"""Self-checks for the pipeline pieces that do not need the real dataset.

The 52.7k-pipe Ljubljana table is not in the repository (Git-LFS), so these
tests build a tiny synthetic table with the same schema and exercise the
loader, the preprocessor, the metrics and the search spaces.

    python -m pytest tests/ -q
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src import metrics
from src.data_loader import CAT_FEATURES, load_and_clean_data
from src.preprocessing import make_preprocessor

N_NO_LEAK = 40
N_LEAK = 10

# Column order and coding follow data/attributes.txt.
_CATEGORICAL_CODES = {
    "DW_atr1": [241, 249, 253],  # cast iron / PVC / ductile iron
    "DW_atr4": [6, 7, 8],  # primary / secondary / tertiary network
    "RO_atr1": [0, 9, 11, 15],  # no data / city road / collector / public path
    "RR_atr2": ["", 13, 15],  # nothing / main line / industrial track
    "EE_atr2": ["", 17, 20],  # nothing / 20 kV / 10 kV
    "EK_atr1": [281, 285, 286],  # cable ducting / buried / buried
    "ZP_atr4": ["", 10, 12],  # nothing / DN 90-110 / DN 160-225
    "DH_atr2": ["", 80, 84],  # nothing / DN 25-50 / DN 100-125
    "WW_atr1": [254, 255, 256],  # sewage / combined / stormwater
}


def write_synthetic_csv(path, missing_token=""):
    """Write a small CSV with the schema of data/modified.csv."""
    rng = np.random.RandomState(0)
    rows = []
    for i in range(N_NO_LEAK + N_LEAK):
        leaking = i >= N_NO_LEAK
        if leaking:
            leak_no = rng.randint(1, 4)
            leak_after_inst = rng.randint(1, 40)
        else:
            leak_no = missing_token
            leak_after_inst = missing_token
        row = {
            "DW_fid": i + 1,
            "DW_leto_izgra": rng.randint(1950, 2020),
            "DW_DIM_tl": rng.choice([50, 100, 150, 200]),
            "DE_Pipe_Age": rng.randint(1, 60),
            "DW_Leak_No": leak_no,
            "DW_Leak_After_Inst": leak_after_inst,
        }
        for col, choices in _CATEGORICAL_CODES.items():
            row[col] = rng.choice(choices)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def synthetic_csv(tmp_path):
    return write_synthetic_csv(tmp_path / "modified.csv")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def test_every_nominal_code_column_is_declared_categorical(synthetic_csv):
    X, _, cat_features, num_features = load_and_clean_data(synthetic_csv)

    assert set(cat_features) == set(CAT_FEATURES)
    assert set(cat_features) | set(num_features) == set(X.columns)
    assert not set(cat_features) & set(num_features)


@pytest.mark.parametrize("missing_token", ["NULL", "", "NA", "No leak", "-"])
def test_any_missing_marker_in_the_leak_count_column_means_no_leak(tmp_path, missing_token):
    """Only a real leak record may set the label.

    pandas turns "NULL"/"NA"/"" into NaN on read, but a utility export can use
    any token (the column is free text). Anything that is not a number must be
    treated as "no leak record", never as a leak.
    """
    path = write_synthetic_csv(tmp_path / "markers.csv", missing_token=missing_token)

    _, y, _, _ = load_and_clean_data(path)

    assert y.sum() == N_LEAK


def test_leak_derived_columns_never_reach_the_model(synthetic_csv):
    X, y, _, _ = load_and_clean_data(synthetic_csv)

    for column in ("DW_Leak_No", "LeakPresence", "DW_Leak_After_Inst"):
        assert column not in X.columns
    assert set(y.unique()) == {0, 1}
    assert y.sum() == N_LEAK


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocessor_handles_numeric_category_codes_with_missing_values(synthetic_csv):
    X, _, cat_features, num_features = load_and_clean_data(synthetic_csv)

    transformed = make_preprocessor(cat_features, num_features).fit_transform(X)

    matrix = transformed.to_numpy(dtype=float)
    assert matrix.shape == (len(X), len(num_features) + len(cat_features))
    assert np.isfinite(matrix).all()


def test_preprocessor_tolerates_unseen_categories(synthetic_csv):
    X, _, cat_features, num_features = load_and_clean_data(synthetic_csv)
    preprocessor = make_preprocessor(cat_features, num_features).fit(X)

    unseen = X.copy()
    unseen["RO_atr1"] = 999  # never seen at fit time
    unseen.loc[unseen.index[0], "WW_atr1"] = np.nan

    transformed = preprocessor.transform(unseen)

    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_preprocessor_keeps_the_width_of_an_entirely_missing_column(synthetic_csv):
    X, _, cat_features, num_features = load_and_clean_data(synthetic_csv)
    preprocessor = make_preprocessor(cat_features, num_features)

    empty = X.copy()
    empty["DW_DIM_tl"] = np.nan  # numeric column missing for every pipe
    empty["ZP_atr4"] = np.nan  # a spatial feature absent network-wide
    preprocessor.fit(empty)

    assert preprocessor.transform(X).shape[1] == len(num_features) + len(cat_features)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _constant_predictor_aucc(prevalence, max_cost_ratio):
    """Closed form for a model that predicts the same score for every pipe.

    Its only two operating points are all-negative (fpr=0, fnr=1) and
    all-positive (fpr=1, fnr=0), so EC(Pc) = min(Pc, 1 - Pc) and the AUCC is
    that envelope integrated between Pc_min = prevalence and Pc_max.
    """
    p = prevalence
    pc_max = (p * max_cost_ratio) / (p * max_cost_ratio + (1 - p))
    return (0.25 - p**2) / 2 + (pc_max - 0.5) - (pc_max**2 - 0.25) / 2


def test_aucc_of_a_constant_predictor_is_the_min_cost_envelope():
    y_true = np.zeros(1000, dtype=int)
    y_true[:20] = 1  # prevalence 0.02
    y_prob = np.zeros(1000)  # every pipe scored 0

    for max_cost_ratio in (10**2.5, 10**3.2):
        aucc = metrics.area_under_cost_curve(y_true, y_prob, max_cost_ratio=max_cost_ratio)

        _, pc = metrics.cost_ratio_grid(y_true, max_cost_ratio)
        # a constant score has only two operating points, so the lower envelope
        # of the expected cost is min(Pc, 1 - Pc)
        assert aucc == pytest.approx(np.trapezoid(np.minimum(pc, 1 - pc), pc), rel=1e-12)


def test_aucc_of_a_constant_predictor_matches_the_closed_form():
    """The discretisation error must stay far below the differences reported."""
    y_true = np.zeros(1000, dtype=int)
    y_true[:20] = 1

    for max_cost_ratio in (10**2.5, 10**3.2):
        aucc = metrics.area_under_cost_curve(y_true, np.zeros(1000), max_cost_ratio=max_cost_ratio)
        assert aucc == pytest.approx(_constant_predictor_aucc(0.02, max_cost_ratio), abs=1e-3)


def test_default_aucc_range_is_the_one_stated_in_the_manuscript():
    assert metrics.DEFAULT_MAX_COST_RATIO == pytest.approx(10**3.2)


def test_aucc_range_is_configurable_and_widening_it_raises_the_area():
    y_true = np.zeros(1000, dtype=int)
    y_true[:20] = 1

    narrow = metrics.area_under_cost_curve(y_true, np.zeros(1000), max_cost_ratio=10**2.5)
    wide = metrics.area_under_cost_curve(y_true, np.zeros(1000))

    assert narrow < wide


def test_aucc_of_a_perfect_ranking_is_zero():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    assert metrics.area_under_cost_curve(y_true, y_prob) == pytest.approx(0.0, abs=1e-12)


def test_probabilistic_scores_match_their_definitions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.2, 0.8, 0.8])

    # Brier = mean((p - y)^2) = 0.2^2 for every pipe
    assert metrics.brier_score(y_true, y_prob) == pytest.approx(0.04)
    # LogLoss = -log(0.8) for every pipe
    assert metrics.log_loss_score(y_true, y_prob) == pytest.approx(-np.log(0.8))
    assert metrics.brier_score(y_true, np.full(4, 0.5)) == pytest.approx(0.25)


def test_metrics_reject_a_single_class_ground_truth():
    with pytest.raises(ValueError):
        metrics.area_under_cost_curve(np.zeros(10, dtype=int), np.zeros(10))


# --------------------------------------------------------------------------- #
# Model factory and search spaces
# --------------------------------------------------------------------------- #
def test_model_factory_does_not_require_the_optional_dl_stack():
    """tabarena must only be imported when a tabular DL model is requested."""
    import importlib
    import sys

    sys.modules.pop("src.models.ModelFactory", None)
    sys.modules.pop("tabarena", None)

    factory = importlib.import_module("src.models.ModelFactory")

    assert "tabarena" not in sys.modules
    assert factory.ModelFactory.get_model("logistic") is not None
    with pytest.raises(ImportError):
        factory.ModelFactory.get_model("tabm")


class _FirstChoiceTrial:
    """Minimal trial double: every suggest_* returns its first candidate."""

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_search_spaces_follow_the_available_device(monkeypatch):
    from src.models import spaces

    trial = _FirstChoiceTrial()

    monkeypatch.setattr(spaces, "USE_GPU", False)
    assert spaces.get_optuna_space(trial, "catboost")["task_type"] == "CPU"
    assert "devices" not in spaces.get_optuna_space(trial, "catboost")
    assert spaces.get_optuna_space(trial, "lgbm")["device"] == "cpu"
    assert spaces.get_optuna_space(trial, "xgb")["device"] == "cpu"

    monkeypatch.setattr(spaces, "USE_GPU", True)
    assert spaces.get_optuna_space(trial, "catboost")["task_type"] == "GPU"
    assert spaces.get_optuna_space(trial, "lgbm")["device"] == "gpu"
    assert spaces.get_optuna_space(trial, "xgb")["device"] == "cuda"


@pytest.mark.parametrize("use_gpu", [False, True])
def test_gradient_boosted_models_train_on_the_selected_device(synthetic_csv, monkeypatch, use_gpu, tmp_path):
    """Both branches must produce a space the learners actually accept."""
    from src.data_loader import load_and_clean_data
    from src.models import spaces
    from src.models.ModelFactory import ModelFactory
    from src.preprocessing import make_preprocessor

    if use_gpu and not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    monkeypatch.setattr(spaces, "USE_GPU", use_gpu)
    # CatBoost writes its training logs into the working directory
    monkeypatch.chdir(tmp_path)

    X, y, cat_features, num_features = load_and_clean_data(synthetic_csv)
    Xt = make_preprocessor(cat_features, num_features).fit_transform(X)

    trial = _FirstChoiceTrial()
    for name in ("xgb", "lgbm", "catboost"):
        model = ModelFactory.get_model(name, params=spaces.get_optuna_space(trial, name))
        model.fit(Xt, y)
        assert model.predict_proba(Xt).shape == (len(X), 2)


def test_detected_device_matches_torch():
    from src.models import spaces

    assert spaces.USE_GPU is torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #
def test_benchmark_defaults_match_the_paper_configuration():
    import inspect

    import main_bulletproof

    defaults = inspect.signature(main_bulletproof.run_benchmark).parameters

    assert defaults["resampling_methods"].default is None  # -> all four
    assert defaults["neightboors"].default == [3, 5, 7, 9]
    assert set(main_bulletproof.RESAMPLING_METHODS) == {
        None,
        "SMOTE",
        "ADASYN",
        "RandomUnderSampler",
    }


def test_gis_map_uses_the_best_aucc_model():
    """The manuscript maps the best-AUCC model (ICL), not the best LogLoss one."""
    from results_analisys import select_map_row

    table = pd.DataFrame(
        {
            "model": ["lgbm", "icl"],
            "resampling": ["None", "None"],
            "calculated_logloss": [0.0799, 0.0800],
            "aucc": [0.1080, 0.1059],
        }
    )

    assert select_map_row(table)["model"] == "icl"


def test_relative_cost_compares_each_resampling_against_the_baseline():
    from results_analisys import relative_cost

    # the positive pipe ranks below a negative one in the baseline, and last of
    # all for SMOTE, so SMOTE must cost more at every ratio
    y_true = np.array([0, 0, 0, 0, 1])
    table = pd.DataFrame(
        {
            "Model": ["ICL", "ICL"],
            "Resampling": ["Baseline", "SMOTE"],
            "predictions": [[0.1, 0.2, 0.3, 0.4, 0.35], [0.4, 0.3, 0.2, 0.1, 0.05]],
        }
    )

    relative = relative_cost(table, y_true, cost_ratio=25.0)

    assert relative.loc["ICL", "Baseline"] == pytest.approx(0.0)
    assert relative.loc["ICL", "SMOTE"] > 0


def test_table_rows_come_out_in_the_published_order():
    from results_analisys import published_order

    table = pd.DataFrame(
        {
            "Model": ["XGB", "LogReg", "XGB"],
            "Resampling": ["SMOTE", "Baseline", "Baseline"],
        }
    )

    ordered = published_order(table)

    assert list(zip(ordered["Model"], ordered["Resampling"])) == [
        ("LogReg", "Baseline"),
        ("XGB", "Baseline"),
        ("XGB", "SMOTE"),
    ]


def test_metrics_table_scores_the_pooled_oof_of_the_best_logloss_trial():
    from results_analisys import metrics_table
    y_true = np.array([0] * 95 + [1] * 5)
    perfect = np.where(y_true == 1, 0.9, 0.05)
    table = pd.DataFrame(
        {
            "model": ["logistic", "logistic"],
            "resampling": ["None", "None"],
            "k": [np.nan, np.nan],
            "mean_log_loss": [0.90, 0.20],  # the second trial wins
            "predictions": [list(np.full(100, 0.5)), list(perfect)],
        }
    )

    scored = metrics_table(table, y_true)

    assert len(scored) == 1
    assert scored["calculated_logloss"].iloc[0] == pytest.approx(
        (95 * -np.log(0.95) + 5 * -np.log(0.9)) / 100
    )
    # a perfectly separated ranking costs nothing at any cost ratio
    assert scored["aucc"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert scored["brier_score"].iloc[0] == pytest.approx((95 * 0.05**2 + 5 * 0.1**2) / 100)


def test_analysis_writes_every_manuscript_artefact(tmp_path, synthetic_csv, monkeypatch):
    """End to end: Table 2, Appendix C, the map CSV and both figures."""
    import sys

    import results_analisys
    from src.data_loader import load_and_clean_data

    _, y, _, _ = load_and_clean_data(synthetic_csv)
    y_true = y.to_numpy()

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "knn",
                "resampling": "None",
                "k": np.nan,
                "mean_log_loss": 0.5,
                "predictions": list(np.full(len(y_true), 0.05)),
            },
            {
                "model": "knn",
                "resampling": "SMOTE",
                "k": 3,
                "mean_log_loss": 0.4,
                "predictions": list(np.where(y_true == 1, 0.9, 0.05)),
            },
            # the Dummy baseline is scored but must stay out of the figures
            {
                "model": "dummy",
                "resampling": "None",
                "k": np.nan,
                "mean_log_loss": 0.6,
                "predictions": list(np.zeros(len(y_true))),
            },
        ]
    ).to_parquet(trials_dir / "trials.parquet")

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "results_analisys.py",
            "--data",
            str(synthetic_csv),
            "--trials-dir",
            str(trials_dir),
            "--out-dir",
            str(out_dir),
        ],
    )

    results_analisys.main()

    for artefact in (
        "tables/table_2.csv",
        "tables/table_c_resampling_cost.csv",
        "gis_figure/best_predictions.csv",
        "figures/figure_1.png",
        "figures/figure_2.png",
    ):
        assert (out_dir / artefact).is_file(), artefact


def test_tabarena_fit_budget_never_asks_for_an_absent_gpu():
    from src.models.ModelFactory import tabarena_fit_kwargs

    kwargs = tabarena_fit_kwargs()

    assert kwargs["num_gpus"] == int(torch.cuda.is_available())
    assert kwargs["num_cpus"] >= 1


def test_aucc_reproduces_the_published_dummy_value():
    """Golden value from manuscript Table 2.

    52,700 pipes with the prevalence implied by the published Dummy LogLoss give
    Dummy AUCC = 0.243004 at R <= 10**2.5. This pins the envelope, the cost grid
    and the range semantics to the published number, which is what silently
    drifted when the metric relied on sklearn.metrics.det_curve.
    """
    y_true = np.zeros(52700, dtype=int)
    y_true[:1242] = 1  # the paper's ~1240 leak records at 52.7k pipes

    aucc = metrics.area_under_cost_curve(y_true, np.zeros(52700), max_cost_ratio=10**2.5)

    assert aucc == pytest.approx(0.243004, abs=1e-5)
