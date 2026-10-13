"""Turn exported Optuna trials into the manuscript's scores, tables and figures.

For every ``(model, resampling)`` configuration the best trial by mean LogLoss is
re-selected and its stacked out-of-fold probabilities are scored with LogLoss,
Brier Score and AUCC. The results back manuscript Table 2, Figure 1 (metric
heatmap), Figure 2 (cost curves), Appendix C (cost of resampling relative to the
baseline) and the GIS map.

    python results_analisys.py --data data/modified.csv --max-cost-ratio 10**2.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import metrics
from src.data_loader import DEFAULT_DATA_PATH, load_and_clean_data

MODEL_LABELS = {
    "catboost": "CatBoost",
    "dummy": "Dummy",
    "et": "ET",
    "icl": "ICL",
    "knn": "kNN",
    "lgbm": "LGBM",
    "logistic": "LogReg",
    "mlp": "MLP",
    "nca": "NCA",
    "rf": "RF",
    "rusb": "RUSB",
    "tabm": "TabM",
    "tabpfn": "TabPFN",
    "xgb": "XGB",
}

RESAMPLING_LABELS = {
    None: "Baseline",
    "None": "Baseline",
    "NaN": "Baseline",
    "SMOTE": "SMOTE",
    "ADASYN": "ADASYN",
    "RandomUnderSampler": "RUS",
}

#: Row/column order used by Figure 1 in the manuscript.
MODEL_ORDER = ["LogReg", "kNN", "ET", "RF", "RUSB", "CatBoost", "LGBM", "XGB", "TabM", "ICL", "MLP", "NCA"]

RESAMPLING_ORDER = ["Baseline", "SMOTE", "ADASYN", "RUS"]


def merge_trials(directory):
    """Concatenate every Optuna trial export in ``directory``."""
    paths = sorted(Path(directory).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no trial parquet files found in {directory}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _probabilities(value):
    """Decode a stored OOF prediction vector (list column or string)."""
    if isinstance(value, str):
        return np.array(value.strip().strip("[]").split(), dtype=float)
    return np.asarray(value, dtype=float)


def metrics_table(trials, y_true, max_cost_ratio=metrics.DEFAULT_MAX_COST_RATIO):
    """Score the best trial of every ``(model, resampling)`` configuration.

    The trial is chosen by mean LogLoss because that is the optimisation
    objective; the three reported metrics are then recomputed from its pooled
    out-of-fold probabilities.
    """
    trials = trials[trials["mean_log_loss"].notna()]
    best = trials.loc[trials.groupby(["model", "resampling"])["mean_log_loss"].idxmin()].copy()

    probabilities = [_probabilities(value) for value in best["predictions"]]
    y_true = np.asarray(y_true)
    best["calculated_logloss"] = [metrics.log_loss_score(y_true, p) for p in probabilities]
    best["brier_score"] = [metrics.brier_score(y_true, p) for p in probabilities]
    best["aucc"] = [
        metrics.area_under_cost_curve(y_true, p, max_cost_ratio=max_cost_ratio) for p in probabilities
    ]
    return best.reset_index(drop=True)


def select_map_row(table, exclude=("dummy",)):
    """The configuration whose predictions go on the GIS map.

    The manuscript maps the best-AUCC model, not the best-LogLoss one, and the
    Dummy baseline is left out of every reported figure.
    """
    candidates = table[~table["model"].isin(exclude)]
    if candidates.empty:
        raise ValueError("no candidate configurations to map")
    return candidates.loc[candidates["aucc"].idxmin()]


def relative_cost(labelled, y_true, cost_ratio, max_cost_ratio=metrics.DEFAULT_MAX_COST_RATIO):
    """Cost of each resampling strategy relative to the baseline, in percent.

    Reproduces Appendix C: the expected cost of a configuration at a fixed cost
    ratio, divided by that of the same model without resampling. Expects the
    labelled table (``Model`` / ``Resampling`` columns).
    """
    prevalence = float(np.asarray(y_true).mean())
    _, pc = metrics.cost_ratio_grid(y_true, max_cost_ratio)
    target = (prevalence * cost_ratio) / (prevalence * cost_ratio + (1 - prevalence))
    column = int(np.argmin(np.abs(pc - target)))

    rows = []
    for (model, resampling), group in labelled.groupby(["Model", "Resampling"], dropna=False):
        probabilities = _probabilities(group["predictions"].iloc[0])
        cost = metrics.normalized_expected_cost(y_true, probabilities, pc)[column]
        rows.append({"Model": model, "Resampling": resampling, "expected_cost": cost})

    wide = pd.DataFrame(rows).pivot(index="Model", columns="Resampling", values="expected_cost")
    if "Baseline" not in wide:
        raise ValueError("the baseline configuration is required to compute relative cost")
    return (wide.div(wide["Baseline"], axis=0) - 1.0) * 100.0


def published_order(labelled):
    """Sort rows the way the manuscript tables and figures list them."""
    model_rank = {label: rank for rank, label in enumerate(MODEL_ORDER)}
    resampling_rank = {label: rank for rank, label in enumerate(RESAMPLING_ORDER)}
    return (
        labelled.assign(
            _model=labelled["Model"].map(model_rank).fillna(len(MODEL_ORDER)),
            _resampling=labelled["Resampling"].map(resampling_rank).fillna(len(RESAMPLING_ORDER)),
        )
        .sort_values(["_model", "_resampling"])
        .drop(columns=["_model", "_resampling"])
    )


def _labelled(table):
    """Copy of ``table`` with the display names and the published row order."""
    labelled = table.copy()
    labelled["Model"] = labelled["model"].map(MODEL_LABELS)
    labelled["Resampling"] = labelled["resampling"].map(RESAMPLING_LABELS)
    return labelled


def _pivot(labelled, metric):
    return (
        labelled.pivot_table(index="Model", columns="Resampling", values=metric, aggfunc="min")
        .reindex(index=MODEL_ORDER, columns=RESAMPLING_ORDER)
    )


def plot_metric_heatmap(labelled, path):
    """Three-panel LogLoss / Brier Score / AUCC heatmap (manuscript Figure 1)."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for axis, metric, letter in zip(axes, ["calculated_logloss", "brier_score", "aucc"], "ABC"):
        panel = _pivot(labelled, metric)
        # the Brier score is reported x10 in the paper
        if metric == "brier_score":
            panel = panel * 10
        best = panel.min().min()

        sns.heatmap(
            panel,
            ax=axis,
            annot=True,
            fmt=".3f",
            cmap="Spectral_r",
            linewidths=0.5,
            linecolor="black",
            cbar=False,
        )
        axis.set_title(f"{letter}) {metric.replace('calculated_', '').replace('_', ' ').upper()}")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=0)

        for text in axis.texts:
            if np.isclose(float(text.get_text()), best, atol=1e-9):
                text.set_fontweight("bold")

    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_cost_curves(labelled, path, y_true, max_cost_ratio=metrics.DEFAULT_MAX_COST_RATIO):
    """Normalised expected cost against the cost ratio (manuscript Figure 2)."""
    ratios, pc = metrics.cost_ratio_grid(y_true, max_cost_ratio)
    models = [m for m in MODEL_ORDER if m in set(labelled["Model"])]
    colors = dict(zip(models, plt.cm.tab20(np.linspace(0, 1, max(len(models), 1)))))
    linestyles = dict(zip(RESAMPLING_ORDER, ["--", "-.", ":", "-"]))

    figure, axes = plt.subplots(
        1 + len(models),
        1,
        figsize=(12, 3 * (1 + len(models))),
        sharex=True,
        sharey=True,
    )
    for axis in axes:
        axis.set_ylabel("NEC")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(r"Ratio ($C_{FN}/C_{FP}$)")

    for _, row in labelled.iterrows():
        probabilities = _probabilities(row["predictions"])
        costs = metrics.normalized_expected_cost(y_true, probabilities, pc)
        style = dict(
            color=colors.get(row["Model"], "grey"),
            linestyle=linestyles.get(row["Resampling"], "-"),
        )
        axes[0].plot(ratios, costs, alpha=0.6, **style)
        for axis, model in zip(axes[1:], models):
            if row["Model"] == model:
                axis.plot(ratios, costs, label=row["Resampling"], **style)
                axis.set_title(model, loc="left")

    axes[0].set_xscale("log")
    axes[0].set_title("All configurations", loc="left")
    axes[1].legend(loc="upper right", fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Score and plot leak benchmark trials")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="dataset CSV")
    parser.add_argument("--trials-dir", default="results/optuna_trials")
    parser.add_argument("--out-dir", default="manuscript")
    parser.add_argument("--cost-ratio", type=float, default=25.0,
                        help="C_FN/C_FP used for the relative-cost table")
    parser.add_argument("--max-cost-ratio", type=float, default=metrics.DEFAULT_MAX_COST_RATIO,
                        help="upper bound of the AUCC sweep (10**2.5 reproduces the published table)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    X, y, _, _ = load_and_clean_data(args.data)
    y_true = y.to_numpy()
    trials = merge_trials(args.trials_dir)
    scored = metrics_table(trials, y_true, max_cost_ratio=args.max_cost_ratio)
    labelled = published_order(_labelled(scored))
    # the paper reports the twelve benchmarked learners; Dummy and TabPFN are
    # scored but excluded from Table 2 and both figures
    reported = labelled[labelled["Model"].isin(MODEL_ORDER)]

    reported[["Model", "Resampling", "calculated_logloss", "brier_score", "aucc"]].rename(
        columns={"calculated_logloss": "LogLoss", "brier_score": "Brier Score"}
    ).to_csv(tables_dir / "table_2.csv", index=False)

    relative_cost(reported, y_true, args.cost_ratio).to_csv(tables_dir / "table_c_resampling_cost.csv")

    map_row = select_map_row(scored)
    gis_dir = out_dir / "gis_figure"
    gis_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fid": X.index, "predictions": _probabilities(map_row["predictions"])}).to_csv(
        gis_dir / "best_predictions.csv", index=False
    )
    print(f"GIS map model: {map_row['model']} / {map_row['resampling']} (AUCC {map_row['aucc']:.4f})")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_heatmap(reported, figures_dir / "figure_1.png")
    plot_cost_curves(reported, figures_dir / "figure_2.png", y_true, args.max_cost_ratio)
    print(f"Wrote scores and figures under {out_dir}")


if __name__ == "__main__":
    main()
