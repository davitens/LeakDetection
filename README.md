# LeakDetection

Benchmark code for pipe leak risk prediction in water distribution networks. The project compares model families and class-imbalance treatments using probabilistic and cost-sensitive evaluation rather than only threshold-based classification metrics.

## Associated Paper

This repository is associated with the paper:

https://doi.org/10.1016/j.watres.2026.126622

Abert-Fernández, D., Monclús, H., Fetai, B., & Kozelj, D. (2026). *AI-driven risk management for sustainable water distribution: A comparative study of resampling strategies and cost-sensitive predictive modeling for leakage failure.* Water Research, 306, 126622.

## Data

The dataset is **not distributed with this repository**: `data/modified.csv` is a Git-LFS pointer and `data/` is not tracked. It is the Ljubljana water supply network table described in `data/attributes.txt` (~52.7k pipe segments, ~1.2k leak records, 2.4% positive).

Two filenames appear in the project's history, and they are not interchangeable:

| File | Used by | Default |
|---|---|---|
| `data/modified.csv` | `main_bulletproof.py`, `src/data_loader.py` | yes |
| `data/modified_real.csv` | the manuscript's analysis scripts | no |

Pass `--data <path>` to either entry point to select one explicitly.

## What This Evaluates

The benchmark predicts pipe leak presence/probability from tabular infrastructure data: pipe attributes plus spatially derived indicators (nearby roads, rail, wastewater, cable, gas and district heating features).

The experimental factors are:

- Model architecture: baseline learners, tree-based ensembles, gradient-boosted decision trees, imbalance-aware ensembles, and modern tabular deep learning models.
- Resampling strategy: no resampling, SMOTE, ADASYN, and random undersampling.
- Neighbour settings for synthetic oversampling: SMOTE/ADASYN are evaluated with `k` in `{3, 5, 7, 9}`.

## Models

Model names accepted by `src.models.ModelFactory`:

- `logistic`, `knn`, `dummy` — baseline learners
- `rf`, `et`, `xgb`, `lgbm`, `catboost` — tree-based learners
- `rusb` — RUSBoost, an imbalance-aware ensemble
- `mlp` (RealMLP), `nca` (ModernNCA), `tabm` (TabM), `icl` (TabICL) — tabarena deep learning models
- `tabpfn` (TabPFNv2.5) — supported by the factory but **not part of the paper's twelve benchmarked models**

The twelve models reported in the paper are Logistic Regression, kNN, ET, RF, RUSB, CatBoost, LGBM, XGB, MLP, NCA, TabM and ICL. `dummy` is scored as a reference but excluded from every reported table and figure.

The tabarena models require the extra install step in `setup_env.sh`; they are imported lazily, so the scikit-learn and boosting models run without it.

The default `main_bulletproof.py` run enables only a subset of the models. Edit `models_to_run` and `n_trials` there for full experiments or focused smoke runs. `RESAMPLING_METHODS` and `K_NEIGHBORS` hold the paper's configuration.

## Metrics

Optuna minimises the mean 5-fold out-of-fold LogLoss. The reported scores are computed by `src/metrics.py` from the pooled out-of-fold probabilities:

- **LogLoss** — proper scoring rule for probabilistic classification.
- **Brier Score** — squared-error proper scoring rule; reported `x10` in Figure 1.
- **AUCC** — area under the minimum-expected-cost curve, integrating

  `EC(Pc) = min_t [ Pc * FNR(t) + (1 - Pc) * FPR(t) ]`

  over the normalised cost space `Pc` built from the cost ratio `R = C_FN/C_FP`.

The upper bound of the swept cost ratio is configurable: `--max-cost-ratio`. It defaults to `10**3.2`, the value stated in the manuscript (~1654, the ratio at which flagging every pipe becomes optimal). **The published Table 2 and Figure 1 were computed with `10**2.5`**; pass `--max-cost-ratio 10**2.5` to reproduce those numbers. The lower bound is always `R = 1`.

The DET-curve operating points are computed explicitly in `src/metrics.py` rather than via `sklearn.metrics.det_curve`, because whether that function pads the curve with the trivial all-negative corner changed between scikit-learn releases and silently changes the AUCC.

## Running The Benchmark

Install the dependencies described in `setup_env.sh`, then run from the repository root so relative paths resolve:

```bash
python main_bulletproof.py <run_name> [--data data/modified.csv]
```

Outputs are written under `results/` (gitignored): Optuna trial exports, status CSVs, and the SQLite study registry.

Then score and plot the trials:

```bash
python results_analisys.py --data data/modified.csv --trials-dir results/optuna_trials --out-dir manuscript
```

That writes:

- `manuscript/tables/table_2.csv` — LogLoss, Brier Score and AUCC per model and resampling (Table 2)
- `manuscript/tables/table_c_resampling_cost.csv` — cost of each resampling strategy relative to the baseline (Appendix C)
- `manuscript/gis_figure/best_predictions.csv` — per-pipe probabilities of the best-AUCC model, for the GIS map
- `manuscript/figures/figure_1.png` — metric heatmap
- `manuscript/figures/figure_2.png` — cost curves

The GIS map is produced from `best_predictions.csv` in QGIS; the manuscript figure styling is not reproduced by the plotting code.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_pipeline.py` builds a small synthetic table with the real schema, so it runs without the dataset. It covers the loader's label and feature handling, the preprocessor, the scoring rules (including the closed-form AUCC of a constant predictor), device selection for the learners, and the analysis outputs end to end.

## Repository Notes

- `data/`, `results/`, `old/`, Python caches and manuscript artifacts are gitignored.
- The manuscript figure PDFs are kept trackable: `figure_aucc_auxiliar.pdf`, `figure_1.pdf`, `figure_2.pdf`, `figure_leaks.pdf`. The manuscript LaTeX sources are not distributed; figure fonts are not included.
- Known limits: the tabarena models receive a fixed `num_cpus=6` budget per fit, and `main_bulletproof.py` runs Optuna trials with `n_jobs=-1`, so CPU-bound fits can oversubscribe the machine.
- `src/workers.py`, `src/log.py` and `src/utils.py::get_best_model_oofs` are currently unreferenced.
