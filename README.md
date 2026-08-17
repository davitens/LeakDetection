# LeakDetection

Benchmark code for pipe leak risk prediction in water distribution networks. The project compares model families and class-imbalance treatments using probabilistic and cost-sensitive evaluation rather than only threshold-based classification metrics.

## Associated Paper

This repository is associated with the paper:

https://doi.org/10.1016/j.watres.2026.126622

## What This Evaluates

The benchmark predicts pipe leak presence/probability from tabular infrastructure data. The data combine pipe attributes with spatially derived infrastructure indicators such as nearby roads, rail, wastewater, cable, gas, and district heating features.

The main experimental factors are:

- Model architecture: baseline learners, tree-based ensembles, gradient-boosted decision trees, imbalance-aware ensembles, and modern tabular deep learning models.
- Resampling strategy: no resampling, SMOTE, ADASYN, and random undersampling.
- Neighbor settings for synthetic oversampling: SMOTE/ADASYN are evaluated with multiple `k` values in the benchmark pipeline.

## Models

The code supports the model names used by `src.models.ModelFactory`:

- `logistic`: logistic regression baseline
- `knn`: k-nearest neighbors baseline
- `dummy`: majority-class dummy classifier
- `rf`: random forest
- `et`: extra trees
- `xgb`: XGBoost
- `lgbm`: LightGBM
- `catboost`: CatBoost
- `rusb`: RUSBoost
- `mlp`: RealMLP via `tabarena`
- `nca`: ModernNCA via `tabarena`
- `tabm`: TabM via `tabarena`
- `icl`: TabICL via `tabarena`
- `tabpfn`: TabPFNv2.5 via `tabarena`

The default `main_bulletproof.py` run currently enables only a subset of these models; edit `models_to_run`, `resampling_methods`, and `n_trials` in that file for full experiments or focused smoke runs.

## Metrics

The Optuna objective optimizes mean 5-fold out-of-fold LogLoss. The manuscript analysis evaluates probabilistic and operational performance using:

- LogLoss: proper scoring rule for probabilistic classification.
- Brier Score: squared-error proper scoring rule for predicted probabilities.
- AUCC: Area Under the Cost Curve, summarizing expected operational cost across false-positive/false-negative cost ratios.

## Running The Benchmark

Install dependencies from `requirements.txt` and the external `tabarena` setup described in the lowercase `readme.md` setup script. Run commands from the repository root so relative paths resolve.

```bash
python main_bulletproof.py <run_name>
```

Outputs are written under `results/`, including Optuna trial exports, status CSV files, and out-of-fold predictions. These generated results are intentionally ignored by Git.

## Repository Notes

- `data/`, `results/`, `old/`, Python caches, and manuscript text/build artifacts are ignored.
- The manuscript figure PDFs referenced by `manuscript/manuscript_final_revision.tex` are kept trackable: `figure_aucc_auxiliar.pdf`, `figure_1.pdf`, `figure_2.pdf`, and `figure_leaks.pdf`.
- See `AGENTS.md` for repo-specific setup, execution, and verification notes for future OpenCode sessions.
