# Probabilistic Classifier Chains — Inference Evaluation

Evaluation framework for Bayes-optimal inference rules in multi-label classification using Probabilistic Classifier Chains (PCC), with Binary Relevance baseline.

## Models

| Model | Joint distribution |
|---|---|
| `ProbabilisticClassifierChainCustom` (PCC) | Chain rule: P(y\|x) = Π_j P(y_j \| x, y_<j) |
| `BinaryRelevance` (BR) | Independence: P(y\|x) = Π_j P(y_j \| x) |

Both expose the same `predict_*` inference rules (below). Base estimator is configurable (LogisticRegression, RandomForest, AdaBoost).

## Inference rules

| Inference rule | Optimises | Method |
|---|---|---|
| `predict_hamming` | Hamming Accuracy | Threshold marginals at 0.5 |
| `predict_subset` | Subset Accuracy | MAP over joint distribution |
| `predict_precision` | Precision | Predict single most probable label |
| `predict_npv` | NPV | All-ones except least probable label |
| `predict_recall` | Recall | All-ones (trivially achieves Recall = 1) |
| `predict_markedness` | Markedness | Top-l expectation: arg max E[0.5·(NPV + Pre)] |
| `predict_fmeasure` | F-beta | Pairwise probability aggregation |
| `predict_informedness` | Informedness | Pairwise: include label j iff q_sens[j] + q_spec_cost[j] > C |
| `predict_marginal_scores` | — (continuous) | Returns marginals as scores for ranking metrics |

## Metrics

Binary-prediction metrics (applied to each `predict_*` output):
`hamming_accuracy`, `subset_accuracy`, `precision_score`, `recall_score`,
`negative_predictive_value`, `f_beta`, `macro_f1`, `micro_f1`,
`markedness`, `informedness`.

Ranking metrics (applied to marginal probability scores, Schapire & Singer 2000):
`one_error_score`, `coverage_score`, `ranking_loss_score`, `average_precision_score`.

All metrics are reported in **higher-is-better** form, e.g., "Hamming Accuracy" = 1 − Hamming Loss; "Coverage Score" = 1 − normalised coverage error.

## Implementation notes

- `predict()` uses **prefix-tree batched inference**: for each chain level j, all N × 2^j partial label prefixes share a single batched `predict_proba` call. This is numerically equivalent to brute-force enumeration but ~hundreds of times faster, enabling L=14 datasets (yeast, Water-quality) in seconds. Verified in `tests/test_predict_equivalence.py` against the reference per-element implementation.
- Per-fold `StandardScaler` is fit on training data only (no leakage into test).

## Installation

```bash
conda create --name inference_prob_mlc python=3.10
conda activate inference_prob_mlc
pip install -r requirements.txt
```

## Usage

```bash
python inference_evaluate_models.py
# or
make eval
```

Edit `dataset_names` in `inference_evaluate_models.py` to select which datasets to evaluate.

Results are saved to:
- `result/result_<dataset_name>.csv`
- `result/result_<dataset_name>_crosstab.csv`

## Datasets

Standard MULAN benchmark datasets (`.arff` format) in `datasets/`:
`emotions`, `scene`, `yeast`, `Water-quality`, `CHD_49`, `VirusGO_sparse`

NIH Chest X-ray (pre-extracted `.npy` feature vectors):
`chest_xray_nih__densenet`, `chest_xray_nih__resnet`, `chest_xray_nih__resnetae`

## Running tests

```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- numpy, pandas, scikit-learn, scipy, joblib

See `requirements.txt` for pinned versions.

## License

MIT License — see `LICENSE`.
