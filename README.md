# Probabilistic Classifier Chains — Inference Evaluation

Evaluation framework for Bayes-optimal inference rules in multi-label classification using Probabilistic Classifier Chains (PCC).

## Overview

This codebase implements and evaluates optimal inference algorithms for different loss functions in multi-label classification:

| Inference rule | Optimises | Method |
|---|---|---|
| `predict_hamming` | Hamming Accuracy | Threshold marginals at 0.5 |
| `predict_subset` | Subset Accuracy | MAP over joint distribution |
| `predict_precision` | Precision | Predict single most probable label |
| `predict_npv` | NPV | All-ones except least probable label |
| `predict_recall` | Recall | All-ones (trivially achieves Recall = 1) |
| `predict_markedness` | Markedness | Expectation over marginals |
| `predict_fmeasure` | F-beta | Pairwise probability aggregation |
| `predict_informedness` | Informedness | Pairwise probability aggregation |

All metrics are reported in **higher-is-better** form (e.g., "Hamming Accuracy" = 1 − Hamming Loss).

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
