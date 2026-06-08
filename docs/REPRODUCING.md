# Reproducing the paper's results

This guide covers the full experimental protocol, datasets, sweeps, results table,
repository layout, and testing for **DaCaF**. For a quick overview and library usage,
see the [main README](../README.md).

The paper uses **Probabilistic Classifier Chains (PCC)** with an **L2-regularised
logistic-regression** base learner, **10-fold cross-validation**, and the **exact
computation paradigm** (enumerate all `2^L` labelings, so it is limited to a small or
moderate number of labels). The exact published protocol is recorded in
[`docs/paper.yaml`](paper.yaml).

## Datasets in the paper (6)

| Dataset | #labels (L) | #instances | Type |
|---|---:|---:|---|
| Emotions | 6 | 593 | tabular |
| CHD-49 | 6 | 555 | tabular |
| Scene | 6 | 2407 | tabular |
| Water-quality | 14 | 1060 | tabular |
| Yeast | 14 | 2417 | tabular |
| ChestX-ray8 | 8 | 25596 | image (ResNet / resnetAE / DenseNet features) |

For the chest-X-ray data we extract features with a pretrained backbone via
[TorchXRayVision](https://github.com/mlmed/torchxrayvision); the raw NIH features are
**not redistributed** (see
[`dacaf_mlc/chest_xray_dataset/Readme.md`](../dacaf_mlc/chest_xray_dataset/Readme.md)).

## Running the experiments

**One command** for the tractable (tabular) subset, runs the 5 tabular datasets × 5
seeds and aggregates:

```bash
make reproduce          # = bash scripts/reproduce_tabular.sh
```

**Full sweep** (heavy, use a cluster):

```bash
dacaf-mlc --dataset CHD_49 --seed 1        # one job per (dataset, seed); repeat as needed
python scripts/aggregate.py                # aggregate when jobs finish
```

Aggregated outputs per dataset: `result/result_<dataset>.csv` (long format),
`_summary.csv` (mean ± std), and `_crosstab.csv` (target × evaluation pivot).

## Run it online (Code Ocean)

A one-click reproducible capsule is available:
**<https://codeocean.com/capsule/1580907/tree>**. Click **Reproducible Run** to rebuild
the environment and reproduce the CHD-49 target × evaluation table
(`result_CHD_49_crosstab.csv`) on CPU in seconds — every diagonal entry is the maximum
of its column, the paper's central claim. The capsule entry point is
[`run`](../run); dependencies are pinned in
[`requirements-core.txt`](../requirements-core.txt).

## Results in full

**The headline finding: mismatch hurts.** When you evaluate with metric *E* but optimise
for a different metric *T* during prediction, performance usually drops. Optimising the
metric you actually care about is (almost always) best. This is verified on 5 tabular
datasets plus a chest-X-ray image dataset, using the *exact* computation paradigm (no
approximation blurring the picture).

You read the table **column by column**: each column is one evaluation metric, each row
is the metric you optimised for. The **bold diagonal** (optimise the metric you evaluate)
should be the largest value in its column.

**Example: CHD-49 (PCC + logistic regression, mean over 5 seeds × 10-fold, the fastest
dataset).** Values are percentages, higher is better. Bold = the maximum of its column =
the rule that targets that metric.

| Target ↓ \ Eval → | F₁ | Hamming | Markedness | Precision | NPV | Recall | Subset |
|---|---:|---:|---:|---:|---:|---:|---:|
| **F₁** | **67.1** | 69.3 | 71.9 | 62.7 | 81.1 | 79.2 | 15.1 |
| **Hamming** | 63.9 | **70.8** | 71.3 | 66.6 | 75.7 | 66.9 | 18.1 |
| **Markedness** | 34.2 | 63.7 | **77.0** | 33.4 | 71.1 | 40.5 | 8.4 |
| **Precision** | 40.5 | 64.8 | 68.3 | **73.5** | 63.1 | 29.1 | 3.1 |
| **NPV** | 58.4 | 43.0 | 71.5 | 43.0 | **100.0** | 99.5 | 0.0 |
| **Recall** | 58.4 | 43.0 | 71.5 | 43.0 | 100.0 | **99.5** | 0.0 |
| **Subset** | 64.0 | 69.4 | 70.4 | 64.2 | 76.5 | 69.8 | **18.9** |

In all **7 of 7** columns the diagonal (target = evaluation) is the maximum: to score
best on a metric, optimise that metric. The NPV and Recall rows are identical because
both BOPs are the all-ones vector `1…1`.

## Repository layout

```
dacaf_mlc/                           # installable package
  probability_classifier_chains.py   # PCC + the 7 per-metric Bayes-optimal predict_* rules
  evaluation_metrics.py              # the 7 paper metrics (higher-is-better form)
  arff_dataset.py                    # MULAN ARFF loader + 10-fold CV
  datasets.py                        # dataset registry + loaders
  metrics_registry.py                # which metrics run on which inference rule
  pipeline.py                        # training / k-fold eval / run_single
  evaluate.py                        # CLI entry point (dacaf-mlc): parse_args + main
  config.py                          # paths + protocol constants
  utils.py                           # result aggregation
  chest_xray_dataset/                # NIH feature extractor + loader ([image] extra)
  skmultiflow/                       # vendored ClassifierChain base
pyproject.toml                       # packaging + deps (core / [image] / [dev])
scripts/                             # reproduce_tabular.sh + Slurm cluster scripts
tests/                               # unit tests + brute-force optimality + e2e
docs/                                # paper.yaml protocol manifest + CONVENTIONS.md + REPRODUCING.md
datasets/                            # the paper's MULAN ARFFs (+ chest-xray label CSV)
result/                              # aggregated result CSVs
CONTRIBUTING.md  CITATION.cff
paper/                               # local copy of the paper source (not tracked)
```

## Testing

```bash
python -m pytest tests/ -v
```

Every inference rule is checked against **brute-force enumeration** of the expected
metric, so the closed-form rules are provably correct on small cases. A batched predictor
(one `predict_proba` call per chain level instead of `N·L·2^L`) is verified numerically
equivalent to the reference enumeration.
</content>
</invoke>
