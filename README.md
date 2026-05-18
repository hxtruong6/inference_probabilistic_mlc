# Bayes-Optimal Inference for Probabilistic Classifier Chains

> Reference implementation and reproducibility artifact for loss-specific Bayes-optimal inference in multi-label classification with Probabilistic Classifier Chains (PCC) [[Dembczyński et al., 2010](#references)].

This repository accompanies our study of metric-aware decoding for PCC. It provides exact inference rules for eight target losses, two corrected derivations relative to the original literature, a batched joint-inference algorithm that reduces the complexity of exact PCC prediction by orders of magnitude, and a fully reproducible evaluation pipeline over eleven standard multi-label benchmarks.

---

## Highlights

1. **Eight Bayes-optimal inference rules** for PCC — Hamming, subset 0/1, precision, recall, NPV, F-β, Markedness, and Informedness — all sharing the same joint estimator, enabling a clean isolation of *inference* from *modelling*.
2. **Two corrected derivations** with respect to the published PCC line of work:
   - **Markedness**: original closed form assumed vacuous precision = 1; we re-derive it consistently with the scikit-learn convention (`zero_division = 0`) standard in the modern multi-label literature.
   - **Informedness**: original rule was degenerate (always predicted ŷ = 0); we provide a correct decomposition based on the joint pairwise distribution P(y_j = 1, |y| = s | x).
   Both derivations are verified against brute-force enumeration in the test suite.
3. **Prefix-tree batched PCC inference**: the naive predictor performs N · L · 2^L base-classifier calls; our implementation performs exactly **L** — empirical speed-up ≈ 300× at L = 6 and ≈ 8000× at L = 10, with numerical equivalence (atol = 1e-12) to the reference enumeration up to L = 7.
4. **Full reproducibility**: stratified 10-fold cross-validation, five seeds, three base learners (Logistic Regression, Random Forest, AdaBoost), eleven MULAN benchmarks; pinned dependencies; Slurm submission scripts; paper-format aggregation utilities.
5. **Both binary-prediction and ranking metric families** reported (example-based, label-based, ranking-based; all in higher-is-better form).

---

## Contents

- [Method](#method)
  - [Models](#models)
  - [Bayes-optimal inference rules](#bayes-optimal-inference-rules)
  - [Corrected derivations](#corrected-derivations)
  - [Evaluation metrics](#evaluation-metrics)
- [Algorithmic contribution: batched joint inference](#algorithmic-contribution-batched-joint-inference)
- [Reproducibility](#reproducibility)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations and scope](#limitations-and-scope)
- [References](#references)

---

## Method

### Models

| Model | Joint distribution | Role |
|---|---|---|
| `ProbabilisticClassifierChainCustom` (PCC) | P(y \| x) = Π_j P(y_j \| x, y_<j) | Chain rule — captures label dependencies |
| `BinaryRelevance` (BR) | P(y \| x) = Π_j P(y_j \| x) | Independence baseline (no chain) |

Both classes expose the **same inference interface**; they differ only in how the joint label distribution is constructed. This isolates the contribution of chain-aware inference for each target metric. Any scikit-learn-compatible base estimator supporting `predict_proba` is admissible.

### Bayes-optimal inference rules

For a given joint P(y | x), each rule returns the prediction ŷ that maximises the expected value of its target metric. Let L denote the number of labels.

| Method | Optimises | Decision rule |
|---|---|---|
| `predict_hamming` | Hamming accuracy | ŷ_j = 1 ⇔ P(y_j = 1 \| x) > ½ |
| `predict_subset` | Subset accuracy (0/1) | ŷ = arg max_y P(y \| x) over all 2^L vectors |
| `predict_precision` | Precision | Predict only the label with the highest marginal |
| `predict_recall` | Recall | ŷ = 1 (trivially optimal; recall = 1) |
| `predict_npv` | Negative predictive value | All ones except the label with the lowest marginal |
| `predict_markedness` | Markedness = ½(Precision + NPV) | arg max over top-l candidates — see [below](#corrected-derivations) |
| `predict_fmeasure` | F-β | Pairwise probability aggregation [Dembczyński et al., 2011] |
| `predict_informedness` | Informedness = ½(Sens. + Spec.) | Per-label threshold on q_sens + q_spec_cost — see [below](#corrected-derivations) |
| `predict_marginal_scores` | — (continuous) | Returns marginals; consumed by ranking metrics |

### Corrected derivations

Two rules from the original PCC formulation are inconsistent with the metric conventions adopted by scikit-learn (`zero_division = 0`) and the broader modern multi-label literature. We provide corrected derivations and verify them against brute-force enumeration.

#### `predict_markedness`

The original closed form assumed Precision = 1 in the degenerate case `|ŷ| = 0 ∧ ∃ y_j = 1` (vacuous precision). Under the scikit-learn convention, the same case returns 0. Re-deriving the expected markedness under that convention, for the candidate "predict the top-l labels by marginal" with sorted marginals π_(1) ≥ … ≥ π_(L) and A_l = Σ_{i ≤ l} π_(i):

- l = 0: E[M | x] = ½ · ( P(y = 0 | x) + 1 − Σ π_j / L )
- 0 < l < L: E[M | x] = ½ · ( A_l / l + 1 − (Σ π_j − A_l) / (L − l) )
- l = L: E[M | x] = ½ · ( Σ π_j / L + P(y = 1 | x) )

Verified against brute-force enumeration on toy cases in `tests/`.

#### `predict_informedness`

The original implementation produced all-zero predictions because E[ŷ = 0] was set to 1 + P(y = 0 | x), dominating every alternative. Starting from the joint pairwise distribution P(y_j = 1, |y| = s | x), we obtain:

**Include label j iff q_sens[j] + q_spec_cost[j] > C**, where

- q_sens[j] = Σ_{s = 1..L} P(y_j = 1, |y| = s | x) / s
- q_spec_cost[j] = Σ_{s = 1..L − 1} P(y_j = 1, |y| = s | x) / (L − s)
- C = P(|y| = 0 | x) / L + Σ_{s = 1..L − 1} P(|y| = s | x) / ( s · (L − s) )

Verified on five samples against brute-force enumeration; in every case the predicted ŷ matched the enumeration-based optimum.

### Evaluation metrics

All metrics are reported in **higher-is-better** form (e.g. `coverage_score` = 1 − normalised coverage error) to remove sign ambiguity from cross-tabulations.

- **Binary-prediction metrics** — applied to each `predict_*` output: `hamming_accuracy`, `subset_accuracy`, `precision_score`, `recall_score`, `negative_predictive_value`, `f_beta`, `macro_f1`, `micro_f1`, `markedness`, `informedness`.
- **Ranking metrics** — applied to marginal scores [Schapire & Singer, 2000]: `one_error_score`, `coverage_score`, `ranking_loss_score`, `average_precision_score`.

Edge-case behaviour matches scikit-learn (`zero_division = 0`) and is documented per metric. The example-based Precision/NPV adopt the "vacuous = 0 when there are unpredicted positives" convention, standard in the multi-label literature; this is precisely the convention that conflicts with the original Markedness derivation, motivating the correction above.

---

## Algorithmic contribution: batched joint inference

The naive PCC predictor invokes the base estimator `N · L · 2^L` times — once per sample, per chain level, per candidate prefix. Our implementation performs a **single batched `predict_proba` call per chain level**, exploiting the prefix-tree structure of the chain. At level j, all `N · 2^j` (sample, prefix) pairs are passed to the j-th base classifier together; outputs are expanded via an interleaved sample-major layout that preserves the canonical MSB-first index ↔ label-vector mapping.

Total `predict_proba` calls: **N · L · 2^L → L**.

| L | Speed-up vs. reference enumeration (50 samples, 10 features) |
|---:|---|
| 6 | ~300× |
| 10 | ~8000× |

Numerical equivalence (atol = 1e-12) to the reference is verified by `tests/test_predict_equivalence.py` for L ∈ {2, 3, 5, 7}.

---

## Reproducibility

### Environment

```bash
conda create -n inference_prob_mlc python=3.10
conda activate inference_prob_mlc
pip install -r requirements.txt
```

Pinned versions (subset): `numpy==1.26.4`, `pandas==2.0.3`, `scikit-learn==1.2.2`, `scipy==1.12.0`, `joblib==1.3.2`.

### Protocol

| Setting | Value |
|---|---|
| Global random seed | `SEED = 6` |
| Cross-validation | `KFold`, `n_splits = 10`, `shuffle = True` |
| Feature scaling | `StandardScaler` fit on training fold only (no leakage) |
| LogisticRegression | `max_iter = 5 × 10⁶`, default L2 |
| RandomForestClassifier | `n_estimators = 100`, `n_jobs = 1` (outer joblib parallelism) |
| AdaBoostClassifier | `n_estimators = 50` |
| Fold parallelism | `joblib.Parallel(n_jobs = −1)` |

### Running the evaluation

Single (dataset, seed, estimator) combination:

```bash
python inference_evaluate_models.py \
    --dataset emotions --seed 1 --estimator lr --output-dir result
```

Writes `result/<dataset>/seed<S>_<est-tag>.csv` (and `_crosstab.csv`) for that one combination.

Full paper sweep (no arguments) — iterates over `DEFAULT_DATASET_NAMES × DEFAULT_SEEDS × all estimators`. Practical only for small datasets on a workstation; for the full sweep we recommend Slurm:

```bash
python inference_evaluate_models.py     # local
# or: see slurm/README.md                # cluster, one job per (dataset × seed × estimator)
python slurm/aggregate.py                # aggregate once jobs finish
```

After aggregation, per dataset:

- `result/result_<dataset>.csv` — long format, one row per `(dataset, model, inference_rule, metric, seed)`.
- `result/result_<dataset>_summary.csv` — mean ± std across seeds.
- `result/result_<dataset>_crosstab.csv` — pivot: rows = `(model × inference_rule)`, cols = metrics, cells = `mean ± std`.

### Benchmarks

| Dataset | L | N | Source | Notes |
|---|---:|---:|---|---|
| `flags` | 7 | 194 | MULAN | |
| `emotions` | 6 | 593 | MULAN | |
| `scene` | 6 | 2407 | MULAN | |
| `CHD_49` | 6 | 555 | MULAN | |
| `Water-quality` | 14 | 1060 | MULAN | HPC recommended |
| `yeast` | 14 | 2417 | MULAN | HPC recommended |
| `VirusGO_sparse` | 6 | 207 | MULAN | sparse ARFF (`src/arff_dataset.py`) |
| `PlantPseAAC` | 12 | 978 | MULAN | sparse ARFF |
| `chest_xray_nih__{densenet,resnet,resnetae}` | 8 | ~112k | NIH ChestX-ray14 | pre-extracted `.npy` features — see [`src/chest_xray_dataset/Readme.md`](src/chest_xray_dataset/Readme.md) |

For `L = 14`, exact enumeration over `2^L = 16 384` label vectors is required. The batched implementation makes this tractable on a single CPU; for `L ≥ 18`, approximate inference is recommended.

The NIH ChestX-ray14 features are **not redistributed** in this repository (50 MB+ per backbone). To regenerate `datasets/nih_feature_vectors_{densenet,resnet,resnetae}.npy`, follow the instructions in `src/chest_xray_dataset/Readme.md`.

---

## Project layout

```
src/
  probability_classifier_chains.py   # PCC, BinaryRelevance, predict_* rules
  evaluation_metrics.py              # all metrics (binary + ranking)
  arff_dataset.py                    # MULAN ARFF loader + 10-fold CV
  utils.py                           # result aggregation
  skmultiflow/                       # vendored ClassifierChain base
  chest_xray_dataset/                # NIH feature loader
inference_evaluate_models.py         # pipeline: train × predict × evaluate
slurm/                               # cluster submission + aggregation
tests/                               # 46 unit tests + equivalence test
datasets/                            # ARFF + .npy feature files
result/                              # per-dataset CSVs
```

---

## Testing

```bash
python -m pytest tests/ -v
```

The suite contains **46 unit tests** covering every metric (including documented edge cases) and **parametrised equivalence tests** that verify the batched predictor against the brute-force reference at L ∈ {2, 3, 5, 7}. The corrected Markedness and Informedness rules are validated against enumeration-based optima on the same suite.

---

## Limitations and scope

- **Exact enumeration** over `2^L` scales poorly for `L ≥ 18` even with the batched implementation (memory ≈ `N · 2^L · 8` bytes for the joint cache). Approximate inference (e.g. ε-A, beam search, Monte-Carlo) is the natural extension and is intentionally outside the scope of this artifact.
- All base estimators must implement `predict_proba`. `SGDClassifier` is admissible with `loss = "log_loss"`; probability calibration is the user's responsibility.

---

## References

- K. Dembczyński, W. Cheng, E. Hüllermeier. **Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains**. *ICML 2010*.
- K. Dembczyński, W. Waegeman, W. Cheng, E. Hüllermeier. **An Exact Algorithm for F-Measure Maximization**. *NeurIPS 2011*.
- R. E. Schapire, Y. Singer. **BoosTexter: A Boosting-Based System for Text Categorization**. *Machine Learning, 39(2/3):135–168, 2000*. (one-error, coverage, ranking loss, average precision.)
- D. M. W. Powers. **Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation**. *Journal of Machine Learning Technologies, 2(1):37–63, 2011*.
- G. Tsoumakas, I. Katakis, I. Vlahavas. **Mining Multi-label Data**. *Data Mining and Knowledge Discovery Handbook*, 2010. (MULAN.)

## License

MIT — see [LICENSE](LICENSE).
