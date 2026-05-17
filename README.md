# Bayes-Optimal Inference for Probabilistic Classifier Chains

Reference implementation and evaluation framework for **Bayes-optimal inference rules** in multi-label classification under Probabilistic Classifier Chains (PCC) [[Dembczyński et al., 2010]](#references). The framework implements eight loss-specific inference rules, a Binary Relevance baseline, and a comprehensive metric suite (example-based, label-based, and ranking-based), and reports stratified 10-fold cross-validation results on standard MULAN benchmarks.

## Method

### Models

| Model | Joint distribution | Reference |
|---|---|---|
| `ProbabilisticClassifierChainCustom` (PCC) | Chain rule: P(y \| x) = Π_j P(y_j \| x, y_<j) | [Dembczyński et al., 2010] |
| `BinaryRelevance` (BR) | Independence: P(y \| x) = Π_j P(y_j \| x) | Standard baseline |

Both classes expose the same inference rules; the only difference is how the joint label distribution is constructed. This isolates the contribution of chain-aware inference. Each can wrap any sklearn-compatible binary base estimator (we evaluate LogisticRegression, RandomForestClassifier, AdaBoostClassifier).

### Bayes-optimal inference rules

Given joint P(y | x), each rule returns the prediction ŷ that maximises the expected value of its target metric. L = number of labels.

| Method | Optimises | Rule |
|---|---|---|
| `predict_hamming` | Hamming Accuracy | ŷ_j = 1 ⇔ P(y_j=1 \| x) > 0.5 |
| `predict_subset` | Subset Accuracy (0/1 loss) | ŷ = arg max_y P(y \| x) over all 2^L vectors |
| `predict_precision` | Precision | Predict only the label with the highest marginal |
| `predict_recall` | Recall | ŷ = 1 (trivially optimal; recall = 1) |
| `predict_npv` | NPV | All ones except the label with the lowest marginal |
| `predict_markedness` | Markedness ½·(Precision + NPV) | arg max over candidate top-l predictions; see § Derivations |
| `predict_fmeasure` | F-β | Pairwise probability aggregation [Dembczyński et al., 2011] |
| `predict_informedness` | Informedness ½·(Sens. + Spec.) | Include label j iff q_sens[j] + q_spec_cost[j] > C; see § Derivations |
| `predict_marginal_scores` | — (continuous) | Returns marginals as scores; consumed by ranking metrics |

### Derivations (corrections relative to the original codebase)

Two inference rules required mathematical correction:

- **`predict_markedness`** — the original closed form assumed Precision = 1 in the degenerate case |ŷ| = 0, ∃ y_j = 1 (vacuous precision). This contradicts the sklearn convention used by the evaluator, where the same case returns 0. We re-derived the expected markedness under the sklearn convention. For the candidate "predict the top-l labels by marginal" with marginals π_j sorted descending and A_l = Σ_{i≤l} π_(i):

  - l = 0:        E[M | x] = ½ · (P(y = 0 | x) + 1 − Σ π_j / L)
  - 0 < l < L:    E[M | x] = ½ · (A_l / l + 1 − (Σ π_j − A_l) / (L − l))
  - l = L:        E[M | x] = ½ · (Σ π_j / L + P(y = 1 | x))

  Verified against brute-force enumeration on toy cases (`tests/`).

- **`predict_informedness`** — the original implementation produced all-zero predictions because E[0] was set to 1 + P(y=0|x), which always dominated. We derived the correct rule directly from the joint pairwise distribution P(y_j = 1, |y| = s | x):

  Include label j iff q_sens[j] + q_spec_cost[j] > C, where
  - q_sens[j] = Σ_{s=1..L} P(y_j=1, |y|=s | x) / s
  - q_spec_cost[j] = Σ_{s=1..L-1} P(y_j=1, |y|=s | x) / (L−s)
  - C = P(|y|=0 | x) / L + Σ_{s=1..L-1} P(|y|=s | x) / (s · (L−s))

  Verified against brute-force on five samples (all five matched the enumeration-based optimum).

### Metrics

All metrics are reported in **higher-is-better** form (e.g. `coverage_score` = 1 − normalised coverage error).

**Binary-prediction metrics** (applied to each `predict_*` output):
`hamming_accuracy`, `subset_accuracy`, `precision_score`, `recall_score`, `negative_predictive_value`, `f_beta`, `macro_f1`, `micro_f1`, `markedness`, `informedness`.

**Ranking metrics** (applied to marginal probability scores, [Schapire & Singer, 2000]):
`one_error_score`, `coverage_score`, `ranking_loss_score`, `average_precision_score`.

Edge-case conventions match scikit-learn (`zero_division=0`) and are documented in each metric's docstring. The example-based Precision/NPV use the convention "vacuous = 0 when there are unpredicted positives", which is the standard in the multi-label literature; this conflicts with the original PCC paper's derivation of `predict_markedness`, hence the re-derivation above.

## Reproducibility

### Environment

```bash
conda create -n inference_prob_mlc python=3.10
conda activate inference_prob_mlc
pip install -r requirements.txt
```

Pinned versions (relevant subset): `numpy==1.26.4`, `pandas==2.0.3`, `scikit-learn==1.2.2`, `scipy==1.12.0`, `joblib==1.3.2`.

### Hyperparameters and protocol

| Setting | Value |
|---|---|
| Random seed (global) | `SEED = 6` |
| Cross-validation | KFold, `n_splits = 10`, `shuffle = True` |
| Feature scaling | `StandardScaler` fit on training fold only (no leakage) |
| LogisticRegression | `max_iter = 5 × 10⁶`, default L2 regularisation |
| RandomForestClassifier | `n_estimators = 100`, `n_jobs = 1` (joblib outer parallelism) |
| AdaBoostClassifier | `n_estimators = 50` |
| Folds run in parallel via | `joblib.Parallel(n_jobs=-1)` |

### Running the evaluation

**Single combo (local):**

```bash
python inference_evaluate_models.py \
    --dataset emotions --seed 1 --estimator lr --output-dir result
```

Writes `result/<dataset>/seed<S>_<est-tag>.csv` (+ `_crosstab.csv`) for that one (dataset, seed, estimator) combination.

**Full paper sweep (no args):**

```bash
python inference_evaluate_models.py
```

Iterates over `DEFAULT_DATASET_NAMES × DEFAULT_SEEDS × all estimators`. Practical only for small datasets on a laptop; for the full sweep use Slurm (below).

**Slurm cluster:** see [`slurm/README.md`](slurm/README.md). One job per (dataset × seed × estimator); aggregate with `python slurm/aggregate.py` once jobs finish.

Output (per dataset, after aggregation):

- `result/result_<dataset>.csv` — long format: one row per (dataset, model, inference_rule, metric, seed)
- `result/result_<dataset>_summary.csv` — mean ± std across seeds
- `result/result_<dataset>_crosstab.csv` — pivot: rows = (model × inference_rule), cols = metrics, cells = "mean ± std"

### Datasets

| Dataset | L | N | Source | Status |
|---|---:|---:|---|---|
| `flags` | 7 | 194 | MULAN | supported |
| `emotions` | 6 | 593 | MULAN | supported |
| `scene` | 6 | 2407 | MULAN | supported |
| `CHD_49` | 6 | 555 | MULAN | supported |
| `Water-quality` | 14 | 1060 | MULAN | supported, HPC recommended |
| `yeast` | 14 | 2417 | MULAN | supported, HPC recommended |
| `VirusGO_sparse` | 6 | 207 | MULAN | supported (sparse ARFF, parsed by `src/arff_dataset.py`) |
| `PlantPseAAC` | 12 | 978 | MULAN | supported (sparse ARFF) |
| `chest_xray_nih__{densenet,resnet,resnetae}` | 8 | ~112k | NIH ChestX-ray14 | supported via pre-extracted `.npy` features; see [`src/chest_xray_dataset/Readme.md`](src/chest_xray_dataset/Readme.md) |

For L = 14, exact enumeration over 2^L = 16 384 label vectors is required. The batched implementation (below) makes this tractable on a single CPU; for L ≥ 18, approximate inference is recommended.

The NIH ChestX-ray14 features are not redistributed in this repo (50 MB+ per backbone). To regenerate `datasets/nih_feature_vectors_{densenet,resnet,resnetae}.npy`, follow the instructions in `src/chest_xray_dataset/Readme.md`.

## Implementation

### Prefix-tree batched joint inference

The naive PCC predictor invokes the base estimator N · L · 2^L times (per sample, per label position, per candidate vector). The implementation in this repository performs a **single batched `predict_proba` call per chain level**, exploiting the fact that all candidate vectors share the prefix-tree structure of the chain. At chain level j, all N · 2^j (sample, prefix) pairs are passed to the j-th base classifier together; outputs are expanded via an interleaved sample-major layout that preserves the canonical MSB-first index ↔ label-vector mapping.

Total `predict_proba` calls: **N · L · 2^L → L**. Empirical speedup over the reference enumeration:

| L | Speedup (50 samples, 10 features) |
|---:|---|
| 6 | ~300× |
| 10 | ~8000× |

Verified numerically equivalent (atol = 1e-12) by `tests/test_predict_equivalence.py` over L ∈ {2, 3, 5, 7}.

### Project layout

```
src/
  probability_classifier_chains.py   # PCC, BinaryRelevance, predict_* rules
  evaluation_metrics.py              # all metrics (binary + ranking)
  arff_dataset.py                    # MULAN ARFF loader + 10-fold CV
  utils.py                           # result aggregation
  skmultiflow/                       # vendored ClassifierChain base
  chest_xray_dataset/                # NIH feature loader
inference_evaluate_models.py         # pipeline: train × predict × evaluate
tests/                               # 46 unit tests + equivalence test
datasets/                            # ARFF + .npy feature files
result/                              # per-dataset CSVs
```

## Testing

```bash
python -m pytest tests/ -v
```

46 unit tests cover all metrics (edge cases included) and 5 parametrised tests verify the optimized predict against the brute-force reference.

## Limitations

- **Exact enumeration** over 2^L scales poorly for L ≥ 18 even with the batched implementation (memory ≈ N · 2^L · 8 bytes for the joint cache). Approximate inference (e.g. ε-A or beam search) is the natural extension and is not implemented here.
- All base estimators must implement `predict_proba`. `SGDClassifier` works with `loss="log_loss"`; calibration is the user's responsibility.

## References

- K. Dembczyński, W. Cheng, E. Hüllermeier. **Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains**. *ICML 2010*.
- K. Dembczyński, W. Waegeman, W. Cheng, E. Hüllermeier. **An Exact Algorithm for F-Measure Maximization**. *NeurIPS 2011*.
- R. E. Schapire, Y. Singer. **BoosTexter: A Boosting-Based System for Text Categorization**. *Machine Learning, 39(2/3):135–168, 2000*. (Definitions of one-error, coverage, ranking loss, average precision.)
- D. M. W. Powers. **Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation**. *Journal of Machine Learning Technologies, 2(1):37–63, 2011*.
- G. Tsoumakas, I. Katakis, I. Vlahavas. **Mining Multi-label Data**. *Data Mining and Knowledge Discovery Handbook*, 2010. (MULAN datasets.)

## License

MIT — see [LICENSE](LICENSE).
