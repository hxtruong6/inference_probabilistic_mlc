# Extending DaCaF

DaCaF separates three concerns so each can be extended on its own, usually by
adding **data to a registry** rather than editing control flow:

- **Evaluation metrics** — how a prediction is *scored* (`evaluation_metrics.py`,
  registered in `metrics_registry.py`).
- **Inference rules (targets)** — the Bayes-optimal predictor you *optimise for*
  (`probability_classifier_chains.py`, registered in `metrics_registry.py`).
- **Datasets** — what you run on (`datasets.py`).

The pipeline (`pipeline.py`) and CLI (`evaluate.py`) iterate these registries
generically, so adding a column, a row, or a dataset never requires touching
them. Everything below is covered by the test suite — write the test first
(`tests/`), watch it fail, then implement.

---

## 1. Add a new evaluation metric

Use this when you only want to **measure** a metric, not optimise for it. It
becomes a new **column** in every dataset's crosstab, scored against all
inference rules. Example: Jaccard.

**Step 1 — add the function** to `EvaluationMetrics` in
`dacaf_mlc/evaluation_metrics.py`, matching the existing vectorised style and
declaring your vacuous (empty-prediction) convention explicitly:

```python
@staticmethod
def jaccard(y_true, y_pred):
    EvaluationMetrics._check_dimensions(y_true, y_pred)
    inter = (y_true * y_pred).sum(axis=1)
    union = ((y_true + y_pred) > 0).sum(axis=1)
    j = inter / np.where(union == 0, 1, union)   # safe denominator
    j = np.where(union == 0, 1.0, j)             # both empty → 1 (your choice)
    return j.mean()
```

**Step 2 — register it** in `dacaf_mlc/metrics_registry.py`:

```python
BINARY_METRICS = (
    ...
    EvalMetric("Jaccard", EvaluationMetrics.jaccard),
)
```

Metrics with parameters use the `options` dict, e.g.
`EvalMetric("F2", EvaluationMetrics.f_beta, {"beta": 2})` — the pipeline passes
`**options` to the function.

**Test first:** add a `tests/test_metrics_vectorized.py`-style test comparing a
plain reference loop to the vectorised version on random binary arrays, plus the
vacuous edge cases (all-zero true/pred).

---

## 2. Add a new inference rule (target)

Use this for a new Bayes-optimal predictor — a new **row** you optimise for.
The refactor split this into two independent decisions: *what the rule
computes* and *what probabilistic statistics it needs*.

**Step 1 — write a pure predictor** in
`dacaf_mlc/probability_classifier_chains.py`. Its only input is an
`InferenceStats`; it returns an `(n, L)` binary array:

```python
def bop_myrule(stats):
    # available fields, populated according to the rule's `needs` (below):
    #   stats.marginals       p_j = P(y_j = 1 | x)           (N, L)
    #   stats.pairwise        P(y_j = 1, |y| = s | x)        (N, L, L+1)
    #   stats.map_prediction  argmax_y P(y | x)              (N, L)
    #   stats.p_empty, stats.p_full                          (N, 1)
    p = stats.marginals
    return np.where(<your decision rule>, 1, 0)
```

**Step 2 — register it** in `PREDICT_FUNCTIONS` (`metrics_registry.py`),
declaring the minimal statistics it consumes via `needs`:

```python
InferenceRule("Predict MyRule", bop_myrule, frozenset({"marginal"}), BINARY_METRICS),
```

`needs` is the key to cost. The pipeline computes the **union** of all rules'
`needs` once per fold and shares it, so a rule that only needs `"marginal"`
never materialises the `2^L` joint and adds essentially zero extra inference
cost if another rule already required marginals.

**Optional — a library method.** The `pcc.predict_*` methods are thin
hand-written wrappers; add one if you want `pcc.predict_myrule(X)` for direct
library use:

```python
def predict_myrule(self, X):
    return bop_myrule(self.compute_stats(X, needs={"marginal"}))
```

**The one core change to be aware of:** the `needs` vocabulary is a closed set
(`"map"`, `"marginal"`, `"pairwise"`). If your rule needs a *new* kind of
statistic (e.g. triplet probabilities), you must add a field to `InferenceStats`
and compute it in `compute_stats` / `_compute_stats_batch` — a real change to
the inference core, not just registration.

**Test first:** verify optimality against brute-force enumeration of the
expected metric over all `2^L` candidates, in the style of
`tests/test_inference_optimality.py`.

---

## 3. Add a new dataset

Datasets live in one registry: `DATASET_SPECS` in `dacaf_mlc/datasets.py`. A
`DatasetSpec` records the label count, file orientation, loader, and whether the
dataset is part of the default sweep — so `DEFAULT_DATASET_NAMES`, the label
counts, and the loader dispatch are all derived from this single list.

### A standard MULAN ARFF (the common case)

Drop `datasets/mydata.arff` in place and append one spec:

```python
DATASET_SPECS = (
    ...
    DatasetSpec("mydata", 8, _load_arff, note="L=8"),
    # labels in the LEADING columns instead of trailing? add target_at_first=True
    # not part of the default sweep? add in_default_sweep=False
)
```

Then run `dacaf-mlc --dataset mydata --seed 1`. No other file changes — the CLI
`--dataset` choices and the no-arg sweep both read from the registry.

### A CSV dataset

A built-in `_load_csv` ships in `datasets.py`. Drop `datasets/mydata.csv` in
place and register it; labels can be positional or selected by column name:

```python
from dacaf_mlc.datasets import _load_csv, DatasetSpec

# positional: last 8 columns are labels (or target_at_first=True for the first 8)
DatasetSpec("mydata", 8, _load_csv, note="L=8")

# labels by explicit name (features = all other columns); custom separator
DatasetSpec("mydata", 2, _load_csv,
            label_columns=("disease_a", "disease_b"), csv_sep=";")
```

`label_columns` must have length `n_labels` (validated at construction). Label
columns are cast to `int`; everything else becomes the float feature matrix.

### A non-ARFF / non-CSV custom source

If your data isn't a whole-file ARFF (e.g. pre-extracted feature `.npy`, like
the ChestX-ray datasets), write a small loader and reference it from the spec.
A loader takes `(spec, folder_path)` and returns a `MultiLabelArffDataset`:

```python
def _load_mysource(spec, folder_path):
    X_df, Y_df = my_loading_logic(spec.name, folder_path)   # both pandas DataFrames
    return MultiLabelArffDataset(dataset_name=spec.name, X=X_df, Y=Y_df)

DATASET_SPECS = (
    ...
    DatasetSpec("mysource__variant", 12, _load_mysource, note="L=12"),
)
```

The `MultiLabelArffDataset(X=, Y=)` data path infers `L` from `Y` and applies no
scaling (the pipeline scales per fold to avoid leakage). `_load_nih_features` in
`datasets.py` is the worked example to copy.

### The dataloader contract

A loader is just `loader(spec, folder_path) -> dataset_handler`. Using
`MultiLabelArffDataset` is the convenient shortcut, but **not required** — the
pipeline only depends on this duck-typed interface, so any object providing it
works:

| Member | Type / shape | Notes |
|---|---|---|
| `.dataset_name` | `str` | used as the result CSV key |
| `.X` | `ndarray (n, d)`, float | **raw / unscaled** — the pipeline applies `StandardScaler` per fold |
| `.Y` | `ndarray (n, L)`, binary int | multi-label targets |
| `.get_cross_validation_folds(n_splits, random_state)` | yields `(train_idx, test_idx)` | e.g. `sklearn.model_selection.KFold` |

Return raw features (no scaling) so the pipeline can fit the scaler on the
training fold only and avoid leakage.

**Test first:** the registry invariants are pinned in
`tests/test_dataset_registry.py` (label counts, orientation, sweep order, error
on unknown names) — add your dataset's expected values there.

---

## Summary

| You want to… | Touch | Core change? | Effort |
|---|---|---|---|
| Score a new metric | `evaluation_metrics.py` + 1 line in `metrics_registry.py` | No | trivial |
| New target using existing stats (marginal/pairwise/map) | one `bop_*` fn + 1 line in `metrics_registry.py` | No | small |
| New target needing a *new* statistic | above **+** `compute_stats` / `InferenceStats` | **Yes** | medium |
| New ARFF dataset | 1 `DatasetSpec` in `datasets.py` | No | trivial |
| New CSV dataset | 1 `DatasetSpec` with `_load_csv` | No | trivial |
| New custom-source dataset | a loader fn + 1 `DatasetSpec` | No | small |

See [`CONVENTIONS.md`](CONVENTIONS.md) for the exact metric definitions and
vacuous conventions, and [`REPRODUCING.md`](REPRODUCING.md) for the full
experimental protocol.
