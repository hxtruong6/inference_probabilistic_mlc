# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-06-08

Internal refactor and performance work on the inference core. **Results are
unchanged** — the full target × evaluation table is byte-for-byte identical to
1.2.0 across the tabular datasets (verified end-to-end on emotions, CHD-49,
scene, and Water-quality over multiple seeds). The documented public API
(`predict_*`, `predict`, `EvaluationMetrics`) is preserved.

### Added
- `ProbabilisticClassifierChain.compute_stats(X, needs=…, batch_size=…)` and the
  `InferenceStats` dataclass: compute only the probabilistic statistics a rule
  needs, skipping the `2^L` joint entirely for the trivial Recall/NPV rules.
- Pure Bayes-optimal predictor functions (`bop_hamming`, `bop_subset`,
  `bop_precision`, `bop_npv`, `bop_recall`, `bop_markedness`, `bop_fmeasure`).
- Typed registry dataclasses `EvalMetric` and `InferenceRule`.
- Optional sample-chunking via `batch_size` / the `DACAF_INFERENCE_BATCH_SIZE`
  environment variable, to bound peak memory on large-N datasets. Off by default.
- Type hints across the public inference API; module/paper-formula documentation.
- `DatasetSpec` registry (`dacaf_mlc.datasets.DATASET_SPECS`): a single source of
  truth for each dataset's label count, file orientation, loader, and
  default-sweep membership, replacing the previously scattered `_LABEL_COUNTS`,
  `DATASET_WHOLE_FILES`, and `DATASET_WHOLE_FILES_TARGET_AT_FIRST` lists.
- Built-in CSV loader (`_load_csv`) with positional or by-name label columns
  (`label_columns`) and a configurable separator (`csv_sep`).
- `docs/EXTENDING.md`: how to add a new evaluation metric, inference target, or
  dataset (ARFF / CSV / custom source), each as a small registry-based change,
  plus the duck-typed dataloader contract.

### Changed
- Vectorized `predict_fmeasure` (~10× faster at L=10), `predict_precision`,
  `bop_markedness`, and the example-based evaluation metrics — all bit-identical
  to the previous implementations (pinned by regression tests).
- The pipeline computes inference statistics once per fold and dispatches the
  rules via the registry, replacing the previous string-based `getattr` dispatch.
- Logging via the `logging` module instead of `print`; metric errors now
  propagate instead of being silently dropped.
- Invalid input raises `ValueError` instead of a bare `Exception`.
- `MultiLabelArffDataset` takes the label count via an explicit `n_labels`
  argument (supplied by the dataset registry) instead of an internal lookup
  table; the in-memory `X=`/`Y=` path is unchanged and infers it from `Y`.

### Removed
- The undocumented model-held prediction cache (`set_cache_key`,
  `prediction_cache`, `cache_key`). Inference no longer keeps fold-specific state
  on the estimator; the pipeline reuses one `compute_stats` call per fold instead.

### Fixed
- Shared mutable default `base_estimator` — each instance now gets its own
  estimator instead of one module-level instance.

## [1.2.0] - 2026-06-08

- Reproducible DaCaF package, core class rename. See the release notes / Zenodo.

[1.3.0]: https://github.com/hxtruong6/inference_probabilistic_mlc/releases/tag/v1.3.0
[1.2.0]: https://github.com/hxtruong6/inference_probabilistic_mlc/releases/tag/v1.2.0
