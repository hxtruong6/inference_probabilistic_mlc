# Refactor plan — final release

Scope (approved): **full restructure**, **paper + extras kept**. Package name: `dacaf_mlc`.
Invariant: the 56-test suite (esp. `tests/test_inference_optimality.py`, the brute-force
BOP optimality proofs) must stay green after every phase. Paper reproduction on Emotions
(NPV=100, Recall=100, Markedness≈84, F1≈66) must be preserved.

## Phase 0 — Safety net  ✅ DONE
- [x] `git tag pre-refactor`
- [x] Golden signatures test (`test_paper_table_signatures`) locks paper reproduction
      (NPV/Recall all-ones, vacuous=1 diagonals, precision single-label).

## Phase 1 — Packaging & structure
- [x] 1a. `git mv src dacaf_mlc`; fixed imports, slurm shim, Makefile. `pyproject.toml`
      (setuptools, package=`dacaf_mlc`); `pip install -e .` works.
- [x] 1b. Moved driver → `dacaf_mlc/evaluate.py`; console entry point `dacaf-mlc`; thin
      root shim kept for backward compatibility.
- [ ] 1c. Split the driver internals into `inference/`, `metrics/`, `data/`, `pipeline/`.
      (Behavior-preserving moves only; verify against golden test.)  ← REMAINING
- [x] 1d. Vendored `skmultiflow/` made a proper package (added `__init__.py`); excluded
      from lint. (Optional: move under `_vendor/` later.)

## Phase 2 — Dependencies  ✅ DONE
- [x] Removed macOS-specific `environment.yml`; deps now in `pyproject.toml`.
- [x] Optional `[image]` extra for torch/torchvision/torchxrayvision; core stays light
      (verified: importing the CLI needs no torch).
- [x] Modernised Dockerfile (python:3.10-slim, `pip install .`, `CMD dacaf-mlc`).

## Phase 3 — Correctness & paper fidelity  ✅ DONE
- [x] `CONVENTIONS.md`: the vacuous-convention asymmetry + the informedness appendix-bug
      note (per-label rule is the correct BOP).
- [x] README scope reconciled: paper-first headline + "Beyond the paper" section.

## Phase 4 — Reproducibility  ← REMAINING
- [ ] `configs/paper.yaml` (PCC+LR, seeds, 10-fold, scaling). `make reproduce`.
- [ ] Document/script NIH ChestX-ray feature regeneration; ship checksums not the .npy.

## Phase 5 — CI & quality  ✅ DONE
- [x] GitHub Actions (`.github/workflows/ci.yml`): ruff lint + pytest on py3.10/3.12.
- [x] ruff configured (excludes vendored + data-prep scripts; ignores E741 for `l`); clean.

## Phase 6 — Docs & release
- [x] `CITATION.cff` (DOI 10.1016/j.inffus.2026.104517).
- [ ] CONTRIBUTING, API reference; version tag, Zenodo code DOI.  ← REMAINING

Order: 0 → 1 → 2 → (3,5 parallel) → 4 → 6.
