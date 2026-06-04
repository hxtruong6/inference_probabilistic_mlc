# Refactor plan — final release

Scope (approved): **full restructure**, **paper + extras kept**. Package name: `dacaf_mlc`.
Invariant: the 56-test suite (esp. `tests/test_inference_optimality.py`, the brute-force
BOP optimality proofs) must stay green after every phase. Paper reproduction on Emotions
(NPV=100, Recall=100, Markedness≈84, F1≈66) must be preserved.

## Phase 0 — Safety net  ✅ tag + golden test
- [ ] `git tag pre-refactor`
- [ ] Golden end-to-end test: Emotions, 1 fold, fixed seed → assert target×eval crosstab
      matches a committed reference (locks paper reproduction).

## Phase 1 — Packaging & structure  (foundational)
- [ ] 1a. `git mv src dacaf_mlc`; update the 12 `from src.` import lines, slurm sbatch,
      Makefile. Add `pyproject.toml` (setuptools, package=`dacaf_mlc`). `pip install -e .`.
- [ ] 1b. Move `inference_evaluate_models.py` → `dacaf_mlc/evaluate.py`; expose a console
      entry point `dacaf-mlc` (argparse subcommands: `evaluate`, `aggregate`). Keep a thin
      root shim for backward compatibility.
- [ ] 1c. Split the driver internals into `inference/`, `metrics/`, `data/`, `pipeline/`.
      (Behavior-preserving moves only; verify against golden test.)
- [ ] 1d. Decide vendored `skmultiflow/`: keep slim vendored copy under
      `dacaf_mlc/_vendor/skmultiflow` with LICENSE/attribution.

## Phase 2 — Dependencies
- [ ] Remove macOS-specific `environment.yml` conda lock; define deps in `pyproject.toml`.
- [ ] Optional extra `[image]` for torch/torchvision/torchxrayvision; core stays light.

## Phase 3 — Correctness & paper fidelity
- [ ] `CONVENTIONS.md`: the intentional vacuous-convention asymmetry (precision sklearn vs
      NPV/markedness/F vacuous=1) + the informedness appendix-bug note (per-label rule is
      the correct BOP).
- [ ] README scope reconcile: paper-first headline, single "Beyond the paper" section.

## Phase 4 — Reproducibility
- [ ] `configs/paper.yaml` (PCC+LR, seeds, 10-fold, scaling). `make reproduce`.
- [ ] Document/script NIH ChestX-ray feature regeneration; ship checksums not the .npy.

## Phase 5 — CI & quality
- [ ] GitHub Actions: fast (no-torch) test suite + ruff lint/format; pre-commit config.

## Phase 6 — Docs & release
- [ ] `CITATION.cff` (DOI 10.1016/j.inffus.2026.104517); CONTRIBUTING; API reference.
- [ ] Version, tag, Zenodo code DOI.

Order: 0 → 1 → 2 → (3,5 parallel) → 4 → 6.
