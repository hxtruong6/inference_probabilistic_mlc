# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest + ruff   (add ,image for ChestX-ray)
```

## Before opening a PR

```bash
ruff check dacaf_mlc tests       # lint (must be clean)
pytest tests/ -q                 # all tests must pass
```

CI runs the same checks on Python 3.10 and 3.12.

## Ground rules

- **The paper is the source of truth.** Every inference rule must maximise the
  expected value of its target metric. New or changed rules must be covered by a
  brute-force optimality check in `tests/test_inference_optimality.py`
  (`argmax_ŷ E[metric]` over all `2^L` predictions).
- Respect the documented metric conventions in `CONVENTIONS.md` (notably the
  intentional vacuous-convention asymmetry, and the corrected Informedness rule).
- Keep vendored code (`dacaf_mlc/skmultiflow/`) unmodified.
- Match the surrounding style; `l` is allowed as a variable (paper notation for
  the prediction size).
