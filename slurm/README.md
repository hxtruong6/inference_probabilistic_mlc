# Slurm sweep driver

This directory turns `inference_evaluate_models.py` into a per-job entry point
and submits one Slurm job per **(dataset × seed × estimator)** combination.

## Files

| File | Purpose |
|---|---|
| `run_eval.sbatch` | One-job sbatch template. Reads `DATASET`, `SEED`, `ESTIMATOR`, `OUTPUT_DIR`, `CONDA_ENV` from env. JAIST defaults (partition `compute`, 10 cpus, 32G, 24h) — adjust the `#SBATCH` block. |
| `submit_all.sh` | Loops over `(DATASETS × SEEDS × ESTIMATORS)` and submits one `sbatch` per combo with per-dataset resource overrides (heavier wall-time/mem for chest-xray + L=14). |
| `aggregate.py` | Walks `result/<dataset>/seed*_*.csv` and produces `result_<dataset>.csv` (long), `result_<dataset>_summary.csv` (mean ± std), and `result_<dataset>_crosstab.csv` (pivot). |
| `logs/` | Per-job stdout/stderr (`<job>_<id>.{out,err}`). |

## Typical run

```bash
# 1. Submit the full paper sweep (11 datasets × 5 seeds × 3 estimators = 165 jobs).
./slurm/submit_all.sh

# 2. Watch progress.
squeue -u $USER
tail -f slurm/logs/pcc_emotions_s1_lr_*.out

# 3. When all jobs are COMPLETE, aggregate per-job CSVs into per-dataset
#    summaries with mean ± std across seeds.
python slurm/aggregate.py --result-dir result
```

## Per-job invocation (no Slurm)

The same entry point runs locally — useful for debugging one combo:

```bash
python inference_evaluate_models.py \
    --dataset emotions --seed 1 --estimator lr --output-dir result
```

Output lands at `result/emotions/seed1_lr.csv` (+ `_crosstab.csv`).

## Customising the sweep

Override any axis at submission time via env vars to `submit_all.sh`:

```bash
# Only LR baseline on the two L=14 MULAN datasets, 3 seeds:
DATASETS="Water-quality yeast" SEEDS="1 2 3" ESTIMATORS="lr" ./slurm/submit_all.sh

# Single dataset, all estimators, seeds 1..10:
DATASETS="emotions" SEEDS="$(seq 1 10)" ./slurm/submit_all.sh
```

## Resource notes (defaults baked into `submit_all.sh`)

| Dataset class | Time | Mem | CPUs |
|---|---|---|---|
| `chest_xray_nih__*` (N≈112k, L=8) | 48h | 64G | 16 |
| `yeast`, `Water-quality` (L=14)   | 24h | 32G | 10 |
| Everything else                   |  4h | 16G |  8 |

`inference_evaluate_models.py` honours `SLURM_CPUS_PER_TASK` for joblib's fold
parallelism — set `--cpus-per-task` to the number of folds you want in
parallel (max 10 with `KFOLD_SPLIT_NUMBER=10`).

## What about reproducibility?

Each per-job CSV is keyed by `(dataset, seed, estimator-tag)`; rerunning a
failed job with the same env vars overwrites only that CSV. The aggregator
picks up whatever seeds happen to exist, so partial sweeps still produce
valid (lower-`N`) summaries.
