"""Command-line entry point for the evaluation sweep.

Thin CLI over the pipeline. With no arguments it runs the full sweep over
DEFAULT_DATASET_NAMES × DEFAULT_SEEDS; with --dataset it runs a single
(dataset, seed) job (the Slurm per-job mode).

Run as ``dacaf-mlc`` (console script) or ``python -m dacaf_mlc.evaluate``.
"""
import argparse
import logging
import os

from dacaf_mlc.config import BASE_DIR, SEED, DEFAULT_SEEDS
from dacaf_mlc.datasets import DEFAULT_DATASET_NAMES
from dacaf_mlc.pipeline import ESTIMATOR_FACTORIES, run_single


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PCC inference rules. With no args, runs the full "
                    "sweep over DEFAULT_DATASET_NAMES × DEFAULT_SEEDS."
    )
    p.add_argument("--dataset", choices=DEFAULT_DATASET_NAMES,
                   help="Run a single dataset (Slurm per-job mode).")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Single-job: RNG seed used for KFold shuffling and the base estimator.")
    p.add_argument("--estimator", choices=list(ESTIMATOR_FACTORIES) + ["all"], default="all",
                   help="Single-job: base estimator to train (paper uses 'lr').")
    p.add_argument("--output-dir", default=os.path.join(BASE_DIR, "result"),
                   help="Root output dir; per-job CSVs land in <output-dir>/<dataset>/.")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=os.environ.get("DACAF_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    estimator_names = list(ESTIMATOR_FACTORIES) if args.estimator == "all" else [args.estimator]

    if args.dataset is not None:
        # Per-job mode (one row of the sweep, e.g. invoked from sbatch).
        run_single(args.dataset, args.seed, estimator_names, args.output_dir)
        return

    # Full sweep — kept for local interactive use; for large runs prefer
    # one --dataset job at a time and aggregate with scripts/aggregate.py.
    for dataset_name in DEFAULT_DATASET_NAMES:
        for seed in DEFAULT_SEEDS:
            run_single(dataset_name, seed, estimator_names, args.output_dir)


if __name__ == "__main__":
    main()
