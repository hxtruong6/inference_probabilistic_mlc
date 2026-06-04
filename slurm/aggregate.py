"""Aggregate per-job CSVs (one per seed × estimator) into per-dataset summaries.

Per-job CSVs are written by inference_evaluate_models.py::run_single to
    <result-dir>/<dataset>/seed<S>_<est-tag>.csv

This script walks that layout and produces, per dataset:
    <result-dir>/result_<dataset>.csv          # long format, one row per
                                               # (Dataset, Model, Predict, Metric, Seed)
    <result-dir>/result_<dataset>_summary.csv  # mean ± std across seeds (long)
    <result-dir>/result_<dataset>_crosstab.csv # pivot showing "mean ± std"
                                               # for each (model × predict) × metric

Existing single-seed CSVs (no seed suffix in filename) are skipped.
"""
import argparse
import glob
import os
import re

import pandas as pd

_SEED_RE = re.compile(r"^seed(?P<seed>\d+)_(?P<est>[A-Za-z0-9-]+)\.csv$")


def _iter_dataset_dirs(result_dir):
    for entry in sorted(os.listdir(result_dir)):
        full = os.path.join(result_dir, entry)
        if os.path.isdir(full):
            yield entry, full


def _load_per_job_csvs(dataset_dir):
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(dataset_dir, "seed*_*.csv"))):
        if csv_path.endswith("_crosstab.csv"):
            continue
        name = os.path.basename(csv_path)
        m = _SEED_RE.match(name)
        if not m:
            continue
        df = pd.read_csv(csv_path)
        df["Seed"] = int(m.group("seed"))
        df["Estimator Tag"] = m.group("est")
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _summarise(long_df):
    grouped = long_df.groupby(
        ["Dataset", "Model", "Predict Function of Model", "Metric Function"]
    )["Score"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "Mean", "std": "Std", "count": "N"})
    summary["Std"] = summary["Std"].fillna(0.0)
    return summary


def _crosstab(summary_df, out_path):
    summary_df = summary_df.copy()
    summary_df["Display"] = summary_df.apply(
        lambda r: f"{r['Mean'] * 100:.2f} ± {r['Std'] * 100:.2f}", axis=1
    )
    pivot = pd.crosstab(
        index=[summary_df["Dataset"], summary_df["Model"], summary_df["Predict Function of Model"]],
        rownames=["Dataset", "Base Model", "Predict Func"],
        columns=summary_df["Metric Function"],
        colnames=["Metric"],
        values=summary_df["Display"],
        aggfunc="first",
    )
    pivot.to_csv(out_path)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", default="result",
                    help="Root dir containing one subdir per dataset (default: result)")
    args = ap.parse_args()

    if not os.path.isdir(args.result_dir):
        raise SystemExit(f"result dir not found: {args.result_dir}")

    n_datasets = 0
    for dataset, dataset_dir in _iter_dataset_dirs(args.result_dir):
        long_df = _load_per_job_csvs(dataset_dir)
        if long_df is None:
            print(f"[skip] {dataset}: no seed*.csv files")
            continue

        long_out  = os.path.join(args.result_dir, f"result_{dataset}.csv")
        summ_out  = os.path.join(args.result_dir, f"result_{dataset}_summary.csv")
        cross_out = os.path.join(args.result_dir, f"result_{dataset}_crosstab.csv")

        long_df.to_csv(long_out, index=False)
        summary = _summarise(long_df)
        summary.to_csv(summ_out, index=False)
        _crosstab(summary, cross_out)

        seeds = sorted(long_df["Seed"].unique())
        ests = sorted(long_df["Estimator Tag"].unique())
        print(f"[{dataset}] {len(long_df)} rows | seeds={seeds} | est-tags={ests}")
        n_datasets += 1

    print(f"\nAggregated {n_datasets} dataset(s).")


if __name__ == "__main__":
    main()
