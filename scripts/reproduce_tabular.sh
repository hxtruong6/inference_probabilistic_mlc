#!/usr/bin/env bash
# Reproduce the paper's tabular results (the tractable subset; the ChestX-ray
# image experiments need pre-extracted features and a GPU — see the README).
#
# Runs PCC + Logistic Regression over the 5 tabular datasets × 5 seeds with
# 10-fold CV, then aggregates per-seed CSVs into the paper-format tables.
#
# Usage:  bash scripts/reproduce_tabular.sh [OUTPUT_DIR]
set -euo pipefail

OUTPUT_DIR="${1:-result}"
DATASETS=(emotions CHD_49 scene Water-quality yeast)
SEEDS=(1 2 3 4 5)

echo "Reproducing tabular results into '${OUTPUT_DIR}' (PCC + LR, 10-fold, seeds ${SEEDS[*]})"
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "==> ${dataset}  seed=${seed}"
    dacaf-mlc --dataset "${dataset}" --seed "${seed}" --estimator lr --output-dir "${OUTPUT_DIR}"
  done
done

echo "==> Aggregating"
python slurm/aggregate.py --result-dir "${OUTPUT_DIR}"
echo "Done. See ${OUTPUT_DIR}/result_<dataset>_crosstab.csv for the target × evaluation tables."
