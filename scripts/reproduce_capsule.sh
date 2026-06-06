#!/usr/bin/env bash
# Reproducibility-capsule run (Code Ocean): the paper's CHD-49 result table.
#
# PCC + Logistic Regression, 10-fold CV, seeds 1-5 — CPU-only, finishes in
# seconds. Produces <out>/result_CHD_49_crosstab.csv (target x evaluation),
# which matches the CHD-49 table in the metapaper (every diagonal entry is the
# maximum of its column — the paper's central claim).
#
# Usage:  bash scripts/reproduce_capsule.sh [OUTPUT_DIR]   (default: /results)
set -euo pipefail

OUTPUT_DIR="${1:-/results}"
DATASET=CHD_49
SEEDS=(1 2 3 4 5)
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${OUTPUT_DIR}"
echo "Reproducing ${DATASET} into '${OUTPUT_DIR}' (PCC + LR, 10-fold, seeds ${SEEDS[*]})"
for seed in "${SEEDS[@]}"; do
  echo "==> ${DATASET}  seed=${seed}"
  dacaf-mlc --dataset "${DATASET}" --seed "${seed}" --estimator lr --output-dir "${OUTPUT_DIR}"
done

echo "==> Aggregating"
python "${HERE}/aggregate.py" --result-dir "${OUTPUT_DIR}"
echo "Done. See ${OUTPUT_DIR}/result_${DATASET}_crosstab.csv"
