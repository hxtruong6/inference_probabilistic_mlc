#!/usr/bin/env bash
# Loop submitter: one sbatch per (dataset × seed × estimator).
#
# Defaults reproduce the paper's full sweep. Override any axis with env vars:
#   DATASETS="emotions scene" SEEDS="1 2" ESTIMATORS="lr" ./slurm/submit_all.sh
#
# Per-dataset overrides for chest_xray (heavier) are baked into the script
# to bump time/mem; tweak below if your cluster needs different limits.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

DATASETS="${DATASETS:-flags emotions scene CHD_49 VirusGO_sparse PlantPseAAC Water-quality yeast chest_xray_nih__densenet chest_xray_nih__resnet chest_xray_nih__resnetae}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ESTIMATORS="${ESTIMATORS:-lr rf adaboost}"
OUTPUT_DIR="${OUTPUT_DIR:-result}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-inference_prob_mlc}"

submitted=0
for dataset in ${DATASETS}; do
    # Heavier resources for L=14 + the 112k-sample chest-xray rows.
    case "${dataset}" in
        chest_xray_nih__*)        time_lim="48:00:00"; mem="64G"; cpus="16" ;;
        yeast|Water-quality)      time_lim="24:00:00"; mem="32G"; cpus="10" ;;
        *)                        time_lim="04:00:00"; mem="16G"; cpus="8"  ;;
    esac

    for seed in ${SEEDS}; do
        for est in ${ESTIMATORS}; do
            job_name="pcc_${dataset}_s${seed}_${est}"
            sbatch \
                --job-name="${job_name}" \
                --time="${time_lim}" \
                --mem="${mem}" \
                --cpus-per-task="${cpus}" \
                --export=ALL,DATASET="${dataset}",SEED="${seed}",ESTIMATOR="${est}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV_NAME}" \
                slurm/run_eval.sbatch
            submitted=$((submitted + 1))
        done
    done
done

echo "Submitted ${submitted} jobs. Tail logs under slurm/logs/. When all complete:"
echo "    python slurm/aggregate.py --result-dir ${OUTPUT_DIR}"
