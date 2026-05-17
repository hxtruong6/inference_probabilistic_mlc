#!/bin/bash
# One Slurm job = one (DATASET, SEED, ESTIMATOR) combo.
# Submitted by slurm/submit_all.sh; do not invoke directly without env vars.
#
# JAIST HPC defaults — adjust partition / time / mem / account for your project.
#SBATCH --job-name=pcc_eval
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

: "${DATASET:?DATASET env var required (e.g. DATASET=emotions)}"
: "${SEED:?SEED env var required (e.g. SEED=1)}"
: "${ESTIMATOR:=all}"
: "${OUTPUT_DIR:=result}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate the conda env that has the pinned requirements; adjust as needed.
# Example for JAIST:
#   module load anaconda3
#   source activate inference_prob_mlc
if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[$(date -Is)] DATASET=${DATASET} SEED=${SEED} ESTIMATOR=${ESTIMATOR}"
echo "[$(date -Is)] CPUS=${SLURM_CPUS_PER_TASK:-?} MEM=${SLURM_MEM_PER_NODE:-?}"

python inference_evaluate_models.py \
    --dataset   "${DATASET}" \
    --seed      "${SEED}" \
    --estimator "${ESTIMATOR}" \
    --output-dir "${OUTPUT_DIR}"

echo "[$(date -Is)] done"
