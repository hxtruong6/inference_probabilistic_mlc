#!/usr/bin/env bash
# Loop submitter: one sbatch per (dataset × seed × estimator).
#
# Defaults reproduce the paper's full sweep. Override any axis with env vars:
#   DATASETS="emotions scene" SEEDS="1 2" ESTIMATORS="lr" ./scripts/submit_all.sh
#
# Per-dataset overrides for chest_xray (heavier) are baked into the script
# to bump time/mem; tweak below if your cluster needs different limits.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p scripts/logs

DATASETS="${DATASETS:-flags emotions scene CHD_49 VirusGO_sparse PlantPseAAC Water-quality yeast chest_xray_nih__densenet chest_xray_nih__resnet chest_xray_nih__resnetae}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ESTIMATORS="${ESTIMATORS:-lr rf adaboost}"
OUTPUT_DIR="${OUTPUT_DIR:-result}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-inference_mlc_env}"

# Snapshot job names already in the queue so we don't double-submit combos
# that are pending/running. Resubmitting a finished combo is also a no-op
# because we skip when the per-job CSV already exists.
in_queue="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"

submitted=0
skipped_done=0
skipped_queued=0
blocked=0
for dataset in ${DATASETS}; do
    # Heavier resources for L=14 + the 112k-sample chest-xray rows.
    case "${dataset}" in
        chest_xray_nih__*)        time_lim="48:00:00"; mem="64G"; cpus="16" ;;
        yeast|Water-quality)      time_lim="24:00:00"; mem="64G"; cpus="10" ;;
        *)                        time_lim="04:00:00"; mem="24G"; cpus="10" ;;
    esac

    for seed in ${SEEDS}; do
        for est in ${ESTIMATORS}; do
            job_name="pcc_${dataset}_s${seed}_${est}"
            out_csv="${OUTPUT_DIR}/${dataset}/seed${seed}_${est}.csv"

            if [[ -f "${out_csv}" ]]; then
                skipped_done=$((skipped_done + 1))
                continue
            fi
            if grep -qx "${job_name}" <<<"${in_queue}"; then
                skipped_queued=$((skipped_queued + 1))
                continue
            fi

            if sbatch \
                --job-name="${job_name}" \
                --time="${time_lim}" \
                --mem="${mem}" \
                --cpus-per-task="${cpus}" \
                --export=ALL,DATASET="${dataset}",SEED="${seed}",ESTIMATOR="${est}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV_NAME}" \
                scripts/run_eval.sbatch >/dev/null 2>&1
            then
                submitted=$((submitted + 1))
                in_queue="${in_queue}"$'\n'"${job_name}"
            else
                # Most likely cause: QOSMaxSubmitJobPerUserLimit. Stop the
                # loop early so the next invocation retries from here.
                blocked=$((blocked + 1))
                echo "sbatch refused ${job_name}; stopping early (re-run later to pick up the rest)." >&2
                break 3
            fi
        done
    done
done

echo "Submitted: ${submitted}  | already-done (CSV present): ${skipped_done}  | already-queued: ${skipped_queued}  | blocked: ${blocked}"
echo "Logs: scripts/logs/. Re-run this script to submit any remaining combos. When all complete:"
echo "    python scripts/aggregate.py --result-dir ${OUTPUT_DIR}"
