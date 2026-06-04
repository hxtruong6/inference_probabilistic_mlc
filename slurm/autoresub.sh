#!/usr/bin/env bash
# Watch the sweep; for any pcc_* combo whose CSV is missing and whose job is
# not currently queued, look up its most recent failure mode in sacct and
# resubmit with an appropriate fix:
#   OUT_OF_MEMORY -> double mem (cap 192G)
#   TIMEOUT       -> double walltime (cap 96h)
#   FAILED/other  -> retry once at same resources, log the .err tail
# Exits when all 165 CSVs exist or after MAX_TICKS iterations.
#
# Resource baselines come from slurm/submit_all.sh (must match).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
LOG="slurm/logs/autoresub.log"
RETRY_DB="slurm/logs/autoresub_retries.tsv"   # name<TAB>attempts<TAB>last_mem<TAB>last_time
touch "${RETRY_DB}"

DATASETS="flags emotions scene CHD_49 VirusGO_sparse PlantPseAAC Water-quality yeast chest_xray_nih__densenet chest_xray_nih__resnet chest_xray_nih__resnetae"
SEEDS="1 2 3 4 5"
ESTIMATORS="lr rf adaboost"
SLEEP_BETWEEN_TICKS="${SLEEP_BETWEEN_TICKS:-300}"
MAX_TICKS="${MAX_TICKS:-144}"   # 144*5min = 12h

baseline_resources() {
    case "$1" in
        chest_xray_nih__*)        echo "48:00:00 64G 16" ;;
        yeast|Water-quality)      echo "24:00:00 64G 10" ;;
        *)                        echo "04:00:00 24G 10" ;;
    esac
}

mem_to_gb()    { echo "${1%G}"; }
time_to_hours() { local t="$1"; echo "${t%%:*}"; }

bump_mem_gb()  { local g="$1"; local d=$((g*2)); (( d > 192 )) && d=192; echo "${d}G"; }
bump_time()    { local h; h=$(time_to_hours "$1"); local d=$((h*2)); (( d > 96 )) && d=96; printf "%02d:00:00" "${d}"; }

log()  { echo "[$(date -Is)] $*" >> "${LOG}"; }

resubmit_combo() {
    local ds="$1" seed="$2" est="$3"
    local name="pcc_${ds}_s${seed}_${est}"
    local csv="result/${ds}/seed${seed}_${est}.csv"
    [[ -f "${csv}" ]] && return 0

    # already queued? skip.
    if grep -qx "${name}" /tmp/autoresub_inqueue.$$; then
        return 0
    fi

    read -r base_time base_mem base_cpus <<<"$(baseline_resources "${ds}")"

    # look up most recent terminal state for this job name
    local last_state mem time cpus
    last_state="$(sacct -u "$USER" -X -n -o State --name "${name}" 2>/dev/null \
                  | awk 'NF{print}' | tail -1 | awk '{print $1}')"

    # current resource targets default to baseline; bump based on failure mode
    mem="${base_mem}"; time="${base_time}"; cpus="${base_cpus}"

    # pull last attempt's resources from our retry DB if present
    local prev_line prev_mem prev_time
    prev_line="$(awk -F'\t' -v n="${name}" '$1==n{print}' "${RETRY_DB}" | tail -1)"
    if [[ -n "${prev_line}" ]]; then
        prev_mem="$(echo "${prev_line}" | cut -f3)"
        prev_time="$(echo "${prev_line}" | cut -f4)"
        mem="${prev_mem:-${mem}}"
        time="${prev_time:-${time}}"
    fi

    case "${last_state:-NONE}" in
        OUT_OF_MEMORY|OUT_OF_ME*)
            mem="$(bump_mem_gb "$(mem_to_gb "${mem}")")"
            log "OOM detected for ${name}; bumping mem -> ${mem}"
            ;;
        TIMEOUT)
            time="$(bump_time "${time}")"
            log "TIMEOUT detected for ${name}; bumping time -> ${time}"
            ;;
        FAILED|NODE_FAIL|CANCELLED*|"")
            # log .err tail if we can find it
            local err
            err="$(ls -t "slurm/logs/${name}"_*.err 2>/dev/null | head -1)"
            if [[ -n "${err}" ]]; then
                log "FAILED ${name} (state=${last_state}); last .err tail:"
                tail -3 "${err}" | sed 's/^/    /' >> "${LOG}"
            else
                log "FAILED ${name} (state=${last_state}); no .err file yet"
            fi
            ;;
        NONE)
            log "${name} never ran before; submitting at baseline ${mem}/${time}"
            ;;
        *)
            log "${name} unknown last_state=${last_state}; retry at ${mem}/${time}"
            ;;
    esac

    # cap retries per combo
    local attempts
    attempts="$(awk -F'\t' -v n="${name}" '$1==n{print $2}' "${RETRY_DB}" | tail -1)"
    attempts="${attempts:-0}"
    if (( attempts >= 5 )); then
        log "${name} has had ${attempts} attempts; giving up."
        return 0
    fi

    if sbatch --job-name="${name}" --time="${time}" --mem="${mem}" --cpus-per-task="${cpus}" \
              --export=ALL,DATASET="${ds}",SEED="${seed}",ESTIMATOR="${est}",OUTPUT_DIR=result,CONDA_ENV=inference_mlc_env \
              slurm/run_eval.sbatch >/dev/null 2>>"${LOG}"
    then
        attempts=$((attempts + 1))
        printf "%s\t%d\t%s\t%s\n" "${name}" "${attempts}" "${mem}" "${time}" >> "${RETRY_DB}"
        log "submitted ${name} attempt=${attempts} mem=${mem} time=${time} cpus=${cpus}"
        echo "${name}" >> /tmp/autoresub_inqueue.$$   # mark so the next loop iter doesn't double-submit
    else
        log "sbatch refused ${name} (likely QOS); will retry next tick"
    fi
}

log "autoresub started (PID $$)"
trap 'rm -f /tmp/autoresub_inqueue.$$' EXIT

for tick in $(seq 1 "${MAX_TICKS}"); do
    done_n=$(find result -name 'seed*_*.csv' ! -name '*_crosstab.csv' 2>/dev/null | wc -l)
    if (( done_n >= 165 )); then
        log "all 165 CSVs present, exiting."
        break
    fi

    squeue -h -u "$USER" -o '%j' > /tmp/autoresub_inqueue.$$

    for ds in ${DATASETS}; do
        for s in ${SEEDS}; do
            for e in ${ESTIMATORS}; do
                resubmit_combo "${ds}" "${s}" "${e}"
            done
        done
    done

    queued_n=$(grep -c '^pcc_' /tmp/autoresub_inqueue.$$ || true)
    log "tick ${tick}: done=${done_n} queued=${queued_n}"
    sleep "${SLEEP_BETWEEN_TICKS}"
done

log "autoresub exiting after ${tick} ticks."
