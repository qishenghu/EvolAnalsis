#!/usr/bin/env bash
# Wait for the exact ALFWorld Qwen3.5 baseline launcher, require complete
# step-100 validation/checkpoint artifacts, then hand lane B to the WebShop
# s200 baseline.  This script never kills a process and never broad-matches a
# user's GPU workload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ALFWORLD_EXPERIMENT="alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max"
ALFWORLD_LAUNCHER_PID="${ALFWORLD_LAUNCHER_PID:?set the exact current launcher PID}"
ALFWORLD_TASKRUNNER_PID="${ALFWORLD_TASKRUNNER_PID:?set the exact current TaskRunner PID}"
ALFWORLD_GPU_WORKER_PIDS_CSV="${ALFWORLD_GPU_WORKER_PIDS:?set four GPU worker PIDs in GPU4-7 order}"
ROLLOUT_GPU_PIDS_CSV="${GPU47_ROLLOUT_GPU_PIDS:?set four vLLM EngineCore PIDs in GPU4-7 order}"
IFS=',' read -r -a ALFWORLD_GPU_WORKER_PIDS <<< "${ALFWORLD_GPU_WORKER_PIDS_CSV}"
IFS=',' read -r -a ROLLOUT_GPU_PIDS <<< "${ROLLOUT_GPU_PIDS_CSV}"
ALFWORLD_LOG="logs/${ALFWORLD_EXPERIMENT}.log"
ALFWORLD_VALIDATION="experiments/alfworld/${ALFWORLD_EXPERIMENT}/validation_log/100.jsonl"
ALFWORLD_CKPT_ROOT="checkpoints/agentevolver/${ALFWORLD_EXPERIMENT}"
ALFWORLD_CKPT="${ALFWORLD_CKPT_ROOT}/global_step_100/actor"
HANDOFF_LOCK="logs/.gpu47_alfworld_to_webshop_handoff.lock"
LANE_LOCK="logs/.gpu47_training_lane.lock"
ARM_MARKER="logs/.gpu47_alfworld_to_webshop_armed_at"
STOP_FILE="logs/STOP_GPU47_HANDOFF"
WEBSHOP_QUEUE="${REPO_ROOT}/run_iclr_gpu47_queue.sh"

mkdir -p logs
exec 8>"${HANDOFF_LOCK}"
if ! flock -n 8; then
    echo "FATAL: another ALFWorld-to-WebShop handoff already owns ${HANDOFF_LOCK}." >&2
    exit 1
fi
exec 7>"${LANE_LOCK}"
if ! flock -n 7; then
    echo "FATAL: another compliant trainer owns GPU4-7 via ${LANE_LOCK}." >&2
    exit 1
fi

unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE
unset WANDB_DISABLED
export WANDB_MODE=online

process_alive() {
    local pid="$1"
    [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

require_pid_shape() {
    local pid="$1"
    local label="$2"
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: invalid ${label} PID ${pid}." >&2
        exit 1
    fi
}

require_exact_process() {
    local pid="$1"
    local expected_title="$2"
    local label="$3"
    require_pid_shape "${pid}" "${label}"
    if ! process_alive "${pid}"; then
        echo "FATAL: ${label} PID ${pid} is not alive." >&2
        exit 1
    fi
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    if [[ "${cmdline}" != *"${expected_title}"* ]]; then
        echo "FATAL: PID ${pid} is not the expected ${label}." >&2
        exit 1
    fi
}

check_gpu_lane_exact() {
    local phase="$1"
    local include_workers="$2"
    local offset gpu expected_csv observed_csv
    for offset in 0 1 2 3; do
        gpu=$((offset + 4))
        expected_csv="${ROLLOUT_GPU_PIDS[${offset}]}"
        if [ "${include_workers}" = "1" ]; then
            expected_csv="${expected_csv},${ALFWORLD_GPU_WORKER_PIDS[${offset}]}"
        fi
        observed_csv="$(
            nvidia-smi -i "${gpu}" --query-compute-apps=pid \
                --format=csv,noheader,nounits \
                | sed 's/[[:space:]]//g' \
                | sort -n \
                | paste -sd, -
        )"
        expected_csv="$(printf '%s\n' "${expected_csv}" | tr ',' '\n' | sort -n | paste -sd, -)"
        if [ "${observed_csv}" != "${expected_csv}" ]; then
            echo "FATAL: ${phase} GPU${gpu} ownership mismatch; expected exact PIDs ${expected_csv}, observed ${observed_csv:-none}." >&2
            exit 1
        fi
    done
}

wait_for_exact_process_exit() {
    local pid="$1"
    local label="$2"
    local attempts=0
    [ -n "${pid}" ] || return 0
    while process_alive "${pid}"; do
        if [ "${attempts}" -ge 20 ]; then
            echo "FATAL: ${label} PID ${pid} remained alive after the ALFWorld launcher exited; refusing overlap." >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        sleep 15
    done
}

if [ "${#ALFWORLD_GPU_WORKER_PIDS[@]}" -ne 4 ] || [ "${#ROLLOUT_GPU_PIDS[@]}" -ne 4 ]; then
    echo "FATAL: handoff requires exactly four ALFWorld workers and four rollout engines in GPU4-7 order." >&2
    exit 1
fi
require_exact_process "${ALFWORLD_LAUNCHER_PID}" "launcher.py" "ALFWorld launcher"
launcher_cmdline="$(tr '\0' ' ' < "/proc/${ALFWORLD_LAUNCHER_PID}/cmdline")"
if [[ "${launcher_cmdline}" != *"${ALFWORLD_EXPERIMENT}"* ]]; then
    echo "FATAL: ALFWorld launcher PID ${ALFWORLD_LAUNCHER_PID} is for another experiment." >&2
    exit 1
fi
require_exact_process "${ALFWORLD_TASKRUNNER_PID}" "ray::TaskRunner.run" "ALFWorld TaskRunner"
for offset in 0 1 2 3; do
    require_exact_process "${ALFWORLD_GPU_WORKER_PIDS[${offset}]}" "ray::WorkerDict" "ALFWorld GPU$((offset + 4)) worker"
    require_exact_process "${ROLLOUT_GPU_PIDS[${offset}]}" "VLLM::EngineCore" "GPU$((offset + 4)) rollout engine"
done
check_gpu_lane_exact "at arm time" 1

if [ -e "${ALFWORLD_VALIDATION}" ] || [ -e "${ALFWORLD_CKPT_ROOT}/global_step_100" ]; then
    echo "FATAL: step-100 artifacts already existed before this handoff was armed; refusing ambiguous stale evidence." >&2
    exit 1
fi

# Verify every dependency and W&B authentication now, while ALFWorld still
# owns the GPUs.  The production queue repeats this preflight at launch time.
bash "${WEBSHOP_QUEUE}" --preflight-only
touch "${ARM_MARKER}"

echo "[$(date '+%F %T %Z')] handoff armed; waiting for ALFWorld launcher PID ${ALFWORLD_LAUNCHER_PID}"
last_status=0
while process_alive "${ALFWORLD_LAUNCHER_PID}"; do
    if [ -e "${STOP_FILE}" ]; then
        echo "STOP file detected at ${STOP_FILE}; exiting without launching WebShop."
        exit 0
    fi
    now="$(date +%s)"
    if [ $((now - last_status)) -ge 300 ]; then
        echo "[$(date '+%F %T %Z')] still waiting for ALFWorld; WebShop remains preflight-ready"
        last_status="${now}"
    fi
    sleep 15
done

echo "[$(date '+%F %T %Z')] ALFWorld launcher exited; checking exact worker teardown and artifacts"
wait_for_exact_process_exit "${ALFWORLD_TASKRUNNER_PID}" "ALFWorld TaskRunner"
for offset in 0 1 2 3; do
    wait_for_exact_process_exit "${ALFWORLD_GPU_WORKER_PIDS[${offset}]}" "ALFWorld GPU$((offset + 4)) WorkerDict"
done
for offset in 0 1 2 3; do
    require_exact_process "${ROLLOUT_GPU_PIDS[${offset}]}" "VLLM::EngineCore" "GPU$((offset + 4)) rollout engine"
done
check_gpu_lane_exact "after ALFWorld teardown" 0

if [ ! -f "${ALFWORLD_VALIDATION}" ]; then
    echo "FATAL: missing step-100 validation ${ALFWORLD_VALIDATION}; WebShop will not start." >&2
    exit 1
fi
python -c \
    'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]; assert len(rows)==200, len(rows); assert {r.get("step") for r in rows}=={100}; ids=[str(r["task_id"]) for r in rows]; assert len(set(ids))==200, len(set(ids))' \
    "${ALFWORLD_VALIDATION}"
if [ ! "${ALFWORLD_VALIDATION}" -nt "${ARM_MARKER}" ] || \
   [ ! "${ALFWORLD_CKPT_ROOT}/global_step_100" -nt "${ARM_MARKER}" ]; then
    echo "FATAL: step-100 artifacts are not newer than this handoff's arm marker." >&2
    exit 1
fi

if [ "$(tr -d '[:space:]' < "${ALFWORLD_CKPT_ROOT}/latest_checkpointed_iteration.txt")" != "100" ]; then
    echo "FATAL: ALFWorld latest checkpoint marker is not 100." >&2
    exit 1
fi
if ! grep -aFq "step:100 -" "${ALFWORLD_LOG}"; then
    echo "FATAL: ALFWorld log lacks a completed step-100 metric line." >&2
    exit 1
fi
if grep -aEq 'Traceback|OutOfMemory|non-finite' "${ALFWORLD_LOG}"; then
    echo "FATAL: ALFWorld log contains a hard failure marker; inspect before launching WebShop." >&2
    exit 1
fi

model_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'model_world_size_4_rank_*.pt' -size +1G | wc -l)"
optim_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'optim_world_size_4_rank_*.pt' -size +1G | wc -l)"
extra_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'extra_state_world_size_4_rank_*.pt' | wc -l)"
if [ "${model_shards}" -ne 4 ] || [ "${optim_shards}" -ne 4 ] || [ "${extra_shards}" -ne 4 ]; then
    echo "FATAL: incomplete ALFWorld checkpoint shards: model=${model_shards}, optim=${optim_shards}, extra=${extra_shards}." >&2
    exit 1
fi

if [ -e "${STOP_FILE}" ]; then
    echo "STOP file detected after ALFWorld completion; exiting without launching WebShop."
    exit 0
fi

echo "[$(date '+%F %T %Z')] ALFWorld step 100 is complete and auditable; handing GPUs 4-7 to WebShop"
export GPU47_HANDOFF_VERIFIED=1
export GPU47_LANE_LOCK_FD=7
export GPU47_ROLLOUT_GPU_PIDS="${ROLLOUT_GPU_PIDS_CSV}"
exec bash "${WEBSHOP_QUEUE}"
