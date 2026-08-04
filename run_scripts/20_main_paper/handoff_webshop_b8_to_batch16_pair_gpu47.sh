#!/usr/bin/env bash
# Atomically hand GPUs 4-7 from the live WebShop batch-8 baseline to two
# fresh Qwen3.5-4B batch-16 runs:
#   1. WebShop, 1,600 tasks, one epoch, 100 steps
#   2. ALFWorld, 1,600 tasks, one epoch, 100 steps
#
# The queue blocks on the same lane lock held by the current trainer.  It never
# kills a trainer, Ray process, rollout server, or environment service.  Every
# transition is fail-closed on exact process ownership and complete artifacts.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env_config.sh"

CURRENT_EXPERIMENT="webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200"
WEBSHOP_EXPERIMENT="webshop_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100"
ALFWORLD_EXPERIMENT="alfworld_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100"
WEBSHOP_CONFIG="config/duet_paper_experiments_configs/iclr2027/webshop/${WEBSHOP_EXPERIMENT}.yaml"
ALFWORLD_CONFIG="config/duet_paper_experiments_configs/iclr2027/alfworld/${ALFWORLD_EXPERIMENT}.yaml"
VERIFY_SCRIPT="scripts/verify_qwen35_batch16_contracts.py"

CURRENT_QUEUE_PID="${CURRENT_QUEUE_PID:?set the exact live batch-8 queue PID}"
CURRENT_LAUNCHER_PID="${CURRENT_LAUNCHER_PID:?set the exact live batch-8 launcher PID}"
CURRENT_MAIN_PID="${CURRENT_MAIN_PID:?set the exact live batch-8 main_ppo PID}"
CURRENT_RAYLET_PID="${CURRENT_RAYLET_PID:?set the exact live batch-8 raylet PID}"
CURRENT_TASKRUNNER_PID="${CURRENT_TASKRUNNER_PID:?set the exact live batch-8 TaskRunner PID}"
CURRENT_GPU_WORKER_PIDS_CSV="${CURRENT_GPU_WORKER_PIDS:?set four live batch-8 GPU worker PIDs}"
ROLLOUT_GPU_PIDS_CSV="${GPU47_ROLLOUT_GPU_PIDS:?set four live rollout EngineCore PIDs}"
IFS=',' read -r -a CURRENT_GPU_WORKER_PIDS <<< "${CURRENT_GPU_WORKER_PIDS_CSV}"
IFS=',' read -r -a ROLLOUT_GPU_PIDS <<< "${ROLLOUT_GPU_PIDS_CSV}"

LANE_GPUS="4,5,6,7"
CURRENT_RAY_ROOT="/data/ray/ws35s200"
LANE_LOCK="logs/.gpu47_training_lane.lock"
QUEUE_LOCK="logs/.gpu47_batch16_pair.lock"
STOP_FILE="logs/STOP_GPU47_BATCH16_PAIR"
ARM_MARKER="logs/.gpu47_batch16_pair_seed2025_armed_at"
QUEUE_MANIFEST="logs/gpu47_batch16_pair_seed2025.manifest"
ROLLOUT_MANIFEST=""
QUEUE_STATE="ACTIVE"

mkdir -p logs
exec 8>"${QUEUE_LOCK}"
if ! flock -n 8; then
    echo "FATAL: another GPU4-7 batch-16 pair queue owns ${QUEUE_LOCK}." >&2
    exit 1
fi
if [ -e "${ARM_MARKER}" ] || [ -e "${QUEUE_MANIFEST}" ]; then
    echo "FATAL: prior batch-16 queue evidence exists; refusing ambiguous re-arm." >&2
    exit 1
fi
: > "${QUEUE_MANIFEST}"

log_event() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_MANIFEST}"
}

on_exit() {
    local rc=$?
    if [ "${QUEUE_STATE}" != "COMPLETE" ]; then
        printf '[%s] status=FAILED exit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" >> "${QUEUE_MANIFEST}"
    fi
}
trap on_exit EXIT

fatal() {
    log_event "status=FAILED reason=$*"
    exit 1
}

process_instance_alive() {
    local pid="$1" expected_tick="$2" state observed_tick
    [ -r "/proc/${pid}/stat" ] || return 1
    state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
    observed_tick="$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || true)"
    [ "${state}" != "Z" ] && [ "${observed_tick}" = "${expected_tick}" ]
}

require_exact_process() {
    local pid="$1" expected="$2" label="$3" cmdline
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || fatal "invalid ${label} PID ${pid}"
    [ -r "/proc/${pid}/cmdline" ] || fatal "${label} PID ${pid} is not alive"
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    [[ "${cmdline}" == *"${expected}"* ]] || fatal "PID ${pid} is not the expected ${label}"
}

start_tick() {
    awk '{print $22}' "/proc/$1/stat"
}

check_http() {
    local url="$1" label="$2"
    curl -fsS --connect-timeout 2 --max-time 10 "${url}" >/dev/null || \
        fatal "${label} is not healthy at ${url}"
}

select_rollout_manifest() {
    local candidate
    shopt -s nullglob
    for candidate in logs/rollout_servers_8211_*.manifest; do
        if [ -z "${ROLLOUT_MANIFEST}" ] || [ "${candidate}" -nt "${ROLLOUT_MANIFEST}" ]; then
            ROLLOUT_MANIFEST="${candidate}"
        fi
    done
    shopt -u nullglob
    [ -n "${ROLLOUT_MANIFEST}" ] || fatal "no rollout manifest for port 8211"
}

require_manifest_field() {
    grep -aFxq "$1" "${ROLLOUT_MANIFEST}" || \
        fatal "rollout manifest lacks required field: $1"
}

check_rollout_contract() {
    select_rollout_manifest
    require_manifest_field "model_dir=/data/shared_models/Qwen3.5-4B-think"
    require_manifest_field "gpus=4,5,6,7"
    require_manifest_field "base_port=8211"
    require_manifest_field "max_model_len=32768"
    require_manifest_field "max_num_seqs=1"
    require_manifest_field "gpu_memory_utilization=0.25"
    require_manifest_field "logprobs_mode=processed_logprobs"
    require_manifest_field "enforce_eager=0"
    require_manifest_field "vllm_enable_fla_packed_recurrent_decode=0"
    require_manifest_field "health_verified_ports=8211 8212 8213 8214"
}

check_rollout_health() {
    local port
    for port in 8211 8212 8213 8214; do
        check_http "http://127.0.0.1:${port}/health" "external vLLM port ${port}"
    done
}

check_webshop_health() {
    check_http "http://127.0.0.1:36003/" "AgentGym WebShop"
    check_http "http://127.0.0.1:8083/healthz" "WebShop env_service"
}

check_alfworld_health() {
    check_http "http://127.0.0.1:36001/" "AgentGym ALFWorld"
    check_http "http://127.0.0.1:8081/healthz" "ALFWorld env_service"
}

sorted_gpu_pids() {
    nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed 's/[[:space:]]//g' | sort -n | paste -sd, -
}

sorted_csv() {
    printf '%s\n' "$1" | tr ',' '\n' | sort -n | paste -sd, -
}

check_gpu_lane_with_current_workers() {
    local offset gpu expected observed
    for offset in 0 1 2 3; do
        gpu=$((offset + 4))
        expected="$(sorted_csv "${ROLLOUT_GPU_PIDS[${offset}]},${CURRENT_GPU_WORKER_PIDS[${offset}]}")"
        observed="$(sorted_gpu_pids "${gpu}")"
        [ "${observed}" = "${expected}" ] || \
            fatal "GPU${gpu} arm ownership mismatch: expected ${expected}, observed ${observed:-none}"
    done
}

check_gpu_lane_rollouts_only() {
    local offset gpu expected observed pid
    for offset in 0 1 2 3; do
        gpu=$((offset + 4))
        pid="${ROLLOUT_GPU_PIDS[${offset}]}"
        process_instance_alive "${pid}" "${ROLLOUT_START_TICKS[${offset}]}" || \
            fatal "GPU${gpu} rollout EngineCore instance ${pid} changed or exited"
        expected="${pid}"
        observed="$(sorted_gpu_pids "${gpu}")"
        [ "${observed}" = "${expected}" ] || \
            fatal "GPU${gpu} is not exclusively owned by rollout PID ${expected}; observed ${observed:-none}"
    done
}

check_fresh_output() {
    local benchmark="$1" experiment="$2"
    [ ! -e "checkpoints/agentevolver/${experiment}" ] || fatal "checkpoint output already exists for ${experiment}"
    [ ! -e "experiments/${benchmark}/${experiment}" ] || fatal "experiment output already exists for ${experiment}"
    [ ! -e "launcher_record/${experiment}" ] || fatal "launcher record already exists for ${experiment}"
    [ ! -e "logs/${experiment}.log" ] || fatal "training log already exists for ${experiment}"
}

verify_config_contracts() {
    python "${VERIFY_SCRIPT}" --webshop "${WEBSHOP_CONFIG}" --alfworld "${ALFWORLD_CONFIG}" || \
        fatal "batch-16 config contract preflight failed"
}

preflight_wandb() {
    local experiment="$1"
    python -c \
        "from agentevolver.utils.tracking import preflight_wandb_online; preflight_wandb_online(project_name='agentevolver', experiment_name='${experiment}')" || \
        fatal "online W&B preflight failed for ${experiment}"
}

source_contract_hash() {
    sha256sum \
        "${BASH_SOURCE[0]}" \
        env_config.sh \
        launcher.py \
        config/agentevolver.yaml \
        external/config_fallback/ppo_trainer.yaml \
        config/duet_paper_experiments_configs/iclr2027/webshop/webshop_qwen35_4b_grpo_snapshot_gate.yaml \
        config/duet_paper_experiments_configs/iclr2027/webshop/webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200.yaml \
        config/duet_paper_experiments_configs/iclr2027/alfworld/alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v4_serial_decode_safe.yaml \
        agentevolver/main_ppo.py \
        agentevolver/module/context_manager/context_policy.py \
        agentevolver/module/context_manager/cmt_linear.py \
        agentevolver/module/env_manager/env_manager.py \
        agentevolver/module/exp_manager/het_actor.py \
        agentevolver/module/exp_manager/het_fsdp_worker.py \
        agentevolver/module/trainer/ae_ray_trainer.py \
        agentevolver/module/trainer/external_llm_server_manager.py \
        agentevolver/utils/tracking.py \
        "${WEBSHOP_CONFIG}" \
        "${ALFWORLD_CONFIG}" \
        "${VERIFY_SCRIPT}" \
        | sha256sum | awk '{print $1}'
}

wait_instance_exit() {
    local pid="$1" tick="$2" label="$3" attempts=0
    while process_instance_alive "${pid}" "${tick}"; do
        [ "${attempts}" -lt 60 ] || fatal "${label} did not exit within 10 minutes"
        attempts=$((attempts + 1))
        sleep 10
    done
}

wait_ray_root_clear() {
    local ray_root="$1" attempts=0
    while pgrep -af -- "${ray_root}/session_" >/dev/null 2>&1; do
        [ "${attempts}" -lt 60 ] || fatal "Ray processes under ${ray_root} did not exit within 10 minutes"
        attempts=$((attempts + 1))
        sleep 10
    done
}

validate_validation_file() {
    local path="$1" expected_step="$2"
    python -c \
        'import json,sys; p=sys.argv[1]; step=int(sys.argv[2]); rows=[json.loads(x) for x in open(p,encoding="utf-8") if x.strip()]; assert len(rows)==200,len(rows); assert {int(r.get("step")) for r in rows}=={step}; ids=[str(r["task_id"]) for r in rows]; assert len(set(ids))==200,len(set(ids))' \
        "${path}" "${expected_step}" || fatal "invalid fixed-200 validation artifact ${path}"
}

validate_complete_run() {
    local benchmark="$1" experiment="$2" step="$3" marker="$4"
    local log="logs/${experiment}.log"
    local val="experiments/${benchmark}/${experiment}/validation_log/${step}.jsonl"
    local ckpt_root="checkpoints/agentevolver/${experiment}"
    local actor_dir="${ckpt_root}/global_step_${step}/actor"
    local step_count model_count optim_count extra_count sync_line test_line wandb_url

    [ -f "${log}" ] || fatal "missing log for ${experiment}"
    step_count="$(grep -acF "step:${step} -" "${log}" || true)"
    [ "${step_count}" -eq 1 ] || fatal "${experiment} has ${step_count} completed step-${step} metric lines"
    if grep -aEq 'Traceback|CUDA out of memory|OutOfMemoryError|non-finite|Error executing job|Error running subprocess|rollout identity gate failed' "${log}"; then
        fatal "${experiment} log contains a hard failure marker"
    fi
    grep -aFq "wandb: Syncing run ${experiment}" "${log}" || \
        fatal "${experiment} lacks an online W&B syncing record"
    sync_line="$(awk '/external rollout weight sync verified/{print NR; exit}' "${log}")"
    test_line="$(awk '/Epoch test/{print NR; exit}' "${log}")"
    [ -n "${sync_line}" ] && [ -n "${test_line}" ] && [ "${sync_line}" -lt "${test_line}" ] || \
        fatal "${experiment} did not prove base-weight sync before initial validation"

    [ -f "${val}" ] || fatal "missing final validation ${val}"
    validate_validation_file "${val}" "${step}"
    [ "${val}" -nt "${marker}" ] || fatal "${val} is not newer than its launch marker"
    [ -d "${actor_dir}" ] || fatal "missing final actor checkpoint ${actor_dir}"
    [ "${ckpt_root}/global_step_${step}" -nt "${marker}" ] || \
        fatal "final checkpoint for ${experiment} is not newer than its launch marker"
    [ "$(tr -d '[:space:]' < "${ckpt_root}/latest_checkpointed_iteration.txt")" = "${step}" ] || \
        fatal "latest checkpoint marker for ${experiment} is not ${step}"
    model_count="$(find "${actor_dir}" -maxdepth 1 -type f -name 'model_world_size_4_rank_*.pt' -size +1G | wc -l)"
    optim_count="$(find "${actor_dir}" -maxdepth 1 -type f -name 'optim_world_size_4_rank_*.pt' -size +1G | wc -l)"
    extra_count="$(find "${actor_dir}" -maxdepth 1 -type f -name 'extra_state_world_size_4_rank_*.pt' | wc -l)"
    [ "${model_count}" -eq 4 ] && [ "${optim_count}" -eq 4 ] && [ "${extra_count}" -eq 4 ] || \
        fatal "incomplete checkpoint shards for ${experiment}: model=${model_count} optim=${optim_count} extra=${extra_count}"

    wandb_url="$(grep -aoE 'https://wandb.ai/[^[:space:]]+/runs/[[:alnum:]]+' "${log}" | tail -1 || true)"
    log_event "verified=${experiment} validation_sha256=$(sha256sum "${val}" | awk '{print $1}') checkpoint_shards=${model_count}+${optim_count}+${extra_count} wandb_url=${wandb_url:-missing}"
}

validate_batch16_first_step() {
    local experiment="$1" log="logs/${experiment}.log" line
    line="$(grep -aF 'step:1 -' "${log}" | tail -1 || true)"
    [[ "${line}" == *"training/num_not_none_traj:128.000"* ]] || fatal "${experiment} step 1 did not contain 128 trajectories"
    [[ "${line}" == *"rollout/behavior/onpolicy_sample_count:128.000"* ]] || fatal "${experiment} step 1 on-policy count is not 128"
    [[ "${line}" == *"duet/group_total_count:16.000"* ]] || fatal "${experiment} step 1 group count is not 16"
}

launch_one() {
    local benchmark="$1" experiment="$2" config="$3" ray_prefix="$4" env_check="$5"
    local ray_root run_marker run_lock run_fd rc
    check_fresh_output "${benchmark}" "${experiment}"
    [ "$(source_contract_hash)" = "${SOURCE_CONTRACT_SHA}" ] || fatal "source/config contract changed after queue arm"
    verify_config_contracts
    check_rollout_health
    check_rollout_contract
    "${env_check}"
    preflight_wandb "${experiment}"
    check_gpu_lane_rollouts_only

    run_marker="logs/.${experiment}.started_at"
    [ ! -e "${run_marker}" ] || fatal "stale launch marker ${run_marker}"
    touch "${run_marker}"
    ray_root="$(mktemp -d -p /data/ray "${ray_prefix}.XXXXXX")"
    [ "${#ray_root}" -le 30 ] || fatal "Ray root is too long for AF_UNIX sockets: ${ray_root}"
    run_lock="logs/.${experiment}.lock"
    exec {run_fd}>"${run_lock}"
    flock -n "${run_fd}" || fatal "another launcher owns ${run_lock}"

    log_event "launching=${experiment} benchmark=${benchmark} ray_root=${ray_root} config_sha256=$(sha256sum "${config}" | awk '{print $1}')"
    set +e
    CUDA_VISIBLE_DEVICES="${LANE_GPUS}" RAY_TMPDIR="${ray_root}" \
        python launcher.py --conf "${config}" > "logs/${experiment}.log" 2>&1
    rc=$?
    set -e
    log_event "exited=${experiment} rc=${rc}"
    [ "${rc}" -eq 0 ] || fatal "${experiment} launcher exited rc=${rc}"
    wait_ray_root_clear "${ray_root}"
    check_gpu_lane_rollouts_only
    validate_complete_run "${benchmark}" "${experiment}" 100 "${run_marker}"
    validate_batch16_first_step "${experiment}"
    exec {run_fd}>&-
}

if [ "${#CURRENT_GPU_WORKER_PIDS[@]}" -ne 4 ] || [ "${#ROLLOUT_GPU_PIDS[@]}" -ne 4 ]; then
    fatal "queue requires exactly four current workers and four rollout EngineCore PIDs"
fi

unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE
unset WANDB_DISABLED
export WANDB_MODE=online
CONDA_ENV_TRAIN="${CONDA_ENV_DUET2:-duet2}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_TRAIN}"

require_exact_process "${CURRENT_QUEUE_PID}" "run_iclr_gpu47_queue.sh" "current queue"
require_exact_process "${CURRENT_LAUNCHER_PID}" "${CURRENT_EXPERIMENT}" "current launcher"
require_exact_process "${CURRENT_MAIN_PID}" "agentevolver.main_ppo" "current main_ppo"
require_exact_process "${CURRENT_RAYLET_PID}" "${CURRENT_RAY_ROOT}/session_" "current raylet"
require_exact_process "${CURRENT_TASKRUNNER_PID}" "ray::TaskRunner.run" "current TaskRunner"
for offset in 0 1 2 3; do
    require_exact_process "${CURRENT_GPU_WORKER_PIDS[${offset}]}" "ray::WorkerDict" "current GPU$((offset + 4)) worker"
    require_exact_process "${ROLLOUT_GPU_PIDS[${offset}]}" "VLLM::EngineCore" "GPU$((offset + 4)) rollout engine"
done

CURRENT_QUEUE_TICK="$(start_tick "${CURRENT_QUEUE_PID}")"
CURRENT_LAUNCHER_TICK="$(start_tick "${CURRENT_LAUNCHER_PID}")"
CURRENT_MAIN_TICK="$(start_tick "${CURRENT_MAIN_PID}")"
CURRENT_RAYLET_TICK="$(start_tick "${CURRENT_RAYLET_PID}")"
CURRENT_TASKRUNNER_TICK="$(start_tick "${CURRENT_TASKRUNNER_PID}")"
CURRENT_GPU_WORKER_START_TICKS=()
ROLLOUT_START_TICKS=()
for offset in 0 1 2 3; do
    CURRENT_GPU_WORKER_START_TICKS+=("$(start_tick "${CURRENT_GPU_WORKER_PIDS[${offset}]}")")
    ROLLOUT_START_TICKS+=("$(start_tick "${ROLLOUT_GPU_PIDS[${offset}]}")")
done

[ ! -e "experiments/webshop/${CURRENT_EXPERIMENT}/validation_log/200.jsonl" ] || \
    fatal "current step-200 validation already existed before queue arm"
[ ! -e "checkpoints/agentevolver/${CURRENT_EXPERIMENT}/global_step_200" ] || \
    fatal "current step-200 checkpoint already existed before queue arm"
check_fresh_output webshop "${WEBSHOP_EXPERIMENT}"
check_fresh_output alfworld "${ALFWORLD_EXPERIMENT}"
verify_config_contracts
check_rollout_health
check_rollout_contract
check_webshop_health
check_alfworld_health
check_gpu_lane_with_current_workers
preflight_wandb "${WEBSHOP_EXPERIMENT}"
preflight_wandb "${ALFWORLD_EXPERIMENT}"
SOURCE_CONTRACT_SHA="$(source_contract_hash)"

touch "${ARM_MARKER}"
log_event "status=ARMED current=${CURRENT_EXPERIMENT} current_queue_pid=${CURRENT_QUEUE_PID} current_launcher_pid=${CURRENT_LAUNCHER_PID} current_main_pid=${CURRENT_MAIN_PID} current_raylet_pid=${CURRENT_RAYLET_PID} current_taskrunner_pid=${CURRENT_TASKRUNNER_PID} current_workers=${CURRENT_GPU_WORKER_PIDS_CSV} rollout_pids=${ROLLOUT_GPU_PIDS_CSV}"
log_event "source_contract_sha256=${SOURCE_CONTRACT_SHA} rollout_manifest=${ROLLOUT_MANIFEST} rollout_manifest_sha256=$(sha256sum "${ROLLOUT_MANIFEST}" | awk '{print $1}')"
log_event "waiting_for_lane_lock=${LANE_LOCK}"

exec 7>"${LANE_LOCK}"
flock 7
log_event "status=LANE_ACQUIRED"

if [ -e "${STOP_FILE}" ]; then
    fatal "STOP file detected before WebShop batch-16 launch"
fi

wait_instance_exit "${CURRENT_QUEUE_PID}" "${CURRENT_QUEUE_TICK}" "current queue"
wait_instance_exit "${CURRENT_LAUNCHER_PID}" "${CURRENT_LAUNCHER_TICK}" "current launcher"
wait_instance_exit "${CURRENT_MAIN_PID}" "${CURRENT_MAIN_TICK}" "current main_ppo"
wait_instance_exit "${CURRENT_RAYLET_PID}" "${CURRENT_RAYLET_TICK}" "current raylet"
wait_instance_exit "${CURRENT_TASKRUNNER_PID}" "${CURRENT_TASKRUNNER_TICK}" "current TaskRunner"
for offset in 0 1 2 3; do
    wait_instance_exit "${CURRENT_GPU_WORKER_PIDS[${offset}]}" "${CURRENT_GPU_WORKER_START_TICKS[${offset}]}" "current GPU$((offset + 4)) worker"
done
wait_ray_root_clear "${CURRENT_RAY_ROOT}"
check_gpu_lane_rollouts_only
validate_complete_run webshop "${CURRENT_EXPERIMENT}" 200 "${ARM_MARKER}"

launch_one webshop "${WEBSHOP_EXPERIMENT}" "${WEBSHOP_CONFIG}" wsb16 check_webshop_health

if [ -e "${STOP_FILE}" ]; then
    fatal "STOP file detected before ALFWorld batch-16 launch"
fi
launch_one alfworld "${ALFWORLD_EXPERIMENT}" "${ALFWORLD_CONFIG}" afb16 check_alfworld_health

QUEUE_STATE="COMPLETE"
log_event "status=COMPLETE sequence=${CURRENT_EXPERIMENT},${WEBSHOP_EXPERIMENT},${ALFWORLD_EXPERIMENT}"
