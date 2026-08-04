#!/usr/bin/env bash
# ICLR 2027 lane-B production queue (GPUs 4-7).
#
# This runner is deliberately non-destructive: it never calls launcher --kill,
# ray stop, pkill, or a broad process cleanup.  The supervised external vLLM
# lane on 8211-8214 may be reused across same-base-model experiments; a new
# trainer force-syncs its own FSDP weights before initial validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/env_config.sh"

EXPERIMENT_NAME="webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200"
CONFIG_PATH="config/duet_paper_experiments_configs/iclr2027/webshop/${EXPERIMENT_NAME}.yaml"
LANE_GPUS="4,5,6,7"
# Ray embeds the full temp root inside AF_UNIX socket names (hard limit: 107
# bytes).  Keep this production lane name intentionally short; the original
# experiment-name path was rejected before Ray/TaskRunner initialization.
LANE_RAY_TMPDIR="${RAY_TMPDIR}/ws35s200"
TRAIN_LOG="logs/${EXPERIMENT_NAME}.log"
LOCK_FILE="logs/.${EXPERIMENT_NAME}.lock"
ROLLOUT_PORTS=(8211 8212 8213 8214)
ROLLOUT_MANIFEST=""

mkdir -p logs "${LANE_RAY_TMPDIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "FATAL: another ${EXPERIMENT_NAME} queue/launch owns ${LOCK_FILE}." >&2
    exit 1
fi

# Future experiments authenticate from a mode-0600 credential file.  Never
# let a plaintext key or token-file path enter Ray or a GPU subprocess.
unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE
unset WANDB_DISABLED
export WANDB_MODE=online

CONDA_ENV_TRAIN="${CONDA_ENV_DUET2:-duet2}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_TRAIN}"

check_http() {
    local url="$1"
    local label="$2"
    if ! curl -fsS --connect-timeout 2 --max-time 10 "${url}" >/dev/null; then
        echo "FATAL: ${label} is not healthy at ${url}." >&2
        exit 1
    fi
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
    if [ -z "${ROLLOUT_MANIFEST}" ]; then
        echo "FATAL: no rollout-server contract manifest exists for port 8211." >&2
        exit 1
    fi
}

require_manifest_field() {
    local expected="$1"
    if ! grep -aFxq "${expected}" "${ROLLOUT_MANIFEST}"; then
        echo "FATAL: rollout manifest ${ROLLOUT_MANIFEST} lacks required contract field: ${expected}" >&2
        exit 1
    fi
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

preflight() {
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "FATAL: missing production config ${CONFIG_PATH}." >&2
        exit 1
    fi
    if [ "${#LANE_RAY_TMPDIR}" -gt 30 ]; then
        echo "FATAL: Ray temp root is too long for worst-case AF_UNIX socket paths: ${LANE_RAY_TMPDIR}." >&2
        exit 1
    fi

    local port
    for port in "${ROLLOUT_PORTS[@]}"; do
        check_http "http://127.0.0.1:${port}/health" "external vLLM port ${port}"
    done
    check_rollout_contract
    check_http "http://127.0.0.1:36003/" "AgentGym WebShop server"
    check_http "http://127.0.0.1:8083/healthz" "WebShop env_service"

    python -c \
        "from agentevolver.utils.tracking import preflight_wandb_online; preflight_wandb_online(project_name='agentevolver', experiment_name='${EXPERIMENT_NAME}')"

    if [ -e "checkpoints/agentevolver/${EXPERIMENT_NAME}" ] || \
       [ -e "experiments/webshop/${EXPERIMENT_NAME}" ]; then
        echo "FATAL: output path for ${EXPERIMENT_NAME} already exists; refusing an ambiguous overwrite/resume." >&2
        exit 1
    fi
    echo "Preflight passed: WebShop services, four contract-matched rollout servers, W&B online auth, and fresh outputs."
}

preflight
if [ "${1:-}" = "--preflight-only" ]; then
    exit 0
fi

# A healthy shared rollout endpoint does not prove that the previous trainer
# has released the lane: two trainers could otherwise hot-reload different
# weights into the same four servers.  Only the artifact-checked handoff is
# allowed to enter the launch path.
if [ "${GPU47_HANDOFF_VERIFIED:-0}" != "1" ]; then
    echo "FATAL: production launch requires GPU47_HANDOFF_VERIFIED=1 from the audited ALFWorld handoff." >&2
    echo "Run run_scripts/20_main_paper/handoff_alfworld_to_webshop_qwen35_gpu47.sh instead of launching this queue directly." >&2
    exit 1
fi
if [ "${GPU47_LANE_LOCK_FD:-}" != "7" ] || [ ! -e "/proc/$$/fd/7" ]; then
    echo "FATAL: production launch lacks the inherited GPU4-7 lane lock." >&2
    exit 1
fi
if ! flock -n 7; then
    echo "FATAL: inherited GPU4-7 lane lock is not held by this handoff." >&2
    exit 1
fi

IFS=',' read -r -a rollout_gpu_pids <<< "${GPU47_ROLLOUT_GPU_PIDS:-}"
if [ "${#rollout_gpu_pids[@]}" -ne 4 ]; then
    echo "FATAL: production launch requires four exact rollout EngineCore PIDs." >&2
    exit 1
fi
for offset in 0 1 2 3; do
    gpu=$((offset + 4))
    pid="${rollout_gpu_pids[${offset}]}"
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || [ ! -e "/proc/${pid}/cmdline" ] || \
       [[ "$(tr '\0' ' ' < "/proc/${pid}/cmdline")" != *"VLLM::EngineCore"* ]]; then
        echo "FATAL: GPU${gpu} rollout EngineCore PID ${pid:-missing} is not the expected live process." >&2
        exit 1
    fi
    observed_pids="$(
        nvidia-smi -i "${gpu}" --query-compute-apps=pid \
            --format=csv,noheader,nounits \
            | tr -d '[:space:]'
    )"
    if [ "${observed_pids}" != "${pid}" ]; then
        echo "FATAL: GPU${gpu} is not exclusively owned by rollout EngineCore PID ${pid}; refusing overlap." >&2
        exit 1
    fi
done

echo "[$(date '+%F %T %Z')] launching ${EXPERIMENT_NAME} on CUDA_VISIBLE_DEVICES=${LANE_GPUS}"
set +e
CUDA_VISIBLE_DEVICES="${LANE_GPUS}" \
RAY_TMPDIR="${LANE_RAY_TMPDIR}" \
python launcher.py --conf "${CONFIG_PATH}" >"${TRAIN_LOG}" 2>&1
run_status=$?
set -e
echo "[$(date '+%F %T %Z')] ${EXPERIMENT_NAME} exited rc=${run_status}; log=${TRAIN_LOG}"
exit "${run_status}"
