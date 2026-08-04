#!/bin/bash
# ==============================================================================
# Launch/stop a lane of external vLLM rollout servers (env 'vllm2') for
# rollout.name == "external_vllm" training (Qwen3.5 students).
#
# Usage:
#   MODEL_DIR=/data/shared_models/Qwen3.5-4B-think GPUS=0,1,2,3 bash start_rollout_servers.sh
#   bash start_rollout_servers.sh run       # keep lane supervised; stop on exit
#   bash start_rollout_servers.sh stop      # kills by exact PIDs from the pidfile
#   bash start_rollout_servers.sh status
#
# Tunables (env vars):
#   MODEL_DIR     model to serve            (default /data/shared_models/Qwen3.5-4B-think)
#   GPUS          comma list, one TP1 server per GPU (default "0,1,2,3")
#   BASE_PORT     first port; server i gets BASE_PORT+i (default 8201)
#   GPU_MEM_UTIL  --gpu-memory-utilization  (default 0.25)
#   MAX_MODEL_LEN --max-model-len           (default 32768)
#   MAX_NUM_SEQS  concurrent sequences/server (default 2)
#   MAX_NUM_BATCHED_TOKENS prefill scheduling chunk (default 8192)
#   CHUNKED_PREFILL 1 to explicitly enable chunked prefill (default 1)
#   VLLM_ENV      conda env with the pinned vLLM runtime (default vllm2)
#   SLEEP_MODE    1 to pass --enable-sleep-mode (default 0)
#   LOGPROBS_MODE behavior-policy logprobs returned by vLLM
#                  (default processed_logprobs; must include temperature)
#   ENFORCE_EAGER  1 to disable torch.compile/CUDA graphs (default 0)
#   GDN_PREFILL_BACKEND optional Qwen3.5 GDN backend: triton or flashinfer
#   VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE 0 disables packed recurrent
#                  decode for Qwen3.5 GDN numerical safety (default 0)
#   VLLM_CACHE_BASE parent for a distinct compile cache per server/port
#                  (default ${RAY_TMPDIR}/duet_vllm_cache)
#   EXPERIMENT_TAG optional launcher/run identifier written to the manifest
#   EXPERIMENT_CONFIG optional YAML path hashed into the manifest
#
# Each server runs with VLLM_SERVER_DEV_MODE=1 (exposes /sleep /wake_up
# /collective_rpc /reset_prefix_cache) and the DUET weight-reload worker
# extension so the trainer can hot-swap weights after every PPO step.
# ==============================================================================
set -euo pipefail

# External rollout inference never logs to W&B.  Scrub credentials before
# sourcing helpers or spawning any manifest/vLLM subprocess so rollout servers
# cannot inherit training authentication material.
unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_config.sh"

MODEL_DIR="${MODEL_DIR:-/data/shared_models/Qwen3.5-4B-think}"
GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8201}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
VLLM_ENV="${VLLM_ENV:-vllm2}"
SLEEP_MODE="${SLEEP_MODE:-0}"
LOGPROBS_MODE="${LOGPROBS_MODE:-processed_logprobs}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-}"
VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE:-0}"
VLLM_CACHE_BASE="${VLLM_CACHE_BASE:-${RAY_TMPDIR}/duet_vllm_cache}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-unbound}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-}"

LOG_DIR="${SCRIPT_DIR}/logs"
PIDFILE="${LOG_DIR}/rollout_servers_${BASE_PORT}.pids"
VLLM_BIN="${CONDA_PATH}/envs/${VLLM_ENV}/bin/vllm"
# verl's ChatCompletionScheduler sends model = last two path components of model.path
SERVED_NAME="$(basename "$(dirname "${MODEL_DIR}")")/$(basename "${MODEL_DIR}")"
# worker extension lives in a standalone module (no agentevolver import in vllm2 env)
EXT_DIR="${SCRIPT_DIR}/external/verl_t5x_patches"

mkdir -p "${LOG_DIR}"

cmd="${1:-start}"

case "${cmd}" in
run)
    cleanup_supervised_lane() {
        bash "${SCRIPT_DIR}/start_rollout_servers.sh" stop || true
    }
    exit_on_signal() {
        exit 130
    }
    trap cleanup_supervised_lane EXIT
    trap exit_on_signal INT TERM

    bash "${SCRIPT_DIR}/start_rollout_servers.sh" start
    echo "Supervising rollout lane ${BASE_PORT}..$((BASE_PORT + 3)); Ctrl-C stops only PIDs in ${PIDFILE}."
    while true; do
        sleep 15
        status_output="$(bash "${SCRIPT_DIR}/start_rollout_servers.sh" status)"
        echo "${status_output}"
        if echo "${status_output}" | grep -Eq 'DEAD|/health=([^2]|2[^0]|20[^0])'; then
            echo "ERROR: supervised rollout lane is unhealthy." >&2
            exit 1
        fi
    done
    ;;
start)
    if [ -f "${PIDFILE}" ]; then
        echo "ERROR: ${PIDFILE} exists — servers may be running (use 'stop' first)." >&2
        exit 1
    fi
    if [ ! -x "${VLLM_BIN}" ]; then
        echo "ERROR: ${VLLM_BIN} not found (env '${VLLM_ENV}' missing?)" >&2
        exit 1
    fi
    extra_flags=()
    if [ "${SLEEP_MODE}" = "1" ]; then
        extra_flags+=(--enable-sleep-mode)
    fi
    if [ "${CHUNKED_PREFILL}" = "1" ]; then
        extra_flags+=(--enable-chunked-prefill)
    fi
    if [ "${ENFORCE_EAGER}" = "1" ]; then
        extra_flags+=(--enforce-eager)
    fi
    if [ -n "${GDN_PREFILL_BACKEND}" ]; then
        if [ "${GDN_PREFILL_BACKEND}" != "triton" ] && [ "${GDN_PREFILL_BACKEND}" != "flashinfer" ]; then
            echo "ERROR: GDN_PREFILL_BACKEND must be empty, triton, or flashinfer." >&2
            exit 1
        fi
        extra_flags+=(--gdn-prefill-backend "${GDN_PREFILL_BACKEND}")
    fi
    if [ "${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE}" != "0" ] && [ "${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE}" != "1" ]; then
        echo "ERROR: VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE must be 0 or 1." >&2
        exit 1
    fi
    i=0
    run_stamp="$(date +%Y%m%d_%H%M%S)"
    archive_dir="${LOG_DIR}/rollout_archive"
    mkdir -p "${archive_dir}"
    manifest="${LOG_DIR}/rollout_servers_${BASE_PORT}_${run_stamp}.manifest"
    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "model_dir=${MODEL_DIR}"
        echo "served_name=${SERVED_NAME}"
        echo "gpus=${GPUS}"
        echo "base_port=${BASE_PORT}"
        echo "max_model_len=${MAX_MODEL_LEN}"
        echo "max_num_seqs=${MAX_NUM_SEQS}"
        echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
        echo "chunked_prefill=${CHUNKED_PREFILL}"
        echo "gpu_memory_utilization=${GPU_MEM_UTIL}"
        echo "logprobs_mode=${LOGPROBS_MODE}"
        echo "enforce_eager=${ENFORCE_EAGER}"
        echo "gdn_prefill_backend=${GDN_PREFILL_BACKEND:-auto}"
        echo "vllm_enable_fla_packed_recurrent_decode=${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE}"
        echo "vllm_cache_base=${VLLM_CACHE_BASE}"
        echo "sleep_mode=${SLEEP_MODE}"
        echo "experiment_tag=${EXPERIMENT_TAG}"
        echo "experiment_config=${EXPERIMENT_CONFIG}"
        echo "vllm_bin=${VLLM_BIN}"
        echo "worker_extension=${EXT_DIR}/duet_vllm_worker_ext.py"
        echo "resolved_model_dir=$(readlink -f "${MODEL_DIR}")"
        "${CONDA_PATH}/envs/${VLLM_ENV}/bin/python" -c \
            'import importlib.metadata as m, platform; print("python=" + platform.python_version()); [print(f"{p}={m.version(p)}") for p in ("vllm", "torch", "transformers", "tokenizers", "safetensors", "numpy", "triton", "openai") if (lambda: True)()]' \
            2>/dev/null || true
        contract_paths=(
            "${MODEL_DIR}/chat_template.jinja"
            "${MODEL_DIR}/tokenizer_config.json"
            "${MODEL_DIR}/tokenizer.json"
            "${MODEL_DIR}/vocab.json"
            "${MODEL_DIR}/merges.txt"
            "${MODEL_DIR}/config.json"
            "${MODEL_DIR}/generation_config.json"
            "${MODEL_DIR}/preprocessor_config.json"
            "${MODEL_DIR}/video_preprocessor_config.json"
            "${MODEL_DIR}/model.safetensors.index.json"
        )
        for shard in "${MODEL_DIR}"/model.safetensors-*.safetensors; do
            if [ -f "${shard}" ]; then
                contract_paths+=("${shard}")
            fi
        done
        for contract_path in "${contract_paths[@]}"; do
            if [ -f "${contract_path}" ]; then
                contract_file="$(basename "${contract_path}")"
                echo "resolved_${contract_file}=$(readlink -f "${contract_path}")"
                sha256sum "${contract_path}"
            fi
        done
        sha256sum \
            "${SCRIPT_DIR}/start_rollout_servers.sh" \
            "${SCRIPT_DIR}/env_config.sh" \
            "${EXT_DIR}/duet_vllm_worker_ext.py"
        if [ -n "${EXPERIMENT_CONFIG}" ] && [ -f "${EXPERIMENT_CONFIG}" ]; then
            echo "resolved_experiment_config=$(readlink -f "${EXPERIMENT_CONFIG}")"
            sha256sum "${EXPERIMENT_CONFIG}"
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,uuid,name,driver_version \
                --format=csv,noheader || true
        fi
    } > "${manifest}"
    IFS=',' read -ra gpu_arr <<< "${GPUS}"
    mkdir -p "${VLLM_CACHE_BASE}"
    : > "${PIDFILE}"
    launched_ports=()
    for gpu in "${gpu_arr[@]}"; do
        port=$((BASE_PORT + i))
        server_cache_root="${VLLM_CACHE_BASE}/${port}"
        mkdir -p "${server_cache_root}"
        log="${LOG_DIR}/rollout_server_${port}.log"
        if [ -s "${log}" ]; then
            mv "${log}" "${archive_dir}/rollout_server_${port}_${run_stamp}.log"
        fi
        echo "server_${port}_argv=CUDA_VISIBLE_DEVICES=${gpu} VLLM_CACHE_ROOT=${server_cache_root} VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE} VLLM_SERVER_DEV_MODE=1 PYTHONPATH=${EXT_DIR} ${VLLM_BIN} serve ${MODEL_DIR} --host 0.0.0.0 --port ${port} --served-model-name ${SERVED_NAME} --return-tokens-as-token-ids --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEM_UTIL} --max-num-seqs ${MAX_NUM_SEQS} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --logprobs-mode ${LOGPROBS_MODE} --worker-extension-cls duet_vllm_worker_ext.RolloutWeightReloadExtension ${extra_flags[*]}" >> "${manifest}"
        echo "Starting rollout server: GPU ${gpu} port ${port} model ${SERVED_NAME} (log: ${log})"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        VLLM_CACHE_ROOT="${server_cache_root}" \
        VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE}" \
        VLLM_SERVER_DEV_MODE=1 \
        PYTHONPATH="${EXT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        nohup "${VLLM_BIN}" serve "${MODEL_DIR}" \
            --host 0.0.0.0 \
            --port "${port}" \
            --served-model-name "${SERVED_NAME}" \
            --return-tokens-as-token-ids \
            --max-model-len "${MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --logprobs-mode "${LOGPROBS_MODE}" \
            --worker-extension-cls duet_vllm_worker_ext.RolloutWeightReloadExtension \
            "${extra_flags[@]+"${extra_flags[@]}"}" \
            > "${log}" 2>&1 &
        echo "${port}:$!" >> "${PIDFILE}"
        launched_ports+=("${port}")
        i=$((i + 1))
    done
    echo "Launched ${i} servers on ports ${BASE_PORT}..$((BASE_PORT + i - 1)); pids in ${PIDFILE}"
    echo "Waiting for all servers to become ready..."
    all_ready=0
    for attempt in $(seq 1 120); do
        all_ready=1
        for port in "${launched_ports[@]}"; do
            if ! curl -sf --connect-timeout 1 --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
                all_ready=0
                break
            fi
        done
        if [ "${all_ready}" = "1" ]; then
            break
        fi
        sleep 2
    done
    if [ "${all_ready}" != "1" ]; then
        echo "ERROR: rollout servers did not all become healthy; inspect logs and run stop." >&2
        exit 1
    fi
    for port in "${launched_ports[@]}"; do
        if ! curl -sf --max-time 5 "http://127.0.0.1:${port}/v1/models" | grep -Fq "${SERVED_NAME}"; then
            echo "ERROR: ${port} is healthy but does not advertise ${SERVED_NAME}." >&2
            exit 1
        fi
    done
    {
        echo "verified_at=$(date --iso-8601=seconds)"
        echo "health_verified_ports=${launched_ports[*]}"
        echo "served_name_verified=${SERVED_NAME}"
        for port in "${launched_ports[@]}"; do
            server_log="${LOG_DIR}/rollout_server_${port}.log"
            kv_line="$(grep -F 'Available KV cache memory:' "${server_log}" | tail -1 || true)"
            concurrency_line="$(grep -F 'Maximum concurrency for' "${server_log}" | tail -1 || true)"
            echo "server_${port}_kv_cache=${kv_line}"
            echo "server_${port}_max_concurrency=${concurrency_line}"
        done
    } >> "${manifest}"
    echo "All rollout servers are healthy and advertise ${SERVED_NAME}."
    echo "Contract manifest: ${manifest}"
    ;;
stop)
    if [ ! -f "${PIDFILE}" ]; then
        echo "No pidfile at ${PIDFILE} — nothing to stop."
        exit 0
    fi
    # Project rule: NEVER broad pkill — kill only the exact PIDs we launched.
    while IFS=':' read -r port pid; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "Stopping server on port ${port} (pid ${pid})"
            kill "${pid}" 2>/dev/null || true
        else
            echo "Server on port ${port} (pid ${pid}) already gone"
        fi
    done < "${PIDFILE}"
    sleep 5
    while IFS=':' read -r port pid; do
        if kill -0 "${pid}" 2>/dev/null; then
            echo "Force-killing pid ${pid} (port ${port})"
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done < "${PIDFILE}"
    rm -f "${PIDFILE}"
    echo "Stopped."
    ;;
status)
    if [ ! -f "${PIDFILE}" ]; then
        echo "No pidfile at ${PIDFILE}."
        exit 0
    fi
    while IFS=':' read -r port pid; do
        if kill -0 "${pid}" 2>/dev/null; then
            health="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health" || true)"
            echo "port ${port} pid ${pid}: running, /health=${health}"
        else
            echo "port ${port} pid ${pid}: DEAD"
        fi
    done < "${PIDFILE}"
    ;;
*)
    echo "Usage: bash start_rollout_servers.sh [run|start|stop|status]" >&2
    exit 2
    ;;
esac
