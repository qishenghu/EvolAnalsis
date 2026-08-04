#!/usr/bin/env bash
# Recover the canonical WebShop s200 baseline after the audited first launch
# failed inside ray.init solely because its AF_UNIX socket path was too long.
# No TaskRunner, W&B run, checkpoint, validation, or rollout was created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ALFWORLD_EXPERIMENT="alfworld_qwen35_4b_grpo_32k_baseline_gpu47_v5_resume_diagnostic_max"
WEBSHOP_EXPERIMENT="webshop_qwen35_4b_grpo_32k_baseline_gpu47_v1_s200"
ALFWORLD_VALIDATION="experiments/alfworld/${ALFWORLD_EXPERIMENT}/validation_log/100.jsonl"
ALFWORLD_CKPT_ROOT="checkpoints/agentevolver/${ALFWORLD_EXPERIMENT}"
ALFWORLD_CKPT="${ALFWORLD_CKPT_ROOT}/global_step_100/actor"
FAILED_LOG="logs/${WEBSHOP_EXPERIMENT}.log"
FAILED_LOG_ARCHIVE="logs/${WEBSHOP_EXPERIMENT}.failed_preray_afunix_20260802_161627.log"
FAILED_RECORD="launcher_record/${WEBSHOP_EXPERIMENT}"
FAILED_RECORD_ARCHIVE="launcher_record/${WEBSHOP_EXPERIMENT}.failed_preray_afunix_20260802_161627"
WEBSHOP_QUEUE="${REPO_ROOT}/run_iclr_gpu47_queue.sh"
HANDOFF_LOCK="logs/.gpu47_alfworld_to_webshop_handoff.lock"
LANE_LOCK="logs/.gpu47_training_lane.lock"
STOP_FILE="logs/STOP_GPU47_HANDOFF"
ROLLOUT_GPU_PIDS_CSV="${GPU47_ROLLOUT_GPU_PIDS:?set four vLLM EngineCore PIDs in GPU4-7 order}"
IFS=',' read -r -a ROLLOUT_GPU_PIDS <<< "${ROLLOUT_GPU_PIDS_CSV}"

mkdir -p logs
exec 8>"${HANDOFF_LOCK}"
if ! flock -n 8; then
    echo "FATAL: another ALFWorld-to-WebShop handoff/recovery is active." >&2
    exit 1
fi
exec 7>"${LANE_LOCK}"
if ! flock -n 7; then
    echo "FATAL: another compliant trainer owns GPU4-7." >&2
    exit 1
fi

unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE WANDB_CREDENTIALS_FILE
unset WANDB_DISABLED
export WANDB_MODE=online

if [ -e "${STOP_FILE}" ]; then
    echo "STOP file detected at ${STOP_FILE}; recovery will not launch WebShop."
    exit 0
fi

if [ "${#ROLLOUT_GPU_PIDS[@]}" -ne 4 ]; then
    echo "FATAL: recovery requires exactly four rollout EngineCore PIDs in GPU4-7 order." >&2
    exit 1
fi
for offset in 0 1 2 3; do
    gpu=$((offset + 4))
    pid="${ROLLOUT_GPU_PIDS[${offset}]}"
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || [ ! -e "/proc/${pid}/cmdline" ] || \
       [[ "$(tr '\0' ' ' < "/proc/${pid}/cmdline")" != *"VLLM::EngineCore"* ]]; then
        echo "FATAL: GPU${gpu} rollout EngineCore PID ${pid:-missing} is not live." >&2
        exit 1
    fi
    observed_pids="$(
        nvidia-smi -i "${gpu}" --query-compute-apps=pid \
            --format=csv,noheader,nounits \
            | tr -d '[:space:]'
    )"
    if [ "${observed_pids}" != "${pid}" ]; then
        echo "FATAL: GPU${gpu} is not exclusively owned by rollout EngineCore PID ${pid}." >&2
        exit 1
    fi
done

if [ ! -f "${ALFWORLD_VALIDATION}" ] || [ ! -d "${ALFWORLD_CKPT}" ]; then
    echo "FATAL: completed ALFWorld step-100 evidence is missing." >&2
    exit 1
fi
python -c \
    'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]; assert len(rows)==200; assert {r.get("step") for r in rows}=={100}; assert len({str(r["task_id"]) for r in rows})==200' \
    "${ALFWORLD_VALIDATION}"
if [ "$(tr -d '[:space:]' < "${ALFWORLD_CKPT_ROOT}/latest_checkpointed_iteration.txt")" != "100" ]; then
    echo "FATAL: ALFWorld latest checkpoint marker is not 100." >&2
    exit 1
fi
model_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'model_world_size_4_rank_*.pt' -size +1G | wc -l)"
optim_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'optim_world_size_4_rank_*.pt' -size +1G | wc -l)"
extra_shards="$(find "${ALFWORLD_CKPT}" -maxdepth 1 -type f -name 'extra_state_world_size_4_rank_*.pt' | wc -l)"
if [ "${model_shards}" -ne 4 ] || [ "${optim_shards}" -ne 4 ] || [ "${extra_shards}" -ne 4 ]; then
    echo "FATAL: incomplete ALFWorld step-100 checkpoint shards." >&2
    exit 1
fi

if [ ! -f "${FAILED_LOG}" ] || [ ! -d "${FAILED_RECORD}" ]; then
    echo "FATAL: canonical pre-Ray failure evidence is missing." >&2
    exit 1
fi
if ! grep -aFq "AF_UNIX path length cannot exceed 107 bytes" "${FAILED_LOG}" || \
   ! grep -aFq "ray.init(" "${FAILED_LOG}" || \
   grep -aFq "TaskRunner pid=" "${FAILED_LOG}"; then
    echo "FATAL: prior WebShop failure is not the audited pre-TaskRunner AF_UNIX case." >&2
    exit 1
fi
if [ -e "checkpoints/agentevolver/${WEBSHOP_EXPERIMENT}" ] || \
   [ -e "experiments/webshop/${WEBSHOP_EXPERIMENT}" ]; then
    echo "FATAL: WebShop training artifacts exist; refusing a fresh recovery." >&2
    exit 1
fi
if [ -e "${FAILED_LOG_ARCHIVE}" ] || [ -e "${FAILED_RECORD_ARCHIVE}" ]; then
    echo "FATAL: pre-Ray provenance archive target already exists." >&2
    exit 1
fi

# Repeat all service/manifest/W&B/fresh-output checks before changing the
# canonical failed-launch paths.  This does not touch GPUs or create a run.
bash "${WEBSHOP_QUEUE}" --preflight-only

# Preserve the complete failed attempt, then make the canonical run paths
# fresh.  Both moves are same-filesystem and recoverable.
mv "${FAILED_LOG}" "${FAILED_LOG_ARCHIVE}"
mv "${FAILED_RECORD}" "${FAILED_RECORD_ARCHIVE}"

echo "[$(date '+%F %T %Z')] audited pre-Ray failure archived; relaunching canonical WebShop s200"
export GPU47_HANDOFF_VERIFIED=1
export GPU47_LANE_LOCK_FD=7
export GPU47_ROLLOUT_GPU_PIDS="${ROLLOUT_GPU_PIDS_CSV}"
exec bash "${WEBSHOP_QUEUE}"
