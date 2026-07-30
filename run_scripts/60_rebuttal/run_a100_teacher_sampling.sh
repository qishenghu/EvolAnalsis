#!/bin/bash
# ============================================================================
# NeurIPS 2026 rebuttal: collect a MID-QUALITY teacher cache on ALFWorld
# for the teacher-quality ablation (reviewers UyKJ Q1, y9x6).
#
# Default: Qwen2.5-14B-Instruct on ONE spare GPU (6), using the AUX ALFWorld
# stack (ports 36011/8091) so it runs concurrently with the main queue.
#
# Usage:
#   bash start_env_alfworld_aux.sh                    # once, before sampling
#   nohup bash run_a100_teacher_sampling.sh > logs/teacher14b_sampling.log 2>&1 &
#   # 32B variant (needs 1 GPU, ~65GB weights fit on A100-80G w/ gpu_mem_util 0.95? use TP=2):
#   MODEL=/data/shared_models/Qwen2.5-32B-Instruct MODEL_NAME=qwen32b GPUS=6 TP=1 \
#       bash run_a100_teacher_sampling.sh
#
# Pipeline (same as the paper's 72B cache):
#   collect (raw) -> filter (success only) -> convert react->tags -> stats
# Final: data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}_filtered_react_tags.pkl
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/env_config.sh"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

GPUS="${GPUS:-6}"
TP="${TP:-1}"
MODEL="${MODEL:-/data/shared_models/Qwen2.5-14B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen14b}"
ENV_URL="${ENV_URL:-http://localhost:8091}"

OUTDIR=data/teacher_trajectories/${MODEL_NAME}
RAW=$OUTDIR/alfworld_${MODEL_NAME}_raw.jsonl
FILTERED_BASE=$OUTDIR/alfworld_${MODEL_NAME}_filtered
FINAL_PKL=${FILTERED_BASE}_react_tags.pkl
mkdir -p "$OUTDIR"

if [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: model not found at $MODEL — run run_download_teacher_models.sh first."
    exit 1
fi
if ! lsof -ti:${ENV_URL##*:} >/dev/null 2>&1; then
    echo "ERROR: aux env_service not running at $ENV_URL — run start_env_alfworld_aux.sh first."
    exit 1
fi

echo "[1/3] Collecting ${MODEL_NAME} rollouts: $(wc -l < "${TASK_FILE:-data/alfworld/task_ids_800_seed2026.txt}") tasks x ${N_PER_TASK:-5}, workers=${MAX_WORKERS:-32}, GPUs=$GPUS"
CUDA_VISIBLE_DEVICES=$GPUS python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --env_url "$ENV_URL" \
    --backend vllm \
    --model_path ${MODEL} \
    --task_file "${TASK_FILE:-data/alfworld/task_ids_800_seed2026.txt}" \
    --output "$RAW" \
    --tensor_parallel_size $TP \
    --gpu_memory_utilization 0.90 \
    --max_num_seqs 256 \
    --n_per_task "${N_PER_TASK:-5}" \
    --max_workers "${MAX_WORKERS:-32}" \
    --max_steps 30 \
    --temperature 0.6 \
    --max_tokens 4096 \
    --no_filter_success \
    --max_retries 3 \
    --save_every 100 \
    --resume

echo "[2/3] Filtering to successful trajectories (reward >= 1.0)..."
python scripts/filter_teacher_trajectories.py \
    --input "$RAW" \
    --output "$FILTERED_BASE" \
    --threshold 1.0

echo "[3/3] Converting react -> react_tags format..."
python scripts/convert_alfworld_react_to_tags.py \
    --input "${FILTERED_BASE}.jsonl" \
    --output "${FILTERED_BASE}_react_tags.jsonl" \
    --output-pkl "$FINAL_PKL"

echo ""
echo "=== ${MODEL_NAME} teacher cache stats (for the rebuttal) ==="
python - <<PY
import pickle
from collections import Counter
with open("$FINAL_PKL","rb") as f:
    d = pickle.load(f)
tasks = Counter(t.get("task_id") for t in d)
print(f"${MODEL_NAME} cache: {len(d)} successful trajectories, {len(tasks)} unique tasks, "
      f"mean {len(d)/max(1,len(tasks)):.2f} traj/task")
print("(72B reference: 19497 trajectories, 2348 tasks, 8.30 traj/task)")
PY
echo "DONE: $FINAL_PKL"
