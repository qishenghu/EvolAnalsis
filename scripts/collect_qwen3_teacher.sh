#!/bin/bash
# ==============================================================================
# Qwen3 Teacher Trajectory Collection (Direct Rollout, Targeted 800 Tasks)
#
# Collects teacher trajectories from Qwen3-30B-A3B-Thinking via direct rollout.
# All environments use the same approach: teacher interacts with env directly.
# Only the 800 tasks matching training config (seed=2026, max_train_tasks=800)
# are collected, for efficiency.
#
# Run on a 4xA100-80G server while main experiments run on the 8xA100 server.
#
# Prerequisites:
#   1. Clone repo and install environment (bash setup_envs.sh)
#   2. Download teacher model:
#      huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
#          --local-dir /data/shared_models/Qwen3-30B-A3B-Thinking
#   3. Start the target environment (ALFWorld, WebShop, or SciWorld)
#
# Usage:
#   bash scripts/collect_qwen3_teacher.sh alfworld
#   bash scripts/collect_qwen3_teacher.sh webshop
#   bash scripts/collect_qwen3_teacher.sh sciworld
#   bash scripts/collect_qwen3_teacher.sh all       # Run all sequentially
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Source environment config
source "$PROJECT_ROOT/env_config.sh"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

# ===== Configuration =====
TEACHER_MODEL="/data/shared_models/Qwen3-30B-A3B-Thinking"
OUTPUT_DIR="data/teacher_trajectories/qwen3_30b"
MAX_WORKERS=8             # Parallel workers
TEMPERATURE=0.6
SAVE_EVERY=50
# Per-env max attempts (stop_on_success mode: stop after first success)
ALFWORLD_MAX_ATTEMPTS=10
WEBSHOP_MAX_ATTEMPTS=10
SCIWORLD_MAX_ATTEMPTS=20   # SciWorld is harder, allow more attempts

mkdir -p "$OUTPUT_DIR" logs

# ===== Verify model exists =====
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: Teacher model not found at $TEACHER_MODEL"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 --local-dir $TEACHER_MODEL"
    exit 1
fi

collect_env() {
    local ENV=$1
    local ENV_URL=$2
    local TASK_FILE=$3
    local OUTPUT_FILE=$4
    local ENV_START_SCRIPT=$5

    local MAX_ATTEMPTS=$6

    echo "=== Collecting $ENV Teacher Trajectories ==="
    echo "  Model: $TEACHER_MODEL"
    echo "  Mode: stop_on_success (max $MAX_ATTEMPTS attempts per task)"
    echo "  Tasks: $(wc -l < $TASK_FILE)"
    echo "  Output: $OUTPUT_FILE"
    echo ""

    # Check if env is running, start if not
    if ! curl -s "$ENV_URL" >/dev/null 2>&1; then
        if [ -n "$ENV_START_SCRIPT" ]; then
            echo "Starting $ENV environment..."
            bash "$ENV_START_SCRIPT"
        else
            echo "ERROR: $ENV environment not running on $ENV_URL"
            exit 1
        fi
    fi

    python scripts/collect_teacher_trajectories.py \
        --env "$ENV" \
        --env_url "$ENV_URL" \
        --backend vllm \
        --model_path "$TEACHER_MODEL" \
        --task_file "$TASK_FILE" \
        --output "$OUTPUT_FILE" \
        --use_qwen3 \
        --n_per_task $MAX_ATTEMPTS \
        --stop_on_success \
        --filter_success \
        --max_workers $MAX_WORKERS \
        --temperature $TEMPERATURE \
        --save_every $SAVE_EVERY \
        2>&1 | tee "logs/collect_qwen3_${ENV}.log"

    # Post-process: filter and convert to pkl
    echo ""
    echo "Post-processing $ENV..."
    python -c "
import json, pickle
input_file = '$OUTPUT_FILE'
output_pkl = input_file.replace('.jsonl', '_filtered.pkl')
trajs = []
with open(input_file) as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            if t.get('success', False):
                trajs.append(t)
task_ids = set(t['task_id'] for t in trajs)
print(f'  Filtered: {len(trajs)} successful trajectories from {len(task_ids)} tasks')
with open(output_pkl, 'wb') as f:
    pickle.dump(trajs, f)
print(f'  Saved: {output_pkl}')
"
    echo "=== $ENV collection complete ==="
    echo ""
}

ENV=${1:?Usage: bash scripts/collect_qwen3_teacher.sh [alfworld|webshop|sciworld|all]}

case $ENV in
    alfworld)
        collect_env "alfworld" \
            "http://127.0.0.1:8081" \
            "data/alfworld/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/alfworld_qwen3_30b.jsonl" \
            "start_env_alfworld.sh" \
            "$ALFWORLD_MAX_ATTEMPTS"
        ;;

    webshop)
        collect_env "webshop" \
            "http://127.0.0.1:8083" \
            "data/webshop/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/webshop_qwen3_30b.jsonl" \
            "start_env_webshop.sh" \
            "$WEBSHOP_MAX_ATTEMPTS"
        ;;

    sciworld)
        collect_env "sciworld" \
            "http://127.0.0.1:8085" \
            "data/sciworld/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/sciworld_qwen3_30b.jsonl" \
            "" \
            "$SCIWORLD_MAX_ATTEMPTS"
        ;;

    all)
        echo "============================================"
        echo " Collecting ALL environments sequentially"
        echo " $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================"

        collect_env "alfworld" \
            "http://127.0.0.1:8081" \
            "data/alfworld/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/alfworld_qwen3_30b.jsonl" \
            "start_env_alfworld.sh" \
            "$ALFWORLD_MAX_ATTEMPTS"

        bash start_env_alfworld.sh stop 2>/dev/null || true

        collect_env "webshop" \
            "http://127.0.0.1:8083" \
            "data/webshop/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/webshop_qwen3_30b.jsonl" \
            "start_env_webshop.sh" \
            "$WEBSHOP_MAX_ATTEMPTS"

        bash start_env_webshop.sh stop 2>/dev/null || true

        collect_env "sciworld" \
            "http://127.0.0.1:8085" \
            "data/sciworld/task_ids_800_seed2026.txt" \
            "$OUTPUT_DIR/sciworld_qwen3_30b.jsonl" \
            "" \
            "$SCIWORLD_MAX_ATTEMPTS"

        echo "============================================"
        echo " ALL collections complete"
        echo " $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================"
        ;;

    *)
        echo "Unknown environment: $ENV"
        echo "Usage: bash scripts/collect_qwen3_teacher.sh [alfworld|webshop|sciworld|all]"
        exit 1
        ;;
esac
