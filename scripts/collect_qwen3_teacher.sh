#!/bin/bash
# ==============================================================================
# Qwen3 Teacher Trajectory Collection
#
# Collects teacher trajectories from Qwen3-30B-A3B-Thinking for DUET training.
# Run on a 4xA100-80G server while main experiments run on the 8xA100 server.
#
# Prerequisites:
#   1. Clone repo and install environment (bash setup_envs.sh)
#   2. Download teacher model:
#      huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
#          --local-dir /data/shared_models/Qwen3-30B-A3B-Thinking
#   3. Start the target environment (ALFWorld or WebShop)
#
# Usage:
#   bash scripts/collect_qwen3_teacher.sh alfworld   # Collect ALFWorld trajectories
#   bash scripts/collect_qwen3_teacher.sh webshop    # Collect WebShop trajectories
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
N_PER_TASK=2              # Rollouts per task (collect more, filter for success)
MAX_WORKERS=8             # Parallel workers
TEMPERATURE=0.6
SAVE_EVERY=100

mkdir -p "$OUTPUT_DIR" logs

ENV=${1:?Usage: bash scripts/collect_qwen3_teacher.sh [alfworld|webshop]}

# ===== Verify model exists =====
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: Teacher model not found at $TEACHER_MODEL"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 --local-dir $TEACHER_MODEL"
    exit 1
fi

case $ENV in
    alfworld)
        ENV_URL="http://127.0.0.1:8081"
        TASK_FILE="data/alfworld/task_ids.txt"
        OUTPUT_FILE="$OUTPUT_DIR/alfworld_qwen3_30b.jsonl"

        echo "=== Collecting ALFWorld Teacher Trajectories ==="
        echo "  Model: $TEACHER_MODEL"
        echo "  Tasks: $(wc -l < $TASK_FILE) tasks × $N_PER_TASK rollouts"
        echo "  Output: $OUTPUT_FILE"
        echo ""

        # Check if env is running
        if ! curl -s "$ENV_URL" >/dev/null 2>&1; then
            echo "Starting ALFWorld environment..."
            bash start_env_alfworld.sh
        fi

        python scripts/collect_teacher_trajectories.py \
            --env alfworld \
            --env_url "$ENV_URL" \
            --backend vllm \
            --model_path "$TEACHER_MODEL" \
            --task_file "$TASK_FILE" \
            --output "$OUTPUT_FILE" \
            --use_qwen3 \
            --n_per_task $N_PER_TASK \
            --filter_success \
            --max_workers $MAX_WORKERS \
            --temperature $TEMPERATURE \
            --save_every $SAVE_EVERY \
            2>&1 | tee "logs/collect_qwen3_alfworld.log"
        ;;

    webshop)
        ENV_URL="http://127.0.0.1:8083"
        TASK_FILE="data/webshop/task_ids.txt"
        OUTPUT_FILE="$OUTPUT_DIR/webshop_qwen3_30b.jsonl"

        echo "=== Collecting WebShop Teacher Trajectories ==="
        echo "  Model: $TEACHER_MODEL"
        echo "  Tasks: $(wc -l < $TASK_FILE) tasks × $N_PER_TASK rollouts"
        echo "  Output: $OUTPUT_FILE"
        echo ""

        # Check if env is running
        if ! curl -s "$ENV_URL" >/dev/null 2>&1; then
            echo "Starting WebShop environment..."
            bash start_env_webshop.sh
        fi

        python scripts/collect_teacher_trajectories.py \
            --env webshop \
            --env_url "$ENV_URL" \
            --backend vllm \
            --model_path "$TEACHER_MODEL" \
            --task_file "$TASK_FILE" \
            --output "$OUTPUT_FILE" \
            --use_qwen3 \
            --n_per_task $N_PER_TASK \
            --filter_success \
            --max_workers $MAX_WORKERS \
            --temperature $TEMPERATURE \
            --save_every $SAVE_EVERY \
            2>&1 | tee "logs/collect_qwen3_webshop.log"
        ;;

    sciworld)
        ENV_URL="http://127.0.0.1:8085"
        TASK_FILE="data/sciworld/task_ids_800_seed2026.txt"
        OUTPUT_FILE="$OUTPUT_DIR/sciworld_qwen3_30b.jsonl"

        echo "=== Collecting SciWorld Teacher Trajectories ==="
        echo "  Model: $TEACHER_MODEL"
        echo "  Tasks: $(wc -l < $TASK_FILE) tasks × $N_PER_TASK rollouts (seed=2026 subset)"
        echo "  Output: $OUTPUT_FILE"
        echo ""

        # Check if env is running
        if ! curl -s "$ENV_URL" >/dev/null 2>&1; then
            echo "ERROR: SciWorld environment not running on $ENV_URL"
            echo "Start it first (check env_service docs for SciWorld setup)"
            exit 1
        fi

        python scripts/collect_teacher_trajectories.py \
            --env sciworld \
            --env_url "$ENV_URL" \
            --backend vllm \
            --model_path "$TEACHER_MODEL" \
            --task_file "$TASK_FILE" \
            --output "$OUTPUT_FILE" \
            --use_qwen3 \
            --n_per_task $N_PER_TASK \
            --filter_success \
            --max_workers $MAX_WORKERS \
            --temperature $TEMPERATURE \
            --save_every $SAVE_EVERY \
            2>&1 | tee "logs/collect_qwen3_sciworld.log"
        ;;

    *)
        echo "Unknown environment: $ENV"
        echo "Usage: bash scripts/collect_qwen3_teacher.sh [alfworld|webshop|sciworld]"
        exit 1
        ;;
esac

echo ""
echo "=== Collection Complete ==="
echo "  Output: $OUTPUT_FILE"

# Post-processing: filter and convert to pkl
echo "Post-processing: filtering successful trajectories..."
python -c "
import json, pickle, sys
input_file = '$OUTPUT_FILE'
output_pkl = input_file.replace('.jsonl', '_filtered.pkl')
output_stats = input_file.replace('.jsonl', '_filtered_stats.json')

trajs = []
with open(input_file) as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            if t.get('success', False):
                trajs.append(t)

# Stats
task_ids = set(t['task_id'] for t in trajs)
print(f'Filtered: {len(trajs)} successful trajectories from {len(task_ids)} tasks')

with open(output_pkl, 'wb') as f:
    pickle.dump(trajs, f)
with open(output_stats, 'w') as f:
    json.dump({'total': len(trajs), 'unique_tasks': len(task_ids)}, f, indent=2)

print(f'Saved: {output_pkl}')
"
