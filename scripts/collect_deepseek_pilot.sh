#!/bin/bash
# DeepSeek-v4-flash teacher PILOT collection via OpenRouter (ICLR2027_PLAN §3.1).
# 50 ALFWorld tasks, stop-on-success — measures success rate, per-trajectory cost,
# and validates the reasoning->!<think> merge (openai_teacher_llm.py) end to end.
# No GPU needed; requires ALFWorld env service on :8081.
# Usage: bash scripts/collect_deepseek_pilot.sh [num_tasks]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
source env_config.sh

NUM_TASKS=${1:-50}
MODEL="${PILOT_MODEL:-deepseek/deepseek-v4-flash}"   # testing phase: ~5x cheaper than v4-pro (in $0.09/out $0.18 vs $0.435/$0.87 per M tok)
OUT=data/teacher_trajectories/deepseek_v4/alfworld_dsv4_pilot.jsonl
mkdir -p "$(dirname "$OUT")" logs

# key lives in test_openrouter.py (user-provided); read it without echoing
export OPENAI_API_KEY=$(grep -oE 'sk-or-v1-[a-f0-9]+' /data/home/qisheng/test_openrouter.py | head -1)
[ -n "$OPENAI_API_KEY" ] || { echo "FATAL: OpenRouter key not found"; exit 1; }
curl -sf -o /dev/null http://127.0.0.1:8081/docs || { echo "FATAL: ALFWorld env service (8081) not up"; exit 1; }

head -n "$NUM_TASKS" data/alfworld/task_ids_800_seed2026.txt > /tmp/pilot_task_ids.txt

/data/home/qisheng/miniconda3/envs/duet/bin/python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --env_url http://127.0.0.1:8081 \
    --backend openai \
    --model_path "$MODEL" \
    --api_base https://openrouter.ai/api/v1 \
    --task_file /tmp/pilot_task_ids.txt \
    --stop_on_success --max_retries 5 \
    --filter_success \
    --temperature 0.6 \
    --max_tokens 8192 \
    --max_workers 4 \
    --output "$OUT" \
    2>&1 | tee logs/deepseek_pilot_alfworld.log

echo "=== PILOT DONE ==="
/data/home/qisheng/miniconda3/envs/duet/bin/python - << 'EOF'
import json
recs = [json.loads(l) for l in open("data/teacher_trajectories/deepseek_v4/alfworld_dsv4_pilot.jsonl")]
succ = [r for r in recs if r.get("success")]
think = sum(1 for r in recs for m in r.get("messages", []) if m["role"] == "assistant" and "<think>" in m.get("content", ""))
print(f"trajectories: {len(recs)}, success: {len(succ)}, assistant msgs with <think>: {think}")
EOF
