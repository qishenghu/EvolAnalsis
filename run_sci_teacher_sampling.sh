#!/bin/bash
# Synthesize SciWorld teacher trajectories with Thought reasoning.
#
# Prerequisites:
#   1. Gold trajectories collected: data/sciworld/gold_trajectories.jsonl
#   2. conda activate agentevolver
#
# Usage:
#   bash run_sci_teacher_sampling.sh test     # Local test (1 GPU, 3B, 3 tasks)
#   bash run_sci_teacher_sampling.sh 800      # 800-task subset (matches training config)
#   bash run_sci_teacher_sampling.sh full     # All ~4400 tasks
#   bash run_sci_teacher_sampling.sh          # Default = 800

set -e

MODE="${1:-800}"

# Common settings
INPUTS="data/sciworld/gold_trajectories.jsonl"
HISTORY_STEPS=5
MAX_TOKENS=128

case "${MODE}" in
  test)
    echo "=== TEST MODE: 1 GPU, 3B model, 3 tasks ==="
    CUDA_VISIBLE_DEVICES=0 \
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_sciworld_teacher_from_gold.py \
      --model_path Qwen/Qwen2.5-3B-Instruct \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.85 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/sciworld_gold_3b_synth_test.jsonl \
      --max_tasks 3 \
      --success_only \
      --collect_log_prob false \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS}
    ;;

  800)
    echo "=== 800-TASK SUBSET: 4 GPUs, 72B model (matches training config) ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_sciworld_teacher_from_gold.py \
      --model_path /data/code/exp/models/Qwen/Qwen2.5-72B-Instruct \
      --tensor_parallel_size 4 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/sciworld_gold_qwen72b_800_synth.jsonl \
      --task_subset 800 --task_seed 2026 \
      --success_only \
      --resume \
      --collect_log_prob false \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS} \
      --export_base data/teacher_trajectories/sciworld_gold_qwen72b_800_filtered \
      --export_threshold 1.0
    ;;

  full)
    echo "=== FULL RUN: 4 GPUs, 72B model, all tasks ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_sciworld_teacher_from_gold.py \
      --model_path /data/code/exp/models/Qwen/Qwen2.5-72B-Instruct \
      --tensor_parallel_size 4 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/sciworld_gold_qwen72b_synth.jsonl \
      --success_only \
      --resume \
      --collect_log_prob false \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS} \
      --export_base data/teacher_trajectories/sciworld_gold_qwen72b_filtered \
      --export_threshold 1.0
    ;;

  *)
    echo "Usage: bash run_sci_teacher_sampling.sh [test|800|full]"
    exit 1
    ;;
esac
