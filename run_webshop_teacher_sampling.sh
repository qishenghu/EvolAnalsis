#!/bin/bash
# Synthesize WebShop teacher trajectories with Thought reasoning from verified gold actions.
#
# Prerequisites:
#   1. Gold action trajectories already collected
#   2. conda activate agentevolver
#
# Usage:
#   bash run_webshop_teacher_sampling.sh test
#   bash run_webshop_teacher_sampling.sh 800
#   bash run_webshop_teacher_sampling.sh full

set -e

MODE="${1:-800}"
INPUTS="${INPUTS:-analysis_outputs/webshop_gold_train_multisearch_full_chunked.jsonl}"
HISTORY_STEPS=5
MAX_TOKENS=400
ACTION_FORMAT="${ACTION_FORMAT:-react_tags}"

case "${MODE}" in
  test)
    echo "=== TEST MODE: GPU1, 3B model, 3 tasks ==="
    CUDA_VISIBLE_DEVICES=1 \
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_webshop_teacher_from_gold.py \
      --model_path Qwen/Qwen2.5-3B-Instruct \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.85 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/qwen3b/webshop_qwen3b_synth_test.jsonl \
      --max_tasks 3 \
      --collect_log_prob false \
      --action_format ${ACTION_FORMAT} \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS}
    ;;

  800)
    echo "=== 800-TASK SUBSET: matches training config ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_webshop_teacher_from_gold.py \
      --model_path /data/code/exp/models/Qwen/Qwen2.5-72B-Instruct \
      --tensor_parallel_size 4 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/qwen72b/webshop_qwen72b_800_synth.jsonl \
      --task_subset 800 \
      --task_seed 2026 \
      --resume \
      --collect_log_prob false \
      --action_format ${ACTION_FORMAT} \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS} \
      --export_base data/teacher_trajectories/qwen72b/webshop_qwen72b_800_filtered \
      --export_threshold 1.0
    ;;

  full)
    echo "=== FULL RUN: all collected gold trajectories ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/synthesize_webshop_teacher_from_gold.py \
      --model_path /data/code/exp/models/Qwen/Qwen2.5-72B-Instruct \
      --tensor_parallel_size 4 \
      --inputs ${INPUTS} \
      --output data/teacher_trajectories/qwen72b/webshop_qwen72b_synth.jsonl \
      --resume \
      --collect_log_prob false \
      --action_format ${ACTION_FORMAT} \
      --history_steps ${HISTORY_STEPS} \
      --max_tokens ${MAX_TOKENS} \
      --export_base data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered \
      --export_threshold 1.0
    ;;

  *)
    echo "Usage: bash run_webshop_teacher_sampling.sh [test|800|full]"
    exit 1
    ;;
esac
