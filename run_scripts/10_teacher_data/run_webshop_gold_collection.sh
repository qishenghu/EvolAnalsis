#!/bin/bash
# Collect verified WebShop gold action trajectories aligned with training task ids.
#
# Prerequisites:
#   1. AgentGym WebShop server is running on 36003
#   2. EnvService WebShop proxy is running on 8083
#   3. conda activate agentevolver (or use conda run as below)
#
# Usage:
#   bash run_webshop_gold_collection.sh test
#   bash run_webshop_gold_collection.sh 800
#   bash run_webshop_gold_collection.sh full

set -e

MODE="${1:-800}"
ENV_URL="http://127.0.0.1:8083"
# Prefer instruction-matched rollouts, but keep verified successful replays even
# when the live instruction text differs slightly from the locally rebuilt goal.
COMMON_ARGS="--env_url ${ENV_URL} --split train --max_steps 20 --target_rollouts_per_task 5 --target_multisearch_rollouts 4 --instruction_match_policy strict"

case "${MODE}" in
  test)
    echo "=== TEST MODE: collect multi-search gold for 3 train tasks ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/collect_webshop_gold_trajectories.py \
      ${COMMON_ARGS} \
      --max_tasks 3 \
      --output analysis_outputs/webshop_gold_train_multisearch_test.jsonl
    ;;

  800)
    echo "=== 800-TASK SUBSET: multi-search gold, matches training config ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/collect_webshop_gold_trajectories.py \
      ${COMMON_ARGS} \
      --task_subset 800 \
      --task_seed 2026 \
      --resume \
      --output analysis_outputs/webshop_gold_train_multisearch_800.jsonl
    ;;

  full)
    echo "=== FULL TRAIN SPLIT: multi-search gold ==="
    conda run -n agentevolver --no-capture-output \
      python scripts/collect_webshop_gold_trajectories.py \
      ${COMMON_ARGS} \
      --resume \
      --output analysis_outputs/webshop_gold_train_multisearch_full.jsonl
    ;;

  *)
    echo "Usage: bash run_webshop_gold_collection.sh [test|800|full]"
    exit 1
    ;;
esac
