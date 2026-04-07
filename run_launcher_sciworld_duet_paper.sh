#!/bin/bash
# ==============================================================================
# DUET Paper Experiments: SciWorld 3B (GRPO / LUFFY / CHORD / DUET)
#
# Teacher data: data/teacher_trajectories/sciworld_gold_qwen72b_800_filtered.pkl
#   - 792 trajectories, 792 unique tasks, 100% success rate
#
# Prerequisites:
#   - AgentGym SciWorld server running at 127.0.0.1:36004
#   - SciWorld env service running at 127.0.0.1:8084
#     (launch: bash env_service/launch_script/sciworld.sh)
#
# Each experiment runs sequentially on 4 GPUs.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================"
echo " DUET Paper: SciWorld 3B Experiments"
echo "============================================"

echo ""
echo "[1/4] Running GRPO on-policy (sciworld_3b_onpolicy)..."
echo "============================================"
# python launcher.py \
#   --conf config/duet_paper_experiments_configs/sciworld/sciworld_3b_onpolicy.yaml

echo ""
echo "[4/4] Running DUET (sciworld_3b_duet)..."
echo "============================================"
python launcher.py \
  --conf config/duet_paper_experiments_configs/sciworld/sciworld_3b_duet_0406.yaml


echo ""
echo "[2/4] Running LUFFY (sciworld_3b_luffy)..."
echo "============================================"
#python launcher.py \
#  --conf config/duet_paper_experiments_configs/sciworld/sciworld_3b_luffy.yaml



echo ""
echo "[3/4] Running CHORD (sciworld_3b_chord)..."
echo "============================================"
#python launcher.py \
#  --conf config/duet_paper_experiments_configs/sciworld/sciworld_3b_chord.yaml

echo ""
echo "============================================"
echo " All SciWorld 3B experiments completed."
echo "============================================"
