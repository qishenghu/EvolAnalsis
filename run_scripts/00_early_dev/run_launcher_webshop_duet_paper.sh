#!/bin/bash
# ==============================================================================
# DUET Paper Experiments: WebShop 3B (LUFFY / DUET / CHORD)
#
# Teacher data: data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl
#   - 26,178 trajectories, 5,691 unique tasks, react_tags format, 100% success
#
# WebShop env runs on port 8083 (configured in yaml).
# Each experiment runs sequentially on 4 GPUs.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================"
echo " DUET Paper: WebShop 3B Experiments"
echo "============================================"

echo ""
echo "[1/3] Running LUFFY (webshop_3b_luffy)..."
echo "============================================"
python launcher.py \
  --conf config/duet_paper_experiments_configs/webshop/webshop_3b_luffy.yaml

echo ""
echo "[2/3] Running DUET (webshop_3b_duet)..."
echo "============================================"
python launcher.py \
  --conf config/duet_paper_experiments_configs/webshop/webshop_3b_duet.yaml

echo ""
echo "[3/3] Running CHORD (webshop_3b_chord)..."
echo "============================================"
python launcher.py \
  --conf config/duet_paper_experiments_configs/webshop/webshop_3b_chord.yaml

echo ""
echo "============================================"
echo " All WebShop 3B experiments completed."
echo "============================================"
