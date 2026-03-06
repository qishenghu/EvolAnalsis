#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="config/paper_alfworld_frc_lite.yaml"
TEACHER_DATA="/home/qisheng/agent/AgentEvolver/data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered.pkl"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate agentevolver

if [ ! -f "$TEACHER_DATA" ]; then
  echo "Teacher data not found: $TEACHER_DATA" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "============================================"
echo "Running AlfWorld FRC-lite"
echo "Config:  $CONFIG_PATH"
echo "Teacher: $TEACHER_DATA"
echo "CUDA:    $CUDA_VISIBLE_DEVICES"
echo "============================================"

python launcher.py --with-alfworld --conf "$CONFIG_PATH"
