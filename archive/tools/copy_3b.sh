#!/bin/bash

set -e

REMOTE="remote"

paths=(
    # Onpolicy (GRPO)
    "checkpoints/agentevolver/alfworld_3b_grpo_react_tags/Trajectory/"
    "experiments/alfworld/alfworld_3b_grpo_react_tags/validation_log/"
    # LUFFY
    "checkpoints/agentevolver/alfworld_3b_luffy/Trajectory/"
    "experiments/alfworld/alfworld_3b_luffy/validation_log/"
    # DUET
    "checkpoints/agentevolver/alfworld_3b_duet_0329/Trajectory/"
    "experiments/alfworld/alfworld_3b_duet_0329/validation_log/"
)

for p in "${paths[@]}"; do
    echo "=========================================="
    echo "Downloading: $p"
    echo "=========================================="
    rclone copy "${REMOTE}:${p}" "$p" --progress --transfers 8
done

echo "All downloads complete."
