#!/bin/bash

set -e

REMOTE="remote"

paths=(
    "checkpoints/agentevolver/webshop_7b_onpolicy/Trajectory/"
    "experiments/webshop/webshop_7b_onpolicy/validation_log/"
    "checkpoints/agentevolver/webshop_7b_luffy/Trajectory/"
    "experiments/webshop/webshop_7b_luffy/validation_log/"
    "checkpoints/agentevolver/webshop_7b_chord/Trajectory/"
    "experiments/webshop/webshop_7b_chord/validation_log/"
    "checkpoints/agentevolver/webshop_7b_duet/Trajectory/"
    "experiments/webshop/webshop_7b_duet/validation_log/"
)

for p in "${paths[@]}"; do
    echo "=========================================="
    echo "Downloading: $p"
    echo "=========================================="
    rclone copy "${REMOTE}:${p}" "$p" --progress --transfers 8
done

echo "All downloads complete."
