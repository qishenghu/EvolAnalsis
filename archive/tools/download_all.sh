#!/bin/bash

set -e

REMOTE="remote"

paths=(
    # Onpolicy (GRPO)
    "checkpoints/agentevolver/webshop_3b_onpolicy/Trajectory/"
    "experiments/webshop/webshop_3b_onpolicy/validation_log/"
    # LUFFY
    "checkpoints/agentevolver/webshop_3b_luffy/Trajectory/"
    "experiments/webshop/webshop_3b_luffy/validation_log/"
    # CHORD
    "checkpoints/agentevolver/webshop_3b_chord_mu_0410/Trajectory/"
    "experiments/webshop/webshop_3b_chord_mu_0410/validation_log/"
    # DUET
    "checkpoints/agentevolver/webshop_3b_duet_0409_ema/Trajectory/"
    "experiments/webshop/webshop_3b_duet_0409_ema/validation_log/"
)

for p in "${paths[@]}"; do
    echo "=========================================="
    echo "Downloading: $p"
    echo "=========================================="
    rclone copy "${REMOTE}:${p}" "$p" --progress --transfers 8
done

echo "All downloads complete."
