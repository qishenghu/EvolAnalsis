#!/bin/bash
# ==============================================================================
# WS v39b Sweep — Phase A: 12 single-seed configs (~42h total)
#
# Goal: find a v39b-series config that beats DUET v1 (53%) on WebShop.
#
# Run order (priority — diagnostic first, then sweep):
#   01) v1cfg_no_bc          (sanity: should reproduce v1 ~53%)
#   02) v39b_default         (baseline: confirm v39b's true ballpark)
#   03-07) BC peak sweep:    {0.2, 0.3, 0.4, 0.5, 0.7} on v1cfg base
#   08-09) EMA sweep:        {0.2 slow, 0.8 fast} on v1cfg base
#   10-12) Strong-BC combos: peak=.5 + (slow ema, high valley, both)
#
# Each run: stop ALL env services → fresh start webshop env → run launcher
# Defends against WebShop env memory leak (~50GB after 26h of long-running env).
# ==============================================================================

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

echo "[$(date '+%m-%d %H:%M')] WS sweep Phase A starting"

# Cleanup any leftover training/ray processes
pgrep -f "launcher.py" 2>/dev/null | xargs -r kill -9 2>/dev/null
pgrep -f "ray::" 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 5

run_one() {
    local config=$1
    local name=$2

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (config missing): $name"
        return 0
    fi

    # ⭐ Per-run env restart (defense against WebShop env memory leak)
    echo "[$(date '+%m-%d %H:%M')] [pre-$name] stopping ALL env services..."
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh stop 2>&1 | tail -1
    sleep 8

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] starting fresh webshop env..."
    bash start_env_webshop.sh
    sleep 5

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED ($rc): $name"
    fi
}

# Run all 12 configs in numeric order (numbers encode priority).
for cfg in config/duet_paper_experiments_configs/webshop/sweep/ws_swA_*.yaml; do
    name=$(basename "$cfg" .yaml)
    run_one "$cfg" "$name"
done

bash start_env_webshop.sh stop 2>&1 | tail -1

echo ""
echo "[$(date '+%m-%d %H:%M')] WS sweep Phase A complete!"
echo "Run analyzer: python scripts/analyze_ws_sweep.py --phase A"
echo ""
echo "=== Summary of val@100 ==="
for cfg in config/duet_paper_experiments_configs/webshop/sweep/ws_swA_*.yaml; do
    name=$(basename "$cfg" .yaml)
    f="experiments/webshop/${name}/validation_log/100.jsonl"
    if [ -f "$f" ]; then
        echo "  ✓ $name"
    else
        echo "  ✗ MISSING: $name"
    fi
done
