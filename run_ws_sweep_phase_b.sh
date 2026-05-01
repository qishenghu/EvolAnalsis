#!/bin/bash
# ==============================================================================
# WS v39b Sweep — Phase B: 20 aggressive configs (~70h total)
#
# Beyond Phase A's BC schedule sweep, Phase B explores:
#   - Stronger BC peaks (0.8, 1.0)
#   - Token weighting (chord_use_token_weighting=true)
#   - DR3 policy_shaping_beta sweep
#   - KL coef variation
#   - State Channel boost
#   - Lower temperature (variance reduction)
#   - Combined best-bets
#
# Priority order: most likely to beat 53% (DUET v1) first.
# Each run does fresh env restart per the per-run pattern.
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

echo "[$(date '+%m-%d %H:%M')] WS sweep Phase B starting"

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

    # Skip if already complete (allow resumption)
    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (already done): $name"
        return 0
    fi

    # Per-run env restart (defense vs WebShop env memory leak)
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
        # Quick check if this is a winner (≥ 53%)
        local valf="experiments/webshop/${name}/validation_log/100.jsonl"
        if [ -f "$valf" ]; then
            local sr=$(python -c "
import json
n=0; sr=0
with open('$valf') as fh:
  for line in fh:
    d=json.loads(line); n+=1
    if d.get('score',d.get('reward',0))>=1.0: sr+=1
print(f'{sr/n*100:.1f}')
" 2>/dev/null)
            echo "[$(date '+%m-%d %H:%M')] $name val@100 success = ${sr}%"
            # Mark winner if ≥53%
            if [ -n "$sr" ] && python -c "import sys; sys.exit(0 if float('$sr')>=53.0 else 1)" 2>/dev/null; then
                echo "[$(date '+%m-%d %H:%M')] 🏆 WINNER: $name beat DUET v1 (53%) at ${sr}%"
                touch "logs/.phase_b_winner_${name}"
            fi
        fi
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED ($rc): $name"
    fi
}

# Priority order: most likely to win first.
priority=(
    # Tier 0+++ — ULTRA SAFE 3B combos (only peak + floor, no ema/valley risk).
    # Phase A 9/12 evidence: ema=.2 → 2.0%, valley=.10 (with pk=.5) → 19.5%.
    # Both 1.5B-recommended levers (ema=.2, valley=.10) FAIL on 3B.
    # Safe path: combine 3B-validated single winners (peak=.5 + floor=.4).
    "ws_swB_27_3Bbest_pk05_fl04"     # pk=.5 + floor=.4 (combine 3B top 2)  ⭐⭐⭐
    "ws_swB_31_pk05_fl03"            # pk=.5 + floor=.3 (push floor lower)  ⭐⭐
    "ws_swB_32_pk05_fl035"           # pk=.5 + floor=.35  ⭐⭐
    "ws_swB_33_pk045_fl04"           # pk=.45 + floor=.4 (gentler peak)  ⭐⭐
    "ws_swB_34_pk05_fl04_v03"        # pk=.5 + floor=.4 + valley=.03 (try lower)  ⭐
    "ws_swB_30_pk04_fl04"            # pk=.4 + floor=.4  ⭐
    # Tier 0+ — 3B + 1.5B valley combos (RISKY: valley=.10 alone hurt on 3B)
    "ws_swB_28_3Bbest_pk05_fl04_v10" # + valley=.10
    "ws_swB_29_3Bbest_pk05_fl04_v15" # + valley=.15
    # Tier 0 — 1.5B-inspired winners (1.5B server found: valley=0.10 + floor=0.6 + ema=0.2 → 36.0%)
    # Risk note: 3B Phase A showed ema=0.2 ALONE is catastrophic (swA_07 = 2.0%, swA_10 = 17%).
    # These work only if the 3-lever combo creates synergy not visible in single-lever data.
    "ws_swB_22_15Brecipe_pk05"       # 1.5B recipe + our peak=0.5 finding
    "ws_swB_21_15Bwinner_exact"      # exact 1.5B recipe on 3B
    "ws_swB_24_pk05_floor07"         # push floor further with strong BC
    "ws_swB_23_floor07"              # push floor only
    "ws_swB_25_pk02_15Brecipe"       # low peak (1.5B winner uses peak=0.3)
    "ws_swB_26_pk08_15Brecipe"       # very strong BC + 1.5B recipe
    # Tier 1: combined best-bets (Phase B original)
    "ws_swB_19_pk05_tw_ema02_v10"
    "ws_swB_18_pk07_ema02_v10"
    "ws_swB_03_pk08_ema02"
    "ws_swB_06_pk05_tw"
    "ws_swB_10_pk05_psb015"
    # Tier 2: extension of Phase A direction
    "ws_swB_04_pk10_ema02"
    "ws_swB_01_pk08"
    "ws_swB_02_pk10"
    "ws_swB_20_pk05_psb015_ema02"
    "ws_swB_08_pk05_tw_ema02"
    # Tier 3: unique mechanism tests
    "ws_swB_05_pk03_tw"
    "ws_swB_17_pk05_T04"
    "ws_swB_16_pk03_T04"
    "ws_swB_07_pk07_tw"
    "ws_swB_14_pk05_sc_b04"
    "ws_swB_15_pk05_sc_eta10"
    # Tier 4: secondary
    "ws_swB_12_pk05_kl_low"
    "ws_swB_13_pk05_kl_high"
    "ws_swB_09_pk05_psb005"
    "ws_swB_11_pk05_psb020"
)

# Re-run any Phase A configs that don't have val@100 (failed/killed)
for swA_yaml in config/duet_paper_experiments_configs/webshop/sweep/ws_swA_*.yaml; do
    name=$(basename "$swA_yaml" .yaml)
    if [ ! -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] Phase A retry: $name (no val@100)"
        run_one "$swA_yaml" "$name"
    fi
done

# Then run Phase B priority list
for name in "${priority[@]}"; do
    cfg="config/duet_paper_experiments_configs/webshop/sweep_phase_b/${name}.yaml"
    run_one "$cfg" "$name"
done

bash start_env_webshop.sh stop 2>&1 | tail -1

echo ""
echo "[$(date '+%m-%d %H:%M')] WS sweep Phase B complete!"

# Show summary
python scripts/analyze_ws_sweep.py --phase B 2>&1 | tail -30
