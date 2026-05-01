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
    # Tier S — focused 3B-data-driven candidates ONLY
    # All other Phase B configs (with ema=.2, valley=.10, peak>.5, 1.5B recipes,
    # token weighting, KL/SC variations) are SKIPPED based on Phase A signal:
    #   - ema=.2 always tanked (3/3 < 20%)
    #   - valley=.10 always tanked (2/2 < 20%)
    #   - peak ≥ .6 always tanked (2/2 ~20%)
    # Tier S = pure peak + floor sweep (3B-validated safe space).
    "ws_swB_27_3Bbest_pk05_fl04"     # ⭐⭐⭐ pk=.5 + floor=.4 (combine 3B top 2)
    "ws_swB_31_pk05_fl03"            # pk=.5 + floor=.3
    "ws_swB_32_pk05_fl035"           # pk=.5 + floor=.35
    "ws_swB_33_pk045_fl04"           # pk=.45 + floor=.4
    "ws_swB_30_pk04_fl04"            # pk=.4 + floor=.4
    "ws_swB_35_pk05_fl045"           # pk=.5 + floor=.45 (gap fill)
    "ws_swB_36_pk05_fl06"            # pk=.5 + floor=.6 (test 1.5B floor without other toxic)
    "ws_swB_37_pk055_fl04"           # pk=.55 + floor=.4
    "ws_swB_38_pk04_fl03"            # pk=.4 + floor=.3
    "ws_swB_34_pk05_fl04_v03"        # pk=.5 + floor=.4 + valley=.03 (lower valley, safe)
    "ws_swB_39_pk05_fl04_psb015"     # pk=.5 + floor=.4 + DR3 boost
)

# Phase A retries DISABLED (saves time; ws_swA_02/03 are informational, not SOTA candidates).
# If you want to retry them, uncomment the loop below.
# for swA_yaml in config/duet_paper_experiments_configs/webshop/sweep/ws_swA_*.yaml; do
#     name=$(basename "$swA_yaml" .yaml)
#     if [ ! -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
#         run_one "$swA_yaml" "$name"
#     fi
# done

# Phase B priority list (Tier S only — see priority array above for rationale)
for name in "${priority[@]}"; do
    cfg="config/duet_paper_experiments_configs/webshop/sweep_phase_b/${name}.yaml"
    run_one "$cfg" "$name"
done

bash start_env_webshop.sh stop 2>&1 | tail -1

echo ""
echo "[$(date '+%m-%d %H:%M')] WS sweep Phase B complete!"

# Show summary
python scripts/analyze_ws_sweep.py --phase B 2>&1 | tail -30
