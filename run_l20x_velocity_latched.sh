#!/bin/bash
# ==============================================================================
# L20X velocity-LATCHED queue (2026-05-02 hot-fix):
#
# After pk05 collapsed (22% → 1.5%) due to μ whip-sawing between peak/valley,
# we added a monotonic latch in het_actor.py: once rising_strength drops below
# threshold (default 0.3) AND velocity history is full, latch rs=0 permanently.
#
# All 5 configs use the SAME hot-fixed code. They differ only in chord_mu_*
# parameters. Total wall-clock: 5 × 3.5h ≈ 17.5h.
#
# Stale results (from broken whip-saw) are renamed *_broken so we don't confuse
# them with the new clean runs.
# ==============================================================================

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source env_config.sh

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
LOG="logs/l20x_velocity_latched.log"
mkdir -p logs

# Move stale (broken-velocity) results aside so the script doesn't skip them
for stale in ws_swC_v_pk05 ws_swC_v_pk03_aggr; do
    if [ -d "experiments/webshop/${stale}" ] && [ ! -d "experiments/webshop/${stale}_broken" ]; then
        mv "experiments/webshop/${stale}" "experiments/webshop/${stale}_broken"
        echo "[$(date '+%m-%d %H:%M')] archived stale: ${stale} → ${stale}_broken" | tee -a "$LOG"
    fi
done

echo "[$(date '+%m-%d %H:%M')] L20X velocity-LATCHED queue starting" | tee -a "$LOG"

run_one() {
    local config=$1
    local name=$2

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (missing): $name" | tee -a "$LOG"
        return 1
    fi
    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (done): $name" | tee -a "$LOG"
        return 0
    fi

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] env restart..." | tee -a "$LOG"
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh  stop 2>&1 | tail -1
    sleep 8
    bash start_env_webshop.sh
    sleep 5

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?

    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        local sr=$(python -c "
import json
n=0; sr=0
with open('experiments/webshop/${name}/validation_log/100.jsonl') as fh:
  for line in fh:
    d=json.loads(line); n+=1
    if d.get('score',d.get('reward',0))>=1.0: sr+=1
print(f'{sr/n*100:.1f}')
" 2>/dev/null)
        echo "[$(date '+%m-%d %H:%M')] DONE: $name val@100 SR=${sr}% (rc=$rc)" | tee -a "$LOG"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED: $name (rc=$rc)" | tee -a "$LOG"
    fi
}

# Queue: 5 runs in priority order. valley=0 variants first since they're our
# strongest hypothesis (BC fully off after latch = pure DUET v1 algorithm).
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05_v00.yaml"     "ws_swC_v_pk05_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk07_v00.yaml"     "ws_swC_v_pk07_v00"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00_K15.yaml" "ws_swC_v_pk03_v00_K15"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05.yaml"         "ws_swC_v_pk05"
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_aggr.yaml"    "ws_swC_v_pk03_aggr"

bash start_env_webshop.sh stop 2>&1 | tail -1

echo "" | tee -a "$LOG"
echo "[$(date '+%m-%d %H:%M')] L20X velocity-LATCHED queue COMPLETE." | tee -a "$LOG"

# Final leaderboard
echo "=== Final WS velocity-LATCHED leaderboard ===" | tee -a "$LOG"
python <<'PYEOF' | tee -a "$LOG"
import json, os
runs = [
    "ws_swC_v_pk05_v00", "ws_swC_v_pk07_v00", "ws_swC_v_pk03_v00_K15",
    "ws_swC_v_pk05", "ws_swC_v_pk03_aggr", "ws_swC_v_pk03_v00"  # last is from 4×A100 if shared FS
]
results = {}
for n in runs:
    f = f"experiments/webshop/{n}/validation_log/100.jsonl"
    if os.path.exists(f):
        cnt=0; sr=0
        with open(f) as fh:
            for line in fh:
                d=json.loads(line); cnt += 1
                if d.get("score", d.get("reward", 0)) >= 1.0: sr += 1
        if cnt > 0: results[n] = sr/cnt*100
for n,s in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"  {s:5.1f}%  {n}")
print(f"  Targets: LUFFY 49.5% | DUET v1 53.0%")
PYEOF
