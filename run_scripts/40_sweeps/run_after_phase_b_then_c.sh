#!/bin/bash
# ==============================================================================
# Phase C orchestrator: after Phase B done, take top-1 config and run 3 seeds
# to get robust mean±std for paper main table.
#
# Logic:
#   1. Wait for Phase B (run_ws_sweep_phase_b.sh) to finish
#   2. Find top-1 config across Phase A + Phase B by val@100 success_rate
#   3. Generate 3-seed yamls (seeds 42, 7, 1234) of that config
#   4. Run each (~3.5h each, ~11h total)
#   5. Final analysis report
# ==============================================================================

set +e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source env_config.sh

if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV_DUET}"

LOG="logs/after_phase_b_then_c.log"
GPUS="${CUDA_GPUS:-0,1,2,3}"

# Wait for Phase B to START (file appears) AND finish (file done marker).
# Use file-based markers to avoid pgrep self-match issues.
wait_for_phase_b() {
    # Phase 1: wait until Phase B starts (its log file is created and not stale)
    echo "[$(date '+%m-%d %H:%M')] waiting for Phase B to START (looking for logs/ws_sweep_phase_b.log)..." >> "$LOG"
    while [ ! -f logs/ws_sweep_phase_b.log ]; do
        sleep 60
    done
    # Confirm Phase B is fresh (modified within last 10 min)
    while [ -f logs/ws_sweep_phase_b.log ] && [ $(( $(date +%s) - $(stat -c %Y logs/ws_sweep_phase_b.log) )) -gt 600 ]; do
        sleep 60
    done
    echo "[$(date '+%m-%d %H:%M')] Phase B started ($(date)). Waiting for completion marker..." >> "$LOG"
    # Phase 2: wait for Phase B to write its "complete" marker
    # (Phase B orchestrator writes "WS sweep Phase B complete!" at end)
    while ! grep -q "WS sweep Phase B complete!" logs/ws_sweep_phase_b.log 2>/dev/null; do
        sleep 300
    done
    echo "[$(date '+%m-%d %H:%M')] Phase B completion marker found." >> "$LOG"
}

wait_for_phase_b
echo "[$(date '+%m-%d %H:%M')] Phase B done. Picking top winner..." >> "$LOG"

# Find top-1 by success_rate across Phase A + Phase B
top_config=$(python << 'PYEOF'
import json, glob, os
results = {}
for d in glob.glob("experiments/webshop/ws_sw[AB]_*/validation_log/100.jsonl"):
    name = os.path.basename(os.path.dirname(os.path.dirname(d)))
    n=0; sr=0
    with open(d) as f:
        for line in f:
            x=json.loads(line); n+=1
            if x.get("score", x.get("reward", 0)) >= 1.0: sr+=1
    if n > 0:
        results[name] = sr/n*100
if not results:
    print("NONE")
else:
    best = max(results.items(), key=lambda x: x[1])
    print(f"{best[0]}|{best[1]:.1f}")
PYEOF
)

if [ "$top_config" = "NONE" ]; then
    echo "[$(date '+%m-%d %H:%M')] No Phase A/B results found. Aborting Phase C." >> "$LOG"
    exit 1
fi

best_name=$(echo "$top_config" | cut -d'|' -f1)
best_score=$(echo "$top_config" | cut -d'|' -f2)
echo "[$(date '+%m-%d %H:%M')] Top winner: $best_name with success=$best_score%" >> "$LOG"

# Generate 3-seed yamls
mkdir -p config/duet_paper_experiments_configs/webshop/phase_c
src_yaml=$(find config/duet_paper_experiments_configs/webshop -name "${best_name}.yaml" | head -1)
if [ -z "$src_yaml" ]; then
    echo "[$(date '+%m-%d %H:%M')] ERROR: source yaml for $best_name not found" >> "$LOG"
    exit 1
fi

for seed in 42 7 1234; do
    out="config/duet_paper_experiments_configs/webshop/phase_c/ws_swC_${best_name}_seed${seed}.yaml"
    cp "$src_yaml" "$out"
    sed -i "s|experiment_name: ${best_name}|experiment_name: ws_swC_${best_name}_seed${seed}|" "$out"
    sed -i "s|seed: 2026|seed: ${seed}|" "$out"
    echo "  generated: $out" >> "$LOG"
done

# Run them
run_one() {
    local cfg=$1
    local name=$2

    if [ -f "experiments/webshop/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (done): $name" >> "$LOG"
        return 0
    fi

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] env restart..." >> "$LOG"
    bash start_env_alfworld.sh stop 2>&1 | tail -1 >> "$LOG"
    bash start_env_webshop.sh stop 2>&1 | tail -1 >> "$LOG"
    sleep 8
    bash start_env_webshop.sh >> "$LOG" 2>&1
    sleep 5

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$cfg" \
        > "logs/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] DONE: $name (rc=$?)" >> "$LOG"
}

for seed in 42 7 1234; do
    name="ws_swC_${best_name}_seed${seed}"
    cfg="config/duet_paper_experiments_configs/webshop/phase_c/${name}.yaml"
    run_one "$cfg" "$name"
done

bash start_env_webshop.sh stop 2>&1 | tail -1 >> "$LOG"

echo "[$(date '+%m-%d %H:%M')] Phase C complete!" >> "$LOG"
echo "" >> "$LOG"
python scripts/analyze_ws_sweep.py --phase all >> "$LOG" 2>&1
