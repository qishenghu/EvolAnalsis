#!/bin/bash
# ==============================================================================
# L20X velocity queue (2026-05-02): 2 WS runs + auto Phase C 3-seed of winner.
#
# Coordinated with 4×A100 server which is running:
#   T+0    ws_swC_v_pk03         (peak=0.3)
#   T+3.5  af_swC_v_pk05         (AF SOTA verification, ~10h)
#
# This script (L20X side):
#   T+0    ws_swC_v_pk05         (peak=0.5, ⭐ main candidate)
#   T+3.5  ws_swC_v_pk03_aggr    (window=5, vt=0.005, aggressive plateau)
#   T+7    auto-pick winner of {pk05, pk03, pk03_aggr} → 3 seeds (42, 7, 1234)
#
# Total wall-clock target: ~17.5h (2 main + 3 seeds × 3.5h)
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
mkdir -p logs
LOG="logs/l20x_velocity_queue.log"

echo "[$(date '+%m-%d %H:%M')] L20X velocity queue starting" | tee -a "$LOG"

run_one() {
    local config=$1
    local name=$2
    local env=${3:-webshop}

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (missing): $name" | tee -a "$LOG"
        return 1
    fi
    if [ -f "experiments/${env}/${name}/validation_log/100.jsonl" ]; then
        echo "[$(date '+%m-%d %H:%M')] SKIP (done): $name" | tee -a "$LOG"
        return 0
    fi

    echo "[$(date '+%m-%d %H:%M')] [pre-$name] env restart..." | tee -a "$LOG"
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh stop 2>&1 | tail -1
    sleep 8
    bash "start_env_${env}.sh"
    sleep 5

    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    echo "[$(date '+%m-%d %H:%M')] RUN: $name" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
    local rc=$?

    if [ -f "experiments/${env}/${name}/validation_log/100.jsonl" ]; then
        local sr=$(python -c "
import json
n=0; sr=0
with open('experiments/${env}/${name}/validation_log/100.jsonl') as fh:
  for line in fh:
    d=json.loads(line); n+=1
    if d.get('score',d.get('reward',0))>=1.0: sr+=1
print(f'{sr/n*100:.1f}')
" 2>/dev/null)
        echo "[$(date '+%m-%d %H:%M')] DONE: $name val@100 SR=${sr}% (rc=$rc)" | tee -a "$LOG"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED: $name (rc=$rc, no val@100)" | tee -a "$LOG"
    fi
    return $rc
}

# Phase 1 — main velocity runs (L20X share)
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05.yaml"      "ws_swC_v_pk05"      webshop
run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_aggr.yaml" "ws_swC_v_pk03_aggr" webshop

# Phase 2 — pick best WS winner across velocity runs (L20X + 4×A100 sides)
echo "[$(date '+%m-%d %H:%M')] Picking WS winner among velocity runs..." | tee -a "$LOG"
WINNER=$(python <<'PYEOF'
import json, glob, os
results = {}
# Look at all velocity runs (both L20X-produced and any 4×A100-produced if shared FS)
candidates = ["ws_swC_v_pk05", "ws_swC_v_pk03_aggr", "ws_swC_v_pk03"]
for name in candidates:
    f = f"experiments/webshop/{name}/validation_log/100.jsonl"
    if os.path.exists(f):
        n=0; sr=0
        with open(f) as fh:
            for line in fh:
                d=json.loads(line); n+=1
                if d.get("score", d.get("reward", 0)) >= 1.0: sr += 1
        if n > 0:
            results[name] = sr/n*100

if not results:
    print("NONE")
else:
    best = max(results.items(), key=lambda x: x[1])
    print(f"{best[0]}|{best[1]:.1f}")
PYEOF
)

if [ "$WINNER" = "NONE" ]; then
    echo "[$(date '+%m-%d %H:%M')] No velocity WS results found — Phase 2 aborted." | tee -a "$LOG"
    exit 1
fi

BEST_NAME=$(echo "$WINNER" | cut -d'|' -f1)
BEST_SR=$(echo "$WINNER" | cut -d'|' -f2)
echo "[$(date '+%m-%d %H:%M')] Winner: $BEST_NAME @ ${BEST_SR}%" | tee -a "$LOG"

# Decision rule
if python -c "import sys; sys.exit(0 if float('$BEST_SR') >= 45.0 else 1)" 2>/dev/null; then
    echo "[$(date '+%m-%d %H:%M')] ${BEST_SR}% ≥ 45% → proceeding with 3-seed Phase C." | tee -a "$LOG"
else
    echo "[$(date '+%m-%d %H:%M')] ${BEST_SR}% < 45% → velocity capped, skipping 3-seed (fallback to AF SOTA narrative)." | tee -a "$LOG"
    bash start_env_webshop.sh stop 2>&1 | tail -1
    exit 0
fi

# Phase 3 — generate + run 3 seeds
SRC="config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST_NAME}.yaml"
for seed in 42 7 1234; do
    OUT="config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST_NAME}_seed${seed}.yaml"
    cp "$SRC" "$OUT"
    sed -i "s|experiment_name: ${BEST_NAME}|experiment_name: ${BEST_NAME}_seed${seed}|" "$OUT"
    sed -i "s|seed: 2026|seed: ${seed}|" "$OUT"
    echo "[$(date '+%m-%d %H:%M')] generated: $OUT" | tee -a "$LOG"
done

for seed in 42 7 1234; do
    name="${BEST_NAME}_seed${seed}"
    cfg="config/duet_paper_experiments_configs/webshop/sweep_phase_c/${name}.yaml"
    run_one "$cfg" "$name" webshop
done

bash start_env_webshop.sh stop 2>&1 | tail -1

echo "[$(date '+%m-%d %H:%M')] L20X velocity queue COMPLETE." | tee -a "$LOG"

# Final 3-seed summary
python <<'PYEOF' | tee -a "$LOG"
import json, glob, os, statistics
import sys
prefix = os.environ.get("BEST_NAME", "")
if not prefix:
    # Try to detect from latest experiments
    cands = sorted(glob.glob("experiments/webshop/ws_swC_v_*_seed*/validation_log/100.jsonl"), key=os.path.getmtime, reverse=True)
    if cands:
        prefix = "_".join(os.path.basename(os.path.dirname(os.path.dirname(cands[0]))).split("_")[:-1])
print(f"=== 3-seed summary for {prefix} ===")
seeds = [42, 7, 1234]
srs = []
for seed in seeds:
    f = f"experiments/webshop/{prefix}_seed{seed}/validation_log/100.jsonl"
    if os.path.exists(f):
        n=0; sr=0
        with open(f) as fh:
            for line in fh:
                d=json.loads(line); n+=1
                if d.get("score", d.get("reward", 0)) >= 1.0: sr += 1
        if n > 0:
            v = sr/n*100
            srs.append(v)
            print(f"  seed {seed}: {v:.1f}%")
if len(srs) >= 2:
    print(f"  mean: {statistics.mean(srs):.2f}%")
    print(f"  std : {statistics.stdev(srs):.2f}%")
PYEOF
