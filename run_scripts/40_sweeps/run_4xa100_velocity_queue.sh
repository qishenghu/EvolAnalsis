#!/bin/bash
# ==============================================================================
# 4xA100 Velocity Sprint — sequential queue with early-stop
#
# WS phase (run sequentially, EARLY-STOP if any hits success_rate >= 50%):
#   1. ws_swC_v_pk03_v00              ~3.5h   (peak=0.3 valley=0  K=10 vt=0.01)  [mandatory]
#   2. ws_swC_v_pk04_v00              ~3.5h   (peak=0.4 valley=0  K=10 vt=0.01)
#   3. ws_swC_v_pk05_v00_K5_vt005     ~3.5h   (peak=0.5 valley=0  K=5  vt=0.005)
#   4. ws_swC_v_pk04_v00_K5_vt005     ~3.5h   (peak=0.4 valley=0  K=5  vt=0.005)
#
# AF phase (always runs, regardless of WS outcome):
#   5. af_swC_v_pk05                  ~10h    (AF guardrail; must be ≥75%)
#
# Total: ≤24h with full WS sweep, ~14h if early stop on first run.
# Per-run env restart (memory leak prevention).
# Reports val@100 to analysis_reports/handoff/results_log.md after each run.
# Threshold: WS_HIT_THRESHOLD=0.50 (success_rate that triggers early-stop)
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
WS_HIT_THRESHOLD=0.50
mkdir -p logs analysis_reports/handoff

RESULTS_LOG="analysis_reports/handoff/results_log.md"
if [ ! -f "$RESULTS_LOG" ]; then
    cat > "$RESULTS_LOG" <<EOF
# Velocity Sprint — cross-server results log

| timestamp | server | name | success_rate | reward_mean | notes |
|---|---|---|---|---|---|
EOF
fi

wait_for_gpu_clean() {
    local _i
    local our_gpus="${CUDA_GPUS:-0,1,2,3}"
    for _i in {1..30}; do
        local used
        # Tolerate cohabitant tenants on multi-user box (threshold 30GB; our 3B+vLLM needs ~60GB).
        used=$(nvidia-smi --id="$our_gpus" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean (checked: $our_gpus)"; return 0; fi
        echo "  Waiting for GPU $our_gpus ($used busy)... ${_i}/30"; sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n training Ray actors..."
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    local env=$1
    echo "  Restarting $env env service..."
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_${env}.sh
    sleep 12
}

# Parses val@100 jsonl, appends row to results_log.md, AND prints
# raw success_rate (0.0-1.0) to stdout for shell parsing.
parse_and_log() {
    local name=$1
    local env=$2
    local val_log="experiments/${env}/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime, sys
val_log = "${val_log}"
name = "${name}"
env = "${env}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
sr_frac = 0.0
if not os.path.exists(val_log):
    line = f"| {ts} | 4xA100 | {name} | — | — | val@100 missing ({val_log}) |"
else:
    n = sr = 0
    rw = 0.0
    with open(val_log) as fh:
        for ln in fh:
            try:
                x = json.loads(ln)
            except Exception:
                continue
            n += 1
            s = x.get("score", x.get("reward", 0.0))
            rw += s
            if s >= 1.0:
                sr += 1
    if n == 0:
        line = f"| {ts} | 4xA100 | {name} | — | — | val@100 empty |"
    else:
        sr_frac = sr / n
        line = f"| {ts} | 4xA100 | {name} | {sr_frac*100:.1f}% (n={n}) | {rw/n:.4f} | env={env} |"
print(line, file=sys.stderr)
with open(results_log, "a") as fh:
    fh.write(line + "\n")
print(f"{sr_frac:.4f}")
PYEOF
}

run_one() {
    local config=$1
    local name=$2
    local env=$3
    local idx=$4
    local total=$5
    local ray_tmp="${RAY_TMPDIR}/${name}"

    echo ""
    echo "============================================"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    echo "============================================"
    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env "$env"
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    if [ ! -f "$config" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] MISSING config: $config — skipping"
        echo "0.0"
        return
    fi

    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name (env=$env)"
    (
        CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
            python launcher.py --conf "$config" \
            > "logs/${name}.log" 2>&1
    )
    local rc=$?
    if [ "$rc" = "0" ]; then
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] DONE: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] [$idx/$total] FAILED (rc=$rc): $name — continuing"
    fi
    parse_and_log "$name" "$env"
    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " 4xA100 Velocity Sprint (early-stop @ ${WS_HIT_THRESHOLD})"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

WS_QUEUE=(
    "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00.yaml|ws_swC_v_pk03_v00"
    "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk04_v00.yaml|ws_swC_v_pk04_v00"
    "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05_v00_K5_vt005.yaml|ws_swC_v_pk05_v00_K5_vt005"
    "config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk04_v00_K5_vt005.yaml|ws_swC_v_pk04_v00_K5_vt005"
)
WS_TOTAL=${#WS_QUEUE[@]}
GRAND_TOTAL=$((WS_TOTAL + 1))   # +1 for AF

# ----- WS phase with early-stop -----
ws_hit=0
idx=0
for entry in "${WS_QUEUE[@]}"; do
    idx=$((idx+1))
    IFS='|' read -r cfg name <<< "$entry"
    sr=$(run_one "$cfg" "$name" "webshop" "$idx" "$GRAND_TOTAL")
    echo ">>> ${name} success_rate=${sr} (threshold=${WS_HIT_THRESHOLD})"
    hit=$(awk -v sr="$sr" -v thr="$WS_HIT_THRESHOLD" 'BEGIN{print (sr+0 >= thr+0) ? 1 : 0}')
    if [ "$hit" = "1" ]; then
        ws_hit=1
        echo ""
        echo "============================================"
        echo " ⭐ WS HIT: ${name} reached ${sr} (≥${WS_HIT_THRESHOLD})"
        echo " Skipping remaining WS variants, proceeding to AF guardrail."
        echo "============================================"
        echo "| $(date '+%Y-%m-%d %H:%M') | 4xA100 | EARLY-STOP | — | — | hit ${WS_HIT_THRESHOLD} on ${name}, skipping remaining WS |" >> "$RESULTS_LOG"
        break
    fi
done

if [ "$ws_hit" = "0" ]; then
    echo ""
    echo "(All ${WS_TOTAL} WS variants exhausted; none reached ${WS_HIT_THRESHOLD}. Proceeding to AF.)"
    echo "| $(date '+%Y-%m-%d %H:%M') | 4xA100 | WS-EXHAUSTED | — | — | none reached ${WS_HIT_THRESHOLD} |" >> "$RESULTS_LOG"
fi

# ----- AF phase (always runs) -----
af_sr=$(run_one \
    "config/duet_paper_experiments_configs/alfworld/sweep_phase_c/af_swC_v_pk05.yaml" \
    "af_swC_v_pk05" "alfworld" "$GRAND_TOTAL" "$GRAND_TOTAL")
echo ">>> af_swC_v_pk05 success_rate=${af_sr}"

bash start_env_webshop.sh stop 2>/dev/null || true
bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " 4xA100 Velocity Sprint complete"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "Results: $RESULTS_LOG"
