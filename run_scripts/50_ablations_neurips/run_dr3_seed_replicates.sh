#!/bin/bash
# ============================================================================
# DR3 ablation seed replicates — 1.5B-AF, seed 2025 and 2027
# Waits for GPUs 0-3 to free (currently busy with 3B WS-SC), then runs the
# two seed variants sequentially. Appends val@100 results to ablation_results.md.
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="0,1,2,3"
mkdir -p logs NeurIPS_2026_Latex/data
RESULTS_LOG="NeurIPS_2026_Latex/data/ablation_results.md"
STOP_FILE="logs/ABLATION_DR3_SEED_STOP"
MASTER_LOG="logs/dr3_seed_orchestrator.log"

# ---------------------- helpers ----------------------

wait_for_gpu_clean() {
    local _i
    for _i in {1..240}; do
        local used
        used=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>30000' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPUs 0-3 clean"; return 0; fi
        echo "  Waiting for GPUs 0-3 ($used busy)... ${_i}/240"; sleep 60
    done
    echo "  WARN: timed out waiting; proceeding anyway"
}

kill_ray_stragglers() {
    local n
    n=$(ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::TaskRunner|ray::AsyncvLLMServer|ray::WorkerDict|ray::WorkerGroupRegisterCenter" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

restart_env() {
    bash start_env_alfworld.sh stop 2>/dev/null || true
    bash start_env_webshop.sh stop 2>/dev/null || true
    sleep 8
    bash start_env_alfworld.sh
    sleep 12
}

parse_and_log() {
    local name=$1; local seed=$2
    local val_log="experiments/alfworld/${name}/validation_log/100.jsonl"
    python - <<PYEOF
import json, os, datetime
val_log = "${val_log}"; name = "${name}"; seed = "${seed}"
results_log = "${RESULTS_LOG}"
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if not os.path.exists(val_log):
    line = f"| {ts} | 4xA100 | 1.5B-AF | -DR3 (seed {seed}) | — | — | — | — | val@100 MISSING |"
else:
    n = 0; sr_strict = 0; sr_lenient = 0; rw = 0.0
    with open(val_log) as fh:
        for ln in fh:
            try: x = json.loads(ln)
            except Exception: continue
            n += 1
            s = float(x.get("score", x.get("reward", 0.0)))
            rw += s
            if s >= 1.0: sr_strict += 1
            if s >= 0.9: sr_lenient += 1
    if n == 0:
        line = f"| {ts} | 4xA100 | 1.5B-AF | -DR3 (seed {seed}) | — | — | — | 0 | EMPTY |"
    else:
        line = (f"| {ts} | 4xA100 | 1.5B-AF | -DR3 (seed {seed}) | "
                f"{sr_strict/n*100:.1f}% | {sr_lenient/n*100:.1f}% | "
                f"{rw/n:.4f} | {n} | OK |")
print(line)
with open(results_log, "a") as fh: fh.write(line + "\n")
PYEOF
}

run_one() {
    local seed=$1
    local config="config/duet_paper_experiments_configs/ablations_neurips/alfworld/alfworld_qwen1.5b_duet_minus_dr3_seed${seed}.yaml"
    local name="alfworld_qwen1.5b_duet_minus_dr3_seed${seed}"
    local short_hash=$(echo "$name" | md5sum | head -c 8)
    local ray_tmp="${RAY_TMPDIR}/r${short_hash}"

    if [ -f "$STOP_FILE" ]; then
        echo "[$(date '+%m-%d %H:%M')] STOP_FILE detected — exiting before seed $seed"
        return 99
    fi

    echo ""
    echo "=================================================="
    echo "[$(date '+%m-%d %H:%M')] PREP: $name"
    echo "  config=$config"
    echo "  ray_tmp=$ray_tmp"
    echo "=================================================="

    kill_ray_stragglers
    wait_for_gpu_clean
    restart_env
    mkdir -p "$ray_tmp"
    rm -rf "$ray_tmp"/session_* 2>/dev/null || true

    if [ ! -f "$config" ]; then
        echo "  MISSING config: $config — skipping"
        return 1
    fi

    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    (CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" > "logs/${name}.log" 2>&1)
    local rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M')] DONE rc=0: $name"
    else
        echo "[$(date '+%m-%d %H:%M')] FAILED rc=$rc: $name"
    fi

    parse_and_log "$name" "$seed"
    kill_ray_stragglers
    sleep 5
}

# ---------------------- main ----------------------

echo "=================================================="
echo " DR3 SEED REPLICATE QUEUE  $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPUs: $GPUS"
echo " Seeds: 2025, 2027 (original was 2026)"
echo " Stop switch: touch $STOP_FILE"
echo "=================================================="

for seed in 2025 2027; do
    run_one "$seed"
    rc=$?
    if [ $rc -eq 99 ]; then
        echo "Queue stopped by STOP_FILE."
        break
    fi
done

bash start_env_alfworld.sh stop 2>/dev/null || true
bash start_env_webshop.sh stop 2>/dev/null || true

echo ""
echo "=================================================="
echo " DR3 SEED QUEUE COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
