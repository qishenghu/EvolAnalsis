#!/bin/bash
# ==============================================================================
# DUET Round 12 — ALFWorld-only pivot (3 experiments)
#
# WebShop env service on this server is fundamentally broken (Round 11 v5
# confirmed: even single env_service with deep clean still ReadTimeouts within
# 7 min — agentenv-webshop conda env or its indices are corrupted).
#
# Pivoting to ALFWorld which has worked fine (Round 9 ran v39b ALFWorld through
# step 99 successfully; only val@100 hit OOM, now fixed by gpu_mem 0.75→0.70).
#
# We accept Phase 1's v39b WebShop = 19.0% success / 0.605 reward as the
# WebShop number (bug-polluted by B2 cross-rank desync but available).
#
# 3 experiments (sequential, ~15h total):
#   1. v39b_postfix ALFWorld 1.5B (~5h) — Phase 1 winner, validates B1+B2+U1 fixes
#   2. v39_postfix  ALFWorld 1.5B (~5h) — original α=0.2
#   3. v39c_postfix ALFWorld 1.5B (~5h) — d_floor=0.4 sweep
#
# Resilience: same as Round 11 (no set -e, watchdog parallel).
# ==============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_DUET}"

GPUS="${CUDA_GPUS:-0,1,2,3}"
mkdir -p logs

wait_for_gpu_clean() {
    for i in {1..30}; do
        local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
        if [ "$used" = "0" ]; then echo "  GPU clean"; return 0; fi
        echo "  Waiting for GPU to clear ($used GPUs busy)... ${i}/30"; sleep 10
    done
    echo "  WARN: GPU not fully clean after 300s, proceeding anyway"
}

kill_ray_stragglers() {
    # NOTE: ALFWorld env_service uses different actors; only kill ray:: workers
    # not bound to env_service's session.
    local n=$(ps -ef | grep -E "ray::|raylet" | grep -v grep | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "  Killing $n stale Ray processes..."
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
        sleep 5
        ps -ef | grep -E "ray::|raylet" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        sleep 3
    fi
}

run_experiment() {
    local config=$1
    local name=$2
    local idx=$3
    local total=$4
    # FIX: use short path to avoid AF_UNIX 107-byte limit on socket file.
    local ray_tmp="${RAY_TMPDIR}"
    echo ""
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] PREP: $name"
    kill_ray_stragglers
    wait_for_gpu_clean
    mkdir -p "$ray_tmp"
    echo "[$(date '+%m-%d %H:%M')] [$idx/$total] RUN: $name"
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
    kill_ray_stragglers
    sleep 5
}

echo "============================================"
echo " DUET Round 12: ALFWorld-only pivot (3 experiments)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " GPU: $GPUS"
echo " Skip WebShop (env service broken on this host)"
echo "============================================"

if ! curl -s http://127.0.0.1:8081 >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] Starting ALFWorld environment..."
    bash start_env_alfworld.sh
fi
sleep 5

TOTAL=3
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39b_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39b_postfix" 1 $TOTAL
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39_postfix" 2 $TOTAL
run_experiment "config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39c_postfix.yaml" \
    "alfworld_qwen1.5b_duet_v39c_postfix" 3 $TOTAL

bash start_env_alfworld.sh stop 2>/dev/null || true

echo ""
echo "============================================"
echo " Round 12 complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
