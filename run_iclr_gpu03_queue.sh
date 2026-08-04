#!/bin/bash
# ============================================================================
# ICLR2027 lane A queue (GPUs 0-3): Qwen3.5-4B GRPO baselines.
#   1. alfworld_qwen35_4b_grpo
#   2. sciworld_qwen35_4b_grpo
#
# PRE-FLIGHT (run these FIRST, in this order — this script only verifies):
#   1. bash start_rollout_servers.sh lane_a     # external vLLM (env 'vllm2') on
#                                               # GPUs 0-3 -> 127.0.0.1:8201-8204
#   2. bash start_env_alfworld.sh               # AgentGym :36001 + env_service :8081
#   3. bash start_env_sciworld.sh               # AgentGym :36004 + env_service :8084
#      (needs the agentenv-sciworld conda env — see start_env_sciworld.sh header)
#
# Launch: nohup bash run_iclr_gpu03_queue.sh > logs/iclr_gpu03_queue.log 2>&1 &
#
# NEVER calls launcher.py --kill: lane B and the rollout servers share this
# machine, and --kill takes down every Ray process on the host.
# Rollout ports 8201-8204 / 8211-8214 sit BELOW the ephemeral range
# (32768-60999), so probing them never touches training internals.
# ============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/env_config.sh"
mkdir -p logs

# Training env for the Qwen3.5 stack (transformers 5.5.1 + vendored verl 0.4).
CONDA_ENV_TRAIN="${CONDA_ENV_DUET2:-duet2}"
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV_TRAIN"

LANE_GPUS="0,1,2,3"
LANE_RAY_TMPDIR="${RAY_TMPDIR}/iclr_lane_a"
ROLLOUT_PORTS="8201 8202 8203 8204"
CFG_ROOT="config/duet_paper_experiments_configs/iclr2027"

check_rollout_servers() {
    local port bad=0
    for port in $ROLLOUT_PORTS; do
        if ! curl -sf --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "FATAL: external vLLM rollout server 127.0.0.1:${port} does not answer /health."
            bad=1
        fi
    done
    if [ "$bad" -ne 0 ]; then
        echo "Start lane A rollout servers first (see pre-flight checklist in this script)."
        exit 1
    fi
    echo "  rollout servers OK: $ROLLOUT_PORTS"
}

check_env_service() {
    local port=$1 name=$2
    if ! lsof -ti:"$port" >/dev/null 2>&1; then
        echo "FATAL: ${name} env_service not listening on :${port}. Run: bash start_env_${name}.sh"
        exit 1
    fi
    echo "  ${name} env_service OK on :${port}"
}

run_one() {
    local name=$1 cfg=$2
    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    CUDA_VISIBLE_DEVICES="$LANE_GPUS" RAY_TMPDIR="$LANE_RAY_TMPDIR" \
        python launcher.py --conf "$cfg" > "logs/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] rc=$? : $name"
}

mkdir -p "$LANE_RAY_TMPDIR"
echo "[$(date '+%m-%d %H:%M')] lane A queue starting (GPUs $LANE_GPUS)"
check_rollout_servers

# --- 1/2: ALFWorld ---
check_env_service 8081 alfworld
run_one alfworld_qwen35_4b_grpo "${CFG_ROOT}/alfworld/alfworld_qwen35_4b_grpo.yaml"

# --- 2/2: SciWorld ---
check_rollout_servers
check_env_service 8084 sciworld
run_one sciworld_qwen35_4b_grpo "${CFG_ROOT}/sciworld/sciworld_qwen35_4b_grpo.yaml"

echo "[$(date '+%m-%d %H:%M')] LANE A QUEUE COMPLETE"
