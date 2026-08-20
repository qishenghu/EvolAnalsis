#!/bin/bash
# =============================================================================
# Checkpoint sweep driver — greedy vs sampled decoding on the frozen 128-task
# ALFWorld held-out val set, one merged HF checkpoint at a time.
#
# Called by run_ckpt_sweep.pbs (which owns the PBS resources, the preflight and
# the trap-based cleanup).  Kept separate so it can be dry-run by hand.
#
# Per checkpoint:
#   restart AlfWorld stack -> start 4x TP1 vLLM on that checkpoint
#   -> greedy   (temperature 0,   top_p 1.0, 1 rollout/task  = 128 episodes)
#   -> sampled  (temperature 0.9, top_p 1.0, 4 rollouts/task = 512 episodes)
#   -> stop vLLM
#
# The AlfWorld restart per checkpoint is load-bearing: every env reset dlopens a
# fresh copy of libdownward.so, and the server dies with "failed to map segment
# from shared object" after a few thousand resets.  640 episodes per checkpoint
# keeps each server process well under that ceiling; a whole 7-checkpoint sweep
# in one process (~4500 resets) would not.
#
# Env knobs: STEPS, SWEEP_OUT, HF_ROOT, BASE_PORT, SHARDS, WORKERS_PER_SHARD,
#            MAX_NUM_SEQS, GPU_MEM_UTIL, PKD, MODE_RETRIES
# =============================================================================
set -u

REPO=/home/qisheng001/DUET_H200/EvolAnalsis
SCRATCH=/projects_vol/gp_wangwy/qisheng/duet_h200
STEPS="${STEPS:-30 40 50 60 70 20 10}"
SWEEP_OUT="${SWEEP_OUT:-$SCRATCH/ckpt_sweep}"
HF_ROOT="${HF_ROOT:-$SCRATCH/ckpt_hf/p0_catalyst_af_s0}"
BASE_PORT="${BASE_PORT:-8301}"
SHARDS="${SHARDS:-4}"
WORKERS_PER_SHARD="${WORKERS_PER_SHARD:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
PKD="${PKD:-1}"
MODE_RETRIES="${MODE_RETRIES:-3}"
# Per-attempt wall clocks. Measured budget on an L40S: ~3.5 min per episode at
# concurrency 3; on 4x H200 at concurrency 64 greedy (128 episodes) lands in
# ~10-15 min and sampled (512) in ~40-60 min. These caps are ~4x that.
TIMEOUT_GREEDY="${TIMEOUT_GREEDY:-90m}"
TIMEOUT_SAMPLED="${TIMEOUT_SAMPLED:-240m}"
ALFWORLD_TMPDIR="${ALFWORLD_TMPDIR:-/tmp/duet_alfworld_tmp_$(id -un)}"
mkdir -p "$ALFWORLD_TMPDIR"
COLLECT_CFG="$REPO/config/duet_paper_experiments_configs/iclr2027/collect_h200/alfworld_qwen35_4b_collect_h200.yaml"
TASK_FILE="$REPO/data/alfworld/val_task_ids_128_seed2025.txt"
ENV_URL="http://127.0.0.1:8081"
# per-sweep log dir: two methods can share step numbers (catalyst 50 vs grpo 50)
# and would otherwise overwrite each other's collector logs.
LOGD="${SWEEP_LOGD:-$SCRATCH/logs/ckpt_sweep}"
mkdir -p "$SWEEP_OUT" "$LOGD"
cd "$REPO" || exit 1

say() { echo "[sweep $(date '+%F %T')] $*"; }

start_env_stack() {
    say "restarting AlfWorld stack"
    env -u CONDA_ENV_DUET TMPDIR="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
        TMP="$ALFWORLD_TMPDIR" bash start_env_alfworld.sh stop >/dev/null 2>&1
    sleep 5
    env -u CONDA_ENV_DUET TMPDIR="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
        TMP="$ALFWORLD_TMPDIR" bash start_env_alfworld.sh \
        >> "$LOGD/env_stack.log" 2>&1 || return 1
    for i in $(seq 1 60); do
        curl -sf --max-time 10 "$ENV_URL/docs" >/dev/null && return 0
        sleep 5
    done
    return 1
}

stop_env_stack() {
    env -u CONDA_ENV_DUET bash start_env_alfworld.sh stop >/dev/null 2>&1
}

start_vllm() {
    local model_dir=$1 tag=$2
    say "starting 4x TP1 vLLM on $model_dir"
    MODEL_DIR="$model_dir" \
    GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    BASE_PORT="$BASE_PORT" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN=32768 \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="$PKD" \
    WAIT_ATTEMPTS=900 \
    EXPERIMENT_TAG="ckpt_sweep_$tag" \
    EXPERIMENT_CONFIG="$COLLECT_CFG" \
    bash start_rollout_servers.sh start >> "$LOGD/vllm_${tag}.log" 2>&1
}

stop_vllm() {
    BASE_PORT="$BASE_PORT" bash start_rollout_servers.sh stop >/dev/null 2>&1
    sleep 10
}

# served model name is derived by start_rollout_servers.sh the same way
served_name() { echo "$(basename "$(dirname "$1")")/$(basename "$1")"; }

# landed rollout records for a mode (one line per rollout across the shards).
# grep -c exits 1 on zero matches; command substitution still captures the "0",
# and the driver does not run under `set -e`, so the count stays usable.
count_records() {
    local step=$1 mode=$2 n
    # [0-9] not *: `shard*.jsonl` also matches the `.jsonl.attempts.jsonl`
    # telemetry sidecars (2 events per rollout), which inflated this count 3x
    # (384/128) and could declare a mode "complete" while rollouts were missing.
    n=$(cat "$SWEEP_OUT"/${step}_${mode}_shard[0-9].jsonl 2>/dev/null | grep -c '"rollout_id"')
    echo "${n:-0}"
}

run_mode() {
    local step=$1 mode=$2 temp=$3 rpt=$4 model_dir=$5 tmo=$6
    local expect=$((128 * rpt))
    local sname; sname=$(served_name "$model_dir")
    local attempt
    for attempt in $(seq 1 "$MODE_RETRIES"); do
        local have; have=$(count_records "$step" "$mode")
        if [ "$have" -ge "$expect" ]; then
            say "step $step $mode already complete ($have/$expect)"
            return 0
        fi
        say "step $step $mode attempt $attempt/$MODE_RETRIES (have $have/$expect)"
        local pids=() idx
        for idx in $(seq 0 $((SHARDS - 1))); do
            # A wedged env instance would otherwise stall the shard until
            # walltime; the retry loop restarts the stack and --resume picks up
            # exactly the rollout slots that never landed.
            timeout --signal=TERM --kill-after=120 "$tmo" \
            python scripts/collect_eval_rollouts.py \
                --shard-index "$idx" --shard-count "$SHARDS" \
                --config "$COLLECT_CFG" \
                --env-url "$ENV_URL" \
                --task-file "$TASK_FILE" \
                --expected-task-count 128 \
                --output "$SWEEP_OUT/${step}_${mode}_shard${idx}.jsonl" \
                --model "$sname" \
                --api-base "http://127.0.0.1:$((BASE_PORT + idx))/v1" \
                --api-key-env DUET_EVAL_API_KEY \
                --temperature "$temp" --top-p 1.0 \
                --rollouts-per-task "$rpt" --max-attempts-per-rollout 1 \
                --max-workers "$WORKERS_PER_SHARD" \
                --resume --skip-live-profile-check \
                > "$LOGD/collect_${step}_${mode}_s${idx}.log" 2>&1 &
            pids+=($!)
        done
        local rc=0 p
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        have=$(count_records "$step" "$mode")
        say "step $step $mode attempt $attempt finished rc=$rc records=$have/$expect"
        [ "$have" -ge "$expect" ] && return 0
        # Partial output almost always means the AlfWorld server hit the
        # libdownward dlopen ceiling; a fresh stack is the documented cure.
        say "incomplete — restarting AlfWorld stack before retry"
        start_env_stack || say "WARNING env stack restart failed"
    done
    say "step $step $mode STILL INCOMPLETE after $MODE_RETRIES attempts"
    return 1
}

overall_rc=0
for step in $STEPS; do
    MODEL_DIR="$HF_ROOT/step_${step}"
    if [ ! -f "$MODEL_DIR/MERGE_DONE.json" ]; then
        say "SKIP step $step — no merged checkpoint at $MODEL_DIR"
        overall_rc=1
        continue
    fi
    say "================ checkpoint step $step ================"
    start_env_stack || { say "ABORT step $step: env stack"; overall_rc=1; continue; }
    if ! start_vllm "$MODEL_DIR" "step${step}"; then
        say "ABORT step $step: vLLM failed to come up"
        tail -30 "$LOGD/vllm_step${step}.log"
        stop_vllm
        overall_rc=1
        continue
    fi
    run_mode "$step" greedy  0    1 "$MODEL_DIR" "$TIMEOUT_GREEDY"  || overall_rc=1
    run_mode "$step" sampled 0.9  4 "$MODEL_DIR" "$TIMEOUT_SAMPLED" || overall_rc=1
    stop_vllm
    say "step $step done: greedy=$(count_records "$step" greedy)/128 sampled=$(count_records "$step" sampled)/512"
done
stop_env_stack
say "sweep finished rc=$overall_rc"
exit "$overall_rc"
