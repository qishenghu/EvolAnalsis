#!/bin/bash
# Upload all Trajectory folders to Google Cloud via rclone (concurrent)
# Usage: bash upload_trajectories.sh [MAX_PARALLEL]
#   MAX_PARALLEL: max concurrent rclone processes (default: 4)

set -euo pipefail

MAX_PARALLEL=${1:-4}
CHECKPOINT_BASE="checkpoints/agentevolver"
REMOTE_BASE="remote:checkpoints/agentevolver"
LOG_DIR="/tmp/rclone_upload_logs"

mkdir -p "$LOG_DIR"

# Find all Trajectory directories
TRAJ_DIRS=($(find "$CHECKPOINT_BASE" -maxdepth 2 -type d -name "Trajectory" | sort))

echo "Found ${#TRAJ_DIRS[@]} Trajectory directories to upload"
echo "Max parallel uploads: $MAX_PARALLEL"
echo "---"

RUNNING=0
TOTAL=${#TRAJ_DIRS[@]}
DONE=0
FAILED=0
PIDS=()
DIRS=()

wait_for_slot() {
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" && true
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 0 ]; then
                    echo "[DONE] ${DIRS[$i]}"
                    DONE=$((DONE + 1))
                else
                    echo "[FAIL] ${DIRS[$i]} (exit code $EXIT_CODE, see log)"
                    FAILED=$((FAILED + 1))
                fi
                unset 'PIDS[i]'
                unset 'DIRS[i]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        sleep 1
    done
}

for traj_dir in "${TRAJ_DIRS[@]}"; do
    # e.g. checkpoints/agentevolver/webshop_3b_duet_0409_ema/Trajectory
    # parent: webshop_3b_duet_0409_ema
    parent=$(basename "$(dirname "$traj_dir")")
    remote_dest="${REMOTE_BASE}/${parent}"
    log_file="${LOG_DIR}/${parent}.log"

    wait_for_slot

    echo "[START] $parent -> $remote_dest"
    rclone copy "$traj_dir" "${remote_dest}/Trajectory" --progress > "$log_file" 2>&1 &
    pid=$!
    PIDS+=($pid)
    DIRS+=("$parent")
    RUNNING=$((RUNNING + 1))
done

# Wait for remaining
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && true
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[DONE] ${DIRS[$i]}"
        DONE=$((DONE + 1))
    else
        echo "[FAIL] ${DIRS[$i]} (exit code $EXIT_CODE)"
        FAILED=$((FAILED + 1))
    fi
done

echo "---"
echo "Finished: $DONE succeeded, $FAILED failed out of $TOTAL total"
echo "Logs in $LOG_DIR/"
