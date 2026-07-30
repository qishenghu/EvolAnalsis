#!/bin/bash
# ==============================================================================
# Watchdog for Round 11 — kills stuck experiments to keep orchestrator moving
#
# Checks every 5 min:
#   1. Identifies the currently-running launcher.py process
#   2. Finds its log file
#   3. Checks whether "step:N" lines are being written
#   4. If 30 min with no new step → kill the launcher (orchestrator continues)
#
# Round 10 lesson: v39c ran 28h with 0 training steps due to env service deadlock.
# This watchdog would have killed it after 30 min, freeing 27.5h of GPU time.
#
# Run alongside the orchestrator:
#   nohup bash run_duet_round11_postfix_full.sh > logs/round11_orchestrator.log 2>&1 &
#   nohup bash run_duet_round11_watchdog.sh > logs/round11_watchdog.log 2>&1 &
# ==============================================================================

set -u
LOGS_DIR="/data/home/qisheng/EvolAnalsis/logs"
CHECK_INTERVAL=300       # check every 5 min
STUCK_THRESHOLD=1800     # 30 min with no progress = stuck
ORCHESTRATOR_NAME="run_duet_round11_postfix_full.sh"

last_step_count=""
last_progress_time=$(date +%s)
last_run_name=""

while true; do
    # Stop watchdog if orchestrator is done
    if ! pgrep -f "$ORCHESTRATOR_NAME" > /dev/null; then
        echo "[$(date '+%m-%d %H:%M')] Orchestrator no longer running — watchdog exiting."
        exit 0
    fi

    # Find current running launcher
    launcher_line=$(ps -ef | grep "launcher.py --conf" | grep "_postfix" | grep -v grep | head -1)
    if [ -z "$launcher_line" ]; then
        echo "[$(date '+%m-%d %H:%M')] No launcher running yet — sleeping."
        last_step_count=""
        last_progress_time=$(date +%s)
        sleep $CHECK_INTERVAL
        continue
    fi

    launcher_pid=$(echo "$launcher_line" | awk '{print $2}')
    config_path=$(echo "$launcher_line" | grep -oP 'config/[^ ]+')
    run_name=$(basename "$config_path" .yaml)
    log_file="${LOGS_DIR}/${run_name}.log"

    # Reset progress timer if we moved to a new experiment
    if [ "$run_name" != "$last_run_name" ]; then
        echo "[$(date '+%m-%d %H:%M')] New experiment detected: $run_name (PID $launcher_pid)"
        last_run_name="$run_name"
        last_step_count=""
        last_progress_time=$(date +%s)
    fi

    if [ ! -f "$log_file" ]; then
        echo "[$(date '+%m-%d %H:%M')] $run_name: log not yet created"
        sleep $CHECK_INTERVAL
        continue
    fi

    # Count step:N lines (training steps logged) — use wc -l to ensure single integer
    current_step_count=$(grep "step:[0-9]" "$log_file" 2>/dev/null | wc -l | tr -d ' \n')
    [ -z "$current_step_count" ] && current_step_count=0
    [ -z "$last_step_count" ] && last_step_count=0

    now=$(date +%s)
    if [ "$current_step_count" -ne "$last_step_count" ]; then
        # Progress detected
        delta=$((current_step_count - last_step_count))
        echo "[$(date '+%m-%d %H:%M')] $run_name: progressing ($current_step_count step lines, +$delta since last check)"
        last_step_count=$current_step_count
        last_progress_time=$now
    else
        # No progress — check how long
        elapsed=$((now - last_progress_time))
        echo "[$(date '+%m-%d %H:%M')] $run_name: NO progress for ${elapsed}s (threshold: ${STUCK_THRESHOLD}s)"
        if [ $elapsed -ge $STUCK_THRESHOLD ]; then
            echo "[$(date '+%m-%d %H:%M')] $run_name: STUCK — killing launcher PID $launcher_pid + ray + main_ppo"
            # Kill launcher
            kill $launcher_pid 2>/dev/null || true
            sleep 5
            # Kill Ray cluster + main_ppo for this experiment
            ps -ef | grep -E "ray::|raylet|main_ppo.*${run_name}" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null || true
            sleep 5
            ps -ef | grep -E "ray::|raylet|main_ppo.*${run_name}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
            echo "[$(date '+%m-%d %H:%M')] $run_name: kill complete. Orchestrator will continue to next experiment."
            # Reset for next experiment
            last_run_name=""
            last_step_count=""
            last_progress_time=$(date +%s)
            sleep 60  # give orchestrator time to detect failure and move on
        fi
    fi

    sleep $CHECK_INTERVAL
done
