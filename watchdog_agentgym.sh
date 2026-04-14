#!/bin/bash
# Watchdog for AgentGym ALFWorld server
# Monitors RSS memory and auto-restarts when it exceeds threshold.
#
# Usage: nohup bash watchdog_agentgym.sh &
#   To stop: kill $(cat /tmp/watchdog_alfworld.pid)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=36001
ENVSERVICE_PORT=8081
RSS_THRESHOLD_MB=30000   # Auto-restart at 30GB (create fails around 40GB from C-level fragmentation)
CHECK_INTERVAL=60
PYTHON="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"

WATCHDOG_LOG="$DUET_PROJECT_ROOT/logs/watchdog_alfworld.log"
mkdir -p "$(dirname $WATCHDOG_LOG)"

echo $$ > /tmp/watchdog_alfworld.pid

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $1" | tee -a "$WATCHDOG_LOG"
}

restart_env_stack() {
    log "Restarting AgentGym + env_service..."

    # 1. Kill env_service first (it connects to AgentGym)
    local envsvc_pids=$(lsof -ti:$ENVSERVICE_PORT 2>/dev/null)
    if [ -n "$envsvc_pids" ]; then
        # Only kill the LISTENING process, not clients connected to it
        for p in $envsvc_pids; do
            if ps -p $p -o cmd --no-headers 2>/dev/null | grep -q "env_service"; then
                kill $p 2>/dev/null
                log "Killed env_service PID $p"
            fi
        done
        sleep 2
    fi

    # 2. Kill AgentGym
    local agentgym_pids=$(lsof -ti:$AGENTGYM_PORT 2>/dev/null)
    if [ -n "$agentgym_pids" ]; then
        echo "$agentgym_pids" | xargs kill 2>/dev/null
        sleep 2
    fi

    # 3. Clean env_service Ray sessions
    rm -rf "${RAY_TMPDIR}/envsvc/session_"* 2>/dev/null || true

    # 4. Restart AgentGym
    nohup $ALFWORLD_BIN --host 127.0.0.1 --port $AGENTGYM_PORT \
        >> "$DUET_PROJECT_ROOT/logs/alfworld_agentgym.log" 2>&1 &
    disown

    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:$AGENTGYM_PORT/ 2>/dev/null | grep -q "AlfWorld"; then
            log "AgentGym restarted (PID: $(lsof -ti:$AGENTGYM_PORT | tail -1))"
            break
        fi
        if [ $i -eq 30 ]; then
            log "ERROR: AgentGym failed to restart!"
            return 1
        fi
        sleep 1
    done

    # 5. Restart env_service
    export RAY_TMPDIR="${RAY_TMPDIR}/envsvc"
    mkdir -p "$RAY_TMPDIR"
    PYTHONPATH="$DUET_PROJECT_ROOT:$PYTHONPATH" \
    nohup $PYTHON -m env_service.env_service --env alfworld --portal 127.0.0.1 --port $ENVSERVICE_PORT \
        >> "$DUET_PROJECT_ROOT/logs/alfworld_envservice.log" 2>&1 &
    disown

    for i in $(seq 1 30); do
        if lsof -ti:$ENVSERVICE_PORT >/dev/null 2>&1; then
            log "env_service restarted (PID: $(lsof -ti:$ENVSERVICE_PORT | head -1))"
            return 0
        fi
        sleep 1
    done
    log "ERROR: env_service failed to restart!"
    return 1
}

log "Watchdog started. RSS threshold: ${RSS_THRESHOLD_MB}MB, check interval: ${CHECK_INTERVAL}s"

while true; do
    PID=$(lsof -ti:$AGENTGYM_PORT 2>/dev/null | tail -1)

    if [ -z "$PID" ]; then
        log "AgentGym not running! Restarting..."
        restart_env_stack
        sleep $CHECK_INTERVAL
        continue
    fi

    RSS_KB=$(ps -p $PID -o rss --no-headers 2>/dev/null | tr -d ' ')
    if [ -z "$RSS_KB" ]; then
        sleep $CHECK_INTERVAL
        continue
    fi

    RSS_MB=$((RSS_KB / 1024))
    if [ $RSS_MB -gt $RSS_THRESHOLD_MB ]; then
        log "RSS=${RSS_MB}MB exceeds threshold ${RSS_THRESHOLD_MB}MB. Restarting..."
        restart_env_stack
    fi

    sleep $CHECK_INTERVAL
done
