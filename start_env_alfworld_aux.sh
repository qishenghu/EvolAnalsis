#!/bin/bash
# Auxiliary ALFWorld stack on dedicated ports (18011/18091), so teacher sampling
# can run concurrently with the main rebuttal queue (which owns 36001/8081 and
# restarts it between runs). Usage: bash start_env_alfworld_aux.sh [stop]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=18011
ENVSERVICE_PORT=18091
PYTHON="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"
export ALFWORLD_SERVER_URL="http://127.0.0.1:${AGENTGYM_PORT}"
export RAY_TMPDIR="${RAY_TMPDIR}/envsvc_aux"
mkdir -p "$RAY_TMPDIR"

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)
    local pid args
    for pid in $pids; do
        args=$(ps -p "$pid" -o args= 2>/dev/null)
        # The ephemeral port range on this host is 32768-60999, which OVERLAPS the
        # service ports below. vLLM/Ray actors bind RANDOM ports in that range, so a
        # blind kill-by-port can take down a running training job. Never kill a
        # process that looks like training infrastructure.
        case "$args" in
            *ray::*|*vllm*|*EngineCore*|*main_ppo*|*launcher.py*)
                echo "  REFUSING to kill PID $pid on port $port (training process): ${args:0:80}"
                continue
                ;;
        esac
        echo "Killing process on port $port: $pid"
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
}

if [ "${1:-}" = "stop" ]; then
    echo "Stopping AUX ALFWorld services..."
    kill_port $AGENTGYM_PORT
    kill_port $ENVSERVICE_PORT
    echo "Done."
    exit 0
fi

echo "=== Starting AUX ALFWorld Environment Stack (ports $AGENTGYM_PORT/$ENVSERVICE_PORT) ==="
kill_port $AGENTGYM_PORT
kill_port $ENVSERVICE_PORT

echo "[1/2] Starting AgentGym ALFWorld server on port $AGENTGYM_PORT..."
nohup $ALFWORLD_BIN --host 127.0.0.1 --port $AGENTGYM_PORT \
    > "$LOGDIR/alfworld_agentgym_aux.log" 2>&1 &
disown

for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:$AGENTGYM_PORT/ | grep -q "AlfWorld" 2>/dev/null; then
        echo "  AUX AgentGym server ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ERROR: AUX AgentGym server failed. Check $LOGDIR/alfworld_agentgym_aux.log"
        exit 1
    fi
    sleep 1
done

echo "[2/2] Starting AUX env_service wrapper on port $ENVSERVICE_PORT..."
rm -rf "${RAY_TMPDIR}/session_"* 2>/dev/null || true
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
nohup $PYTHON -m env_service.env_service --env alfworld --portal 127.0.0.1 --port $ENVSERVICE_PORT \
    > "$LOGDIR/alfworld_envservice_aux.log" 2>&1 &
disown

for i in $(seq 1 60); do
    if lsof -ti:$ENVSERVICE_PORT >/dev/null 2>&1; then
        echo "  AUX env_service ready at http://127.0.0.1:$ENVSERVICE_PORT"
        exit 0
    fi
    sleep 2
done
echo "  ERROR: AUX env_service failed. Check $LOGDIR/alfworld_envservice_aux.log"
kill_port $AGENTGYM_PORT
exit 1
