#!/bin/bash
# Start SciWorld environment stack (AgentGym server + env_service wrapper)
# Usage: bash start_env_sciworld.sh
#   To stop: bash start_env_sciworld.sh stop
#
# Port convention (matches env_service/launch_script/sciworld.sh and the
# sciworld_* experiment configs): AgentGym :26004, env_service :8084.
#
# PREREQ: the AgentGym SciWorld backend needs its own conda env (python 3.8
# + Java 1.8+), analogous to agentenv-webshop:
#   conda create --name agentenv-sciworld python=3.8
#   conda activate agentenv-sciworld && pip install -e AgentGym/agentenv-sciworld
# Override CONDA_ENV_SCIWORLD / SCIWORLD_BIN below if it lives elsewhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"

AGENTGYM_PORT=26004   # below the ephemeral range (32768+): Ray workers grab those
ENVSERVICE_PORT=8084
CONDA_ENV_SCIWORLD="${CONDA_ENV_SCIWORLD:-agentenv-sciworld}"
SCIWORLD_BIN="${SCIWORLD_BIN:-${CONDA_PATH}/envs/${CONDA_ENV_SCIWORLD}/bin/sciworld}"
PYTHON_DUET="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"
export RAY_TMPDIR="${RAY_TMPDIR}/envsvc"
mkdir -p "$RAY_TMPDIR"

LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

# Preserve the outgoing environment logs instead of truncating them. These files
# used to be overwritten by every queue launch, so a finished run's environment
# could never be audited afterwards — which is exactly the evidence we wanted when
# a result looked wrong. Rotate by timestamp; the queue log gives the run/time map.
ENV_ARCHIVE="$LOGDIR/env_archive"
mkdir -p "$ENV_ARCHIVE"
for _old in "$LOGDIR/sciworld_agentgym.log" "$LOGDIR/sciworld_envservice.log"; do
    if [ -s "$_old" ]; then
        mv "$_old" "$ENV_ARCHIVE/$(basename "$_old" .log)_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
done

kill_port() {
    local port=$1
    # -sTCP:LISTEN 只杀监听者:lsof 会把连着该端口的客户端一并列出,
    # 2026-08-08 曾因此误杀正在采集的驱动进程(rc=143)。
    local pids=$(lsof -ti:$port -sTCP:LISTEN 2>/dev/null)
    local pid args
    for pid in $pids; do
        args=$(ps -p "$pid" -o args= 2>/dev/null)
        # The ephemeral port range on this host is 32768-60999, which OVERLAPS the
        # service ports below. vLLM/Ray actors bind RANDOM ports in that range, so a
        # blind kill-by-port can take down a running training job. Never kill a
        # process that looks like training infrastructure.
        case "$args" in
            *ray::*|*vllm*|*EngineCore*|*main_ppo*|*launcher.py*|*collect_openrouter*|*collect_student*)
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
    echo "Stopping SciWorld services..."
    kill_port $AGENTGYM_PORT
    kill_port $ENVSERVICE_PORT
    echo "Done."
    exit 0
fi

if [ ! -x "$SCIWORLD_BIN" ]; then
    echo "ERROR: sciworld launcher not found at $SCIWORLD_BIN"
    echo "The AgentGym SciWorld backend is not installed. Set it up with:"
    echo "  conda create --name agentenv-sciworld python=3.8"
    echo "  conda activate agentenv-sciworld"
    echo "  pip install -e $SCRIPT_DIR/AgentGym/agentenv-sciworld   # needs Java 1.8+"
    echo "or export SCIWORLD_BIN=/path/to/sciworld and re-run."
    exit 1
fi

echo "=== Starting SciWorld Environment Stack ==="

kill_port $AGENTGYM_PORT
kill_port $ENVSERVICE_PORT

echo "[1/2] Starting AgentGym SciWorld server on port $AGENTGYM_PORT..."
nohup $SCIWORLD_BIN --host 127.0.0.1 --port $AGENTGYM_PORT \
    > "$LOGDIR/sciworld_agentgym.log" 2>&1 &
disown

for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:$AGENTGYM_PORT/ | grep -q "ScienceWorld" 2>/dev/null; then
        echo "  AgentGym server ready (PID: $(lsof -ti:$AGENTGYM_PORT | tail -1))"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: AgentGym server failed to start. Check $LOGDIR/sciworld_agentgym.log"
        exit 1
    fi
    sleep 1
done

echo "[2/2] Starting env_service wrapper on port $ENVSERVICE_PORT..."
# Clean old Ray sessions to speed up startup
rm -rf "${RAY_TMPDIR}/session_"* 2>/dev/null || true

PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" \
SCIWORLD_SERVER_URL=http://127.0.0.1:$AGENTGYM_PORT \
nohup $PYTHON_DUET -m env_service.env_service --env sciworld --portal 127.0.0.1 --port $ENVSERVICE_PORT \
    > "$LOGDIR/sciworld_envservice.log" 2>&1 &
disown

for i in $(seq 1 30); do
    if lsof -ti:$ENVSERVICE_PORT >/dev/null 2>&1; then
        echo "  env_service ready (PID: $(lsof -ti:$ENVSERVICE_PORT | head -1))"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ERROR: env_service failed to start. Check $LOGDIR/sciworld_envservice.log"
        kill_port $AGENTGYM_PORT
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== SciWorld Environment Stack Running ==="
echo "  AgentGym server: http://127.0.0.1:$AGENTGYM_PORT"
echo "  env_service:     http://127.0.0.1:$ENVSERVICE_PORT"
echo "  Logs:            $LOGDIR/sciworld_*.log"
echo "  To stop: bash start_env_sciworld.sh stop"
