#!/bin/bash
# Start DeepSearch environment stack (BM25 retrieval server + env_service wrapper)
# Usage: bash start_env_deepsearch.sh          To stop: bash start_env_deepsearch.sh stop
#
# Components:
#   [1] retrieval_server.py — BM25 over wiki-18, runs in the agentenv-webshop
#       conda env (pyserini + JDK11), port 25011 (outside the 32768+ ephemeral range)
#   [2] env_service wrapper  — duet env, --env deepsearch, port 8086

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_config.sh"
duet_activate || { echo "ERROR: cannot activate duet env"; exit 1; }

RETRIEVAL_PORT=25011
ENVSERVICE_PORT=8086
BM25_INDEX="${DEEPSEARCH_BM25_INDEX:-/projects_vol/gp_wangwy/qisheng/duet_h200/deepsearch/bm25_wiki18}"
WSPY="${CONDA_PATH}/envs/${CONDA_ENV_WEBSHOP:-agentenv-webshop}/bin/python"
PYTHON="${CONDA_PATH}/envs/${CONDA_ENV_DUET}/bin/python"
export RAY_TMPDIR="${RAY_TMPDIR}/envsvc_ds"
mkdir -p "$RAY_TMPDIR"

LOGDIR="$SCRIPT_DIR/logs"
ENV_ARCHIVE="$LOGDIR/env_archive"
mkdir -p "$LOGDIR" "$ENV_ARCHIVE"
for _old in "$LOGDIR/deepsearch_retrieval.log" "$LOGDIR/deepsearch_envservice.log"; do
    if [ -s "$_old" ]; then
        mv "$_old" "$ENV_ARCHIVE/$(basename "$_old" .log)_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
done

kill_port() {
    local port=$1
    # -sTCP:LISTEN 只杀监听者(lsof 会把连着该端口的客户端一并列出,
    # 2026-08-08 曾误杀采集驱动 rc=143)。
    local pids=$(lsof -ti:$port -sTCP:LISTEN 2>/dev/null)
    local pid args owner me
    me=$(id -un)
    for pid in $pids; do
        owner=$(ps -p "$pid" -o user= 2>/dev/null | tr -d ' ')
        if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
            echo "  REFUSING to kill PID $pid on port $port (owned by '$owner', not '$me')"
            continue
        fi
        args=$(ps -p "$pid" -o args= 2>/dev/null)
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
    echo "Stopping DeepSearch services..."
    kill_port $RETRIEVAL_PORT
    kill_port $ENVSERVICE_PORT
    echo "Done."
    exit 0
fi

echo "=== Starting DeepSearch Environment Stack ==="
[ -d "$BM25_INDEX" ] || { echo "ERROR: BM25 index missing at $BM25_INDEX (build with pyserini first)"; exit 1; }
[ -x "$WSPY" ] || { echo "ERROR: agentenv-webshop python missing at $WSPY"; exit 1; }

kill_port $RETRIEVAL_PORT
kill_port $ENVSERVICE_PORT

echo "[1/2] Starting BM25 retrieval server on port $RETRIEVAL_PORT..."
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" \
nohup $WSPY "$SCRIPT_DIR/env_service/launch_script/retrieval_server.py" \
    --index "$BM25_INDEX" --port $RETRIEVAL_PORT \
    > "$LOGDIR/deepsearch_retrieval.log" 2>&1 &
disown

for i in $(seq 1 60); do
    if curl -sf --max-time 5 http://127.0.0.1:$RETRIEVAL_PORT/health | grep -q "ok" 2>/dev/null; then
        echo "  retrieval server ready (PID: $(lsof -ti:$RETRIEVAL_PORT | tail -1))"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ERROR: retrieval server failed to start. Check $LOGDIR/deepsearch_retrieval.log"
        exit 1
    fi
    sleep 2
done

echo "[2/2] Starting env_service wrapper on port $ENVSERVICE_PORT..."
rm -rf "${RAY_TMPDIR}/session_"* 2>/dev/null || true
export DEEPSEARCH_RETRIEVAL_URL="http://127.0.0.1:$RETRIEVAL_PORT"
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" \
nohup $PYTHON -m env_service.env_service --env deepsearch --portal 127.0.0.1 --port $ENVSERVICE_PORT \
    > "$LOGDIR/deepsearch_envservice.log" 2>&1 &
disown

for i in $(seq 1 30); do
    if lsof -ti:$ENVSERVICE_PORT >/dev/null 2>&1; then
        echo "  env_service ready (PID: $(lsof -ti:$ENVSERVICE_PORT | head -1))"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ERROR: env_service failed to start. Check $LOGDIR/deepsearch_envservice.log"
        kill_port $RETRIEVAL_PORT
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== DeepSearch Environment Stack Running ==="
echo "  retrieval (BM25): http://127.0.0.1:$RETRIEVAL_PORT"
echo "  env_service:      http://127.0.0.1:$ENVSERVICE_PORT"
echo "  Logs:             $LOGDIR/deepsearch_*.log"
echo "  To stop: bash start_env_deepsearch.sh stop"
