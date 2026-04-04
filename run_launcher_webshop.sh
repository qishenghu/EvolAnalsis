export CUDA_VISIBLE_DEVICES=0,1,2,3

WEBSHOP_PORT=36003
WEBSHOP_HOST=0.0.0.0
WEBSHOP_DIR="AgentGym/agentenv-webshop"

# 函数：kill占用指定端口的进程
kill_port() {
    local port=$1
    local pids=""

    # 方法1: 使用 lsof (如果可用)
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:$port 2>/dev/null)
    # 方法2: 使用 fuser (如果可用)
    elif command -v fuser >/dev/null 2>&1; then
        pids=$(fuser $port/tcp 2>/dev/null | awk '{print $1}')
    # 方法3: 使用 ss + awk (通常可用)
    elif command -v ss >/dev/null 2>&1; then
        pids=$(ss -tlnp 2>/dev/null | grep ":$port " | awk '{print $6}' | sed 's/.*pid=\([0-9]*\).*/\1/' | sort -u)
    # 方法4: 使用 netstat (备用)
    elif command -v netstat >/dev/null 2>&1; then
        pids=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | sort -u)
    fi

    if [ -n "$pids" ]; then
        echo "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    else
        echo "No process found on port $port"
    fi
}

# WebShop 依赖 Java（pyserini/jnius）
export JAVA_HOME="/data/code/exp/anaconda3/envs/agentenv-webshop/lib/jvm"
export PATH="$JAVA_HOME/bin:$PATH"

# WebShop 的实际 python binary（避免 conda run 后台启动的问题）
WEBSHOP_BIN="/data/code/exp/anaconda3/envs/agentenv-webshop/bin/webshop"

# 函数：重启 WebShop 底层服务
restart_webshop() {
    echo ""
    echo "=========================================="
    echo "  Restarting WebShop service (port $WEBSHOP_PORT)..."
    echo "=========================================="

    # 1. Kill 现有 webshop 进程
    kill_port $WEBSHOP_PORT
    pkill -f "webshop --host" 2>/dev/null || true
    sleep 2

    # 2. 直接用 binary 后台启动（跳过 conda run wrapper）
    echo "Starting webshop service..."
    $WEBSHOP_BIN --host $WEBSHOP_HOST --port $WEBSHOP_PORT > /tmp/webshop_service.log 2>&1 &
    WEBSHOP_PID=$!
    echo "WebShop PID: $WEBSHOP_PID"

    # 3. 等待服务就绪
    echo "Waiting for WebShop to be ready on port $WEBSHOP_PORT..."
    for i in $(seq 1 180); do
        if ! kill -0 $WEBSHOP_PID 2>/dev/null; then
            echo "ERROR: WebShop process (PID $WEBSHOP_PID) died."
            echo "Last log lines:"
            tail -20 /tmp/webshop_service.log 2>/dev/null
            return 1
        fi
        if curl -s --max-time 2 http://127.0.0.1:${WEBSHOP_PORT}/ > /dev/null 2>&1; then
            echo "WebShop service is ready! (took ${i}s)"
            return 0
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "  Still waiting... (${i}s elapsed)"
            tail -3 /tmp/webshop_service.log 2>/dev/null
        fi
        sleep 1
    done

    echo "WARNING: WebShop not ready after 180s. Last log:"
    tail -10 /tmp/webshop_service.log 2>/dev/null
    return 1
}

ENV_SERVICE_PORT=8083

# 函数：确保 env_service (FastAPI gateway on 8083) 在运行
ensure_env_service() {
    if curl -s --max-time 2 http://127.0.0.1:${ENV_SERVICE_PORT}/ > /dev/null 2>&1; then
        echo "env_service (port $ENV_SERVICE_PORT) is already running."
        return 0
    fi

    echo "Starting env_service on port $ENV_SERVICE_PORT..."
    conda run -n agentevolver \
        python -m env_service.env_service --env webshop --portal 127.0.0.1 --port $ENV_SERVICE_PORT \
        > /tmp/env_service.log 2>&1 &
    ENV_SERVICE_PID=$!

    for i in $(seq 1 30); do
        if curl -s --max-time 2 http://127.0.0.1:${ENV_SERVICE_PORT}/ > /dev/null 2>&1; then
            echo "env_service is ready! (took ${i}s)"
            return 0
        fi
        sleep 1
    done

    echo "WARNING: env_service may not be ready."
    tail -10 /tmp/env_service.log 2>/dev/null
    return 1
}

# 函数：清理训练残留进程（不杀 env_service 和 webshop）
cleanup_training() {
    echo "Cleaning up training processes..."
    # 杀 vLLM 相关
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    # 杀 main_ppo 相关
    pkill -f "agentevolver.main_ppo" 2>/dev/null || true
    # 杀 Ray worker（但保留 Ray head 和 env_service 的 Ray）
    pkill -f "ray::TaskRunner" 2>/dev/null || true
    pkill -f "ray::RolloutWorker" 2>/dev/null || true
    pkill -f "ray::ActorRolloutRef" 2>/dev/null || true
    sleep 3
    # 释放 GPU 显存
    nvidia-smi > /dev/null 2>&1 || true
}

# 函数：运行单个实验（带 webshop 重启）
run_experiment() {
    local config=$1
    local name=$2

    echo ""
    echo "############################################"
    echo "  Running: $name"
    echo "  Config:  $config"
    echo "############################################"
    echo ""

    restart_webshop
    ensure_env_service

    python launcher.py --conf "$config"
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "WARNING: $name exited with code $exit_code"
    fi

    # 只清理训练相关进程，不杀 env_service / webshop
    cleanup_training

    return $exit_code
}

# ==========================================
#  WebShop 3B Paper Experiments
# ==========================================

# WebShop GRPO (on-policy baseline)
# run_experiment \
#     config/duet_paper_experiments_configs/webshop/webshop_3b_onpolicy.yaml \
#     "WebShop 3B GRPO"

# WebShop LUFFY
# run_experiment \
#     config/duet_paper_experiments_configs/webshop/webshop_3b_luffy.yaml \
#     "WebShop 3B LUFFY"

# # WebShop CHORDß
# run_experiment \
#     config/duet_paper_experiments_configs/webshop/webshop_3b_chord.yaml \
#     "WebShop 3B CHORD"

# ==========================================
#  0405 Experiments: Attribute-Aware SC + Bug Fixes
# ==========================================

# DUET 0405: DR3 (disc_temp=1.0) + attribute_aware SC
run_experiment \
    config/duet_paper_experiments_configs/webshop/webshop_3b_duet_0405.yaml \
    "WebShop 3B DUET 0405"

# DUET Hybrid 0405: DR3 w_hat × LUFFY p/(p+β) + attribute_aware SC
run_experiment \
    config/duet_paper_experiments_configs/webshop/webshop_3b_duet_hybrid_0405.yaml \
    "WebShop 3B DUET_Hybrid 0405"

# LUFFY+SC 0405: LUFFY p/(p+β) + attribute_aware SC
run_experiment \
    config/duet_paper_experiments_configs/webshop/webshop_3b_luffy_sc_0405.yaml \
    "WebShop 3B LUFFY+SC 0405"



# 清理
echo ""
echo "All WebShop 3B 0405 experiments completed!"
kill_port $WEBSHOP_PORT
pkill -f "webshop --host" 2>/dev/null || true
