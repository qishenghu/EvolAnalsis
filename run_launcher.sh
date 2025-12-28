export CUDA_VISIBLE_DEVICES=0,1,2,3

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

# 在启动主程序之前，先kill掉8081和8124端口的程序
# echo "Checking and killing processes on ports 8081 and 8124..."
# kill_port 8081
# kill_port 8124

# Run GRPO baseline (default)
python launcher.py \
  --conf config/alfworld_grpo_3b_exp_replay.yaml
 # --with-alfworld

# Run GRPO with Experience Replay
# python launcher.py \
#   --conf config/alfworld_grpo_7b_exp_replay.yaml
 # --with-alfworld
