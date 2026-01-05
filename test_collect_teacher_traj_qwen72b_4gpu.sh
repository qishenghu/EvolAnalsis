#!/bin/bash
# ============================================================================
# 配置示例：4卡80G + Qwen-72B-Instruct 大规模轨迹采集
# ============================================================================
# 
# 硬件配置：
#   - 4 × NVIDIA GPU (80GB each, e.g., A100/H100)
#   - 模型：Qwen-72B-Instruct
# 
# 推荐配置：
#   - tensor_parallel_size: 4 (使用所有4张卡)
#   - gpu_memory_utilization: 0.85-0.9 (72B模型在4卡80G上足够)
#   - max_workers: 12-16 (根据环境服务能力调整)
#   - max_num_seqs: 512-1024 (vLLM并发序列数，根据GPU内存调整)
# 
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY=""
export OPENAI_API_BASE="https://api.openai.com/v1"

# 设置 PYTHONPATH 指向项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 模型配置
MODEL=/data/models/Qwen2.5-72B-Instruct  # 修改为你的模型路径
MODEL_NAME=qwen72b

# ============================================================================
# 推荐配置（2000 tasks × 10 rollouts）
# ============================================================================
python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --env_url http://localhost:8081 \
    --backend vllm \
    --model_path ${MODEL} \
    --task_file data/alfworld/task_ids.txt \
    --output data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}.jsonl \
    \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.88 \
    --max_num_seqs 512 \
    \
    --n_per_task 10 \
    --max_workers 12 \
    --max_steps 30 \
    --temperature 0.6 \
    --max_tokens 4096 \
    \
    --no_filter_success \
    --max_retries 3 \
    --save_every 100 \
    --resume

# ============================================================================
# 配置说明：
# ============================================================================
# 
# 【GPU 配置】
# --tensor_parallel_size 4          # 使用4张GPU进行张量并行
# --gpu_memory_utilization 0.88    # GPU内存使用率88%（留12%缓冲）
# --max_num_seqs 512                # vLLM并发序列数（512-1024之间，根据内存调整）
# 
# 【采集配置】
# --n_per_task 10                  # 每个task采集10个rollouts
# --max_workers 12                  # 12个并行worker（可根据环境服务能力调整到8-16）
# --max_steps 30                    # 每个轨迹最多30步
# --temperature 0.6                 # 采样温度0.6（已设为默认值）
# --max_tokens 4096                 # 每次LLM调用最多生成4096 tokens
# 
# 【其他配置】
# --no_filter_success               # 不过滤失败轨迹（保留所有轨迹）
# --max_retries 3                   # 每个rollout最多重试3次
# --save_every 100                  # 每100个轨迹保存一次检查点
# --resume                          # 支持断点续采
# 
# ============================================================================
# 性能优化建议：
# ============================================================================
# 
# 1. 【max_workers 调整】
#    - 如果环境服务是瓶颈：降低到 8-10
#    - 如果GPU利用率低：可以提高到 16-20
#    - 监控GPU利用率：nvidia-smi -l 1
# 
# 2. 【max_num_seqs 调整】
#    - 如果GPU内存不足（OOM）：降低到 256
#    - 如果GPU利用率低：可以提高到 1024
#    - 72B模型在4卡80G上，512通常是最优值
# 
# 3. 【gpu_memory_utilization 调整】
#    - 如果经常OOM：降低到 0.8-0.85
#    - 如果内存充足：可以提高到 0.9
# 
# 4. 【监控指标】
#    - GPU利用率：应该保持在 80-95%
#    - GPU内存：应该保持在 70-85GB（每卡）
#    - 环境服务延迟：应该 < 1秒
# 
# ============================================================================
# 预期性能：
# ============================================================================
# 
# - 总任务数：2000 tasks × 10 rollouts = 20,000 个rollouts
# - 并行加速：12 workers 理论上可达到 8-12倍加速
# - 预计时间：取决于每个rollout的平均时间
#   * 如果每个rollout平均30秒：串行需要 166小时，并行需要 14-21小时
#   * 如果每个rollout平均60秒：串行需要 333小时，并行需要 28-42小时
# 
# ============================================================================

