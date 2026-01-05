#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=""
export OPENAI_API_BASE="https://api.openai.com/v1"

# 设置 PYTHONPATH 指向项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


# 激活 conda 环境（如果需要）
# source ~/anaconda3/bin/activate agentevolver

# 确保 alfworld 环境服务已启动
# 可以通过 env_service/launch_script/alfworld.sh 启动
# 例如: bash env_service/launch_script/alfworld.sh

# # MODEL=Qwen/Qwen2.5-7B-Instruct
# # MODEL_NAME=qwen7b

# python scripts/collect_teacher_trajectories.py \
#     --env alfworld \
#     --env_url http://localhost:8081 \
#     --backend vllm \
#     --model_path ${MODEL} \
#     --task_file data/alfworld/task_ids.txt \
#     --output data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}.jsonl \
#     --no_filter_success \
#     --max_tasks 2 \
#     --tensor_parallel_size 2 \
#     --n_per_task 2 \
#     --max_steps 30

# python scripts/validate_teacher_trajectories.py \
#     --input data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}.jsonl \
#     --verbose


MODEL=gpt-4.1-mini
MODEL_NAME=gpt41mini

python scripts/collect_teacher_trajectories.py \
    --env alfworld \
    --env_url http://localhost:8081 \
    --backend openai \
    --model_path ${MODEL} \
    --task_file data/alfworld/task_ids.txt \
    --output data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}.jsonl \
    --no_filter_success \
    --max_tasks 2 \
    --tensor_parallel_size 2 \
    --n_per_task 2 \
    --collect_log_prob true \
    --max_steps 30

# python scripts/validate_teacher_trajectories.py \
#     --input data/teacher_trajectories/${MODEL_NAME}/alfworld_${MODEL_NAME}.jsonl \
#     --verbose