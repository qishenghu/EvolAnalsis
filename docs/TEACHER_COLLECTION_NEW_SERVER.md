# Qwen3-30B Teacher Collection — New Server Setup Guide

## 目标

在新服务器上用 `Qwen3-30B-A3B-Thinking-2507` 采集 **WebShop** 和 **SciWorld** 的 teacher trajectories，用于辅助 `Qwen3-1.7B` 的 DUET 训练。

ALFWorld 已在另一台 4×H100 服务器上采集。

## 硬件要求

- **GPU**: 4×H200 144GB（或任何 ≥2 张 80GB+ 的 GPU）
- **磁盘**: ≥100GB 可用空间（模型 61GB + 环境数据 + 输出）
- **内存**: ≥64GB RAM

## 模型信息

| 项目 | 值 |
|------|-----|
| Teacher 模型 | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| 架构 | `qwen3_moe` (MoE, ~30B total / ~3B active) |
| 大小 | ~61GB (BF16, 16 safetensor shards) |
| 推理配置 | TP=2, gpu_memory_utilization=0.90 |
| Student 模型 | `Qwen/Qwen3-1.7B` (same tokenizer, log_probs directly reusable) |
| vLLM 版本要求 | ≥ 0.8.5 (支持 `qwen3_moe`) |

## 采集参数

| 参数 | WebShop | SciWorld |
|------|---------|----------|
| Tasks | 800 (seed=2026) | 800 (seed=2026) |
| Task file | `data/webshop/task_ids_800_seed2026.txt` | `data/sciworld/task_ids_800_seed2026.txt` |
| Mode | stop_on_success | stop_on_success |
| Max attempts/task | 10 | 20 |
| Max steps/turn | 15 | 40 |
| Action format | react_tags (`<think>/<action>`) | react_tags (`<think>/<action>`) |
| Success threshold | default (reward ≥ 1.0) | 70 (score > 70) |
| Workers per instance | 4 | 4 |
| Parallel instances | 2 (GPU 0,1 + GPU 2,3) | 2 (GPU 0,1 + GPU 2,3) |

## 输出格式

每条轨迹是 JSONL 格式：
```json
{
  "task_id": "123",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "content": "<think>\n..reasoning..\n</think>\n<action>\naction text\n</action>"},
    {"role": "user", "content": "..env observation.."}
  ],
  "reward": 1.0,
  "success": true,
  "teacher_model": "Qwen3-30B-A3B-Thinking-2507",
  "log_probs": [float, ...],
  "log_probs_per_turn": [...],
  "metadata": {"is_teacher": true, "has_log_prob": true, ...}
}
```

## 快速开始

```bash
# 1. Clone repo
git clone https://github.com/qishenghu/EvolAnalsis.git
cd EvolAnalsis

# 2. 运行自动化设置脚本
bash setup_new_server.sh

# 3. 采集（后台运行）
nohup bash run_qwen3_teacher_parallel.sh webshop > logs/teacher_webshop.log 2>&1 &
nohup bash run_qwen3_teacher_parallel.sh sciworld > logs/teacher_sciworld.log 2>&1 &

# 或一起跑（顺序执行 webshop → sciworld）
nohup bash run_qwen3_teacher_parallel.sh all > logs/teacher_parallel.log 2>&1 &
```

## 详细设置步骤（如果自动脚本失败）

### Step 1: Conda 环境

需要以下 conda 环境：

```bash
# 主环境（duet）— 用于采集脚本和 vLLM
conda create -n duet python=3.11 -y
conda activate duet
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5
pip install transformers huggingface_hub loguru omegaconf tqdm jieba

# WebShop 环境
conda create -n agentenv-webshop python=3.9 -y
conda activate agentenv-webshop
cd AgentGym/agentenv-webshop
pip install -e .
# WebShop 需要 Java:
conda install -c conda-forge openjdk=11 -y

# SciWorld 环境
conda create -n agentenv-sciworld python=3.8 -y
conda activate agentenv-sciworld
cd AgentGym/agentenv-sciworld
pip install -e .
# SciWorld 需要 Java 1.8+:
conda install -c conda-forge openjdk=11 -y
```

### Step 2: 下载模型

```bash
conda activate duet
huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --local-dir models/Qwen/Qwen3-30B-A3B-Thinking-2507

# 可选：下载 student 模型用于验证
huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir models/Qwen/Qwen3-1.7B
```

### Step 3: 配置路径

编辑 `run_qwen3_teacher_parallel.sh` 中的路径：
```bash
export CONDA_PATH="/path/to/your/anaconda3"  # 修改为实际 conda 路径
```

或设置环境变量：
```bash
export CONDA_PATH=/path/to/anaconda3
```

### Step 4: 验证环境

```bash
# 测试 WebShop 环境
bash run_qwen3_teacher_parallel.sh webshop  # 先前台跑看输出

# 测试 SciWorld 环境
bash run_qwen3_teacher_parallel.sh sciworld
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `run_qwen3_teacher_parallel.sh` | 主采集脚本（2×TP=2 并行，400+400 task 拆分） |
| `scripts/collect_qwen3_teacher.sh` | 单实例采集脚本（被 parallel 脚本内部调用） |
| `scripts/collect_teacher_trajectories.py` | 底层采集逻辑 |
| `scripts/validate_teacher_for_training.py` | 验证采集数据可用于 1.7B 训练 |
| `data/{webshop,sciworld}/task_ids_800_seed2026.txt` | 定向 800 task IDs |
| `agentevolver/module/teacher/vllm_teacher_llm.py` | vLLM teacher LLM 封装（含 `<think>` 前缀修复） |

## 已知问题和解决方案

### 1. Ray 磁盘空间检查 (>95% 报错)
Ray 的 `file_system_monitor` 会在磁盘 >95% 满时拒绝工作。解决：
```bash
# 在 run_qwen3_teacher_parallel.sh 中设置 RAY_TMPDIR 到空间充足的分区
export RAY_TMPDIR="/tmp/ray_envsvc"  # 而非 /data/ray
```

### 2. Conda 在 nohup 中不可用
nohup 后台运行时 conda 可能未初始化。脚本已处理：用 `$CONDA_PATH/envs/duet/bin/python` 绝对路径。

### 3. SciWorld 需要 Java
SciWorld 依赖 Java。确保 JAVA_HOME 设置正确：
```bash
export JAVA_HOME="${CONDA_PATH}/envs/agentenv-sciworld/lib/jvm"
export PATH="${JAVA_HOME}/bin:${PATH}"
```

### 4. `<think>` 标签不出现
Qwen3 chat template 会将 `<think>\n` 作为 generation prompt 的一部分追加到 prompt 末尾，但 vLLM `.text` 只返回新生成的 token。代码已修复（`vllm_teacher_llm.py`）：检测到 prompt 以 `<think>` 结尾时自动补回前缀。

### 5. 端口冲突
脚本启动时会自动清理端口，但如果遇到问题：
```bash
# 手动清理
for port in 8081 8083 8085 36001 36003 36004; do
    lsof -ti:$port | xargs kill -9 2>/dev/null
done
ray stop --force
```

## 监控采集进度

```bash
# 实时日志
tail -f logs/collect_webshop_partA.log
tail -f logs/collect_sciworld_partB.log

# 进度统计
for part in A B; do
    echo "Part $part:"
    grep -c "success=True" logs/collect_webshop_part${part}.log 2>/dev/null
    echo "successes"
done

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

## 采集完成后

1. 验证数据：
```bash
python scripts/validate_teacher_for_training.py \
    --teacher_data data/teacher_trajectories/qwen3_30b/webshop_qwen3_30b.jsonl \
    --student_model models/Qwen/Qwen3-1.7B
```

2. 将 `data/teacher_trajectories/qwen3_30b/` 目录下的 `.jsonl` 和 `.pkl` 文件传回训练服务器。

3. 训练时在 config 中指定：
```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: data/teacher_trajectories/qwen3_30b/webshop_qwen3_30b_filtered.pkl
```
