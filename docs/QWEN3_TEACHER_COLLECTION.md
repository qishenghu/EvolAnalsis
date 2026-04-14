# Qwen3 Teacher Trajectory Collection Guide

## 概述

使用 `Qwen/Qwen3-30B-A3B-Thinking-2507`（MoE, 3B 活跃参数）作为 teacher 模型，为 Qwen3-1.7B student 模型采集 teacher trajectories。

## 为什么需要 Qwen3 teacher？

- Qwen3 是 reasoning 模型，有 native thinking mode（`<think>` special tokens）
- Qwen2.5-72B 是普通 instruct 模型，其推理风格可能不适合 Qwen3 student
- Reasoning teacher + reasoning student = 更自然的知识传递

## 环境准备（4xA100-80G 服务器）

### 1. Clone 并安装

```bash
git clone <repo_url>
cd EvolAnalsis
bash setup_envs.sh        # 安装 conda env + ALFWorld + WebShop
source env_config.sh
```

### 2. 下载 Teacher 模型

```bash
# Qwen3-30B-A3B-Thinking (~8GB, MoE 模型)
huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --local-dir /data/shared_models/Qwen3-30B-A3B-Thinking

# 也下载 student 模型（后续训练用）
huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir /data/shared_models/Qwen3-1.7B
```

### 3. 下载 Teacher 数据所需的模型（用于 vLLM 推理）

Qwen3-30B-A3B 是 MoE 模型，4x80G 可以轻松跑。vLLM 原生支持 MoE 推理。

## 采样执行

### ALFWorld（2,348 tasks）

```bash
bash start_env_alfworld.sh
nohup bash scripts/collect_qwen3_teacher.sh alfworld \
    > logs/collect_qwen3_alfworld.log 2>&1 &
```

### WebShop（5,691 tasks）

```bash
bash start_env_webshop.sh
nohup bash scripts/collect_qwen3_teacher.sh webshop \
    > logs/collect_qwen3_webshop.log 2>&1 &
```

### SciWorld（800 tasks，seed=2026 定向子集）

SciWorld 只采集训练时会用到的 800 个 task（通过 `seed=2026` + `max_train_tasks=800` 确定的子集），而非全部 4137 个 train task。Task ID 文件已生成在 `data/sciworld/task_ids_800_seed2026.txt`。

```bash
# 启动 SciWorld 环境（需要先配置）
# bash start_env_sciworld.sh

nohup bash scripts/collect_qwen3_teacher.sh sciworld \
    > logs/collect_qwen3_sciworld.log 2>&1 &
```

## 采样参数

| 参数 | 值 | 说明 |
|------|------|------|
| `--use_qwen3` | true | 启用 native thinking mode，non-final turns 追加 `/no_think` |
| `--n_per_task` | 2 | 每个 task 采 2 条轨迹，过滤成功的 |
| `--filter_success` | true | 只保留 reward=1.0 的轨迹 |
| `--temperature` | 0.6 | 采样温度 |
| `--max_workers` | 8 | 并行采样 workers |
| `--backend` | vllm | 使用 vLLM 本地推理（收集 log_probs） |

## 输出文件

```
data/teacher_trajectories/qwen3_30b/
├── alfworld_qwen3_30b.jsonl              # 原始采集数据
├── alfworld_qwen3_30b_filtered.pkl       # 过滤后（仅成功）
├── alfworld_qwen3_30b_filtered_stats.json
├── webshop_qwen3_30b.jsonl
├── webshop_qwen3_30b_filtered.pkl
└── webshop_qwen3_30b_filtered_stats.json
```

## 数据格式

每条轨迹结构：
```json
{
    "task_id": "123",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "OK. I'll follow..."},
        {"role": "user", "content": "You are in a room..."},
        {"role": "assistant", "content": "<think>\n...\n</think>\n<action>\ngo to desk 1\n</action>"},
        {"role": "user", "content": "On the desk, you see..."}
    ],
    "reward": 1.0,
    "success": true,
    "teacher_model": "Qwen3-30B-A3B-Thinking",
    "log_probs": [...],
    "log_probs_per_turn": [...]
}
```

**注意**：Qwen3 的 `<think>` 是 native special token（ID 151667/151668），但存储为文本时就是普通的 `<think>` 字符串。Student tokenizer 会自动正确 re-tokenize。

## 预期数据量

| 环境 | Tasks | 采样 | 预期成功 | 过滤后 |
|------|-------|------|---------|--------|
| ALFWorld | 2,348 | 4,696 | ~60-80% | ~3,000-4,000 |
| WebShop | 5,691 | 11,382 | ~50-70% | ~6,000-8,000 |
| SciWorld | 800 | 1,600 | ~40-60% | ~600-1,000 |

> SciWorld 只采 800 个 task（与训练的 `max_train_tasks=800, seed=2026` 对齐），确保 teacher 数据覆盖训练时用到的所有 task。

Qwen3-30B-A3B-Thinking 是 reasoning 模型，预期比 Qwen2.5-72B 有更高的成功率（尤其在需要推理的任务上）。

## 后续步骤

采样完成后：
1. 将 `data/teacher_trajectories/qwen3_30b/*.pkl` 拷贝到 8xA100 训练服务器
2. 创建 Qwen3-1.7B 的训练 configs（指向新 teacher 数据路径）
3. 运行训练实验

## 故障排查

- **vLLM OOM**：Qwen3-30B-A3B MoE 只有 3B 活跃参数，4x80G 应该不会 OOM。如果 OOM，减少 `--max_workers`
- **环境连接失败**：确认 env_service 在对应端口运行（ALFWorld: 8081, WebShop: 8083）
- **采样中断**：脚本支持断点续传（`--save_every`），重新运行会跳过已采集的 task
- **成功率过低**：检查 `logs/collect_qwen3_*.log` 中的轨迹样本，确认模型输出格式正确
