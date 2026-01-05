# Teacher Trajectory Pipeline 测试指南

## 快速测试命令

```bash
cd /home/qisheng/agent/AgentEvolver
source ~/anaconda3/bin/activate agentevolver

# 运行快速测试
python tests/quick_test_teacher.py

# 运行完整测试
python tests/test_teacher_trajectory_pipeline.py
```

## 测试数据

当前测试数据：`data/teacher_trajectories/alfworld_qwen7b.jsonl`

包含：
- 2 个 task（1430, 799）
- 每个 task 2 个 trajectories
- 共 4 个 trajectories

## 手动验证步骤

### Step 1: 验证 JSONL 格式

```python
import json
import numpy as np

with open('data/teacher_trajectories/alfworld_qwen7b.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if line.strip():
            d = json.loads(line)
            print(f"Traj {i}: task_id={d.get('task_id')}, reward={d.get('reward')}")
            print(f"  Keys: {list(d.keys())}")
            
            # 检查 log_probs
            if 'log_probs' in d:
                lp = d['log_probs']
                print(f"  log_probs: len={len(lp)}, mean={np.mean(lp):.4f}")
```

### Step 2: 验证 ExperienceManager 加载

```python
from agentevolver.module.exp_manager.exp_manager import ExperienceManager

config = {
    "teacher_experience": {
        "enable": True,
        "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
        "exp_ratio": 0.25,
        "select_mode": "random",
        "use_log_prob": True,
    },
    "experience_replay": {"enable": False}
}

exp_manager = ExperienceManager(config)

# 应该看到：
# - teacher_enabled: True
# - 2 个 task 被加载
# - 每个 task 2 个 trajectories

print(f"Teacher enabled: {exp_manager.teacher_enabled}")
print(f"Tasks: {list(exp_manager.teacher_task2trajectories.keys())}")

for tid, trajs in exp_manager.teacher_task2trajectories.items():
    print(f"Task {tid}: {len(trajs)} trajectories")
    for t in trajs:
        print(f"  - has_log_prob: {t.metadata.get('has_log_prob')}")
        print(f"  - is_teacher: {t.metadata.get('is_teacher')}")
```

### Step 3: 验证轨迹选择

```python
# 测试不同的 select_mode
task_ids = list(exp_manager.teacher_task2trajectories.keys())

# random 模式
trajs = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
print(f"Random: got {len(trajs)} trajectories")

# confidence 模式（如果有 log_prob）
exp_manager.teacher_select_mode = "confidence"
trajs = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
print(f"Confidence: got {len(trajs)} trajectories")

# entropy 模式
exp_manager.teacher_select_mode = "entropy"
trajs = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
print(f"Entropy: got {len(trajs)} trajectories")
```

### Step 4: 验证 log_prob 对齐逻辑

```python
# 检查 log_probs 格式是否适合对齐
traj = trajs[0]

if traj.metadata.get("has_log_prob"):
    log_probs = traj.metadata.get("old_log_probs", [])
    print(f"old_log_probs length: {len(log_probs)}")
    
    # 这些 log_probs 将在 env_manager._align_teacher_log_probs() 中
    # 与 tokenized response 的 loss_mask 对齐
    
    # 对齐逻辑：
    # 1. loss_mask 标记 LLM 响应位置
    # 2. 按顺序将 teacher log_probs 填充到 loss_mask=1 的位置
    # 3. 其他位置填充 0
```

### Step 5: 验证 Batch 格式

```python
import torch

# 模拟 batch 中的关键字段
batch_size = 4
response_length = 100

# exp_mask: 标记 self-generated off-policy 样本
exp_mask = torch.zeros(batch_size, response_length)

# teacher_mask: 标记 teacher off-policy 样本
teacher_mask = torch.zeros(batch_size, response_length)
teacher_mask[0, :] = 1  # 第一个样本是 teacher

# recorded_old_log_probs: 历史 log_probs
recorded_old_log_probs = torch.zeros(batch_size, response_length)
# teacher 样本的 log_probs 会在这里填充

# loss_mask: 标记需要计算 loss 的位置
loss_mask = torch.ones(batch_size, response_length)

print(f"exp_mask shape: {exp_mask.shape}")
print(f"teacher_mask shape: {teacher_mask.shape}")
print(f"recorded_old_log_probs shape: {recorded_old_log_probs.shape}")
```

## 集成测试：使用小配置运行训练

创建一个测试配置来验证端到端流程：

```yaml
# config/test_teacher_only.yaml
# 使用很小的设置来快速验证

data:
  train_batch_size: 2
  max_train_tasks: 4

exp_manager:
  experience_replay:
    enable: false
  
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_qwen7b.jsonl"
    exp_ratio: 0.5  # 50% teacher
    select_mode: "random"
    use_log_prob: true  # 使用 Qwen 的 log_prob

trainer:
  total_epochs: 1
  save_freq: 1000  # 不保存
  test_freq: 1000  # 不测试
```

然后运行：

```bash
python launcher.py --config config/test_teacher_only.yaml
```

观察日志中是否出现：
- `Teacher experience enabled`
- `Loaded X teacher trajectories`
- `teacher_exp_ratio: 0.5`
- Actor loss 包含 `teacher_pg_loss` 指标

## 关键验证点

| 验证点 | 如何检查 | 预期结果 |
|--------|----------|----------|
| 加载正确 | `exp_manager.teacher_task2trajectories` | 2 个 task，4 个 trajectories |
| has_log_prob | `traj.metadata["has_log_prob"]` | 根据采集时是否收集 log_prob |
| is_teacher | `traj.metadata["is_teacher"]` | True |
| select_mode | 切换不同模式 | 轨迹顺序变化 |
| batch 格式 | 检查 `batch["teacher_mask"]` | 正确标记 teacher 样本 |
| loss 计算 | 查看 `actor/teacher_pg_loss` 指标 | 非零值 |

## 常见问题

### 1. log_probs 对齐问题

如果 Teacher 和 Policy 使用不同的 tokenizer：
- 设置 `use_log_prob: false`
- 系统将使用 LUFFY 风格的 importance ratio

### 2. 轨迹格式问题

确保 JSONL 中的每条轨迹包含：
- `task_id`: 任务 ID
- `messages`: 消息列表
- `reward`: 奖励值
- `metadata.is_teacher`: True

### 3. 没有 log_probs

如果采集时没有收集 log_probs：
- `has_log_prob` 会是 False
- 使用 `use_log_prob: false` 配置

