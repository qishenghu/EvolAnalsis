# Teacher Experience Replay - 完整流程 Manual

本文档详细 walk through 使用 Teacher Experience Replay（LUFFY）进行训练的完整流程，包括代码位置、数据流和关键细节。

## 目录

1. [概述](#概述)
2. [初始化阶段](#初始化阶段)
3. [训练循环 - LUFFY Rollout 级别混合](#训练循环---luffy-rollout-级别混合)
4. [训练循环 - Task 级别混合（兼容 ExGRPO）](#训练循环---task-级别混合兼容-exgrpo)
5. [Loss 计算](#loss-计算)
6. [关键数据结构](#关键数据结构)
7. [配置项详解](#配置项详解)
8. [常见问题排查](#常见问题排查)

---

## 概述

### LUFFY vs ExGRPO

Teacher Experience Replay 支持两种混合模式：

| 特性 | LUFFY (Rollout-level) | ExGRPO (Task-level) |
|------|----------------------|---------------------|
| **混合粒度** | Rollout 级别 | Task 级别 |
| **设计理念** | 每个 task 内混合 on-policy 和 teacher rollouts | batch 中某些 task 完全是 teacher experience |
| **典型配置** | `n_teacher_rollouts_per_task: 1` | `teacher_exp_ratio: 0.2` |
| **GRPO 影响** | Teacher 的高 reward 影响同 task 内所有 rollouts 的 baseline | Teacher tasks 有独立的 baseline |
| **适用场景** | 引导策略学习 teacher 的行为模式 | 补充稀疏 reward 的正样本 |

### 混合粒度详解

**LUFFY Rollout-level 混合**:
```
Batch 8 tasks, n_rollout=8, n_teacher_per_task=1:
  Task 1: 7 on-policy + 1 teacher  → GRPO 计算 advantage 时 teacher 影响 baseline
  Task 2: 7 on-policy + 1 teacher
  Task 3: 7 on-policy + 1 teacher
  ...
  Task 8: 7 on-policy + 1 teacher
```

**ExGRPO Task-level 混合**:
```
Batch 8 tasks, exp_ratio=0.5, teacher_exp_ratio=0.25:
  Task 1-2: 完全是 teacher off-policy (2 tasks × 8 rollouts = 16 teacher rollouts)
  Task 3-4: 完全是 self-generated off-policy
  Task 5-8: 完全是 on-policy
```

### 两种使用场景

1. **只使用 Teacher Experience Replay**（LUFFY 模式）
   - 无需等待训练积累经验
   - 从第一个 epoch 就可以开始使用
   - 适合有高质量 teacher 轨迹的场景

2. **Teacher + Self-generated 混合使用**
   - Teacher 提供高质量示范
   - Self-generated 提供分布内样本
   - 两者互补，效果更好

---

## 初始化阶段

### 1.1 ExperienceManager 初始化

**位置**: `agentevolver/module/exp_manager/exp_manager.py:52-100`

```python
class ExperienceManager:
    def __init__(self, config: DictConfig):
        # ... 基础配置 ...
        
        # ⭐ Teacher Experience 配置
        teacher_config = config.get("teacher_experience", {})
        self.teacher_enabled = teacher_config.get("enable", False)
        self.teacher_data_path = teacher_config.get("data_path", None)
        
        # ⭐ LUFFY 风格：混合模式配置
        self.teacher_mix_mode = teacher_config.get("mix_mode", "rollout_level")
        self.n_teacher_rollouts_per_task = teacher_config.get("n_teacher_rollouts_per_task", 1)
        
        # Task 级别配置（兼容 ExGRPO）
        self.teacher_exp_ratio = teacher_config.get("exp_ratio", 0.2)
        
        self.teacher_max_per_task = teacher_config.get("max_trajectories_per_task", 3)
        self.teacher_select_mode = teacher_config.get("select_mode", "random")
        self.teacher_use_log_prob = teacher_config.get("use_log_prob", False)
        
        # Teacher 轨迹存储（与 self-generated 分开存储）
        self.teacher_task2trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
        
        # 如果配置了 data_path，尝试加载 Teacher 轨迹
        if self.teacher_enabled and self.teacher_data_path:
            self.load_teacher_trajectories(self.teacher_data_path)
```

### 1.2 加载 Teacher 轨迹

**位置**: `agentevolver/module/exp_manager/exp_manager.py:635-690`

**支持格式**: JSONL 和 PKL

```python
def load_teacher_trajectories(self, data_path: str) -> int:
    """加载 Teacher 轨迹数据"""
    count = 0
    
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                traj_dict = json.loads(line)
                traj = self._dict_to_teacher_trajectory(traj_dict)
                traj.metadata["is_teacher"] = True
                self.teacher_task2trajectories[traj.task_id].append(traj)
                count += 1
                
    elif data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
            for item in trajectories:
                if isinstance(item, dict):
                    traj = self._dict_to_teacher_trajectory(item)
                else:
                    traj = item  # 已经是 Trajectory 对象
                traj.metadata["is_teacher"] = True
                self.teacher_task2trajectories[traj.task_id].append(traj)
                count += 1
    
    logger.info(f"Loaded {count} teacher trajectories from {data_path}")
    return count
```

### 1.3 Teacher 轨迹数据格式

**JSONL 格式** (`data/teacher_trajectories/alfworld_gpt4.jsonl`):

```json
{
  "task_id": "pick_cool_then_place_in_recep-Tomato-None-CounterTop-27",
  "rollout_id": "0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Task: put a cool tomato in countertop."},
    {"role": "assistant", "content": "THOUGHT: I need to find a tomato..."},
    {"role": "user", "content": "You arrive at loc 1. On the countertop 1..."}
  ],
  "reward": 1.0,
  "success": true,
  "teacher_model": "Qwen2.5-72B-Instruct",
  "log_probs": [-0.5, -0.3, -0.8, ...],
  "log_probs_per_turn": [
    {
      "turn_idx": 0,
      "log_probs": [-0.5, -0.3, -0.1],
      "token_ids": [100, 200, 300],
      "tokens": ["I", " need", " to"]
    },
    {
      "turn_idx": 1,
      "log_probs": [-0.8, -0.2],
      "token_ids": [400, 500],
      "tokens": ["find", " tomato"]
    }
  ],
  "metadata": {
    "is_teacher": true,
    "has_log_prob": true,
    "total_generated_tokens": 159,
    "num_turns": 5
  }
}
```

**字段说明**：

| 字段 | 必需 | 说明 |
|------|------|------|
| `task_id` | ✅ | 任务唯一标识 |
| `rollout_id` | ✅ | 同一 task 的 rollout 序号 |
| `messages` | ✅ | 对话历史 |
| `reward` | ✅ | 奖励值 |
| `success` | ✅ | 是否成功 |
| `teacher_model` | ⚠️ | Teacher 模型名（便于追溯） |
| `log_probs` | ⚠️ | 累积的 log_probs（当 `use_log_prob: true` 时需要） |
| `log_probs_per_turn` | ⭐ | 分轮的详细信息（**推荐**，用于精确对齐） |
| `log_probs_per_turn[].token_ids` | ⭐ | 生成的 token ids（**用于精确对齐 chat template 问题**） |
| `metadata.has_log_prob` | ✅ | 标记是否有 log_prob |
| `metadata.is_teacher` | ✅ | 标记为 teacher 轨迹 |

> ⭐ **重要**：当使用 `use_log_prob: true` 时，强烈推荐包含 `log_probs_per_turn` 和 `token_ids`，
> 以支持精确对齐，避免 chat template 导致的位置偏移问题。

---

## 训练循环 - LUFFY Rollout 级别混合

### 2.1 判断混合模式

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1903-1910`

```python
# ⭐ LUFFY vs ExGRPO: 检查混合模式
teacher_mix_mode = teacher_exp_config.get("mix_mode", "rollout_level")
use_luffy_rollout_level = (
    enable_teacher_exp and 
    teacher_mix_mode == "rollout_level"
)
```

### 2.2 创建 LUFFY Mixer

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1918-1935`

```python
if use_luffy_rollout_level:
    n_teacher_rollouts = teacher_exp_config.get("n_teacher_rollouts_per_task", 1)
    n_rollout = self.config.actor_rollout_ref.rollout.n
    
    logger.info(
        f"[LUFFY] Rollout-level mixing: "
        f"n_rollout={n_rollout}, n_teacher_per_task={n_teacher_rollouts}, "
        f"n_onpolicy_per_task={n_rollout - n_teacher_rollouts}"
    )
    
    # 创建 LUFFY mixer
    luffy_mixer = LUFFYTeacherRolloutMixer(
        exp_manager=self.exp_manager,
        n_teacher_rollouts_per_task=n_teacher_rollouts,
        n_rollout=n_rollout,
    )
```

### 2.3 生成 On-policy Rollouts

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2075-2082`

```python
# ⭐ LUFFY 模式：生成完整的 n_rollout on-policy rollouts
# 后续会用 teacher 替换部分（如果 teacher 不足，保留更多 on-policy）
trajectories = self.env_manager.rollout(
    tasks, 
    task_exp_configs, 
    mode="sample", 
    epoch=f"train.{epoch}.{i}",
)
```

**关键点**:
- LUFFY 模式下，**生成完整的 `n_rollout` on-policy rollouts**
- 不预先减少 on-policy 数量
- 这样可以自动处理 teacher 不足的情况

### 2.4 LUFFY 混合（替换模式）

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2085-2109`

```python
if use_luffy_rollout_level and luffy_mixer:
    # 1. 更新 difficulty2task_dict（用完整的 on-policy 轨迹）
    self.exp_manager.update_difficulty2task_dict(trajectories)
    
    # 2. 使用 LUFFY mixer 用 teacher rollouts 替换部分 on-policy
    all_trajectories, luffy_stats = luffy_mixer.mix_trajectories(
        on_policy_cmt_array=trajectories,
        tasks=tasks,
        env_manager=self.env_manager,
        config=self.config,
        tokenizer=self.tokenizer,
    )
    
    # 更新 metrics
    metrics.update({
        "luffy/tasks_with_full_teacher": luffy_stats.get("tasks_with_full_teacher", 0),
        "luffy/tasks_with_partial_teacher": luffy_stats.get("tasks_with_partial_teacher", 0),
        "luffy/tasks_without_teacher": luffy_stats.get("tasks_without_teacher", 0),
        "luffy/total_teacher_rollouts": luffy_stats["total_teacher"],
    })
```

### 2.5 LUFFYTeacherRolloutMixer 内部逻辑

**位置**: `agentevolver/module/exp_manager/experience_collate.py:326-520`

**核心逻辑（替换模式）**:

```python
def mix_trajectories(self, on_policy_cmt_array, tasks, env_manager, config, tokenizer):
    """将 teacher rollouts 混入 on-policy trajectories（替换模式）"""
    
    # 1. 按 task_id 分组 on-policy CMT
    task_id_to_onpolicy: Dict[str, List] = defaultdict(list)
    for cmt in on_policy_cmt_array:
        task_id_to_onpolicy[cmt.task_id].append(cmt)
    
    # 2. 获取每个 task 的 teacher rollouts
    teacher_rollouts_map = self.exp_manager.get_teacher_rollouts_for_luffy_mixing(
        tasks=tasks,
        n_teacher_rollouts_per_task=self.n_teacher_rollouts_per_task,
    )
    
    # 3. 转换 teacher trajectories 为 CMT 对象
    teacher_cmt_array = env_manager.convert_offpolicy_to_cmt(
        offpolicy_trajectories=all_teacher_trajs,
        config=config,
        tokenizer=tokenizer,
        task_id_to_data_id=task_id_to_data_id,
    )
    
    # 4. 混合轨迹
    for task in tasks:
        task_id = task.task_id
        onpolicy_cmts = task_id_to_onpolicy.get(task_id, [])
        teacher_cmts = task_id_to_teacher.get(task_id, [])
        actual_teacher_count = len(teacher_cmts)
        
        # ⭐ 自动处理 teacher 不足：保留更多 on-policy
        n_onpolicy_to_keep = self.n_rollout - actual_teacher_count
        kept_onpolicy_cmts = onpolicy_cmts[:n_onpolicy_to_keep]
        
        # 合并：保留的 on-policy + teacher
        task_cmts = kept_onpolicy_cmts + teacher_cmts
        mixed_cmt_array.extend(task_cmts)
    
    return mixed_cmt_array, stats
```

**自动处理 Teacher 不足**:
```
例如：n_rollout=8, n_teacher_rollouts_per_task=2

Task A 有 2 条 teacher → 6 on-policy + 2 teacher = 8 总计 ✓
Task B 有 1 条 teacher → 7 on-policy + 1 teacher = 8 总计 ✓
Task C 有 0 条 teacher → 8 on-policy + 0 teacher = 8 总计 ✓
```

---

## 训练循环 - Task 级别混合（兼容 ExGRPO）

### 3.1 使用 TeacherExperienceMixCollateFn

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1949-1978`

当 `mix_mode == "task_level"` 时：

```python
if enable_teacher_exp and teacher_mix_mode == "task_level":
    # ⭐ Task 级别：Teacher 从总 off-policy 比例中分走一部分
    total_exp_ratio = exp_replay_config.get("exp_ratio", 0.5)
    teacher_exp_ratio = teacher_exp_config.get("exp_ratio", 0.2)
    teacher_exp_ratio = min(teacher_exp_ratio, total_exp_ratio)
    self_exp_ratio = max(0.0, total_exp_ratio - teacher_exp_ratio)
    
    experience_mix_collate = TeacherExperienceMixCollateFn(
        exp_manager=self.exp_manager,
        train_task_manager=self.train_task_manager,
        self_exp_ratio=self_exp_ratio,
        teacher_exp_ratio=teacher_exp_ratio,
        teacher_exp_enabled=True,
        replay_start_ratio=replay_start_ratio,
        offpolicy_trajectories_per_task=exp_replay_config.get("offpolicy_trajectories_per_task", 1),
        n_rollout=self.config.actor_rollout_ref.rollout.n,
    )
    
    # 返回三元组
    experience_tasks, teacher_exp_tasks, on_policy_tasks = experience_mix_collate(
        training_tasks=tasks,
        training_progress=training_progress,
        enable_replay=True,
    )
```

### 3.2 TeacherExperienceMixCollateFn 内部逻辑

**位置**: `agentevolver/module/exp_manager/experience_collate.py:165-295`

```python
def __call__(self, training_tasks, training_progress, enable_replay):
    batch_size = len(training_tasks)
    
    # 1. 检查是否达到 replay 开始条件
    if not enable_replay or training_progress < self.replay_start_ratio:
        return [], [], training_tasks
    
    # 2. 计算各类型的 task 数量
    target_self_exp_count = int(batch_size * self.self_exp_ratio)
    target_teacher_exp_count = int(batch_size * self.teacher_exp_ratio)
    
    # 3. 采样 self-generated experience tasks
    valid_self_exp_task_ids = self.exp_manager.get_valid_replay_task_ids()
    n_self_exp = min(len(valid_self_exp_task_ids), target_self_exp_count)
    sampled_self_exp_task_ids = random.sample(valid_self_exp_task_ids, n_self_exp)
    
    # 4. 采样 teacher experience tasks（避免与 self_exp 重复）
    valid_teacher_task_ids = self.exp_manager.get_valid_teacher_task_ids()
    available_teacher_task_ids = [
        tid for tid in valid_teacher_task_ids 
        if tid not in sampled_self_exp_task_ids
    ]
    n_teacher_exp = min(len(available_teacher_task_ids), target_teacher_exp_count)
    sampled_teacher_task_ids = random.sample(available_teacher_task_ids, n_teacher_exp)
    
    # 5. 转换为 Task 对象
    self_exp_tasks = self._task_ids_to_tasks(sampled_self_exp_task_ids, is_teacher=False)
    teacher_exp_tasks = self._task_ids_to_tasks(sampled_teacher_task_ids, is_teacher=True)
    
    # 6. 补充 on-policy tasks
    n_on_policy = batch_size - len(self_exp_tasks) - len(teacher_exp_tasks)
    on_policy_tasks = remaining_tasks[:n_on_policy]
    
    return self_exp_tasks, teacher_exp_tasks, on_policy_tasks
```

### 3.3 自动调整 n_offpolicy_trajectories

**位置**: `agentevolver/module/exp_manager/experience_collate.py:297-343`

⭐ **新增修复**：根据实际可用的 trajectories 数量设置 `n_offpolicy_trajectories`

```python
def _task_ids_to_tasks(self, task_ids, is_teacher=False):
    tasks = []
    for task_id in task_ids:
        task = self._get_task_by_id(task_id)
        if task is not None:
            task.metadata = task.metadata or {}
            
            # ⭐ 根据实际可用的 trajectories 数量设置 n_offpolicy_trajectories
            if is_teacher:
                actual_count = len(self.exp_manager.teacher_task2trajectories.get(task_id, []))
            else:
                actual_count = len(self.exp_manager.task2trajectories.get(task_id, []))
            
            actual_offpolicy = min(self.offpolicy_trajectories_per_task, actual_count)
            
            if actual_count < self.offpolicy_trajectories_per_task:
                logger.warning(
                    f"Task {task_id} has only {actual_count} trajectories "
                    f"(requested {self.offpolicy_trajectories_per_task}). "
                    f"Will use {actual_offpolicy} off-policy + "
                    f"{self.n_rollout - actual_offpolicy} on-policy."
                )
            
            task.metadata["n_offpolicy_trajectories"] = actual_offpolicy
            task.metadata["is_teacher_task"] = is_teacher
            tasks.append(task)
    return tasks
```

### 3.4 获取 Teacher Off-policy Trajectories

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2036-2065`

```python
if teacher_exp_tasks:
    teacher_num_per_task = teacher_exp_config.get("max_trajectories_per_task", 1)
    teacher_offpolicy_trajectories = self.exp_manager.get_teacher_offpolicy_batch(
        tasks=teacher_exp_tasks,
        num_trajectories_per_task=teacher_num_per_task,
    )
    
    if teacher_offpolicy_trajectories:
        # 可选：使用当前 policy 的 entropy 选择最优 teacher 轨迹
        teacher_select_mode = teacher_exp_config.get("select_mode", "random")
        if teacher_select_mode == "entropy":
            teacher_offpolicy_trajectories = self._select_best_teacher_by_current_entropy(
                teacher_trajectories=teacher_offpolicy_trajectories,
                tasks=teacher_exp_tasks,
                num_trajectories_per_task=teacher_num_per_task,
            )
        
        # 转换为 CMT 对象
        teacher_offpolicy_cmt_array = self.env_manager.convert_offpolicy_to_cmt(
            offpolicy_trajectories=teacher_offpolicy_trajectories,
            config=self.config,
            tokenizer=self.tokenizer,
            task_id_to_data_id=task_id_to_data_id
        )
        
        # 标记为 teacher
        for cmt in teacher_offpolicy_cmt_array:
            cmt.metadata["is_teacher"] = True
```

### 3.5 合并所有轨迹

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2111-2133`

```python
# ExGRPO 风格：Task 级别混合
self.exp_manager.update_difficulty2task_dict(trajectories)

# 合并 on-policy、self-generated off-policy、teacher off-policy 轨迹
all_trajectories = trajectories.copy()

if offpolicy_cmt_array:
    all_trajectories.extend(offpolicy_cmt_array)
    logger.info(f"Added {len(offpolicy_cmt_array)} self-generated off-policy trajectories")

if teacher_offpolicy_cmt_array:
    all_trajectories.extend(teacher_offpolicy_cmt_array)
    logger.info(f"Added {len(teacher_offpolicy_cmt_array)} teacher off-policy trajectories")
```

---

## Loss 计算

### 4.1 数据准备

**位置**: `agentevolver/module/env_manager/env_manager.py:882-920`

```python
# ⭐ Experience Replay: 处理 recorded_old_log_probs
for sample in samples:
    is_teacher = sample.extras.get("is_teacher", False)
    has_log_prob = sample.extras.get("has_log_prob", False)
    response_length = len(sample.response_ids)
    
    if is_teacher and has_log_prob:
        # ⭐ Teacher Experience: 对齐 teacher_log_probs 到 response_loss_mask
        teacher_log_probs = sample.extras.get("teacher_log_probs")
        teacher_token_ids = sample.extras.get("teacher_token_ids")  # 用于精确对齐
        aligned_log_probs = self._align_teacher_log_probs(
            teacher_log_probs=teacher_log_probs,
            response_loss_mask=sample.response_loss_mask,
            response_length=response_length,
            response_ids=sample.response_ids,       # 用于精确对齐
            teacher_token_ids=teacher_token_ids,    # 用于精确对齐
        )
        recorded_old_log_probs_list.append(aligned_log_probs)
        teacher_mask_list.append(torch.ones(response_length, dtype=torch.int))
        
    elif is_teacher:
        # ⭐ Teacher Experience 但没有 log_prob：使用 LUFFY 模式
        recorded_old_log_probs_list.append(torch.zeros(response_length))
        teacher_mask_list.append(torch.ones(response_length, dtype=torch.int))
        
    else:
        # Self-generated Experience 或 on-policy
        old_log_probs = sample.extras.get("old_log_probs")
        # ... 处理 self-generated 的 old_log_probs ...
        teacher_mask_list.append(torch.zeros(response_length, dtype=torch.int))
```

**关键字段**:
- `recorded_old_log_probs`: 历史策略的 log_prob（teacher 或 self-generated）
- `teacher_mask`: 标记哪些样本是 teacher 轨迹（1=teacher, 0=非 teacher）
- `exp_mask`: 标记哪些样本是 off-policy（1=off-policy, 0=on-policy）
- `teacher_token_ids`: Teacher 生成的 token ids（用于精确对齐）

### 4.1.1 ⭐ Log Prob 对齐问题（重要）

当 `use_log_prob: true` 且使用同系列模型（如 Qwen-3B 学习 Qwen-72B）时，需要处理 **Chat Template 对齐问题**。

#### 问题根源

**采集时（vLLM）**：
- `log_probs` 只包含 **LLM 实际生成的 token**
- **不包含** chat template 添加的特殊标记

**训练时（tokenize_steps）**：
- `response_loss_mask=1` 包含了整个 assistant 消息
- 包括 chat template 的特殊标记（如 `<|im_start|>assistant\n`, `<|im_end|>`）

**Qwen 的 chat template**：
```
<|im_start|>assistant\n  ← 采集时不记录 log_prob（约 3-4 tokens）
[实际生成的内容]          ← 只有这部分有 log_prob
<|im_end|>               ← 采集时不记录 log_prob（1 token）
```

**结果**：训练时的 LLM positions 比采集时的 log_probs 多

#### 解决方案

`_align_teacher_log_probs` 实现了两种对齐策略：

**模式 1：精确对齐（推荐）**

**位置**: `agentevolver/module/env_manager/env_manager.py:469-493`

使用采集时保存的 `token_ids` 进行匹配：

```python
# ⭐ 模式 1：精确对齐（使用 token_ids）
if response_ids is not None and teacher_token_ids is not None:
    filled_count = self._align_by_token_ids(
        aligned_log_probs=aligned_log_probs,
        teacher_log_probs=teacher_log_probs,
        teacher_token_ids=teacher_token_ids,
        response_ids=response_ids,
        llm_positions=llm_positions,
    )
```

**模式 2：简单对齐（fallback）**

**位置**: `agentevolver/module/env_manager/env_manager.py:495-521`

从后往前填充（末尾对齐），开头的 chat template 标记位置自动填充 0：

```python
# 从后往前填充（末尾对齐）
for i in range(num_to_fill):
    teacher_idx = len(teacher_log_probs) - 1 - i
    llm_idx = len(llm_positions) - 1 - i
    pos = llm_positions[llm_idx]
    aligned_log_probs[pos] = teacher_log_probs[teacher_idx]
```

#### 日志说明

训练时可能看到如下日志：

**简单对齐模式**：
```
[Teacher Log Prob Alignment] Fewer teacher log_probs (159) than LLM positions (204). 
45 leading positions (likely chat template tokens) will be 0.
```

**精确对齐模式**：
```
[Teacher Log Prob Alignment] Token-based alignment: 159/159 tokens matched (100.0%).
```

这些是正常的，表示对齐逻辑正在正确处理 chat template 标记。

#### 确保精确对齐

要启用精确对齐，确保 teacher 轨迹数据包含 `log_probs_per_turn`：

```json
{
  "log_probs": [-0.5, -0.3, ...],
  "log_probs_per_turn": [
    {"turn_idx": 0, "log_probs": [-0.5, -0.3], "token_ids": [100, 200]},
    {"turn_idx": 1, "log_probs": [-0.8, -0.2], "token_ids": [300, 400]}
  ]
}
```

使用 `scripts/collect_teacher_trajectories.py` 采集的数据会自动包含这些字段。

### 4.2 替换 Off-policy 的 old_log_prob

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2224-2233`

```python
# ⭐ Experience Replay: 替换 off-policy 数据的 old_log_prob
if enable_exp_replay and "recorded_old_log_probs" in batch.batch:
    exp_is_correct = exp_replay_config.get("exp_is_correct", True)
    if exp_is_correct:
        batch = self._replace_recorded_old_log_probs(
            batch=batch,
            current_old_log_prob=old_log_prob,
            entropys=entropys,
        )
```

### 4.3 het_compute_teacher_aware_loss

**位置**: `agentevolver/module/exp_manager/het_core_algos.py:156-335`

这是 Teacher Experience Replay 的核心 loss 计算函数：

```python
def het_compute_teacher_aware_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    teacher_mask,  # ⭐ 标记 Teacher 轨迹
    # ... clipping 参数 ...
    # ⭐ Teacher 专用配置
    teacher_use_log_prob: bool = False,
    teacher_policy_shaping_enable: bool = True,
    teacher_policy_shaping_mode: str = "p_div_p_beta",
    teacher_policy_shaping_beta: float = 0.1,
    teacher_use_clip: bool = False,
):
    """
    计算混合 on-policy、self-generated off-policy 和 teacher off-policy 的 loss。
    
    Teacher 轨迹的两种处理模式：
    1. teacher_use_log_prob=True: ExGRPO 形式，ratio = π_current / π_teacher
    2. teacher_use_log_prob=False: LUFFY 形式，ratio = π_current（分母=1）
    """
```

### 4.4 两种 Teacher Loss 计算模式

**模式 1: 使用 log_prob（ExGRPO 形式）**

当 `teacher_use_log_prob=True` 时（适用于 Qwen 等同 tokenizer 的 teacher）：

```python
if teacher_use_log_prob:
    # Teacher 轨迹有 log_prob，使用标准重要性采样
    # ratio = π_current / π_teacher = exp(log_prob - old_log_prob)
    teacher_ratio = ratio  # 直接使用标准 ratio
```

**数学公式**:
```
ratio = exp(log_prob_current - log_prob_teacher)
     = π_current(a|s) / π_teacher(a|s)
```

**模式 2: 无 log_prob（LUFFY 形式）**

当 `teacher_use_log_prob=False` 时（适用于 GPT-4 等不同 tokenizer 的 teacher）：

```python
else:
    # Teacher 轨迹无 log_prob，使用简化计算
    # 假设 π_teacher = 1，ratio = π_current / 1 = π_current
    teacher_ratio = torch.exp(log_prob)
```

**数学公式**:
```
ratio = π_current(a|s) / 1 = π_current(a|s)
```

### 4.5 Policy Shaping

**位置**: `agentevolver/module/exp_manager/het_core_algos.py:275-291`

LUFFY 推荐使用 Policy Shaping 来放大低概率动作的梯度信号：

```python
# Policy shaping（LUFFY 推荐使用）
if teacher_policy_shaping_enable:
    teacher_ratio = _apply_policy_shaping(
        teacher_ratio,
        mode=teacher_policy_shaping_mode,
        beta=teacher_policy_shaping_beta,
    )

def _apply_policy_shaping(ratio, mode, beta=0.1):
    if mode == "p_div_p_beta":
        # LUFFY 的 policy shaping: f(x) = x / (x + β)
        # 放大低概率信号，抑制高概率信号
        return ratio / (ratio + beta)
    elif mode == "sqrt":
        return torch.sqrt(ratio)
    else:
        return ratio
```

**Policy Shaping 的作用**:
- 原始 ratio 接近 0 的 token（当前 policy 认为概率很低的 teacher 动作）
- 经过 `f(x) = x/(x+β)` 变换后，梯度信号被放大
- 有助于学习 teacher 的"非典型"但成功的动作

### 4.6 Loss 混合

**位置**: `agentevolver/module/exp_manager/het_core_algos.py:292-320`

```python
# Teacher loss 计算
teacher_off_pg_losses_raw = -advantages * teacher_ratio

if teacher_use_clip:
    teacher_off_pg_losses2 = -advantages * torch.clamp(teacher_ratio, 1 - cliprange_low, 1 + off_cliprange_high)
    teacher_off_pg_losses = torch.maximum(teacher_off_pg_losses_raw, teacher_off_pg_losses2)
else:
    teacher_off_pg_losses = teacher_off_pg_losses_raw  # LUFFY 建议不 clip

# =============== 混合 off-policy loss ===============
# 根据 teacher_mask 选择使用哪种 off-policy loss
off_pg_losses = torch.where(
    teacher_mask_float.bool(),
    teacher_off_pg_losses,      # teacher 轨迹
    self_off_pg_losses          # self-generated 轨迹
)

# =============== 合并 loss ===============
pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
```

---

## 关键数据结构

### 5.1 DataProto.batch

```python
{
    "input_ids": torch.Tensor,           # [batch_size, seq_len]
    "attention_mask": torch.Tensor,      # [batch_size, seq_len]
    "position_ids": torch.Tensor,        # [batch_size, seq_len]
    "loss_mask": torch.Tensor,           # [batch_size, seq_len]
    "response_mask": torch.Tensor,       # [batch_size, response_len]
    "group_ids": torch.Tensor,           # [batch_size] - 基于 data_id
    
    # ⭐ Experience Replay 相关
    "exp_mask": torch.Tensor,            # [batch_size, seq_len] - 1=off-policy, 0=on-policy
    "recorded_old_log_probs": torch.Tensor,  # [batch_size, response_len]
    
    # ⭐ Teacher Experience 相关
    "teacher_mask": torch.Tensor,        # [batch_size, response_len] - 1=teacher, 0=非 teacher
    
    # Loss 计算相关
    "old_log_probs": torch.Tensor,       # [batch_size, response_len]
    "log_probs": torch.Tensor,           # [batch_size, response_len]
    "advantages": torch.Tensor,          # [batch_size, response_len]
    "token_level_rewards": torch.Tensor, # [batch_size, response_len]
}
```

### 5.2 ExperienceManager 数据结构

```python
{
    # Self-generated experience (ExGRPO)
    "difficulty2task_dict": Dict[int, List[str]],      # 难度 -> task_id 列表
    "task2trajectories": Dict[str, List[Trajectory]],  # task_id -> 轨迹列表
    "skip_uid_set": Set[str],                          # 已完全解决的 task_id
    
    # ⭐ Teacher experience (LUFFY)
    "teacher_task2trajectories": Dict[str, List[Trajectory]],  # task_id -> teacher 轨迹列表
}
```

### 5.3 Trajectory.metadata

```python
{
    # ⭐ Teacher 特有字段
    "is_teacher": True,                    # 标记为 teacher 轨迹
    "is_experience_replay": True,          # 标记为 off-policy
    "teacher_model": "gpt-4",              # teacher 模型名称
    
    # Log prob 相关（可选）
    "has_log_prob": True,                  # 是否有 log_prob
    "log_probs": [float, ...],             # 累积的 log_probs
    "log_probs_per_turn": [[float], ...],  # 分轮的 log_probs
}
```

---

## 配置项详解

### 6.1 完整配置示例

```yaml
exp_manager:
  # ==================== Teacher Experience (LUFFY) ====================
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_gpt4.jsonl"
    
    # ⭐ 混合模式
    # - "rollout_level": LUFFY 风格，每个 task 内混合 on-policy 和 teacher rollouts
    # - "task_level": ExGRPO 风格，batch 中某些 task 完全是 teacher experience
    mix_mode: "rollout_level"
    
    # LUFFY 风格配置
    n_teacher_rollouts_per_task: 1    # 每个 task 的 n 个 rollouts 中有多少是 teacher
    
    # Task 级别配置（当 mix_mode="task_level" 时使用）
    exp_ratio: 0.2                     # Teacher experience tasks 的比例
    
    # 轨迹选择
    max_trajectories_per_task: 3       # 每个 task 最多存储的 teacher 轨迹数
    select_mode: "random"              # 选择模式：random, entropy, confidence
    
    # Log prob 配置
    use_log_prob: false                # Teacher 是否有 log_prob
                                       # - true: Qwen 等同 tokenizer 的 teacher
                                       # - false: GPT-4 等不同 tokenizer 的 teacher
    
    # Policy shaping 配置（LUFFY 推荐）
    policy_shaping:
      enable: true
      mode: "p_div_p_beta"             # f(x) = x / (x + β)
      beta: 0.1
  
  # ==================== Self-generated Experience (ExGRPO) ====================
  experience_replay:
    enable: true                       # 是否启用 self-generated experience
    exp_ratio: 0.5                     # 总 experience ratio（包含 teacher）
    replay_start_ratio: 0.35           # 训练进度达到 35% 时开始 replay
    offpolicy_trajectories_per_task: 1
    use_current_policy_entropy: true   # 使用当前 policy 计算 entropy 选择轨迹
    exp_is_correct: true               # 使用 recorded_old_log_probs

# Actor 配置
actor_rollout_ref:
  actor:
    # ⭐ Teacher Experience Loss 配置
    teacher_use_log_prob: false        # 是否使用 teacher 的 log_prob
    teacher_policy_shaping_enable: true
    teacher_policy_shaping_mode: "p_div_p_beta"
    teacher_policy_shaping_beta: 0.1
    teacher_use_clip: false            # LUFFY 建议不使用 clipping
    
  rollout:
    n: 8                               # 每个 task 的 rollout 数量
```

### 6.2 只使用 Teacher Experience 的配置

```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_gpt4.jsonl"
    mix_mode: "rollout_level"
    n_teacher_rollouts_per_task: 1
    use_log_prob: false
    policy_shaping:
      enable: true
      mode: "p_div_p_beta"
      beta: 0.1
  
  # ⭐ 禁用 self-generated experience
  experience_replay:
    enable: false
```

### 6.3 Teacher + Self-generated 混合配置

```yaml
exp_manager:
  teacher_experience:
    enable: true
    data_path: "data/teacher_trajectories/alfworld_gpt4.jsonl"
    mix_mode: "task_level"             # 使用 Task 级别混合
    exp_ratio: 0.2                     # 20% teacher tasks
    use_log_prob: false
  
  experience_replay:
    enable: true
    exp_ratio: 0.5                     # 总 50% experience (20% teacher + 30% self)
    replay_start_ratio: 0.35
```

---

## 常见问题排查

### 7.1 Teacher 轨迹没有被使用

**症状**: `luffy/total_teacher_rollouts` 始终为 0

**检查步骤**:
1. 确认配置 `teacher_experience.enable: true`
2. 检查 `data_path` 是否正确，文件是否存在
3. 检查 JSONL 中的 `task_id` 是否与训练任务匹配
4. 查看日志中 `Loaded X teacher trajectories` 的数量

### 7.2 Teacher Loss 为 NaN

**症状**: `teacher_off_pg_loss` 为 NaN

**可能原因**:
1. `teacher_mask` 全为 0，导致 masked_mean 计算 NaN
2. `recorded_old_log_probs` 对齐问题

**解决**:
- 检查 `teacher_mask.sum()` 是否大于 0
- 检查 `recorded_old_log_probs` 的值是否合理

### 7.3 重要性采样权重过大

**症状**: `ratio` 值非常大，训练不稳定

**可能原因**:
1. Teacher 使用不同 tokenizer，但 `teacher_use_log_prob=true`
2. Policy 与 teacher 差异太大

**解决**:
- 如果 teacher 使用不同 tokenizer（如 GPT-4），设置 `teacher_use_log_prob: false`
- 启用 policy shaping：`teacher_policy_shaping_enable: true`

### 7.4 Teacher Trajectory 不足

**症状**: 部分 task 没有足够的 teacher rollouts

**LUFFY 模式**:
- 自动保留更多 on-policy rollouts 补足
- 查看日志中的 `tasks_with_partial_teacher` 和 `tasks_without_teacher`

**Task 级别模式**:
- 自动调整 `n_offpolicy_trajectories`
- 查看 warning 日志中的具体数量

### 7.5 GRPO 分组错误

**症状**: Advantage 计算不准确

**检查步骤**:
1. 确认 `uid` 基于 `group_ids` 设置
2. 确认同一 task 的 on-policy 和 teacher rollouts 共享相同的 `data_id`

### 7.6 Log prob 对齐问题（Chat Template）

**症状**: 日志显示 `Fewer teacher log_probs (X) than LLM positions (Y)`

**原因**: Chat template 在训练时的 tokenization 包含了特殊标记（如 `<|im_start|>assistant\n`, `<|im_end|>`），但采集时的 log_probs 只包含 LLM 实际生成的 token。

**这是正常的！** 差值通常约为 `4-5 tokens × 对话轮数`。

**解决方案**:

1. **推荐**：确保 teacher 轨迹包含 `log_probs_per_turn` 和 `token_ids`，启用精确对齐：
   ```json
   {
     "log_probs_per_turn": [
       {"turn_idx": 0, "log_probs": [...], "token_ids": [100, 200, ...]},
       ...
     ]
   }
   ```

2. **备选**：如果没有 `token_ids`，系统会使用简单对齐（从后往前填充）

**验证对齐效果**:
```
# 精确对齐成功
[Teacher Log Prob Alignment] Token-based alignment: 159/159 tokens matched (100.0%).

# 简单对齐
[Teacher Log Prob Alignment] Fewer teacher log_probs (159) than LLM positions (204). 
45 leading positions (likely chat template tokens) will be 0.
```

### 7.7 Tokenizer 不匹配

**症状**: `ratio` 值异常大或为 NaN

**原因**: Teacher 模型和 Student 模型使用不同的 tokenizer

**解决**:
- 设置 `teacher_experience.use_log_prob: false`
- 使用 LUFFY 简化形式（假设 π_old = 1）
- 启用 policy shaping 提升信号：`teacher_policy_shaping_enable: true`

---

## 快速测试

### 验证 Teacher 轨迹加载

```python
from omegaconf import OmegaConf
from agentevolver.module.exp_manager.exp_manager import ExperienceManager

config = OmegaConf.load("config/alfworld_grpo_3b_teacher_only.yaml")
exp_manager = ExperienceManager(config.exp_manager)

print(f"Teacher enabled: {exp_manager.teacher_enabled}")
print(f"Teacher tasks: {len(exp_manager.teacher_task2trajectories)}")
print(f"Total teacher trajectories: {sum(len(v) for v in exp_manager.teacher_task2trajectories.values())}")

# 检查一个 task 的 teacher 轨迹
for task_id, trajs in list(exp_manager.teacher_task2trajectories.items())[:1]:
    print(f"\nTask {task_id}:")
    for traj in trajs:
        print(f"  - has_log_prob: {traj.metadata.get('has_log_prob', False)}")
        print(f"  - reward: {traj.reward.outcome if traj.reward else 'N/A'}")
```

### 验证 LUFFY Mixer

```python
from agentevolver.module.exp_manager.experience_collate import LUFFYTeacherRolloutMixer

mixer = LUFFYTeacherRolloutMixer(
    exp_manager=exp_manager,
    n_teacher_rollouts_per_task=1,
    n_rollout=8,
)

print(f"n_onpolicy_per_task: {mixer.get_n_onpolicy_rollouts_per_task()}")
```

---

## 总结

Teacher Experience Replay 的完整流程：

1. **初始化**: 加载 teacher 轨迹到 `teacher_task2trajectories`
2. **混合模式选择**:
   - LUFFY: Rollout 级别混合，每个 task 内混入 teacher
   - ExGRPO: Task 级别混合，部分 task 完全是 teacher
3. **Rollout**: 生成 on-policy trajectories
4. **混合**: 用 teacher 替换部分 on-policy（LUFFY）或追加（ExGRPO）
5. **Loss 计算**:
   - 使用 `teacher_mask` 区分 teacher 和 self-generated
   - `teacher_use_log_prob=True`: ExGRPO 形式
   - `teacher_use_log_prob=False`: LUFFY 形式 + Policy Shaping

关键设计：
- **LUFFY Rollout-level**: Teacher 影响同 task 内所有 rollouts 的 GRPO baseline
- **自动补足**: Teacher 不足时自动保留更多 on-policy
- **Policy Shaping**: 放大低概率 teacher 动作的梯度信号

