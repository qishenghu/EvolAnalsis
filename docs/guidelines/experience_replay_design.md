# Experience Replay 机制设计方案

## 1. 概述

### 1.1 背景
现有的 experience-guided 机制将经验作为 in-context example 放入 prompt 中，这是一种轻量级的经验利用方式。而 Experience Replay 是一种更直接的方式，将 off-policy trajectory 作为训练数据与 on-policy rollout 混合，通过重要性采样（Importance Sampling）来利用历史经验。

**详细说明**：
- **Experience-Guided**：通过检索相似经验片段，将其作为 in-context example 插入到 prompt 中，模型在生成时会参考这些经验，但不会直接在这些经验上计算 loss。这种方式轻量级，但影响是间接的。
- **Experience Replay**：直接获取历史任务的完整轨迹（off-policy trajectory），将其作为训练数据与当前生成的 on-policy 数据混合，在同一个 batch 中训练。通过重要性采样校正 off-policy 数据的权重，使其能够有效利用历史经验。这种方式更直接，但需要处理策略差异（通过重要性采样）。
- **为什么需要 Experience Replay**：在强化学习中，agent 需要从历史经验中学习。传统的 on-policy 方法只能使用当前策略生成的数据，效率较低。Experience Replay 允许 agent 重复利用历史成功经验，提高样本效率。

### 1.2 核心思想
- **Off-policy Trajectory**: 从 ExperienceManager 的内存存储中获取历史任务的完整轨迹
- **内存管理**: 在 ExperienceManager 中维护 `difficulty2task_dict` 和 `task2trajectories`，直接管理 replay 任务和轨迹
- **Off-policy Loss 计算**: Off-policy 数据参与 loss 计算，通过 `exp_mask` 区分 on-policy 和 off-policy，使用不同的 cliprange
- **重要性采样**: 使用 `off_ratio = exp(log_prob_current - old_log_prob_historical)` 来校正 off-policy 数据的权重
- **混合训练**: On-policy 和 off-policy 数据在同一个 batch 中混合训练

**详细说明**：
- **Off-policy Trajectory**：历史任务在之前某个策略版本下生成的完整轨迹，包含 messages（对话历史）、response（历史响应）、reward（奖励）和 old_log_probs（历史策略的 log 概率）。这些轨迹存储在 ExperienceManager 的内存中（`task2trajectories`），可以通过 task_id 检索。
- **Off-policy Loss 计算**：Off-policy trajectory 的 LLM 消息保持 `author="llm"`，确保 `loss_mask=1`，参与 off-policy loss 计算。通过 `exp_mask=1` 标记 off-policy 数据，在 `het_compute_token_on_off_policy_loss` 中使用不同的 cliprange 计算 off-policy loss。
- **重要性采样（Importance Sampling）**：由于 off-policy 数据来自历史策略，而当前策略可能已经改变，需要校正权重。重要性采样权重 `ratio = π_current / π_historical = exp(log_prob_current - old_log_prob_historical)` 用于校正 off-policy loss，确保训练稳定。
- **混合训练**：On-policy 和 off-policy 数据在同一个 batch 中混合，通过 `exp_mask` 区分。On-policy 数据使用标准 PPO loss，off-policy 数据使用带重要性采样的 PPO loss。这样可以同时利用新生成的数据和历史经验。

**Off-Policy 数据的处理方式**

参考 ExGRPO 的设计，Off-policy 数据参与 loss 计算，通过 `exp_mask` 区分 on-policy 和 off-policy：

1. **统一训练流程**：
   - On-policy 和 off-policy 使用相同的训练路径
   - 在同一个 batch 中混合处理，无需额外的数据管道
   - 通过 `exp_mask` 区分数据类型，应用不同的 loss 计算方式

2. **Off-policy Loss 计算**：
   - Off-policy 数据的 LLM 消息保持 `author="llm"`，确保 `loss_mask=1`
   - 使用 `exp_mask=1` 标记 off-policy 数据
   - 在 `het_compute_token_on_off_policy_loss` 中使用不同的 cliprange 计算 off-policy loss

3. **重要性采样**：
   - Off-policy 数据使用 `recorded_old_log_probs`（历史策略的 log_prob）
   - 计算重要性采样权重：`ratio = exp(log_prob_current - old_log_prob_historical)`
   - 使用 `off_cliprange_high` 稳定训练

**与 ExGRPO 的对比**：

| 特性 | ExGRPO 的设计 | 本设计 |
|------|---------------------|---------------------|
| **场景** | Single-turn（单轮对话） | Multi-turn（多轮对话） |
| **Mask 机制** | `prefix_mask` 标记 prefix 位置 | `exp_mask` 标记 off-policy token |
| **Loss 计算** | `on_pg_loss` + `off_pg_loss` 分别计算 | `het_compute_token_on_off_policy_loss` 分别计算 |
| **LLM 消息 author** | N/A（单轮） | 保持 `author="llm"`，确保 `loss_mask=1` |
| **重要性采样** | 使用 `recorded_old_log_probs` | 使用 `recorded_old_log_probs` |

**关键设计决策**：
1. **保持 `author="llm"`**：确保 off-policy 数据的 LLM 消息的 `loss_mask=1`，参与 loss 计算
2. **使用 `exp_mask` 区分**：不使用 `loss_mask=0` 来区分 off-policy，而是使用独立的 `exp_mask`
3. **兼容 multi-turn**：在多轮对话场景中，每轮 LLM 响应都参与 off-policy loss 计算

### 1.3 与现有机制的区别

| 特性 | Experience-Guided (现有) | Experience Replay (新) |
|------|-------------------------|----------------------|
| 经验使用方式 | 作为 in-context example | 作为训练数据 |
| 数据来源 | 检索相似经验片段 | 获取完整 off-policy trajectory |
| 模型行为 | 生成新响应 | 计算历史响应的 log_prob |
| 训练方式 | 间接影响（通过 prompt） | 直接训练（通过 loss） |
| old_log_prob | 使用当前策略计算 | 使用历史策略（记录） |

## 2. 整体架构

### 2.1 数据流

```
训练循环开始
    ↓
[0] 动态选择 Tasks（如果 training_progress >= replay_start_ratio）
    ├─ 从 training tasks 中选择 (batch_size - replay_task_count) 个
    ├─ 从 replaytaskpool 中选择 replay_task_count 个
    └─ 合并组成新的 training batch
    ↓
[1] 获取 Off-policy Trajectory（仅对 replay tasks）
    ├─ 从 ExperienceManager.task2trajectories 获取历史轨迹
    ├─ 转换为 Trajectory 对象（Linear_CMT 格式）
    ├─ 设置 metadata["is_experience_replay"] = True
    ├─ 保存 old_log_probs 到 metadata
    └─ 设置合适的 data_id（确保 GRPO 分组正确）
    ↓
[2] 生成 On-policy Rollout
    ├─ 对于 replay tasks：调整 rollout_n = n_rollout - offpolicy_trajectories_per_task
    │   └─ 例如：n_rollout=8, offpolicy=2 → on-policy rollout_n=6
    ├─ 对于非 replay tasks：使用原始的 rollout_n = n_rollout
    │   └─ 例如：n_rollout=8 → on-policy rollout_n=8
    ├─ 执行环境交互（EnvWorker.execute → AgentFlow.execute）
    ├─ 生成新轨迹（Trajectory/Linear_CMT）
    └─ 标记 metadata["is_experience_replay"] = False
    ↓
[2.5] 更新 difficulty2task_dict
    ├─ 按 task_id 分组统计 rollout 结果
    ├─ 计算每个 task 的 difficulty（reward=1 的次数）
    └─ 更新 difficulty2task_dict[difficulty].append(task_id)
    ↓
[3] 混合 Trajectory 列表
    ├─ 合并 on-policy 和 off-policy trajectory
    └─ 统一转换为 DataProto（to_dataproto）
        ├─ trajectories_to_samples: group_tokenize() → Sample[]
        ├─ samples_to_dataproto: padding + batching
        └─ 创建 exp_mask（基于 Sample.extras["is_experience_replay"]）
    ↓
[4] 计算 old_log_prob
    ├─ 对所有数据计算当前策略的 old_log_prob
    ├─ 使用 exp_mask 区分 on/off-policy
    └─ 替换 off-policy 的 old_log_prob 为 recorded_old_log_probs
    ↓
[5] 计算 Reward 和 Advantage
    ├─ 计算 token_level_rewards
    ├─ 计算 advantages（GRPO/GAE）
    └─ GRPO 基于 group_ids（data_id）分组，off-policy 使用独立分组
    ↓
[6] 计算 Loss
    ├─ 使用 exp_mask 区分 on/off-policy
    ├─ 重要性采样权重: off_ratio = exp(log_prob - old_log_prob)
    ├─ On-policy loss: 标准 PPO loss（exp_mask=0）
    └─ Off-policy loss: 带重要性采样的 PPO loss（exp_mask=1）
    ↓
[7] 反向传播更新模型
    ↓
[8] 保存 old_log_prob（可选，只保存 reward=1 的轨迹）
    └─ 将成功轨迹的 old_log_prob 保存到 ExperienceManager.task2trajectories
```

**数据流详细说明**：

**[0] 准备 Training Tasks**：
- **目的**：从 TaskManager 获取 training tasks，并过滤掉已经全部做对的 tasks
- **逻辑**：
  1. 从 TaskManager 获取 batch_size 个 training tasks
  2. **⭐ 过滤 skip_uid_set**：排除已经在 `skip_uid_set` 中的 tasks（这些 tasks 的 n_rollout 全部做对，不再需要训练）
  3. 如果过滤后数量不足，从原始 tasks 中补充
- **关键点**：
  - `skip_uid_set` 是一个集合，存储已经全部做对的 task_id
  - 如果某个 task 的 n_rollout 全部 reward=1，说明该 task 已经完全掌握，不再需要训练
  - 这样可以避免浪费计算资源在已经掌握的任务上

**[1] 获取 Off-policy Trajectory**：
- **目的**：为 replay tasks 获取历史轨迹，用于后续的 off-policy 训练
- **时机**：在生成 on-policy rollout 之前，因为需要知道哪些 tasks 是 replay tasks
- **数据来源**：从 ExperienceManager.task2trajectories 获取
- **关键点**：
  - 只对 replay tasks 获取 off-policy trajectory
  - 每个 replay task 可以获取多个历史轨迹（通过 `offpolicy_trajectories_per_task` 配置）
  - Off-policy trajectory 必须包含 `old_log_probs`（历史策略的 log 概率），用于重要性采样

**[2] 生成 On-policy Rollout**：
- **目的**：对所有 tasks（包括 replay tasks）生成新的 on-policy 轨迹
- **关键设计：混合数量控制**：
  - **对于 replay tasks**：
    - 如果 `n_rollout=8`，`offpolicy_trajectories_per_task=2`
    - 则 on-policy rollout 数量 = `8 - 2 = 6`
    - 这样确保每个 replay task 总共有 8 条轨迹（6 条 on-policy + 2 条 off-policy）
  - **对于非 replay tasks**：
    - 使用原始的 `rollout_n = n_rollout`（例如 8）
    - 只生成 on-policy 轨迹，总数为 8
- **实现方式**：
  - 在调用 `env_manager.rollout()` 之前，为每个 task 计算其 on-policy rollout 数量
  - 对于 replay tasks：`on_policy_rollout_n = n_rollout - offpolicy_trajectories_per_task`
  - 对于非 replay tasks：`on_policy_rollout_n = n_rollout`
  - 可以通过修改 `task_exp_configs` 或传递不同的 `rollout_n` 参数来实现
- **原因**：
  1. **更新 difficulty**：replay tasks 也需要生成新的 on-policy rollout 来更新其 difficulty
  2. **混合训练**：replay tasks 的 on-policy 轨迹也会参与训练，与 off-policy 轨迹混合
  3. **保存经验**：成功轨迹会被保存到内存，供后续使用
  4. **保持总数一致**：确保每个 task 的总轨迹数（on-policy + off-policy）等于 `n_rollout`，便于 GRPO 分组计算
- **关键点**：
  - 所有 on-policy 轨迹（无论来自 training tasks 还是 replay tasks）都标记为 `is_experience_replay = False`
  - 每个 replay task 的总轨迹数 = on-policy 数量 + off-policy 数量 = `n_rollout`

**[2.5] 更新 difficulty2task_dict**：
- **目的**：根据当前 step 的 rollout 结果，更新任务难度分桶
- **计算方式**：对每个 task，统计其在该 step 中 reward=1 的轨迹数量，作为 difficulty
- **更新规则**：如果 task 不在对应难度的桶中，则添加（允许同一个 task 在不同难度桶中存在）
- **关键点**：只统计 on-policy rollout 的结果，不包括 off-policy trajectory

**[3] 混合 Trajectory 列表**：
- **目的**：将 on-policy 和 off-policy 轨迹合并，统一转换为 DataProto 格式
- **合并方式**：`all_trajectories = on_policy_trajectories + off_policy_trajectories`
- **转换流程**：
  1. `trajectories_to_samples`：调用每个 trajectory 的 `group_tokenize()` 方法，将轨迹转换为 Sample 列表
  2. `samples_to_dataproto`：将 Sample 列表进行 padding 和 batching，转换为 DataProto
  3. 在转换过程中，根据 `Sample.extras["is_experience_replay"]` 创建 `exp_mask`
- **关键点**：
  - Off-policy trajectory 的 `is_experience_replay = True`，会在 `exp_mask` 中标记
  - On-policy trajectory 的 `is_experience_replay = False`，`exp_mask` 中不标记（除非使用了 experience-guided）

**[4] 计算 old_log_prob**：
- **目的**：计算当前策略对所有数据的 log 概率，用于 PPO loss 计算和重要性采样
- **处理逻辑**（参考 ExGRPO 的三步流程）：
  1. **步骤 1：收集经验时保存 old_log_prob**
     - 在生成轨迹后，对于成功轨迹（reward=1），保存其 `old_log_prob`（来自当时的 policy）
     - 保存位置：`trajectory.metadata["old_log_probs"]`，同时保存 `policy_version`（global_steps）
     - 这些 `old_log_prob` 是生成经验时的旧 policy 的 log 概率
  2. **步骤 2：训练时计算当前 policy 的 log_prob**
     - 使用当前 policy model 计算所有样本（包括 on-policy 和 off-policy）的 `log_prob`
     - 这是当前策略对所有数据的 log 概率，用于计算重要性采样权重
  3. **步骤 3：替换 Off-Policy 样本的 old_log_prob**
     - **log_prob**：由当前 policy model 计算（所有样本，包括 on-policy 和 off-policy）
     - **old_log_prob**：
       - **On-policy**：由当前 policy model 计算（与 log_prob 相同，ratio ≈ 1.0）
       - **Off-policy**：使用 `recorded_old_log_prob`（收集经验时保存的旧 policy 的 log_prob）
     - 通过 `exp_mask` 区分，只替换 off-policy 数据的 old_log_prob
- **为什么这样设计**：
  - **重要性采样需要 π_old 和 π_new 的 log_prob**：
    - `recorded_old_log_prob` 是生成经验时的旧 policy 的 log_prob（π_old）
    - `log_prob` 是当前 policy 的 log_prob（π_new）
    - 两者结合可计算重要性采样权重：`ratio = exp(log_prob - old_log_prob) = π_new / π_old`
  - **统一计算流程**：
    - 先对所有数据计算当前策略的 log_prob，确保计算方式一致
    - 然后只替换 off-policy 数据的 old_log_prob，保持 on-policy 数据不变
    - 这样设计简化了实现，避免了重复计算
- **关键点**：
  - Off-policy 数据使用历史策略的 old_log_prob，这是重要性采样的基础
  - On-policy 数据使用当前策略的 old_log_prob，这是标准 PPO 的做法
  - 通过 `exp_mask` 区分，确保不同类型的数据使用正确的 old_log_prob

**[5] 计算 Reward 和 Advantage**：
- **目的**：计算每个 token 的奖励和优势值，用于策略优化
- **Reward 计算**：根据环境反馈或 reward model 计算 `token_level_rewards`
- **Advantage 计算**：
  - GRPO：基于 `group_ids`（data_id）分组计算 advantage
  - Off-policy 数据使用独立的 `data_id`（整数格式，例如：`1000000 + task_id_int * 1000 + index`），确保单独分组
- **关键点**：GRPO 的分组确保 off-policy 数据的 advantage 计算不会影响 on-policy 数据

**[6] 计算 Loss**：
- **目的**：计算策略梯度损失，用于模型更新
- **重要性采样**：
  - 对于 off-policy 数据：`ratio = exp(log_prob_current - old_log_prob_historical)`
  - 对于 on-policy 数据：`ratio = exp(log_prob_current - old_log_prob_current) = 1.0`
- **Loss 计算**：
  - On-policy：标准 PPO loss，使用 `cliprange_low` 和 `cliprange_high`
  - Off-policy：带重要性采样的 PPO loss，使用 `cliprange_low` 和 `off_cliprange_high`（通常更小，如 1.0）
- **关键点**：通过 `exp_mask` 区分 on/off-policy，应用不同的 loss 计算方式

**[7] 反向传播更新模型**：
- **目的**：根据计算的 loss 更新模型参数
- **处理**：标准的反向传播和优化器更新

**[8] 保存 old_log_prob**：
- **目的**：将成功轨迹的 old_log_prob 保存到 ExperienceManager 的内存存储中，供后续作为 off-policy 数据使用
- **保存位置**：`ExperienceManager.task2trajectories`（内存字典，key 为 task_id，value 为 Trajectory 列表）
- **保存条件**：
  1. 只保存 `reward == 1.0` 的成功轨迹
  2. 需要配置 `experience_replay.enable = true`
- **保存内容**：完整的 Trajectory 对象，包括 messages、reward、old_log_probs、policy_version 等
- **内存管理**：使用 `max_trajectories_per_task` 限制每个 task 的轨迹数量，超过时使用 FIFO 策略删除最旧的轨迹
- **关键点**：保存的 old_log_prob 是当前策略计算的，在后续训练中会作为历史策略的 old_log_prob 使用

### 2.2 关键组件

1. **ExperienceManager**: 
   - 维护 `difficulty2task_dict` (replaytaskpool) 按难度分桶存储 task_id
   - 维护 `task2trajectories` 按 task_id 存储 Trajectory 列表（包含 old_log_probs）
   - 根据 rollout 结果动态更新 difficulty 分桶
   - 提供从内存中获取 off-policy trajectory 的方法
2. **AgentEvolverRayPPOTrainer**: 
   - 在训练循环中集成 off-policy 数据
   - 从 replaytaskpool 动态选择 tasks 组成训练 batch
   - 将成功轨迹保存到 ExperienceManager 的内存存储中
3. **Loss 计算**: 修改 loss 计算函数以支持重要性采样

## 3. 详细设计

### 3.1 ExperienceManager 扩展

**位置**: `agentevolver/module/exp_manager/exp_manager.py`

**作用**：ExperienceManager 负责管理经验相关的所有操作，包括 experience-guided 和 experience replay。扩展后需要维护 `difficulty2task_dict`（replaytaskpool）和 `task2trajectories`（轨迹存储），并根据 rollout 结果动态更新。所有 replay 相关的数据都存储在内存中，不依赖 ReMe。

**新增属性和方法**:

#### 3.1.1 初始化 difficulty2task_dict、task2trajectories 和 skip_uid_set

```python
def __init__(self, config: DictConfig):
    # ... 现有初始化代码 ...
    
    # Experience Replay 相关
    self.difficulty2task_dict: Dict[int, List[str]] = defaultdict(list)  # ⭐ 按难度分桶存储 task_id
    self.task2trajectories: Dict[str, List[Trajectory]] = defaultdict(list)  # ⭐ 按 task_id 存储 Trajectory 列表
    self.skip_uid_set: Set[str] = set()  # ⭐ 存储已经全部做对的 task_id，不再参与 replay
    self.replay_start_ratio = self.exp_manager_config.get("experience_replay", {}).get("replay_start_ratio", 0.35)
    self.max_trajectories_per_task = self.exp_manager_config.get("experience_replay", {}).get("max_trajectories_per_task", 10)  # ⭐ 每个 task 最多保存的轨迹数量
    self.experience_lbound = self.exp_manager_config.get("experience_replay", {}).get("experience_lbound", 0)  # ⭐ 加入 replay pool 的最小成功数
    self.experience_rbound = self.exp_manager_config.get("experience_replay", {}).get("experience_rbound", 8)  # ⭐ 加入 replay pool 的最大成功数（默认 8，通常等于 n_rollout）
    self.exp_select_mode = self.exp_manager_config.get("experience_replay", {}).get("exp_select_mode", "argmin")  # ⭐ 轨迹选择模式 (argmin/argmax/random)
```

**详细说明**：
- **difficulty2task_dict**：字典结构，key 为 difficulty（整数），value 为 task_id 列表。例如：`{3: ['task_2', 'task_22'], 5: ['task_1']}` 表示 difficulty=3 的桶中有 task_2 和 task_22，difficulty=5 的桶中有 task_1。
- **task2trajectories**：字典结构，key 为 task_id（字符串），value 为 Trajectory 列表。每个 Trajectory 包含完整的消息序列、reward、old_log_probs 等。例如：`{'task_2': [traj1, traj2, ...], 'task_22': [traj3, ...]}`。
- **max_trajectories_per_task**：每个 task 最多保存的轨迹数量，用于限制内存占用。当超过限制时，使用 FIFO（先进先出）策略删除最旧的轨迹。
- **replay_start_ratio**：训练进度阈值，当 `global_steps / total_training_steps >= replay_start_ratio` 时，开始使用 replay。默认值 0.35 表示训练进行到 35% 时开始 replay。
- **初始化时机**：在 `ExperienceManager.__init__()` 中初始化，整个训练过程中保持同一个实例。

#### 3.1.2 更新 difficulty2task_dict

```python
def update_difficulty2task_dict(
    self,
    trajectories: List[Trajectory]
) -> None:
    """
    根据当前 step 的 rollout 结果更新 difficulty2task_dict。
    
    Difficulty 定义：在一个 training step 中，某个 task 的 n 次 rollout 中，reward=1 的次数。
    
    Args:
        trajectories: 当前 step 的所有 trajectory 列表
    """
    # 按 task_id 分组统计
    task_id_to_trajectories = defaultdict(list)
    for traj in trajectories:
        task_id = traj.task_id
        task_id_to_trajectories[task_id].append(traj)
    
    # 计算每个 task 的 difficulty（reward=1 的数量）
    for task_id, trajs in task_id_to_trajectories.items():
        success_count = sum(1 for traj in trajs if traj.reward.outcome == 1.0)
        new_difficulty = success_count
        
        # 检查 task_id 是否已经在某个难度的桶中
        old_difficulty = None
        for diff, task_list in self.difficulty2task_dict.items():
            if task_id in task_list:
                old_difficulty = diff
                break
        
        # 如果 task_id 已经在旧桶中，且新 difficulty 不同，需要重新归类
        if old_difficulty is not None and old_difficulty != new_difficulty:
            # 从旧桶中移除
            self.difficulty2task_dict[old_difficulty].remove(task_id)
            logger.info(f"Moved task {task_id} from difficulty {old_difficulty} to difficulty {new_difficulty}")
        
        # 如果 task_id 不在新桶中，添加到新桶
        if task_id not in self.difficulty2task_dict[new_difficulty]:
            self.difficulty2task_dict[new_difficulty].append(task_id)
            if old_difficulty is None:
                logger.info(f"Added task {task_id} to difficulty {new_difficulty} bucket")
```

**详细说明**：
- **调用时机**：每个 training step 生成轨迹后立即调用，用于更新任务难度分桶
- **输入数据**：当前 step 的所有 trajectory 列表，包括来自 training tasks 和 replay tasks 的 on-policy 轨迹
- **计算逻辑**：
  1. 按 `task_id` 分组所有轨迹
  2. 对每个 task，统计其在该 step 中 `reward.outcome == 1.0` 的轨迹数量
  3. 该数量即为该 task 在当前 step 的 difficulty
- **更新规则**：
  1. **检查旧桶**：遍历所有 difficulty 桶，查找 task_id 是否已经存在于某个桶中
  2. **重新归类**：如果 task_id 已经在旧桶中，且新 difficulty 与旧 difficulty 不同：
     - 从旧桶中移除 task_id
     - 将 task_id 添加到新 difficulty 的桶中
     - 记录移动日志
  3. **首次添加**：如果 task_id 不在任何桶中，直接添加到新 difficulty 的桶中
  4. **保持不变**：如果 task_id 已经在新 difficulty 的桶中，不做任何操作
- **关键点**：
  - 每个 task_id 在某个时刻只属于一个 difficulty 桶（基于最新的 difficulty）
  - 当 task 的 difficulty 发生变化时，会自动从旧桶移动到新桶
  - 这确保了 replaytaskpool 中的任务难度信息始终是最新的
- **示例**：
  - Step 100：task_2 的 difficulty=3，添加到 `difficulty2task_dict[3] = ['task_2']`
  - Step 200：task_2 的 difficulty=5，从 `difficulty2task_dict[3]` 移除，添加到 `difficulty2task_dict[5] = ['task_2']`
- **用途**：后续从 replaytaskpool 选择 tasks 时，可以根据 difficulty 选择特定难度的任务，且每个任务只会在一个难度桶中

#### 3.1.3 保存轨迹到内存

```python
def save_trajectories_to_memory(self, trajectories: List[Trajectory]) -> None:
    """
    将轨迹及其 old_log_probs 保存到内存中的 task2trajectories。
    如果某个 task 的轨迹数量超过 max_trajectories_per_task，则根据 exp_select_mode 替换。
    
    Args:
        trajectories: 包含 old_log_probs 和 entropys 的轨迹列表
    """
    for traj in trajectories:
        task_id = traj.task_id
        if task_id not in self.task2trajectories:
            self.task2trajectories[task_id] = []
        
        # 如果当前轨迹列表已满，根据 exp_select_mode 决定是否替换
        if len(self.task2trajectories[task_id]) >= self.max_trajectories_per_task:
            # 找到当前列表中 entropy 最高（或最低）的轨迹
            current_entropies = [t.metadata.get("entropy", float('inf')) for t in self.task2trajectories[task_id]]
            
            if self.exp_select_mode == "argmin":  # 保留 entropy 最低的
                # 如果新轨迹的 entropy 更低，则替换掉当前最高的
                if traj.metadata.get("entropy", float('inf')) < max(current_entropies):
                    max_entropy_idx = current_entropies.index(max(current_entropies))
                    self.task2trajectories[task_id][max_entropy_idx] = traj
                    logger.debug(f"Replaced trajectory for task {task_id} with lower entropy.")
            elif self.exp_select_mode == "argmax":  # 保留 entropy 最高的
                # 如果新轨迹的 entropy 更高，则替换掉当前最低的
                if traj.metadata.get("entropy", float('-inf')) > min(current_entropies):
                    min_entropy_idx = current_entropies.index(min(current_entropies))
                    self.task2trajectories[task_id][min_entropy_idx] = traj
                    logger.debug(f"Replaced trajectory for task {task_id} with higher entropy.")
            else:  # 默认 FIFO
                self.task2trajectories[task_id].pop(0)  # 移除最旧的轨迹
                self.task2trajectories[task_id].append(traj)
                logger.debug(f"Replaced trajectory for task {task_id} using FIFO.")
        else:
            self.task2trajectories[task_id].append(traj)
```

**详细说明**：
- **调用时机**：在筛选出符合条件的轨迹后，保存到内存存储中
- **输入**：筛选后的轨迹列表，每个轨迹包含 `old_log_probs`、`entropy` 和 `policy_version`
- **内存管理策略**：
  - 如果某个 task 的轨迹数量未达到 `max_trajectories_per_task`，直接添加
  - 如果已满，根据 `exp_select_mode` 决定是否替换：
    - `"argmin"`：如果新轨迹的 entropy 更低，替换掉当前 entropy 最高的轨迹
    - `"argmax"`：如果新轨迹的 entropy 更高，替换掉当前 entropy 最低的轨迹
    - 默认（FIFO）：删除最旧的轨迹，添加新轨迹
- **关键点**：
  - 只保存符合条件的轨迹（高 reward + 低 entropy）
  - 使用 `max_trajectories_per_task` 限制内存占用
  - 根据 `exp_select_mode` 保持轨迹池的质量

#### 3.1.4 从内存获取 off-policy 轨迹

```python
def get_offpolicy_trajectories_from_memory(
    self, 
    task_id: str, 
    num_trajectories: int = 1
) -> List[Trajectory]:
    """
    从内存中的 task2trajectories 获取指定任务的 off-policy trajectory。
    根据 exp_select_mode 选择轨迹。
    
    Args:
        task_id: 任务 ID
        num_trajectories: 获取的轨迹数量
        
    Returns:
        List[Trajectory]: Off-policy trajectory 列表
    """
    available_trajectories = self.task2trajectories.get(task_id, [])
    if not available_trajectories:
        return []
    
    # 根据 exp_select_mode 选择轨迹
    if self.exp_select_mode == "argmin":  # 选择 entropy 最低的
        available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('inf')))
    elif self.exp_select_mode == "argmax":  # 选择 entropy 最高的
        available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('-inf')), reverse=True)
    # 默认或 random 模式下，不排序，直接随机选择
    
    # 采样 num_trajectories 个轨迹（允许重复）
    sampled_trajectories = random.choices(
        available_trajectories, 
        k=min(num_trajectories, len(available_trajectories))
    )
    
    # 标记为 experience replay
    for traj in sampled_trajectories:
        traj.metadata["is_experience_replay"] = True
        # ⭐ Experience Replay: LLM 消息保持原样（author="llm"），确保 loss_mask=1
        # 使用 exp_mask=1 来标记 off-policy 数据，而不是使用 loss_mask=0
    
    return sampled_trajectories
```

**详细说明**：
- **调用时机**：在训练循环中，为 replay tasks 获取 off-policy trajectory 时调用
- **输入**：task_id 和需要获取的轨迹数量
- **选择策略**：
  - `exp_select_mode="argmin"`：选择 entropy 最低的轨迹（模型最自信的成功经验）
  - `exp_select_mode="argmax"`：选择 entropy 最高的轨迹
  - 默认：随机选择
- **Off-policy Loss 计算**：
  - LLM 消息保持 `author="llm"`，确保 `loss_mask=1`，参与 off-policy loss 计算
  - 使用 `exp_mask=1` 标记 off-policy 数据
  - 在 `het_compute_token_on_off_policy_loss` 中使用不同的 cliprange 计算 off-policy loss
- **返回**：Off-policy trajectory 列表，已标记为 `is_experience_replay=True`

#### 3.1.5 更新 skip_uid_set 并筛选轨迹

```python
def update_skip_uid_set_and_filter_trajectories(
    self,
    trajectories: List[Trajectory],
    n_rollout: int,
    entropys: torch.Tensor,  # (bs, response_len)
    response_mask: torch.Tensor  # (bs, response_len)
) -> List[Trajectory]:
    """
    根据 rollout 结果更新 skip_uid_set，并筛选符合条件的轨迹（非全对非全错，且选择 entropy 最低的成功轨迹）。
    
    Args:
        trajectories: 当前 step 的所有 on-policy trajectory 列表
        n_rollout: 每个 task 的 rollout 数量
        entropys: 当前 step 所有 on-policy 轨迹的 token 级 entropy (bs, response_len)
        response_mask: 当前 step 所有 on-policy 轨迹的 response mask (bs, response_len)
        
    Returns:
        List[Trajectory]: 筛选后符合条件的轨迹列表，用于保存到 task2trajectories
    """
    filtered_trajectories_to_save = []
    
    # 按 task_id 分组统计
    task_id_to_trajectories = defaultdict(list)
    task_id_to_entropy_list = defaultdict(list)
    for i, traj in enumerate(trajectories):
        task_id = traj.task_id
        task_id_to_trajectories[task_id].append(traj)
        
        # 计算轨迹的平均 entropy
        traj_entropys = entropys[i].cpu().numpy()
        traj_response_mask = response_mask[i].cpu().numpy()
        valid_entropys = traj_entropys[traj_response_mask.astype(bool)]
        avg_entropy = np.mean(valid_entropys) if len(valid_entropys) > 0 else 0.0
        traj.metadata["entropy"] = avg_entropy  # 保存平均 entropy 到 metadata
        task_id_to_entropy_list[task_id].append((avg_entropy, traj))
    
    for task_id, trajs in task_id_to_trajectories.items():
        success_count = sum(1 for traj in trajs if traj.reward.outcome == 1.0)
        
        # 1. 更新 skip_uid_set
        if success_count == n_rollout:  # 全部做对
            if task_id not in self.skip_uid_set:
                self.skip_uid_set.add(task_id)
                # 从 difficulty2task_dict 中移除
                for diff, task_list in list(self.difficulty2task_dict.items()):
                    if task_id in task_list:
                        self.difficulty2task_dict[diff].remove(task_id)
                        if not self.difficulty2task_dict[diff]:
                            del self.difficulty2task_dict[diff]  # 如果桶为空则删除
                # 从 task2trajectories 中删除
                if task_id in self.task2trajectories:
                    del self.task2trajectories[task_id]
                logger.info(f"Task {task_id} fully solved, added to skip_uid_set and removed from replay pool.")
            continue  # 不再考虑加入经验池
        elif task_id in self.skip_uid_set:  # 如果之前在 skip_uid_set 但现在没全对，则移除
            self.skip_uid_set.remove(task_id)
            logger.info(f"Task {task_id} no longer fully solved, removed from skip_uid_set.")
        
        # 2. 筛选加入 experience replay pool 的 tasks (非全对非全错)
        if self.experience_lbound < success_count < self.experience_rbound:
            # 选择 entropy 最低的成功轨迹
            successful_trajs_with_entropy = [
                (e, t) for e, t in task_id_to_entropy_list[task_id] if t.reward.outcome == 1.0
            ]
            if successful_trajs_with_entropy:
                if self.exp_select_mode == "argmin":
                    best_traj = min(successful_trajs_with_entropy, key=lambda x: x[0])[1]
                elif self.exp_select_mode == "argmax":
                    best_traj = max(successful_trajs_with_entropy, key=lambda x: x[0])[1]
                else:  # 默认随机选择一个成功的
                    best_traj = random.choice([t for e, t in successful_trajs_with_entropy])
                filtered_trajectories_to_save.append(best_traj)
        else:
            logger.debug(f"Task {task_id} (success_count={success_count}) not within bounds [{self.experience_lbound}, {self.experience_rbound}], skipping for experience pool.")
    
    return filtered_trajectories_to_save
```

**详细说明**：
- **调用时机**：在每个 training step 生成轨迹并计算 entropy 后调用
- **输入**：
  - `trajectories`：当前 step 的所有 on-policy 轨迹
  - `n_rollout`：每个 task 的 rollout 数量
  - `entropys`：所有轨迹的 token 级 entropy，形状 `(bs, response_len)`
  - `response_mask`：所有轨迹的 response mask，形状 `(bs, response_len)`
- **处理流程**：
  1. **按 task_id 分组**：将所有轨迹按 `task_id` 分组
  2. **计算 entropy**：对每个轨迹，计算其平均 entropy（基于 `entropys` 和 `response_mask`）
  3. **更新 skip_uid_set**：
     - 如果某个 task 的 `success_count == n_rollout`（全部做对），加入 `skip_uid_set`，并从 `difficulty2task_dict` 和 `task2trajectories` 中移除
     - 如果之前在 `skip_uid_set` 但现在没全对，从 `skip_uid_set` 中移除
  4. **筛选轨迹**：
     - 只考虑"部分成功"的任务：`experience_lbound < success_count < experience_rbound`
     - 对于每个符合条件的任务，从所有成功轨迹中选择 entropy 最低（或最高）的轨迹
- **返回**：筛选后的轨迹列表，用于保存到 `task2trajectories`
- **关键点**：
  - Entropy 计算基于 `entropys` 和 `response_mask`，只计算有效 token 的平均 entropy
  - 保存的 entropy 用于后续的轨迹选择（保存和 replay 时）
  - 确保每个 task 只保存一个最优轨迹（entropy 最低的成功轨迹）

#### 3.1.6 从 replaytaskpool 选择 tasks

```python
def sample_tasks_from_replaypool(
    self,
    difficulty: Optional[int] = None,
    num_tasks: int = 2,
    strategy: str = "random"
) -> List[str]:
    """
    从 replaytaskpool 中采样指定数量的 task_id。
    
    Args:
        difficulty: 指定的难度值。如果为 None，则随机选择一个难度
        num_tasks: 需要采样的 task 数量
        strategy: 采样策略，"random" 或 "uniform"（按难度均匀采样）
        
    Returns:
        List[str]: 采样得到的 task_id 列表
    """
    if not self.difficulty2task_dict:
        return []
    
    if difficulty is None:
        # 随机选择一个难度
        available_difficulties = [d for d, tasks in self.difficulty2task_dict.items() if len(tasks) > 0]
        if not available_difficulties:
            return []
        difficulty = random.choice(available_difficulties)
    
    # 从指定难度的 task 列表中采样
    available_tasks = self.difficulty2task_dict.get(difficulty, [])
    if len(available_tasks) == 0:
        return []
    
    # 采样（允许重复）
    sampled_tasks = random.choices(available_tasks, k=min(num_tasks, len(available_tasks)))
    return sampled_tasks
```

**详细说明**：
- **调用时机**：在训练循环中，当需要从 replaytaskpool 选择 tasks 时调用
- **参数说明**：
  - `difficulty`：如果指定，从该难度的桶中采样；如果为 None，随机选择一个有任务的难度桶
  - `num_tasks`：需要采样的 task 数量，通常等于 `replay_task_count` 配置
  - `strategy`：采样策略，目前支持 "random"（随机采样），未来可以扩展为 "uniform"（按难度均匀采样）
- **采样逻辑**：
  1. 如果 `difficulty` 为 None，从所有有任务的难度桶中随机选择一个
  2. 从选定难度的 task 列表中随机采样 `num_tasks` 个
  3. 允许重复采样（同一个 task 可能被采样多次）
- **返回**：task_id 列表，需要后续转换为 Task 对象
- **边界情况**：
  - 如果 `difficulty2task_dict` 为空，返回空列表
  - 如果指定难度的桶为空，返回空列表
  - 如果可用任务数少于 `num_tasks`，返回所有可用任务

#### 3.1.7 获取 off-policy batch

```python
def get_offpolicy_batch(
    self, 
    tasks: List[Task], 
    num_trajectories_per_task: int = 1
) -> List[Trajectory]:
    """
    为给定的任务列表从内存中获取 off-policy trajectory。
    
    Args:
        tasks: 任务列表
        num_trajectories_per_task: 每个任务获取的轨迹数量
        
    Returns:
        List[Trajectory]: Off-policy trajectory 列表
    """
    offpolicy_trajectories = []
    
    for task in tasks:
        try:
            trajs = self.get_offpolicy_trajectories_from_memory(
                task_id=task.task_id,
                num_trajectories=num_trajectories_per_task
            )
            offpolicy_trajectories.extend(trajs)
        except Exception as e:
            logger.warning(f"Failed to get off-policy trajectory for task {task.task_id}: {e}")
            continue
    
    return offpolicy_trajectories
```

**详细说明**：
- **调用时机**：在训练循环中，为 replay tasks 获取 off-policy trajectory 时调用
- **输入**：Task 对象列表（replay tasks），每个任务需要获取的轨迹数量
- **处理流程**：
  1. 遍历每个 task，调用 `get_offpolicy_trajectories_from_memory()` 从内存中获取历史轨迹
  2. 轨迹已经是 `Trajectory` 对象格式，直接使用
  3. 轨迹已经包含 `is_experience_replay = True` 标记和 `old_log_probs`
  4. 收集所有轨迹并返回
- **错误处理**：如果某个 task 获取失败，记录警告但继续处理其他 task，不影响整体流程
- **返回**：`Trajectory` 对象列表，每个包含完整的消息序列、奖励和 old_log_probs
- **优势**：直接从内存获取，速度快，无需网络请求

### 3.2 Off-policy Trajectory 转换为 Linear_CMT

**位置**: `agentevolver/module/env_manager/env_manager.py`

**作用**：将 off-policy Trajectory 转换为与 on-policy 相同格式的 Linear_CMT 对象，确保可以使用相同的 tokenize 和 batch 处理流程。这是数据流中的关键转换步骤。

**关键点**:
1. Off-policy trajectory 需要转换为与 on-policy 相同格式的 `Trajectory` 对象（Linear_CMT）
2. 使用相同的 `to_dataproto()` 流程：`Trajectory` → `group_tokenize()` → `Sample[]` → `samples_to_dataproto()` → `DataProto`
3. 在 `Sample.extras` 中标记为 off-policy，这样 `samples_to_dataproto()` 会自动创建 `exp_mask`
4. 保存 `recorded_old_log_probs` 到 Trajectory metadata，后续在训练循环中使用

**为什么需要转换**：
- On-policy 轨迹是通过 `EnvWorker.execute()` → `AgentFlow.execute()` 生成的，已经是 Linear_CMT 格式
- Off-policy 轨迹是从 ReMe 获取的原始数据（messages 列表），需要转换为 Linear_CMT 格式
- 只有转换为相同格式，才能使用统一的 `to_dataproto()` 流程，避免重复代码

**实现步骤**:

#### Step 1: 将 Off-policy Trajectory 转换为 Linear_CMT

```python
# 在 ParallelEnvManager 中新增方法
def convert_offpolicy_to_cmt(
    self,
    offpolicy_trajectories: List[Trajectory],
    config: DictConfig,
    tokenizer
) -> List[Linear_CMT]:
    """
    将 off-policy Trajectory 转换为 Linear_CMT 对象，以便使用相同的 tokenize 流程。
    
    Args:
        offpolicy_trajectories: Off-policy trajectory 列表
        config: 配置对象
        tokenizer: Tokenizer 实例
        
    Returns:
        List[Linear_CMT]: 转换后的 Linear_CMT 对象列表
    """
    cmt_array = []
    
    for traj in offpolicy_trajectories:
        # 根据 context_template 创建对应的 CMT 对象
        if config.actor_rollout_ref.rollout.context_template == "linear":
            cmt = Linear_CMT(config, tokenizer)
        elif config.actor_rollout_ref.rollout.context_template == "linear_think":
            cmt = LinearThinkCMT(config, tokenizer)
        else:
            raise ValueError(f"Unsupported context template: {config.actor_rollout_ref.rollout.context_template}")
        
        # 设置基本信息
        # ⭐ 关键：data_id 必须是整数或可以转换为整数（因为 group_ids = torch.tensor([int(s.data_id) for s in samples])）
        # 为 off-policy 数据分配独立的 data_id，避免与 on-policy 数据混合分组（GRPO 需要）
        # 使用一个大的偏移量（例如 1000000）确保 off-policy data_id 不会与 on-policy data_id 冲突
        # ⭐ 注意：Trajectory 对象没有直接的 task_id 属性，需要从 metadata 中获取
        task_id = traj.metadata.get("task_id", "unknown")
        traj_index = len(cmt_array)  # 使用索引确保唯一性
        # 将 task_id 转换为整数（如果可能），否则使用 hash
        try:
            task_id_int = int(task_id)
        except (ValueError, TypeError):
            task_id_int = hash(task_id) % 100000  # 使用 hash 确保是整数
        
        # data_id 格式：使用大偏移量 + task_id + index，确保唯一且是整数
        # 例如：1000000 + task_id_int * 1000 + traj_index
        cmt.data_id = str(1000000 + task_id_int * 1000 + traj_index)
        cmt.rollout_id = traj.metadata.get("rollout_id", "0")
        cmt.task_id = task_id  # Linear_CMT 有 task_id 属性
        cmt.query = traj.query or traj.metadata.get("query", "")
        cmt.reward = traj.reward
        cmt.is_terminated = traj.is_terminated
        
        # 将 messages 转换为 ExtendedMessage 并填充 full_context
        # ⭐ Experience Replay: 对于 off-policy 数据，所有 LLM 消息的 author 保持为 "llm"
        # 这样 loss_mask 不会为 0，LLM 消息会参与 off-policy loss 计算
        # 使用 exp_mask=1 来区分 on-policy 和 off-policy，而不是用 loss_mask=0
        from agentevolver.module.context_manager.cmt_linear import ExtendedMessage
        for msg in traj.steps:
            role = msg.get("role", "user")
            author = msg.get("role", "user")
            # ⭐ Experience Replay: LLM 消息保持 author="llm"，用于计算 off-policy loss
            # 使用 exp_mask 区分 on/off-policy，而不是让 loss_mask=0
            if role == "assistant":
                author = "llm"  # 保持为 "llm"，loss_mask=1，参与 off-policy loss 计算
            elif role == "user":
                author = "user"  # 保持为user
            else:
                author = role  # 其他角色保持原样
            
            ext_msg = ExtendedMessage(
                author=author,
                role=role,
                content=msg.get("content", ""),
                token_generator='auto',
                tokenizer=tokenizer,
            )
            cmt.full_context.append(ext_msg)
        
        # 标记为 experience replay
        cmt.metadata["is_experience_replay"] = True
        cmt.metadata["old_log_probs"] = traj.metadata.get("old_log_probs")
        cmt.metadata["policy_version"] = traj.metadata.get("policy_version")
        
        cmt_array.append(cmt)
    
    return cmt_array
```

**详细说明**：
- **调用时机**：在获取 off-policy trajectory 后，转换为 Linear_CMT 格式时调用
- **输入**：Off-policy Trajectory 列表（从 `task2trajectories` 内存存储中获取，已经是 Trajectory 对象）
- **转换流程**：
  1. 根据配置的 `context_template` 创建对应的 CMT 对象（Linear_CMT 或 LinearThinkCMT）
  2. 设置基本信息：`data_id`、`rollout_id`、`task_id`、`query`、`reward` 等
  3. 将 messages 转换为 `ExtendedMessage` 对象，填充到 `full_context`
  4. **⭐ Experience Replay 关键**：保持 LLM 消息的 `author="llm"`，确保 `loss_mask=1`，参与 off-policy loss 计算
  5. 设置 `is_experience_replay = True` 和 `old_log_probs` 到 metadata
- **Off-policy Loss 计算**：
  - LLM 消息保持 `author="llm"`，确保 `loss_mask=1`
  - 使用 `exp_mask=1` 标记 off-policy 数据
  - 在 `het_compute_token_on_off_policy_loss` 中：
    - `on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)` - on-policy
    - `off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)` - off-policy
  - 两者合并：`pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)`

**`tokenize_steps` 中的特殊处理**：

对于 off-policy 数据，需要在 `tokenize_steps` 中进行特殊处理：
  ```python
  # 在 tokenize_steps 方法中
  is_experience_replay = self.metadata.get("is_experience_replay", False)
  
  # 对于 experience replay，允许 role="assistant" 且 author="llm" 成为分割点
  if (split_prompt_reponse_index == -1) and (ext_msg.need_training or (is_experience_replay and ext_msg.role == "assistant")):
      split_prompt_reponse_index = len(input_ids)
  
  # 对于 experience replay，LLM 消息的 loss_mask 应该为 1
  if is_experience_replay and ext_msg.role == "assistant":
      msg_loss_mask = [1] * len(ext_msg.token_arr)  # 参与 off-policy loss 计算
  else:
      msg_loss_mask = ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)
  ```

**On-policy vs Off-policy 对比**：

| 数据类型 | author | loss_mask | exp_mask | loss 计算 |
|----------|--------|-----------|----------|-----------|
| On-policy | `"llm"` | 1 | 0 | `on_pg_loss` |
| Off-policy | `"llm"` | 1 | 1 | `off_pg_loss` |

**关键设计决策**：
1. **保持 `author="llm"`**：确保 off-policy 数据的 LLM 消息的 `loss_mask=1`，参与 loss 计算
2. **使用 `exp_mask` 区分**：不使用 `loss_mask=0` 来区分 off-policy，而是使用独立的 `exp_mask`
3. **参考 ExGRPO**：ExGRPO 的 `mix_core_alg.py` 也是分别计算 `on_pg_loss` 和 `off_pg_loss`，然后合并
- **关键设计**：
  - `data_id` 使用整数格式（例如：`1000000 + task_id_int * 1000 + traj_index`），确保可以转换为整数
  - **重要**：现有代码中 `group_ids = torch.tensor([int(s.data_id) for s in samples])` 要求 data_id 必须是整数或可以转换为整数
  - 使用大偏移量（1000000）确保 off-policy data_id 不会与 on-policy data_id（通常是 0, 1, 2, ...）冲突
  - 独立的 `data_id` 确保 GRPO 分组时 off-policy 数据不会与 on-policy 数据混合
  - 保持与 on-policy 轨迹相同的结构，便于后续统一处理

#### Step 2: 修改 tokenize_steps 以支持 is_experience_replay

```python
# 在 Linear_CMT.tokenize_steps() 方法中修改（约第 609 行）
def tokenize_steps(self, ext_steps: List[ExtendedMessage], debug=False) -> dict:
    # ... 现有代码（移除最后一个非LLM消息，处理experience等）...
    
    # ⭐ 检查是否为 experience replay 数据
    is_experience_replay = self.metadata.get("is_experience_replay", False)
    
    # mapping
    input_ids = []
    attention_mask = []
    loss_mask = []
    split_prompt_reponse_index = -1
    
    for ext_msg in ext_steps:
        # find split index, this have to be done before input_ids += ext_msg.token_arr
        # ⭐ Experience Replay: 对于 off-policy 数据，所有 LLM 消息都应该参与 loss 计算
        # 使用 exp_mask 区分 on-policy 和 off-policy，而不是让 loss_mask=0
        if (split_prompt_reponse_index == -1) and (ext_msg.need_training or (is_experience_replay and ext_msg.role == "assistant")):
            split_prompt_reponse_index = len(input_ids)
            # 对于 experience replay，允许 author 为 "llm(do_not_train)"
            if not is_experience_replay:
                assert ext_msg.author == 'llm', "The first message after initialization should be from LLM, not from env or user"
        input_ids += ext_msg.token_arr
        attention_mask += [1] * len(ext_msg.token_arr)
        # ⭐ Experience Replay: 对于 off-policy 数据，LLM 消息的 loss_mask 应该为 1（参与 loss 计算）
        # 使用 exp_mask 来区分 on/off-policy，而不是用 loss_mask=0
        if is_experience_replay and ext_msg.role == "assistant":
            # Off-policy LLM 消息：loss_mask = 1（参与 off-policy loss 计算）
            msg_loss_mask = [1] * len(ext_msg.token_arr)
        else:
            msg_loss_mask = ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)
        loss_mask += msg_loss_mask
    
    # ⭐ 如果是 experience replay 数据且没有找到 split_prompt_reponse_index
    # 说明没有 LLM 消息（异常情况），设置为第一个位置
    if is_experience_replay and split_prompt_reponse_index == -1:
        split_prompt_reponse_index = 0  # 所有内容都是 response（用于计算 loss）
    
    assert split_prompt_reponse_index != -1, "split_prompt_reponse_index should not be -1, at least one message should be in the context"
    
    # ... 后续代码保持不变（分离 prompt 和 response）...
```

**详细说明**：
- **调用时机**：在 `group_tokenize()` 中调用，用于将 trajectory 转换为 tokenized 格式
- **关键修改**：
  1. 对于 experience replay 数据，LLM 消息（role="assistant"）也会成为分割点
  2. 对于 experience replay 数据，LLM 消息的 `loss_mask=1`（参与 off-policy loss 计算）
  3. 使用 `exp_mask=1` 标记 off-policy 数据，在 loss 计算时区分 on/off-policy
- **为什么需要这个修改**：
  - Off-policy 数据需要参与 loss 计算（参考 ExGRPO）
  - 使用 `exp_mask` 区分 on-policy 和 off-policy，而不是使用 `loss_mask=0`
  - 确保 off-policy 数据的 LLM 消息的 `loss_mask=1`，参与 off-policy loss 计算
- **结果**：
  - Off-policy 数据：LLM 消息的 `loss_mask=1`，`exp_mask=1`，参与 `off_pg_loss` 计算
  - On-policy 数据：LLM 消息的 `loss_mask=1`，`exp_mask=0`，参与 `on_pg_loss` 计算

#### Step 3: 修改 get_extra 以支持 Experience Replay

```python
# 在 ParallelEnvManager.get_extra() 中
def get_extra(self, cmt):
    """
    获取 trajectory 的额外信息，用于创建 Sample.extras。
    """
    extras = {
        "add_exp": cmt.metadata.get("add_exp", None),
        "task_train_expmode": cmt.metadata.get("task_train_exp_mode", None),
        "experience_list": cmt.metadata.get("experience_list", []),
        "is_experience_replay": cmt.metadata.get("is_experience_replay", False),  # ⭐ 新增：标记是否为 experience replay
        "old_log_probs": cmt.metadata.get("old_log_probs"),  # ⭐ 新增：历史策略的 log_prob
    }
    return extras
```

**详细说明**：
- **调用时机**：在 `trajectories_to_samples()` 中，为每个 trajectory 创建 Sample 时调用
- **作用**：提取 trajectory 的额外信息，存储到 `Sample.extras` 中，供后续处理使用
- **新增字段**：
  - `is_experience_replay`：标记是否为 experience replay 数据，用于创建 `exp_mask`
  - `old_log_probs`：历史策略的 log 概率，用于重要性采样
- **兼容性**：同时保留原有的 `add_exp`、`task_train_expmode`、`experience_list` 字段，确保与 experience-guided 机制兼容

#### Step 4: 修改 samples_to_dataproto 以支持 Experience Replay

```python
# 在 ParallelEnvManager.samples_to_dataproto() 中修改 exp_mask 的创建逻辑
# 第 521-527 行修改为：

# Create experience mask: 
# 1. 如果是 experience replay 数据（is_experience_replay=True），整个 response 都标记为 off-policy
# 2. 如果是 on-policy 但使用了 experience-guided（add_exp=True 且 task_train_expmode="discard"），也标记
# ⭐ 注意：优先级是 experience replay > experience-guided > on-policy
if sample.extras.get("is_experience_replay", False):
    # Experience Replay: response 部分全为 1（off-policy）
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
elif sample.extras.get("add_exp", False) and sample.extras.get("task_train_expmode", None)=="discard":
    # Experience-guided: prompt 和 response 都标记（因为 experience 在 prompt 中）
    prompt_exp_mask_list.append(torch.ones(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
else:
    # On-policy without experience: 全为 0
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))
```

**详细说明**：
- **调用时机**：在 `samples_to_dataproto()` 中，为每个 Sample 创建 `exp_mask` 时执行
- **exp_mask 的作用**：
  - `exp_mask` 是一个二进制 mask，标识哪些 token 是 off-policy 的
  - 在 loss 计算时，通过 `exp_mask` 区分 on-policy 和 off-policy 数据，应用不同的 cliprange
- **三种情况的处理**：
  1. **Experience Replay**（`is_experience_replay=True`）：
     - Prompt 部分：全为 0（prompt 本身不是 off-policy）
     - Response 部分：全为 1（整个 response 都是 off-policy）
  2. **Experience-Guided**（`add_exp=True` 且 `task_train_expmode="discard"`）：
     - Prompt 和 Response 都标记为 1（因为 experience 被插入到 prompt 中）
  3. **纯 On-policy**：
     - Prompt 和 Response 都标记为 0（标准 on-policy 训练）
- **关键点**：`exp_mask` 会在后续的 loss 计算中使用，确保 off-policy 数据使用正确的 old_log_prob 和 cliprange

#### Step 5: 保存 recorded_old_log_probs 到 DataProto

```python
# 在 samples_to_dataproto() 的最后，添加 recorded_old_log_probs
recorded_old_log_probs_list = []
for sample in samples:
    old_log_probs = sample.extras.get("old_log_probs")
    if old_log_probs is not None:
        # 转换为 tensor 并对齐长度
        if isinstance(old_log_probs, (list, np.ndarray)):
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        # 对齐到 response_length
        response_length = len(sample.response_ids)
        if len(old_log_probs) > response_length:
            old_log_probs = old_log_probs[:response_length]
        elif len(old_log_probs) < response_length:
            old_log_probs = torch.cat([
                old_log_probs,
                torch.zeros(response_length - len(old_log_probs), dtype=torch.float32)
            ])
        recorded_old_log_probs_list.append(old_log_probs)
    else:
        # 如果没有记录，创建零向量（后续会重新计算）
        recorded_old_log_probs_list.append(
            torch.zeros(len(sample.response_ids), dtype=torch.float32)
        )

# Pad 到 max_response_length
recorded_old_log_probs = pad_sequence(recorded_old_log_probs_list, batch_first=True, padding_value=0.0)
recorded_old_log_probs = pad_sequence_to_length(
    recorded_old_log_probs, max_response_length_this_batch, 0.0
)

# 添加到 batch（在 TensorDict 构造之前）
# ⭐ 注意：recorded_old_log_probs 需要添加到 batch TensorDict 中
# 在 samples_to_dataproto() 的最后，在构造 TensorDict 时添加：
batch = TensorDict(
    {
        "prompts": prompt_ids,
        "responses": response_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "exp_mask": exp_mask,
        "step_ids": step_ids_pad,
        "group_ids": group_ids,
        "recorded_old_log_probs": recorded_old_log_probs,  # ⭐ 新增：历史策略的 old_log_probs
    },
    batch_size=len(samples),
)
```

**详细说明**：
- **调用时机**：在 `samples_to_dataproto()` 的最后，将所有 Sample 的 `old_log_probs` 收集并添加到 batch 中
- **处理流程**：
  1. 遍历所有 Sample，从 `extras["old_log_probs"]` 获取历史策略的 log 概率
  2. 如果 `old_log_probs` 存在，转换为 tensor 并对齐到 `response_length`
  3. 如果 `old_log_probs` 不存在（on-policy 数据），创建零向量（后续会重新计算）
  4. 将所有 `old_log_probs` 进行 padding，对齐到 batch 的 `max_response_length`
  5. 添加到 batch 中，供后续训练使用
- **长度对齐**：
  - 如果历史 `old_log_probs` 更长：截断到 `response_length`
  - 如果历史 `old_log_probs` 更短：用 0 填充（这些位置在后续会使用当前策略的 old_log_prob）
- **关键点**：`recorded_old_log_probs` 只对 off-policy 数据有效，on-policy 数据对应位置为零向量，后续会被替换为当前计算的 old_log_prob

### 3.4 Data Collate 机制（Experience Mix Collate Function）

**位置**: `agentevolver/module/env_manager/env_manager.py` 或新建 `agentevolver/module/exp_manager/experience_collate.py`

**作用**：参考 ExGRPO 的设计，实现一个 data collate 函数，在数据准备阶段混合 on-policy 和 off-policy tasks。这个函数负责：
1. 根据 `exp_ratio` 计算目标 experience task 数量
2. 从 replaytaskpool 采样 experience task_ids
3. 如果 experience tasks 不足，用 on-policy tasks 补充
4. 确保最终 batch_size 保持不变

**设计要点**：
- **Exp Ratio**：控制 experience tasks 的比例，默认 0.5（50%）
- **Batch Size 不变**：无论是否有 experience replay，batch_size 始终为配置值
- **自动补充**：如果 experience pool 中的任务数量不足，自动用 on-policy tasks 补充
- **Rollout 总数保证**：总 rollouts = batch_size * n_rollout

**⭐ 与 ExGRPO 的差异**：
- **ExGRPO**：在 DataLoader 的 `collate_fn` 中进行混合（数据样本级别），接收已经准备好的数据样本（row_dict），从 `experience_pool` 采样 off-policy 样本，与 on-policy 样本混合
- **AgentEvolver**：在训练循环中进行混合（Task 级别），在 Task 构建之后、rollout 之前，从 `replaytaskpool` 采样 replay tasks，与 training tasks 混合
- **关键差异**：ExGRPO 在数据样本级别混合，AgentEvolver 在 Task 级别混合，但两者都使用 `exp_ratio` 控制混合比例，确保 batch_size 不变

**实现代码**：

```python
class ExperienceMixCollateFn:
    """
    混合 on-policy 和 off-policy tasks 的 collate 函数。
    参考 ExGRPO 的 ExperienceMixCollateFn 设计。
    """
    def __init__(
        self,
        exp_manager: ExperienceManager,
        train_task_manager: TaskManager,  # ⭐ 新增：用于获取 Task 对象
        exp_ratio: float = 0.5,
        replay_start_ratio: float = 0.35,
        offpolicy_trajectories_per_task: int = 1,
        n_rollout: int = 8,
    ):
        self.exp_manager = exp_manager
        self.train_task_manager = train_task_manager  # ⭐ 新增：用于从 task_id 获取 Task 对象
        self.exp_ratio = exp_ratio  # ⭐ Experience tasks 的比例（默认 0.5）
        self.replay_start_ratio = replay_start_ratio
        self.offpolicy_trajectories_per_task = offpolicy_trajectories_per_task
        self.n_rollout = n_rollout
    
    def __call__(
        self,
        training_tasks: List[Task],
        training_progress: float,
        enable_replay: bool = True,
    ) -> Tuple[List[Task], List[Task]]:
        """
        混合 on-policy 和 off-policy tasks。
        
        Args:
            training_tasks: 原始 training tasks 列表（batch_size 个）
            training_progress: 当前训练进度（global_steps / total_training_steps）
            enable_replay: 是否启用 replay（需要同时满足 training_progress >= replay_start_ratio）
            
        Returns:
            Tuple[List[Task], List[Task]]: (experience_tasks, on_policy_tasks)
            - experience_tasks: 从 replaytaskpool 选择的 tasks（需要获取 off-policy trajectories）
            - on_policy_tasks: 纯 on-policy tasks（不需要 off-policy trajectories）
            - 总数量 = len(experience_tasks) + len(on_policy_tasks) = len(training_tasks)
        """
        batch_size = len(training_tasks)
        
        # 检查是否启用 replay
        if not enable_replay or training_progress < self.replay_start_ratio:
            # 阶段 1：只使用 training tasks，不进行 replay
            return [], training_tasks
        
        # 阶段 2：混合 experience tasks 和 on-policy tasks
        # 计算目标 experience task 数量
        target_exp_count = int(batch_size * self.exp_ratio)  # 例如：64 * 0.5 = 32
        
        # 从 replaytaskpool 采样 experience task_ids
        valid_exp_task_ids = []
        for difficulty, task_ids in self.exp_manager.difficulty2task_dict.items():
            # 检查这些 task_ids 是否有可用的轨迹
            for task_id in task_ids:
                if task_id in self.exp_manager.task2trajectories:
                    if len(self.exp_manager.task2trajectories[task_id]) > 0:
                        valid_exp_task_ids.append(task_id)
        
        # 采样 experience task_ids（最多 target_exp_count 个）
        n_exp = min(len(valid_exp_task_ids), target_exp_count)
        if n_exp > 0:
            # 随机采样（可以后续支持按难度采样）
            import random
            sampled_exp_task_ids = random.sample(valid_exp_task_ids, n_exp)
        else:
            sampled_exp_task_ids = []
        
        # 将 experience task_ids 转换为 Task 对象
        experience_tasks = []
        for task_id in sampled_exp_task_ids:
            # ⭐ 从 train_task_manager 获取 Task 对象
            task = self.train_task_manager.get_task_by_id(task_id)
            if task is not None:
                experience_tasks.append(task)
            else:
                logger.warning(f"Failed to get Task object for task_id={task_id}, skipping")
        
        # 计算需要补充的 on-policy tasks 数量
        n_exp_actual = len(experience_tasks)
        n_on_policy = batch_size - n_exp_actual  # 确保总数为 batch_size
        
        # 从 training_tasks 中选择 on-policy tasks
        on_policy_tasks = training_tasks[:n_on_policy]
        
        # 验证总数
        assert len(experience_tasks) + len(on_policy_tasks) == batch_size, \
            f"Total tasks mismatch: {len(experience_tasks)} + {len(on_policy_tasks)} != {batch_size}"
        
        logger.info(
            f"Mixed batch: {len(experience_tasks)} experience tasks + "
            f"{len(on_policy_tasks)} on-policy tasks = {batch_size} total"
        )
        
        return experience_tasks, on_policy_tasks
```

**详细说明**：
- **调用时机**：在每个 training step 开始时，生成 batch 之前
- **输入**：原始 training tasks 列表（batch_size 个）
- **输出**：两个列表
  - `experience_tasks`：从 replaytaskpool 选择的 tasks（需要获取 off-policy trajectories）
  - `on_policy_tasks`：纯 on-policy tasks（不需要 off-policy trajectories）
- **关键逻辑**：
  1. 计算目标 experience task 数量：`target_exp_count = batch_size * exp_ratio`
  2. 从 replaytaskpool 采样 experience task_ids（最多 target_exp_count 个）
  3. 如果 experience tasks 不足，用 on-policy tasks 补充
  4. 确保总数为 batch_size
- **Rollout 数量保证**：
  - Experience tasks：每个 task 有 `offpolicy_trajectories_per_task` 个 off-policy + `(n_rollout - offpolicy_trajectories_per_task)` 个 on-policy = `n_rollout` 个 total
  - On-policy tasks：每个 task 有 `n_rollout` 个 on-policy
  - 总 rollouts = `len(experience_tasks) * n_rollout + len(on_policy_tasks) * n_rollout = batch_size * n_rollout` ✓

### 3.5 训练循环集成

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py`

**作用**：在训练循环中集成 experience replay 机制，包括使用 data collate 函数混合 tasks、获取 off-policy trajectory、生成 rollout、计算 loss 等。这是整个机制的核心集成点。

**修改点**:

1. **使用 Data Collate 函数混合 tasks**:
```python
# 在 fit() 方法中，Task 构建之后、rollout 之前（约第 1138 行之后）
# ⭐ 关键：Task 对象是从 batch_dict 中提取的（第 1130-1138 行）
tasks = [Task(
    task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
    query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
    env_type=self.config.env_service.env_type,
    open_query=gen_batch.non_tensor_batch["extras"][i]['open_query'],
    evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'],
    ground_truth=gen_batch.non_tensor_batch['extras'][i]['ground_truth']
) for i in range(len(gen_batch))]

# ⭐ 插入 Experience Replay 的 Task 混合逻辑
if self.config.exp_manager.experience_replay.get("enable", False):
    # 计算当前训练进度
    training_progress = self.global_steps / self.total_training_steps
    replay_start_ratio = self.config.exp_manager.experience_replay.get("replay_start_ratio", 0.35)
    enable_replay = training_progress >= replay_start_ratio
    
    if enable_replay:
        # 使用 ExperienceMixCollateFn 混合 tasks
        experience_mix_collate = ExperienceMixCollateFn(
            exp_manager=self.exp_manager,
            train_task_manager=self.train_task_manager,  # ⭐ 传入 TaskManager
            exp_ratio=self.config.exp_manager.experience_replay.get("exp_ratio", 0.5),
            replay_start_ratio=replay_start_ratio,
            offpolicy_trajectories_per_task=self.config.exp_manager.experience_replay.get("offpolicy_trajectories_per_task", 1),
            n_rollout=self.config.actor_rollout_ref.rollout.n,
        )
        
        experience_tasks, on_policy_tasks = experience_mix_collate(
            training_tasks=tasks,
            training_progress=training_progress,
            enable_replay=True,
        )
        
        # 合并 tasks
        tasks = experience_tasks + on_policy_tasks
        logger.info(
            f"Mixed batch: {len(experience_tasks)} experience tasks + "
            f"{len(on_policy_tasks)} on-policy tasks = {len(tasks)} total"
        )
    else:
        logger.info(f"Training progress {training_progress:.2%} < {replay_start_ratio}, using only training tasks")

# 继续原有的 rollout 流程（约第 1144 行）
task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="sample")
trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")
```

**详细说明**：
- **调用时机**：在每个 training step 开始时，生成 batch 之前
- **进度计算**：`training_progress = global_steps / total_training_steps`，范围 [0, 1]
- **条件判断**：
  - 如果 `training_progress < replay_start_ratio`：只使用 training tasks（积累经验阶段）
  - 如果 `training_progress >= replay_start_ratio` 且 `enable = true`：混合 training tasks 和 replay tasks
- **Task 选择逻辑**：
  1. 从 replaytaskpool 采样 `replay_task_count` 个 task_id
  2. 将 task_id 转换为 Task 对象（通过 `_get_task_by_id`）
  3. 从原始 training tasks 中选择 `(batch_size - replay_task_count)` 个
  4. 合并组成新的 batch
- **关键点**：
  - Replay tasks 和 training tasks 在后续处理中完全相同，都会生成 on-policy rollout
  - 如果 replaytaskpool 为空或采样失败，自动降级为只使用 training tasks
  - `_get_task_by_id` 需要从 task_manager 或其他地方获取 Task 对象，需要根据实际实现

2. **在 Task 混合之后获取 off-policy trajectory**:
```python
# 在 fit() 方法中，Task 混合之后、rollout 之前（约第 951 行之后）
# ⭐ 关键：需要在 Task 混合之后，才能知道哪些是 replay tasks
with _timer("get_offpolicy", timing_raw):
    # 获取 off-policy trajectory（仅对 replay tasks，从内存中获取）
    offpolicy_trajectories = []
    if enable_replay and self.config.exp_manager.experience_replay.get("enable", False):
        # 只为 replay tasks 获取 off-policy trajectory
        # ⭐ 注意：experience_tasks 是混合后的 tasks 中的前 n_exp_actual 个
        n_exp_actual = len(experience_tasks) if 'experience_tasks' in locals() else 0
        replay_tasks = tasks[:n_exp_actual] if n_exp_actual > 0 else []
        if replay_tasks:
            # 从 ExperienceManager 的内存存储中获取
            offpolicy_trajectories = self.exp_manager.get_offpolicy_batch(
                tasks=replay_tasks,
                num_trajectories_per_task=self.config.exp_manager.experience_replay.get(
                    "offpolicy_trajectories_per_task", 1
                )
            )
    
    # 将 off-policy trajectory 转换为 Linear_CMT
    if offpolicy_trajectories:
        offpolicy_cmt_array = self.env_manager.convert_offpolicy_to_cmt(
            offpolicy_trajectories=offpolicy_trajectories,
            config=self.config,
            tokenizer=self.tokenizer
        )
    else:
        offpolicy_cmt_array = []
```

**详细说明**：
- **调用时机**：在生成 on-policy rollout 之前，只为 replay tasks 获取 off-policy trajectory
- **获取方式**：从 `ExperienceManager.task2trajectories` 内存存储中获取，无需 HTTP 请求
- **获取范围**：只对 replay tasks 获取，因为只有这些 tasks 需要 off-policy 数据
- **获取数量**：每个 replay task 获取 `offpolicy_trajectories_per_task` 个历史轨迹（通常为 1）
- **转换流程**：
  1. 调用 `exp_manager.get_offpolicy_batch()` 从内存中获取 off-policy Trajectory 列表
  2. 调用 `env_manager.convert_offpolicy_to_cmt()` 转换为 Linear_CMT 格式
  3. 转换后的 `offpolicy_cmt_array` 会在后续与 on-policy 轨迹合并
- **关键点**：
  - Off-policy trajectory 已经包含完整的消息序列和 old_log_probs（从内存中获取时已包含）
  - 转换后的 Linear_CMT 对象与 on-policy 生成的格式完全相同
  - 如果获取失败（例如 task2trajectories 中没有该 task 的轨迹），`offpolicy_cmt_array` 为空，不影响主流程
- **优势**：直接从内存获取，速度快，无需网络请求

**详细说明**：
- **调用时机**：在生成 on-policy rollout 之前，只为 replay tasks 获取 off-policy trajectory
- **获取范围**：只对 replay tasks 获取，因为只有这些 tasks 需要 off-policy 数据
- **获取数量**：每个 replay task 获取 `offpolicy_trajectories_per_task` 个历史轨迹（通常为 1）
- **转换流程**：
  1. 调用 `exp_manager.get_offpolicy_batch()` 获取 off-policy Trajectory 列表
  2. 调用 `env_manager.convert_offpolicy_to_cmt()` 转换为 Linear_CMT 格式
  3. 转换后的 `offpolicy_cmt_array` 会在后续与 on-policy 轨迹合并
- **关键点**：
  - Off-policy trajectory 已经包含完整的消息序列和 old_log_probs
  - 转换后的 Linear_CMT 对象与 on-policy 生成的格式完全相同
  - 如果获取失败，`offpolicy_cmt_array` 为空，不影响主流程

2. **生成 on-policy rollout（调整 replay tasks 的数量）并混合**:
```python
# 在生成 on-policy trajectory 时（约第 1144 行）
# ⭐ 关键：需要为 replay tasks 调整 rollout_n
n_rollout = self.config.actor_rollout_ref.rollout.n  # 例如 8
offpolicy_trajectories_per_task = self.config.exp_manager.experience_replay.get(
    "offpolicy_trajectories_per_task", 1
) if enable_replay else 0

# 分离 replay tasks 和 non-replay tasks
replay_tasks = tasks[-replay_task_count:] if enable_replay and replay_task_count > 0 else []
non_replay_tasks = tasks[:-replay_task_count] if enable_replay and replay_task_count > 0 else tasks

# 为 replay tasks 和 non-replay tasks 分别生成 rollout
# 方式1：分别调用 rollout（推荐）
if replay_tasks:
    # Replay tasks: on-policy rollout_n = n_rollout - offpolicy_trajectories_per_task
    replay_task_exp_configs = self.exp_manager.get_complete_exp_configs(replay_tasks, mode="sample")
    # ⭐ 临时修改 self.env_manager.rollout_n（rollout() 方法使用 self.rollout_n，第 295 行）
    original_rollout_n = self.env_manager.rollout_n
    self.env_manager.rollout_n = n_rollout - offpolicy_trajectories_per_task
    try:
        replay_trajectories = self.env_manager.rollout(
            replay_tasks, replay_task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}"
        )
    finally:
        self.env_manager.rollout_n = original_rollout_n  # 恢复原始值（使用 try-finally 确保恢复）
else:
    replay_trajectories = []

if non_replay_tasks:
    # Non-replay tasks: on-policy rollout_n = n_rollout
    non_replay_task_exp_configs = self.exp_manager.get_complete_exp_configs(non_replay_tasks, mode="sample")
    non_replay_trajectories = self.env_manager.rollout(
        non_replay_tasks, non_replay_task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}"
    )
else:
    non_replay_trajectories = []

# 合并所有 on-policy 轨迹
trajectories = non_replay_trajectories + replay_trajectories

# 合并 on-policy 和 off-policy trajectory
if offpolicy_cmt_array:
    # 合并轨迹列表
    all_trajectories = trajectories + offpolicy_cmt_array
    logger.info(
        f"Merged {len(non_replay_trajectories)} non-replay on-policy + "
        f"{len(replay_trajectories)} replay on-policy + "
        f"{len(offpolicy_cmt_array)} off-policy trajectories"
    )
    logger.info(
        f"Replay tasks: {len(replay_trajectories)} on-policy + {len(offpolicy_cmt_array)} off-policy = "
        f"{len(replay_trajectories) + len(offpolicy_cmt_array)} total (expected: {len(replay_tasks) * n_rollout})"
    )
else:
    all_trajectories = trajectories

# 转换为 DataProto（使用相同的 to_dataproto 流程）
gen_batch_output = self.env_manager.to_dataproto(all_trajectories)
```

**详细说明**：
- **调用时机**：在生成所有 on-policy 轨迹后，将 on-policy 和 off-policy 轨迹合并
- **关键设计：混合数量控制**：
  - **Replay tasks**：
    - On-policy rollout 数量 = `n_rollout - offpolicy_trajectories_per_task`
    - Off-policy trajectory 数量 = `offpolicy_trajectories_per_task`
    - 总轨迹数 = `n_rollout`（例如：6 + 2 = 8）
  - **Non-replay tasks**：
    - On-policy rollout 数量 = `n_rollout`
    - Off-policy trajectory 数量 = 0
    - 总轨迹数 = `n_rollout`（例如：8）
- **实现方式**：
  - 在调用 `env_manager.rollout()` 之前，临时修改 `self.env_manager.rollout_n` 为 `n_rollout - offpolicy_trajectories_per_task`（仅对 replay tasks）
  - 或者分别调用 rollout，为 replay tasks 和 non-replay tasks 使用不同的 `rollout_n`
  - 确保每个 task 的总轨迹数（on-policy + off-policy）等于 `n_rollout`，便于 GRPO 分组计算
- **合并逻辑**：
  - `all_trajectories = non_replay_on_policy_trajectories + replay_on_policy_trajectories + off_policy_trajectories`
  - 使用相同的 `to_dataproto()` 流程统一转换
- **转换流程**：
  1. `trajectories_to_samples()`：调用每个 trajectory 的 `group_tokenize()`，转换为 Sample 列表
  2. `samples_to_dataproto()`：将 Sample 列表进行 padding 和 batching，转换为 DataProto
  3. 在转换过程中，根据 `is_experience_replay` 创建 `exp_mask` 和 `recorded_old_log_probs`
- **关键点**：
  - On-policy 和 off-policy 轨迹使用完全相同的转换流程，确保格式一致
  - Off-policy 轨迹的 `is_experience_replay = True`，会在 `exp_mask` 中标记
  - 合并后的 DataProto 包含所有数据，通过 `exp_mask` 区分类型
  - **每个 replay task 的总轨迹数必须等于 `n_rollout`**，这样 GRPO 才能正确分组计算 advantage

3. **计算 old_log_prob 时的处理（参考 ExGRPO 的三步流程）**:
```python
# ⭐ 步骤 1：收集经验时保存 old_log_prob（在生成轨迹后，计算 old_log_prob 之前）
# 这部分逻辑在 [8] 保存轨迹时执行，见下文

# ⭐ 步骤 2：训练时计算当前 policy 的 log_prob（约第 1218 行）
with _timer("old_log_prob", timing_raw):
    # 使用当前 policy model 计算所有样本（包括 on-policy 和 off-policy）的 log_prob
    # ⭐ 注意：这里计算的是当前策略对所有数据的 log_prob，不替换
    current_old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    batch = batch.union(current_old_log_prob)
    
    # 同时计算 entropy（用于后续选择轨迹）
    entropys = current_old_log_prob.batch["entropys"]
    
    # ⭐ 步骤 3：替换 Off-Policy 样本的 old_log_prob（如果配置了 exp_is_correct）
    # 参考 ExGRPO: replace_recorded_old_log_probs(self, batch, off_policy_mask, prefix_mask)
    if "recorded_old_log_probs" in batch.batch:
        # 检查配置：是否使用记录的 old_log_prob
        exp_is_correct = self.config.exp_manager.experience_replay.get("exp_is_correct", True)
        if exp_is_correct:
            batch = self._replace_recorded_old_log_probs(
                batch=batch,
                current_old_log_prob=current_old_log_prob
            )
            logger.info("Replaced off-policy old_log_probs with recorded ones")
        else:
            logger.info("exp_is_correct=False, using current policy's old_log_prob for all data")
    
    # 后续处理保持不变...
```

**辅助方法**:
```python
def _replace_recorded_old_log_probs(
    self,
    batch: DataProto,
    current_old_log_prob: DataProto
) -> DataProto:
    """
    替换 off-policy 数据的 old_log_prob 为记录的 historical old_log_prob。
    
    使用 exp_mask 来区分 on-policy 和 off-policy 数据：
    - exp_mask == 1: off-policy 数据，使用 recorded_old_log_probs
    - exp_mask == 0: on-policy 数据，使用当前计算的 old_log_probs
    
    Args:
        batch: 包含 exp_mask 和 recorded_old_log_probs 的 batch
        current_old_log_prob: 当前策略计算的 old_log_prob
        
    Returns:
        DataProto: 更新后的 batch
    """
    if "exp_mask" not in batch.batch or "recorded_old_log_probs" not in batch.batch:
        # 如果没有 off-policy 数据，直接使用当前计算的
        return batch.union(current_old_log_prob)
    
    exp_mask = batch.batch["exp_mask"]  # (bs, total_len)
    recorded_old_log_probs = batch.batch["recorded_old_log_probs"]  # (bs, response_len)
    current_old_log_probs = current_old_log_prob.batch["old_log_probs"]  # (bs, response_len)
    
    # 获取 response 部分的 exp_mask
    prompt_length = batch.batch["prompts"].shape[1]
    response_exp_mask = exp_mask[:, prompt_length:]  # (bs, response_len)
    
    # 对于 off-policy 数据（response_exp_mask == 1），使用记录的 old_log_prob
    # 对于 on-policy 数据（response_exp_mask == 0），使用当前计算的 old_log_prob
    old_log_probs = torch.where(
        response_exp_mask.bool(),
        recorded_old_log_probs,
        current_old_log_probs
    )
    
    # 更新 batch
    current_old_log_prob.batch["old_log_probs"] = old_log_probs
    batch = batch.union(current_old_log_prob)
    
    return batch
```

**详细说明**：
- **调用时机**：在计算 old_log_prob 时，对所有数据计算当前策略的 log 概率后调用
- **处理逻辑**（参考 ExGRPO 的实现）：
  1. **步骤 1：收集经验时保存 old_log_prob**
     - 在生成轨迹后，对于成功轨迹（reward=1），保存其 `old_log_prob`（来自当时的 policy）
     - 保存位置：`trajectory.metadata["old_log_probs"]`，同时保存 `policy_version`（global_steps）
     - 这些 `old_log_prob` 是生成经验时的旧 policy 的 log 概率
  2. **步骤 2：训练时计算当前 policy 的 log_prob**
     - 使用当前 policy model 计算所有样本（包括 on-policy 和 off-policy）的 `log_prob`
     - 这是当前策略对所有数据的 log 概率，用于计算重要性采样权重
  3. **步骤 3：替换 Off-Policy 样本的 old_log_prob**
     - 检查 batch 中是否存在 `recorded_old_log_probs`（off-policy 数据的历史 old_log_prob）
     - 检查配置 `exp_is_correct`：如果为 True，才替换 off-policy 数据的 old_log_prob
     - 如果存在且 `exp_is_correct=True`，使用 `exp_mask` 区分：
       - **Off-policy 数据**（`exp_mask == 1`）：使用 `recorded_old_log_probs`（历史策略的）
       - **On-policy 数据**（`exp_mask == 0`）：使用当前计算的 `old_log_prob`（当前策略的）
     - 合并后的 `old_log_probs` 用于后续的 loss 计算
- **为什么这样设计**：
  - **重要性采样需要 π_old 和 π_new 的 log_prob**：
    - `recorded_old_log_prob` 是生成经验时的旧 policy 的 log_prob（π_old）
    - `log_prob` 是当前 policy 的 log_prob（π_new）
    - 两者结合可计算重要性采样权重：`ratio = exp(log_prob - old_log_prob) = π_new / π_old`
  - **统一计算流程**：
    - 先对所有数据计算当前策略的 log_prob，确保计算方式一致
    - 然后只替换 off-policy 数据的 old_log_prob，保持 on-policy 数据不变
    - 这样设计简化了实现，避免了重复计算
- **关键点**：
  - **log_prob**：由当前 policy model 计算（所有样本，包括 on-policy 和 off-policy）
  - **old_log_prob**：
    - **On-policy**：由当前 policy model 计算（与 log_prob 相同，ratio ≈ 1.0）
    - **Off-policy**：如果 `exp_is_correct=True`，使用 `recorded_old_log_prob`（收集经验时保存的旧 policy 的 log_prob）；否则使用当前 policy 计算的 log_prob
  - Off-policy 数据使用历史策略的 old_log_prob，这是重要性采样的基础
  - On-policy 数据使用当前策略的 old_log_prob，这是标准 PPO 的做法
  - 通过 `torch.where()` 根据 `exp_mask` 选择使用哪个 old_log_prob

### 3.5 Loss 计算修改

**位置**: `agentevolver/module/exp_manager/het_core_algos.py`

**作用**：修改 loss 计算函数以支持重要性采样。该函数计算 on-policy 和 off-policy 的混合 loss，通过 `exp_mask` 区分数据类型，应用不同的 cliprange 和重要性采样权重。

**修改 `het_compute_token_on_off_policy_loss` 函数**:

当前实现已经支持 on-policy 和 off-policy 的区分（通过 `exp_mask`），但需要确保：
1. 使用正确的 `old_log_prob`（off-policy 使用记录的，on-policy 使用新计算的）
2. 计算重要性采样权重

**函数输入说明**：
- `old_log_prob`：已经通过 `_replace_recorded_old_log_probs` 处理过，off-policy 使用历史策略的，on-policy 使用当前策略的
- `log_prob`：当前策略对所有数据的 log 概率
- `advantages`：计算得到的优势值
- `response_mask`：标识哪些 token 是 response 部分
- `exp_mask`：标识哪些 token 是 off-policy 的（1=off-policy, 0=on-policy）

**关键修改**:
```python
def het_compute_token_on_off_policy_loss(
    old_log_prob,  # 已经替换过的 old_log_prob（off-policy 用记录的，on-policy 用新计算的）
    log_prob,      # 当前策略的 log_prob
    advantages,
    response_mask,
    exp_mask,      # 标识 off-policy 数据（1=off-policy, 0=on-policy）
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    计算 on-policy 和 off-policy 的 loss，使用重要性采样。
    
    注意：
    - old_log_prob 已经通过 _replace_recorded_old_log_probs 处理过
    - off-policy 数据的 old_log_prob 是历史策略的（记录的）
    - on-policy 数据的 old_log_prob 是当前策略的（新计算的）
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # 重要性采样权重
    # 对于 off-policy: ratio = exp(log_prob_current - old_log_prob_historical) = π_current / π_historical
    # 对于 on-policy: ratio = exp(log_prob_current - old_log_prob_current) = 1.0
    ratio = torch.exp(negative_approx_kl)
    
    def compute_pg_losses(cliprange_low, cliprange_high):
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
        clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
        return pg_losses, clipfrac, clipfrac_lower

    # On-policy calculations
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses, on_pg_clipfrac, on_pg_clipfrac_lower = compute_pg_losses(cliprange_low, cliprange_high)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)

    # Off-policy calculations (使用重要性采样)
    off_cliprange_low = cliprange_low
    off_pg_losses, off_pg_clipfrac, off_pg_clipfrac_lower = compute_pg_losses(off_cliprange_low, off_cliprange_high)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss

    # Combine on-policy and off-policy losses
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "off_pg_clipfrac": off_pg_clipfrac,
        "off_pg_clipfrac_lower": off_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
        "importance_ratio": ratio,  # 添加重要性采样权重用于监控
    }
```

**在训练循环中使用**:
```python
# 在计算 loss 时（在 update_actor 中，约第 1344 行）
# exp_mask 已经在 samples_to_dataproto 中创建，直接使用即可
exp_mask = batch.batch["exp_mask"]  # (bs, total_len)

# 获取 response 部分的 exp_mask
prompt_length = batch.batch["prompts"].shape[1]
response_exp_mask = exp_mask[:, prompt_length:]  # (bs, response_len)

loss_dict = het_compute_token_on_off_policy_loss(
    old_log_prob=batch.batch["old_log_probs"],
    log_prob=actor_output.batch["log_probs"],
    advantages=batch.batch["advantages"],
    response_mask=batch.batch["response_mask"],
    exp_mask=response_exp_mask,  # 只使用 response 部分的 mask
    cliprange=self.config.algorithm.cliprange,
    cliprange_low=self.config.algorithm.cliprange_low,
    cliprange_high=self.config.algorithm.cliprange_high,
    off_cliprange_high=self.config.exp_manager.experience_replay.get("off_cliprange_high", 1.0),
    loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
)
```

**详细说明**：
- **调用时机**：在 `update_actor` 中，计算 loss 时调用
- **exp_mask 使用**：
  - `exp_mask` 已经在 `samples_to_dataproto()` 中创建，直接使用即可
  - 需要提取 response 部分的 mask：`response_exp_mask = exp_mask[:, prompt_length:]`
- **Loss 计算**：
  - On-policy loss：使用 `cliprange_low` 和 `cliprange_high`（标准 PPO）
  - Off-policy loss：使用 `cliprange_low` 和 `off_cliprange_high`（通常更小，如 1.0）
  - 通过 `exp_mask` 分别计算两种 loss，然后合并
- **重要性采样**：
  - 对于 off-policy：`ratio = exp(log_prob_current - old_log_prob_historical)`
  - 对于 on-policy：`ratio = exp(log_prob_current - old_log_prob_current) = 1.0`
  - Ratio 用于校正 off-policy 数据的权重，确保训练稳定

### 3.6 更新 difficulty2task_dict 和保存轨迹

**在生成轨迹后更新 difficulty2task_dict 并保存轨迹**:

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py`

**作用**：在每个 training step 生成轨迹后，更新任务难度分桶，并将成功轨迹的 old_log_prob 保存到 `task2trajectories` 内存存储中，供后续作为 off-policy 数据使用。这是经验积累的关键步骤。

```python
# ⭐ 注意：这里的 trajectories 只包含 on-policy 轨迹（replay_trajectories + non_replay_trajectories）
# 不包含 off-policy 轨迹（offpolicy_cmt_array）

# 更新 difficulty2task_dict（无论是否启用 replay）
# ⭐ 关键：只统计 on-policy rollout 的结果，不包括 off-policy trajectory
if self.config.exp_manager.experience_replay.get("enable", False):
    self.exp_manager.update_difficulty2task_dict(trajectories)  # 只传入 on-policy 轨迹
    logger.info(f"Updated difficulty2task_dict: {dict(self.exp_manager.difficulty2task_dict)}")

# ⭐ 步骤 1：收集经验时保存 old_log_prob（在计算 old_log_prob 之后，约第 1227 行）
# 参考 ExGRPO: collect_experience_entries(self, batch, reward_tensor, success_value, metrics)
if self.config.exp_manager.experience_replay.get("enable", False):
    # old_log_prob 已经计算并添加到 batch
    # ⭐ 关键：使用当前计算的 old_log_prob（来自当前 policy），不是 recorded_old_log_prob
    old_log_probs = batch.batch["old_log_probs"]  # (bs, response_len)
    entropys = batch.batch.get("entropys")  # (bs, response_len) - 从 compute_log_prob 返回
    response_mask = batch.batch["response_mask"]  # (bs, response_len)
    
    # ⭐ 关键：确保 trajectories 和 batch 的顺序一致
    # rollout 返回的轨迹已按 (data_id, rollout_id) 排序（env_manager.py 第 367 行）
    # to_dataproto 转换后的 batch 顺序与轨迹顺序一致
    # 因此，batch 中前 len(trajectories) 个是 on-policy 数据，后面是 off-policy 数据
    
    # 将 old_log_prob 和 entropy 保存到 trajectory metadata
    # ⭐ 注意：batch 中包含 on-policy 和 off-policy 数据，但只保存 on-policy 轨迹（trajectories）
    for i, traj in enumerate(trajectories):
        if i < old_log_probs.shape[0]:
            # 获取该 trajectory 对应的 old_log_prob
            # ⭐ 关键：使用当前计算的 old_log_prob（来自当前 policy），这是生成经验时的 policy
            traj_old_log_prob = old_log_probs[i].cpu().numpy()
            traj_response_mask = response_mask[i].cpu().numpy()
            traj_old_log_prob = traj_old_log_prob[traj_response_mask.astype(bool)]
            
            # ⭐ 保存 old_log_prob 和 policy_version，供后续作为 recorded_old_log_prob 使用
            traj.metadata["old_log_probs"] = traj_old_log_prob.tolist()
            traj.metadata["policy_version"] = self.global_steps
    
    # ⭐ 更新 skip_uid_set 并筛选符合条件的轨迹（entropy 最低的成功轨迹）
    n_rollout = self.config.actor_rollout_ref.rollout.n
    # ⭐ 关键：entropys 需要只取 on-policy 部分（前 len(trajectories) 个）
    on_policy_entropys = entropys[:len(trajectories)] if entropys is not None else None
    on_policy_response_mask = response_mask[:len(trajectories)]
    
    filtered_trajectories = self.exp_manager.update_skip_uid_set_and_filter_trajectories(
        trajectories=trajectories,
        n_rollout=n_rollout,
        entropys=on_policy_entropys,  # ⭐ 传入 on-policy 部分的 entropy
        response_mask=on_policy_response_mask,  # ⭐ 传入 on-policy 部分的 response_mask
    )
    
    # 保存筛选后的轨迹到 ExperienceManager 的内存存储中
    if filtered_trajectories:
        self.exp_manager.save_trajectories_to_memory(filtered_trajectories)
        logger.info(
            f"Filtered and saved {len(filtered_trajectories)} trajectories to memory "
            f"(skip_uid_set size: {len(self.exp_manager.skip_uid_set)})"
        )
```

**详细说明**：
- **更新 difficulty2task_dict**：
  - 调用时机：每个 training step 生成轨迹后立即调用
  - 输入：当前 step 的所有 on-policy 轨迹（包括来自 training tasks 和 replay tasks 的）
  - 作用：根据 rollout 结果更新任务难度分桶，为后续的 replay 选择提供依据
  - 关键点：只统计 on-policy rollout 的结果，不包括 off-policy trajectory
- **保存轨迹到内存**：
  - 调用时机：在计算 old_log_prob 之后，只保存成功轨迹（reward=1）
  - 保存位置：`ExperienceManager.task2trajectories`（内存存储）
  - 保存内容：完整的 Trajectory 对象，包括 messages、reward、old_log_probs、policy_version、entropy 等
  - 作用：将成功经验保存到内存，供后续作为 off-policy 数据使用
  - **⭐ Experience 优选逻辑（参考 ExGRPO）**：
    - **第一步：筛选高 reward 轨迹**：只保存 `reward == 1.0` 的成功轨迹，确保数据质量
    - **第二步：筛选符合条件的任务**：只保存"部分成功"的任务（`experience_lbound < success_count < experience_rbound`），排除全对和全错的任务
    - **第三步：选择最优轨迹**：对于每个符合条件的任务，从所有成功轨迹中选择 **entropy 最低的轨迹**（模型最自信的成功经验）
    - **第四步：保存到内存**：将筛选后的最优轨迹保存到 `task2trajectories`，每个 task 最多保存 `max_trajectories_per_task` 个
    - **第五步：内存管理**：如果某个 task 的轨迹数量超过 `max_trajectories_per_task`，根据 `exp_select_mode` 替换：
      - `exp_select_mode="argmin"`：如果新轨迹的 entropy 更低，则替换掉当前 entropy 最高的轨迹
      - `exp_select_mode="argmax"`：如果新轨迹的 entropy 更高，则替换掉当前 entropy 最低的轨迹
      - 默认（FIFO）：删除最旧的轨迹
  - **关键点**：
    - **高 reward + 低 entropy = 最优 experience**：参考 ExGRPO 的设计，最优的 experience 是高 reward（reward=1）同时 entropy 最低的轨迹
    - **Entropy 计算**：在 `update_skip_uid_set_and_filter_trajectories` 中，计算每个成功轨迹的平均 entropy（基于 `entropys` 和 `response_mask`）
    - **保存的 old_log_prob**：是当前策略计算的，在后续训练中会作为历史策略的 old_log_prob 使用
    - **需要确保 trajectories 和 batch 的顺序一致**：才能正确匹配 old_log_prob 和 entropy
    - **使用 `max_trajectories_per_task` 限制每个 task 的轨迹数量**：避免内存无限增长
    - **Replay 时的选择**：在 `get_offpolicy_trajectories_from_memory` 中，也根据 `exp_select_mode` 选择 entropy 最低（或最高）的轨迹进行 replay

## 4. 配置项

**位置**: `config/*.yaml`

**作用**：通过配置文件控制 experience replay 的行为，包括是否启用、何时开始、混合比例等。所有配置项都有合理的默认值，便于使用和调试。

**新增配置项**:
```yaml
exp_manager:
  # ... 现有配置 ...
  
  # Experience Replay 配置
  experience_replay:
    enable: true  # 是否启用 Experience Replay
    replay_start_ratio: 0.35  # ⭐ 训练进度达到此比例时开始使用 replay（0.0-1.0）
    exp_ratio: 0.5  # ⭐ Experience tasks 的比例（0.0-1.0），默认 0.5（50%）
    offpolicy_trajectories_per_task: 1  # 每个任务获取的 off-policy 轨迹数量
    max_trajectories_per_task: 10  # ⭐ 每个 task 最多保存的轨迹数量（内存管理）
    off_cliprange_high: 1.0  # Off-policy 的 cliprange_high
    # 可选：指定从哪个难度采样
    # replay_difficulty: null  # null 表示随机选择，或指定具体难度值
    
  # ReMe 配置（现有，用于 experience-guided，不影响 experience replay）
  reme:
    base_url: "http://localhost:8001"
    workspace_id: "default"
    enable_context_generator: true
    enable_summarizer: true
```

**配置项详细说明**：
- **enable**：是否启用 Experience Replay 机制。如果为 false，所有相关功能都不会执行。
- **replay_start_ratio**：训练进度阈值（0.0-1.0）。当 `global_steps / total_training_steps >= replay_start_ratio` 时，开始使用 replay。在此之前只积累经验（更新 difficulty2task_dict 和保存轨迹到内存）。默认值 0.35。
- **exp_ratio**：Experience tasks 的比例（0.0-1.0）。例如，如果 batch_size=64，exp_ratio=0.5，则会有 32 个 experience tasks 和 32 个 on-policy tasks。默认值 0.5（50%）。
- **offpolicy_trajectories_per_task**：每个 experience task 获取的 off-policy 轨迹数量。例如，如果 n_rollout=8，offpolicy_trajectories_per_task=2，则每个 experience task 会有 2 个 off-policy + 6 个 on-policy = 8 个 total。默认值 2。
- **max_trajectories_per_task**：每个 task 最多保存的轨迹数量（内存管理）。当超过限制时，使用 FIFO 策略删除最旧的轨迹。建议设置为 5-20，根据内存情况调整。默认值 10。
- **experience_lbound**：加入 replay pool 的最小成功数。如果某个 task 的 `num_success <= experience_lbound`（全错或几乎全错），则不会加入 replay pool。默认值 0。
- **experience_rbound**：加入 replay pool 的最大成功数。如果某个 task 的 `num_success > experience_rbound`（接近全对），则不会加入 replay pool。如果为 null，则不限制。默认值 8（通常等于 n_rollout）。
- **exp_select_mode**：选择轨迹的模式。"argmin" 表示选择 entropy 最低的轨迹（模型最自信的成功经验），"argmax" 表示选择 entropy 最高的轨迹，"random" 表示随机选择。默认值 "argmin"。
- **exp_is_correct**：是否使用记录的 old_log_prob。如果为 true，off-policy 数据使用 `recorded_old_log_prob`（收集经验时保存的旧 policy 的 log_prob）；如果为 false，使用当前 policy 计算的 old_log_prob（等同于 on-policy）。默认值 true。
- **off_cliprange_high**：Off-policy 数据的 cliprange_high，通常设置为 1.0（比 on-policy 的 cliprange_high 更小），用于稳定重要性采样。默认值 1.0。
- **replay_difficulty**（可选）：指定从哪个难度采样。如果为 null，随机选择一个难度；如果指定具体值，从该难度的桶中采样。默认值 null（随机选择）。

**配置示例**：
```yaml
# 示例：batch_size=64, n_rollout=8, exp_ratio=0.5, offpolicy_trajectories_per_task=2
# 结果：
# - Experience tasks: 32 个，每个 task 有 2 个 off-policy + 6 个 on-policy = 8 个 total
# - On-policy tasks: 32 个，每个 task 有 8 个 on-policy
# - 总 rollouts = 32 * 8 + 32 * 8 = 512 = 64 * 8 ✓
```

## 5. 实现步骤

### Phase 1: 基础框架
1. ✅ 在 `ExperienceManager` 中：
   - 添加 `difficulty2task_dict` 属性（按难度分桶存储 task_id）
   - 添加 `task2trajectories` 属性（按 task_id 存储 Trajectory 列表）
   - 实现 `update_difficulty2task_dict` 方法
   - 实现 `save_trajectories_to_memory` 方法（保存轨迹到内存）
   - 实现 `get_offpolicy_trajectories_from_memory` 方法（从内存获取轨迹）
   - 实现 `sample_tasks_from_replaypool` 方法
   - 添加 `get_offpolicy_batch` 方法（从内存获取）
2. ✅ 在 `ParallelEnvManager` 中实现 `convert_offpolicy_to_cmt` 方法
3. ✅ 修改 `get_extra` 以支持 `is_experience_replay` 字段
4. ✅ 修改 `samples_to_dataproto` 以支持 off-policy 数据的 `exp_mask` 和 `recorded_old_log_probs`

### Phase 2: 训练循环集成
4. ✅ 在训练循环中实现动态 task 选择（training tasks + replay tasks）
5. ✅ 实现 `replay_start_ratio` 控制逻辑
6. ✅ 在生成轨迹后更新 `difficulty2task_dict`
7. ✅ 在训练循环中获取 off-policy trajectory（仅对 replay tasks）
8. ✅ 实现 batch 合并逻辑
9. ✅ 实现 `_replace_recorded_old_log_probs` 方法

### Phase 3: Loss 计算
7. ✅ 修改 loss 计算函数以支持重要性采样
8. ✅ 在训练循环中使用 prefix_mask 区分 on/off-policy

### Phase 4: 数据保存
9. ✅ 实现保存轨迹到 `task2trajectories` 的逻辑（只保存 reward=1 的轨迹）
10. ✅ 在生成新轨迹时保存轨迹到内存
11. ✅ 实现内存管理逻辑（FIFO 策略，限制每个 task 的轨迹数量）
12. ✅ 实现 `_get_task_by_id` 辅助方法（用于从 task_id 获取 Task 对象）

### Phase 5: 测试和优化
11. ✅ 单元测试
12. ✅ 集成测试
13. ✅ 性能优化

## 6. 关键注意事项

**作用**：列出实现和使用 experience replay 机制时需要注意的关键点，包括数据流、边界情况、性能考虑等。这些注意事项有助于正确实现和调试。

### 6.0 Multi-turn 场景的特殊处理

**背景**：与 ExGRPO 的 single-turn 数学推理任务不同，AgentEvolver 处理的是 multi-turn 任务（如 ALFworld），其中 LLM 会与 Environment/User 进行多轮对话完成任务。

**Multi-turn 场景的关键特点**：
1. **Response 部分包含多轮交互**：一个 trajectory 的 response 部分包含多轮 LLM-Environment 交替对话
2. **只有 LLM 响应需要计算 loss**：Environment 响应不参与 loss 计算
3. **loss_mask 标记 LLM 响应**：在 multi-turn 中，`loss_mask` 只对 LLM 响应位置设置为 1

**关键实现细节**：

#### 6.0.1 response_mask 的来源（het_actor.py:127-130）

```python
if multi_turn:
    response_mask = data["loss_mask"][:, -response_length:]  # 使用 loss_mask
else:
    response_mask = attention_mask[:, -response_length:]  # 使用 attention_mask
```

在 multi-turn 场景中，`response_mask` 基于 `loss_mask`，确保只有 LLM 响应参与 loss 计算。

#### 6.0.2 exp_mask 的创建（env_manager.py:610-620）

```python
if sample.extras.get("is_experience_replay", False):
    # ⭐ Multi-turn 关键：使用 response_loss_mask 而不是全 1
    # 确保只标记 LLM 响应为 off-policy
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))
```

`exp_mask` 只对 LLM 响应位置设置为 1，Environment 响应位置为 0。

#### 6.0.3 old_log_probs 的保存（ae_ray_trainer.py:1407-1416）

```python
# ⭐ Multi-turn 关键：保存完整的 response 部分的 old_log_probs（不过滤）
traj.metadata["old_log_probs"] = traj_old_log_prob.tolist()
traj.metadata["response_mask"] = traj_response_mask.tolist()  # 保存 mask 用于后续对齐
```

保存完整的 `old_log_probs`（不过滤），同时保存 `response_mask` 用于后续对齐。

#### 6.0.4 与 ExGRPO 的对比

| 特性 | ExGRPO (Single-turn) | AgentEvolver (Multi-turn) |
|------|---------------------|---------------------------|
| Response 结构 | 单轮 LLM 响应 | 多轮 LLM-Environment 交替 |
| response_mask | `attention_mask[:, -response_length:]` | `loss_mask[:, -response_length:]` |
| exp_mask | 整个 response 部分为 1 | 只有 LLM 响应位置为 1 |
| old_log_probs | 整个 response | 完整保存（不过滤） |

### 6.0.5 Difficulty2Task 机制

#### 6.0.1 Difficulty 计算
- **定义**：在一个 training step 中，某个 task 的 n 次 rollout 中，reward=1 的次数
- **示例**：task_id='2' 在当前 step 有 8 次 rollout，其中 3 次 reward=1，则 difficulty=3
- **更新时机**：每个 training step 生成轨迹后立即更新

#### 6.0.2 ReplayTaskPool 维护
- `difficulty2task_dict` 按难度分桶存储 task_id
- 同一个 task 可能在不同 step 有不同的 difficulty，会被添加到对应的桶中
- 如果 task 已经在某个 difficulty 桶中，不会重复添加（但可以跨难度存在）

#### 6.0.3 Replay Start Ratio
- **作用**：控制何时开始使用 replay
- **阶段 1**（training_progress < replay_start_ratio）：
  - 只更新 `difficulty2task_dict`
  - 只保存 reward=1 的轨迹到 `task2trajectories` 内存存储
  - 不进行 replay（积累经验阶段）
- **阶段 2**（training_progress >= replay_start_ratio）：
  - 从 replaytaskpool 选择 tasks
  - 获取这些 tasks 的 off-policy trajectory
  - 与 on-policy 数据混合训练

#### 6.0.4 Task 选择策略
- **混合比例**：从 training tasks 中选择 `(original_batch_size - replay_task_count)` 个
- **Replay tasks**：从 replaytaskpool 中选择 `replay_task_count` 个
- **采样策略**：可以随机选择难度，或指定特定难度

**详细说明**：
- **Difficulty 计算示例**：
  - Step 100：task_2 有 8 次 rollout，其中 3 次 reward=1 → difficulty=3
  - Step 200：task_2 有 8 次 rollout，其中 5 次 reward=1 → difficulty=5
  - 结果：`difficulty2task_dict[3] = ['task_2']`，`difficulty2task_dict[5] = ['task_2']`（同一个 task 可以在不同难度桶中）
- **ReplayTaskPool 维护**：
  - `difficulty2task_dict` 在整个训练过程中持续更新
  - 同一个 task 在不同 step 可能有不同的 difficulty，会被添加到对应的桶中
  - 如果 task 已经在某个难度的桶中，不会重复添加（但可以跨难度存在）
  - 随着训练进行，pool 会逐渐积累不同难度的任务
- **Replay Start Ratio 的作用**：
  - **阶段 1**（training_progress < replay_start_ratio）：
    - 只更新 `difficulty2task_dict`，积累任务难度信息
    - 只保存 reward=1 的轨迹到 ReMe，积累成功经验
    - 不进行 replay，因为 pool 还不够丰富
  - **阶段 2**（training_progress >= replay_start_ratio）：
    - 从 replaytaskpool 选择 tasks，开始使用历史经验
    - 获取这些 tasks 的 off-policy trajectory
    - 与 on-policy 数据混合训练
- **Task 选择策略**：
  - 混合比例确保 batch size 保持不变
  - 可以随机选择难度，也可以指定特定难度（例如只选择 difficulty=3 的任务）
  - 允许重复采样（同一个 task 可能被采样多次）

### 6.1 Experience 优选逻辑（参考 ExGRPO）

**核心原则**：最优的 experience 是**高 reward（reward=1）同时 entropy 最低**的轨迹。

**详细说明**：

#### 6.1.1 为什么选择高 Reward + 低 Entropy？

参考 ExGRPO 的设计，最优的 experience 应该满足两个条件：
1. **高 Reward（reward=1）**：确保是成功的经验，模型能够正确完成任务
2. **低 Entropy**：表示模型对生成结果最自信，不确定性最低

**Entropy 的含义**：
- Entropy 衡量模型输出的不确定性
- 低 entropy 表示模型对生成的 token 概率分布更集中（更自信）
- 高 entropy 表示模型对生成的 token 概率分布更分散（更不确定）

**为什么选择低 Entropy 的成功轨迹**：
- 这些轨迹是模型最自信的成功经验，质量更高
- 在 replay 时，这些轨迹能够提供更稳定、更可靠的学习信号
- 有助于模型更快地学习到正确的行为模式

#### 6.1.2 Experience 优选流程

**步骤 1：筛选高 Reward 轨迹**
- 只考虑 `reward == 1.0` 的成功轨迹
- 排除所有失败的轨迹（reward=0）

**步骤 2：筛选符合条件的任务**
- 只保存"部分成功"的任务：`experience_lbound < success_count < experience_rbound`
- 排除全对的任务（`success_count == n_rollout`）：这些任务已经掌握，加入 `skip_uid_set`
- 排除全错的任务（`success_count <= experience_lbound`）：这些任务太难，不适合 replay

**步骤 3：计算 Entropy**
- 对于每个成功轨迹，计算其平均 entropy：
  ```python
  traj_entropys = entropys[i].cpu().numpy()  # (response_len,)
  traj_response_mask = response_mask[i].cpu().numpy()  # (response_len,)
  valid_entropys = traj_entropys[traj_response_mask.astype(bool)]
  avg_entropy = np.mean(valid_entropys)  # 平均 entropy
  traj.metadata["entropy"] = avg_entropy  # 保存到 metadata
  ```

**步骤 4：选择最优轨迹**
- 对于每个符合条件的任务，从所有成功轨迹中选择 entropy 最低的轨迹：
  ```python
  successful_trajs_with_entropy = [
      (e, t) for e, t in task_id_to_entropy_list[task_id] 
      if t.reward.outcome == 1.0
  ]
  if exp_select_mode == "argmin":
      best_traj = min(successful_trajs_with_entropy, key=lambda x: x[0])[1]  # 选择 entropy 最低的
  elif exp_select_mode == "argmax":
      best_traj = max(successful_trajs_with_entropy, key=lambda x: x[0])[1]  # 选择 entropy 最高的
  ```

**步骤 5：保存到内存**
- 将筛选后的最优轨迹保存到 `task2trajectories[task_id]`
- 每个 task 最多保存 `max_trajectories_per_task` 个轨迹

**步骤 6：内存管理**
- 如果某个 task 的轨迹数量超过 `max_trajectories_per_task`，根据 `exp_select_mode` 替换：
  - `exp_select_mode="argmin"`：如果新轨迹的 entropy 更低，则替换掉当前 entropy 最高的轨迹
  - `exp_select_mode="argmax"`：如果新轨迹的 entropy 更高，则替换掉当前 entropy 最低的轨迹
  - 默认（FIFO）：删除最旧的轨迹

#### 6.1.3 Replay 时的轨迹选择

在 `get_offpolicy_trajectories_from_memory` 中，也根据 `exp_select_mode` 选择轨迹：
```python
if self.exp_select_mode == "argmin":  # 选择 entropy 最低的
    available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('inf')))
elif self.exp_select_mode == "argmax":  # 选择 entropy 最高的
    available_trajectories.sort(key=lambda t: t.metadata.get("entropy", float('-inf')), reverse=True)
# 然后采样 num_trajectories 个轨迹
```

**总结**：
- **保存时**：选择高 reward + 低 entropy 的轨迹
- **Replay 时**：也选择低 entropy 的轨迹（如果 `exp_select_mode="argmin"`）
- **一致性**：保存和 replay 都遵循相同的优选逻辑，确保使用最高质量的 experience

### 6.2 Off-policy 数据处理
- Off-policy trajectory 通过相同的 `group_tokenize()` 流程转换为 Sample
- Off-policy 数据的 LLM 消息保持 `author="llm"`，确保 `loss_mask=1`，参与 off-policy loss 计算
- 通过 `exp_mask=1` 标识 off-policy 数据，在 `het_compute_token_on_off_policy_loss` 中分别计算 on/off-policy loss
- 使用 historical old_log_prob 进行重要性采样

**详细说明**：
- **数据转换**：Off-policy trajectory 通过完全相同的 `group_tokenize()` 流程转换为 Sample，确保格式一致
- **Loss Mask 处理**：
  - Off-policy 数据的 LLM 消息保持 `author="llm"`，确保 `loss_mask=1`
  - **不使用 `loss_mask=0` 来区分 off-policy**，而是使用独立的 `exp_mask`
  - 这与 ExGRPO 的设计一致，off-policy 数据参与 loss 计算
- **Log Prob 计算**：模型在 off-policy 数据上计算 log_prob 的方式与 on-policy 完全相同，都是计算当前策略对响应的概率
- **Old Log Prob 使用**：
  - On-policy：使用当前策略计算的 old_log_prob（标准 PPO）
  - Off-policy：使用历史策略的 old_log_prob（从 `task2trajectories` 获取的 recorded_old_log_probs）
  - 这个区别是重要性采样的基础
- **Exp Mask 的作用**：
  - 标识哪些 token 是 off-policy 的（`exp_mask=1`）
  - 在 `het_compute_token_on_off_policy_loss` 中：
    - `on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)` - on-policy
    - `off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)` - off-policy
  - 两者合并：`pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)`
  - off-policy 数据使用 `off_cliprange_high`（通常更小），稳定训练

### 6.4 old_log_prob 对齐
- 历史策略的 old_log_prob 长度可能与当前 response 长度不一致
- 在 `samples_to_dataproto()` 中，通过 padding 对齐到 batch 的 max_response_length
- 对于长度不匹配的情况：
  - 如果历史 old_log_prob 更长：截断到 response_length
  - 如果历史 old_log_prob 更短：用 0 填充（这些位置会使用当前策略的 old_log_prob）
- 如果没有记录的 old_log_prob，使用当前策略计算的 old_log_prob（等同于 on-policy）

**详细说明**：
- **长度不匹配的原因**：
  - 历史轨迹的 response 长度可能与当前轨迹不同
  - 不同策略版本生成的响应长度可能不同
  - Tokenization 的差异也可能导致长度不一致
- **对齐策略**：
  - 在 `samples_to_dataproto()` 中，先将每个 Sample 的 old_log_probs 对齐到其 response_length
  - 然后通过 padding 对齐到 batch 的 max_response_length
  - 确保所有数据的 old_log_probs 具有相同的形状
- **填充处理**：
  - 如果历史 old_log_prob 更长：截断到 response_length，丢弃多余部分
  - 如果历史 old_log_prob 更短：用 0 填充，这些位置在后续会被替换为当前策略的 old_log_prob
- **缺失处理**：如果 off-policy trajectory 没有 old_log_probs（例如旧数据），使用当前策略计算的 old_log_prob，等同于 on-policy 处理

### 6.5 重要性采样稳定性
- 重要性采样权重 `ratio = exp(log_prob - old_log_prob)` 可能非常大或非常小
- 需要添加 clipping 来稳定训练
- 可以添加监控指标来观察 importance ratio 的分布

**详细说明**：
- **Ratio 不稳定的原因**：
  - 如果当前策略与历史策略差异很大，`log_prob - old_log_prob` 可能很大
  - `exp()` 函数会放大这种差异，导致 ratio 非常大或非常小
  - 这会导致 off-policy loss 不稳定，影响训练
- **稳定化方法**：
  - 使用 `off_cliprange_high`（通常为 1.0）限制 ratio 的上界
  - 在 PPO loss 计算中，对 ratio 进行 clipping：`torch.clamp(ratio, 1 - cliprange_low, 1 + off_cliprange_high)`
  - 这确保了 off-policy loss 不会因为 ratio 过大而爆炸
- **监控建议**：
  - 记录 `importance_ratio_mean`、`importance_ratio_max`、`importance_ratio_min`
  - 如果 ratio 持续很大或很小，说明策略变化太快，可能需要调整学习率或 cliprange

### 6.6 数据质量
- 只使用高质量的 off-policy trajectory（例如 reward > threshold）
- 考虑 trajectory 的年龄（policy_version 差异）
- 可以添加过滤机制

**详细说明**：
- **质量筛选**：
  - 当前设计只保存 `reward == 1.0` 的成功轨迹，确保数据质量
  - 在获取 off-policy trajectory 时，可以进一步过滤（例如只获取 reward > 0.8 的）
  - 低质量的轨迹可能导致负面的训练效果
- **策略版本考虑**：
  - `policy_version` 记录了轨迹生成时的策略版本（global_steps）
  - 如果策略版本太旧（例如相差很多 steps），策略差异可能太大，重要性采样可能不稳定
  - 可以考虑只使用较新的轨迹（例如 policy_version 在最近 N 个 steps 内）
- **过滤机制**：
  - 在 `get_offpolicy_trajectory` 中可以添加过滤逻辑
  - 例如：只获取 `policy_version >= (current_steps - max_age)` 的轨迹
  - 或者：只获取 `reward >= min_reward_threshold` 的轨迹

### 6.7 GRPO 计算考虑
- GRPO 基于 `group_ids`（通常是 `data_id`）进行分组计算 advantage
- Off-policy 数据应该使用独立的 `data_id`，避免与 on-policy 数据混合分组
- 在 `convert_offpolicy_to_cmt()` 中，为 off-policy trajectory 分配唯一的 `data_id`（整数格式，例如：`1000000 + task_id_int * 1000 + index`）
- 这样 GRPO 会为 off-policy 数据单独计算 advantage，不会影响 on-policy 数据的 advantage 计算
- **关键：总轨迹数必须等于 n_rollout**：
  - 对于 replay tasks：on-policy 数量 + off-policy 数量 = `n_rollout`
  - 对于 non-replay tasks：on-policy 数量 = `n_rollout`
  - 这确保了 GRPO 分组时，每个 task 的总轨迹数一致，advantage 计算正确

**详细说明**：
- **GRPO 分组机制**：
  - GRPO（Group Relative Policy Optimization）基于 `group_ids`（通常是 `data_id`）进行分组
  - 同一组内的轨迹共享相同的 advantage 计算（基于组内的 reward 分布）
  - 不同组之间的 advantage 计算是独立的
- **为什么需要独立分组**：
  - Off-policy 数据来自历史策略，其 reward 分布可能与当前 on-policy 数据不同
  - 如果混合分组，off-policy 数据的 reward 会影响 on-policy 数据的 advantage 计算
  - 这会导致 advantage 计算不准确，影响训练效果
- **实现方式**：
  - 在 `convert_offpolicy_to_cmt()` 中，为每个 off-policy trajectory 分配唯一的 `data_id`
  - 格式：使用整数格式（例如：`1000000 + task_id_int * 1000 + traj_index`），确保可以转换为整数
  - **重要**：现有代码要求 `data_id` 必须是整数或可以转换为整数（`int(s.data_id)`）
  - 使用大偏移量（1000000）确保 off-policy data_id 不会与 on-policy data_id 冲突
  - 这样每个 off-policy trajectory 都是独立的一组，不会与其他数据混合
- **优势**：
  - On-policy 数据的 advantage 计算不受 off-policy 数据影响
  - Off-policy 数据的 advantage 基于其自身的 reward 分布计算
  - 两种数据的 advantage 计算都是准确的

### 6.8 性能考虑
- 内存管理：使用 `max_trajectories_per_task` 限制每个 task 的轨迹数量
- Batch 大小会增加，需要确保内存足够
- 可以限制每个 batch 中 off-policy 数据的比例，避免过度增加 batch size

**详细说明**：
- **内存管理**：
  - 使用 `task2trajectories` 在内存中存储轨迹，访问速度快
  - 通过 `max_trajectories_per_task` 限制每个 task 的轨迹数量，避免内存无限增长
  - 使用 FIFO 策略删除最旧的轨迹，保持内存占用可控
  - 建议 `max_trajectories_per_task` 设置为 5-20，根据内存情况调整
- **内存占用估算**：
  - 每个 Trajectory 包含 messages、reward、old_log_probs 等
  - 假设每个轨迹平均 1000 tokens，old_log_probs 为 float32，每个轨迹约占用 4KB
  - 如果有 100 个 tasks，每个 task 保存 10 个轨迹，总内存约 4MB（可接受）
- **获取速度**：
  - 从内存获取 off-policy trajectory 非常快，无需 HTTP 请求
  - 相比从 ReMe 获取，速度提升显著
- **Batch 大小**：
  - 混合 on-policy 和 off-policy 数据会增加 batch size
  - 如果 `offpolicy_trajectories_per_task > 1`，batch size 会进一步增加
  - 需要确保 GPU 内存足够，或者限制 off-policy 数据的比例
- **比例控制**：
  - 通过 `replay_task_count` 和 `offpolicy_trajectories_per_task` 控制 off-policy 数据的比例
  - 建议 off-policy 数据不超过 batch 的 30-50%，确保训练稳定

## 7. 监控指标

**作用**：监控 experience replay 机制的运行状态，包括数据比例、重要性采样权重、loss 对比等。这些指标有助于调试和优化训练过程。

建议添加以下监控指标：

1. **Off-policy 数据比例**: `offpolicy_ratio = (exp_mask == 1).sum() / exp_mask.numel()`
2. **重要性采样权重统计**: `importance_ratio_mean`, `importance_ratio_max`, `importance_ratio_min`
3. **On/Off-policy Loss 对比**: `on_pg_loss` vs `off_pg_loss`
4. **Old Log Prob 差异**: 比较记录的 old_log_prob 和当前计算的 old_log_prob 的差异

**详细说明**：
- **Off-policy 数据比例**：
  - 监控每个 batch 中 off-policy 数据的占比
  - 如果比例过高（>50%），可能影响训练稳定性
  - 如果比例过低（<10%），可能 replay 效果不明显
- **重要性采样权重统计**：
  - `importance_ratio_mean`：平均权重，应该接近 1.0
  - `importance_ratio_max`：最大权重，如果过大（>10），说明策略变化太快
  - `importance_ratio_min`：最小权重，如果过小（<0.1），说明策略变化太大
  - 这些指标有助于判断重要性采样是否稳定
- **Loss 对比**：
  - 比较 on-policy 和 off-policy 的 loss 大小
  - 如果 off-policy loss 远大于 on-policy loss，可能需要调整 `off_cliprange_high`
  - 如果 off-policy loss 持续为 0 或 NaN，说明 off-policy 数据处理有问题
- **Old Log Prob 差异**：
  - 比较记录的 old_log_prob 和当前计算的 old_log_prob 的差异
  - 如果差异很大，说明策略变化很大，重要性采样可能不稳定
  - 可以用于判断是否需要调整学习率或 cliprange
- **Difficulty2Task 统计**：
  - 监控 `difficulty2task_dict` 的大小和分布
  - 例如：每个难度桶中有多少个 task
  - 这有助于了解 replaytaskpool 的丰富程度

## 8. 与现有机制的兼容性

**作用**：说明 Experience Replay 与现有机制（特别是 Experience-Guided）的兼容性，确保两种机制可以同时使用而不冲突。

- **Experience Replay 与 Experience-Guided 可以同时使用**：
  - Experience-Guided：通过 `add_exp` 和 `task_train_expmode` 控制，在 prompt 中插入经验
  - Experience Replay：通过 `is_experience_replay` 标识，作为训练数据混合
  - 两者通过不同的 `exp_mask` 逻辑区分（见 `samples_to_dataproto` 中的实现）
- **可以通过配置项控制是否启用**：
  - `exp_manager.experience_replay.enable` 控制 Experience Replay
  - `exp_manager.reme.enable_context_generator` 控制 Experience-Guided
- **如果 `task2trajectories` 中没有 off-policy trajectory，自动降级为纯 on-policy 训练**
- **不影响现有的 experience-guided 功能**：
  - `get_extra()` 中同时支持 `add_exp` 和 `is_experience_replay`
  - `samples_to_dataproto()` 中的 `exp_mask` 逻辑兼容两种机制

**详细说明**：
- **同时使用的场景**：
  - 一个 trajectory 可以同时使用两种机制
  - 例如：在 prompt 中插入 experience（Experience-Guided），同时使用 off-policy trajectory 作为训练数据（Experience Replay）
  - 两种机制通过不同的字段和逻辑区分，不会冲突
- **Exp Mask 的优先级**：
  - 在 `samples_to_dataproto()` 中，先检查 `is_experience_replay`
  - 如果是 experience replay，response 部分标记为 1
  - 如果不是，再检查 `add_exp` 和 `task_train_expmode`
  - 这确保了两种机制的 exp_mask 逻辑不会冲突
- **降级机制**：
  - 如果 `task2trajectories` 中没有 off-policy trajectory，`get_offpolicy_batch()` 返回空列表
  - 训练流程会自动降级为纯 on-policy 训练，不影响主流程
  - 这确保了机制的健壮性
- **向后兼容**：
  - 如果 `experience_replay.enable = false`，所有相关代码都不会执行
  - 现有的 experience-guided 功能完全不受影响
  - 这确保了向后兼容性

## 9. 未来扩展

1. **自适应混合比例**: 根据训练进度动态调整 on/off-policy 比例
2. **Trajectory 选择策略**: 基于相似度、奖励等选择最有价值的 off-policy trajectory
3. **多策略融合**: 支持从多个历史策略版本获取 trajectory
4. **异步 Experience Replay**: 在后台异步获取和准备 off-policy 数据

## 10. 参考资料

- PPO with Importance Sampling
- Off-Policy Reinforcement Learning
- Experience Replay in RL
- Prefix-LM Training

