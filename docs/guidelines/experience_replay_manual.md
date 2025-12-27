# Experience Replay with GRPO - 完整流程 Manual

本文档详细 walk through 使用 GRPO 进行 Experience Replay 训练的完整流程，包括每个步骤的代码位置、数据流和关键细节。

## 目录

1. [概述](#概述)
2. [初始化阶段](#初始化阶段)
3. [训练循环 - 完整流程](#训练循环---完整流程)
4. [关键数据结构](#关键数据结构)
5. [常见问题排查](#常见问题排查)

---

## 概述

Experience Replay 机制允许模型复用历史成功轨迹（off-policy data）来提高样本效率。结合 GRPO（Group Relative Policy Optimization），我们可以：

1. **复用历史经验**：将过去成功的轨迹作为 off-policy 数据参与训练
2. **Off-policy Loss 计算**：Off-policy 轨迹参与 loss 计算，使用 `exp_mask` 区分 on/off-policy
3. **重要性采样**：使用 `ratio = exp(log_prob_current - old_log_prob_historical)` 来校正 off-policy 数据的权重
4. **混合训练**：在同一个 batch 中混合 on-policy 和 off-policy 数据

### Multi-turn 场景的特殊处理

与 ExGRPO 的 single-turn 数学推理任务不同，AgentEvolver 处理的是 **multi-turn 任务**（如 ALFworld），其中 LLM 会与 Environment/User 进行多轮对话完成任务。

**关键区别**：
- **Response 结构**：包含多轮 LLM-Environment 交替对话
- **response_mask**：在 multi-turn 中，基于 `loss_mask` 而不是 `attention_mask`
- **exp_mask**：只对 LLM 响应位置设置为 1，而不是整个 response
- **old_log_probs**：保存完整的 response 部分（不过滤），确保位置对齐

### 轨迹数量控制

**设计原则**：每个 task 的总轨迹数量恒定为 `n_rollout`，确保 GRPO 分组均衡。

**实现方式**：
- Experience task：`n_exp` 条 off-policy + `(n_rollout - n_exp)` 条 on-policy = `n_rollout` 条
- On-policy task：`n_rollout` 条 on-policy = `n_rollout` 条

**示例**（`n_rollout=8`, `offpolicy_trajectories_per_task=2`）：
- Experience task：2 off-policy + 6 on-policy = 8 条总轨迹
- On-policy task：8 on-policy = 8 条总轨迹

---

## 初始化阶段

### 1.1 Trainer 初始化

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:306` - `AgentEvolverRayPPOTrainer.__init__()`

**关键步骤**:

```python
# 1. 初始化 ExperienceManager
self.exp_manager = ExperienceManager(self.config.exp_manager)

# 2. 加载 experience pool（如果存在）
if exp_replay_config.get("enable", False):
    exp_pool_base_dir = os.path.join(self.config.trainer.default_local_dir, "experience_pool")
    if os.path.exists(exp_pool_base_dir):
        # 找到最新的 step 目录并加载
        self.exp_manager.load_experience_pool_from_disk(exp_pool_load_dir)
```

**数据结构初始化**:

- `ExperienceManager.difficulty2task_dict`: `Dict[int, List[str]]` - 按成功次数（难度）分组的 task_id
- `ExperienceManager.task2trajectories`: `Dict[str, List[Trajectory]]` - 每个 task 的轨迹列表
- `ExperienceManager.skip_uid_set`: `Set[str]` - 已完全解决的 task_id（n_rollout 全部成功）

**配置项**:

```yaml
exp_manager:
  experience_replay:
    enable: true
    exp_ratio: 0.5  # Experience tasks 的比例
    replay_start_ratio: 0.35  # 训练进度达到 35% 时开始 replay
    offpolicy_trajectories_per_task: 1  # 每个任务获取的 off-policy 轨迹数
    experience_lbound: 0  # 成功次数下界（包含）
    experience_rbound: 8  # 成功次数上界（不包含，通常是 n_rollout）
    exp_select_mode: "argmin"  # 轨迹选择模式：argmin（最低 entropy）、argmax、random
    exp_is_correct: true  # 是否使用 recorded_old_log_probs
    max_trajectories_per_task: 5  # 每个任务最多保存的轨迹数
```

---

## 训练循环 - 完整流程

### 2.1 准备 Training Tasks

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1206-1214`

**步骤**:

1. 从 `gen_batch` 中提取 tasks：
   ```python
   tasks = [Task(
       task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
       query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
       # ... 其他属性
   ) for i in range(len(gen_batch))]
   ```

2. **关键点**:
   - `tasks` 是原始的 on-policy tasks，数量 = `batch_size`
   - 此时还没有进行 Experience Replay 混合

---

### 2.2 Experience Replay 混合 Tasks

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1216-1259`

**详细流程**:

#### Step 1: 检查是否启用 Experience Replay

```python
exp_replay_config = self.config.exp_manager.get("experience_replay", {})
enable_exp_replay = exp_replay_config.get("enable", False)
training_progress = self.global_steps / self.total_training_steps
replay_start_ratio = exp_replay_config.get("replay_start_ratio", 0.35)
```

**条件**:
- `enable_exp_replay == True`
- `training_progress >= replay_start_ratio`

如果条件不满足，跳过 Experience Replay，直接使用原始 `tasks`。

#### Step 2: 使用 ExperienceMixCollateFn 混合 Tasks

```python
experience_mix_collate = ExperienceMixCollateFn(
    exp_manager=self.exp_manager,
    train_task_manager=self.train_task_manager,
    exp_ratio=exp_replay_config.get("exp_ratio", 0.5),
    replay_start_ratio=replay_start_ratio,
    offpolicy_trajectories_per_task=exp_replay_config.get("offpolicy_trajectories_per_task", 1),
    n_rollout=self.config.actor_rollout_ref.rollout.n,
)

experience_tasks, on_policy_tasks = experience_mix_collate(
    training_tasks=tasks,
    training_progress=training_progress,
    enable_replay=True,
)
```

**ExperienceMixCollateFn 内部逻辑** (`agentevolver/module/exp_manager/experience_collate.py:49-118`):

1. **计算目标 experience task 数量**:
   ```python
   target_exp_count = int(batch_size * self.exp_ratio)  # 例如：64 * 0.5 = 32
   ```

2. **从 replaytaskpool 采样 experience task_ids**:
   ```python
   valid_exp_task_ids = self.exp_manager.get_valid_replay_task_ids()  # 排除 skip_uid_set
   n_exp = min(len(valid_exp_task_ids), target_exp_count)
   sampled_exp_task_ids = random.sample(valid_exp_task_ids, n_exp)
   ```

3. **转换为 Task 对象**:
   ```python
   experience_tasks = []
   for task_id in sampled_exp_task_ids:
       task = self._get_task_by_id(task_id)  # 从 train_task_manager 获取
       if task is not None:
           experience_tasks.append(task)
   ```

4. **补充 on-policy tasks**:
   ```python
   n_on_policy = batch_size - len(experience_tasks)  # 确保总数为 batch_size
   on_policy_tasks = training_tasks[:n_on_policy]
   ```

**结果**:
- `experience_tasks`: 从 replaytaskpool 选择的 tasks（需要获取 off-policy trajectories）
- `on_policy_tasks`: 纯 on-policy tasks（不需要 off-policy trajectories）
- `len(experience_tasks) + len(on_policy_tasks) == batch_size`

#### Step 3: 获取 Off-policy Trajectories

```python
if experience_tasks:
    # ⭐ ExGRPO 设计：使用当前 policy 计算 entropy 选择最优轨迹
    use_current_policy_entropy = exp_replay_config.get("use_current_policy_entropy", True)
    num_trajectories_per_task = exp_replay_config.get("offpolicy_trajectories_per_task", 1)
    
    if use_current_policy_entropy:
        # 获取所有候选轨迹
        task_to_candidates = self.exp_manager.get_all_candidates_batch(tasks=experience_tasks)
        # 使用当前 policy 计算 entropy 选择最优轨迹
        offpolicy_trajectories = self._select_best_offpolicy_by_current_entropy(
            task_to_candidates=task_to_candidates,
            tasks=experience_tasks,
            num_trajectories_per_task=num_trajectories_per_task,
        )
    else:
        # 使用保存时的 entropy 选择轨迹
        offpolicy_trajectories = self.exp_manager.get_offpolicy_batch(
            tasks=experience_tasks,
            num_trajectories_per_task=num_trajectories_per_task,
            use_saved_entropy=True
        )
    
    if offpolicy_trajectories:
        # ⭐ ExGRPO 设计：构建 task_id 到 data_id 的映射
        # 确保 off-policy trajectory 使用与对应 on-policy trajectory 相同的 data_id
        # tasks = experience_tasks + on_policy_tasks，experience_tasks 在前面
        task_id_to_data_id = {
            task.task_id: idx
            for idx, task in enumerate(tasks)
        }
        
        offpolicy_cmt_array = self.env_manager.convert_offpolicy_to_cmt(
            offpolicy_trajectories=offpolicy_trajectories,
            config=self.config,
            tokenizer=self.tokenizer,
            task_id_to_data_id=task_id_to_data_id  # ⭐ 确保 GRPO 分组正确
        )
```

**⭐ GRPO 分组关键设计**：
- 同一个 task 的所有 rollouts（on-policy 和 off-policy）应该共享同一个 `data_id`
- GRPO 使用 `data_id` 来分组计算 advantage
- `task_id_to_data_id` 映射确保 off-policy trajectory 使用与 on-policy trajectory 相同的 data_id

**`get_offpolicy_batch` 内部逻辑** (`agentevolver/module/exp_manager/exp_manager.py:473-501`):

1. 为每个 experience task 调用 `get_offpolicy_trajectories_from_memory()` 或 `get_all_candidates_batch()`
2. ⭐ **ExGRPO 设计**：如果 `use_current_policy_entropy=True`，则获取所有候选轨迹，由 `_select_best_offpolicy_by_current_entropy` 使用当前 policy 计算 entropy 后选择最优的
3. 如果 `use_current_policy_entropy=False`，则根据保存时的 entropy 选择轨迹（argmin/argmax/random）
4. 标记轨迹为 `is_experience_replay=True`

**`_select_best_offpolicy_by_current_entropy` 内部逻辑** (`agentevolver/module/trainer/ae_ray_trainer.py:851-970`):

⭐ **ExGRPO 设计**：使用当前 policy 计算 entropy，选择每个 task 的最优 off-policy 轨迹。

1. **获取候选轨迹**:
   ```python
   task_to_candidates = self.exp_manager.get_all_candidates_batch(tasks=experience_tasks)
   ```

2. **转换为 CMT 并计算 entropy**:
   ```python
   candidate_cmts = self.env_manager.convert_offpolicy_to_cmt(candidates, ...)
   candidate_batch = self.env_manager.to_dataproto(candidate_cmts)
   log_prob_result = self.actor_rollout_wg.compute_log_prob(candidate_batch)
   entropys = log_prob_result.batch["entropys"]
   ```

3. **计算平均 entropy（只考虑 LLM 响应部分）**:
   ```python
   for i in range(len(candidates)):
       traj_entropy = entropys[i].cpu().numpy()
       traj_response_mask = response_masks[i].cpu().numpy()
       # ⭐ Multi-turn 关键：只计算 response_mask=1 的位置（LLM 响应）
       valid_entropys = traj_entropy[traj_response_mask.astype(bool)]
       avg_entropy = float(np.mean(valid_entropys))
       avg_entropys.append(avg_entropy)
   ```

4. **根据 exp_select_mode 选择最优轨迹**:
   ```python
   if exp_select_mode == "argmin":
       sorted_indices = np.argsort(avg_entropys)  # 选择 entropy 最低的
   elif exp_select_mode == "argmax":
       sorted_indices = np.argsort(avg_entropys)[::-1]  # 选择 entropy 最高的
   ```

**关键点**:
- ⭐ **使用当前 policy 的 entropy**：更准确反映当前策略下哪个轨迹最优
- ⭐ **Multi-turn 关键**：只对 LLM 响应部分（`response_mask=1`）计算 entropy
- 如果计算失败，回退到使用保存时的 entropy

**`convert_offpolicy_to_cmt` 内部逻辑** (`agentevolver/module/env_manager/env_manager.py:390-470`):

1. **创建 Linear_CMT 对象**:
   ```python
   cmt = Linear_CMT(config, tokenizer)
   ```

2. **设置 data_id（关键）**:
   ```python
   # ⭐ ExGRPO 设计：使用 task_id_to_data_id 映射
   # 确保 off-policy trajectory 使用与 on-policy trajectory 相同的 data_id
   if task_id_to_data_id and task_id in task_id_to_data_id:
       data_id = task_id_to_data_id[task_id]
   else:
       data_id = int(task_id) if possible else hash(task_id) % 100000
   cmt.data_id = str(data_id)
   ```
   - ⭐ **关键变更**：同一个 task 的 on-policy 和 off-policy 共享相同的 data_id
   - GRPO 会将同一 task 的所有 rollouts 放在同一个分组中计算 advantage

3. **保持 LLM 消息的 author**:
   ```python
   # ⭐ Experience Replay: LLM 消息保持 author="llm"，用于计算 off-policy loss
   # 使用 exp_mask 区分 on/off-policy，而不是让 loss_mask=0
   if role == "assistant":
       author = "llm"  # 保持为 "llm"，loss_mask=1，参与 off-policy loss 计算
   ```
   - 这样 `loss_mask=1`，LLM 消息会参与 off-policy loss 计算
   - 使用 `exp_mask=1` 来标记 off-policy 数据

4. **设置 metadata**:
   ```python
   cmt.metadata["is_experience_replay"] = True
   cmt.metadata["old_log_probs"] = traj.metadata.get("old_log_probs")
   cmt.metadata["policy_version"] = traj.metadata.get("policy_version")
   ```

#### Step 4: 合并 Tasks

```python
tasks = experience_tasks + on_policy_tasks
```

**最终 tasks 列表**:
- 前 `len(experience_tasks)` 个：experience tasks（有对应的 off-policy trajectories）
- 后 `len(on_policy_tasks)` 个：on-policy tasks（没有 off-policy trajectories）

---

### 2.3 Rollout 生成轨迹

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1266`

```python
trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", epoch=f"train.{epoch}.{i}")
```

**关键点**:

1. **只对 tasks 进行 rollout**:
   - `experience_tasks` 和 `on_policy_tasks` 都会进行 rollout
   - 每个 task 生成 `n_rollout` 个轨迹

2. **Off-policy trajectories 不参与 rollout**:
   - `offpolicy_cmt_array` 已经是从内存中获取的历史轨迹
   - 它们不需要重新生成，直接使用

3. **Rollout 结果**:
   - `trajectories`: 所有 on-policy 轨迹（包括 experience tasks 和 on-policy tasks 的 rollout 结果）
   - 数量 = `len(tasks) * n_rollout`

---

### 2.4 更新 difficulty2task_dict 并合并轨迹

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1270-1287`

#### Step 1: 更新 difficulty2task_dict

```python
if enable_exp_replay:
    self.exp_manager.update_difficulty2task_dict(trajectories)
```

**`update_difficulty2task_dict` 内部逻辑** (`agentevolver/module/exp_manager/exp_manager.py:518-550`):

1. **按 task_id 分组统计成功次数**:
   ```python
   task_id_to_success_count: Dict[str, int] = defaultdict(int)
   for traj in trajectories:
       if traj.reward and traj.reward.outcome == 1.0:
           task_id_to_success_count[traj.task_id] += 1
   ```

2. **更新 difficulty2task_dict**:
   ```python
   for task_id in task_ids_seen:
       success_count = task_id_to_success_count.get(task_id, 0)
       # 从旧的 difficulty bucket 中移除
       # 加入新的 difficulty bucket
       if task_id not in self.skip_uid_set:
           self.difficulty2task_dict[success_count].append(task_id)
   ```

#### Step 2: 合并 On-policy 和 Off-policy 轨迹

```python
if offpolicy_cmt_array:
    all_trajectories = trajectories + offpolicy_cmt_array
else:
    all_trajectories = trajectories
```

**最终轨迹列表**:
- 前 `len(trajectories)` 个：on-policy 轨迹（来自 rollout）
- 后 `len(offpolicy_cmt_array)` 个：off-policy 轨迹（来自内存）

---

### 2.5 转换为 DataProto

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1287`

```python
gen_batch_output = self.env_manager.to_dataproto(all_trajectories)
```

**`to_dataproto` 流程** (`agentevolver/module/env_manager/env_manager.py:380-390`):

1. **`trajectories_to_samples`**: 将轨迹转换为 Sample 对象
   - 调用 `cmt.tokenize_steps()` 进行 tokenization
   - 对于 off-policy 数据，`is_experience_replay=True`，`split_prompt_reponse_index = len(input_ids)`（整个 trajectory 作为 prompt）

2. **`samples_to_dataproto`**: 将 Sample 对象转换为 DataProto
   - Padding 和 batching
   - 创建 `exp_mask`（基于 `Sample.extras["is_experience_replay"]`）
   - 创建 `recorded_old_log_probs`（从 `Sample.extras["old_log_probs"]`）

**关键数据结构**:

- `gen_batch_output.batch["exp_mask"]`: `(batch_size, seq_len)` - 1 表示 off-policy，0 表示 on-policy
- `gen_batch_output.batch["recorded_old_log_probs"]`: `(batch_size, response_len)` - 历史策略的 old_log_probs
- `gen_batch_output.batch["group_ids"]`: `(batch_size,)` - 基于 `data_id` 的分组 ID（用于 GRPO）

---

### 2.6 合并 Batch 并设置 UID

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1329-1331`

```python
batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)
batch.batch["response_mask"] = compute_response_mask(batch)
```

**关键修复** (`agentevolver/module/trainer/ae_ray_trainer.py:1323-1331`):

```python
# ⭐ GRPO 分组关键：uid 必须基于 data_id（group_ids）来设置，而不是随机 UUID
if "group_ids" in batch.batch:
    group_ids = batch.batch["group_ids"].cpu().numpy()
    batch.non_tensor_batch["uid"] = np.array([str(int(gid)) for gid in group_ids], dtype=object)
else:
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
```

**为什么重要**:
- GRPO 使用 `uid` 来分组计算 advantage
- 同一 `data_id` 的轨迹应该在同一组（例如，同一个 task 的 n_rollout 个轨迹）
- 如果使用随机 UUID，GRPO 分组会错误，导致 advantage 计算不准确

---

### 2.7 计算 Reward

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1346-1356`

```python
if self.use_rm:
    reward_tensor = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(reward_tensor)

if self.config.reward_model.launch_reward_fn_async:
    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
else:
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

**关键点**:
- 所有轨迹（on-policy 和 off-policy）都会计算 reward
- Off-policy 轨迹的 reward 是历史记录的，但会重新计算以验证一致性

---

### 2.8 计算 old_log_prob

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1358-1379`

#### Step 1: 计算当前策略的 old_log_prob

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
entropys = old_log_prob.batch["entropys"]
response_masks = batch.batch["response_mask"]
```

**关键点**:
- 对所有数据（on-policy 和 off-policy）都计算当前策略的 `old_log_prob`
- 这是重要性采样中的 `π_current` 的 log_prob

#### Step 2: 替换 Off-policy 的 old_log_prob

```python
if enable_exp_replay and "recorded_old_log_probs" in batch.batch:
    exp_is_correct = exp_replay_config.get("exp_is_correct", True)
    if exp_is_correct:
        batch = self._replace_recorded_old_log_probs(
            batch=batch,
            current_old_log_prob=old_log_prob,
            entropys=entropys,
        )
```

**`_replace_recorded_old_log_probs` 内部逻辑** (`agentevolver/module/trainer/ae_ray_trainer.py:800-850`):

1. **提取 exp_mask 和 recorded_old_log_probs**:
   ```python
   exp_mask = batch.batch["exp_mask"]  # [batch, seq_len]
   response_exp_mask = exp_mask[:, prompt_length:]  # [batch, response_len]
   recorded_old_log_probs = batch.batch["recorded_old_log_probs"]  # [batch, response_len]
   current_old_log_probs = current_old_log_prob.batch["old_log_probs"]  # [batch, response_len]
   ```

2. **条件替换**:
   ```python
   # off-policy (exp_mask=1): 使用 recorded_old_log_probs
   # on-policy (exp_mask=0): 使用 current_old_log_probs
   new_old_log_probs = torch.where(
       response_exp_mask.bool(),
       recorded_old_log_probs,
       current_old_log_probs[:, :min_len]
   )
   ```

**最终 old_log_prob**:
- **On-policy**: 使用当前策略计算的 `old_log_prob`（`π_current`）
- **Off-policy**: 使用 `recorded_old_log_probs`（`π_historical`，收集经验时的旧策略）

**重要性采样权重**:
```python
ratio = exp(log_prob_current - old_log_prob)
# On-policy: ratio = exp(log_prob_current - log_prob_current) = 1.0
# Off-policy: ratio = exp(log_prob_current - old_log_prob_historical)
```

---

### 2.9 计算 Advantage（GRPO）

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1480-1499`

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,
    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
    config=self.config.algorithm,
)
```

**`compute_advantage` 内部逻辑（GRPO）** (`agentevolver/module/trainer/ae_ray_trainer.py:263-278`):

```python
elif adv_estimator == AdvantageEstimator.GRPO:
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],  # ⭐ 使用 uid 分组
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

**`compute_grpo_outcome_advantage` 内部逻辑** (`agentevolver/module/trainer/ae_ray_trainer.py:158-218`):

1. **计算每个轨迹的总 reward**:
   ```python
   scores = token_level_rewards.sum(dim=-1)  # [batch_size]
   ```

2. **按 uid 分组**:
   ```python
   id2score = defaultdict(list)
   for i in range(bsz):
       id2score[index[i]].append(scores[i])  # index 是 uid
   ```

3. **计算组内均值和标准差**:
   ```python
   for idx in id2score:
       if len(id2score[idx]) > 1:
           id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
           id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
   ```

4. **计算相对 advantage**:
   ```python
   for i in range(bsz):
       if norm_adv_by_std_in_grpo:
           scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
       else:
           scores[i] = scores[i] - id2mean[index[i]]
   ```

5. **广播到 token 级别**:
   ```python
   scores = scores.unsqueeze(-1) * response_mask  # [batch_size, response_len]
   ```

**GRPO 分组机制** (⭐ ExGRPO 设计):

- **同一 task 的所有 rollouts 共享相同的 `uid`**: 无论是 on-policy 还是 off-policy，同一 task 的所有轨迹使用相同的 `data_id`/`uid`
- **混合分组计算 advantage**: 例如，experience_task_0 有 1 条 off-policy + 7 条 on-policy，它们都使用 `data_id=0`，在同一组内计算相对 advantage

**为什么这样设计**:
- 引入 off-policy rollouts 的目的是为了干涉 rollout batch 做 rollouts 的 distribution engineering
- 属于同一个 task 的 on-policy 和 off-policy rollouts 应该一起计算 advantages
- Off-policy 轨迹通常是成功的轨迹，可以帮助提高该 task 的 reward 基准线

---

### 2.10 计算 Loss

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1536-1540`

```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```

**`update_actor` 内部会调用 loss 计算函数**，例如 `het_compute_token_on_off_policy_loss`。

**Loss 计算逻辑** (`agentevolver/module/exp_manager/het_core_algos.py:45-121`):

1. **计算重要性采样权重**:
   ```python
   negative_approx_kl = log_prob - old_log_prob
   ratio = torch.exp(negative_approx_kl)  # ratio = π_current / π_old
   ```

2. **计算 PPO loss**:
   ```python
   pg_losses1 = -advantages * ratio
   pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
   pg_losses = torch.maximum(pg_losses1, pg_losses2)
   ```

3. **区分 On-policy 和 Off-policy**:
   ```python
   # On-policy loss
   on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)
   
   # Off-policy loss
   off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
   ```

4. **合并 Loss**:
   ```python
   pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
   pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
   ```

**关键点**:
- On-policy 数据使用标准 PPO loss（`ratio ≈ 1.0`）
- Off-policy 数据使用重要性采样校正的 PPO loss（`ratio = exp(log_prob_current - old_log_prob_historical)`）
- `exp_mask` 确保 on-policy 和 off-policy 的 loss 不会相互干扰

---

### 2.11 保存轨迹到内存

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1381-1421`

#### Step 1: 更新 skip_uid_set 并筛选轨迹

```python
filtered_trajectories = self.exp_manager.update_skip_uid_set_and_filter_trajectories(
    trajectories=trajectories,  # 只用 on-policy 轨迹
    n_rollout=n_rollout,
    entropys=on_policy_entropys,
    response_mask=on_policy_response_mask,
)
```

**`update_skip_uid_set_and_filter_trajectories` 内部逻辑** (`agentevolver/module/exp_manager/exp_manager.py:354-435`):

1. **按 task_id 分组统计**:
   ```python
   task_id_to_trajectories: Dict[str, List[Trajectory]] = defaultdict(list)
   for traj in trajectories:
       task_id_to_trajectories[traj.task_id].append(traj)
   ```

2. **更新 skip_uid_set**:
   ```python
   for task_id, trajs in task_id_to_trajectories.items():
       success_count = sum(1 for traj in trajs if traj.reward and traj.reward.outcome == 1.0)
       
       if success_count == n_rollout:  # 全部做对
           self.skip_uid_set.add(task_id)
           # 从 difficulty2task_dict 和 task2trajectories 中移除
   ```

3. **筛选符合条件的轨迹**:
   ```python
   if self.experience_lbound < success_count < self.experience_rbound:
       # ⭐ ExGRPO 设计：保存所有 reward 为正的轨迹，取用时再选最优的
       successful_trajs = [
           t for e, t in task_id_to_entropy_list[task_id] 
           if t.reward and t.reward.outcome > 0  # 所有 reward 为正的轨迹
       ]
       if successful_trajs:
           # 将所有成功轨迹都加入待保存列表（不再只选一个最优的）
           filtered_trajectories_to_save.extend(successful_trajs)
   ```

**筛选条件**:
- 非全对非全错：`experience_lbound < success_count < self.experience_rbound`
- ⭐ **ExGRPO 设计**：保存所有 reward 为正的轨迹（`outcome > 0`），不在存储时筛选最优的
- 取用时使用当前 policy 计算 entropy，选择最优的轨迹

#### Step 2: 保存 old_log_probs 到轨迹 metadata

```python
old_log_probs_tensor = old_log_prob.batch["old_log_probs"]
for idx, traj in enumerate(trajectories):
    if idx < old_log_probs_tensor.shape[0]:
        traj_old_log_prob = old_log_probs_tensor[idx].cpu().numpy()
        traj_response_mask = response_masks[idx].cpu().numpy()
        traj_old_log_prob = traj_old_log_prob[traj_response_mask.astype(bool)]
        traj.metadata["old_log_probs"] = traj_old_log_prob.tolist()
        traj.metadata["policy_version"] = self.global_steps
```

**关键点**:
- 只保存 on-policy 轨迹的 `old_log_probs`（off-policy 轨迹已经有历史记录）
- `old_log_probs` 会被用于后续的重要性采样

#### Step 3: 保存轨迹到内存

```python
if filtered_trajectories:
    self.exp_manager.save_trajectories_to_memory(filtered_trajectories)
```

**`save_trajectories_to_memory` 内部逻辑** (`agentevolver/module/exp_manager/exp_manager.py:270-308`):

1. **保存所有传入的轨迹**:
   ```python
   for traj in trajectories:
       task_id = traj.task_id
       if task_id not in self.task2trajectories:
           self.task2trajectories[task_id] = []
   ```

2. **内存管理（使用 max_trajectories_per_task 限制）**:
   ```python
   if len(self.task2trajectories[task_id]) >= self.max_trajectories_per_task:
       # 根据 exp_select_mode 决定替换策略
       if self.exp_select_mode == "argmin":
           # 如果新轨迹的 entropy 更低，则替换掉当前 entropy 最高的
           if traj_entropy < max(current_entropies):
               max_entropy_idx = current_entropies.index(max(current_entropies))
               self.task2trajectories[task_id][max_entropy_idx] = traj
       # ... 其他模式（argmax/FIFO）
   else:
       self.task2trajectories[task_id].append(traj)  # 直接添加
   ```

3. **关键点**:
   - ⭐ **ExGRPO 设计**：存储时保存所有传入的成功轨迹，不再只保存一个最优的
   - 使用 `max_trajectories_per_task` 限制每个 task 的最大轨迹数
   - 当超过限制时，根据 `exp_select_mode` 决定替换策略
   - 取用时使用当前 policy 计算 entropy，选择最优的轨迹

---

### 2.12 保存 Checkpoint 和 Experience Pool

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1583-1595`

```python
if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
    self._save_checkpoint()
    
    if enable_exp_replay:
        exp_pool_save_dir = os.path.join(
            self.config.trainer.default_local_dir,
            "experience_pool",
            f"step_{self.global_steps}"
        )
        self.exp_manager.save_experience_pool_to_disk(exp_pool_save_dir)
```

**保存内容**:
- `difficulty2task_dict.json`: 难度分组字典
- `skip_uid_set.json`: 跳过集合
- `task2trajectories.pkl`: 轨迹字典（包含 old_log_probs）

---

## 关键数据结构

### 3.1 DataProto.batch

```python
{
    "prompts": torch.Tensor,  # [batch_size, prompt_len]
    "responses": torch.Tensor,  # [batch_size, response_len]
    "input_ids": torch.Tensor,  # [batch_size, seq_len]
    "attention_mask": torch.Tensor,  # [batch_size, seq_len]
    "position_ids": torch.Tensor,  # [batch_size, seq_len]
    "loss_mask": torch.Tensor,  # [batch_size, seq_len]
    "exp_mask": torch.Tensor,  # [batch_size, seq_len] - 1=off-policy, 0=on-policy
    "response_mask": torch.Tensor,  # [batch_size, response_len]
    "group_ids": torch.Tensor,  # [batch_size] - 基于 data_id
    "recorded_old_log_probs": torch.Tensor,  # [batch_size, response_len]
    "old_log_probs": torch.Tensor,  # [batch_size, response_len]
    "log_probs": torch.Tensor,  # [batch_size, response_len]
    "advantages": torch.Tensor,  # [batch_size, response_len]
    "returns": torch.Tensor,  # [batch_size, response_len]
    "token_level_rewards": torch.Tensor,  # [batch_size, response_len]
}
```

### 3.2 DataProto.non_tensor_batch

```python
{
    "task_ids": np.ndarray,  # [batch_size]
    "rollout_ids": np.ndarray,  # [batch_size]
    "uid": np.ndarray,  # [batch_size] - 用于 GRPO 分组（基于 group_ids）
    "extras": np.ndarray,  # [batch_size] - 包含 is_experience_replay, old_log_probs 等
}
```

### 3.3 ExperienceManager 数据结构

```python
{
    "difficulty2task_dict": Dict[int, List[str]],  # 难度 -> task_id 列表
    "task2trajectories": Dict[str, List[Trajectory]],  # task_id -> 轨迹列表
    "skip_uid_set": Set[str],  # 已完全解决的 task_id
}
```

---

## 常见问题排查

### 4.1 GRPO 分组错误

**症状**: Advantage 计算不准确，训练不稳定

**原因**:
- `uid` 使用随机 UUID 而不是基于 `data_id`
- Off-policy 数据的 `data_id` 与对应 on-policy 数据不一致

**解决**:
- 确保 `uid` 基于 `group_ids`（`data_id`）设置
- ⭐ **ExGRPO 设计**：确保同一 task 的 on-policy 和 off-policy 使用相同的 `data_id`
- 使用 `task_id_to_data_id` 映射确保 data_id 一致

### 4.2 Off-policy 数据的 Loss 计算问题

**症状**: Off-policy 轨迹没有参与 loss 计算

**原因**:
- `author` 被错误设置为 `"llm(do_not_train)"`，导致 `loss_mask=0`
- Off-policy 数据的 LLM 消息应该保持 `author="llm"`

**解决**:
- 在 `convert_offpolicy_to_cmt` 中确保所有 LLM 消息的 `author="llm"`（而不是 `"llm(do_not_train)"`）
- 使用 `exp_mask=1` 来标记 off-policy 数据，而不是使用 `loss_mask=0`
- 在 `tokenize_steps` 中检查 `is_experience_replay`，正确设置 `split_prompt_reponse_index`

### 4.3 Multi-turn 场景中的对齐问题

**症状**: Off-policy 数据的 old_log_probs 位置错位

**原因**:
- 保存 old_log_probs 时过滤了 Environment 响应，导致加载时位置不对

**解决**:
- 保存完整的 old_log_probs（不过滤），同时保存 response_mask
- exp_mask 只对 LLM 响应位置（loss_mask=1）设置为 1

### 4.4 Environment 响应被错误计算 loss

**症状**: Environment 响应也参与了 loss 计算

**原因**:
- exp_mask 对整个 response 设置为 1，而不是只对 LLM 响应

**解决**:
- 使用 `response_loss_mask` 创建 exp_mask，确保只标记 LLM 响应为 off-policy

### 4.3 Importance Sampling 权重错误

**症状**: Off-policy loss 过大或过小

**原因**:
- `old_log_probs` 没有正确替换
- `exp_is_correct=False` 但应该使用 `recorded_old_log_probs`

**解决**:
- 检查 `_replace_recorded_old_log_probs` 是否正确替换
- 确保 `exp_is_correct=True` 时使用 `recorded_old_log_probs`

### 4.4 Experience Pool 为空

**症状**: 没有 off-policy 数据参与训练

**原因**:
- `replay_start_ratio` 太大，还没开始 replay
- `experience_lbound` 和 `experience_rbound` 设置不合理
- 所有任务都在 `skip_uid_set` 中

**解决**:
- 检查 `training_progress >= replay_start_ratio`
- 检查 `get_valid_replay_task_ids()` 是否返回空列表
- 调整 `experience_lbound` 和 `experience_rbound`

---

## 总结

Experience Replay 与 GRPO 的完整流程包括：

1. **准备阶段**: 混合 on-policy 和 off-policy tasks，获取历史轨迹
2. **Rollout 阶段**: 只对 tasks 进行 rollout，off-policy 轨迹直接使用
3. **数据转换**: 将轨迹转换为 DataProto，设置 `exp_mask` 和 `recorded_old_log_probs`
4. **GRPO 分组**: 基于 `uid`（来自 `group_ids`）分组计算 advantage
5. **Loss 计算**: 使用重要性采样区分 on-policy 和 off-policy loss
6. **保存阶段**: 筛选并保存成功轨迹到内存

关键点：
- **Off-policy Loss 计算**: Off-policy 轨迹参与 loss 计算，使用 `exp_mask` 区分 on/off-policy
- **⭐ 共享 data_id**: 同一 task 的 on-policy 和 off-policy 使用相同的 `data_id`，GRPO 在同一组内计算 advantage
- **重要性采样**: 使用 `recorded_old_log_probs` 校正 off-policy 数据权重
- **⭐ ExGRPO 存储策略**: 存储所有 reward 为正的轨迹，取用时使用当前 policy 计算 entropy 选择最优的
- **LLM 消息保持 author="llm"**: 确保 `loss_mask=1`，参与 off-policy loss 计算

