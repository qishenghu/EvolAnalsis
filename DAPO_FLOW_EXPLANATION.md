# DAPO 完整流程：从环境 Reward 到模型更新

## 📊 流程图概览

```
环境执行 → Reward (0/1) → Token-Level Reward → DAPO 处理 → Advantage → Loss → 模型更新
   ↓            ↓                ↓                  ↓            ↓        ↓         ↓
Trajectory   outcome      token_level_scores   Overlong      GRPO    DAPO     Actor
             (scalar)      (sparse tensor)    Shaping      Adv      Clip-    Update
                                                           Calc     Higher
```

---

## 🔄 详细流程步骤

### **Step 1: 环境执行与 Reward 获取**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1637`

```python
trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="sample", ...)
```

- **输入**: Tasks (prompts)
- **输出**: `Trajectory` 对象列表，每个包含：
  - `traj.reward.outcome`: **Sequence-level reward** (0.0 或 1.0)
  - `traj.steps`: 多轮对话步骤
  - `traj.is_terminated`: 是否正常结束

**关键点**: 此时 reward 是**标量**，表示整个轨迹的成功/失败。

---

### **Step 2: Trajectory → DataProto 转换**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1658`

```python
gen_batch_output = self.env_manager.to_dataproto(all_trajectories)
```

- 将 `Trajectory` 转换为 `DataProto`，包含：
  - `batch["prompts"]`, `batch["responses"]`
  - `non_tensor_batch["reward_scores"]`: 每个样本的 `{"outcome": 0.0/1.0}`

---

### **Step 3: Reward 转换为 Token-Level Tensor**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1735` → `parse_reward_from_dataproto()`

```python
# agentevolver/module/trainer/ae_ray_trainer.py:73-112
def parse_reward_from_dataproto(data: DataProto, return_dict=False):
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, seq_len)
    
    # 获取每个样本的 response 长度
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs,)
    
    # 获取环境返回的 outcome (0.0 或 1.0)
    reward_scores = torch.tensor([item["outcome"] for item in data.non_tensor_batch["reward_scores"]])
    
    # ⭐ 关键：将 reward 放在最后一个有效 token 位置（稀疏放置）
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores
    
    return reward_tensor  # Shape: (batch_size, response_length)
```

**关键点**:
- Reward 是**稀疏的**：只在最后一个有效 token 位置有值（0.0 或 1.0）
- 其他位置都是 0
- Shape: `(batch_size, response_length)`，例如 `(64, 4096)`

---

### **Step 4: DAPO Overlong Reward Shaping**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1921-2020`

```python
# 检测截断样本
is_truncated_by_length = (actual_response_lengths >= max_response_length - 1)
is_truncated_by_termination = ~traj.is_terminated  # 多轮对话未正常结束
is_truncated = is_truncated_by_length | is_truncated_by_termination

# 应用软惩罚
reward_tensor = dapo_overlong_reward_shaping(
    rewards=reward_tensor,
    is_truncated=is_truncated,
    truncation_penalty=-0.5,  # 配置中的值
    soft_penalty_mode="additive",
)
```

**`dapo_overlong_reward_shaping` 实现** (`het_core_algos.py:572-681`):

```python
# 对于 2D tensor (batch_size, seq_len)
for i in range(rewards.shape[0]):
    if not is_truncated[i]:
        continue
    
    # 获取轨迹级 reward（sum 所有 token）
    traj_reward = rewards[i].sum()  # 例如：1.0
    
    # 应用惩罚（additive 模式）
    new_traj_reward = traj_reward + truncation_penalty  # 1.0 + (-0.5) = 0.5
    
    # ⭐ 关键：只在最后一个非零位置放置新 reward
    reward_pos = non_zero_positions[-1]  # 找到 reward 位置
    modified_rewards[i] = 0  # 清空所有位置
    modified_rewards[i, reward_pos] = new_traj_reward  # 只在一个位置设置
```

**结果**:
- 正常样本：reward 保持 0.0 或 1.0
- 截断样本：reward 变为 -0.5 或 0.5（1.0 - 0.5）
- **仍然稀疏**：每个样本只有一个非零位置

---

### **Step 5: Token-Level Rewards 设置**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2022-2027`

```python
if self.config.algorithm.use_kl_in_reward:
    # 如果启用 KL penalty in reward，会减去每个 token 的 KL divergence
    batch, kl_metrics = apply_kl_penalty(batch, ...)
else:
    # ⭐ DAPO 配置中 use_kl_in_reward: false，所以直接复制
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
```

**关键点**: 
- `token_level_rewards` = `token_level_scores`（DAPO 配置下）
- 仍然是稀疏的：每个样本只有一个非零位置

---

### **Step 6: DAPO Dynamic Sampling（过滤样本）**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2060-2099`

```python
if use_dapo and dapo_config.get("dynamic_sampling", {}).get("enable", False):
    # 获取同一 prompt 的所有 rollouts 的 rewards
    filter_rewards = batch.batch["token_level_rewards"].sum(dim=-1)  # (batch_size,)
    
    # 按 group_id (uid) 分组，计算每组准确率
    keep_mask = dapo_filter_samples(
        rewards=filter_rewards,
        group_ids=group_ids,  # 同一 prompt 的 rollouts 有相同的 uid
        n_rollout=8,
        filter_mode="strict",  # 过滤全对或全错的组
    )
    
    # ⭐ 关键：不删除样本（会破坏 GRPO 分组），而是将 advantage 置零
    batch.batch["dapo_keep_mask"] = keep_mask.float()
```

**`dapo_filter_samples` 逻辑** (`het_core_algos.py:495-569`):

```python
# 对每个 prompt 组（相同 uid）：
for group_id in unique_groups:
    group_rewards = [rewards[i] for i where uid[i] == group_id]
    accuracy = sum(group_rewards > 0) / len(group_rewards)
    
    if filter_mode == "strict":
        if accuracy == 0.0 or accuracy == 1.0:
            # 全错或全对 → 过滤（标记为 False）
            keep_mask[group_indices] = False
```

**结果**:
- 被过滤的样本：advantage 会被置零（Step 8）
- 保持 batch 结构不变，GRPO 分组仍然有效

---

### **Step 7: GRPO Advantage 计算**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2111` → `compute_advantage()` → `compute_grpo_outcome_advantage()`

```python
# agentevolver/module/trainer/ae_ray_trainer.py:158-218
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    # Step 7.1: 计算每个样本的总 reward
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)
    # 例如：scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]  # 8 个 rollouts
    
    # Step 7.2: 按 group_id (uid) 分组计算均值和标准差
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    
    for idx in id2score:
        if len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))  # 组内均值
            id2std[idx] = torch.std(torch.tensor(id2score[idx]))     # 组内标准差
    
    # Step 7.3: 计算 normalized advantage
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    
    # Step 7.4: ⭐ 关键：将 advantage 扩展到每个 token
    scores = scores.unsqueeze(-1) * response_mask  # (batch_size, seq_len)
    # 例如：如果 advantage = 0.935，response 有 3000 个 token
    # 那么每个 token 的 advantage 都是 0.935
    
    return scores, scores  # advantages 和 returns 相同（GRPO 特性）
```

**关键点**:
- Advantage 是**密集的**：每个有效 token 都有相同的 advantage 值
- Advantage 值 = `(reward - group_mean) / group_std`
- 范围通常在 [-2, 2] 左右（取决于组内方差）

---

### **Step 8: DAPO Dynamic Sampling 应用（置零 advantage）**

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:2122-2132`

```python
if "dapo_keep_mask" in batch.batch:
    dapo_keep_mask = batch.batch["dapo_keep_mask"]  # (batch_size,)
    dapo_keep_mask = dapo_keep_mask.unsqueeze(-1)  # (batch_size, 1)
    
    # ⭐ 将被过滤样本的 advantage 置零
    batch.batch["advantages"] = batch.batch["advantages"] * dapo_keep_mask
```

**结果**: 被过滤样本的 advantage 全为 0，不会产生梯度。

---

### **Step 9: DAPO Policy Loss 计算（Clip-Higher）**

**位置**: `agentevolver/module/exp_manager/het_actor.py:162-178` → `dapo_compute_policy_loss()`

```python
# agentevolver/module/exp_manager/het_core_algos.py:319-492
def dapo_compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, ...):
    # Step 9.1: 计算 importance sampling ratio
    ratio = torch.exp(log_prob - old_log_prob)  # π_new / π_old
    
    # Step 9.2: ⭐ DAPO Clip-Higher 核心逻辑
    # 对于 A > 0（鼓励的动作）：
    ratio_clipped_pos = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    # 例如：clip 到 [0.8, 1.28]（cliprange_low=0.2, cliprange_high=0.28）
    
    # 对于 A < 0（不鼓励的动作）：
    ratio_clipped_neg = torch.clamp(ratio, 1 - cliprange_low, clip_ratio_c)
    # 例如：clip 到 [0.8, 3.0]（移除上界，允许更大的减少）
    
    # Step 9.3: 根据 advantage 符号选择 clipped ratio
    on_pg_losses_clipped = torch.where(
        advantages >= 0,
        -advantages * ratio_clipped_pos,  # A > 0: 标准 clip
        -advantages * ratio_clipped_neg,  # A < 0: 移除上界
    )
    
    # Step 9.4: PPO-style max（取更保守的 loss）
    on_pg_losses = torch.maximum(
        -advantages * ratio,           # 未 clip 的 loss
        on_pg_losses_clipped           # clip 后的 loss
    )
    
    # Step 9.5: ⭐ Token-Level 聚合（DAPO 的第三个改进）
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode="token-mean",  # 按 token 平均，而不是按序列
    )
    
    return {"pg_loss": pg_loss, ...}
```

**DAPO Clip-Higher 的关键**:
- **A > 0**: 标准 PPO clip `[1-ε_low, 1+ε_high]`，限制概率增加
- **A < 0**: 移除上界，允许低概率 token 进一步减少，**防止熵崩塌**

**Token-Level 聚合**:
- `loss_agg_mode: token-mean` 确保每个 token 对 loss 的贡献相等
- 避免短序列被过度加权

---

### **Step 10: 完整 Loss 计算与模型更新**

**位置**: `agentevolver/module/exp_manager/het_actor.py:62-220`

```python
# Step 10.1: 计算完整 loss
total_loss = (
    pg_loss +                    # DAPO policy loss
    kl_loss_coef * kl_loss +    # KL divergence loss（如果启用）
    entropy_coeff * entropy_loss # Entropy bonus（通常为 0）
)

# Step 10.2: 反向传播
total_loss.backward()

# Step 10.3: 梯度裁剪
torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), grad_clip)

# Step 10.4: 优化器更新
optimizer.step()
```

---

## 📈 数据流形状变化总结

| 步骤 | 数据 | Shape | 稀疏性 | 说明 |
|------|------|-------|--------|------|
| 1. 环境 | `outcome` | `(1,)` | - | 标量：0.0 或 1.0 |
| 2. Token-Level | `token_level_scores` | `(bs, seq_len)` | ✅ 稀疏 | 只在最后一个 token 有值 |
| 3. Overlong Shaping | `token_level_scores` | `(bs, seq_len)` | ✅ 稀疏 | 截断样本：reward - 0.5 |
| 4. Token Rewards | `token_level_rewards` | `(bs, seq_len)` | ✅ 稀疏 | = token_level_scores |
| 5. Advantage | `advantages` | `(bs, seq_len)` | ❌ 密集 | 每个有效 token 都有值 |
| 6. Policy Loss | `pg_losses` | `(bs, seq_len)` | ❌ 密集 | 每个 token 的 loss |
| 7. Aggregated Loss | `pg_loss` | `(1,)` | - | Token-level 平均 |

---

## 🎯 DAPO 的三个核心改进

### 1. **Clip-Higher（解耦非对称裁剪）**
- **位置**: `dapo_compute_policy_loss()` (Step 9)
- **效果**: A < 0 时移除上界，防止熵崩塌

### 2. **Dynamic Sampling（动态采样）**
- **位置**: `dapo_filter_samples()` (Step 6)
- **效果**: 过滤全对/全错的 prompt 组，减少无效梯度

### 3. **Token-Level Policy Gradient**
- **位置**: `loss_agg_mode: token-mean` (Step 9.5)
- **效果**: 按 token 平均 loss，避免短序列被过度加权

### 4. **Overlong Reward Shaping（截断奖励塑造）**
- **位置**: `dapo_overlong_reward_shaping()` (Step 4)
- **效果**: 对截断样本应用软惩罚（-0.5），保留部分学习信号

---

## 🔍 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 环境执行 | `ae_ray_trainer.py` | 1637 |
| Reward 转换 | `ae_ray_trainer.py` | 73-112 |
| Overlong Shaping | `ae_ray_trainer.py` | 1921-2020 |
| Overlong 实现 | `het_core_algos.py` | 572-681 |
| Dynamic Sampling | `ae_ray_trainer.py` | 2060-2099 |
| Dynamic 实现 | `het_core_algos.py` | 495-569 |
| GRPO Advantage | `ae_ray_trainer.py` | 158-218 |
| DAPO Loss | `het_core_algos.py` | 319-492 |
| 模型更新 | `het_actor.py` | 62-220 |

---

## 💡 常见问题

### Q1: 为什么 `critic/rewards_last/mean` 和 `critic/rewards_sum/mean` 不同？

**A**: 
- `rewards_sum` = `token_level_rewards.sum(-1)`：每个样本的总 reward
- `rewards_last` = `token_level_rewards[batch_idx, last_resp_idx]`：最后一个有效 token 的 reward

如果不同，可能原因：
1. **Token-level shaping**：某些 token 有额外的 penalty（但 DAPO 配置下不应该）
2. **位置不一致**：reward 实际位置 ≠ metrics 认为的最后一个 token（多轮对话常见）

### Q2: Advantage 为什么是密集的？

**A**: GRPO 将 sequence-level advantage 扩展到每个 token，这样每个 token 都能获得相同的学习信号。这是 GRPO 的设计，不是 bug。

### Q3: 为什么 validation 的 reward 正常（0-1），但训练时会有负数？

**A**: Validation 不经过 `dapo_overlong_reward_shaping`，所以 reward 保持原始的 0.0/1.0。训练时截断样本会被减去 0.5，所以可能变成 -0.5 或 0.5。

