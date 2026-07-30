# Experience Replay 中 Off-Policy 数据的处理机制

## 概述

在 Experience Replay 机制中，off-policy 轨迹（历史成功轨迹）需要**参与训练 loss 计算**。关键设计：

1. **Off-policy LLM 消息保持 `author="llm"`**：确保 `loss_mask=1`，参与 loss 计算
2. **使用 `exp_mask=1` 区分 on-policy 和 off-policy**：`het_compute_token_on_off_policy_loss` 使用 `exp_mask` 来分别计算 on/off-policy loss
3. **重要性采样**：off-policy 数据使用 `recorded_old_log_probs` 计算重要性采样权重

## ⚠️ 重要设计原则

参考 ExGRPO 的 `mix_core_alg.py`，off-policy 数据**应该参与 loss 计算**：

```python
# ExGRPO mix_core_alg.py
off_pg_losses = -advantages * off_ratio
off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)  # ⭐ off-policy 计算 loss

pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)  # 合并
```

**关键区别**：
- 使用 `exp_mask`（或 ExGRPO 中的 `prefix_mask`）区分 on-policy 和 off-policy
- **不使用 `loss_mask=0` 来区分 off-policy**

---

## 核心机制

### 1. Off-Policy 数据的 Author 设置

**位置**: `agentevolver/module/env_manager/env_manager.py:435-458`

```python
def convert_offpolicy_to_cmt(self, offpolicy_trajectories, config, tokenizer):
    for traj in offpolicy_trajectories:
        cmt = Linear_CMT(config, tokenizer)
        # ...
        for msg in steps:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                
                # ⭐ Experience Replay: LLM 消息保持 author="llm"，用于计算 off-policy loss
                # 使用 exp_mask 区分 on/off-policy，而不是让 loss_mask=0
                if role == "assistant":
                    author = "llm"  # 保持为 "llm"，loss_mask=1，参与 off-policy loss 计算
                elif role == "user":
                    author = "user"
                else:
                    author = role
                
                ext_msg = ExtendedMessage(
                    author=author,
                    role=role,
                    content=msg.get("content", ""),
                    # ...
                )
                cmt.full_context.append(ext_msg)
```

**关键点**:
- 所有 `role="assistant"` 的消息，`author` 保持为 `"llm"`
- 这确保了 `need_training = True`，`loss_mask = 1`
- Off-policy 数据参与 loss 计算

---

### 2. `tokenize_steps` 中的特殊处理

**位置**: `agentevolver/module/context_manager/cmt_linear.py:608-640`

```python
def tokenize_steps(self, ext_steps: List[ExtendedMessage], debug=False) -> dict:
    # 检查是否为 experience replay 数据
    is_experience_replay = self.metadata.get("is_experience_replay", False)
    
    # ... tokenization ...
    
    split_prompt_reponse_index = -1
    for ext_msg in ext_steps:
        # ⭐ Experience Replay: 对于 off-policy 数据，所有 LLM 消息都应该参与 loss 计算
        if (split_prompt_reponse_index == -1) and (ext_msg.need_training or (is_experience_replay and ext_msg.role == "assistant")):
            split_prompt_reponse_index = len(input_ids)
            if not is_experience_replay:
                assert ext_msg.author == 'llm', "The first message after initialization should be from LLM"
        
        input_ids += ext_msg.token_arr
        attention_mask += [1] * len(ext_msg.token_arr)
        
        # ⭐ Experience Replay: 对于 off-policy 数据，LLM 消息的 loss_mask 应该为 1
        if is_experience_replay and ext_msg.role == "assistant":
            msg_loss_mask = [1] * len(ext_msg.token_arr)  # 参与 off-policy loss 计算
        else:
            msg_loss_mask = ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)
        loss_mask += msg_loss_mask
    
    # ⭐ 如果是 experience replay 数据且没有找到 split_prompt_reponse_index
    if is_experience_replay and split_prompt_reponse_index == -1:
        split_prompt_reponse_index = 0  # 所有内容都是 response（用于计算 loss）
```

**关键点**:
- 对于 experience replay 数据，LLM 消息也会成为分割点
- 对于 experience replay 数据，LLM 消息的 `loss_mask=1`（参与 off-policy loss 计算）

---

### 3. `exp_mask` 的创建

**位置**: `agentevolver/module/env_manager/env_manager.py:605-623`

```python
# Create experience mask: 
if sample.extras.get("is_experience_replay", False):
    # Experience Replay: response 部分全为 1（off-policy）
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
elif sample.extras.get("add_exp", False) and sample.extras.get("task_train_expmode", None)=="discard":
    # Experience-guided: prompt 和 response 都标记为 1
    prompt_exp_mask_list.append(torch.ones(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
else:
    # On-policy without experience: 全为 0
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))
```

**关键点**:
- Off-policy 数据的 `exp_mask=1`（response 部分）
- On-policy 数据的 `exp_mask=0`

---

### 4. Loss 计算

**位置**: `agentevolver/module/exp_manager/het_core_algos.py:45-121`

```python
def het_compute_token_on_off_policy_loss(
    old_log_prob, log_prob, advantages, response_mask, exp_mask, ...
):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    
    # On-policy loss (exp_mask=0)
    on_pg_losses, on_pg_clipfrac, _ = compute_pg_losses(cliprange_low, cliprange_high)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)
    
    # Off-policy loss (exp_mask=1)
    off_pg_losses, off_pg_clipfrac, _ = compute_pg_losses(off_cliprange_low, off_cliprange_high)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    
    # Combine
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

**关键点**:
- On-policy 数据：使用 `(1.0 - exp_mask) * response_mask` 选择，计算 `on_pg_loss`
- Off-policy 数据：使用 `exp_mask * response_mask` 选择，计算 `off_pg_loss`
- 两者合并：`pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)`

---

## 完整数据流

### On-policy 数据（正常训练）

```
轨迹: [user_msg, llm_msg, env_msg, llm_msg, ...]
      ↓
author: ["user", "llm", "env", "llm", ...]
      ↓
need_training: [False, True, False, True, ...]
      ↓
loss_mask: [0, 1, 0, 1, ...]  (只有 llm 消息计算 loss)
      ↓
exp_mask: [0, 0, 0, 0, ...]  (全为 0，标记为 on-policy)
      ↓
在 het_compute_token_on_off_policy_loss 中计算 on_pg_loss
```

### Off-policy 数据（Experience Replay）

```
轨迹: [user_msg, llm_msg, env_msg, llm_msg, ...]
      ↓
author: ["user", "llm", "env", "llm", ...]  (保持 "llm"，不是 "llm(do_not_train)")
      ↓
need_training: [False, True, False, True, ...]  (llm 消息为 True)
      ↓
loss_mask: [0, 1, 0, 1, ...]  (llm 消息的 loss_mask=1，参与 loss 计算)
      ↓
exp_mask: [0, 1, 0, 1, ...]  (response 部分为 1，标记为 off-policy)
      ↓
在 het_compute_token_on_off_policy_loss 中计算 off_pg_loss
```

---

## On-policy vs Off-policy 对比

| 数据类型 | author | loss_mask | exp_mask | loss 计算 | old_log_prob |
|----------|--------|-----------|----------|-----------|--------------|
| On-policy | `"llm"` | 1 | 0 | `on_pg_loss` | 当前策略计算 |
| Off-policy | `"llm"` | 1 | 1 | `off_pg_loss` | `recorded_old_log_probs` |

---

## 重要性采样（Importance Sampling）

Off-policy 数据需要重要性采样来校正权重：

```python
ratio = exp(log_prob_current - old_log_prob_historical)
```

- **log_prob_current**：当前策略对 off-policy 数据的 log_prob（新计算）
- **old_log_prob_historical**：历史策略的 log_prob（`recorded_old_log_probs`，保存的）
- **ratio**：重要性采样权重

在 `_replace_recorded_old_log_probs` 中替换 off-policy 数据的 old_log_prob：

```python
new_old_log_probs = torch.where(
    response_exp_mask.bool(),
    recorded_old_log_probs,     # off-policy: 使用历史策略的 old_log_prob
    current_old_log_probs       # on-policy: 使用当前策略的 old_log_prob
)
```

---

## 与 ExGRPO 的对比

| 特性 | ExGRPO 的设计 | AgentEvolver 的设计 |
|------|---------------|---------------------|
| **场景** | Single-turn（单轮对话） | Multi-turn（多轮对话） |
| **Mask 名称** | `prefix_mask` | `exp_mask` |
| **LLM 消息 author** | N/A（单轮） | 保持 `author="llm"` |
| **loss_mask** | 正常计算 | 正常计算（off-policy 也为 1） |
| **Loss 函数** | `compute_token_on_off_policy_loss` | `het_compute_token_on_off_policy_loss` |
| **重要性采样** | 使用 `recorded_old_log_probs` | 使用 `recorded_old_log_probs` |
| **response_mask** | `attention_mask[:, -response_length:]` | `loss_mask[:, -response_length:]`（multi-turn） |
| **exp_mask 范围** | 整个 response | 只有 LLM 响应位置（基于 loss_mask） |

---

## Multi-turn 场景的特殊处理

### 关键区别

与 ExGRPO 的 single-turn 场景不同，multi-turn 场景中：

1. **Response 结构**：包含多轮 LLM-Environment 交替对话
2. **只有 LLM 响应需要计算 loss**：Environment 响应不参与 loss 计算

### response_mask 的来源

**位置**: `agentevolver/module/exp_manager/het_actor.py:127-130`

```python
if multi_turn:
    response_mask = data["loss_mask"][:, -response_length:]  # 使用 loss_mask
else:
    response_mask = attention_mask[:, -response_length:]  # 使用 attention_mask
```

在 multi-turn 场景中，`response_mask` 基于 `loss_mask`，确保只有 LLM 响应参与 loss 计算。

### exp_mask 的创建

**位置**: `agentevolver/module/env_manager/env_manager.py:615-620`

```python
if sample.extras.get("is_experience_replay", False):
    # ⭐ Multi-turn 关键：使用 response_loss_mask 而不是全 1
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))
```

`exp_mask` 只对 LLM 响应位置设置为 1，Environment 响应位置为 0。

### old_log_probs 的保存

**位置**: `agentevolver/module/trainer/ae_ray_trainer.py:1407-1416`

```python
# ⭐ Multi-turn 关键：保存完整的 response 部分的 old_log_probs（不过滤）
traj.metadata["old_log_probs"] = traj_old_log_prob.tolist()
traj.metadata["response_mask"] = traj_response_mask.tolist()  # 保存 mask 用于后续对齐
```

保存完整的 `old_log_probs`（不过滤），同时保存 `response_mask` 用于后续对齐。

---

## 关键代码位置总结

1. **设置 author**: `agentevolver/module/env_manager/env_manager.py:444-445`
   ```python
   if role == "assistant":
       author = "llm"  # 保持为 "llm"，loss_mask=1
   ```

2. **tokenize_steps 特殊处理**: `agentevolver/module/context_manager/cmt_linear.py:615-625`
   ```python
   if is_experience_replay and ext_msg.role == "assistant":
       msg_loss_mask = [1] * len(ext_msg.token_arr)  # 参与 off-policy loss 计算
   ```

3. **exp_mask 创建**: `agentevolver/module/env_manager/env_manager.py:612-615`
   ```python
   if sample.extras.get("is_experience_replay", False):
       response_exp_mask_list.append(torch.ones(...))  # off-policy
   ```

4. **Loss 计算**: `agentevolver/module/exp_manager/het_core_algos.py:98-108`
   ```python
   on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)
   off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
   ```

---

## 总结

Off-policy 数据的处理机制：

1. ✅ **保持 `author="llm"`**：确保 `loss_mask=1`，参与 loss 计算
2. ✅ **使用 `exp_mask=1` 标记 off-policy**：不使用 `loss_mask=0` 区分
3. ✅ **分别计算 on/off-policy loss**：`het_compute_token_on_off_policy_loss` 使用 `exp_mask` 分别计算
4. ✅ **重要性采样**：使用 `recorded_old_log_probs` 计算重要性采样权重
5. ✅ **与 ExGRPO 一致**：参考 ExGRPO 的设计，off-policy 数据参与 loss 计算

这是 Experience Replay 机制的核心设计，确保了 off-policy 数据能够正确地参与训练 loss 计算，同时使用重要性采样校正权重。
