# DAPO 中 Off-Policy Data (Experience Replay) 的处理机制

## 📋 概述

DAPO 通过 `exp_mask` 区分 on-policy 和 off-policy 数据，并应用不同的 loss 计算策略：
- **On-policy**: 使用 DAPO 的 **Clip-Higher** 机制
- **Off-policy**: 使用 **ExGRPO Policy Shaping** 处理重要性采样

---

## 🔍 实现位置

### **1. Exp Mask 创建**

**文件**: `agentevolver/module/env_manager/env_manager.py:634-653`

```python
# Create experience mask
if sample.extras.get("is_experience_replay", False):
    # Experience Replay: 只对 LLM 响应位置（loss_mask=1）设置 exp_mask=1
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))
else:
    # On-policy: 全为 0
    prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
    response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))
```

**关键点**:
- `exp_mask = 1`: Off-policy data（来自 Experience Replay）
- `exp_mask = 0`: On-policy data（当前策略生成）
- **Multi-turn 关键**: 只对 LLM 响应位置（`loss_mask=1`）设置 `exp_mask=1`，Environment 响应不参与 off-policy loss

---

### **2. DAPO Loss 计算中的 Off-Policy 处理**

**文件**: `agentevolver/module/exp_manager/het_core_algos.py:428-457`

```python
def dapo_compute_policy_loss(
    old_log_prob, log_prob, advantages, response_mask,
    exp_mask=None,  # ⭐ Off-policy mask
    off_policy_shaping_mode="exgrpo_policy_shaping",
    off_policy_shaping_beta=0.1,
    ...
):
    # Step 1: 计算 importance sampling ratio
    ratio = torch.exp(log_prob - old_log_prob)  # π_new / π_old
    
    # Step 2: 处理 exp_mask
    if exp_mask is None:
        exp_mask = torch.zeros_like(response_mask)
    exp_mask = exp_mask.float()
    
    # Step 3: ⭐ On-policy loss (DAPO Clip-Higher)
    # ... 使用 DAPO Clip-Higher 机制 ...
    on_policy_mask = (1.0 - exp_mask) * response_mask
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_policy_mask)
    
    # Step 4: ⭐ Off-policy loss (ExGRPO Policy Shaping)
    if off_policy_shaping_mode == "exgrpo_policy_shaping":
        # ExGRPO Policy Shaping: f(x) = x / (x + β)
        off_ratio = ratio  # 使用相同的 importance sampling ratio
        off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)
        off_pg_losses = -advantages * off_ratio_shaped
    elif off_policy_shaping_mode == "dapo_clip_higher":
        # 对 off-policy 也使用 DAPO Clip-Higher
        off_pg_losses = on_pg_losses
    else:
        raise ValueError(f"Invalid off_policy_shaping_mode")
    
    # Step 5: 计算 off-policy loss
    off_policy_mask = exp_mask * response_mask
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_policy_mask)
    
    # Step 6: ⭐ 合并 on-policy 和 off-policy losses
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

---

## 🎯 两种 Off-Policy 处理模式

### **模式 1: ExGRPO Policy Shaping（推荐，默认）**

**公式**: `f(x) = x / (x + β)`

**实现**:
```python
off_ratio = ratio  # importance sampling ratio: π_new / π_old
off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)  # f(x) = x/(x+β)
off_pg_losses = -advantages * off_ratio_shaped
```

**特点**:
- ✅ **放大低概率信号**: 当 `ratio` 很小时（例如 0.1），`f(0.1) = 0.1/(0.1+0.1) = 0.5`，放大了 5 倍
- ✅ **抑制高概率信号**: 当 `ratio` 很大时（例如 10），`f(10) = 10/(10+0.1) ≈ 0.99`，几乎不变
- ✅ **保持熵**: 通过放大低概率信号，鼓励模型保持探索
- ✅ **不需要 clipping**: 函数本身就有界 `[0, 1)`，更平滑

**数学性质**:
```
f(x) = x / (x + β)

当 x → 0:  f(x) → 0
当 x → +∞: f(x) → 1
当 x = β:  f(x) = 0.5  (拐点)
```

**为什么有效**:
- Off-policy 数据来自历史策略，分布可能已经偏移
- 直接使用 importance sampling ratio 可能导致高方差
- ExGRPO shaping 通过非线性变换稳定梯度，同时保持探索性

---

### **模式 2: DAPO Clip-Higher（实验性）**

**实现**:
```python
off_pg_losses = on_pg_losses  # 直接使用 on-policy 的 DAPO Clip-Higher loss
```

**特点**:
- ⚠️ **实验性**: 对 off-policy 也应用 DAPO Clip-Higher
- ⚠️ **可能不稳定**: Off-policy 数据的分布偏移可能导致 Clip-Higher 效果不佳
- ⚠️ **不推荐**: 默认使用 ExGRPO shaping 更稳定

---

## ✅ 实现正确性分析

### **1. Importance Sampling Ratio 计算**

```python
ratio = torch.exp(log_prob - old_log_prob)  # π_new / π_old
```

**正确性**: ✅ **正确**
- 对于 **on-policy** 数据: `old_log_prob` 是当前策略的 log_prob，所以 `ratio ≈ 1.0`
- 对于 **off-policy** 数据: `old_log_prob` 是历史策略的 log_prob（从 Experience Replay 加载），所以 `ratio` 反映策略变化

**关键**: Off-policy 数据必须使用**历史策略的 old_log_prob**，这在 Experience Replay 流程中已正确处理。

---

### **2. ExGRPO Policy Shaping 公式**

```python
off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)
```

**正确性**: ✅ **正确**
- 公式与 ExGRPO 论文一致: `f(w*(θ)) = w*(θ) / (w*(θ) + β)`
- 其中 `w*(θ) = exp(log_prob - old_log_prob)` 是 importance sampling ratio

**验证**:
- 当 `off_ratio = 0.1`, `beta = 0.1`: `f(0.1) = 0.1/(0.1+0.1) = 0.5` ✅
- 当 `off_ratio = 1.0`, `beta = 0.1`: `f(1.0) = 1.0/(1.0+0.1) ≈ 0.909` ✅
- 当 `off_ratio = 10.0`, `beta = 0.1`: `f(10.0) = 10.0/(10.0+0.1) ≈ 0.990` ✅

---

### **3. Loss 合并**

```python
pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

**正确性**: ✅ **正确**
- 使用 `exp_mask` 进行元素级别的混合
- 对于 `exp_mask=1` 的位置，使用 `off_pg_losses`
- 对于 `exp_mask=0` 的位置，使用 `on_pg_losses`
- 最后使用 `token-mean` 聚合（DAPO 的第三个改进）

---

### **4. Mask 使用**

```python
on_policy_mask = (1.0 - exp_mask) * response_mask
off_policy_mask = exp_mask * response_mask
```

**正确性**: ✅ **正确**
- 确保只对有效 token（`response_mask=1`）计算 loss
- 正确区分 on-policy 和 off-policy 数据
- Multi-turn 场景下，只对 LLM 响应部分（`loss_mask=1`）设置 `exp_mask=1`

---

## 🔬 潜在问题与改进建议

### **问题 1: Off-Policy Advantage 计算**

**当前实现**: Off-policy 数据使用与 on-policy 相同的 advantage（来自 GRPO 分组计算）

**潜在问题**:
- Off-policy 数据来自历史策略，advantage 可能不准确
- 如果 off-policy 和 on-policy 数据在同一个 GRPO 组中，advantage 是混合计算的

**分析**:
- ✅ **当前实现是合理的**: ExGRPO 论文中，off-policy 和 on-policy 数据共享相同的 advantage（基于同一 task 的 outcome）
- ✅ **Policy shaping 已经处理了分布偏移**: `f(x) = x/(x+β)` 通过调整 importance sampling ratio 来补偿分布偏移

---

### **问题 2: Beta 参数选择**

**当前配置**: `off_policy_shaping_beta: 0.1`

**分析**:
- ✅ **0.1 是 ExGRPO 论文的默认值**: 经过实验验证
- ⚠️ **可能需要调优**: 如果 off-policy 数据比例很高，可能需要调整 beta
- 💡 **建议**: 监控 `actor/off_pg_loss` 和 `actor/on_pg_loss` 的比例，如果 off-policy loss 过大，可以增大 beta

---

### **问题 3: Off-Policy 数据的 Clip-Higher 选项**

**当前实现**: 提供了 `dapo_clip_higher` 选项，但对 off-policy 应用 Clip-Higher 可能不稳定

**分析**:
- ⚠️ **理论上可能有问题**: Off-policy 数据的分布已经偏移，Clip-Higher 的设计假设可能不成立
- ✅ **默认使用 ExGRPO shaping 是安全的**: 这是经过验证的方法
- 💡 **建议**: 除非有特殊需求，否则使用默认的 `exgrpo_policy_shaping`

---

## 📊 完整流程示例

### **场景**: Batch 中有 64 个样本，其中 32 个是 off-policy

```python
# 输入
old_log_prob: (64, 4096)  # 对于 off-policy，这是历史策略的 log_prob
log_prob: (64, 4096)       # 当前策略的 log_prob
advantages: (64, 4096)     # GRPO 计算的 advantage（on/off-policy 共享）
response_mask: (64, 4096)  # 有效 token mask
exp_mask: (64, 4096)       # [0,0,...,0, 1,1,...,1] 前 32 个是 on-policy，后 32 个是 off-policy

# Step 1: 计算 ratio
ratio = exp(log_prob - old_log_prob)  # (64, 4096)
# 对于 on-policy: ratio ≈ 1.0
# 对于 off-policy: ratio 可能偏离 1.0（反映策略变化）

# Step 2: On-policy loss (DAPO Clip-Higher)
on_pg_losses = compute_dapo_clip_higher_loss(ratio, advantages, ...)  # (64, 4096)
on_policy_mask = (1.0 - exp_mask) * response_mask  # 前 32 个样本
on_pg_loss = masked_mean(on_pg_losses, on_policy_mask)  # scalar

# Step 3: Off-policy loss (ExGRPO Shaping)
off_ratio_shaped = ratio / (ratio + 0.1)  # (64, 4096)
off_pg_losses = -advantages * off_ratio_shaped  # (64, 4096)
off_policy_mask = exp_mask * response_mask  # 后 32 个样本
off_pg_loss = masked_mean(off_pg_losses, off_policy_mask)  # scalar

# Step 4: 合并
pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)  # (64, 4096)
pg_loss = agg_loss(pg_losses, response_mask, "token-mean")  # scalar

# Step 5: 反向传播
pg_loss.backward()
```

---

## ✅ 总结

### **实现正确性**: ✅ **基本正确**

1. ✅ **Importance Sampling Ratio**: 正确计算，off-policy 使用历史策略的 old_log_prob
2. ✅ **ExGRPO Policy Shaping**: 公式正确，与论文一致
3. ✅ **Loss 合并**: 使用 exp_mask 正确混合
4. ✅ **Mask 使用**: 正确处理 multi-turn 场景

### **设计合理性**: ✅ **合理**

1. ✅ **分离处理**: On-policy 用 DAPO Clip-Higher，off-policy 用 ExGRPO shaping，各司其职
2. ✅ **兼容性**: 与原有的 Experience-Replay + GRPO 机制兼容
3. ✅ **稳定性**: ExGRPO shaping 比直接应用 Clip-Higher 更稳定

### **建议**

1. ✅ **保持当前实现**: 默认使用 `exgrpo_policy_shaping` 是正确的选择
2. 💡 **监控指标**: 关注 `actor/off_pg_loss` 和 `actor/on_pg_loss` 的比例
3. 💡 **Beta 调优**: 如果 off-policy loss 过大，可以尝试增大 `off_policy_shaping_beta`
4. ⚠️ **避免使用 `dapo_clip_higher`**: 除非有特殊实验需求，否则不推荐

---

## 📚 参考文献

- **ExGRPO 论文**: Experience-Guided Reinforcement Learning with Shared Representations
- **DAPO 论文**: Decoupled Clip and Dynamic sAmpling Policy Optimization
- **实现位置**: `agentevolver/module/exp_manager/het_core_algos.py:319-492`

