# Token-Level Policy Gradient: `token-mean` 实现详解

## 📍 核心实现位置

### **1. 配置设置**

**文件**: `config/alfworld_dapo_3b.yaml`

```yaml
actor_rollout_ref:
  actor:
    # ⭐ Token-Level Policy Gradient (DAPO's 3rd improvement)
    loss_agg_mode: token-mean
```

---

### **2. 核心实现函数**

**文件**: `agentevolver/module/exp_manager/het_core_algos.py:8-41`

```python
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    
    Args:
        loss_mat: shape (bs, response_length) - 每个 token 的 loss
        loss_mask: shape (bs, response_length) - mask，1 表示有效 token
        loss_agg_mode: "token-mean" | "seq-mean-token-sum" | ...
    """
    if loss_agg_mode == "token-mean":
        # ⭐ 关键：直接对所有有效 token 求平均
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        # 先对每个序列求和，再对序列求平均
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # (bs,)
        loss = torch.mean(seq_losses)  # scalar
    # ... 其他模式
    return loss
```

**`masked_mean` 实现** (verl 库):

```python
def masked_mean(values, mask, axis=None):
    """
    计算 masked 元素的平均值
    
    公式: sum(values * mask) / (sum(mask) + 1e-8)
    """
    return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)
```

---

### **3. 在 DAPO Loss 计算中的使用**

**文件**: `agentevolver/module/exp_manager/het_core_algos.py:458`

```python
def dapo_compute_policy_loss(...):
    # ... 计算每个 token 的 loss (pg_losses)
    # pg_losses shape: (batch_size, response_length)
    # 每个位置都有 loss 值（密集的）
    
    # ⭐ 关键调用：使用 token-mean 聚合
    pg_loss = agg_loss(
        loss_mat=pg_losses,           # (bs, seq_len) - 每个 token 的 loss
        loss_mask=response_mask,      # (bs, seq_len) - 有效 token mask
        loss_agg_mode=loss_agg_mode   # "token-mean"
    )
    
    return {"pg_loss": pg_loss, ...}
```

---

### **4. 在 Actor Update 中的调用链**

**文件**: `agentevolver/module/exp_manager/het_actor.py:165-178`

```python
# Step 1: 从配置读取 loss_agg_mode
loss_agg_mode = self.config.get("loss_agg_mode", "token-mean")

# Step 2: 调用 DAPO loss 计算
ret_dict = dapo_compute_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=response_mask,
    exp_mask=exp_mask,
    cliprange_low=clip_ratio_low,
    cliprange_high=clip_ratio_high,
    clip_ratio_c=clip_ratio_c,
    loss_agg_mode=loss_agg_mode,  # ⭐ 传入 "token-mean"
    ...
)

# Step 3: 获取聚合后的 loss
pg_loss = ret_dict["pg_loss"]  # scalar
```

---

## 🔍 具体计算过程

### **示例：理解 `token-mean` vs `seq-mean-token-sum`**

假设有 3 个样本：

```python
# 输入数据
pg_losses = torch.tensor([
    [0.1, 0.2, 0.3, 0.0, 0.0],  # 样本 1: 3 个有效 token
    [0.4, 0.5, 0.0, 0.0, 0.0],  # 样本 2: 2 个有效 token
    [0.6, 0.7, 0.8, 0.9, 1.0],  # 样本 3: 5 个有效 token
])  # shape: (3, 5)

response_mask = torch.tensor([
    [1, 1, 1, 0, 0],  # 样本 1
    [1, 1, 0, 0, 0],  # 样本 2
    [1, 1, 1, 1, 1],  # 样本 3
])  # shape: (3, 5)
```

#### **模式 1: `token-mean` (DAPO 使用)**

```python
# 计算过程
total_loss = (pg_losses * response_mask).sum()  # 0.1+0.2+0.3 + 0.4+0.5 + 0.6+0.7+0.8+0.9+1.0 = 4.5
total_tokens = response_mask.sum()  # 3 + 2 + 5 = 10
loss = total_loss / total_tokens  # 4.5 / 10 = 0.45
```

**特点**:
- ✅ **每个 token 贡献相等**：无论来自哪个序列
- ✅ **短序列不会被过度加权**：样本 2 只有 2 个 token，但每个 token 的权重和其他样本相同
- ✅ **适合长推理链场景**：不同序列长度差异大时，避免短序列主导训练

#### **模式 2: `seq-mean-token-sum` (传统方式)**

```python
# 计算过程
seq_losses = (pg_losses * response_mask).sum(dim=-1)  # [0.6, 0.9, 4.0]
loss = seq_losses.mean()  # (0.6 + 0.9 + 4.0) / 3 = 1.83
```

**特点**:
- ❌ **每个序列贡献相等**：无论序列长短
- ❌ **短序列被过度加权**：样本 2 只有 2 个 token，但和样本 3（5 个 token）权重相同
- ❌ **不适合长推理链**：短序列的每个 token 实际上被赋予了更高的权重

---

## 📊 数学公式对比

### **`token-mean` (DAPO)**

```
L = (1 / N_total) * Σ_{i,j} L_{i,j} * mask_{i,j}

其中:
- N_total = Σ_{i,j} mask_{i,j}  (所有有效 token 的总数)
- L_{i,j}: 样本 i 的 token j 的 loss
- mask_{i,j}: 1 表示有效 token，0 表示 padding
```

**含义**: 所有有效 token 的 loss 直接平均，**不考虑序列边界**。

### **`seq-mean-token-sum` (传统)**

```
L = (1 / N_seq) * Σ_i [Σ_j L_{i,j} * mask_{i,j}]

其中:
- N_seq: 序列数量（batch size）
- 先对每个序列求和，再对序列求平均
```

**含义**: 每个序列的 loss 总和先计算，然后序列之间平均，**序列长度影响权重**。

---

## 🎯 为什么 DAPO 使用 `token-mean`？

### **问题场景**

在长推理链任务（如 AlfWorld）中：
- 不同样本的响应长度差异很大：短的 200 tokens，长的 4000 tokens
- 如果使用 `seq-mean-token-sum`：
  - 短序列（200 tokens）和长序列（4000 tokens）的 loss 权重相同
  - 但短序列的每个 token 实际上被赋予了 **20 倍**的权重（4000/200）
  - 这会导致模型偏向生成短响应

### **DAPO 的解决方案**

使用 `token-mean`：
- 每个 token 的 loss 贡献相等
- 长序列自然有更多 token，所以总贡献更大
- 但每个 token 的权重相同，避免了短序列被过度加权

---

## 🔗 完整调用链

```
配置 (yaml)
  ↓
loss_agg_mode: "token-mean"
  ↓
het_actor.py:174
  ↓ loss_agg_mode 参数传递
dapo_compute_policy_loss(..., loss_agg_mode="token-mean")
  ↓
het_core_algos.py:458
  ↓
agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="token-mean")
  ↓
het_core_algos.py:24
  ↓
verl_F.masked_mean(loss_mat, loss_mask)
  ↓
verl 库实现: (loss_mat * mask).sum() / (mask.sum() + 1e-8)
  ↓
返回: scalar loss
```

---

## 💡 关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 配置设置 | `config/alfworld_dapo_3b.yaml` | 113 |
| 核心实现 | `het_core_algos.py` | 8-41 |
| DAPO 调用 | `het_core_algos.py` | 458 |
| Actor 调用 | `het_actor.py` | 174 |
| 验证检查 | `ae_ray_trainer.py` | 749-754 |

---

## ✅ 总结

**`token-mean` 在 DAPO 中的体现**:

1. **配置层面**: `config/alfworld_dapo_3b.yaml:113` 设置 `loss_agg_mode: token-mean`
2. **实现层面**: `het_core_algos.py:24` 使用 `masked_mean` 直接对所有有效 token 求平均
3. **调用层面**: `dapo_compute_policy_loss()` 在聚合 loss 时调用 `agg_loss(..., loss_agg_mode="token-mean")`
4. **效果**: 确保每个 token 对 loss 的贡献相等，避免短序列被过度加权

这就是 DAPO 的 **Token-Level Policy Gradient** 改进的核心实现！

