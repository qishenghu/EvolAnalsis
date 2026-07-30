# DAPO 算法集成指南

## 概述

DAPO（**D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization）是由清华大学人工智能研究院（AIR）与字节跳动联合实验室共同开发的强化学习算法，是对GRPO（Group Relative Policy Optimization）算法的重要改进。

DAPO通过四项关键技术解决了GRPO在训练过程中可能面临的熵崩溃、奖励噪声和训练不稳定等问题：

1. **Clip-Higher**：解耦上下剪辑范围，防止熵崩溃
2. **Dynamic Sampling**：动态过滤无效梯度样本
3. **Token-Level Policy Gradient Loss**：Token级别策略梯度
4. **Overlong Reward Shaping**：截断样本的软惩罚

## 算法原理

### 1. Clip-Higher（解耦非对称剪辑）

**问题**：标准PPO/GRPO使用对称的剪辑范围 `[1-ε, 1+ε]`，这在长期训练中可能导致熵崩溃（entropy collapse），即模型倾向于只生成高概率token，失去探索能力。

**解决方案**：DAPO解耦上下剪辑边界：

```
对于 A > 0（鼓励的action）: clip ratio 到 [1-ε_low, 1+ε_high]
对于 A < 0（抑制的action）: clip ratio 到 [1-ε_low, +∞) —— 移除上界
```

**原理**：
- 当优势 A > 0 时，保持正常剪辑，防止概率增加过快
- 当优势 A < 0 时，移除上界限制，允许低概率token进一步降低概率
- 这种非对称设计鼓励模型探索低概率token，防止熵崩溃

**数学公式**：

```
L_clip = E[min(r(θ)A, clip(r(θ), 1-ε_low, 1+ε_high)A)]  当 A ≥ 0
L_clip = E[min(r(θ)A, clip(r(θ), 1-ε_low, +∞)A)]        当 A < 0
```

其中 `r(θ) = π_θ(a|s) / π_θ_old(a|s)` 是重要性采样比率。

### 2. Dynamic Sampling（动态采样）

**问题**：在GRPO中，如果某个prompt的所有rollout结果完全相同（全对或全错），则组内优势归一化后方差为0，无法产生有效梯度。

**解决方案**：在训练前过滤掉这些无效样本组：

```
过滤条件：
- 准确率 = 1.0（所有rollout都正确）→ 过滤
- 准确率 = 0.0（所有rollout都错误）→ 过滤
```

**实现方式**：
1. 计算每个prompt组的成功率
2. 标记需要过滤的样本
3. 将被过滤样本的优势值设为0（保持batch结构不变）

### 3. Token-Level Policy Gradient Loss

**问题**：在长推理链场景中，不同样本的响应长度差异很大，使用序列级别的loss聚合可能导致短序列被过度加权。

**解决方案**：在token级别计算策略梯度损失，通过 `loss_agg_mode: token-mean` 实现：

```python
loss = masked_mean(token_losses, response_mask)  # Token级别平均
```

这确保每个token对loss的贡献相等，而不是每个序列贡献相等。

### 4. Overlong Reward Shaping（截断奖励塑造）

**问题**：当响应被截断时（达到最大长度），通常会给予0奖励或负奖励，这引入了奖励噪声，因为截断点可能是任意的。

**解决方案**：对截断样本应用软惩罚，而不是硬性失败：

```python
# 软惩罚模式
reward_shaped = reward + truncation_penalty  # additive
reward_shaped = reward * (1 + truncation_penalty)  # multiplicative
```

这样可以：
- 保留部分正确轨迹的学习信号
- 适度惩罚过长响应
- 减少由任意截断点带来的奖励噪声

## 配置说明

### 完整配置示例

```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  
  # DAPO 配置
  dapo:
    enable: true  # 主开关
    
    # Dynamic Sampling 配置
    dynamic_sampling:
      enable: true
      filter_mode: "strict"  # 过滤模式
    
    # Overlong Reward Shaping 配置
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.5
      soft_penalty_mode: "additive"

actor_rollout_ref:
  actor:
    # Clip-Higher 配置
    use_dapo: true  # 启用 DAPO 的 Clip-Higher
    clip_ratio_low: 0.2   # ε_low (下界)
    clip_ratio_high: 0.28  # ε_high (上界，仅用于 A > 0)
    
    # Token-Level PG 配置
    loss_agg_mode: token-mean
```

### 配置参数详解

#### `algorithm.dapo`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | DAPO 主开关 |

#### `algorithm.dapo.dynamic_sampling`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | 启用动态采样过滤 |
| `filter_mode` | str | `"strict"` | 过滤模式 |

**filter_mode 选项**：
- `"strict"`: 过滤全对(acc=1)或全错(acc=0)的样本组（推荐）
- `"remove_all_correct"`: 仅过滤全对的样本组
- `"remove_all_incorrect"`: 仅过滤全错的样本组
- `"none"`: 不过滤

#### `algorithm.dapo.overlong_reward_shaping`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `false` | 启用截断奖励塑造 |
| `truncation_penalty` | float | `-0.5` | 截断惩罚值 |
| `soft_penalty_mode` | str | `"additive"` | 惩罚应用方式 |

**soft_penalty_mode 选项**：
- `"additive"`: `reward = reward + penalty`
- `"multiplicative"`: `reward = reward * (1 + penalty)`
- `"replace_if_positive"`: 如果截断且reward>0，则 `reward = penalty`
- `"cap"`: 将正奖励限制在 penalty 值

#### `actor_rollout_ref.actor`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_dapo` | bool | `false` | 启用 Clip-Higher 机制 |
| `clip_ratio_low` | float | `0.2` | ε_low，剪辑下界 |
| `clip_ratio_high` | float | `0.28` | ε_high，剪辑上界（仅用于 A>0） |
| `loss_agg_mode` | str | `"token-mean"` | Loss聚合模式 |

## 使用方法

### 方式一：使用预配置的DAPO配置文件

```bash
python launcher.py --config-name alfworld_dapo_3b
```

### 方式二：在现有配置上启用DAPO

在你的配置文件中添加以下内容：

```yaml
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: "strict"
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.5
      soft_penalty_mode: "additive"

actor_rollout_ref:
  actor:
    use_dapo: true
```

### 方式三：仅启用部分DAPO功能

你可以根据需要单独启用DAPO的各项功能：

```yaml
# 仅启用 Clip-Higher
actor_rollout_ref:
  actor:
    use_dapo: true

# 仅启用 Dynamic Sampling
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: "strict"
    overlong_reward_shaping:
      enable: false

# 仅启用 Overlong Reward Shaping
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: false
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.5
```

## 监控指标

启用DAPO后，训练过程中会记录以下额外指标：

### Dynamic Sampling 指标

| 指标 | 说明 |
|------|------|
| `dapo/num_filtered_samples` | 被过滤的样本数量 |
| `dapo/filter_ratio` | 过滤比例 |

### Overlong Reward Shaping 指标

| 指标 | 说明 |
|------|------|
| `dapo/num_truncated_samples` | 被截断的样本数量 |
| `dapo/truncation_ratio` | 截断比例 |
| `dapo/avg_truncation_penalty_applied` | 平均应用的截断惩罚 |

### Clip-Higher 指标

| 指标 | 说明 |
|------|------|
| `actor/on_pg_clipfrac` | 上界剪辑比例 |
| `actor/on_pg_clipfrac_lower` | 下界剪辑比例 |

## Multi-turn 对话场景支持

DAPO完全支持multi-turn对话形式的RL训练。以下是各组件对multi-turn的处理方式：

### Clip-Higher（Loss计算）

在multi-turn场景中，`response_mask`会被替换为`loss_mask`，确保只在LLM生成的token上计算loss：

```python
# het_actor.py 中的处理
if multi_turn:
    response_mask = data["loss_mask"][:, -response_length:]
else:
    response_mask = attention_mask[:, -response_length:]

# 传递给 DAPO
ret_dict = dapo_compute_policy_loss(
    ...,
    response_mask=response_mask,  # multi-turn时为loss_mask
    ...
)
```

这确保了：
- **环境反馈token**（如观察结果）不参与loss计算
- **只有LLM生成的action token**会被优化
- 剪辑统计量正确反映LLM生成部分的行为

### Dynamic Sampling

Dynamic Sampling基于轨迹级别的reward进行过滤，与multi-turn兼容：

```python
# 基于每个轨迹的总reward判断
filter_rewards = batch.batch["token_level_rewards"].sum(dim=-1)

# 按group_id (uid) 分组判断
# 同一prompt的所有rollouts会被一起评估
```

### Overlong Reward Shaping（Multi-turn感知）

在multi-turn场景中，截断判断使用双重检测机制：

1. **长度截断**：响应是否达到`max_response_length`
2. **终止状态**：轨迹的`is_terminated`标志

```python
# 方法1: 检查响应是否达到最大长度
is_truncated_by_length = (actual_response_lengths >= max_response_length - 1)

# 方法2: 检查轨迹是否正常终止（multi-turn关键）
# 如果轨迹未终止(is_terminated=False)，说明被截断
for idx, traj in enumerate(trajectories):
    if not getattr(traj, 'is_terminated', True):
        is_truncated_by_termination[idx] = True

# 两种情况都视为截断
is_truncated = is_truncated_by_length | is_truncated_by_termination
```

这种设计特别适合Agent任务，因为：
- 任务可能因达到`max_steps`而终止（但未完成目标）
- 响应可能因长度限制而被截断
- 两种情况都应该应用软惩罚

### Multi-turn相关监控指标

启用DAPO后，会记录额外的multi-turn感知指标：

| 指标 | 说明 |
|------|------|
| `dapo/num_truncated_by_length` | 因长度限制被截断的样本数 |
| `dapo/num_truncated_by_termination` | 因未正常终止被识别为截断的样本数 |

## 代码结构

DAPO的实现分布在以下文件中：

```
agentevolver/
├── module/
│   ├── exp_manager/
│   │   ├── het_core_algos.py      # DAPO核心算法实现
│   │   │   ├── dapo_compute_policy_loss()      # Clip-Higher
│   │   │   ├── dapo_filter_samples()           # Dynamic Sampling
│   │   │   └── dapo_overlong_reward_shaping()  # Overlong Reward Shaping
│   │   └── het_actor.py           # Actor中集成DAPO
│   └── trainer/
│       └── ae_ray_trainer.py      # 训练器中集成DAPO
config/
├── agentevolver.yaml              # 基础配置（含DAPO选项）
├── alfworld_grpo_3b.yaml          # GRPO配置（含DAPO选项）
└── alfworld_dapo_3b.yaml          # DAPO完整启用配置
```

## 与GRPO的兼容性

DAPO的所有改进都是**可选的**，通过配置开关控制：

- 默认情况下，所有DAPO功能都是**关闭的**
- 原有的GRPO算法**完全不受影响**
- 你可以随时在GRPO和DAPO之间切换

```yaml
# 使用原始GRPO（默认）
algorithm:
  dapo:
    enable: false

actor_rollout_ref:
  actor:
    use_dapo: false

# 使用DAPO
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
    overlong_reward_shaping:
      enable: true

actor_rollout_ref:
  actor:
    use_dapo: true
```

## Experience-Replay 兼容性

DAPO完全兼容Experience-Replay机制。当同时启用Experience-Replay和DAPO时：

### 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│              Experience-Replay + DAPO Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  数据来源:                                                        │
│  ├── On-policy: 当前策略生成的轨迹 (exp_mask = 0)                 │
│  └── Off-policy: 经验池中的历史轨迹 (exp_mask = 1)                │
│                                                                  │
│  Loss 计算:                                                       │
│  ├── On-policy → DAPO Clip-Higher 机制                           │
│  │   ├── A > 0: clip to [1-ε_low, 1+ε_high]                      │
│  │   └── A < 0: clip to [1-ε_low, +∞) (移除上界)                  │
│  │                                                               │
│  └── Off-policy → ExGRPO Policy Shaping                          │
│      └── f(x) = x / (x + β) 处理重要性采样比率                    │
│                                                                  │
│  合并: pg_loss = off_loss * exp_mask + on_loss * (1 - exp_mask)  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计

1. **On-policy 数据**（`exp_mask = 0`）：
   - 使用 DAPO 的 **Clip-Higher** 机制
   - 对正优势使用正常剪辑，对负优势移除上界限制
   - 这鼓励探索低概率token，防止熵崩溃

2. **Off-policy 数据**（`exp_mask = 1`）：
   - 使用 **ExGRPO Policy Shaping**：`f(x) = x / (x + β)`
   - 这与原有的 Experience-Replay + GRPO 行为一致
   - 处理由于策略变化导致的分布偏移

3. **Loss 合并**：
   - 根据 `exp_mask` 自动选择正确的 loss 计算方式
   - 最终 loss 是 on-policy 和 off-policy loss 的加权组合

### 配置示例

```yaml
# Experience-Replay + DAPO 完整配置
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: "strict"
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.5

actor_rollout_ref:
  actor:
    use_dapo: true  # 启用 DAPO Clip-Higher (on-policy)
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    # Off-policy 设置 (用于 Experience-Replay)
    off_policy_shaping_mode: "exgrpo_policy_shaping"  # 推荐
    off_policy_shaping_beta: 0.1

exp_manager:
  experience_replay:
    enable: true
    replay_start_ratio: 0.1
    exp_ratio: 0.5
```

### Off-policy 处理模式

| 模式 | 说明 | 推荐场景 |
|------|------|---------|
| `exgrpo_policy_shaping` | 使用 `f(x)=x/(x+β)` 塑形，与原有 Experience-Replay 行为一致 | **推荐**，兼容性最好 |
| `dapo_clip_higher` | 对 off-policy 也使用 DAPO Clip-Higher | 实验性，可能不稳定 |

### 监控指标

启用 Experience-Replay + DAPO 后的额外指标：

| 指标 | 说明 |
|------|------|
| `actor/on_pg_loss` | On-policy 数据的 loss (DAPO Clip-Higher) |
| `actor/off_pg_loss` | Off-policy 数据的 loss (ExGRPO shaping) |
| `exp_mask_ratio` | Off-policy 数据在 batch 中的比例 |
| `actor/on_pg_clipfrac` | On-policy 数据的剪辑比例 |

### 注意事项

1. **Dynamic Sampling 与 Experience-Replay**：
   - Dynamic Sampling 基于 GRPO 的 `uid`（group_id）分组
   - 同一 task 的 **on-policy 和 off-policy 轨迹共享相同的 uid**
   - 过滤判断基于**整个组**（包含 on+off policy）的 success_rate：
   
   ```
   示例场景（n_rollout=8, off-policy=1）:
   ┌─────────────────────────────────────────────────┐
   │ Task A:                                         │
   │   - 1 off-policy 成功轨迹 (reward > 0)          │
   │   - 7 on-policy 失败轨迹 (reward = 0)           │
   │   → success_rate = 1/8 = 0.125                  │
   │   → 不会被过滤 ✓ (有学习信号)                    │
   ├─────────────────────────────────────────────────┤
   │ Task B:                                         │
   │   - 1 off-policy 成功轨迹                       │
   │   - 7 on-policy 成功轨迹                        │
   │   → success_rate = 8/8 = 1.0                    │
   │   → 被过滤 ✗ (全部成功，无梯度)                  │
   └─────────────────────────────────────────────────┘
   ```
   
   - 这是**期望的行为**：Experience-Replay 引入成功轨迹是为了提供学习信号
   - 如果 on-policy 全部失败，off-policy 的成功轨迹会阻止该组被过滤

2. **Overlong Reward Shaping 与 Off-policy**：
   - Off-policy 轨迹也会被检查是否截断
   - 使用 `is_terminated` 标志进行判断
   - 对于从经验池加载的轨迹，`is_terminated` 状态会被保留

3. **old_log_prob 处理**：
   - Off-policy 数据使用**历史策略的 old_log_prob**
   - 这是正确的重要性采样所需的
   - 通过 `recorded_old_log_probs` 字段实现
   - DAPO 的 ExGRPO shaping 会正确处理这种分布偏移

4. **Clip-Higher 只影响 On-policy**：
   - DAPO 的非对称剪辑（Clip-Higher）**只应用于 on-policy 数据**
   - Off-policy 数据使用 ExGRPO policy shaping，与原有行为一致
   - 这确保了 Experience-Replay 的稳定性

## 推荐配置

### 数学推理任务（如AIME）

```yaml
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: "strict"  # 过滤全对/全错
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.5
      soft_penalty_mode: "additive"

actor_rollout_ref:
  actor:
    use_dapo: true
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    loss_agg_mode: token-mean
```

### Agent任务（如AlfWorld）

```yaml
algorithm:
  dapo:
    enable: true
    dynamic_sampling:
      enable: true
      filter_mode: "strict"
    overlong_reward_shaping:
      enable: true
      truncation_penalty: -0.3  # Agent任务可能需要较长响应
      soft_penalty_mode: "additive"

actor_rollout_ref:
  actor:
    use_dapo: true
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

## DAPO 完整执行流程

以下是DAPO在训练过程中的完整执行流程，可用于验证算法是否正确运行：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DAPO Training Flow                                  │
│                    (Pure On-Policy, No Experience Replay)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [1] Rollout Phase (ae_ray_trainer.py:1540-1660)                            │
│  ├── Generate trajectories via env_manager.rollout()                        │
│  └── Convert to DataProto via env_manager.to_dataproto()                    │
│      └── exp_mask: 全部为 0 (纯 on-policy)                                   │
│                                                                              │
│  [2] Reward Computation (ae_ray_trainer.py:1726-1735)                       │
│  └── compute_reward() → reward_tensor                                       │
│                                                                              │
│  [3] ⭐ DAPO Overlong Reward Shaping (ae_ray_trainer.py:1895-1960)          │
│  ├── 检查配置: algorithm.dapo.enable && overlong_reward_shaping.enable      │
│  ├── 检测截断:                                                               │
│  │   ├── 方法1: response_length >= max_response_length - 1                  │
│  │   └── 方法2: trajectory.is_terminated == False                           │
│  └── 应用软惩罚: reward = reward + truncation_penalty                        │
│                                                                              │
│  [4] ⭐ DAPO Dynamic Sampling (ae_ray_trainer.py:1972-2008)                  │
│  ├── 检查配置: algorithm.dapo.enable && dynamic_sampling.enable             │
│  ├── 按 uid (group_id) 分组计算 success_rate                                 │
│  ├── 过滤条件 (filter_mode="strict"):                                        │
│  │   └── success_rate == 0.0 或 success_rate == 1.0                          │
│  └── 设置 batch.batch["dapo_keep_mask"]                                      │
│                                                                              │
│  [5] Advantage Computation (ae_ray_trainer.py:2020-2029)                    │
│  └── compute_advantage() with GRPO estimator                                │
│                                                                              │
│  [6] ⭐ Apply Dynamic Sampling Mask (ae_ray_trainer.py:2034-2041)           │
│  └── advantages = advantages * dapo_keep_mask (过滤样本的 advantage 归零)   │
│                                                                              │
│  [7] Actor Update (ae_ray_trainer.py:2077-2081)                             │
│  ├── batch.meta_info["multi_turn"] = multi_turn.enable                      │
│  └── actor_rollout_wg.update_actor(batch)                                   │
│                                                                              │
│  [8] ⭐ DAPO Clip-Higher Loss (het_actor.py:162-178)                        │
│  ├── 检查配置: actor.use_dapo == true                                        │
│  ├── 调用 dapo_compute_policy_loss()                                         │
│  │   ├── On-policy (exp_mask=0):                                            │
│  │   │   ├── A > 0: clip to [1-ε_low, 1+ε_high]                             │
│  │   │   └── A < 0: clip to [1-ε_low, clip_ratio_c] (移除上界)               │
│  │   └── Off-policy (exp_mask=1): 不适用 (纯 on-policy 场景)                 │
│  └── 计算 pg_loss 并反向传播                                                 │
│                                                                              │
│  [9] ⭐ Token-Level Aggregation (het_core_algos.py:agg_loss)                │
│  └── loss_agg_mode="token-mean" → 每个 token 等权贡献                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 关键配置检查清单

启用DAPO时，请确保以下配置正确设置：

| 配置项 | 位置 | 值 | 作用 |
|--------|------|-----|------|
| `algorithm.dapo.enable` | algorithm | `true` | 主开关：启用 Dynamic Sampling 和 Overlong Reward Shaping |
| `algorithm.dapo.dynamic_sampling.enable` | algorithm | `true` | 启用 Dynamic Sampling |
| `algorithm.dapo.dynamic_sampling.filter_mode` | algorithm | `"strict"` | 过滤模式 |
| `algorithm.dapo.overlong_reward_shaping.enable` | algorithm | `true` | 启用 Overlong Reward Shaping |
| `algorithm.dapo.overlong_reward_shaping.truncation_penalty` | algorithm | `-0.5` | 截断惩罚值 |
| `actor_rollout_ref.actor.use_dapo` | actor | `true` | 启用 Clip-Higher 机制 |
| `actor_rollout_ref.actor.clip_ratio_low` | actor | `0.2` | ε_low |
| `actor_rollout_ref.actor.clip_ratio_high` | actor | `0.28` | ε_high |
| `actor_rollout_ref.actor.loss_agg_mode` | actor | `"token-mean"` | Token-Level PG |
| `actor_rollout_ref.rollout.multi_turn.enable` | rollout | `true` | Multi-turn 模式 |

### 验证DAPO是否正确运行

训练时观察以下日志和指标：

```bash
# 1. Dynamic Sampling 日志
DAPO Dynamic Sampling: Filtered X samples with strict mode

# 2. 监控指标 (wandb/tensorboard)
dapo/num_filtered_samples     # 被过滤的样本数
dapo/filter_ratio             # 过滤比例
dapo/num_truncated_samples    # 截断样本数
dapo/truncation_ratio         # 截断比例
dapo/avg_truncation_penalty_applied  # 平均截断惩罚

# 3. Actor 指标
actor/on_pg_clipfrac          # Clip-Higher 上界剪辑比例
actor/on_pg_clipfrac_lower    # Clip-Higher 下界剪辑比例
actor/pg_loss                 # 策略损失
```

### 纯 On-Policy DAPO 的特殊行为

当禁用 Experience Replay（`experience_replay.enable: false`）时：

1. **exp_mask 全为 0**：所有数据都是 on-policy
2. **Clip-Higher 全覆盖**：所有 token 都使用 DAPO 的非对称剪辑
3. **off_pg_loss = 0**：没有 off-policy loss
4. **Dynamic Sampling**：基于当前 rollout 的 success_rate 过滤

## 参考文献

1. DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization
   - 开发者：清华大学AIR & 字节跳动SIA Lab
   - 成果：在AIME 2024数学竞赛中使用Qwen2.5-32B达到50分

2. GRPO: Group Relative Policy Optimization
   - DeepSeek-R1相关工作

3. PPO: Proximal Policy Optimization
   - Schulman et al., 2017

