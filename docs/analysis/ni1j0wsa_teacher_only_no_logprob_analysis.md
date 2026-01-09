# ni1j0wsa（LUFFY teacher-only, no logprob）现象解读：reward 先超后落 + 熵先降后升再塌陷

> 目标：用 **可复现指标** + **数学关系** 解释 `ni1j0wsa` 的三类现象：  
> 1) reward 早期/中期超过 GRPO，后期被 GRPO 反超  
> 2) 熵（entropy/entropy_loss）先快速下降，中期回升，中后期显著下降（塌陷趋势）  
> 3) teacher experience 在训练流程中的正负作用如何理解

---

## 0. 实验与数据来源

- **teacher run**：`ni1j0wsa`（teacher-only，`use_log_prob=false`）  
  - on-policy reward 指标：`critic/rewards_onpolicy/mean`
- **GRPO baseline**：`9ggix50f`  
  - reward 指标：`critic/rewards/mean`
- **moving average**：窗口 $w=10$（用于减少单点波动）
- **本地 trajectory 目录**：  
  - `checkpoints/agentevolver/alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_confidence_analysis/Trajectory/`
  - 包含：每步的 `batch_diag_step_*.json`、汇总 `batch_diag_compiled.csv`、以及 `trajectories_step_*.jsonl`
- **本次 run 新增诊断指标**（由代码注入）：
  - batch-level：`diag/*`（group gap、按类型的 adv/entropy、teacher token 比例等）
  - loss-side：`teacher_diag/*`（ratio/adv 分布；本 run 因为 state=crashed，history 列不完全稳定，但本地 `batch_diag_compiled.csv` 已足够解释主现象）

---

## 1. 关键观测（用指标“钉住”现象）

### 1.1 reward “先超后落”的时间区间（证据）

对 reward 曲线做 $w=10$ 滑动平均后：

- **首次超过 GRPO**：约在 `step ≈ 22`
- **最后一次仍高于 GRPO**：约在 `step ≈ 59`
- **LUFFY(on-policy) 峰值**：约 `0.6125 @ step 37`
- **GRPO 峰值**：约 `0.7344 @ step 91`

这说明：teacher experience 的确带来 **前中期加速**，但并没有把最终最优提升到 GRPO 的水平（反而 GRPO 在后期继续爬升）。

### 1.2 teacher 在 group 内长期“抬高 baseline”（证据）

从本地 `batch_diag_compiled.csv`（等价于 `diag/group_teacher_minus_on_reward_mean`）：

- $ \mathbb{E}[\text{teacher\_reward} - \text{on\_reward} \mid \text{within group}] $ **长期为正**
- 数值大致在 **[0.38, 0.78]** 之间波动（越大代表 teacher 相对更强）

直觉含义：**teacher rollout 在每个 task group 内经常是“最强样本”**，从而持续把 group baseline 抬高。

### 1.3 on-policy advantage 被系统性压低（证据）

从 `diag/adv_onpolicy_token_mean` 与 `diag/adv_teacher_token_mean`：

- `adv_onpolicy_token_mean` **全程为负**（约 $-0.07$ 到 $-0.033$）
- `adv_teacher_token_mean` **全程为正**（约 $0.8$ 到 $1.8$）

这是解释后续 “熵塌陷 + 后期 reward 不如 GRPO” 的核心证据链：  
**组内相对优势被 teacher “占据”，on-policy 在组内成为被惩罚的一方。**

### 1.4 teacher token 比例很小但“影响很大”（证据）

`diag/teacher_token_ratio`（token-level）：

- 均值约 $\approx 0.9\%$
- 范围约 $[0.7\%, 1.1\%]$

虽然 teacher token 只占很小比例，但它们拥有 **大且持续为正的 advantage**，并且还会改变 **GRPO 的组内 baseline**（影响所有 rollouts 的 advantage），因此整体影响可以很显著。

---

## 2. 用数学形式化：为什么 teacher 会“抬 baseline → 压 on-policy advantage”

下面用最小化假设把机制讲清楚。

### 2.1 GRPO 的组内优势定义

对同一 task 的一组 rollouts（大小 $n$），令每条 rollout 的回报为 $R_i$，组内平均为：

$$
\bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j
$$

典型 GRPO 的相对优势可以抽象为：

$$
A_i = R_i - \bar{R}
$$

（实际实现可能还有标准化/截断/GAE 等，但 **组内去均值** 是关键结构。）

### 2.2 引入 teacher 混入后的 baseline 上移

设每组里混入 $k$ 条 teacher rollouts，剩余 $n-k$ 条为 on-policy。令：

- teacher 平均回报：$\mu_T$
- on-policy 平均回报：$\mu_O$

则组均值（baseline）为：

$$
\bar{R}
= \frac{k\mu_T + (n-k)\mu_O}{n}
= \mu_O + \frac{k}{n}(\mu_T - \mu_O)
$$

当 teacher 更强（$\mu_T>\mu_O$）时，baseline **相对 on-policy 的上移量**为：

$$
\Delta \triangleq \bar{R} - \mu_O = \frac{k}{n}(\mu_T-\mu_O) > 0
$$

这直接导致 on-policy 的组内优势期望：

$$
\mathbb{E}[A_O]
= \mu_O - \bar{R}
= -\Delta
= -\frac{k}{n}(\mu_T-\mu_O) < 0
$$

**结论**：只要 teacher 平均回报高于 on-policy，且每组混入 $k>0$，那么 on-policy 在组内的期望优势就是负的（被系统性压低）。

> 这与本次 run 的观测完全一致：`diag/group_teacher_minus_on_reward_mean` 长期为正，且 `diag/adv_onpolicy_token_mean` 长期为负。

### 2.3 “为什么 teacher token 比例很小也会造成大影响？”

注意：teacher 的影响有两层：

1) **直接梯度贡献**：teacher token 自身参与 policy gradient（teacher advantage 往往大且为正）  
2) **间接 baseline 影响**：teacher 改变了 $\bar R$，从而改变了每条 on-policy rollout 的 $A_i$

第二层是 "以小搏大" 的关键：即使 teacher token 比例小，只要它们对应的 rollout 回报显著更高，就会在组内把 $\bar R$ 拉上去，影响所有 on-policy 的优势符号与幅度。

---

## 3. 用数学解释 reward “先超后落”

### 3.1 早期/中期：teacher 提供稳定的正向“方向导引”

以 PPO/GRPO 的 policy gradient 形式（省略 clip/ratio 细节）为例：

$$
\nabla_\theta \mathcal{L}_{PG}
\propto
\mathbb{E}\left[\sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$

teacher rollout 往往有：

- $A_T$ 大且正（本 run `diag/adv_teacher_token_mean` 长期为正且显著大于 0）
- 因此 teacher 对更新方向的贡献稳定且强

从优化角度看，它相当于把策略快速推向 teacher 轨迹附近的参数区域，使得 **样本效率提升**，因此 reward 在 `step≈22~59` 超过 GRPO 是合理的。

### 3.2 后期：探索被压制，难以超过“teacher 所覆盖的策略子空间”

上一节已经推出：

$$
\mathbb{E}[A_O] = -\frac{k}{n}(\mu_T-\mu_O)
$$

这意味着 **探索型的 on-policy 行为**（通常短期回报更不稳定）在组内更容易被判为负 advantage，从而在更新中被压制。

在 RL 中，后期进一步提升往往依赖：

- 发现 teacher 轨迹之外的更优策略（或更稳健策略）
- 或在更复杂状态分布下学会处理“长尾失败模式”

而 baseline 被长期抬高会让策略更倾向于收缩在少数高概率模式里（见下一节的熵分析），从而：

- 前中期：快速逼近 teacher 能力范围（reward 提升快）
- 后期：探索不足导致边际收益下降（reward 变慢甚至回落），被 GRPO（保持更强探索）反超

与观测一致：GRPO 峰值出现在更靠后的 `step≈91`，而 teacher run 峰值更早（`step≈37`）。

---

## 4. 用数学解释熵 “先降 → 回升 → 再塌陷”

### 4.1 熵项的梯度方向（为什么会“收缩”）

常见 entropy 正则（最大化熵）写为：

$$
\mathcal{L} = \mathcal{L}_{PG} - \alpha \, \mathbb{E}[H(\pi_\theta(\cdot|s))]
$$

其中 $H(\pi)$ 越大越"探索"。当 policy gradient 部分对某些动作给出持续正 advantage 时，$\pi_\theta$ 会对这些动作 **提高概率**，从而自然降低熵（更确定）。

在本 run 中，teacher 的优势长期强正，而 on-policy 优势长期偏负，这相当于在组内持续施加“朝 teacher 动作集中”的梯度压力 → **早期熵快速下降**是自然结果。

### 4.2 中期回升：baseline 压力减轻 + 训练噪声/正则的再扩散

当模型逐步学到一部分成功策略后：

- on-policy 的回报 $\mu_O$ 上升
- teacher 与 on-policy 的差距 $(\mu_T-\mu_O)$ 可能阶段性缩小
- 则 baseline 上移量 $\Delta=\frac{k}{n}(\mu_T-\mu_O)$ 变小

在这种情况下，on-policy 的负 advantage 压力减轻，entropy 正则和训练噪声可能让策略出现一定“再扩散” → **中期熵回升**。

这与你的本地 rollouts 汇总一致：例如在 `step 40` 周期，reward 较高且 `entropy_mean` 也相对更高（相比 `step 80` 的塌陷段）。

### 4.3 中后期再度下滑/塌陷：长期负优势使探索变成“被惩罚对象”

当 teacher 仍然长期强于 on-policy（观测：`diag/group_teacher_minus_on_reward_mean` 始终为正且不小），则：

- $\Delta$ 不会消失
- $\mathbb{E}[A_O]$ 长期为负

这会导致：

1) on-policy 中“探索导致的短期失败”被更强烈惩罚（优势更负）
2) 策略逐渐只保留少数稳定动作模式（概率集中）

最终表现为 **熵显著下降/塌陷**。  
从数值上看，本 run 的 `diag/entropy_onpolicy_token_mean` 末期大约降到 `~0.09` 量级，且与 late-stage reward 有正相关（粗略相关系数约 0.38）：**熵越低，越容易陷入僵化策略，reward 更难维持高位**。

---

## 5. teacher experience 的正负作用：用“因果分解”来理解

### 5.1 正作用：提供强、稳定、低方差的学习信号（加速）

teacher rollout 贡献的“有效学习信号”来自两点：

1) 它们本身 $A_T$ 往往为正且幅度大  
2) 它们降低了早期探索的随机性，让策略更快进入“能完成任务”的区域

因此会看到 reward 前中期快速抬升，并在一段区间超过 GRPO。

### 5.2 负作用：把 GRPO 的组内比较结构“推向保守收缩”（抑制探索）

teacher 的负作用主要不是“teacher 本身坏”，而是 **teacher 太强 + GRPO 的组内相对基线** 共同作用：

- teacher 强 → baseline 上移 $\Delta>0$
- baseline 上移 → on-policy 期望优势变负 $\mathbb{E}[A_O]<0$
- on-policy 优势为负 → 探索轨迹被惩罚 → 熵下降 → 发现更优策略的概率下降

这会导致后期被 GRPO 反超，或者在更极端情况下出现 reward 崩塌。

### 5.3 “teacher-only + no logprob” 让这种效应更显著

在 `use_log_prob=false` 的设定下，teacher 的 off-policy 修正更弱（本质更像 shaping/imitative 的注入），其更新信号更容易成为“强指向性收缩”，因此：

- 早期提速明显
- 中后期探索受限更明显（更容易看到 entropy_loss 下降得更厉害）

---

## 6. 下一步：如何用你现有指标把结论“做实”

你已经具备验证上述机制的关键观测量，建议把分析动作固定成三条对齐曲线（全部用 $w=10$ 滑动平均）：

1) **reward 对比**：`critic/rewards_onpolicy/mean` vs `critic/rewards/mean`  
2) **baseline 抬高强度**：`diag/group_teacher_minus_on_reward_mean`  
3) **探索被压制的直接表征**：`diag/entropy_onpolicy_token_mean` + `diag/adv_onpolicy_token_mean`

若你看到：

- baseline gap 上升或长期高位  
且
- on-policy adv 长期为负  
且
- entropy 下滑明显  
随后
- reward 停滞/回落  

那么上述因果链条就被非常强地支持。

---

## 7. 附：可复现实验产物位置

- notebook：`analysis/ni1j0wsa_analysis.ipynb`
- wandb 导出 CSV：`analysis/ni1j0wsa/luffy_metrics.csv`、`analysis/ni1j0wsa/grpo_metrics.csv`
- 训练保存的诊断与 rollouts：`checkpoints/agentevolver/.../Trajectory/`


