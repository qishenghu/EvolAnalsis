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
- **moving average**：窗口 `w=10`（用于减少单点波动）
- **本地 trajectory 目录**：  
  - `checkpoints/agentevolver/alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_confidence_analysis/Trajectory/`
  - 包含：每步的 `batch_diag_step_*.json`、汇总 `batch_diag_compiled.csv`、以及 `trajectories_step_*.jsonl`
- **本次 run 新增诊断指标**（由代码注入）：
  - batch-level：`diag/*`（group gap、按类型的 adv/entropy、teacher token 比例等）
  - loss-side：`teacher_diag/*`（ratio/adv 分布；本 run 因为 state=crashed，history 列不完全稳定，但本地 `batch_diag_compiled.csv` 已足够解释主现象）

---

## 1. 关键观测（用指标“钉住”现象）

### 1.1 reward “先超后落”的时间区间（证据）

对 reward 曲线做 `w=10` 滑动平均后：

- **首次超过 GRPO**：约在 `step ≈ 22`
- **最后一次仍高于 GRPO**：约在 `step ≈ 59`
- **LUFFY(on-policy) 峰值**：约 `0.6125 @ step 37`
- **GRPO 峰值**：约 `0.7344 @ step 91`

这说明：teacher experience 的确带来 **前中期加速**，但并没有把最终最优提升到 GRPO 的水平（反而 GRPO 在后期继续爬升）。

### 1.1.1 为什么 `step < 22` 会先明显掉下来（用数据解释）

你观察到的“前 22 步明显掉下来、弱于 GRPO”，本质上来自两个因素叠加：

1) **moving average 在最开始非常敏感**：窗口 `w=10` 时，`step=1~9` 仍处在“样本数不足 10”的阶段，任何 1~2 个极低 reward 点都会把曲线拉下去。  
2) **早期 on-policy 还很弱，而 teacher 近乎满分（reward=1）**：这会造成 *组内 gap 很大*，baseline 被 teacher 强烈抬高，使得 on-policy 在组内更容易处于“相对劣势”（advantage 更偏负），从而早期提升会慢一拍、且波动更大。

下面给出两条直接证据（均来自本地导出的 CSV：`analysis/ni1j0wsa/luffy_metrics.csv` 与 `analysis/ni1j0wsa/grpo_metrics.csv`）：

- **证据 A：早期出现多个极低 on-policy reward 点，会把 `w=10` 曲线直接拉穿**
  - 早期最差的几个点（按 `w=10` 的 `delta_ma = luffy_ma - grpo_ma` 排序）：
    - `step=2`：`luffy_ma=0.2500` vs `grpo_ma=0.4453` → `delta_ma=-0.1953`，当步 `on-policy reward=0.1250`
    - `step=3`：`luffy_ma=0.2202` vs `grpo_ma=0.4219` → `delta_ma=-0.2016`，当步 `on-policy reward=0.1607`
    - `step=4`：`luffy_ma=0.2277` vs `grpo_ma=0.4141` → `delta_ma=-0.1864`，当步 `on-policy reward=0.2500`
  - 这些点在 `w<10` 的阶段权重非常大，所以你会看到“先掉一截”。
  - 到 `step=21`，`w=10` 的差值已经几乎回到 0：`delta_ma_end=-0.00244`（也就是前期下滑更多是“瞬态 + 方差”）。

- **证据 B：早期 teacher 相对更强（group gap 更大），baseline 上移更剧烈**
  - `step=1~21`（early 段）：
    - `on-policy reward` 均值：`0.3658`，而 GRPO 的 `reward` 均值：`0.4256`
    - `diag/group_teacher_minus_on_reward_mean` 均值：`0.6300`（teacher 远强于 on-policy）
    - `diag/reward_teacher_mean`：**恒为 1.0**（teacher rollout 基本满分）
  - `step=22~59`（mid 段，开始“先超”）：
    - `on-policy reward` 均值上升到：`0.5504`
    - `diag/group_teacher_minus_on_reward_mean` 均值下降到：`0.4364`（on-policy 变强，teacher 相对优势变小）

把两点合起来看就很清楚了：

- `step<22` 时，**on-policy 本身还没爬起来**，但 teacher 基本满分 → 组内 gap 大、baseline 上移强，再叠加 `w<10` 的 moving-average 效应 → 曲线看起来会“先掉下来”。  
- 到 `step≈22` 后，on-policy reward 明显提升、group gap 收敛（teacher 相对优势变小）→ moving-average 被“高 reward 区间”填满，于是进入你看到的“先超”阶段。

> 备注：这里的解释是“机制 + 数据证据链”的结合。严格意义上，early 段的绝对高低还会受任务采样/seed 的方差影响，但 **CSV 里 early 段确实存在多个极低 on-policy reward 点**，且 **teacher=1 与 large gap** 的结构性事实也成立，这两者足以解释“先掉再起”的宏观形态。

### 1.2 teacher 在 group 内长期“抬高 baseline”（证据）

从本地 `batch_diag_compiled.csv`（等价于 `diag/group_teacher_minus_on_reward_mean`）：

- `E[teacher_reward - on_reward | within_group]` **长期为正**
- 数值大致在 **[0.38, 0.78]** 之间波动（越大代表 teacher 相对更强）

直觉含义：**teacher rollout 在每个 task group 内经常是“最强样本”**，从而持续把 group baseline 抬高。

### 1.3 on-policy advantage 被系统性压低（证据）

从 `diag/adv_onpolicy_token_mean` 与 `diag/adv_teacher_token_mean`：

- `adv_onpolicy_token_mean` **全程为负**（约 `-0.07` 到 `-0.033`）
- `adv_teacher_token_mean` **全程为正**（约 `0.8` 到 `1.8`）

这是解释后续 “熵塌陷 + 后期 reward 不如 GRPO” 的核心证据链：  
**组内相对优势被 teacher “占据”，on-policy 在组内成为被惩罚的一方。**

### 1.4 teacher token 比例很小但“影响很大”（证据）

`diag/teacher_token_ratio`（token-level）：

- 均值约 `≈ 0.9%`
- 范围约 `[0.7%, 1.1%]`

虽然 teacher token 只占很小比例，但它们拥有 **大且持续为正的 advantage**，并且还会改变 **GRPO 的组内 baseline**（影响所有 rollouts 的 advantage），因此整体影响可以很显著。

---

## 2. 用数学形式化：为什么 teacher 会“抬 baseline → 压 on-policy advantage”

下面用最小化假设把机制讲清楚。

### 2.1 GRPO 的组内优势定义

对同一 task 的一组 rollouts（大小 \(n\)），令每条 rollout 的回报为 \(R_i\)，组内平均为：

\[
\bar{R} = \frac{1}{n}\sum_{j=1}^{n} R_j
\]

典型 GRPO 的相对优势可以抽象为：

\[
A_i = R_i - \bar{R}
\]

（实际实现可能还有标准化/截断/GAE 等，但 **组内去均值** 是关键结构。）

### 2.2 引入 teacher 混入后的 baseline 上移

设每组里混入 \(k\) 条 teacher rollouts，剩余 \(n-k\) 条为 on-policy。令：

- teacher 平均回报：\(\mu_T\)
- on-policy 平均回报：\(\mu_O\)

则组均值（baseline）为：

\[
\bar{R}
= \frac{k\mu_T + (n-k)\mu_O}{n}
= \mu_O + \frac{k}{n}(\mu_T-\mu_O)
\]

当 teacher 更强（\(\mu_T>\mu_O\)）时，baseline **相对 on-policy 的上移量**为：

\[
\Delta \triangleq \bar{R} - \mu_O = \frac{k}{n}(\mu_T-\mu_O) > 0
\]

这直接导致 on-policy 的组内优势期望：

\[
\mathbb{E}[A_O]
= \mu_O - \bar{R}
= -\Delta
= -\frac{k}{n}(\mu_T-\mu_O) < 0
\]

**结论**：只要 teacher 平均回报高于 on-policy，且每组混入 \(k>0\)，那么 on-policy 在组内的期望优势就是负的（被系统性压低）。

> 这与本次 run 的观测完全一致：`diag/group_teacher_minus_on_reward_mean` 长期为正，且 `diag/adv_onpolicy_token_mean` 长期为负。

### 2.3 “为什么 teacher token 比例很小也会造成大影响？”

注意：teacher 的影响有两层：

1) **直接梯度贡献**：teacher token 自身参与 policy gradient（teacher advantage 往往大且为正）  
2) **间接 baseline 影响**：teacher 改变了 \(\bar{R}\)，从而改变了每条 on-policy rollout 的 \(A_i\)

第二层是 "以小搏大" 的关键：即使 teacher token 比例小，只要它们对应的 rollout 回报显著更高，就会在组内把 \(\bar{R}\) 拉上去，影响所有 on-policy 的优势符号与幅度。

---

## 3. 用数学解释 reward “先超后落”

### 3.1 早期/中期：teacher 提供稳定的正向“方向导引”

以 PPO/GRPO 的 policy gradient 形式（省略 clip/ratio 细节）为例：

\[
\nabla_\theta \mathcal{L}_{PG}
\propto
\mathbb{E}\left[\sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]
\]

teacher rollout 往往有：

- `A_T` 大且正（本 run `diag/adv_teacher_token_mean` 长期为正且显著大于 0）
- 因此 teacher 对更新方向的贡献稳定且强

从优化角度看，它相当于把策略快速推向 teacher 轨迹附近的参数区域，使得 **样本效率提升**，因此 reward 在 `step≈22~59` 超过 GRPO 是合理的。

### 3.2 后期：探索被压制，难以超过“teacher 所覆盖的策略子空间”

上一节已经推出：

\[
\mathbb{E}[A_O] = -\frac{k}{n}(\mu_T-\mu_O)
\]

这意味着 **探索型的 on-policy 行为**（通常短期回报更不稳定）在组内更容易被判为负 advantage，从而在更新中被压制。

在 RL 中，后期进一步提升往往依赖：

- 发现 teacher 轨迹之外的更优策略（或更稳健策略）
- 或在更复杂状态分布下学会处理“长尾失败模式”

而 baseline 被长期抬高会让策略更倾向于收缩在少数高概率模式里（见下一节的熵分析），从而：

- 前中期：快速逼近 teacher 能力范围（reward 提升快）
- 后期：探索不足导致边际收益下降（reward 变慢甚至回落），被 GRPO（保持更强探索）反超

与观测一致：GRPO 峰值出现在更靠后的 `step≈91`，而 teacher run 峰值更早（`step≈37`）。

### 3.3 你困惑的核心：为什么“有 teacher”也不会单调学到 teacher（数学角度）

你预期的路径更像 **行为克隆（BC）**：如果一直用 teacher 的 (s, a) 做监督学习，那么确实会单调“逼近 teacher”。  
但我们现在优化的是 **GRPO/PPO 的相对优势目标**，它和 BC 有三个本质差别，导致 reward 可以“先掉、后超、再落”。

#### 3.3.1 目标函数不是“最大化 teacher reward”，而是“最大化组内相对优势”

对每个 task 的一组 rollouts（大小 n，其中 k 条 teacher，n-k 条 on-policy），简化写成：

- 组内 baseline：`R_bar = mean(R_1..R_n)`
- 组内优势：`A_i = R_i - R_bar`

如果 teacher 更强：`mu_T > mu_O`，那你已经看到：

- `E[A_on] = - (k/n) * (mu_T - mu_O) < 0`
- `E[A_teacher] = + ((n-k)/n) * (mu_T - mu_O) > 0`

重要点在于：**当 teacher 固定且强时，on-policy 的“期望优势”长期为负**。  
这意味着训练在优化时会出现一个“结构性偏置”：大多数 on-policy token 会被当作“应该降低概率”的对象。

> 这不是在最大化绝对 reward，而是在最大化 “相对 baseline 的优势”。当 baseline 被 teacher 长期抬高时，on-policy 的相对优势就会长期偏负。

#### 3.3.2 “负优势”并不等价于“朝 teacher 靠拢”，它首先是“远离当前 on-policy 自己”

把 token-level 的 policy gradient 写成（忽略 clip 等细节）：

`grad ~ E[ w_t * A_t * ∇ log pi_theta(a_t | s_t) ]`

当大多数 on-policy token 的 `A_t < 0` 时，更新方向是 **降低这些 on-policy 动作的概率**。  
但“降低自己做过的动作概率”并不保证“提高 teacher 动作概率”，因为 teacher 动作通常是一个很小子集，且在 early 阶段 `pi_theta(a_teacher|s)` 可能极低。

所以 early 很容易出现这种现象：

- teacher 的奖励很高（甚至接近 1）
- 但 student 还没把 teacher 动作学到足够概率
- 同时 on-policy 大范围负优势把 student 现有行为整体“推开”
- 结果是：**策略发生剧烈重分配（甚至塌到某些高概率模板），reward 先掉一段**

这能解释你看到的 “step<22 先掉下来”：early 的 on-policy reward 本来就低（均值 0.3658），再叠加 baseline 抬高带来的负优势压力，波动会被放大。

#### 3.3.3 use_log_prob=false 时，teacher 梯度对“低概率 teacher token”天然很弱（学得慢）

你这个 run 是 `use_log_prob=false`（LUFFY 形式）。在我们实现里，teacher 的 off-policy 权重本质上是 “用当前策略的概率做权重” 的某种变体（再可能叠加 policy shaping）。

直观结论：

- 当 `pi_theta(a_teacher|s)` 很小（early 阶段常见），teacher token 的权重也很小  
  -> **teacher 的正 advantage 再大，也很难产生足够梯度把它学起来**
- 等到中期 `pi_theta(a_teacher|s)` 被逐渐抬起来，teacher 梯度才会开始变“有效”  
  -> 这对应你看到的 **22 之后开始超过 GRPO**

换句话说：**teacher 在 early 并不会像 BC 那样“强制你模仿”**；它更像一个“带权的 RL 信号”，权重还会在 early 被低概率显著削弱。

#### 3.3.4 为什么后期会“落下来”，而不是一直上升？

当 on-policy 水平上升后，会出现两个自然后果：

- (a) `mu_O` 上升使 `(mu_T - mu_O)` 变小，teacher 相对优势变小  
  -> teacher 的“拉动作用”边际变弱（你也观测到 group gap 从 early 的 0.63 降到 mid 的 0.44）
- (b) baseline 仍然被 teacher 抬高，on-policy 优势仍偏负（虽然绝对值会变小）  
  -> 这会长期压制探索，导致策略更容易收缩到少数模式（熵下降/塌陷）

而很多 RL 任务的后期提升（尤其要超过一个强 teacher 的覆盖范围）往往依赖：

- 在 teacher 没覆盖到的状态/失败模式中继续探索并修正
- 学到更鲁棒的策略而非“高概率模板”

当探索被压制时，reward 就会出现“涨不动甚至回落”，同时 GRPO（更纯粹的 on-policy、baseline 不被 teacher 污染）仍可能继续爬升，最终反超。

> 一句话总结：**teacher replay 在这里不是“监督模仿”，而是“相对优势驱动的加权 RL 信号”**。强 teacher 会同时带来“中期加速”与“长期 baseline 抬高/探索压制”，从而自然产生 “先掉—后超—再落” 的非单调曲线。

#### 3.3.5 更形式化：teacher 梯度强度为什么会随 `pi_theta(a_T|s)` 单调变强（解释“中期才开始贴近 teacher”）

把 teacher token 的更新抽象成一个“带权优势梯度”（忽略 clip/normalize，仅用于定性）：

- teacher token 的梯度项：`g_T = E_{(s,a_T)~D_T}[ w_T(s,a_T) * A_T(s,a_T) * ∇ log pi_theta(a_T|s) ]`

这里 `D_T` 是 teacher 轨迹诱导的状态-动作分布；`A_T` 是 teacher token 在当前 batch 里的优势（对这类 run 里通常为正且较大）；`w_T` 是 off-policy 权重。

在两个常见设定下，`w_T` 的量级行为不同：

- **use_log_prob=true（标准重要性比率）**
  - 近似：`w_T ≈ exp( log pi_theta(a_T|s) - log pi_old(a_T|s) )`
  - 对 teacher replay 来说 `pi_old` 常是 teacher：`pi_old = pi_teacher`
  - 则：`w_T ≈ pi_theta(a_T|s) / pi_teacher(a_T|s)`
  - 推论：如果 teacher 对该动作很自信（`pi_teacher` 大），但 student 还没学会（`pi_theta` 小），那么 `w_T` 会非常小。

- **use_log_prob=false（LUFFY 简化）**
  - 近似：`w_T ≈ f( pi_theta(a_T|s) )`
  - 不做 shaping 时可以近似成：`f(x)=x`
  - 做 policy shaping（例如 `f(x)=x/(x+beta)`）时：
    - 当 `x << beta`：`f(x) ≈ x/beta`（仍然与 `x` 成正比，依旧很小）
    - 当 `x >> beta`：`f(x) ≈ 1`（权重饱和，teacher 梯度“完全显现”）

所以无论你是哪一种设定，都有一个共同结论：

- 当 `pi_theta(a_T|s)` 很小（early 常见）时，teacher 梯度的有效系数 `w_T` 很小  
  -> 即使 `A_T` 很大，也“推不动”策略去模仿 teacher（学得慢）
- 当 `pi_theta(a_T|s)` 被逐步抬起来（中期开始）时，`w_T` 随之变大并可能饱和  
  -> teacher 梯度突然变得非常有效（你看到的“22 之后开始超过 GRPO”就符合这一点）

这就是你问的“为什么不会一直按 teacher experience 学”的一个严格答案：  
**teacher 的更新强度不是常数，它是随 student 当前对 teacher 动作的概率增长而自增强的。**

#### 3.3.6 更形式化：给一个“熵塌陷”的（近似）充分条件

我们只看一个状态 `s` 下的离散动作分布（把多 token 展开成多个动作也一样），令：

- `p_i = pi_theta(a_i|s)`
- `H(p) = -sum_i p_i log p_i`

忽略 clip 和 baseline 细节时，PPO/GRPO 的更新会倾向于：

- 对“高优势动作”提高概率
- 对“负优势动作”降低概率

如果在一个较长时间段内，存在一个动作子集 `G`（可以理解成 teacher 常走的一组动作/模板），满足：

- 对所有 `a in G`：`E[A(s,a)] >= +c_pos`
- 对所有 `a notin G`：`E[A(s,a)] <= -c_neg`
- 且 entropy 正则系数 `alpha` 相对较小（不足以抵消上述优势差）

则迭代更新会把质量从 `a notin G` 不断转移到 `G`，导致：

- `sum_{a in G} p(a)` 单调增加
- 概率质量集中到少数动作上
- 熵 `H(p)` 下降（“塌陷”）

把它更直观地写成一句话：

- 如果“同一批训练信号”长期给出 **稳定的正优势方向**（teacher）与 **广泛的负优势惩罚**（on-policy 探索），且 `alpha` 不够大，那么最稳定的解就是“收缩到少数高概率模式”，熵必然下降。

这与你的观测一致：

- 该 run 里 `diag/adv_teacher_token_mean` 长期为正且不小
- `diag/adv_onpolicy_token_mean` 长期为负
- late 段 `diag/entropy_onpolicy_token_mean` 明显更低（~0.09）

因此 reward 的后期回落可以理解为：**当策略在少数模板上塌陷时，它在更复杂状态/失败模式上的修复能力下降，整体 on-policy reward 反而难以维持。**

---

## 4. 用数学解释熵 “先降 → 回升 → 再塌陷”

### 4.1 熵项的梯度方向（为什么会“收缩”）

常见 entropy 正则（最大化熵）写为：

- `L = L_PG - alpha * E[ H(pi_theta(.|s)) ]`

其中 `H(pi)` 越大越"探索"。当 policy gradient 部分对某些动作给出持续正 advantage 时，`pi_theta` 会对这些动作 **提高概率**，从而自然降低熵（更确定）。

在本 run 中，teacher 的优势长期强正，而 on-policy 优势长期偏负，这相当于在组内持续施加“朝 teacher 动作集中”的梯度压力 → **早期熵快速下降**是自然结果。

### 4.2 中期回升：baseline 压力减轻 + 训练噪声/正则的再扩散

当模型逐步学到一部分成功策略后：

- on-policy 的回报 `mu_O` 上升
- teacher 与 on-policy 的差距 `(mu_T - mu_O)` 可能阶段性缩小
- 则 baseline 上移量 `Delta = (k/n)*(mu_T - mu_O)` 变小

在这种情况下，on-policy 的负 advantage 压力减轻，entropy 正则和训练噪声可能让策略出现一定“再扩散” → **中期熵回升**。

这与你的本地 rollouts 汇总一致：例如在 `step 40` 周期，reward 较高且 `entropy_mean` 也相对更高（相比 `step 80` 的塌陷段）。

### 4.3 中后期再度下滑/塌陷：长期负优势使探索变成“被惩罚对象”

当 teacher 仍然长期强于 on-policy（观测：`diag/group_teacher_minus_on_reward_mean` 始终为正且不小），则：

- `Delta` 不会消失
- `E[A_on]` 长期为负

这会导致：

1) on-policy 中“探索导致的短期失败”被更强烈惩罚（优势更负）
2) 策略逐渐只保留少数稳定动作模式（概率集中）

最终表现为 **熵显著下降/塌陷**。  
从数值上看，本 run 的 `diag/entropy_onpolicy_token_mean` 末期大约降到 `~0.09` 量级，且与 late-stage reward 有正相关（粗略相关系数约 0.38）：**熵越低，越容易陷入僵化策略，reward 更难维持高位**。

---

## 5. teacher experience 的正负作用：用“因果分解”来理解

### 5.1 正作用：提供强、稳定、低方差的学习信号（加速）

teacher rollout 贡献的“有效学习信号”来自两点：

1) 它们本身 `A_T` 往往为正且幅度大  
2) 它们降低了早期探索的随机性，让策略更快进入“能完成任务”的区域

因此会看到 reward 前中期快速抬升，并在一段区间超过 GRPO。

### 5.2 负作用：把 GRPO 的组内比较结构“推向保守收缩”（抑制探索）

teacher 的负作用主要不是“teacher 本身坏”，而是 **teacher 太强 + GRPO 的组内相对基线** 共同作用：

- teacher 强 → baseline 上移 `Delta > 0`
- baseline 上移 → on-policy 期望优势变负 `E[A_on] < 0`
- on-policy 优势为负 → 探索轨迹被惩罚 → 熵下降 → 发现更优策略的概率下降

这会导致后期被 GRPO 反超，或者在更极端情况下出现 reward 崩塌。

### 5.3 “teacher-only + no logprob” 让这种效应更显著

在 `use_log_prob=false` 的设定下，teacher 的 off-policy 修正更弱（本质更像 shaping/imitative 的注入），其更新信号更容易成为“强指向性收缩”，因此：

- 早期提速明显
- 中后期探索受限更明显（更容易看到 entropy_loss 下降得更厉害）

---

## 6. 下一步：如何用你现有指标把结论“做实”

你已经具备验证上述机制的关键观测量，建议把分析动作固定成三条对齐曲线（全部用 `w=10` 滑动平均）：

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


