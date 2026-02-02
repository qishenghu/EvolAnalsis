# DR³（Density-Ratio-Repair）在 AgentEvolver 中的完整整理（动机、与 LUFFY 区别、数学推导、实现细节）

> 面向“reviewer 解释稿”：尽量用 **清晰的目标函数 + 采样分布 + 密度比** 来讲清楚“为什么这是分布矫正”，并说明工程上为什么采用 **relative density ratio \(p/m_\alpha\)** 而不是直接用 \(p/q\)。
>
> 代码对应（核心路径）：
> - `agentevolver/module/exp_manager/dr3_ratio.py`：判别器训练、\(D\to w_\alpha\) 映射、ESS/dual clip、alpha 自动估计等
> - `agentevolver/module/exp_manager/het_actor.py`：DR³ 的 observe/apply、teacher fade-out（reward-gap gate）、把 \(w_\alpha\) 注入到 PPO ratio
> - `agentevolver/module/exp_manager/het_core_algos.py`：LUFFY / teacher-aware loss（teacher 无 logprob 时 ratio=exp(log_prob)+policy shaping）

---

## 1. 问题背景与动机（Motivation）

在我们的训练场景里，每个训练 step 的 batch 同时包含：
- **on-policy**（由当前策略 \(\pi\) 采样得到）；
- **teacher/off-policy**（从 teacher 轨迹库中采样得到，通常没有 teacher 的 logprob）。

这会带来一个核心矛盾：
- 我们希望 teacher 数据能 **加速早期学习**；
- 但 teacher 与当前策略分布往往差别很大（尤其早期），如果直接把 teacher 梯度“硬塞进来”，会导致：
  - **训练目标偏移**（优化的不是你想要的 on-policy 目标）；
  - **late-stage asymptotic 性能受损**（模型长期被 teacher 分布牵引，出现 “teacher lock-in”）。

因此我们需要一个机制：
1) 允许使用 teacher 样本，但要把其对更新的影响 **做分布矫正（distribution correction / repair）**；
2) 在策略逐渐追上 teacher 时，teacher 影响应当 **闭环淡出（closed-loop fade-out）**，保证 asymptotic。

DR³ 的核心思想就是：用判别器学习 **on/off 的密度比结构**，并用相对密度比 \(w_\alpha\) 修复 off-policy/teacher 的梯度贡献，同时配合一套稳定化与淡出机制。

---

## 2. LUFFY 在做什么？它的优点与关键缺陷

在 `teacher_use_log_prob=False` 的 LUFFY 模式里，我们缺少 teacher 的 \(\log \pi_{\text{teacher}}\) 或 \(\log \pi_{\text{old}}\)，因此无法构造标准重要性比：
\[
\rho = \frac{\pi_{\text{new}}}{\pi_{\text{old}}} = \exp(\log\pi_{\text{new}} - \log\pi_{\text{old}}).
\]

LUFFY 的做法是对 teacher 轨迹采取“分母=1”的近似：
\[
\rho_{\text{teacher}} \approx \pi_{\text{new}}(a|s) = \exp(\log\pi_{\text{new}}).
\]
并且为了避免 teacher 过强，通常还会叠加 **policy shaping**（例如 \(p/(p+\beta)\) 一类的压缩函数）来削弱 teacher 的影响。

### LUFFY 的优点
- 不需要 teacher 的 logprob；
- 简单、可用、早期能加速。

### LUFFY 的关键缺陷
- **不是分布矫正**：它并没有把 teacher 数据从 \(q\)（teacher 分布）“纠正成” \(p\)（当前策略分布）的任何密度比结构。
- **late-stage 容易锁死**：若 teacher 信号持续存在，模型可能长期向 teacher 的行为靠拢而不是最优策略，导致 asymptotic reward 不如更“自举”的训练。
- **shaping 是启发式的**：shaping 能抑制过强 teacher，但缺少“为什么这等价于矫正”的清晰统计语义。

---

## 3. DR³ 的核心目标：从“混合采样”矫正回 on-policy

### 3.1 采样分布是什么？
我们的训练数据不是纯 teacher，也不是纯 on-policy，而是一个 **混合分布**：
\[
m_\alpha(x) = (1-\alpha)\,p(x) + \alpha\,q(x)
\]
其中：
- \(p(x)\) 表示当前策略产生的样本分布（on-policy）；
- \(q(x)\) 表示 teacher/off-policy 的样本分布；
- \(\alpha\in(0,1)\) 表示 off-policy 在混合中的比例（工程上可能随 step、task 缺 teacher、分布式 gather 等动态变化）。

### 3.2 我们到底想矫正什么？
我们想让“在混合分布 \(m_\alpha\) 上计算的期望”变成“在 \(p\) 上的期望”。这对应重要性采样的恒等式：
\[
\mathbb{E}_{x\sim p}[f(x)]
\,=\,
\mathbb{E}_{x\sim m_\alpha}\!\left[\frac{p(x)}{m_\alpha(x)}\,f(x)\right].
\]
因此 **正确的矫正权重**是：
\[
w_\alpha(x)\;=\;\frac{p(x)}{m_\alpha(x)}\;=\;\frac{p(x)}{(1-\alpha)p(x)+\alpha q(x)}.
\]

> 这就是 DR³ 的“密度比修复（repair）”语义：  
> **用 \(w_\alpha=p/m_\alpha\) 把混合采样上的更新，矫正回 on-policy 目标。**

---

## 4. 判别器的语义：从 \(D(x)\) 得到 \(p/q\) 的结构

DR³ 训练一个二分类判别器 \(D(x)\)：
- 标签 \(y=1\)：样本来自 on-policy（分布 \(p\)）
- 标签 \(y=0\)：样本来自 teacher/off-policy（分布 \(q\)）

当判别器训练充分时，输出近似后验概率：
\[
D(x)\approx P(y=1\mid x).
\]

### 4.1 为什么 class-balanced sampling 很重要？
我们在训练判别器时采用（或近似）**balanced**（50/50）采样，让训练先验 \(P(y=1)=P(y=0)=0.5\)。这样才有最干净的 odds 恒等式：
\[
\frac{D(x)}{1-D(x)} \approx \frac{p(x)}{q(x)}.
\]

如果不 balanced，odds 会混入先验项：
\[
\frac{D}{1-D}\approx \frac{1-\pi}{\pi}\cdot\frac{p}{q}
\quad (\pi=P(y=1)).
\]
这会让 “\(D\to p/q\)” 依赖训练 prior，工程上容易引入不可控偏差。

因此我们在实现里把判别器 batch 采样做成 class-balanced，并且用 **prior-robust 映射**（odds）来削弱 prior 影响。

---

## 5. 从 \(D(x)\) 到 \(w_\alpha(x)=p/m_\alpha\) 的推导（prior-robust mapping）

### 5.1 odds 给出 \(\hat r \approx p/q\)
\[
\hat r(x) \;=\; \frac{D(x)}{1-D(x)} \;\approx\; \frac{p(x)}{q(x)}.
\]

### 5.2 用代数把 \(p/m_\alpha\) 写成 \(r\) 的函数
由定义：
\[
w_\alpha(x)=\frac{p}{(1-\alpha)p+\alpha q}.
\]
两边同除 \(q\)，令 \(r=p/q\)，得到：
\[
w_\alpha(x)=\frac{r}{(1-\alpha)r+\alpha}.
\]

所以我们最终映射为：
\[
w_\alpha(x)=\frac{\hat r(x)}{(1-\alpha)\hat r(x)+\alpha}.
\]

### 5.3 为什么这比直接用 \(p/q\) 更适合 PPO？
一个 reviewer 很喜欢的性质：**\(w_\alpha\) 有天然上界**。

当 \(r\to\infty\)（即 \(q\) 很小、\(p/q\) 爆炸）时，
\[
w_\alpha \to \frac{1}{1-\alpha}.
\]
也就是说：
- **\(p/q\) 是无界的（高方差、极不稳定）**；
- **\(p/m_\alpha\) 是有界的（更低方差、数值更稳）**，特别适合 PPO/GRPO 这类带 clipping 的优化框架。

这就是我们选择 relative density ratio 的核心“可说服 reviewer 的理由”：  
**它既与真实采样分布 \(m_\alpha\) 对齐，又显著降低方差并提供有界性。**

---

## 6. DR³ 最终是怎么作用到 teacher 的 “importance ratio” 的？

### 6.1 PPO 的 ratio 形式
标准 PPO/GRPO 里：
\[
\rho = \exp(\log\pi_{\text{cur}}-\log\pi_{\text{old}}).
\]

### 6.2 DR³ 的 trick：构造“等效 old logprob”，让 \(\rho_{\text{teacher}}\approx w_\alpha\)
我们在 `teacher_no_logprob` 路径里，不直接做 `loss *= w`，而是对 teacher/off-policy 样本构造：
\[
\log\pi_{\text{old,eff}} \leftarrow \log\pi_{\text{cur}}(\text{detach}) - \log w_\alpha.
\]

于是 teacher 的 ratio：
\[
\rho_{\text{teacher}}
=\exp\!\big(\log\pi_{\text{cur}}-\log\pi_{\text{old,eff}}\big)
\approx \exp(\log w_\alpha)=w_\alpha.
\]

### 6.3 与 LUFFY 的关键区别（一句话）
- **LUFFY**：teacher ratio 近似为 \(\rho_{\text{teacher}}\approx \exp(\log\pi_{\text{cur}})\)（分母视为 1） + shaping（启发式压缩）。
- **DR³**：teacher ratio 被“修复”为 \(\rho_{\text{teacher}}\approx w_\alpha=p/m_\alpha\)（有清晰分布矫正语义与天然有界性），并且这条路径不依赖 LUFFY 的 policy shaping。

---

## 7. “为什么不是用 \(p/q\)？我们明明知道 teacher 来自 \(q\)”

### 7.1 关键点：矫正目标取决于你的采样分布
- 若样本全部来自 \(q\)，你确实会写：
  \[
  \mathbb{E}_{p}[f]=\mathbb{E}_{q}\!\left[\frac{p}{q}f\right].
  \]
- 但我们的训练样本来自 **混合采样** \(m_\alpha\)，因此对应恒等式是：
  \[
  \mathbb{E}_{p}[f]=\mathbb{E}_{m_\alpha}\!\left[\frac{p}{m_\alpha}f\right].
  \]
  这时 \(p/m_\alpha\) 才是与数据生成过程完全对齐的密度比。

### 7.2 稳定性：\(p/m_\alpha\) 低方差且有上界，是 PPO 的现实选择
即使你“知道 teacher 来自 \(q\)”，在 PPO 里直接用 \(p/q\) 往往会带来权重爆炸，导致：
- 大量样本被 clipping 成常数（有效梯度信息丢失）；
- 训练高度不稳定；
- 经验上 asymptotic 更差。

因此我们用 \(p/q\) 的“形状”（由判别器 odds 学到），但最终采用 \(p/m_\alpha\) 作为矫正权重，是一个 **统计效率优先** 的选择，并且有非常清晰的理论叙事（relative density ratio）。

---

## 8. DR³ 的工程实现要点（为什么不只是推导）

### 8.1 自动估计 \(\alpha\)（避免固定 prior bias）
工程上 teacher rollout 可能缺失（某些 task 没 teacher），导致固定 \(\alpha\) 会偏。
因此我们支持根据实际 batch / buffer 的 on/off 比例估计 \(\alpha\)，并用 EMA 平滑：
- `alpha_mode: sync_batch_ema`（跨 rank 同步统计更稳）
- `alpha_ema_beta`

### 8.2 判别器过拟合与过自信：为什么需要稳定化
判别器太快 acc=1 并不一定好：它会让 logits 很大、\(D\) 贴近 0/1，进而使 odds/ratio 极端。
我们加入了多种“让 \(D\) 更温和、更泛化”的机制：
- **class-balanced 采样**（削弱训练 prior 偏差）
- **label smoothing**（避免 logit 无界变大）
- **temperature scaling**（推理时软化 logits）
- **age-weighted loss**（让判别器更关注近期分布，缓解漂移）
- **weight decay / dropout**（限制容量，抑制过拟合）
- **disc_train_min_buf_size / apply_min_buf_size / apply_warmup_steps**（先积累再训练/再应用，避免早期噪声）

### 8.3 dual clipping + ESS：约束极端权重并可观测
我们对 \(w_\alpha\) 做 clipping：
- 理论上 \(w_\alpha\le 1/(1-\alpha)\)；
- 实践中仍可能出现异常（判别器误差、特征漂移），因此用 ESS 计算权重退化程度，并用 dual 机制自适应调整 clip 上界。

关键可观测指标（wandb）：
- `dr3/w_off_mean`, `dr3/w_off_p99`, `dr3/w_clipfrac_off`
- `dr3/ess_off_window`
- `dr3/dual_lambda`

---

## 9. Closed-loop teacher fade-out：为什么要在 late-stage 显式淡出

即使 DR³ 的密度比估计更“正确”，也可能出现：teacher 仍然在 late-stage 持续影响策略，导致探索受限、极值点受限。
因此我们增加了基于 reward gap 的闭环门控（gate）：
- trainer 计算 teacher 与 on-policy reward 的 gap，输出 `teacher_loss_scale`（0~1）
- actor 侧对 teacher 样本的优势 \(A\) 做 sample-level 缩放：
  \[
  A_{\text{teacher}} \leftarrow A_{\text{teacher}}\cdot g
  \]
并支持 `gap_gate_power` 做非线性压缩：
  \[
  g \leftarrow g^{\text{power}}
  \]

这提供了一个 reviewer 能接受的解释：
- **DR³**：解决“混合采样下的分布矫正与稳定性”
- **Gap gate**：解决“teacher late-stage 影响过强导致 asymptotic 受损”

两者组合实现“早期加速 + 后期自举”的闭环。

---

## 10. 方法总结（可以直接写进 paper / response）

我们提出/实现了一种在 teacher 缺 logprob 的混合训练中使用的分布矫正机制 DR³：
1) 用 class-balanced 判别器估计 on/off 的密度比结构，得到 \(\hat r \approx p/q\)；
2) 将其转换为 relative density ratio：
   \[
   w_\alpha = \frac{\hat r}{(1-\alpha)\hat r + \alpha} = \frac{p}{(1-\alpha)p+\alpha q}
   \]
   它是从真实采样分布 \(m_\alpha\) 到目标分布 \(p\) 的正确密度比，并具有天然上界 \(1/(1-\alpha)\)；
3) 在 PPO/GRPO 中，通过构造等效旧策略 logprob，使 teacher 的重要性比 \(\rho_{\text{teacher}}\approx w_\alpha\)，实现 teacher 梯度的分布修复；
4) 加入稳定化（label smoothing/temperature/age-weight/dual+ESS）与 reward-gap 闭环淡出，改善 late-stage asymptotic。

---

## 11. 实验/分析建议（对 reviewer 的“可验证性”）

强烈建议在报告中同时展示：
- `dr3/disc_acc`, `dr3/disc_loss`（判别器是否过快饱和）
- `dr3/w_off_mean`, `dr3/w_off_p99`, `dr3/w_clipfrac_off`（权重是否极端）
- `dr3/ess_off_window`, `dr3/dual_lambda`（权重退化与自适应 clip 是否工作）
- `dr3/gap_gate_mean`, `dr3/gap_gate_power`（teacher fade-out 是否在 late-stage 生效）
- （建议新增或手动计算）teacher 的“有效注入强度”近似：`w_off_mean * gap_gate_mean`

---

## 12. 附：常见 reviewer 问答（简版）

### Q1：为什么不用 \(p/q\)？那不是最无偏吗？
A：我们的数据来自混合分布 \(m_\alpha\)，因此从 \(m_\alpha\) 矫正到 \(p\) 的正确密度比是 \(p/m_\alpha\)。此外 \(p/m_\alpha\) 有上界 \(1/(1-\alpha)\)，方差更小、更稳定，适配 PPO/GRPO 的 clipping 框架；直接用 \(p/q\) 在早期分布差异大时会导致权重爆炸与退化。

### Q2：class-balanced 训练是否改变了 prior？
A：是的，因此我们用 odds \(D/(1-D)\) 来消除训练 prior（balanced 的 prior 为 0.5 时 odds 直接对应 \(p/q\)），并最终映射到 \(w_\alpha\)。

### Q3：为什么还需要 teacher fade-out？
A：分布矫正解决的是“使用 teacher 数据时的统计偏差/稳定性”，但 teacher 仍可能在 late-stage 产生策略锁定效应。reward-gap gate 提供闭环淡出，保证 asymptotic。

---

## 13. 关键三问（Q1–Q3）：我们如何把“分布矫正的必要性”讲清楚并用 pilot 证出来？

这一节按 reviewer 的因果链来写：**分布矫正解决什么 → fade-out 解决什么 → 最小 pilot ablation 如何区分二者贡献**。

### Q1：分布矫正的意义是什么？如果最终会 fade-out，为什么还要分布矫正？

**核心观点**：fade-out 解决的是“teacher 影响在训练时间维度上何时退出”；分布矫正解决的是“teacher 仍在起作用的阶段（前/中期），它的更新是否在统计意义上对齐 on-policy 目标，并且是否低方差/可控”。二者是正交的。

可以直接写进论文的一句话：
> DR³ 用相对密度比 \(w_\alpha=p/m_\alpha\) 修复混合采样下 off-policy 梯度的统计偏差与方差；reward-gap gate 作为闭环调度在策略追上 teacher 后逐步关掉 teacher 注入，从而保护 late-stage asymptotic。没有 DR³，前期 teacher 仍可能造成目标偏移/不稳定；没有 gate，后期仍可能出现 teacher-lock-in。

### Q2：有什么 pilot experiment 能证明分布矫正的必要性/自然引出分布矫正？

下面给一套“最省算力但信息量最大”的 pilot（优先级从高到低）。建议每个 pilot 都输出：reward 曲线 + teacher 强度曲线 + 稳定性曲线（极端权重/ESS）。

#### Pilot-1（首选）：同样的 fade-out，比较“有无分布矫正”
目标：排除“只是 gate 更强/更弱导致”的解释。

最小 3 组：
- **LUFFY + gap-gate**（teacher 无 logprob：`teacher_ratio=exp(log_prob)` + policy shaping；开启同一套 adaptive_weight）
- **DR³ + gap-gate**（你当前方案：teacher ratio \(\approx w_\alpha\) 注入 PPO ratio；同一套 adaptive_weight）
- **No-teacher（纯 on-policy）**（下界/参照）

推荐画图：
- `reward_mean` vs step（主结果）
- teacher “有效注入强度”的 proxy：`dr3/w_off_mean * dr3/gap_gate_mean`（或并列曲线）
- 稳定性：`dr3/w_off_p99`, `dr3/w_clipfrac_off`, `dr3/ess_off_window`, `dr3/dual_lambda`

论证逻辑：若两者 gate 曲线近似（teacher 强度近似），但 DR³ reward 更高/更稳，则说明分布矫正改变的是 **teacher 信号质量（权重形状/方差/更新对齐）**，而不是单纯强度。

#### Pilot-2：teacher 强度对齐（最强“不是只有 fade-out”证据）
目标：让 reviewer 没法说“你只是把 teacher 弄弱了”。

做法：在 LUFFY 与 DR³ 两个方法上，强制对齐 teacher 的平均有效系数（例如使每 step 的 \(\mathbb{E}[w\cdot g]\) 相同；LUFFY 侧用一个可调 scale/shaping 使其匹配）。

如果 teacher 强度被对齐后 DR³ 仍更好 → 这就是非常强的机制证据。

#### Pilot-3：机制指标（证明 DR³ 的“矫正”在机制变量上确实生效）
目标：展示 DR³ 不只是一个额外超参，而是确实改变了 off-policy 权重结构。

展示：
- DR³ 侧：`w_off_p50/p90/p99`、`clipfrac_off`、`ESS` 的改善（更少极端、更高 ESS）
- 对照 LUFFY：用 teacher_ratio 的分位数、clipping 命中率（在 `het_core_algos.py` 的 teacher_off_pg_cliphit_mask/统计中补充日志也可）

#### Pilot-4：α 漂移实验（自然引出“为什么需要 auto alpha + prior-robust”）
目标：证明固定 α 会在 teacher 缺失/比例变化时产生系统性偏差，分布矫正需要跟现实采样分布对齐。

做法：
- 人为制造 teacher 缺失（某些 task 没 teacher）或改变 `n_teacher_rollouts_per_task`
- 对比 fixed alpha vs `sync_batch_ema`

展示：`dr3/alpha_raw`, `dr3/alpha_ema` 与实际 teacher ratio 的一致性及 reward 影响。

#### Pilot-5：用 \(p/q\)（odds）当 teacher ratio 的对照（解释“为何不用 p/q”）
目标：把 reviewer 质疑点用实证压实：\(p/q\) 高方差/无界导致退化。

做法：加开关让 teacher ratio 使用 \(\hat r=D/(1-D)\)（强 clip/dual）替代 \(w_\alpha\)；其余不变。

预期：更多极端、更高 clipfrac、更低 ESS、更差 reward。

### Q3：怎么证明最后有效的是分布矫正，而非只是 fade-out？

建议用“必要性 + 剂量反应 + 机制证据 + 反事实对照”四段式：

1) **必要性（ablation）**：固定同一套 gate（同一 adaptive_weight 超参），比较 DR³ on vs off。
2) **剂量反应**：扫 gate 强度（\(\tau,\epsilon,\beta,\text{power}\)），验证 DR³ 的优势在多种 gate 强度下都存在，而不是某个 gate 超参碰巧。
3) **机制证据**：展示 DR³ 在机制变量上降低极端权重/提高 ESS，并减少“全 clip 区”的样本比例。
4) **反事实对照（最强）**：做 Pilot-2（teacher 强度对齐）——强度相同仍更好，reviewer 很难再说“只是 fade-out”。

