# 3PM 导师会议准备 — DUET WebShop 1.5B 深度复盘（第一性原理版）

**日期**: 2026-04-20
**原则**: 每一个术语都定义；每一个论断都可以追溯到代码或数据
**阅读方式**: 从 §1 开始逐节读，术语第一次出现时都在 §0 查好

---

## §0 术语表（所有后续论证用到的基本概念）

| 术语 | 定义 | 在哪里用 |
|---|---|---|
| **π_θ(a\|s)** | 当前策略在状态 s 下选动作 a 的概率；`θ` 是模型权重 | 所有 RL 公式 |
| **z(a)** | softmax 之前的 logit：`π_θ(a\|s) = softmax(z)[a] = exp(z(a))/Σexp(z(a'))` | 梯度推导 |
| **Teacher token** | 一条 teacher demonstration 轨迹里出现的 token（`a*`）。WebShop 里典型例子：`click[bright white]`、`click[3x-large]` | 反复出现 |
| **On-policy token** | 当前模型自己生成的 rollout 里出现的 token | 反复出现 |
| **Log-prob `log π_θ(a*\|s)`** | 模型给 teacher token 的对数概率；越负代表越"不像 teacher" | BC loss |
| **A(τ)** | trajectory-level advantage；这条轨迹比 group 平均回报高多少（GRPO 标准化后） | DR3 loss |
| **w_hat** | 判别器学到的 density ratio 估计 ≈ π_teacher(s,a) / π_θ(s,a)，**是整条轨迹一个标量**，不是每个 token 一个 | DR3 loss |
| **PPO clip** | 把 ratio = π_new/π_old 限制在 `[1−ε, 1+ε]` 的操作，防止单步更新过大 | DR3 loss |
| **SFT loss / BC loss** | Behavior Cloning 损失：`−log π_θ(a*\|s)`；无条件让 teacher token 概率变大 | CHORD, v24 |
| **Unit coefficient** | 损失的系数是 1（或常数 μ），不随样本内容变化 | BC 的特征 |
| **Sequence-level coefficient** | 损失的系数是整条轨迹一个值（如 A(τ) 或 w_hat），一条轨迹里所有 token 共享这个系数 | DR3 的特征 |
| **Per-token coefficient** | 每个 token 的系数独立由 softmax 几何决定 | BC 的特征 |
| **μ_t** | CHORD / v24 里 BC 损失的权重，随训练步数衰减的 schedule；v24 是 0.3 → 0.05 over 25 steps | curriculum 核心参数 |

---

## §1 先把 BC 和 DR3 的梯度公式推到第一性原理

这一节是整份 prep 的基石。你会在这里看到 "BC per-token unit coefficient" 和 "DR3 sequence-level × advantage × clip" 到底是什么意思。

### 1.1 Softmax 梯度的基本事实（谁都逃不掉的）

对于任意 loss `L = −log π_θ(a*|s)`（即把 teacher token `a*` 的概率拉高），对 logit `z(a)` 的梯度是：

$$
\frac{\partial L}{\partial z(a)} = \pi_\theta(a|s) - \mathbb{1}_{a=a^*}
$$

写开来：
- 对 `z(a*)` (teacher 选的那个 token)：梯度是 `π_θ(a*|s) − 1`（负数，push UP）
- 对其他 `z(a')` (非 teacher token)：梯度是 `π_θ(a'|s)`（正数，push DOWN）

**用梯度下降更新 logit**：
$$
\Delta z(a^*) = -\eta \cdot (\pi_\theta(a^*|s) - 1) = \eta \cdot (1 - \pi_\theta(a^*|s))
$$

**这就是 "per-token surprise-weighted" 的数学来源**：
- 如果 `π_θ(a*|s) ≈ 1`（模型已经很确定要出这个 token），`1 - π ≈ 0`，更新几乎为零（不用再学）
- 如果 `π_θ(a*|s) ≈ 0`（模型完全意料之外），`1 - π ≈ 1`，更新量最大（把不会的教给它）
- "Surprise" = 模型当前预期之外 = 低概率 = 梯度大

所以 "surprise-weighted" 不是我们加的什么 trick，**这是 softmax 的自然性质**。BC 直接把这个性质用起来了。

### 1.2 BC loss 对单个 teacher token 的完整梯度

CHORD 的 SFT loss（`het_core_algos.py:1723-1779`）：
$$
L_{BC} = \mu \cdot (-\log \pi_\theta(a^*|s))
$$

对 logit 的梯度：
$$
\frac{\partial L_{BC}}{\partial z(a^*)} = \mu \cdot (\pi_\theta(a^*|s) - 1)
$$

每一步 logit 增量：
$$
\Delta z(a^*) = \mu \cdot (1 - \pi_\theta(a^*|s))
$$

**关键性质**：
- **系数 μ 是 "unit"**：所有 teacher token 共享一个系数（μ），这个系数不依赖 token 内容，不依赖轨迹回报
- **"Per-token" 因为梯度的实际大小由 `(1 − π_θ)` 即 token 自身的 surprise 决定**
- **"Unconditional 正号" 因为 `(1 − π) > 0` 永远成立**，所以梯度**永远是把这个 teacher token 的 logit 推上去**

### 1.3 DR3 loss 对单个 teacher token 的完整梯度

DR3 teacher loss（`het_core_algos.py:393-402`，以 `teacher_use_log_prob: false` 模式）：
$$
L_{DR3} = -A(\tau) \cdot \text{clip}\left(w_\text{hat} \cdot \pi_\theta(a^*|s),\ 1-\epsilon,\ 1+\epsilon\right)
$$

注意 **`w_hat` 是整条 teacher 轨迹的一个判别器分数**，不是每个 token 一个。

对 logit 的梯度（在 PPO clip 未触发区）：
$$
\frac{\partial L_{DR3}}{\partial z(a^*)} = -A(\tau) \cdot w_\text{hat} \cdot \pi_\theta(a^*|s) \cdot (1 - \pi_\theta(a^*|s))
$$

每一步 logit 增量：
$$
\Delta z(a^*) = A(\tau) \cdot w_\text{hat} \cdot \pi_\theta(a^*|s) \cdot (1 - \pi_\theta(a^*|s))
$$

**关键性质**：
- **系数 `A(τ) · w_hat` 是 sequence-level**：整条轨迹共享
- **梯度大小 ∝ `π_θ(1 − π_θ)`**，即 `p(1−p)` —— **这是 Bernoulli 方差，在 p=0.5 处最大，在 p→0 或 p→1 处都趋向 0**
- **"Conditional" 因为正号依赖 A(τ) > 0**：如果 teacher 轨迹在 GRPO 里被判定为低于平均（A<0，罕见但可能），DR3 会把 teacher token **推下去**

### 1.4 具体数值 case study：稀有 SKU token

**场景**：WebShop 产品详情页，teacher 要点 `click[bright white]`。刚训练开始时模型从来没见过这个 SKU 字符串，所以 `π_θ(a* | s) = 10⁻⁴`（非常稀有）。

CHORD 的 μ_peak = 0.9，DR3 的典型 `A(τ) · w_hat ≈ 1 × 1 = 1`。

**Logit 每步增量对比**：

| 方法 | 公式 | 当 p_θ = 10⁻⁴ 时每步 Δz | 10 步后 p_θ | 100 步后 p_θ |
|---|---|---:|---:|---:|
| CHORD (BC, μ=0.9) | μ·(1−p) | 0.9 × 0.9999 ≈ **0.9** | **exp(9) ≈ 8000 倍** → p ≈ 0.4 | p ≈ 1.0 |
| DR3 (PPO clip 未触发) | A·w·p·(1−p) | 1 × 1 × 10⁻⁴ × 1 ≈ **10⁻⁴** | exp(10⁻³) ≈ 1.001 倍 → p 基本没动 | exp(0.01) ≈ 1.01 倍 → p 基本没动 |
| DR3 (PPO clip 触发) | 被 PPO 限速，每步 p_new ≤ 1.2 × p_old | 最多 +20% | 1.2¹⁰ = 6.2 倍 → p ≈ 6×10⁻⁴ | 1.2¹⁰⁰ ≈ 10⁷ 倍 → p ≈ 1.0 |

**直觉解释**：
- CHORD 对 logit 的推力是**常数级**（μ ≈ 0.9），不随 p_θ 变化；10 步能把 p_θ 从 10⁻⁴ 拉到接近 1
- DR3 对 logit 的推力是 `p(1-p)` 的**乘性**，在 p_θ 很小时几乎为零；**梯度消失**在稀有 token 上；必须靠 PPO 的 ratio clip 才有可能缓慢积累

**这就是为什么 CHORD 学得会 `click[bright white]`，DUET v1 学不会** —— 不是 clip 在限制 DR3，而是 DR3 的系数本身在稀有 token 上趋向于零。

**经验证据**：`actor/off_pg_cliphit_rate = 0.000` 全程 —— PPO clip 从未触发过，所以不是 clip 的锅，是 `p·(1-p)` 几何的锅（v25 分析已确认 clip 不 binding）。

### 1.5 总结：三个区分性质，精确定义

| 性质 | 数学定义 | 为什么重要 |
|---|---|---|
| **Teacher-specific** | Loss 只在 teacher token 上计算；anchor 点是 teacher 轨迹 | 决定 policy 往哪里拉；KL-to-ref 往 pretrained 方向拉，不是 teacher 方向 |
| **Per-token surprise-weighted** | 梯度 ∝ `(1 − π_θ)`（BC）而不是 ∝ `π_θ(1-π_θ)`（DR3）；低概率 teacher token 获得最大推力 | 决定能否快速学稀有 token |
| **Unconditional 正号** | 梯度方向永远是"推高 teacher token"；不依赖 A(τ) 或 reward | 确保每一步都在朝 teacher 方向走，没有抵消 |

**BC 三者全占**，**DR3 只占 (teacher-specific)**。这就是本文章核心发现。

---

## §2 Q1: 为什么 CHORD 在 WebShop 1.5B 上比 DUET-v1 好？

### 2.1 问题 restate

事实：
- CHORD Val@100 = **0.603**
- DUET-v1 Val@100 = **0.549**
- Gap = 5.4pp

表面上看，两者都没用 teacher logit，都是 "data-only" 方法，为什么有差距？

### 2.2 第一性原理答案

CHORD 的 teacher 梯度有 (per-token × unit coefficient × unconditional 正号)，**三者都有**。
DUET-v1 的 DR3 teacher 梯度只有 (teacher-specific)，**另外两者都没有**，并且还带 `p·(1−p)` 的乘性抑制。

**后果**：对稀有 SKU token（WebShop 里大量存在），CHORD 能在 10 步内把 `click[<color>]` 从 p=10⁻⁴ 拉到 p=0.5；DR3 即便活了 100 步也只是在 PPO clip 限速下慢慢爬。

**行为级证据**（case-analyst 分析，200 个 val 匹配任务，step 100）：

| 变体 | 任意 option 点击 | Teacher-exact option | 长轨死循环 |
|---|---:|---:|---:|
| **DUET-v1** (DR3+SC，无 BC) | 39.5% | **33.0%** | 8/200 |
| **CHORD** (μ=0.9→0.05) | 92.0% | **72.5%** | 16/200 |
| **v24** (DR3+SC+衰减 BC) | 78.0% | **61.0%** | **0/200** |

看 "Teacher-exact option" 列：33% → 72.5% → 61%。CHORD 比 DUET-v1 在"点对 option"这件事上领先 40 个百分点。这直接对应 +5.4pp 的 val reward（一条轨迹最终 reward 很大一部分取决于有没有点对 color/size）。

### 2.3 DR3 warmup 也有影响（次要因素）

DR3 的 `apply_warmup_steps = 10`，意味着**前 10 步 DR3 完全不激活**。DUET-v1 在 step 1-9 基本只是 GRPO + SC（on-policy 方法）。CHORD 从 step 1 就有 μ=0.9 的强 BC 在推 teacher token。

这个 9 步的 head-start 让 CHORD 先把 teacher 的语法和 SKU 概念印刻到 policy 里。

### 2.4 讲给导师的 1 句话（可直接念）

> "CHORD 的 SFT 梯度是 per-token 的 unit 系数，DR3 的 teacher 梯度是 sequence-level 的 `A·w_hat` 乘 softmax 的 `p(1-p)` 方差。对稀有 SKU token (p = 10⁻⁴)，CHORD 一步推 log π 上升 0.9，DR3 一步只推 10⁻⁴；相差四个数量级。经验上表现为 CHORD 在 200 个验证任务里 72.5% 点对 teacher 指定的 option，DUET v1 只有 33%。这 40pp 的 option-click 能力差直接对应 5.4pp 的 val reward 差。"

---

## §3 Q2: 为什么 v24 work？能否推广到 ALFWorld 和 3B？

### 3.1 v24 是什么

$$
L_\text{v24} = \underbrace{L_\text{DR3-teacher}}_\text{credit assignment} + \underbrace{L_\text{SC-onpolicy}}_\text{state shaping} + \mu_t \cdot \underbrace{L_\text{BC-teacher}}_\text{token identity}
$$

- `μ_t` 从 0.3 衰减到 0.05，over 25 steps
- `chord_use_token_weighting: false`（即 BC 不加 p(1-p) 权重，是纯 unit）
- 代码修改在 `het_actor.py:L1754-1756`：允许 DR3 和 CHORD SFT 共存（之前是 mutually exclusive）

Val@100 = **0.678**，比 CHORD (0.603) 高 7.5pp，比 DUET-v1 (0.549) 高 12.9pp。

### 3.2 为什么只有 BC 能补齐这块？—— 4 个 rescue 失败的精确数字

我们做了 4 个不加 BC 但改其他东西的 "narrative rescue" 变体，想看看能不能在不引入 BC 的情况下达到 v24 水平：

| 变体 | 改了什么 | 改变的数学量 | Val@100 | 结论 |
|---|---|---|---:|---|
| **v28** | `w_hat_ema_alpha: 0.3 → 0.1` | w_hat 的指数平滑系数变小 → w_hat 方差降低 | **0.495** | 降噪不够 |
| **v29** | `off_cliprange_high: 0.6 → 2.0` + `w_hat_ema_alpha: 0.3 → 0.1` + `clip_max: 2.0 → 5.0` | 放宽 PPO clip + 降 w_hat 方差 + 放宽 w_hat 上限 | **0.511** | 组合也不行 |
| **v30** | `kl_loss_coef: 0.001 → 0.01` | KL penalty 增强 10× | **0.520** | KL 防崩但不涨 |
| **v33** | `disc_temperature: 1.5 → 3.0` | 判别器 logit 除以 3 → 软化判别器输出 → w_hat 更接近 1 | **0.520** | 判别器软化无效 |

参照点：
- v12（baseline，DR3 稳定化，无 BC） = **0.431**
- CHORD = 0.603
- **v24（有 BC）= 0.678**

**规律：所有不加 BC 的 rescue 都卡在 0.49-0.52**。这是一个**经验上非常硬的 ceiling**。

### 3.3 为什么每个 rescue 都失败？—— 对照三个性质

把 4 个 rescue 摆到"三性质矩阵"里：

| 变体 | Teacher-specific | Per-token surprise | Unconditional 正号 | 能达到 v24 水平？ |
|---|:-:|:-:|:-:|:-:|
| v28 (EMA 降噪) | ❌ 作用在 w_hat 上，不改变梯度结构 | ❌ 同 | ❌ 同 | ❌ |
| v29 (放宽 clip 组合) | ❌ 只影响 PPO clip 边界 | ❌ 同 | ❌ 同 | ❌ |
| v30 (强 KL) | ❌ **KL anchor 是 ref model（pretrained），不是 teacher** | ✅ 有（softmax 性质） | ⚠️ 仅当 π_θ 偏离 ref | ❌ |
| v33 (软 disc) | ❌ 改 w_hat 分布形状，不改变梯度结构 | ❌ 同 | ❌ 同 | ❌ |
| **v24 (BC)** | ✅ Loss 锚点就是 teacher 轨迹 | ✅ ∂/∂z = 1−π_θ | ✅ 恒正 | ✅ |

**v30 的细节很关键**：KL-to-ref 看起来"有 per-token 性质"（确实），但它的 anchor 是**预训练模型 `π_ref`**，不是 teacher。所以 KL 把 policy 拉**回预训练**，不是拉**向 teacher**。在 WebShop 上，预训练模型根本不会点 SKU option，KL 只是不让 policy 漂移太远，但漂不向 teacher 那侧。

这就是为什么 v30 完美稳定（grad_norm=3.2, kl=0.18），却只到 0.520 —— **稳定但不学 teacher-specific 的 option token**。

### 3.4 μ_valley = 0.05 的第二个作用：语法保护

v25 实验直接证明了这一点。v25 配置：去掉 BC + 放宽 PPO clip（`off_cliprange_high: 0.6 → 2.0`）。结果：

- step 1-97：正常训练，reward 爬升到 ~0.62
- step 98：**单次 toxic gradient**，response_length 从 2053 掉到 1080
- step 100：Val reward = **−0.041**，grad_norm 爆炸到 48.4，policy 完全崩坏

失败模式（case-analyst 分析 v25 @ step 99 rollouts）：
- 47% rollout 输出双开 `<action>` 标签 → env reject
- 69% 在 `<action>` 里套幻觉 wrapper：`<search>`, `<click>`, `<when>`, `<story>` 等等
- 极端：traj 11 输出递归 `<when result="...">` XML 树

**诊断**：没有 BC 持续锚定 `<action>`, `</action>` 这种低概率 grammar token，PPO 的更新会慢慢把这些语法 token 的 logit 漂掉。一旦语法框架破，后续就是雪崩。

**v24 的 μ_valley = 0.05 解决这个问题**：即使过了 decay 阶段，仍保持 5% 的 BC 强度，**永远在锚 grammar token**。

所以 BC 在 v24 里兼任两个角色：
1. **前期（μ=0.3）**：强力推稀有 SKU token 概率
2. **后期（μ=0.05 floor）**：持续守住语法 token

### 3.5 推广到 ALFWorld / 3B / 7B 的预测

BC 的价值取决于三个环境/模型属性：

| 属性 | BC 价值 | 原因 |
|---|---|---|
| **Rare-token gap 大** | ↑ | teacher 轨迹里 SKU-specific token 很多（WebShop）；模板化动作很少（ALFWorld） |
| **Format fragility 高** | ↑ | 语法复杂容易崩（WebShop `<action>click[...]</action>`）vs 简单（ALFWorld `go to countertop 1`） |
| **Model capacity 小** | ↑ | 小模型对稀有 token 的先验支持弱 |

具体预测：

| 环境 × 规模 | rare-token | format fragility | model capacity | 预测 v24 gain vs DUET-without-BC |
|---|:-:|:-:|:-:|---|
| WebShop 1.5B | 高 | 中 | 低 | **+10pp（已验证：+12.9pp）** |
| WebShop 3B | 高 | 中 | 中 | +3~5pp |
| WebShop 7B | 高 | 中 | 高 | +1~2pp（边际） |
| ALFWorld 1.5B | 低（模板） | 低 | 低 | **+0~3pp**（关键待验证） |
| ALFWorld 3B | 低 | 低 | 中 | +0~1pp |
| ALFWorld 7B | 低 | 低 | 高 | **≈ 0**，BC 可以彻底关掉 |

### 3.6 决定性实验：v24 on ALFWorld 1.5B

这**一个**实验能决定整个叙事成立还是崩塌：

- 如果 v24-with-BC ≥ DUET-without-BC（现有 32.5%）→ 框架**成立**，BC curriculum 在不需要时自动熄火（μ=0.05 很小，对高概率 token 的 `(1-p)` 权重本来就小）
- 如果 v24-with-BC **显著低于** DUET-without-BC → 框架**崩塌**，BC 是 WebShop-specific hack，不能作为通用 contribution

**Config diff**（基于 `alfworld_qwen1.5b_duet.yaml`）：
```yaml
actor_rollout_ref:
  actor:
    use_chord: true
    chord_mu_warmup_steps: 0
    chord_mu_decay_steps: 25
    chord_mu_peak: 0.3
    chord_mu_valley: 0.05
    chord_use_token_weighting: false
```

**预估 ~5h on GPU 0-3**。在 WebShop Round 4B 跑完后立即上。

### 3.7 讲给导师的 1 句话

> "v24 work 是因为 BC 同时具备三个 DR3 缺失的性质：teacher-specific（锚点在 teacher 而不是 ref）、per-token surprise（梯度 = 1-p 而不是 p(1-p)）、unconditional 正号（不依赖 A 符号）。我们做了 4 个只加稳定化不加 BC 的 rescue（EMA、放宽 clip、强 KL、软 disc），Val@100 全部卡在 0.49-0.52。唯一加 BC 的 v24 达到 0.678。推广性预测是 BC 价值随 rare-token gap 和 format fragility 上升。**ALFWorld 1.5B v24 是决定这个框架是通用还是 WebShop-only 的唯一实验**，必须立刻跑。"

---

## §4 Q3: 保持 dual-channel 叙事的干净方式？

### 4.1 问题

加了 BC 之后，叙事要怎么写？之前 DUET 卖点是 "DR3 (Action Channel) + SC (State Channel)"，现在多了 BC，是不是变成"三通道"了？

### 4.2 推荐叙事：Teacher-Gradient Curriculum

重新定义 Action Channel：**它是一个 gradient operator curriculum，不是单一算子**。

```
Action Channel (acts on teacher trajectories):
  L_action = μ_t · L_BC + L_DR3
  
  μ_t: 0.3 → 0.05 over 25 steps
  L_BC: teacher-token identity imprint (per-token, unconditional)
  L_DR3: trajectory-level credit assignment (density-ratio corrected)
  μ_valley = 0.05: format-token preservation (long-term anchor)

State Channel (acts on on-policy trajectories):
  L_sc = β · Φ(τ) · ∇log π_θ
  [unchanged]
```

**关键点**：
- 两个 operator 都作用在 teacher 轨迹上 → 逻辑上属于一个 channel
- μ_t 是两者之间的 curriculum：早期 BC 主导（建立 teacher-token 支持），后期 DR3 主导（优化 trajectory credit）
- State Channel 完全不变

### 4.3 为什么这个叙事诚实？

对比之前试过的 framing：

| Framing | 为什么失败 |
|---|---|
| Framing A（时间 curriculum） | μ schedule 看起来是为 WebShop 调的；不解释机制 |
| Framing B（token-level gating） | 代码里根本没实现过，写这个是说谎 |
| Framing C（automatic p_θ specialization） | 数学错（下界在 cold-start 处消失） + 经验证据不存在 + 实际 `chord_use_token_weighting: false` |
| **Teacher-Gradient Curriculum** | 基于 BC/DR3 的**实际数学性质差异**；不做空洞数学；承认 BC 的双重作用（identity + format） |

**诚实之处**：
1. 不声称 "p_θ 自动分工" —— 这是 Framing C 被证伪的
2. 不用 "support gap" cold-start theorem —— 这是 vacuous 的
3. 承认 μ_valley = 0.05 兼任 format 保护（v25 证据）
4. Novelty 在**density-ratio 驱动的自动 curriculum**（w_hat → 1 时 DR3 自然退出），而不是 BC 本身

### 4.4 如何回应审稿人

**R1: "DUET 不就是 CHORD + DR3 吗？"**
> CHORD 的 μ schedule 是人工常数衰减，没有 teacher 样本上的 trajectory credit assignment（CHORD 的 SFT 对所有 teacher token 用同一个系数，不管 trajectory 好坏），也没有自动 fade-out 机制。DUET 的 DR3 提供两者：(a) trajectory credit —— 成功的 teacher 轨迹得到更强的推力；(b) 自动 fade —— `w_hat → 1` 时 DR3 系数自动趋于 0。
>
> 经验验证：v22（constant μ=0.05）= 0.462，v23（constant μ=0.1）= 0.440，v24（decaying μ=0.3→0.05）= 0.678。同样是 BC，只有**衰减调度**才能配合 DR3 达到 0.678；这不是可有可无的细节。

**R2: "既然 BC 是必要的，还要 SC 吗？"**
> BC 解决 action-level 的 token identity 问题；SC 解决 state-visitation 的 credit assignment。两者正交。v4（SC off，保留 DR3+BC）在 WebShop 1.5B 掉到 **0.343** —— 硬证据。
>
> 机制层面：SC 奖励上到达 teacher-visited 状态的 on-policy 轨迹；BC 教具体的 action token。一个是 "去哪"，一个是 "做什么"。

**R3: "v24 的 recipe 是 WebShop-specific 吗？"**
> 理论上 BC 的价值 ∝ (rare-token gap × format fragility × 1/model_capacity)。ALFWorld 上 rare-token gap 小（模板化动作），预测 BC 贡献接近 0，μ curriculum 自动熄火。
>
> **待验证：ALFWorld 1.5B/3B 上 v24 的结果**。如果不劣于 DUET-without-BC，证明通用。

**R4（最危险，防御最弱）: "AWAC 作为单算子 baseline 呢？"**
> 我们没跑 AWAC。**诚实披露为 future work**。我们的 claim 降级为 "两个算子各自有清晰独立的分析" 而不是 "minimal design"。

### 4.5 讲给导师的 1 句话

> "Dual-channel 叙事保留为：Action Channel 是 BC 和 DR3 两个互补算子，μ_t 是算子间的 curriculum；State Channel 不变。Novelty 集中在 density-ratio 驱动的自动 curriculum (`w_hat → 1` 自然退出)，这是 CHORD 做不到的。叙事诚实因为避开了 Framing C 的三个致命错误（数学错、无经验支持、与实现矛盾）。"

---

## §5 必须立刻决定的行动项

### 5.1 最高优先级：v24 on ALFWorld 1.5B（决定命运）

**目的**：验证 v24 框架是通用还是 WebShop-specific

**时间**：~5h on GPU 0-3

**执行**：
```bash
# 创建 config
cp config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet.yaml \
   config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v24.yaml

# 编辑：加 chord_mu_* 参数（见 §3.6）

# 跑
python launcher.py --conf config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v24.yaml
```

### 5.2 次优先级：AWAC baseline（R4 防御）

如果有时间。~5h。否则 future work 中说明。

### 5.3 低优先级：继续 Round 4B

- v36 (const tiny BC μ=0.05) 会在 ~15:00 完成 —— 告诉我们 "最小 BC 剂量"
- v31, v32 —— 会后再看

### 5.4 3B CHORD ALFWorld（之前 kill 了，需要重跑）

非紧急，但 EXPERIMENT_LOG 里有个空格，某个时候要补。

---

## §6 7 分钟 pitch 建议讲法（给导师）

1. **现状（30s）**：WebShop 1.5B 上 v24（DR3 + SC + 衰减 BC）达到 **0.678**，比 CHORD（0.603）高 7.5pp，比原 DUET（0.549）高 12.9pp。
2. **Q1 机制（90s）**：DR3 的 teacher 梯度是 `A(τ) · w_hat · p(1-p)`，p 小时梯度消失；CHORD 的 BC 是 `μ · (1-p)`，p 小时梯度最大。稀有 SKU token (`click[bright white]`) 上 CHORD 快 4 个数量级。[展示 fig2]
3. **Q2 v24 独门（120s）**：BC 同时具备 3 个 DR3 缺的性质。4 个 rescue（EMA/宽 clip/强 KL/软 disc）Val 全部卡在 0.49-0.52；只有加 BC 的 v24 到 0.678。μ_valley 还兼任 format 保护（v25 证据）。[展示 fig3]
4. **Q3 叙事（90s）**：Action Channel = BC + DR3 的 gradient-operator curriculum，State Channel 不变。Novelty 在 `w_hat → 1` 自动 fade-out。[展示 fig5]
5. **关键未验证实验（30s）**：v24 on ALFWorld 1.5B 决定框架通用性，~5h 就能知道。
6. **弱点主动披露（30s）**：AWAC baseline 没跑；作为 future work。

---

## §7 图表快捷引用

| 图 | 一句话说明 | 文件 |
|---|---|---|
| Fig 1 | 35 变体 Val@100 全景：no-BC 全部在 0.50 左右，v24 独峰 0.678 | `figures/fig1_variant_landscape.png` |
| Fig 2 | CHORD vs v1 training dynamics（reward、kl、length、entropy）4 面板 | `figures/fig2_chord_vs_duet_v1_dynamics.png` |
| Fig 3 | No-BC ceiling：v12 + 4 rescue + CHORD + v24 bar chart —— 最强证据 | `figures/fig3_no_bc_ceiling.png` |
| Fig 4 | v12 vs v24 6 面板 metric delta —— BC 的机制级作用 | `figures/fig4_v12_vs_v24_mechanism.png` |
| Fig 5 | 跨 scale 跨环境 gain 预测 —— BC 贡献随 scale 衰减 | `figures/fig5_scaling_prediction.png` |

---

## §8 支撑文档

| 文档 | 内容 |
|---|---|
| `duet_webshop_1.5b_final_retrospective.md` | exp-analyst 全景复盘 |
| `duet_final_theory_synthesis.md` | theory-researcher 数学 synthesis |
| `chord_vs_duet_v1_trajectory_diff.md` | case-analyst trajectory 级比较 |
| `webshop_1.5b_duet_v1_to_v24_ablation_analysis.md` | 24 变体 ablation 详表 |
| `webshop_1.5b_duet_trajectory_case_analysis.md` | v1/v8/v12/v24/CHORD token 级对比 |
| `v25_divergence_analysis.md` | v25 崩溃的 metric 诊断（off_pg_cliphit_rate=0 证据） |
| `v25_trajectory_collapse.md` | v25 语法 token drift 证据（`<story>` 等幻觉） |
| `framing_C_agent_team_verdict.md` | 之前失败 framing 的三轴证伪 |

---

## §9 一页速记（最后的最后，可以背下来）

**DR3 梯度 = `A(τ) · w_hat · p(1-p) · ∇log π_θ`**，p 小时梯度消失，系数是 sequence-level 标量。
**BC 梯度 = `μ · (1-p) · ∇log π_θ`**，p 小时梯度最大，系数是 per-token unit。

**BC 同时有**：teacher-specific、per-token surprise、unconditional 正号。
**DR3 只有**：teacher-specific。
**KL 有**：per-token surprise、unconditional（对 ref），但**不是 teacher-specific**。
**EMA/soft disc**：只是降 w_hat 方差，**不改变梯度结构**，所以不能替代 BC。

**4 个 no-BC rescue 都卡在 0.49-0.52（v28=0.495, v29=0.511, v30=0.520, v33=0.520）；v24 到 0.678**。

**μ_valley=0.05 兼任 format 保护**；v25 去掉 BC 后 step 98 语法崩（`<story>`, `<when>` 幻觉）。

**决定性实验**：v24 on ALFWorld 1.5B。一个实验定框架命运。

---

**DONE — 可以从 §0 开始读了。**
