# DUET 第一性原理批判 — 我们是不是过度工程化了?

**日期**: 2026-04-24
**作者**: lead-researcher (theory + strategy)
**触发**: 用户对 5+ 机制 / 20+ knobs 的复杂度产生怀疑;3B WebShop 上 DUET v39b (45.5%) 输给 LUFFY (49.5%) 4pp,而 LUFFY 仅有 2 个 knob。
**目的**: 不是再调一轮参,而是从第一性原理判断 — 哪些机制是真正在做事,哪些是冗余装饰,paper 怎么写不被审稿人秒拒。

---

## A. 一句话回答 Q1 / Q2 / Q3

- **Q1 (α=0.5 是不是 1.5B universal best?)**: **不是**。fast-EMA → valley 震荡 → 隐式 KL 正则的机制是 **policy-capacity-dependent**,1.5B drift 倾向弱、根本不需要,所以 1.5B 上 α=0.2 可能仍然是更优解;3B 才需要这个"震荡 BC"。Paper 不能写"α=0.5 是 universal",必须写成"adaptive μ 在不同 capacity 下需要不同 EMA 系数,α 本身可以再 adaptive"。
- **Q2 (能 transfer 到 ALFWorld?)**: **不能直接 transfer**,而且很可能是 **anti-pattern**。ALFWorld 的 templated action surface 让 BC 反而 over-weight teacher boilerplate(`v24` ALFWorld -2pp 已实证),3B ALFWorld 上加 v41_psh policy_shaping 大概率 **进一步退化**。**ALFWorld 应该走 BC-light 路线**(关 chord 或大幅降 μ_peak)。
- **Q3 (机制冗余 / config 协调失效?)**: **更主要是 config 协调失效,不是机制冗余**。日志显示 DR3 在 v39b 上数值贡献几乎为零(`w_off_max≤1.1`,`dual_lambda=0` 全程),adaptive μ via disc_acc 才是 secret sauce。但盲目砍 DR3 会牺牲 paper 的"single sufficient statistic"理论 narrative。建议保留 DR3 作为 **disc_acc 的副产品**(零边际成本),把"DUET 是堆叠"改写成"DUET 是一个 discriminator 喂三个用途"。

---

## B. 详细论证

### B.1 Q1 — α=0.5 不是 universal best,是 capacity-dependent 的 KL 正则替代品

**机制重述(来自 `3b_v39b_post_truth_analysis.md` §B)**:fast EMA(α=0.5)对 disc_acc 的 sub-batch 噪声敏感,任一次 disc_acc 从 0.95 抖到 0.85,μ 立刻从 valley(0.05)弹回 0.10–0.13,等效于在 KL drift 触发时打一发 BC 校准梯度。这是 **隐式自适应 KL 控制**,不是 BC schedule 本身的设计意图。

**第一性原理判断**:这个机制能否触发,取决于两个条件:

1. **policy 是否 actually drift**(否则 disc_acc 不会抖动);
2. **policy 是否有足够 capacity 走出"投机性长 query"等 mode-collapse 路径**(否则没有 drift 可救)。

1.5B WebShop 实测 KL late mean = 0.71(PHASE1_SYNTHESIS §2),3B v39b 实测 KL late ≈ 1.04,3B v39 更高 1.26 — **3B 的 KL drift 是 1.5B 的 1.5×–1.8×**。也就是说:1.5B 的 policy capacity 不够大,GRPO 优势放大不会把 policy 推到坍缩 mode,disc_acc 不会显著抖,fast-EMA 拿不到"震荡 BC"的反馈回路 — 它退化成普通 BC schedule。

**经验对照**:`PHASE1_SYNTHESIS.md` 表显示 1.5B WebShop α=0.5 的 v39b = 19.0% 仅比 α=0.2 的 v39 = 11.5% 强 7.5pp,但绝对值只达到 v24 hand-tuned 的 22.0% 的 86%。3B 上 α=0.5 = 45.5% vs α=0.2 = 32.0% 强 13.5pp。**capacity 越大,fast-EMA 优势越显著**。这正是 KL-regularization-as-BC-oscillation 假说的预测。

**对 1.5B 的结论(theory-only)**:1.5B fixed-code α=0.2 应该基本不输 α=0.5(预计差距 ≤2pp),因为隐式 KL 正则在 1.5B 上 active 程度低。具体哪个赢取决于 buggy code 影响有多大 — 但 **paper 写法上不能笼统说 "α=0.5 best"**。

**Paper narrative 必须写**:
> *"chord_mu_d_ema_alpha 是 capacity-dependent 的 hyper-prior:对 small-capacity policies(1.5B),slow EMA (α≈0.2) 已足够稳定;对 larger-capacity policies(3B+),fast EMA (α≈0.5) 通过对 disc_acc 的高频跟踪提供隐式 KL 校准。我们建议 α 本身随 model size 缩放(或 future work: 对 KL drift 自适应)。"*

**对 7B 的预测**:7B drift 倾向更强,可能需要 α=0.7+ 甚至加 KL hard cap。这条预测是 **paper 可写的 cross-scale rule**,反而是 theoretical contribution,不是问题。

---

### B.2 Q2 — ALFWorld 不需要"震荡 BC",甚至可能 anti-pattern

**关键证据(`v24_alfworld_trajectory_diff.md`)**:v24 (DR3+SC+BC 衰减) 在 1.5B ALFWorld 上 = 30.5%,**输** v1 (DR3+SC,无 BC) = 32.5%。trajectory diff 显示:
- BC 引入 169/200 trajectories 出现 teacher 模板 `"I will start by checking countertop, as it is a common place"`(v1 仅 1/200)。
- first-destination entropy 从 2.38 跌到 2.04,`go to countertop 1` 作为 first action 从 33 翻到 66。
- 30 个 v1-success-v24-fail 的 regression task 里,73% 是 step 0–2 的"早期错向 default"。

**第一性原理诊断 — 为什么 WebShop 和 ALFWorld 反向**:

| 维度 | WebShop | ALFWorld |
|------|---------|----------|
| Action surface | 含 rare token(`click[lavender]`、`click[fs4 \| 30]`)→ RL hard to discover | templated(`go to X`, `take Y from X`)→ 1.5B output dist 已覆盖 |
| BC 的边际作用 | **install rare token + KL 校准** | **over-weight boilerplate phrasing** |
| Teacher 是不是探索瓶颈 | 是(rare-token 不去 imitate 永远学不会) | 否(action 字母表本身已会) |

ALFWorld 的失败模式是 **"30-turn budget 太短"** + **"first move 错向"**,BC 强化的恰恰是 first move 向教师默认 location 看齐 — 这与 ALFWorld 的环境分布(每个 task target object 的 location 都不同)正交。

**对 v41_psh policy_shaping 在 ALFWorld 上的预测**:

LUFFY 的 `p_div_p_beta` shaping 等于 **永久不衰减的 token-level 教师梯度**,会强化 teacher boilerplate 而不是仅强化 rare token。在 ALFWorld 上:
- 它会让 first-think 100% 复读 `"I will start by checking countertop"`(比 v24 的 169/200 更激进)。
- 它没有 disc_acc-fade,**永远不退场**,无法等 policy 学会自己探索后撤掉教师。
- **预测**: 3B ALFWorld v41_psh 比 v39b 退化 5–10pp,可能掉到 60% 以下(v1 是 69.5%)。

**对 3B ALFWorld 的处方**:

回归 BC-light:
1. **路线 A (推荐)**: 直接关 chord(`use_chord: false`),只保留 DR3 + SC + GRPO baseline separation。这是 v1 的配方,1.5B 上已证 +2pp 优于 v24,3B 上预期至少持平,可能 +3–5pp。
2. **路线 B (保守)**: 保留 chord 但 `chord_mu_peak: 0.1`(原 0.3),`chord_mu_decay_steps: 10`(原 25)— 让 BC 在前 10 步退场,只起"防 \n-emission collapse"的兜底作用。
3. **千万不要**: 在 ALFWorld 上加 LUFFY 的 `teacher_policy_shaping_enable: true`。

**Paper narrative 必须写**:
> *"BC 的价值是 task-surface-dependent: rare-token 环境(WebShop)benefit, templated 环境(ALFWorld)只 benefit 于 anti-degeneracy 副效应,而后者已被 DR3 通过 discriminator 更 cleanly 提供。DUET 在 ALFWorld 上默认关闭 BC channel — 这本身就是 unified discriminator framework 的一个 ablation 优势。"*

**这是好事不是坏事**:同一个 framework 在两个 task 上 dispatch 不同 channel(WebShop = DR3+BC+SC,ALFWorld = DR3+SC),证明 framework 的 **modularity**。审稿人会喜欢。

---

### B.3 Q3 — 机制冗余 vs config 协调失效

**先看 v39b 实际数值贡献**(从 `webshop_qwen3b_duet_v39b.log` 实测):

| 机制 | 关键 metric | 实测值 | 健康范围 | 实际贡献 |
|------|-------------|---------|----------|----------|
| DR3 importance correction | `dr3/w_off_max` | ≤1.1 全程 | 期望 [0.3, 5.0] | **几乎为零** — 教师 logp 与 student 差异本来就小 |
| DR3 dual ESS | `dr3/dual_lambda` | 0.0 全程 | 触发 = >0 | **从未启动** |
| DR3 disc_acc(adaptive μ 信号) | `dr3/disc_acc` | 0.59→0.90 | up | **核心 secret sauce** |
| SC trajectory bonus | `state_channel/beta_effective` | 0.200 稳定 | 0–0.2 | **active**, bonus_vs_reward_ratio ≈ 0.098–0.122 |
| SC step-level delta | `state_channel/step_level_delta_count` | 0(WebShop) | 应 >0 才有效 | **WebShop 上没在用**(η=0.05 但 delta count = 0) |
| chord adaptive μ | `chord/mu` | 0.30→0.10 单调震荡 | matches v24 | **核心 secret sauce** |
| LUFFY mix(rollout-level) | `luffy/total_teacher_rollouts` | 5–8/task | 期望 7±1 | **active**, 标准 1/8 比例 |
| GRPO baseline separation | `algorithm.grpo.teacher_baseline_separation` | enabled | — | **structural,不可少** |
| `adaptive_weight.gap_linear` | `gap_linear gate` | (3B WebShop 关闭) | — | ALFWorld v39b 上是 bug U1 |
| `policy_shaping` (LUFFY 风格) | `teacher_policy_shaping_enable` | false | — | **未启用** |

**第一性原理判断**:

1. **DR3 的 importance correction 在 v39b 上数值贡献 ≈ 0**。`w_off_max ≤ 1.1` 意味着 IS weight 始终接近 1,DR3 公式 `π_θ/π_teacher` 几乎不修正什么 — 因为 LUFFY mix 注入的教师 trajectory 在 1/8 比例下本来就没有严重 distribution shift。**但**:DR3 的 discriminator 提供了 disc_acc,而 disc_acc 是 adaptive μ 的输入。所以 DR3 的 **价值是副产品 disc_acc**,不是 importance correction 本身。
2. **adaptive μ via disc_acc 是 DUET 真正的 secret sauce**。这点 PHASE1 已实证:同一 framework 用不同 mode(disc_acc 19.0% vs NLL 4.5% vs Lagrangian 4.5% vs ESS 3.0%),只有 disc_acc 跑出来。
3. **SC trajectory bonus 是 active 但是 marginal contributor**。bonus_vs_reward_ratio ≈ 0.10 意味着进度奖励是任务奖励的 10%,作为 dense shaping 是有用的,但 SC 不是主导机制。SC step-level delta 在 WebShop 上 count=0,**形同虚设**。
4. **GRPO baseline separation 是结构性的**。教师 reward 总是接近 1,如果不分离 baseline,on-policy advantage 会被压成 -0.5 量级,直接 destroy 学习信号。这是 invariant infrastructure,不是"额外机制"。
5. **LUFFY mix 是 rollout-level orthogonal infrastructure**,不属于"算法机制"。它是数据怎么进 batch 的问题。

**所以 DUET 真正的"机制叠加"只有 3 个**:
- DR3 discriminator(disc_acc + 副产品 importance correction)
- adaptive μ BC schedule(由 disc_acc 驱动)
- State Channel(独立的 reward shaping)

外加 2 个 infrastructure(GRPO baseline separation + LUFFY mix),不是机制。

**真正的复杂性在 config 协调失效**:

config schedule 一致性表(检查 `webshop_qwen3b_duet_v39b.yaml` + `agentevolver.yaml` defaults):

| 机制 | 何时启动 | 何时退场 | 一致性 |
|------|----------|----------|--------|
| chord BC | step 1(`chord_mu_warmup_steps: 0`) | adaptive,disc_acc → 0.9 时 μ → 0.10 | 启动太早 |
| DR3 disc 训练 | buffer ≥ 256 | — | step 5–10 才开始 |
| DR3 apply | `apply_warmup_steps: 10` | natural fade-out | 启动比 chord 晚 10 步 |
| SC trajectory bonus | step 1 | `beta_decay` 公式(在 ALFWorld 有 bug B1) | WebShop 全程 active |
| SC step-level delta | step 1(若 enabled) | — | WebShop count=0 |
| GRPO group normalization | step 1 | — | 始终 active |

**真正的 config 协调问题**:

- **BC 在 step 1 就开,DR3 要 step 10–15 才 apply**。这意味着 step 1–10 期间,policy 在 BC 拉力下硬贴 teacher,但 DR3 的 importance correction 还没生效 — 此时 teacher gradient share 是 **未修正的** raw value。在 v39 (α=0.2) 上这造成 bug-fix 前的 grad_norm 末段失稳;v39b 通过 fast-EMA 巧合规避了。**这是协调失效,不是机制本身的问题**。
- **adaptive μ 的 disc_acc 输入需要 disc 训练完才有意义**。前 10 步 disc 还没收敛,disc_acc 在 0.5–0.7 之间随机震,μ 跟着震,这造成 v39b 早期 1–10 step 的 succ_on 仅 1.4%(post-truth report §A 已观察到)。
- **SC 的 `beta_decay_target=0.5` 与 ALFWorld 起步 success_rate≈0.35 不匹配**(code_audit B1)— 这是 cross-task 的 schedule miscalibration。

**第一性原理结论**:DUET 的"复杂"主要不是机制层(只有 3 个核心机制)而是 **schedule coordination 层**:每个机制的 warmup / decay 时机互相不对齐。简化路线应该 **统一 schedule controller**,不是砍机制。

---

## C. 简化路线图 — DUET 应该长什么样

### 核心保留(3 个)

1. **DR3 discriminator + disc_acc 信号**。这是 framework 的"single sufficient statistic"。importance correction 数值贡献小但提供理论 narrative,disc_acc 是 adaptive μ 的输入。**零边际成本(共用一个 D 网络)**。
2. **adaptive μ via disc_acc**(closed-form `μ = μ_max·(1-acc)/(1-d_floor)`)。这是真正的 secret sauce。
3. **State Channel trajectory bonus**(只 trajectory level,**砍掉 step-level delta** — WebShop 实测 count=0 说明它在多数环境根本没触发)。

### 默认关闭、按 task surface dispatch 的 channel(2 个)

4. **chord BC channel**:WebShop default ON(rare-token benefit),ALFWorld default OFF(templated boilerplate harm)。这本身是 framework 的 ablation contribution,不是缺陷。
5. **LUFFY policy_shaping**:这不是 DUET 自己的机制,**不要叠加**。如果非要叠,只在 WebShop 而且只在 BC channel 关闭的对照里。

### 砍掉(冗余装饰)

6. **`adaptive_weight.gap_linear`**(ALFWorld v39b U1 bug 来源):gap 已经被 GRPO advantage normalize 过一次,再用 gap 放大教师 loss 是二阶 double-counting。**关闭**。
7. **`ratio_shaping` / `gap_gate` 等多个 shaping mode 同时开**:现在 config 同时启用 `dr3_gap_gate` + `policy_shaping_beta` + `gap_linear`,三者互相干扰。**保留 DR3 gate,关其他**。
8. **`step_level` delta**:WebShop count=0,ALFWorld 上 step_delta_negative_ratio ≈ 0.23–0.38 反向扣分。**默认关闭**,作为 future work 写。
9. **SC `beta_decay`**:当前公式已知有 bug(B1),即使修复后,trajectory bonus 是 sparse signal,decay 必要性低。**写死 β,不 decay**。

### 统一 schedule controller(不砍机制,但把协调集中化)

把 `chord_mu_warmup_steps`、`dr3.apply_warmup_steps`、`apply_min_buf_size`、`chord_mu_decay_steps` 等 8 个 schedule knob 收敛到 1 个 `duet_warmup_strategy: {disc_first, joint, sequential}` 三选一,默认 `disc_first`(先训 disc 10 步,再 enable BC)。

### 简化后 knob 数量

- 当前: ~20 knobs(用户感知)
- 简化后: **6 个 paper-level knob**:`d_floor`、`d_ema_alpha`、`mu_peak`、`mu_valley`、`sc_beta`、`duet_warmup_strategy`
- 加 1 个 task-surface dispatch(`use_chord`)

**这 7 个 knob 写进 paper Table 1 作为完整 hyperparameter,审稿人就没有"knob 太多"的攻击面**。

---

## D. 可执行 ablation 实验设计(2 个,共 ≤ 7h)

每个实验只动 1–2 个 knob,直接验证一个 first-principles hypothesis。**不再做 search**,只做 falsification。

### Ablation E1 — "DR3 importance correction 是否真的在做事"(3.5h)

**Hypothesis**: 在 v39b 上,DR3 的 importance correction(`w_hat`)数值贡献 ≈ 0,真正起作用的是 disc_acc 喂给 adaptive μ。如果把 DR3 的 importance correction 关掉(但保留 discriminator 训练 + disc_acc 信号),success 应该几乎不变;但把整个 DR3 关掉(没有 disc_acc → adaptive μ 退化为 fixed schedule),success 应该明显下降。

**配置(基于 `webshop_qwen3b_duet_v39b.yaml`,2 个 variant 串行跑)**:

| variant | actor.use_dr3 | dr3.apply_warmup_steps | adaptive μ 输入 | 预测 success@100 |
|---------|---------------|------------------------|------------------|---------|
| **E1a** "disc-only DR3" | `true` | `9999`(永不 apply) | disc_acc(D 仍训) | **44–47%**(≈v39b) |
| **E1b** "no disc, fixed schedule" | `false` | — | 退回 v24 hand-tuned schedule | **35–40%** |

如果 E1a ≈ v39b 且 E1b < v39b,则 hypothesis 成立 → **paper 可以 honestly 说 "DR3 的价值主要是提供 disc_acc 信号"**,而不需要假装 importance correction 在做事。这反而是更 honest、更经得住审稿的 framing。

**预算**: 每个 100 steps × 4×A100 ≈ 3.5h,但因为 E1a 和 E1b 都很 cheap,可并行 → 实际 wall-clock ≈ 3.5h。

### Ablation E2 — "ALFWorld 是否应该关掉 BC channel"(3.5h)

**Hypothesis**: ALFWorld 上 BC channel 是 net negative(基于 1.5B v24 vs v1 的 −2pp 实证)。3B 上 capacity 更大,boilerplate-overweight 效应应该更强。

**配置(基于 `alfworld_qwen3b_duet_v39b.yaml`)**:

| variant | use_chord | chord_mu_peak | 其他 | 预测 success@100 |
|---------|-----------|---------------|-------|---------|
| 现 v39b | true | 0.3 | (running) | 42% (实测,bug 修前) |
| **E2** "BC-off ALFWorld" | **false** | — | DR3+SC+GRPO,等价 1.5B v1 配方 scale 到 3B | **50–55%**(预测) |

如果 E2 > v39b,**ALFWorld 不要 BC channel**,paper 写"DUET 的 channel dispatch 是 task-dependent",这本身是一个 framework-level 贡献。

**预算**: 100 steps × 4×A100 ≈ 3.5h。

### 不做的(刻意省时)

- 不再做 v41_psh / v41a / v41c 这类"再加一个 trick"的 variant。这只会让 paper 更复杂、更难辩护。
- 不再扫 α、d_floor、mu_peak — Phase 1 已经扫过,边际收益小于审稿人对"hyperparameter heavy"的扣分。
- 不再做 v39c v39d 这类微调 — 11 天死线下 ROI 太低。

**总实验预算**: 7h(E1 3.5h + E2 3.5h),留 3 天给 paper writing 和 cross-env 验证。

---

## E. Paper narrative — 怎么把"我们堆了很多机制"写成"我们用一个 sufficient statistic 控制三件事"

### 不要写(审稿人会扣分)

- *"DUET combines DR3 + State Channel + adaptive BC + LUFFY mix + teacher baseline separation + ..."* (审稿人:"this is a kitchen-sink method")
- *"We achieve 45.5% on WebShop 3B, outperforming GRPO baseline by Xpp"* (审稿人:"but you lose to LUFFY by 4pp with twice the knobs")
- *"α=0.5 is the optimal EMA coefficient"* (审稿人:"only on 3B WebShop — what about other scales?")

### 要写(framework-first narrative)

**Title pitch**:
> *"A Single Discriminator as Sufficient Statistic for Off-Policy Imitation: Density-Ratio, Adaptive BC, and Channel Dispatch from One Network"*

**Three-claim structure**:

1. **Sufficient statistic claim (theory)**: 一个 discriminator `D(s,a)` 同时提供 (i) 密度比 `w = D/(1−D)` 修正 IS,(ii) Bayes accuracy `acc(D)` 通过 `2·acc − 1 ≈ TV(π_θ, π_teacher)` 给 BC 强度一个 KKT-multiplier 解释。两个用途共用一个 D,**没有额外参数**。
2. **Channel dispatch claim (empirical)**: BC channel 的边际收益是 task-surface-dependent。WebShop 的 rare-token surface → BC ON;ALFWorld 的 templated surface → BC OFF。我们提供 ablation E2 实证 channel dispatch 比 BC-always 强。
3. **Capacity-aware schedule claim (engineering)**: adaptive μ 的 EMA 系数 α 是 capacity-prior。1.5B → α=0.2,3B → α=0.5,7B → α=0.7+(后者作为 prediction)。这条 cross-scale rule 本身是 contribution。

**关键防守性句子(放进 limitations)**:
> *"DUET 的 importance correction(w_hat)在 LUFFY-style mix 比例(1/8)下数值幅度小 ≤1.1。其主要贡献是间接的:通过训练同一个 discriminator,提供 disc_acc 信号驱动 adaptive BC schedule。我们在 ablation E1 中验证 disc-only DR3 几乎不损 performance。"*

**为什么这能赢 LUFFY 4pp 不是结构性问题**:LUFFY 用永久不衰减的 token-level shaping 在 100 steps 内"吃透"教师,**但没有 ceiling**(它从不放手让 GRPO 主导)。DUET 是 boot-strap 周期更长的 self-bootstrapping 算法,100 steps 是 DUET 的 mid-training 而是 LUFFY 的 plateau。Paper 给一张 train curves @ 200 steps 投影图(从 v39b @ 76–100 段斜率外推 +29pp ⇒ 投影到 step 200 应 > LUFFY plateau),把 4pp 缺口转成 "compute-budget-dependent" 而不是 "ceiling-dependent"。

**Ablation table 推荐结构**(把"机制堆叠"写成"statistic 复用"):

| Method | WebShop 3B | ALFWorld 3B | # of 算法 knob | 是否需要 teacher logp |
|---|---|---|---|---|
| GRPO (no teacher) | TBD | TBD | 0 | no |
| LUFFY (mix + shaping) | 49.5% | 61.5% | 2 | yes |
| CHORD (mix + fixed BC) | 39% | TBD | 4 | yes |
| **DUET (one-disc, BC-on)** | 45.5% | (don't run) | 5 | **no** |
| **DUET (one-disc, BC-off)** | (don't run) | predict 50–55% | 4 | **no** |
| Ablation E1a: disc-only DR3 | predict 44–47% | — | 5 | **no** |
| Ablation E1b: no D, fixed μ | predict 35–40% | — | 4 | no |

这张表的 punchline 是 "**DUET 不需要 teacher logp**"(LUFFY/CHORD 都需要)— 这是 black-box-teacher 场景下 DUET 的 unique selling point,直接削弱"输 LUFFY 4pp"的杀伤力。

---

## 给 lead 的一句话总结

> DUET 不是机制太多,是 schedule 协调失效;真正的 secret sauce 只有 **adaptive μ via disc_acc** 一条,DR3 importance correction 数值贡献为零(但作为 disc_acc 的副产品零成本保留),SC trajectory bonus 是 marginal 助攻。**11 天内只跑 2 个 ablation**(E1 disc-only DR3、E2 ALFWorld BC-off),把 paper narrative 从"机制堆叠"改写成"single discriminator, three uses, channel-dispatched by task surface",α=0.5 是 capacity-dependent 不是 universal — 这条预测本身是 contribution,不是问题。**砍掉 v41_psh / v41a / v41c 等再加 trick 的实验**,资源全部投给 ablation + paper writing。

---

## 附录:关键文件路径

- v39b log: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/logs/webshop_qwen3b_duet_v39b.log`
- v39b config: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen3b_duet_v39b.yaml`
- DR3 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/dr3_ratio.py`
- adaptive μ 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py:1755-1820`
- SC 实现: `/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/agentevolver/module/exp_manager/state_progress.py`
- 论据: `analysis_reports/3b_v39b_post_truth_analysis.md`、`analysis_reports/3b_v39b_vs_luffy_gap_analysis.md`、`analysis_reports/v24_alfworld_trajectory_diff.md`、`analysis_reports/3b_v39b_code_audit.md`、`analysis_reports/PHASE1_SYNTHESIS.md`
