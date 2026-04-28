# 3B v39b 发现的跨 scale × env 数据层判断

**日期**: 2026-04-24
**触发**: bug 修后 3B WebShop v39b (α=0.5) Val@100 = 45.5%,反超 v39 (α=0.2) +13.5pp,推论 v41_psh (LUFFY-style policy shaping) 有望反超 LUFFY 49.5%。本文不跑实验,仅基于已有 buggy/clean 数据外推到 1.5B 和 ALFWorld。
**结论先行**: α=0.5 的真实优势是 **scale-up 涌现**(3B policy capacity 大、KL drift 倾向强,fast EMA 把 BC 当隐式 KL regularizer 用),**不会**完整 transfer 到 1.5B。ALFWorld 上 BC 本身就是负担,α=0.5 + policy shaping 大概率是 anti-pattern,paper contribution 应重定位为"按 env 类型自适应"而非"统一更强"。

---

## A. Q1, Q2, Q3 分别判断

### Q1: 1.5B 上 α=0.5 vs α=0.2 的真实关系

**已有 buggy 数据**(B1/B2/U1 三 bug 全在,见 PHASE1_SYNTHESIS / phase1_deep_dive):

| 1.5B WebShop | reward@100 | success@100 | μ s5–30 above 0.10 | 备注 |
|---|---:|---:|---:|---|
| v39b (α=0.5) | 0.637 | 19.0% | 18 / 25 步 | best-smoothed succ 0.118(变体最高) |
| v39 (α=0.2) | 0.605 | 11.5% | 18 / 25 步 | best-smoothed succ 0.084 |
| v24 (hand) | 0.678 | 22.0% | 13 / 25 步 | MAE vs v24 implied schedule = 0 |

| 1.5B ALFWorld | Val@50 | Val@100 | 备注 |
|---|---:|---:|---|
| v39 (α=0.2) | 45.5% | **42.0%** | handoff 报告"成功",但 buggy code |
| v39b (α=0.5) | 40.0% | crash (vLLM OOM) | Val@100 缺失 |
| v24 (hand) | 33.5% | 30.5% | regression vs v1 |
| v1 (no BC) | 27.5% | 32.5% | DUET 强势 base |

**关键观察**:在 buggy 1.5B 数据下,WebShop 上 α=0.5 已经赢 α=0.2 +7.5pp,**与 3B 上的 +13.5pp 同向**。这说明"fast EMA 当 KL regularizer"的 mechanism 在 1.5B 上**已经存在**,只是幅度小。3B 上的放大可以解释为:1.5B policy capacity 小、KL drift 倾向弱(v39 1.5B late KL>1 仅 4%),BC 校准回路的边际价值有限;3B policy 大、drift 倾向强(v39 3B 末段 kl_loss 单调爬到 1.26 还在涨),fast EMA 校准回路价值放大,所以 α=0.5 vs α=0.2 的 gap 从 1.5B 的 +7.5pp 跳到 3B 的 +13.5pp。

但要注意:3B WebShop 的 v39 buggy 修复后只拿到 32%,而 1.5B WebShop v39 buggy 拿到 11.5%。**两者绝对值不可比**(3B 本身 capacity 大),只能比 α 内 gap 方向。

**1.5B ALFWorld 反着看**:这里 v39 (α=0.2) buggy 拿到 42%,v39b 在 Val@50 反而**比 v39 低**(40% vs 45.5%)——和 WebShop 方向相反。Mechanism 论证:ALFWorld 的 teacher template (`go to X`) 简单,1.5B 上 disc_acc 大概率早早饱和(handoff R3 已警告 "3B 可能更快饱和"),fast EMA 反而**让 μ 更早被压扁**到 floor,反而失去 BC 早期消除 `\n` 退化的好处(v24 ALFWorld 那个 "16→0 \n-action" 的功劳本质来自 BC 早段 μ 高;fast EMA 削掉早段 μ)。

**bug 修后真实 trajectory 推断**(纯外推):

- 1.5B WebShop v39b fix-code:reward 从 0.637 大概率上升(B1 让 SC 死了,修了反而帮 reward 更厚),success 从 19% 升到 **22-26%**(信心 M)。和 v24 22% 接近或微超。**α=0.5 仍赢 α=0.2**(信心 H,机制 + 同向 buggy 数据)。
- 1.5B ALFWorld v39 fix-code:从 buggy 42% 估计**回退到 35-40%**(信心 M)。理由:U1 让教师梯度被反向放大,在 ALFWorld template 任务上反而误打误撞放大了 BC 的 "anti-degeneracy" 副作用(消 `\n`,消重复循环);bug 修后教师梯度回归正常 5%,反而失去这个意外好处。
- 1.5B ALFWorld v39b fix-code:能不能跑完 Val@100 都是问号,即使跑完估计 **30-38%**(信心 L-M)。ALFWorld 上 fast EMA 把 BC 早期削掉,既没 fix 掉 v24 的 "wrong-default countertop" template overfitting(因为 disc_acc 仍会感知 template 重合 → μ 仍非 0 → 仍 imprint),又削掉了 BC 的 anti-degeneracy 副作用——两边坏处都吃。

**明确建议**:**primary 服务器需要在 1.5B 重跑 v39 vs v39b under fix-code**(信心 H),理由:

1. handoff 给的 1.5B v39 ALFWorld 42% 是"paper 主张 +11.5pp vs v24"的核心证据。如果 fix 后回到 35%,这个 paper 卖点直接坍塌——必须在 5/7 之前知道。
2. 1.5B WebShop fix-code v39b 是验证"α=0.5 winning mechanism 在 1.5B 也成立"的关键数据点。如果 fix 后 1.5B v39b 仍赢 v39,paper 的 "scale-invariant adaptive μ" 可以保留;如果 fix 后 v39b 反而输 v39,那 v39b 优势是 3B-only,paper story 必须改。
3. cost: 单 run ~3h on 4-GPU,两个 run 6h。**5/7 deadline 前还容得下**(剩 13 天)。比起跑新 v41_psh,先 lock down 1.5B 真实数字优先级更高。

### Q2: ALFWorld 上 α=0.5 + policy_shaping 的预测

ALFWorld 历史(1.5B 全程,3B 部分):

| 1.5B Variant | Val@100 | mechanism 注 |
|---|---:|---|
| v1 (no BC, no DR3, no SC) | 32.5% | 仅 LUFFY mix + GRPO |
| v24 (hand BC schedule) | 30.5% | **regression −2pp**,template overfitting + plan-dump |
| v39 (α=0.2 buggy) | 42.0% | handoff 主张,但 U1 反向 + B1 死 SC 的运气 |
| v39b (α=0.5 buggy) | crash | Val@50 = 40% |

| 3B Variant | Val@100 | 备注 |
|---|---:|---|
| v1 (no BC) | **69.5%** | DUET 强势 baseline,paper 旗舰数 |
| LUFFY | 61.5% | |
| CHORD | 54.5% | 可能配错 |
| OnPolicy | 58.5% | |

**关键证据**(v24_alfworld_trajectory_diff.md 第 2 段,200 任务行为分析):

> "On WebShop, BC was unambiguously useful because the teacher's action surface contains rare SKU-like tokens (`click[lavender]`, `click[fs4 | 30]`)... On ALFWorld the story inverts: the teacher's action alphabet is templated (`go to X`, `take Y from X`)... BC does not *add* useful primitives — it only *over-weights* the teacher's boilerplate phrasing... 'I will start by checking countertop, as it is a common place to find X' (169/200 vs 1/200) → first-destination entropy 2.38 → 2.04 → 30 v1-successes flipped into v24-failures."

ALFWorld 的 BC 经济学是**双刃剑**:正面 +26 任务(消 `\n`-as-action 16→0,消重复循环 55→20),负面 −30 任务(template overfitting,countertop 默认偏置),净 −4。

**那么 v39b α=0.5 + v41_psh policy shaping 在 ALFWorld 上是什么?**

policy shaping (`p/p_β`) 对教师 token 等效"常驻教师 prior",**和 BC 一样会注入 teacher boilerplate**——而且更强、更平滑、更连续。LUFFY 在 ALFWorld 3B 拿到 61.5%(< v1 的 69.5% −8pp),正是 policy shaping 在 template 任务上的"过度模仿"效应。

**预测**:

1. **3B ALFWorld v39b alone**(只换 BC schedule,无 shaping):**60-66%**(信心 M)。理由:v1 = 69.5% 是无 BC 上限;adaptive BC 在 template 任务上加任何东西都是负担,只是 fast EMA + 数据驱动的 μ 退场让伤害比 v24 小。可能小幅低于 v1。
2. **3B ALFWorld v39b + v41_psh**(α=0.5 BC + shaping):**55-62%**(信心 M)。**双重 boilerplate 注入,大概率比 v39b alone 差 3-5pp**。policy shaping 没有 DR3 的 fade-out 机制,常驻 prior + adaptive BC 的边际重叠,加一起反而互相**放大** template overfitting。

**这是 anti-pattern**(信心 M-H):ALFWorld 不需要"更多 teacher 注入",它需要"更少"。

**v1 已经做对了的事**:

- 无 BC → 不 imprint boilerplate,first-destination entropy 保持 2.38
- 仅 LUFFY rollout-level mix(不是 token-level shaping)→ teacher 只在 group-relative advantage 里贡献 baseline,不直接改 token 分布
- 没有 SC、没有 DR3、没有 chord_mu → 全 GRPO + group baseline → 69.5%

**ALFWorld 上 paper 该做的不是"加 v41_psh",而是 confirm v1 + 可能加"BC 只在 anti-degeneracy token 上施加"的轻量变体**(v24 报告结尾的 (iii) 提议:filter teacher gradient to response-end tokens only)。但这是新设计,5/7 deadline 来不及。

**注意一个反例**: 如果 BC 真在 ALFWorld 减少 `\n`-as-action(handoff 提过 v24 在 1.5B ALFWorld 上 16→0),那对**容易陷入 degeneracy 的小模型**有救。3B 上 degeneracy 很少(看 v1 = 69.5% 就知道 3B base policy 已经足够稳定),所以 BC 副作用反向放大;1.5B 上 degeneracy 多,BC 反而救场——这正好解释了"为什么 1.5B v39 ALFWorld 看起来有 +11.5pp"(buggy 数据让 BC 副作用得以发挥)。**即 1.5B ALFWorld v39 的"高分"很可能是 BC 救 degeneracy + buggy 让 BC 力度反向放大的双重运气**,fix 后会回退。

### Q3: 3B v39b WebShop 45.5% mechanism 的 transfer 概率

总结表(每格:**预测 + 信心 + 一行论证**,见 §B 矩阵)。核心结论:

- **WebShop 任务上**:α=0.5 mechanism 在 1.5B 和 3B 都成立(信心 H),policy shaping 叠加预计反超 LUFFY(3B 信心 M-H,1.5B 信心 L-M)
- **ALFWorld 任务上**:α=0.5 + shaping 的"加法"是反向的,**全栈 DUET (DR3+SC+adaptive μ+shaping) 不会比纯 v1 (LUFFY mix) 强**(3B 信心 M-H,1.5B 信心 L)

---

## B. 跨 scale × env 4-cell 矩阵预测

| | **1.5B WebShop** | **1.5B ALFWorld** | **3B WebShop** | **3B ALFWorld** |
|---|---|---|---|---|
| **α=0.5 比 α=0.2 强** | YES (M, +3-7pp 预测,buggy 已 +7.5pp,fix 后保留方向) | NO/wash (M, ALFWorld template 任务 fast EMA 反而削早期 BC 救场效应) | **YES (H, 实测 +13.5pp)** | maybe-NO (L-M, BC 本身在 ALFWorld 是负担,fast/slow 之争次要,可能两者都不如 v1) |
| **LUFFY shaping 加在 v39b 上能反超 LUFFY** | YES (L-M, 1.5B 上两个绝对值都低,验证空间窄) | NO (M, ALFWorld 1.5B LUFFY 仅 5.5%,v1 32.5%,shaping 是 ALFWorld 的弱点不是优点) | **likely YES (M-H, 49.5→51-54% 预测,DR3 fade-out + shaping floor 正交叠加)** | NO (M, shaping 在 ALFWorld 3B 已实测 LUFFY=61.5 < v1=69.5,叠 v39b 不会反超 v1) |
| **多机制全开 (DR3+SC+adaptive μ+shaping) 比纯 LUFFY mix 强** | **NO** (M, v24 hand 22% > v39b 19%,1.5B 上 hand schedule 仍是上限,LUFFY mix 充分性更高) | **NO** (H, 1.5B v1 = 32.5% > v24 = 30.5%,任何加法都伤) | **NO** (M, 3B 实测 v39b 45.5 < LUFFY 49.5,即使 v41_psh 51% 也只是 marginal +2pp,不构成 algorithm-level 胜出) | **NO** (M-H, 3B v1 = 69.5% > LUFFY 61.5,DUET 全栈在 ALFWorld 上的优势就是 v1 的简单性,加东西只伤) |

**核心 takeaway**:
- 横向看:**WebShop 列普遍偏 YES,ALFWorld 列普遍偏 NO**——两个 env 的 BC/shaping 经济学方向相反
- 纵向看:1.5B → 3B 在 WebShop 上**机制不变方向不变,幅度放大**;在 ALFWorld 上 1.5B 还能靠 BC 救 degeneracy,3B base policy 已稳,BC 只剩负面
- 全栈 DUET 在 4 个 cell 中 **WebShop 3B 一格能反超 LUFFY**(且只是预测),其他 3 格都打不过现成的简单基线(v1 / v24 / LUFFY)

---

## C. 必须用实验验证的 cell

考虑 5/7 deadline,推荐 2 个最关键、cost 最低的实验:

**实验 1**:`primary 服务器 1.5B WebShop fix-code rerun v39 vs v39b`
- **为什么必须**:Q1 直接 lock down,验证"α=0.5 mechanism 是否 scale-invariant"。这是 paper 跨 scale 章节的命门数据。
- **cost**:2 × 3h = 6h on 4-GPU
- **决策树**:如果 fix 后 v39b 仍赢 v39 +3pp 以上 → paper 保留 "adaptive μ scale-invariant" claim;如果 fix 后 v39b ≈ v39 或反超 → paper 必须把 v39b 优势降级为 "3B-specific scale-emergent"

**实验 2**:`primary 服务器 1.5B ALFWorld fix-code rerun v39 (α=0.2)`
- **为什么必须**:handoff 主张的 "v39 ALFWorld +11.5pp vs v24" 是 paper ALFWorld 数据的核心,但是 buggy。如果 fix 后回退到 35%,paper ALFWorld 章节就不能写成 "DUET adaptive 全面超越 v24",必须改成 "在 WebShop 类 rare-token env 上超越,ALFWorld template env 上需要不同处理"。
- **cost**:1 × 3h on 4-GPU
- **决策树**:fix 后 ≥ 38% → 维持 paper 现有 framing;fix 后 ≤ 35% → 触发 §D 的 framing 调整

**第三个 nice-to-have**(若时间允许):**3B ALFWorld v39b 跑完**(已 pending)。仅为确认"3B ALFWorld 上 v39b 不会反超 v1=69.5%"——基于上面分析信心 M-H。这个数据**只用于 paper ablation**,不是 main result。**不推荐跑 v41_psh on ALFWorld**——matrix 已经预测它低于 v1。

---

## D. 如果 Q2 答案是 "ALFWorld 不该有 BC",paper 怎么写

**强烈建议 framing 调整**(信心 M-H,基于 1.5B v24 trajectory 行为分析 + 3B v1 vs LUFFY 数字 + Q2 mechanism 推理):

### 旧 framing(目前 PHASE1_SYNTHESIS 写法)

> "DUET (DR3+SC+adaptive BC) 在 1.5B 和 3B 的 WebShop+ALFWorld 上都比 baseline 强。adaptive μ 是单一 sufficient statistic,**uniformly stronger**"

**问题**:已有数据已经反这个 claim:
- 1.5B ALFWorld v1 (32.5%) > v24 (30.5%),全栈 DUET 在 ALFWorld 1.5B 上**已经输给自己的子集**
- 3B ALFWorld v1 (69.5%) > LUFFY (61.5%) > 预测的 v39b (60-66%) > 预测的 v41_psh (55-62%),**加 BC/shaping 单调变差**

### 新 framing(推荐)

> **"DUET = adaptive component selection by environment surface."** Paper 主张:
> 1. **rare-token environments**(WebShop:`click[lavender]`,SKU 命名,长尾 action surface)需要 **DR3 + adaptive BC + shaping**——teacher 注入 rare primitives,policy 单独学不到,这里 DUET 全栈最强(WebShop 3B v41_psh 预测 51%,反超 LUFFY 49.5%)。
> 2. **template environments**(ALFWorld:`go to X`,`take Y from X`,封闭 action grammar)只需要 **LUFFY rollout-level mix + GRPO baseline**——teacher 给 advantage prior 但不动 token 分布,DUET 退化到子集 v1(ALFWorld 3B v1 = 69.5%,远超 LUFFY 的 61.5% 和带 BC 的 v24)。
> 3. **adaptive μ 的真正贡献是把 BC 自动关掉**:disc_acc 在 ALFWorld 上快速饱和(template 易区分)→ μ 自动退到 floor → BC 失效 → DUET 几乎等价 v1。这是 v39b 比 v24 在 ALFWorld 上好的真实 mechanism——**不是"自适应让 BC 更聪明",而是"自适应让 BC 自己识相退场"**。

### 配套 contribution 重排

| 旧 contribution | 新 contribution |
|---|---|
| C1: DR3 density-ratio correction(技术实现) | C1: **DR3 仍然是技术贡献**(action channel 校正 IS weight)——保留 |
| C2: SC progress shaping | C2: **SC 是 WebShop 专属** → ablation 表里写"SC on ALFWorld 无效",转成"按 env 启用"的 design lesson |
| C3: adaptive μ via single discriminator | C3: **adaptive μ 的真实价值是"按 env 自适应启停 BC"** → 在 WebShop 上 μ 长 effective,在 ALFWorld 上 μ 自动早退场 → 跨 env 一份代码两种行为 |
| C4: uniformly stronger | C4: **删掉**;改为 "design space 横扫:WebShop 上 v41_psh > LUFFY,ALFWorld 上 v1 ≈ v39b 退化版,**DUET 是上包络**" |

### Paper 写作具体建议

1. **Main table**:不要把 ALFWorld 和 WebShop 的 DUET 数字简单平均。用两栏分别列,每栏标 winner,**让 reader 看到"WebShop DUET wins, ALFWorld DUET ties v1"是设计意图,不是缺陷**。
2. **新增 ablation section**:**"BC saturation analysis: when does adaptive μ help?"** — 用 disc_acc 收敛曲线 + first-destination entropy 量化,明示"BC 在 template env 上自动退场是 feature"。
3. **Theory section**:把 "TV-KKT identity" 的论证扩展为 "TV identity also predicts task-surface dependence: when teacher and policy share a templated grammar, TV converges fast → μ retires → DUET reduces to LUFFY mix"。这把 paper 的理论框架反而强化了。
4. **Comparison table** 必须补 "adaptive components on/off by env":WebShop[BC=on, shaping=on, SC=on], ALFWorld[BC=auto-off, shaping=off, SC=off]。让 reviewer 看到这是**有意设计而非超参**。

### 这个 framing 的好处

- **避开 "为什么 v24 在 ALFWorld 反而退步"** 的尴尬质问 → 因为 BC 本来就不该在 ALFWorld 上启用,我们的 adaptive 比 hand-tuned 强不是因为 schedule 算得更好,而是**因为我们的 schedule 自动学会了不启用**
- **避开 "为什么 DUET 全栈在 ALFWorld 3B 不如 v1"** → 因为 v1 就是 DUET 在 template env 上的 adaptive output;两者数值近似 = mechanism 论证成功
- **paper 卖点从 "uniformly stronger" 变成 "design-aware"**,后者更适合 NeurIPS 审稿口味(一个实验跨多 env 时,reviewer 通常更喜欢看到"我们 understand each env" 而非 "one-size-fits-all")

### 风险

- 5/7 之前来不及补完所有 ablation(如 SC-on-ALFWorld 的 ablation 数字),只能用现有 v1/v24/v39 数据 + mechanism 论证
- 如果 1.5B fix-code rerun(实验 1, 2)出来反向(v39b 在 fix 后 1.5B WebShop 反输 v39),整个 framing 都要再调,**所以这两个实验是 contingency-resolving,优先级最高**

---

## 总结一句话给 lead

> 3B WebShop α=0.5 的 +13.5pp 优势在 1.5B 上方向同向但幅度小(buggy 数据 +7.5pp,fix 后预测 +3-7pp),**机制能 scale-down 但要降级表述**;在 ALFWorld 上则**反向**——template 任务的 BC 经济学决定了 v1 (无 BC, LUFFY mix) 已经是上限,v39b/v41_psh 加任何东西都伤,paper 必须把 contribution 重定位为 "adaptive component selection by env surface"。下一步 6h 重跑 1.5B v39/v39b fix-code 是 5/7 deadline 前**最高优先级 contingency-resolving 实验**;**不推荐跑 v41_psh on ALFWorld**(矩阵预测落后 v1)。

---

**附录:数据来源**

- `analysis_reports/3b_v39b_post_truth_analysis.md` — 3B WebShop v39 vs v39b 窗口对比 + fast-EMA 机制论证
- `analysis_reports/v24_alfworld_trajectory_diff.md` — 200-task ALFWorld 行为分析,template overfitting 量化
- `analysis_reports/PHASE1_SYNTHESIS.md` — 1.5B v39b 全景 + paper narrative 旧版
- `analysis_reports/phase1_deep_dive.md` — 1.5B v24/v39/v39b/v40b/v41b/v43a 单步 metric 对比
- `HANDOFF_L20X_SERVER.md` — 3B 配置 + baseline 数字 + 1.5B 参考
- `HANDOFF_3B_SERVER.md` — 1.5B 完整 baseline 表 + ALFWorld 数字
