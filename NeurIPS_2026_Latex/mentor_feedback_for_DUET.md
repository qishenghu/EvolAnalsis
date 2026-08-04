我的判断：**不要把主叙事压在“我们大幅改进 LUFFY”上**。LUFFY 不够大众，审稿人不会天然关心“比 LUFFY 好多少”。更稳的主叙事是：

> **Teacher experience replay is the natural cold-start fix for LLM-agent RL, but naïvely putting teacher rollouts into GRPO groups introduces two hidden biases. DUET is the principled replay framework that diagnoses and corrects those biases, then extracts teacher signal through action and state channels.**

也就是说，LUFFY/CHORD 是“已有 teacher replay 实例”，不是故事主角。主角应该是 **teacher replay in GRPO-style LLM-agent RL**。

---

## 1. 叙事定位：从 “Fix LUFFY” 改成 “Principled Teacher Replay”

你们现在 plan 里写的是 “principled fix to LUFFY-style experience replay” 。我建议在论文正文里进一步弱化 “LUFFY-style” 这个词，改成：

> **principled teacher experience replay for GRPO-style LLM-agent RL**

原因是：

第一，LUFFY 不像 GRPO、PPO、DPO 那样是 reviewer 立刻有 schema 的对象。标题和 abstract 里如果隐含“我们修 LUFFY”，有些 reviewer 会问：“为什么我要关心 LUFFY？”

第二，你们真正强的贡献不是“改了某个 baseline”，而是**抽象出了 teacher replay 的两个系统性 failure modes**：baseline contamination 和 off-policy mismatch。这个诊断是 paper 的 intellectual core，plan 里也明确说 “diagnosis first” 是 NeurIPS bar 的关键 。

第三，LUFFY/CHORD 仍然要出现，但位置应该是：**它们是 teacher replay 的代表性实例，说明这个问题已经被大家用起来了，但还没有被原则化处理。**

我建议全篇的命名层级是：

* 大类：**teacher experience replay / teacher-mixed replay**
* 具体已有方法：LUFFY, CHORD
* 你们的方法：DUET
* 你们的贡献：diagnosis + corrections + two-channel utilization

正文里可以写：

> Existing methods such as LUFFY and CHORD instantiate this idea by mixing successful teacher trajectories into student RL updates. However, they treat teacher replay as supervision rather than as off-policy data inside a group-normalized policy-gradient estimator.

这比 “LUFFY has problems, we fix LUFFY” 更容易被广泛接受。

---

## 2. 文章核心卖点应该排序为：Diagnosis > Principle > Results > Mechanisms

我建议你们严格控制 reviewer 的阅读路径：

### 第一层：一句话贡献

> DUET turns teacher replay from a heuristic cold-start trick into a corrected policy-gradient estimator with complementary action- and state-level teacher signals.

这句话比 “DUET has four modules” 更重要。

### 第二层：两个 bias

你们的两个 bias 很好，应该成为全文最反复出现的 backbone：

1. **Baseline contamination**：teacher rollout reward 高，进入 GRPO group 后污染 group baseline，导致 student successful exploration 被压低甚至惩罚。
2. **Off-policy mismatch**：teacher trajectory 来自 $\pi_\beta$，GRPO 的 $\pi_\theta / \pi_{\text{old}}$ 只修正 update-step mismatch，不修正 teacher-student mismatch。

这两个 bias 已经在 narrative 和 plan 里写得很清楚 。我的建议是：**Introduction 第三段、Method §3.2、Ablation §4.3、Discussion 都要用同一套词，不要换说法。**

### 第三层：四个机制，但不要平均用力

四个机制的叙事权重应该不同：

| 机制                  | 叙事角色                                         | 篇幅权重 |
| ------------------- | -------------------------------------------- | ---: |
| Baseline separation | 最强 diagnosis-to-fix，支撑 Bias 1                |    高 |
| DR3                 | 最 principled，支撑 Bias 2                       |   最高 |
| BC                  | cold-start safety net，不要装成理论贡献               |    中 |
| SC                  | complementary state-channel signal，靠 Ng99 兜住 |   中低 |

BC 的 framing 你们现在做对了：不要说 JSD-driven，因为 disc_acc 单调上升会被 reviewer 抓住；要说它是 DR3 稳定前的 cold-start safety net 。

---

## 3. Introduction 应该非常克制：5 段足够

我建议 Introduction 保持你们 plan 里的 5 段结构，但每段的“任务”要非常明确 。

### Paragraph 1：问题，不要从 LUFFY 开始

开头应该从 **weak LLM agents + sparse reward + on-policy RL cold-start** 开始。第一段结尾给出你们最抓人的数字：

> On-policy GRPO achieves only 1.0% / 0.5% success on 1.5B ALFWorld / WebShop.

这比 “LUFFY/CHORD exist” 更能抓 reviewer。

### Paragraph 2：teacher replay 是自然解，但现有做法 naïve

这里引出 LUFFY/CHORD：

> A natural remedy is to replay successful teacher trajectories. Existing methods such as LUFFY and CHORD instantiate this idea by mixing teacher data into student RL batches.

然后马上转折：

> However, teacher replay is not on-policy data. In GRPO-style updates, this distinction matters.

### Paragraph 3：两个 bias，一定要写得像 paper 的发现

这段是你们的核心。不要写得像 implementation bug，要写得像 estimator-level diagnosis：

> We show that teacher-mixed replay induces two systematic biases in group-normalized policy gradients.

然后分别一两句写 baseline contamination 和 off-policy mismatch。

### Paragraph 4：DUET，不要堆 acronym

DUET 介绍顺序：

1. first corrects the estimator：baseline separation + density-ratio correction
2. then extracts extra signal：action imitation + state-progress shaping
3. all teacher influence self-attenuates：no manual schedule

这里不要过早展开 DR3/BC/SC 细节。

### Paragraph 5：结果 + contributions

这里放 3 个 contribution bullets：

* diagnose two biases in teacher-mixed GRPO replay;
* propose DUET with principled corrections and two teacher-signal channels;
* show 4/4 SOTA, +13.0pp average, +17.5pp weakest regime.

主结果表里 DUET 是 4/4 setting 最好，平均超过最强 baseline 13.0pp，weak setting +17.5pp，这些是结果叙事的硬锚点 。

---

## 4. Method 结构：先 estimator，再 architecture

你们现在计划的 §3 结构基本是对的：setup → diagnosis → overview → four mechanisms → combined update 。我建议 Method 的写作重点是：**不要让它看起来像四个 trick 拼装**。

### §3.1 Problem setup

只保留必须符号：

* student policy $\pi_\theta$
* teacher policy/cache $\pi_\beta$
* on-policy group $G^o$
* teacher replay subset $G^\beta$
* GRPO advantage $\hat A = (R-\mu_g)/\sigma_g$

这里不要讲太多 environment。

### §3.2 Two biases

这节应该是 Method 的第一个高潮。建议用一个小标题：

> Why teacher replay is not benign in GRPO

然后给两个 boxed equations：

**Bias 1 equation：**
[
\mu_g = \frac{n_\beta + n_o \bar R^o}{n}
]

**Bias 2 equation：**
[
\rho_{\text{GRPO}}=\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}
\quad \text{but teacher replay requires correction for} \quad
\frac{\pi_\theta}{\pi_\beta}.
]

这两个式子要非常短。reviewer 扫一眼就懂。

### §3.3 DUET overview

建议放 Figure 1。Figure 1 不要做成流程图大杂烩，而要做成 **bias correction + signal extraction 的二层结构**：

```
Teacher replay + Student rollout
              |
              v
   Stage I: Correct the estimator
   - baseline separation
   - density-ratio correction
              |
              v
   Stage II: Extract complementary teacher signal
   - action channel: BC
   - state channel: potential shaping
              |
              v
        DUET update
```

Action/State channel 是好概念，但不要让它抢走 “bias correction” 的主线。你们 plan 里四个机制已经按 “bias correction / signal extraction” 和 “action / state” 映射了 ；Figure 1 就应该把这个二维结构可视化。

### §3.4 Baseline separation

这节要短、狠。写法：

1. 先复述 contamination；
2. 给 separated advantage；
3. 解释 teacher reward 恒定导致 teacher subgroup std 可能为 0，所以实现中 std 用 non-teacher source；
4. 结尾一句 principled justification；
5. 不要在 Method 里提前过多讲 47.5→0.0，最多一句 “validated in §4.3”。

建议句式：

> Baseline separation is not an additional reward heuristic; it restores the zero-mean property of the advantage estimator on the on-policy subgroup.

### §3.5 DR3

这是最长机制。要注意一个潜在风险：你们文档里有时说 $\hat w = \pi_\theta/\pi_\beta$，有时说 $\hat w = \pi_\beta/\pi_\theta$。这个必须统一，否则 reviewer 会直接质疑推导。

我建议在全文里统一成：

* discriminator estimates density ratio clearly；
* 明确 teacher-sample correction 用的是哪个方向；
* 如果 implementation 是通过 `old_log_prob ← log_prob.detach() − log ŵ` 间接实现，正文里用“effective ratio”解释，appendix 放实现细节。

不要让 reviewer 在这里卡住。DR3 的表述要满足三件事：

1. classifier-based density ratio 是标准技术；
2. 它修正 teacher-student off-policy mismatch；
3. as student approaches teacher / replay becomes less informative, correction attenuates teacher influence without manual schedule。

### §3.6 BC

这节一定不要过度 claim。建议标题直接叫：

> Adaptive behavior cloning as a cold-start safety net

不要叫 “principled imitation” 或 “JSD adaptive distillation”。你们自己也已经决定 μ 不应 framed as JSD-driven 。

最好的写法是：

> BC is deliberately decoupled from the policy-gradient correction. Its role is to provide dense token-level supervision before the discriminator-based ratio is reliable.

这样 reviewer 会觉得诚实。

### §3.7 SC

SC 要强调“state channel”是 complementary，不是又一个 reward hack。写作重点：

* potential function 来自 successful teacher trajectories；
* shaping 只作用 on-policy samples；
* teacher samples excluded，避免 double-counting teacher advantage；
* potential-based shaping policy-invariant。

你们 plan 里已经有 “exclude_teacher” 和 “grpo_decouple” 的防守点，建议正文至少保留前者，后者放 appendix 即可。

### 每个 subsection 的收尾

你们 plan 里说每个 subsection 要以 “principled justification” 结尾，这是非常对的 。但我建议不要每节都机械写 “Principled justification: ...”，而是自然写成最后一句：

> Thus, the correction follows from the estimator rather than from a tuned replay schedule.

---

## 5. Experiments：Table 1 是性能，Table 2 是可信度，Figure 3 是“principled”的证据

### Table 1：主结果表

主表不需要太复杂。保留 4 columns：1.5B-AF / 1.5B-WS / 3B-AF / 3B-WS。行顺序建议：

1. On-policy GRPO
2. SFT + GRPO
3. LUFFY
4. CHORD
5. DUET

我会把 SFT+GRPO 放在 LUFFY/CHORD 前后都可以，但不要让 baseline 列表显得混乱。更重要的是：**bold 规则必须清楚**。现在 table 里 SFT+GRPO 的 30.0、18.5 也 bold，DUET 又 bold，会造成 “谁是 best?” 的视觉混乱。建议：

* 每列只 bold best；
* 用 underline 标 strongest baseline；
* 最后一行写 Δ over strongest baseline。

这样 reviewer 一眼看到 DUET 的优势。

### Table 2：ablation table 是全文第二重要证据

Ablation table 不只是 component study，它要服务两个诊断：

* `-baseline_sep: 47.5 → 0.0` 证明 Bias 1 不是编出来的；
* `-DR3: 36.0 → 9.5` 证明 Bias 2 在长 horizon / WebShop 中 load-bearing。

你们 plan 里说 removing baseline separation 让 1.5B-AF 从 47.5% collapse 到 0.0%，这是非常强的 cell 。我建议在 Table 2 旁边加一句：

> The ablation is not merely additive: removing baseline separation collapses learning in the weakest ALFWorld setting, consistent with the predicted exploration-suppression failure mode.

Table 2 不要只写 “w/o X”。建议写成：

| Variant                 | Bias / Signal tested | 1.5B-AF | 1.5B-WS | ... |
| ----------------------- | -------------------- | ------: | ------: | --: |
| DUET                    | —                    |    47.5 |    36.0 | ... |
| w/o baseline separation | Bias 1               |     0.0 |     ... | ... |
| w/o DR3                 | Bias 2               |    47.5 |     9.5 | ... |
| w/o BC                  | action safety net    |     ... |     ... | ... |
| w/o SC                  | state channel        |     ... |     ... | ... |

这样 table 本身就在讲故事。

### Figure 2：training curves

Figure 2 应该选 **1.5B WebShop**，因为这里 OnPolicy 接近 0、LUFFY 低、DUET 高，视觉差距最大。三条曲线足够：

* OnPolicy GRPO
* LUFFY 或 CHORD，二选一；如果版面允许两者都放
* DUET

不要塞所有 baseline。曲线图的作用不是完整比较，而是展示 “cold-start trap 被打开”。

### Figure 3：auto-fade dynamics

这是“principled / no manual schedule”的核心证据。你们计划放 $\hat w_\tau$、$\mu(t)$、disc_acc 三个 panel，非常好 。我建议 Figure 3 的标题不要叫 “Training dynamics”，而叫：

> Self-attenuation of teacher influence

三个 panel：

A. density-ratio weight / effective teacher gradient weight
B. adaptive BC coefficient $\mu(t)$
C. discriminator accuracy or ESS

caption 里写清楚：

> No external replay-decay schedule is used; teacher influence decreases through DUET’s internal correction signals.

这个 figure 会直接防守 “你们不也是 schedule 吗？” 的 reviewer 质疑。

### Figure 1：方法图

Figure 1 放 Method 开头，风格要干净。建议不要用 DALL-E 风格图，NeurIPS 里最好是 TikZ / vector schematic。DALL-E 图很容易显得 marketing。除非时间不够，否则用 TikZ。

### Appendix figures

appendix 放：

* LUFFY 3-way reproducibility；
* w-hat histogram；
* hyperparameter sensitivity；
* maybe per-task examples / trajectories。

LUFFY reproducibility 建议放 appendix，不要主文大讲。你们现在决定 main table 用 L20X 38.0，appendix 给 49.5 / 38.0 / 3.5 的透明报告，这个决策是对的 。

---

## 6. Related Work：不要写成文献综述，要写成“我们填哪个 gap”

Related Work 控制在 0.7–0.9 页。建议四段：

### 2.1 On-policy RL for LLM agents

GRPO / PPO / ReAct-style agent RL。目的：说明 sparse reward agent RL 有 cold-start 问题。

### 2.2 Teacher replay and offline-to-online RL

这里放 LUFFY/CHORD/SFT+RL。关键句：

> These methods demonstrate the usefulness of teacher data, but do not analyze how teacher trajectories interact with group-normalized policy-gradient estimators.

### 2.3 Density-ratio correction and off-policy learning

Goodfellow / Sugiyama / off-policy correction。不要 claim DR3 estimator 新，而是 claim “bringing density-ratio correction into teacher-replayed GRPO for LLM agents”。

### 2.4 Potential-based reward shaping

Ng et al. 1999 + agent reward shaping。强调 SC 是 policy-invariant shaping，不是 learned reward model。

Related Work 里不要攻击 LUFFY/CHORD。用 neutral wording：“do not explicitly correct...”。

---

## 7. 一些具体措辞建议

### 少用

* “fix LUFFY”
* “LUFFY-style” 高频出现
* “our method solves off-policy RL”
* “principled” 但没有紧跟 principle
* “self-calibrating” 过多，容易像 marketing
* “disc_acc measures teacher-student gap”

### 多用

* “teacher-mixed replay”
* “teacher replay inside group-normalized policy gradients”
* “estimator-level bias”
* “baseline contamination”
* “teacher-student distribution mismatch”
* “correct first, then extract signal”
* “adaptive attenuation from internal signals”
* “cold-start safety net”

---

## 8. Abstract 我建议微调一点点

你们 abstract 已经不错，但我会改两点。

第一，最后 “best performance in all used settings” 建议改成 “all evaluated settings”。
第二，abstract 里 “adaptively attenuating teacher influence as the student improves” 如果你们的 $\hat w$ 实际是从 0.50 到 0.05，而理论段又说 $\hat w \to 1$，要特别小心方向不一致。abstract 可以写得更抽象：

> adaptively regulating teacher influence using internal density-ratio and discriminator signals

这样不暴露方向问题。

我会把中间一句改成：

> Together, these components let DUET use teacher data as corrected off-policy replay rather than uncalibrated supervision, while extracting complementary action- and state-level signals.

这句话比 “beyond direct replay” 更硬。

---

## 9. 风险点和防守方式

### 风险 1：DR3 在 1.5B-AF 没掉

不要回避。把它变成 nuance：

> DR3 is most important when the teacher-student gap persists over longer horizons. On ALFWorld, baseline separation and BC close the gap quickly; on WebShop, removing DR3 causes a large drop.

这比硬说 DR3 always helps 更可信。

### 风险 2：四个模块像拼装

解决方式：全文始终坚持二阶段：

1. **Correction**：baseline separation + DR3
2. **Signal extraction**：BC + SC

不要只说 “DUET has four components”。要说 “two are corrections, two are complementary signal channels”。

### 风险 3：LUFFY baseline 复现差异

主文 table footnote 写：

> LUFFY numbers are our reproduction under the same evaluation protocol; we discuss cross-infrastructure discrepancies in Appendix A.

appendix neutral 呈现 49.5 / 38.0 / 3.5，不要用 “failed to reproduce” 这种攻击性措辞。

---

## 10. 推荐最终版排版

9 页主文可以这样分配：

| Section                    |      页数 | 内容                                    |
| -------------------------- | ------: | ------------------------------------- |
| Abstract                   |       — | 已定稿，轻微润色                              |
| 1 Introduction             |     1.0 | 5 段 + contribution bullets            |
| 2 Related Work             |    0.75 | 4 短段                                  |
| 3 Method                   | 3.0–3.3 | Fig.1 + bias equations + 4 mechanisms |
| 4 Experiments              | 2.5–2.8 | Table 1, Table 2, Fig.2, Fig.3        |
| 5 Discussion / Limitations |     0.5 | single seed, variance, scope          |
| 6 Conclusion               |    0.25 | 一段                                    |
| References                 |      剩余 | 正常                                    |

图表顺序：

1. **Figure 1**: DUET schematic, Method 开头
2. **Table 1**: main results, Experiments 开头
3. **Table 2**: ablations, 紧接 Table 1
4. **Figure 2**: training curves, 放 ablation 后
5. **Figure 3**: self-attenuation dynamics, 放最后支撑 no manual schedule

如果版面紧张，Figure 2 和 Figure 3 可以合成一个 2-row figure，但我更建议保留 Figure 3，因为它直接支撑 “principled/self-attenuating” claim。

---

## 11. 我会给这篇 paper 的最终叙事口号

**最推荐：**

> Correct teacher replay before exploiting it.

展开版：

> Teacher trajectories are valuable, but in GRPO they are not harmless supervision. DUET first corrects the estimator-level biases introduced by teacher replay, then uses the same teacher data as complementary action and state signals.

这就是全文应该反复服务的主线。LUFFY 是背景和 baseline，不是故事中心。DUET 的故事中心是：**teacher replay 在 LLM-agent RL 里需要从 heuristic 变成 principled estimator design**。
