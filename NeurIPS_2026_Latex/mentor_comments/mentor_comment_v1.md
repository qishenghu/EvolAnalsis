整体判断：**这版已经具备一篇可投 NeurIPS 初稿的骨架和主叙事**。Abstract、Introduction、Related Work、Method 的大方向是对的，而且“correct teacher replay before exploiting it”这条主线非常好。现在最需要改的是：**降低过强 claim、修正 DR3 表述中的潜在理论/实现不一致、处理未完成 ablation/TBD 对可信度的伤害、增强 Figure/Table 的专业感和审稿防守性**。

下面我按“优先级”给反馈。

---

# 0. 总体评分

如果这是最终提交前 48 小时的版本，我会给：

**叙事：8/10**
**方法呈现：7/10**
**实验可信度：5.5/10，目前最大风险**
**写作成熟度：7.5/10**
**NeurIPS review safety：6/10**

强项是：故事清楚、claim 有记忆点、Introduction 很顺、Table 1 很有冲击力、baseline separation ablation 很强。

最大风险是：**有几处 reviewer 会抓住的“硬伤型问题”**，尤其是 DR3 density-ratio 方向、自称 “unbiased/correctly-weighted”、Table 2 TBD、以及 appendix 中 hyperparameter sensitivity 其实显示较强脆弱性，却被描述成 contiguous high-performing region。

---

# 1. 最重要问题：DR3 的理论表述有潜在方向和 claim 风险

这是我认为必须优先修的地方。

你们 §3.5 写道：如果 teacher likelihood 可得，原则修正是乘以 $\pi_\theta(a|s)/\pi_\beta(a|s)$；随后用 discriminator label on-policy=1, teacher=0，得到 $\hat w=D/(1-D)\to \pi_\theta/\pi_\beta$；再在 teacher sample 上使用 $\hat w\rho_t$，并说这 becomes correctly-weighted off-policy policy-gradient estimator 。

这套数学在“形式方向”上是自洽的，但 reviewer 可能会质疑两个点：

1. **Off-policy PG 通常从 teacher distribution 估计 student objective 时，确实会出现目标/行为 ratio，但你们的 teacher samples 是 filtered successful cache，不是完整 $\pi_\beta$ 分布。**
   因此 $\pi_\theta/\pi_\beta$ 不是完整修正，因为采样分布实际更像 $\pi_\beta(\tau \mid R=1)$，不是 $\pi_\beta(\tau)$。如果你们声称 “correctly-weighted off-policy policy-gradient estimator”，这会被攻击。

2. **你们的 self-attenuation 解释有点怪。**
   文中说 teacher action 在 $\pi_\beta$ 下密度高于 $\pi_\theta$，所以 $\hat w$ 在 teacher samples 上小于 1，并随 discriminator confident 而下降；Figure 2 说从 1.00 降到 0.67 。这与之前你们 plan 里的 “$\hat w\to1$ as student catches up” 叙事不同。现在 draft 改成了“discriminator stronger → w decreases”，但这可能被 reviewer 理解成：**这不是 student improves 的 self-attenuation，而是 discriminator learns to downweight teacher**。二者不完全一样。

建议修改：

把强 claim：

> “eliminating Bias 2”
> “correctly-weighted off-policy policy-gradient estimator”
> “unbiased only when discriminator is well-calibrated”

改弱成：

> “mitigating Bias 2”
> “a density-ratio-corrected surrogate for teacher replay”
> “consistent with the standard density-ratio form under the idealized unfiltered-teacher assumption”

建议 §3.5 加一段 caveat：

```tex
Because our teacher cache is filtered to successful trajectories, the estimated ratio should be interpreted as a correction for the empirical replay distribution rather than an exact likelihood ratio for the original teacher policy. We therefore use DR3 as a bias-mitigating replay weight, not as a claim of exact off-policy evaluation.
```

这会显著降低理论攻击面。

另外，Abstract 里 “density-ratio correction” 可以保留，但不要写 “corrects these biases” 太绝对。现在 abstract 说 DUET “first corrects these biases through baseline separation and density-ratio correction” 。建议改成：

> “mitigates these biases through baseline separation and discriminator-based density-ratio weighting”

Baseline separation 可以说 corrects；DR3 建议说 mitigates。

---

# 2. 第二大问题：Table 2 有 TBD，不能提交

Table 2 当前大量 TBD，而且正文还讨论 “remaining cells will be added in camera-ready” 。这个在 NeurIPS submission 里是严重问题。Reviewer 会直接认为实验不完整，甚至觉得你们还没完成工作。

如果 deadline 前拿不到完整 ablation，我建议：

## 方案 A：把 Table 2 改成 “focused ablations” 而不是 4×4 component matrix

只报告已经完成且强的 cells：

| Variant          | 1.5B-AF | 1.5B-WS | Interpretation           |
| ---------------- | ------: | ------: | ------------------------ |
| DUET             |    47.5 |    36.0 | full                     |
| w/o baseline sep |     0.0 |     0.0 | Bias 1                   |
| w/o DR3          |    47.5 |     9.5 | Bias 2 setting-dependent |

然后正文不要说 “each of four mechanisms”。改成：

> “We first ablate the two estimator corrections, which are the central methodological claims.”

BC/SC ablation 如果没完成，放 appendix 或删掉 “role of each correction” 这种贡献说法。

## 方案 B：如果必须保留四机制表

至少不能有 TBD。可以用 “not run” 也不行。主文必须完整。NeurIPS reviewer 对 TBD 很敏感。

你们 Introduction 贡献第 3 点写 “ablations that validate the role of each correction” 还可以，因为 “correction” 是两个，不是四个；但 abstract/experiments 如果暗示每个 component 都 ablate 了，就要改。现在 Table 2 caption 说 “Each row removes a single mechanism” 且包含 TBD ，需要立即处理。

---

# 3. Baseline contamination 的数学有一个小但危险的表述问题

§3.2 里写：

> A successful on-policy rollout receives advantage $(1-\mu_g)/\sigma_g$, which can be small or even negative whenever the teacher term dominates $\mu_g$ 

如果 teacher rewards are 1 and rewards are bounded by 1，那么 $\mu_g \le 1$，successful rollout $R^o=1$ 的 advantage 不会 negative，只会 small / zero。**Near-successful rollout** 才可能 negative。

建议改成：

> “A fully successful on-policy rollout receives a reduced advantage, while near-successful student rollouts can become negative despite being much better than the current student average.”

这更精确，也更符合你们叙事：teacher baseline suppresses successful/near-successful exploration。

同样 Abstract 里 “successful on-policy student rollouts” 被 suppress 可以；不要说 punished/negative for successful unless reward can exceed? 当前 abstract 没有说 negative，还好 。

---

# 4. “No manual schedule” claim 需要更谨慎

现在全文多处强调 “requires no manually tuned schedule of teacher influence” 。但是你们的 BC $\mu(t)$ 明确有 $\mu_{\max}, \mu_{\min}, d_{\mathrm{floor}}$，而 appendix B 还显示这些 hyperparameters 非常敏感：例如 $d_{\mathrm{floor}}=0.6$ 得 36.0%，0.5 得 21.5%，0.4 得 2.0%；$\mu_{\min}$ 从 0.10 到 0.15 也从 21.5% 到 2.0% 。

所以“no manually tuned schedule”可以说，但必须限定为：

> no manually specified **time-based decay schedule**

而不是 no tuning / no schedule at all。

建议全篇统一改成：

* “no hand-designed time-based replay decay”
* “not annealed as a function of training step”
* “teacher influence is regulated by internal signals, although the mapping contains a small number of hyperparameters”

尤其 §4.2 说 DUET “despite using strictly fewer hyperparameters that depend on training step” ，这句话可以保留，但不要让 reviewer 觉得你们在掩盖 $\mu$ 的三个 knobs。

Appendix B 的最后一句 “SOTA cell sits inside a contiguous high-performing region rather than on a knife-edge”  目前不太被表格支持。表里 36.0 是孤立最优，附近 21.5、13.5、5.5、2.0 很低。建议删掉这句，改成诚实版本：

> “The sweep shows that WebShop is sensitive to the duration of the BC safety net, especially $d_{\mathrm{floor}}$. This sensitivity motivates our framing of BC as a safety mechanism rather than a theoretically calibrated quantity.”

这样反而更可信。

---

# 5. Figure 1 现在信息对，但视觉质量偏弱

Page 5 的 Figure 1 是一个架构图，内容上覆盖了 inputs、bias correction、signal channels、update；caption 也解释了四个机制的角色 。但从渲染图看，整体像自动生成的幻灯片截图，文字小、灰色多、层次不够 NeurIPS 风。

建议换成 TikZ / vector 版本。目标是“少字、大结构”：

```text
Teacher replay τβ        Student rollouts τθ
        \                    /
         \                  /
          mixed GRPO group
                 |
       Estimator corrections
       [Baseline sep] [DR3]
                 |
       Complementary signals
       [BC action] [SC state]
                 |
            DUET update
```

Figure 1 caption 现在太长，几乎像一个 subsection。建议 caption 缩短 40%。图本身表达角色，caption 只补充：

> “DUET separates teacher replay into estimator correction and signal extraction. Baseline separation and DR3 address the two biases in §3.2; BC and SC extract complementary action/state signals after correction.”

---

# 6. Figure 2 的 claim 有点弱于你们原本想讲的故事

Figure 2 当前是 self-attenuation 三 panel，$\hat w$ 从 1.00 到 0.67，BC 从 0.3 到 0.1，disc acc 到 0.99 。问题是：**33% attenuation 听起来不够强**，不太能支撑“teacher influence fades”这个大 claim。之前 plan 里提到可能是 0.50 到 0.05，但现在 draft 写 1.00 到 0.67，这个视觉冲击小很多。

建议：

1. Figure title 从 “Self-attenuation of teacher influence” 改成更保守：

   > “Internal regulation signals for teacher replay”

2. Panel A y-axis 不要叫 “Density-ratio weight” alone，而叫：

   > “Effective teacher replay weight”

3. 如果确实只有 1.00→0.67，不要说 “fade-out”；说 “down-weighting”。
   “fade-out” 会让 reviewer 期待接近 0。

4. 强调 BC 的 $\mu$ 衰减才是 action imitation fade，DR3 是 replay correction weight，不一定要归零。

---

# 7. Table 1 很好，但 caption 太长，且单 seed 要放得更规范

Table 1 是强项：4/4 best、Δ row 清楚、strongest baseline underline 的设计是对的 。

建议小改：

* Caption 缩短，不要在 caption 里写完整 interpretation。
* “single seed” 放 caption 第一行可以，但最好加 “CI in §5”。
* `SFT (no RL)` 只在 1.5B-WebShop 有一格，可能让表显得不整齐。可以移到 appendix 或 footnote reference。如果保留，标成 “reference only”。

Table 1 里 3B-WebShop OnPolicy 2.0% 有 †，但 footnote 说 “collapses without expert guidance” 。这个 † 不太必要。脚注符号最好只用于真正需要解释的 baseline reproduction。

---

# 8. Introduction 已经很强，但可以更像 NeurIPS camera-ready

Introduction 的逻辑非常顺：cold-start → teacher replay → two biases → DUET → results/contributions 。我建议做三类压缩：

## 8.1 减少重复

“teacher trajectories are not benign / not merely supervision / corrected replay plus complementary signal” 在 Introduction 和 Abstract 里出现多次。可以保留主线，但减少重复句。

## 8.2 “state-of-the-art” 要谨慎

贡献里写 “state-of-the-art performance in all evaluated settings” 。如果 baseline 是你们自己 reproduction，且 LUFFY 原 paper 3B-WS 49.5 高于你们 DUET 45.5，那么 “SOTA” 会被 reviewer 挑战。你们 Table 1 用 LUFFY reproduction 38.0，但 appendix 又承认 original 49.5 。

建议全篇把 “SOTA” 改成：

> “best under our controlled evaluation protocol”

或者：

> “best among all baselines reproduced under the same protocol”

这会安全很多。Abstract 里 “best performance in all used settings” 也建议改成：

> “best performance among reproduced baselines in all evaluated settings”

## 8.3 “weakest-agent regime” 要定义

Abstract 和 Introduction 都用 weakest-agent regime 。建议第一次出现时明确：

> “the two 1.5B settings”

否则 reviewer 不知道 weakest 是 1.5B 还是 on-policy 低的 cells。

---

# 9. Related Work 写得稳，但略保守，可以增加一类：offline-to-online / demonstration-guided RL

Related Work 当前四段：LLM agents、group-based RL/teacher guidance、experience replay/demonstrations、density-ratio、reward shaping 。总体很好，能支撑 claim。

我建议增加或微调两点：

1. **把 “demonstration-augmented RL” 和 “offline-to-online RL” 连接起来。**
   因为你们问题本质是 filtered teacher cache + online RL。可以引用 AWAC、IQL、CQL 或 DAPG 这类，但不要过度展开。重点说：classic offline-to-online assumes explicit dataset distribution / value learning；DUET studies group-normalized LLM-agent RL with teacher-mixed replay.

2. **LUFFY/CHORD 描述要中性。**
   现在写 “leave open estimator question” 很好 。不要写 reviewer-criticized/manual schedule 太攻击。

---

# 10. Experiments 目前最大可信度问题：single seed + cross-infra variance + TBD 同时出现

单独 single seed 可以通过 CI 防守；单独 cross-infra variance 可以说透明；单独 TBD 可以说初稿未完成。但三者同时出现，会让 reviewer 感到实验体系不稳。

你们 §5 对 single seed 写得诚实，CI 也不错 。但是 cross-infra variance 的描述过于详细，甚至可能吓到 reviewer：

> LUFFY 38.0 vs 3.5 despite identical code/configs/seeds 

这会引出一个危险问题：如果 baseline 可以差这么多，DUET 的结果是否也可能差？你们说 DUET 4×A100 vs L20X 只差 ±1.5pp，但这个证据只在 appendix 文字里，没有表格化。

建议加一个小 appendix table：

| Method      | 4×A100 | 4×L20X |
| ----------- | -----: | -----: |
| LUFFY 3B-WS |    3.5 |   38.0 |
| DUET 3B-WS  |   44.0 |   45.5 |

这样“DUET 更稳定”才有支撑。

同时主文 §4.5 可以压缩，不要列太多 speculative root causes。现在 “vLLM scheduler timing / FSDP precision / floating-point drift” 这些猜测  可以放 appendix，主文只说：

> We observed substantial cross-infrastructure variance for LUFFY; Appendix A reports all numbers. We use the strongest reproduction under our protocol to avoid overstating DUET's gains.

---

# 11. Appendix C 有一个明显编号错误

Appendix 里有：

* C.1 DR3 Discriminator
* C.2 DR3 Stability
* C.3 BC coefficient
* C.4 State Channel
* C.5 Teacher Baseline Separation

但 PDF text 显示 C.1 “DR3”，C.2 也叫 “DR3” 。建议改成：

* C.1 Discriminator architecture and training
* C.2 Ratio clipping and distributed synchronization
* C.3 Adaptive BC coefficient
* C.4 State-channel implementation
* C.5 Baseline separation implementation

---

# 12. Appendix C.4 和正文 SC 公式不一致，需要修

正文 §3.7 写的是 potential-based step shaping：

[
r'(s_t,a_t,s_{t+1}) = r + \beta(\Phi(s_{t+1})-\Phi(s_t))
]



但 Appendix C.4 写：

> Trajectory progress $P(\tau)=\Phi(s_T)$ is added as a multiplicative bonus $\beta P(\tau)$ with $\beta=0.20$ 

这看起来不是同一个机制。一个是 step-wise potential difference；一个是 trajectory-level terminal potential bonus / multiplicative bonus。这个必须统一。否则 reward shaping 的 Ng et al. policy invariance claim 可能不成立。

如果实现真的是 trajectory-level $\beta P(\tau)$，那正文不能强 claim potential-based invariance。你们需要确认实现。

* 如果实现是 step-wise difference：改 Appendix C.4。
* 如果实现是 terminal progress bonus：正文改成 “progress-based reward shaping inspired by potential shaping”，不要说严格 policy-invariant。
* 如果 WebShop/ALFWorld 两者不同：正文要说明 “we use the potential-difference form when stepwise states are available; implementation details vary by environment”。

这是硬伤级别，优先级很高。

---

# 13. Checklist 还全是 TODO，最终必须清掉

PDF 后面 NeurIPS checklist 仍然是模板，所有 answer 都是 TODO 。这最终提交会直接出问题。建议单独留一小时填写。

尤其以下问题要小心：

* Q2 Limitations：Yes，指 §5。
* Q3 Theory assumptions/proofs：不要说 Yes unless 有 formal proof。可以说 “N/A; we provide estimator motivations but no new theorem.”
* Q7 Statistical significance：最好说 “Yes, binomial CIs as proxy; no multi-seed due compute.” 或 “Partial” 但 checklist 只能 Yes/No/N/A。可以 Yes with caveat。
* Q10 Broader impacts：需要加一小段 broader impact，哪怕很短。
* Q16 LLM usage：如果 Claude/ChatGPT 参与写作，按政策通常写 “LLMs were used for writing assistance, not core method or experiments” 是否需要视 NeurIPS policy；但 checklist 问 core method development，可能 N/A/No。你们自己决定。

---

# 14. 建议的最终主文结构调整

我建议最终 9 页主文这样排：

1. Abstract
2. Introduction：保留，压缩 10%
3. Related Work：保留，稍微加 offline-to-online 一句
4. Method：保留，但修 DR3 和 SC consistency
5. Experiments：

   * Setup
   * Main results
   * **Focused estimator ablations**，除非 Table 2 完整
   * Internal regulation dynamics
   * Reproducibility note，压缩到 1 段
6. Discussion / Limitations：保留，压缩 speculative 部分
7. Conclusion：保留，但缩短 20%

---

# 15. 我会立刻修改的具体句子

## Abstract

原句：

> “best performance in all used settings”

改：

> “best performance among reproduced baselines in all evaluated settings”

原句：

> “first corrects these biases through baseline separation and density-ratio correction”

改：

> “first removes baseline contamination and mitigates teacher-student mismatch through baseline separation and discriminator-based density-ratio weighting”

## §3.2

原句：

> “successful on-policy rollout ... can be small or even negative”

改：

> “a fully successful rollout receives a reduced advantage, while near-successful rollouts can even become negative despite exceeding the student’s current average.”

## §3.5

原句：

> “becomes the correctly-weighted off-policy policy-gradient estimator”

改：

> “matches the standard importance-weighted form under the idealized assumption that the replay cache is sampled from the unfiltered teacher distribution.”

加 caveat：

> “Since our teacher cache is success-filtered, DR3 should be interpreted as correcting the empirical replay distribution rather than recovering exact teacher-policy likelihoods.”

## §4.3

如果 TBD 还在，删：

> “Remaining cells ... will be added in the camera-ready.”

改成 focused ablation framing。

## Appendix B

删：

> “contiguous high-performing region rather than on a knife-edge”

改成：

> “the schedule is sensitive on 1.5B-WebShop, especially to $d_{\mathrm{floor}}$, which is consistent with the role of BC as a cold-start safety net.”

---

# 16. 最终建议：论文主线保留，但把 “principled” 从强理论 claim 改成 “estimator-aware design”

现在版本最容易被攻击的是：你们用了 “principled / correct / unbiased / policy-invariant / no schedule” 这组强词，但实现和实验里有 filtering、heuristic mappings、single seed、sensitivity、TBD。这些强词要稍微降温。

我建议全文把 “principled” 的含义限定为：

> We identify estimator-level failure modes and design mechanisms that correspond to known principles: subgroup-normalized advantages, density-ratio weighting, and potential-based shaping.

而不是：

> We fully correct teacher replay and obtain an unbiased estimator.

这样 reviewer 更难攻击，也不削弱故事。

---

# 一句话结论

这版 draft 的主线已经对了：**teacher replay 不是 benign supervision，DUET 的核心是 correct before exploit**。接下来最重要的是做“审稿安全化”：**修 DR3 ratio/claim，统一 SC 公式与实现，删掉 TBD，弱化 no-schedule 和 SOTA 的绝对表述，诚实呈现 hyperparameter sensitivity**。完成这些后，这篇会从“有潜力但容易被抓硬伤”变成“叙事清楚、风险可控、结果有冲击力”的投稿版本。
