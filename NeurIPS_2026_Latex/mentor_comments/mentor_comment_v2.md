这版**明显比上一版安全很多**：Claude 基本把我上次指出的几个硬伤都处理了，包括 DR3 的 filtered-cache caveat、Table 2 从 TBD component table 改成 focused estimator ablation、SC 从严格 potential-based policy invariance 降级为 progress shaping approximation、以及 “no schedule” 改成 “no step-indexed/time-based schedule”。整体上，这版已经从“有硬伤风险的强叙事”进入了“可投、但还需 polish 和进一步降风险”的状态。

我的总体判断：

**主线：稳了。**
**理论 claim：比上一版安全很多。**
**实验呈现：主文基本可接受。**
**最大剩余风险：Abstract 仍然比正文更激进；SC/Ng99 叙事还略不一致；DR3 仍有一两句容易被攻击；Checklist 未填；Figure 1 视觉质量仍偏弱。**

---

## 1. 最大进步：主叙事现在更审稿安全

Introduction 现在很好地把 “weakest-agent regime” 定义清楚了：不是泛泛而谈，而是指 base model 弱到 pure on-policy RL 几乎找不到成功轨迹，在你们实验里就是两个 1.5B 设置 。这非常重要，因为它避免 reviewer 问 “weakest-agent regime 是什么”。

DUET 的 framing 也比之前稳很多。现在正文说它是 **estimator-aware framework**，不是一上来就说完全 principled correction；DR3 也被说成 per-sample replay weight，而不是绝对的 off-policy estimator 。这个改动是对的。

贡献列表也更安全了：现在写的是 “best strict success rate among reproduced baselines” 和 “focused ablations validate the two estimator corrections”，没有再暗示四个模块都被完整 ablate 。这比上一版显著更可信。

---

## 2. 但 Abstract 还没有完全同步正文的降温

Abstract 仍然有几处比正文更强，建议必须改。

现在 Abstract 仍写：

> “DUET ... first corrects these biases through baseline separation and density-ratio correction” 

但正文 §3.5 已经正确承认：因为 teacher cache 是 success-filtered，DR3 只能解释为 correcting empirical replay distribution，不是 exact teacher-policy likelihood，也不是 exact off-policy evaluation 。所以 abstract 里 “corrects these biases” 对 DR3 来说仍然过强。

建议改成：

```tex
Therefore, we propose DUET, an estimator-aware experience-replay framework that
removes baseline contamination through subgroup normalization, mitigates
teacher--student mismatch through discriminator-based density-ratio weighting,
and extracts complementary teacher signals through two channels...
```

Abstract 还写：

> “potential-based state-progress shaping” 

但正文 §3.7 已经承认你们实际用的是 trajectory-level progress score，是 strict potential-based shaping 的 coarse approximation，不 claim exact policy invariance 。所以 abstract 里最好不要再叫 “potential-based”。建议改成：

```tex
progress-based state shaping in the state channel
```

Abstract 里还有：

> “adaptively attenuating teacher influence as the student improves” 

正文 §3.5 已经说得更准确：$\hat w$ 的下降由 discriminator strengthening 驱动，不 claim 它 calibrated as student-teacher gap 。所以 abstract 这句也建议改：

```tex
adaptively regulating teacher influence through internal discriminator-derived signals
```

最后，Abstract 仍写 “best performance in all used settings” ，但 Introduction 和 Table 1 已经改成 “among reproduced baselines”。Abstract 必须同步：

```tex
Across ALFWorld and WebShop with two model scales, DUET achieves the best
strict success rate among reproduced baselines in all evaluated settings...
```

这是提交前必须修的第一优先级。

---

## 3. Related Work 现在太短，但未必是坏事

Related Work 从上一版的五段压成三段：RL for LLM agents and teacher-guided post-training、density-ratio、reward shaping 。优点是很紧凑，不占页数；缺点是略显 compressed，尤其第一段塞了太多东西。

现在第一段同时覆盖 LLM agents、GRPO/PPO/RLHF、AgentGym/RAGEN/GiGPO、LUFFY/CHORD、demonstration RL、offline-to-online RL 。如果 reviewer 熟悉这些方向，会觉得你们在“点名式 related work”，但考虑到主文篇幅，这可以接受。

我建议只做一个小改：把第一段拆成两个短段，增加可读性：

1. **RL for LLM agents.** ALFWorld/WebShop/ReAct/AgentGym/RAGEN/GiGPO/GRPO。
2. **Teacher-guided and offline-to-online RL.** LUFFY/CHORD/demo RL/AWAC/IQL。

不用增加内容，只是拆段。这样 Related Work 看起来更专业，不像把所有东西压在一段里。

Reward shaping 那段现在说得比较诚实：明确说 DUET state channel 是 trajectory-level progress bonus，是 strict potential-based shaping 的 coarse approximation 。这是安全的。

---

## 4. Method 部分：大体稳，但还有三处要修

### 4.1 “provably biased” 这个词建议删掉或弱化

§3.3 里写：

> “the corrective role is the one without which teacher-mixed GRPO is provably biased” 

这里 “provably biased” 容易被 reviewer 要求 formal theorem/proof。你们没有 theorem，只是 estimator-level derivation。建议改成：

```tex
the corrective role addresses the estimator-level failure modes identified in §3.2.
```

或者：

```tex
without these corrections, teacher-mixed GRPO violates the assumptions behind the group-normalized estimator in §3.2.
```

这样不触发 proof expectation。

### 4.2 Baseline separation 的 “unbiased on-policy gradient” 也略强

§3.4 结尾写：

> “prerequisite to interpreting Eq. (2) as an unbiased on-policy gradient” 

GRPO/PPO clipped surrogate 本身严格来说也不是 unbiased gradient estimator；加上 finite group normalization、clipping、LLM tokenization，更不宜说 unbiased。建议改成：

```tex
... prerequisite to interpreting Eq. (2) as an on-policy, group-relative update whose baseline is not contaminated by replayed teacher rewards.
```

或者：

```tex
... restores the intended zero-mean normalization of the on-policy subgroup.
```

这更准确，也不削弱 Bias 1。

### 4.3 §3.6 第一句 “DR3 correction is unbiased only when...” 需要改

§3.6 开头写：

> “The DR3 correction from §3.5 is unbiased only when the discriminator is well-calibrated” 

但 §3.5 已经花一整段说 DR3 不是 exact unbiased off-policy correction because filtered cache 。所以这里“unbiased”又把风险带回来了。

建议改成：

```tex
The DR3 weight from §3.5 is most meaningful when the discriminator is sufficiently trained and calibrated; in the first tens of training steps, the buffer is small and D_\phi is unreliable, so \hat w is noisy.
```

这句必须改，否则前后矛盾。

---

## 5. SC 部分现在诚实了，但与 §3.3/Conclusion 还有少量不一致

§3.7 已经很好：明确说实际用的是 trajectory-level progress score $P(\tau)$，加 $\beta P(\tau)$，是 strict potential-based shaping 的 coarse approximation，不 claim exact policy invariance 。Appendix C.4 也同步说明了实际实现：trajectory-level mean, $\beta P(\tau)/N_{\text{valid}}$ 分到 token reward，step-wise invariant variant implemented but not used 。这个修得很好。

但 §3.3 仍写：

> “applies potential-based reward shaping ... providing dense per-step signal while leaving the optimal policy unchanged” 

这与 §3.7 的 “we do not claim exact policy invariance” 冲突。必须改。

建议 §3.3 改成：

```tex
The state channel (§3.7) builds a progress score from successful teacher state visitation and adds a trajectory-level progress bonus to on-policy rollouts. This is inspired by potential-based shaping, but we use a coarser trajectory-level form for stability and do not rely on an exact policy-invariance claim.
```

Conclusion 也写：

> “potential-based state-progress shaping” 

也要改成：

```tex
progress-based state shaping
```

全篇凡是 “potential-based state-progress shaping” 出现在 Abstract/Conclusion/§3.3，都建议改成 “progress-based state shaping inspired by potential functions”。

---

## 6. Experiments 现在比上一版安全很多

Table 2 改成 focused estimator ablations，非常正确。现在没有 TBD，而且表 caption 也明确说 auxiliary ablations 在 Appendix E，不再把未完成的 BC/SC ablation 作为主 claim 。

Table 1 也清楚地写了：single seed, binomial CI in §5, baseline numbers are reproductions under same protocol, cross-infrastructure in appendix 。这比上一版好很多。

但有两个小风险：

### 6.1 “removing baseline separation collapses both 1.5B settings” 很强，但要确保 WS 真实完成

Table 2 现在写 w/o baseline separation 在 1.5B-WS 也是 0.0% 。如果这个 cell 已经真实完成，那非常强；如果只是沿用了 running/未充分验证的结果，风险很大。这个数字是目前 paper 里最强证据之一，不可以有半点虚。

如果 WS 的 0.0% 是 val@100 完整结果，保留。
如果不是完整 val@100，而是 early stop proxy，也要像 AF 一样脚注说明。现在脚注写 “both 1.5B settings collapse to 0% within first ~30 steps”，但只具体说明 AF stopped early at step 69 and val@50 proxy 。WS 如果也是 proxy，也必须说明；如果是完整 val@100，就写清楚 “WS was run to step 100”。

### 6.2 “DUET without step-indexed schedule” 对 CHORD 的对比要别太攻击

§4.2 写 CHORD 的 imitation weight is annealed as a function of training step, while DUET has no step-indexed schedule 。这个可以，但 Related Work 里 “manually-scheduled supervised loss” 也出现了 。不要再更攻击了。现在程度刚好。

---

## 7. Figure 2 改得很好；Figure 1 仍然需要升级

Figure 2 现在从 “Self-attenuation” 改成 “Internal regulation signals”，这非常对。caption 也明确说 $\hat w$ 只是 33% down-weighting，$\mu$ 由 discriminator accuracy map，不再过度 claim fade-out 。这个图现在是安全的。

Figure 1 还是上一版图，页面 4 视觉上比较像灰色流程图截图，字体小、细节多。caption 已经压缩了很多，这是好的 。但图本身仍建议换成更 NeurIPS 风格的 TikZ/vector。尤其图里还写着类似 “SC state channel potential shaping r ← r + β(Φ(s′)−Φ(s))”，但你们实际已经改成 trajectory-level progress bonus。这个是**视觉上的不一致**，必须修。

Figure 1 里的 SC box 应该改成：

```text
SC state channel
progress bonus
r(τ) ← r(τ) + βP(τ)
```

不要再画 per-step potential difference。这个非常重要，因为 reviewer 看图比看正文快。

---

## 8. Appendix 现在更诚实，但 Appendix E 有点弱

Appendix A 加了 DUET vs LUFFY cross-infrastructure table，非常好：LUFFY 3.5 vs 38.0，DUET 44.0 vs 45.5，这能支撑你们说 DUET reproducibility 更好 。

Appendix B 也很好地承认了 BC schedule sensitivity，不再说 wide flat region 。这是成熟写法。

Appendix E 目前只是“On ablating the two signal channels”的解释，没有真实 ablation 数字 。这个标题可能让 reviewer 期待看到结果，但里面没有表，只是说 noisy and future work。建议改标题，避免 misleading：

现在：

```tex
E On ablating the two signal channels
```

建议：

```tex
E Scope of auxiliary signal-channel ablations
```

或者：

```tex
E Notes on auxiliary signal channels
```

另外主文 Table 2 caption 说 “Auxiliary ablations ... are reported in Appendix E” ，但 Appendix E 没有 report ablation results。这里不准确。应改成：

```tex
Auxiliary signal-channel ablations require additional multi-seed validation; we discuss their scope in Appendix E.
```

主文 §4.3 也写 “we report ablations of each in Appendix E as they finish on the compute queue” 。这句不能出现在 submission 里，会显得工作未完成。建议改成：

```tex
We therefore do not use single-seed BC/SC ablations as central evidence; Appendix E discusses their expected roles and why robust attribution requires multi-seed evaluation.
```

不要提 “as they finish on the compute queue”。

---

## 9. Checklist 仍然是提交阻断项

NeurIPS checklist 还完全是模板，所有 answer 都是 TODO 。这个必须在最终 PDF 前清理，否则不是 reviewer 质疑，而是格式/desk reject 风险。

你们至少要：

* 删除 checklist instruction block；
* 填所有 [TODO]；
* 对 Q3 小心：不要说有 formal theoretical results。你们是 estimator derivation + known theorem reference，不是新 theorem。
* Q7 统计显著性可以写 Yes/Partial-ish 的 Yes：binomial CI over 200 tasks, no multi-seed due compute。
* Q10 broader impacts 不能空，最好加一段简短 discussion 或在 checklist justification 里说明 foundational agent RL risk。
* Q16 LLM usage：如果只用于写作，不是 core method，按 checklist wording 可以说 N/A 或 No；但如果政策要求披露写作辅助，你们按组内规范来。

---

## 10. 还需要全篇一致性替换的词

我建议做一次 grep，全篇替换/检查以下短语。

### “corrects these biases”

在 Abstract 可改；正文中 baseline separation 可以 correct，DR3 用 mitigate。建议统一：

* Bias 1: removes / corrects baseline contamination
* Bias 2: mitigates teacher-student mismatch

### “potential-based”

仅在讲 Ng et al. strict form 时使用。你们实际方法叫：

* progress-based state shaping
* progress-aware reward bonus
* inspired by potential functions

### “policy invariance”

只用于描述 Ng et al. 的 strict step-wise version，不用于 DUET reported runs。

### “unbiased”

尽量删掉。用：

* estimator-aware
* density-ratio-weighted
* idealized importance-weighted form
* mitigates mismatch

### “as the student improves”

改成：

* as discriminator-derived internal signals evolve
* as the discriminator strengthens
* as training progresses under internal signals

---

## 11. 我建议提交前必须完成的修改清单

按优先级排序：

1. **改 Abstract**：mitigate DR3、progress-based SC、among reproduced baselines、internal discriminator signals。
2. **改 §3.3 的 SC 描述**：不要说 exact policy-invariant，不要说 dense per-step signal。
3. **改 Figure 1 里的 SC box**：从 $r+\beta(\Phi(s')-\Phi(s))$ 改成 $r(\tau)+\beta P(\tau)$。
4. **改 §3.6 第一句**：删除 “unbiased only when”。
5. **改 §3.4 “unbiased on-policy gradient”**：换成 “zero-mean normalization / uncontaminated group-relative update”。
6. **改 Appendix E 和主文引用**：不要说 “reported as they finish on queue”；改成 scope/future multi-seed。
7. **确认 Table 2 WS 0.0 是否完整 val@100**；如果不是，脚注写清楚。
8. **填 NeurIPS checklist**。
9. **Figure 1 vector 化或至少修正文字**。
10. **Conclusion 同步降温**：progress-based shaping、among reproduced baselines、mitigate mismatch。

---

## 12. 修改后这篇的状态

如果完成上面这些，我会把这版评为：

**叙事：8.5/10**
**方法安全性：8/10**
**实验呈现：7/10**
**NeurIPS readiness：7.5/10**

仍然最大的客观弱点是 single seed 和 1.5B-WebShop sensitivity；但现在你们已经诚实处理了这些问题。对于 NeurIPS，诚实不一定扣分，反而比藏着被 reviewer 发现要好。

我的结论：**这版方向是对的，可以继续沿这个版本打磨，不要再大改叙事。** 当前任务不是“重写 paper”，而是做最后一轮 consistency 和 claim calibration。最核心的 paper slogan 仍然成立：

> Correct teacher replay before exploiting it.

只要你们把 DR3/SC/Abstract 的强 claim 再压低一点，这个故事就会比较稳。
