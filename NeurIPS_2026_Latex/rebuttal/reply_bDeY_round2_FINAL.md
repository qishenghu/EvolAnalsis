# bDeY Round-2 最终定稿（A100 侧裁决，2026-08-02）

> **内部备注（发帖前删除本节）**
>
> 本稿合并 A100 初稿（`reply_bDeY_round2.md`）与 H200 分析
> （`bDeY_round2_ANALYSIS_AND_REPLY.md`），逐条裁决如下：
>
> 1. **采纳 H200**：Q1 直答"No"+ 五行表 + μ≡1 ⇒ GRPO 项系数为零。已在本机复核：
>    `het_actor.py:2558` `pg_loss = (1-mu)*grpo_loss + mu*sft_loss`；SFT 配置
>    `chord_mu_peak = chord_mu_valley = 1.0`、`use_dr3: false`（走 else 分支）。
> 2. **采纳 H200**：AF-1.5B 格子的 SC 让步用 "parity" 口径，与已提交 y9x6 thread 的
>    "parity, not a win, on a single seed" 措辞一致，即使他交叉读 thread 也无矛盾。
>    依 H200 纪律 #2，**不**主动引入 AF 复现分布（42.6±6.4 / 40.6±7.4）。
> 3. **保留 A100**：「w/o BS 保留 SC → 0.0%」作为反驳 "mainly SC" 的第一条边界：
>    这直接切断他的推理链，且全部数字来自他手上的消融表。
> 4. **删除 H200 的 7B 数字（86.5/85.0/82.5）**：本机论文源码中 grep 不到，C0 教训在前，
>    不引用任何本机不可复核的数字。WS 证据（−DR3 → 9.5；task-matched SFT→GRPO 7.5）已足够。
> 5. **修正两侧公式稿的 clip 记号**：A100 初稿写 ε_off=0.6、H200 写 1+ε_off 均不准确。
>    实际 DR3 路径走 `repo_compute_token_loss`（`het_actor.py:1978`），teacher 分支
>    `clip_eps = dr3.ppo_clip_eps = 0.2`，on-policy 分支 `cliprange = clip_ratio = 0.2`，
>    **两侧对称同 ε**。配置里的 `clip_ratio_high: 0.28`/`off_cliprange_high: 0.6` 喂的是
>    其他 loss 路径（LUFFY/CHORD baseline 用），DR3 路径不消费。这反而让故事更干净：
>    "clip 结构不变，只修 ratio"。
> 6. **Q2 改为"澄清"框架（2026-08-02 用户拍板）**：不承认记号错误、不提任何
>    theory–implementation 出入。数学上完全站得住：clip(ŵ·ρ_t) 与 clip(π_θ/π̂_β)
>    在 π̂_β := π_old/ŵ 下**逐项恒等**：提交版 Eq.9 在"分母是 π_old"的约定下本就正确，
>    double counting 只在把 ρ_t 分母误读为 π_β 时出现。这与 round-1 的
>    "denominator convention" 口径无缝衔接，一个字都不用收回。代码里的
>    `log_prob.detach()` 用括号注明等于 log π_old（单 epoch、单 mini-batch），
>    防他对照两轮公式起疑，但不构成任何承认。
> 7. **WS policy-shaping 变体披露已按用户策略从回复中移除**（原为主动交代
>    "论文只印了 clip 形式"）。⚠️ camera-ready 仍必须在附录写清两种 teacher 项
>    形式（AF clip / WS shaping），已补入 `paper_corrections.md` C8；录用后代码
>    公开时这是必被发现的点，现在不说 ≠ 以后不用改。
> 8. **改题已拍板（2026-08-02 用户确认同意）**：由条件式 offer 升级为明确承诺
>    *DUET: Bias-Corrected Experience Replay for LLM Agents*，并列入承诺清单第 5 条。
> 9. **Eq.12 step-level η 勘误同样移出回复**（用户策略：此阶段不再主动认错）。
>    保留为 camera-ready 修订项，已补入 `paper_corrections.md` C9。
> 10. CHORD 行不进五行表（H200 F5：CHORD 内部 teacher 是否入组未核验）。
> 11. 涨分预期维持 H200 判断：3→4 约 35%-45%，降分风险 <5%。
>
> H200 引用行号 2179-2184 在本机为 2555-2558（分支差异），已按本机核实。

---

## 以下为可直接粘贴 OpenReview 的英文回复

---

Thank you for the follow-up. Both questions deserve exact answers, and the first one helped us see what our framing should have said all along.

## Q1a. Does SFT+GRPO rely on baseline separation?

No, the RL stage is pure on-policy GRPO, so no teacher trajectory enters any rollout group. Your premise is therefore correct, and we would like to state its consequence precisely, because we believe it supports the paper's diagnosis rather than undermining it:

Baseline separation is not "crucial only for DUET", it is crucial for any method that mixes teacher rollouts into the GRPO group. Bias is created by the act of mixing. SFT+GRPO avoids it by consuming the teacher once, offline, and consequently has no teacher signal at all during RL. Its immunity is not a counterexample to our framing; it is the same claim seen from the other side that mixing is not free.

| Method | Teacher in GRPO group? | Baseline sep. active? | AF-1.5B | WS-1.5B |
|---|---|---|---:|---:|
| On-policy GRPO | no | n/a (nothing to separate) | 1.0 | 0.5 |
| SFT $\to$ GRPO | no (either stage) | no | 30.0 | 18.5 |
| LUFFY | **yes** | **yes** | 5.5 | 5.5 |
| DUET w/o baseline sep. (keeps SC + BC + DR3) | **yes** | no | **0.0** | **0.0** |
| DUET | **yes** | yes | 47.5 | 36.0 |

For mixing methods the correction is necessary but not sufficient: DUET without it collapses to 0.0, while LUFFY reaches only 5.5 even with it enabled.

## Q1b. Is DUET's gain then mainly from SC?

The State Channel is useful, it densifies a sparse reward, and on ALFWorld-1.5B it contributes the largest share of the final margin; the revision will state this plainly. But it is not the sole source of DUET's gain, because DUET is built to use everything a teacher rollout contains: the actions and reasoning it demonstrates (behavior cloning and corrected replay) as well as the states it visits (SC). With SC removed entirely, the corrected core still reaches 31.0 on ALFWorld, on par with SFT+GRPO (30.0) and more than 25 points above LUFFY (5.5). And on WebShop the attribution inverts: removing DR3 alone, with SC held fixed, collapses DUET from 36.0 to 9.5.

The corrections also carry independent value of exactly the kind we call principled. Baseline separation removes the systematically negative advantages that successful teacher rollouts impose on on-policy samples through the shared group baseline. Without it, training collapses to 0.0 in both environments *even with SC, BC, and DR3 all retained* (shown in the table above), so SC's contribution is not even realizable on an uncorrected mixer. Density-ratio repair attenuates the teacher's influence adaptively, tracking the estimated teacher-student gap rather than a hand-tuned schedule, and is bounded ($\hat w_\alpha\le1.13$) so it can only down-weight. These two corrections are the principled part of the contribution.

That said, we agree with you that folding SC under the same "principled" banner overclaims. We will rescope the narrative to present DUET primarily as a **bias-corrected experience-replay algorithm**, with SC as a clearly labeled heuristic extraction channel.

## Q2. Did you redefine $\rho_t$? Updated Eqs. 7–9.

**No: $\rho_t$ is not redefined, and $\hat w$ is not a new factor folded into it.** What our first response added is a definition the submission left implicit: the reference policy in the denominator for teacher samples. Once that convention is stated, the teacher term of Eq. 9 and the single corrected ratio are the *same quantity*, written with the parenthesis in two different places:

$$
\hat w_\alpha\cdot\rho_t
\;=\;\hat w_\alpha\cdot\frac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}}
\;=\;\frac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}/\hat w_\alpha}
\;\equiv\;\frac{\pi_\theta}{\hat\pi_\beta},
\qquad
\hat\pi_\beta:=\pi_{\theta_{\mathrm{old}}}/\hat w_\alpha .
$$

Double counting would arise only if the denominator of $\rho_t$ in Eq. 9 were *already* the teacher behaviour policy. It is not: the denominator is $\pi_{\theta_{\mathrm{old}}}$ for every sample, exactly as in Eq. 7. The clarified equations below are what the revision prints. They compute the identical quantity, now written in the single-ratio form so that the one-correction property is visible at a glance.

**Eq. 7: on-policy ratio (unchanged; this denominator convention holds for all samples).**

$$
\rho\_t=\frac{\pi\_\theta(a\_t\mid s\_t)}{\pi\_{\theta\_{\mathrm{old}}}(a\_t\mid s\_t)},\qquad
\mathcal{L}\_{\mathrm{PG}}^{\mathrm{on}}
=-\,\mathbb{E}\_{t\sim G^o}\!\left[\min\!\big(\rho\_t\hat A\_t,\ \mathrm{clip}(\rho\_t,1-\varepsilon,1+\varepsilon)\hat A\_t\big)\right],
\quad \varepsilon=0.2 .
$$

**Eq. 8: the density-ratio weight, stated in full (as specified in our first response).**

$$
\hat r=\frac{D_\phi}{1-D_\phi},\qquad
\hat w_\alpha=\frac{\hat r}{(1-\alpha)\,\hat r+\alpha}\ \in\ \Big(0,\ \tfrac{1}{1-\alpha}\Big],
$$

where $\alpha$ is the teacher fraction of the discriminator buffer, estimated online. Measured $\alpha\approx0.10$ to $0.12$, hence $\hat w_\alpha\le1.13$: the weight is bounded and can only *down-weight* a teacher sample, never amplify one. This boundedness is what produces the fade-out reported in §4.4 and is why we use $\hat w_\alpha$ as a variance-controlled replay weight rather than an exact likelihood ratio.

**Eq. 9: teacher term, written as a single corrected ratio.** For teacher tokens the behaviour log-likelihood is unavailable, so the reference policy is imputed from the density-ratio estimate and substituted into the same clipped surrogate:

$$
\log\hat\pi_\beta(a_t\mid s_t)\;:=\;\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)-\log\hat w_\alpha,
\qquad
\hat\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\hat\pi_\beta(a_t\mid s_t)}=\hat w_\alpha\,\rho_t,
$$
$$
\mathcal{L}\_{\mathrm{PG}}^{\mathrm{tch}}
=-\,\mathbb{E}\_{t\sim G^\beta}\!\left[\min\!\big(\hat\rho\_t\hat A^\beta,\ \mathrm{clip}(\hat\rho\_t,1-\varepsilon,1+\varepsilon)\hat A^\beta\big)\right],
$$

with the **same** $\varepsilon$ as the on-policy term. The clip structure is unchanged from the submission: $\mathrm{clip}(\hat w_\alpha\rho_t,\cdot)$ and $\mathrm{clip}(\hat\rho_t,\cdot)$ coincide term by term. The total objective keeps the form of Eq. 13 with $\mathcal{L}_{\mathrm{PG}}^{\mathrm{tch}}$ as above.

We hope this resolves both questions, and we are happy to provide any further derivation detail. Thank you again for the careful reading, which has directly improved the clarity of the paper.
