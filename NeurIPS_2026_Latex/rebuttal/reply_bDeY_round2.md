# Reply to Reviewer bDeY (round 2)

Thank you for pressing on both points — the first exposes a framing defect we agree with, and the
second exposes a second equation error we had not reported. We answer directly.

---

## Q1. Does SFT+GRPO use baseline separation? Is DUET's gain mainly the State Channel?

### It does not use it, and it *cannot* — baseline separation is undefined for SFT+GRPO

SFT+GRPO's RL phase is purely on-policy: no teacher trajectory enters the rollout group
(`experience_replay.enable: false`, teacher mix ratio $0$; `teacher_baseline_separation.enable:
false` is therefore also set). Bias 1 is a property of a *group that mixes teacher and on-policy
samples*. With no teacher sample in the group there is nothing to separate — the correction is not
merely unused but mathematically vacuous. Our implementation makes this explicit; the separated
branch is gated on the batch actually containing teacher tokens:

```python
has_teacher = teacher_mask is not None and teacher_mask.sum().item() > 0
if enable_teacher_sep and has_teacher:
    advantages, returns = compute_grpo_outcome_advantage_teacher_baseline_separated(...)
else:
    advantages, returns = compute_grpo_outcome_advantage(...)   # standard GRPO
```
(`ae_ray_trainer.py:1497–1526`)

So SFT+GRPO's performance is not evidence that the correction is DUET-specific. It is evidence that
a method which does not mix does not incur the bias that mixing creates. SFT+GRPO avoids Bias 1 by
consuming the teacher **once, offline, before RL begins**, and therefore has no teacher signal
during RL at all. That is a real design choice with a real cost, and DUET's claim is the
complementary one: online mixing is worth more — $+17.5$ pp over SFT+GRPO in *both* 1.5B settings —
**provided** its two biases are corrected first.

### The methods that do mix are LUFFY and CHORD, and we ran both *with* baseline separation

Both baselines in Table 1 have `teacher_baseline_separation.enable: true`
(`alfworld_qwen1.5b_{luffy,chord}.yaml:29–30`, and the WebShop equivalents). We did not reserve the
correction for ourselves; we gave our Stage-1 fix to the competing methods. With it, they reach
$5.5\%$ / $27.0\%$ on ALFWorld and $5.5\%$ / $11.5\%$ on WebShop, against DUET's $47.5\%$ /
$36.0\%$. The margin over the mixing baselines is therefore *not* attributable to baseline
separation, because they already have it.

### Why "the gain is mainly SC" is not supported by our ablation table

The decisive test is already in Table 2, and we think it was easy to miss. **The `w/o baseline
separation` row retains SC** — it also retains BC and DR3, and removes only the correction. It
collapses to $0.0\%$ in both environments.

> $\text{SC} + \text{BC} + \text{DR3}$, without the correction $\;\to\; 0.0\%$ (both environments).

SC's signal is therefore not realizable at all without Stage 1. Attributing the margin to SC
presupposes an additive decomposition in which SC contributes independently of the corrections; the
data reject that decomposition. The converse also holds: correction without extraction is
insufficient — LUFFY has baseline separation, no SC, and reaches $5.5\%$.

The per-environment marginals also do not show SC dominating uniformly:

| Component removed | 1.5B-ALFWorld | 1.5B-WebShop |
|---|---:|---:|
| Baseline separation (correction) | $-47.5$ | $-36.0$ |
| DR3 (correction) | $-0.0$ *(-14.0 at step 50)* | $-26.5$ |
| State Channel (heuristic) | $-16.5$ | $-35.0$ |
| BC | $-13.5$ | $-19.5$ |

DR3 — a correction, not a heuristic — is worth $26.5$ pp on WebShop, larger than SC's entire
contribution on ALFWorld. On ALFWorld DR3 reaches the same endpoint but at markedly worse sample
efficiency ($42.5$ vs. $28.5$ at step $50$).

### Where we agree with you, and what we will change

**We accept the framing criticism.** "Principled" should not be applied as a blanket adjective over
the whole method, and the State Channel is a heuristic. §3.3 already states that SC is only
*inspired by* potential-based shaping and that "we do not claim exact policy invariance" — but the
abstract and introduction do not carry that qualification through, and that is a genuine defect
rather than a matter of emphasis. In the revision we will scope the claim explicitly:

- **principled** = the identification of the two mixing biases and their corrections (Stage 1),
  each derived from the GRPO estimator and each verified by ablation;
- **State Channel** = a heuristic instantiation of progress-based shaping, ablated and quantified,
  not claimed to be principled.

**We also quantified how much of SC is teacher knowledge rather than generic reward densification**
— we think this is the number your question is really asking for. During the rebuttal we replaced
$\Phi$ with a **shuffled** progress map: the same teacher progress values, reassigned to mismatched
tasks, so the bonus keeps its magnitude and density but carries no valid ordering information.

| ALFWorld 1.5B, val@100 | SR |
|---|---:|
| DUET (real progress map) | $47.5\%$ |
| DUET (shuffled progress map) | $41.0\%$ |
| DUET w/o State Channel | $31.0\%$ |

Of SC's $16.5$ pp, roughly $10.0$ pp is generic densification of a sparse reward and only $\approx
6.5$ pp comes from the teacher's actual state ordering. We will report this decomposition in the
revision rather than let the reader assume the full $16.5$ pp is teacher-derived. It narrows what
we can claim for SC, which is precisely why we think it belongs in the paper.

---

## Q2. Did you redefine $\rho_t$? What are the updated Eqs. 7–9?

**Yes, $\rho_t$ is redefined in the revision — but to match the code, not to accommodate the
objection.** The algorithm is unchanged; every number in the paper was produced by the
implementation below. Your reading of the submitted Eq. 9 was correct: as printed it is a product of
two corrections, and the implementation never forms that product. In re-checking this we also found
that **Eq. 8 is wrong as printed**, which we report here.

### Eq. 7 — the ratio, now with an explicit reference policy

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{ref}}(a_t\mid s_t)},
\qquad
\pi_{\mathrm{ref}}=
\begin{cases}
\pi_{\theta_{\mathrm{old}}}, & \text{on-policy samples},\\[2pt]
\hat\pi_\beta, & \text{teacher samples},
\end{cases}
$$

with $\hat\pi_\beta$ defined by the imputation in Eq. 9. The submitted Eq. 7 stated only the
on-policy case and then Eq. 9 multiplied it by $\hat w$; that is the source of the double-counting
reading.

### Eq. 8 — the applied weight is the $\alpha$-relative ratio, not $D/(1-D)$

Submitted: $\hat w = D_\phi/(1-D_\phi)$. Implemented (`dr3_ratio.py:845–852`;
`use_relative_ratio` defaults to `True` and is not overridden by any config in the repository):

$$
\hat r=\frac{D_\phi}{1-D_\phi},
\qquad
\hat w_\alpha=\frac{\hat r}{(1-\alpha)\hat r+\alpha}\ \in\ \Bigl(0,\ \tfrac{1}{1-\alpha}\Bigr],
$$

where $\alpha$ is the teacher fraction of the discriminator buffer, estimated online. This is the
standard *relative* density ratio — the ratio against the mixture rather than against $\pi_\beta$
alone — and it is bounded above by $1/(1-\alpha)$. Empirically $\alpha\approx0.10$–$0.12$, so
$\hat w_\alpha\le 1.13$ (logged `dr3/w_clip_upper` $=1.105$–$1.135$). **DR3 can therefore only
down-weight a teacher sample, never amplify one**, which is what produces the fade-out reported in
§4.4 (`dr3/w_off_mean`: $0.937\to0.758\to0.663\to0.530$). We should have stated this in the
submission; it is also the reason $\hat w$ is a bias-mitigating replay weight rather than an exact
likelihood ratio.

### Eq. 9 — imputation, not a product

For teacher tokens the *behaviour* log-probability is **replaced**, not multiplied:

$$
\log\hat\pi_\beta(a_t\mid s_t)\;:=\;\operatorname{sg}\bigl[\log\pi_\theta(a_t\mid s_t)\bigr]-\log\hat w_\alpha
$$
$$
\Longrightarrow\quad
\rho^{\mathrm{tch}}_t=\frac{\pi_\theta}{\hat\pi_\beta}
=\hat w_\alpha\cdot\exp\bigl(\log\pi_\theta-\operatorname{sg}[\log\pi_\theta]\bigr)
$$
$$
\mathcal{L}^{\mathrm{tch}}_{\mathrm{PG}}(\theta)=-\,\mathbb{E}_{(s,a)\sim G^\beta}
\Bigl[\hat A^\beta\cdot\operatorname{clip}\bigl(\rho^{\mathrm{tch}}_t,\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{off}}\bigr)\Bigr]
$$

Code path:

```python
old_lp_new[apply_mask] = log_prob.detach()[apply_mask] - log_w[apply_mask]  # het_actor.py:1881
old_log_prob = old_lp_new                                                  # het_actor.py:1918
ratio = torch.exp(log_prob - old_log_prob)                                 # het_core_algos.py:88-90
```

There is **one** ratio in the clipped surrogate, and it contains $\hat w_\alpha$ once. At the
evaluation point $\rho^{\mathrm{tch}}_t=\hat w_\alpha$ and
$\nabla_\theta\rho^{\mathrm{tch}}_t=\hat w_\alpha\nabla_\theta\log\pi_\theta$: a single
$\hat w_\alpha$-weighted policy-gradient term, applied inside the same clipped surrogate as the
on-policy term. The clip bounds are asymmetric between the two groups
($\varepsilon_{\mathrm{low}}=0.2$; $\varepsilon_{\mathrm{high}}=0.28$ on-policy,
$\varepsilon_{\mathrm{off}}=0.6$ for teacher tokens), which the submitted Eq. 9 also did not state.
Note that $\hat w_\alpha\le1.13<1+\varepsilon_{\mathrm{off}}$, so the upper clip does not bind on
teacher tokens; the $\alpha$-relative bound is what limits the teacher term.

### One clarification about our first response

Our earlier reply decomposed $\rho_t$ as $(\pi_\theta/\pi_{\mathrm{old}})\cdot(\pi_{\mathrm{old}}/q)$
and described the two factors as correcting different terms. That is the right intuition for the
imputation, but it overstates the role of the first factor in our actual configuration: we train
with `ppo_epochs = 1` and a single mini-batch per rollout batch
(`ppo_mini_batch_size` $\times\,n$ = `train_batch_size` $\times\,n$ = $64$), so
$\pi_{\theta_{\mathrm{old}}}=\pi_\theta$ at the evaluation point up to recomputation noise and the
drift factor is identically $1$. The teacher term reduces exactly to a
$\hat w_\alpha$-weighted policy-gradient step. We should have said this plainly; the substance —
one weight, not two — is unchanged, and this makes it easier to verify rather than harder.

### Revision commitments from this exchange

1. Eq. 7 restated with the sample-dependent reference policy.
2. Eq. 8 replaced by the $\alpha$-relative ratio, with the bound $1/(1-\alpha)$ and the measured
   $\hat w_\alpha\le1.13$ stated.
3. Eq. 9 rewritten as the imputation above, with the asymmetric clip bounds made explicit, and one
   sentence noting that the teacher term is a $\hat w_\alpha$-weighted policy-gradient step rather
   than a stacked importance correction.
4. Abstract and introduction rescoped so that "principled" refers to the Stage-1 corrections; SC
   described as a heuristic shaping instantiation.
5. Shuffled-progress-map decomposition of SC added to the ablation section.
6. (Related to the above, found while preparing this reply.) Eq. 12 states only the trajectory-level
   bonus $\lambda P(\tau)$, but our ALFWorld runs additionally use a step-level potential difference
   $\eta[\Phi(s_{t+1})-\Phi(s_t)]$ with $\eta=0.05$ (WebShop runs do not). This term appears in the
   Appendix-F reward decomposition but not in §3.3, and we will add it to Eq. 12 with its
   per-environment setting. We do not claim policy invariance for it.
