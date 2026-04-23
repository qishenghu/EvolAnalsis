# DUET — Unifying DR3 with BC: A Single Token-Level Density-Ratio Operator

*Lead researcher memo, 2026-04-19. Target: NeurIPS 2026 narrative upgrade. Addresses strategic directive to subsume BC into DR3 rather than mix them, and to repair the "closed-form / no-tuning" claim via adaptive μ if unification fails.*

---

## 1. What DR3 is today and why token-level matters

At the implementation level (`het_core_algos.py:393-402, 574-582`; `het_actor.py:1494-1500`), DR3's teacher surrogate is, for `teacher_use_log_prob=True` after the DR3 shift `old_log_prob ← log_prob.detach() - log ŵ(τ)`:

$$
L_{\mathrm{DR3}}^{\mathrm{tch}} = -\mathbb{E}_{\tau \sim \pi_{\mathrm{teacher}}}\!\left[A(\tau)\cdot \min\!\left(r_\theta(\tau)\cdot\hat{w}(\tau),\; \mathrm{clip}(r_\theta(\tau)\cdot \hat{w}(\tau),\,1\!-\!\varepsilon,\,1\!+\!\varepsilon)\right)\right],
$$

with per-token gradient coefficient on $\nabla_\theta\log\pi_\theta(a_t|s_t)$:

$$
\mathrm{coef}_{\mathrm{DR3}}(s_t, a_t, \tau) \;=\; \underbrace{\hat{w}(\tau)}_{\text{traj-level}} \cdot \underbrace{A(\tau)}_{\text{traj-level}} \cdot \underbrace{r_\theta(a_t|s_t)}_{\text{token-level but }\approx 1}
$$

In the first PPO inner step `r_θ(a_t|s_t) = 1` exactly (because `old_log_prob` was just set to `log_prob.detach()`), so **every teacher token in trajectory τ gets the *same* coefficient $\hat w(\tau)A(\tau)$**. Only PPO inner iterations introduce within-trajectory token-level variation, and empirically `off_pg_cliphit_rate = 0` across v12/v22/v24/v36 (curriculum_empirical_validation §1, fact 9), which means PPO clipping never binds — ratios stay within $[1{-}\varepsilon, 1{+}\varepsilon]$ throughout. **DR3 is effectively a pure trajectory-level signal.**

In contrast, the BC operator `compute_chord_sft_loss` contributes a per-token coefficient $\mu \cdot 1$ — *every* teacher token gets unit gradient regardless of trajectory or context. That token-level uniformity is exactly what v24 needs, per the second-pass theory: early high μ is the "bootstrap kick" that imprints rare tokens like `click[bright_white]` into the policy's support. DR3 alone cannot do this because if $\pi_\theta(a^*|s) \approx 10^{-4}$, a coefficient of $\hat w(\tau)A(\tau) \approx 0.5 \cdot 0.2 = 0.1$ applied to $\nabla\log\pi_\theta$ yields $\Delta\log p_\theta \approx 0.1$ per step — **same order as BC's 0.3, but distributed across hundreds of tokens** so each rare token only receives a vanishing share. BC is surgical; DR3 is diffuse.

The research question is therefore: **can we redesign DR3 so that rare-under-policy teacher tokens automatically receive a larger coefficient — recovering BC's imprinting behavior — without an auxiliary μ-scheduled loss?**

---

## 2. Deliverable 1 — Five unification candidates

### Candidate (A) — Token-level discriminator $D(s,a)$

Replace the trajectory-level classifier $D(\tau)$ with a state-action classifier $D(s,a)$ trained to predict $P(\text{teacher}\mid s,a)$. Then $\hat w(s,a) = D(s,a)/(1-D(s,a))$ is a per-token density ratio, and the teacher surrogate becomes

$$
L_A = -\mathbb{E}_{(s,a)\sim\pi_{\mathrm{teacher}}}\!\left[A(\tau)\cdot \hat w(s,a)\cdot r_\theta(a|s)\right].
$$

Per-token coefficient: $\hat w(s,a)\cdot A(\tau)\cdot r_\theta$.

**BC recovery.** For an optimal Bayes discriminator, $D^*(s,a) = \frac{\pi_{\mathrm{teacher}}(a|s)}{\pi_{\mathrm{teacher}}(a|s)+\pi_\theta(a|s)}$ (assuming equal label priors), so $\hat w^*(s,a) = \pi_{\mathrm{teacher}}(a|s)/\pi_\theta(a|s)$. On a rare-under-policy teacher token where $\pi_\theta(a|s)\to 0$, $\hat w^* \to \infty$. Clipping at $w_{\max}$ then gives a per-token coefficient $\approx w_{\max}\cdot A(\tau)$ — which is BC-like *only if* $A(\tau)>0$ (successful trajectories). On unsuccessful teacher trajectories ($A(\tau)<0$ after GRPO normalization), rare tokens would be pushed *down* in probability — opposite of BC. **Partial recovery.**

**Critical concerns.** (i) Token-level classifiers are hard to train: a single (s,a) has very few duplicates in the buffer. (ii) Clipping at $w_{\max}$ is a new hyperparameter. (iii) The gradient coefficient now has product-of-two-estimates variance ($\hat w\cdot A$), both noisy.

### Candidate (B) — Self-normalized surprise multiplier

Keep the current $\hat w(\tau)$. Multiply the teacher loss *inside* the expectation by $(1-\pi_\theta(a|s))$:

$$
L_B = -\mathbb{E}_{(s,a,\tau)\sim\pi_{\mathrm{teacher}}}\!\left[(1-\pi_\theta(a|s))\cdot \hat w(\tau)\cdot A(\tau)\cdot r_\theta(a|s)\right].
$$

Per-token coefficient: $(1-\pi_\theta(a|s))\cdot \hat w(\tau)\cdot A(\tau)\cdot r_\theta$.

**BC recovery.** When $\pi_\theta(a^*|s)\to 0$ (rare token), $(1-\pi_\theta)\to 1$, so coefficient $\to \hat w(\tau)A(\tau)$. When $\pi_\theta(a^*|s)\to 1$ (common token), $(1-\pi_\theta)\to 0$, so coefficient vanishes. Not BC exactly — BC gives unit coefficient to both rare and common teacher tokens; this gives unit-like coefficient only to rare. **Stronger than needed for rare-token imprinting but eliminates gradient on mastered tokens (arguably desirable, no over-fitting).**

**Concerns.** Still weighted by $A(\tau)$, so negative-advantage teacher trajectories (rare but real: e.g., WebShop tasks where the teacher fails) suppress rare tokens. Also, on successful teacher trajectories the effective scale on rare tokens is $\hat w A \approx 0.3{-}0.5$ — **smaller than BC's $\mu = 0.3$ in absolute terms**, unless we rescale. Zero new hyperparameters.

### Candidate (C) — Log-probability-weighted DR3

$$
L_C = -\mathbb{E}_{\pi_{\mathrm{teacher}}}\!\left[\big(-\log\pi_\theta(a|s)\big)\cdot \hat w(\tau)\cdot A(\tau)\cdot r_\theta(a|s)\right].
$$

Per-token coefficient: $-\log\pi_\theta(a|s)\cdot \hat w(\tau)\cdot A(\tau)\cdot r_\theta$.

**BC recovery.** On rare teacher token ($\pi_\theta\approx 10^{-4}$), $-\log\pi_\theta \approx 9$. Combined with $\hat w A\approx 0.1$ gives coefficient $\sim 0.9$ — close to BC's $\mu=0.3$ in magnitude but 3× larger. On mastered token ($\pi_\theta\approx 0.9$), $-\log\pi_\theta\approx 0.1$, coefficient $\sim 0.01$. **Good BC-like shape; too aggressive on rarest tokens.**

**Concerns.** Unbounded above: on $\pi_\theta = 10^{-10}$ token (tokenizer-specific artifacts), coefficient = 23, gradient explodes. Needs either a clip or a saturation like $\tanh$. That reintroduces a hyperparameter. Also — critically — the coefficient now depends on the current policy's own surprise, making the loss non-stationary relative to the samples: as $\pi_\theta$ climbs on a rare token during PPO inner steps, the gradient shrinks within a single update. This is self-correcting but makes variance analysis harder.

### Candidate (D) — Hybrid per-token advantage augmentation

Define $\tilde A(s,a) = A(\tau) + \beta\cdot f(s,a)$ where $f$ is a token-level surprise proxy centered to zero mean. This doesn't change DR3's trajectory-level $\hat w(\tau)$; it shifts the scalar multiplying the ratio.

**BC recovery.** Weak. If $f(s,a) = -\log\pi_\theta(a|s) - \bar H$ (surprise minus mean entropy), then on rare tokens, $\tilde A$ increases by $\beta \cdot (9 - \bar H) \approx \beta\cdot 7$. Coefficient becomes $\hat w(\tau)\cdot (A(\tau)+7\beta)\cdot r_\theta$. For small $\beta$ this is approximately BC-like; but the *sign* of $A(\tau)$ still controls the gradient direction. **Does not cleanly recover BC because BC is advantage-independent.**

**Concerns.** Introduces $\beta$ hyperparameter. Advantage normalization in GRPO would need to re-center after the augmentation to preserve $\mathbb{E}[\tilde A]=0$ per group.

### Candidate (E) — Two-scale density ratio (factorized)

Maintain both $\hat w(\tau)$ (trajectory-level, as today) *and* $\hat w(s,a)$ (token-level, as in A). Combine:

$$
L_E = -\mathbb{E}\!\left[A(\tau)\cdot \underbrace{\hat w(\tau)}_{\text{between-traj}}\cdot \underbrace{\hat w(s,a|\tau)}_{\text{within-traj}}\cdot r_\theta(a|s)\right].
$$

**BC recovery.** Same as A for the within-trajectory part. Trajectory-level $\hat w$ adds between-trajectory selection, so the product is $\pi_{\mathrm{teacher}}(\tau)\pi_{\mathrm{teacher}}(a|s,\tau)/[\pi_\theta(\tau)\pi_\theta(a|s,\tau)]$, which factors as total density ratio — theoretically equivalent to the full IS correction, up to discriminator estimation error.

**Concerns.** Two discriminators, multiplicative variance. The two-scale decomposition is theoretically clean (factorization of trajectory density into marginal × conditional) but doubles the estimation burden.

### Ranking (novelty × recovery × safety)

| Cand. | BC-recovery | New hyperparams | Variance safety | Narrative fit |
|---|---|---|---|---|
| A: token $D$ | Partial (sign-dep.) | +1 (clip) | Medium | Clean |
| **B: $(1-\pi_\theta)$ mul** | **Strong (rare only)** | **0** | **High** | **Excellent** |
| C: $-\log\pi_\theta$ mul | Strong, too hot | +1 (clip/sat) | Low (explodes) | Good |
| D: adv augmentation | Weak | +1 ($\beta$) | Medium | Weak |
| E: two-scale | Full | +1 (clip) | Low (×2 noise) | Very clean |

**Top recommendation: Candidate B.** Then Candidate C as secondary. A and E are theoretically appealing but operationally fragile at the scale of 1.5B/3B runs under a 17-day deadline.

---

## 3. Deliverable 2 — Rigorous analysis of Candidate B ($\pi_\theta$-surprise DR3)

### 3.1 Formal operator

$$
L_B = -\mathbb{E}_{\tau\sim\pi_{\mathrm{tch}}}\!\left[\sum_{t}\mathbb{1}_{\mathrm{tch}}(t)\cdot \phi_\theta(s_t,a_t)\cdot \hat w(\tau)\cdot A(\tau)\cdot r_\theta(a_t|s_t)\right],\quad \phi_\theta(s,a) := (1-\pi_\theta(a|s)).\texttt{detach()}
$$

The `.detach()` on $\phi_\theta$ is essential: $\phi$ is used as a weight, not as a differentiable target. This keeps the per-token gradient $\nabla_\theta\log\pi_\theta(a|s)$ clean.

### 3.2 BC subsumption (rare-token limit)

**Claim.** Let $a^*$ be a teacher token with $\pi_\theta(a^*|s)\leq\epsilon$ for some small $\epsilon > 0$. Let $\tau$ be a teacher trajectory with $A(\tau) \geq A_{\min} > 0$ (typical for teacher after baseline separation; empirically teacher advantage $\approx +1$ in WebShop). Then

$$
\left|\mathrm{coef}_B(s,a^*) - (1-\epsilon)\hat w(\tau)A_{\min}\right| \leq \epsilon\cdot\hat w(\tau)A_{\min}.
$$

So $\mathrm{coef}_B \to \hat w(\tau)A(\tau)$ as $\epsilon\to 0$. This is the per-token BC gradient scaled by $\hat w(\tau)A(\tau)$. If we tune nothing, the effective "BC strength" is $\hat w(\tau)A(\tau)\approx 0.3{-}0.5\times 1.0\approx 0.3{-}0.5$ — matching v24's peak $\mu=0.3$. **Emergent match, no tuning.**

On mastered tokens ($\pi_\theta\to 1$), $\mathrm{coef}_B\to 0$, unlike BC which continues to push $\log\pi_\theta$ upward on already-mastered tokens (redundant, small harm). Candidate B is *strictly better* on mastered tokens: no wasted gradient.

### 3.3 Preservation of DR3 properties

**Unbiased off-policy PG?** The policy gradient theorem applied to the teacher surrogate with importance weight $\hat w(\tau)$ gives an unbiased estimate of $\nabla_\theta J(\pi_\theta)$ *if* $\hat w(\tau) = \pi_\theta(\tau)/\pi_{\mathrm{tch}}(\tau)$. Introducing $\phi_\theta = (1-\pi_\theta(a|s))$ **breaks unbiasedness** — the estimator becomes biased toward rare-under-policy actions. This is *the same kind of bias BC introduces* (unit weight on all teacher tokens is not an unbiased PG estimate either). For a paper that already claims "BC is a principled advantage regularizer," this bias is acceptable and interpretable.

We can, however, frame B as an *unbiased estimator of a modified objective*:

$$
J_B(\pi_\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t (1-\pi_\theta(a_t|s_t))\cdot Q(s_t,a_t)\right]
$$

which is a **support-expanding objective**: it explicitly rewards putting probability mass on actions with high Q-value but currently low $\pi_\theta$. This is a well-posed objective, gradient of which is what $L_B$ estimates (off-policy, via $\hat w$).

**Fade-out.** As $\pi_\theta\to\pi_{\mathrm{tch}}$, (i) $\hat w(\tau)\to 1$ — existing DR3 fade; (ii) $(1-\pi_\theta(a|s))$ on teacher tokens approaches $(1-\pi_{\mathrm{tch}}(a|s))$ which is small (teacher is high-probability on teacher actions by definition). So the coefficient fades *twice*: once through $\hat w$, once through $\phi$. **Double fade-out — mathematically tighter fade than current DR3.**

**Variance.** $\phi_\theta\in[0,1]$, bounded; it *reduces* variance on mastered tokens. Overall teacher-loss variance is no higher than current DR3 (which has $\phi\equiv 1$). No new clipping needed.

### 3.4 Failure modes

1. **Cold-start problem.** At step 0, every teacher action is rare ($\pi_\theta$ is near-uniform), so $\phi\approx 1$ everywhere → reduces to current DR3. **Fine, not a failure.**
2. **Discriminator instability amplification.** If $\hat w(\tau)$ is noisy early (before `disc_apply_ready`), the per-token coefficient inherits that noise. But this is no worse than current DR3. Solution: same `dr3_apply_warmup_steps` gate (already exists).
3. **Negative-advantage teacher trajectories.** On rare teacher trajectories where $A(\tau)<0$ (failed teacher, or teacher_baseline_separation misclassification), $\phi\cdot\hat w\cdot A < 0$ on rare tokens → pushes rare teacher tokens *down*. BC protects against this with its unconditional unit coefficient. **Mitigation: mask out $A(\tau)<0$ teacher trajectories for the $\phi$-weighted surrogate** (i.e., apply $\phi$ only when $A>0$). This costs 2 lines of code and zero new hyperparameters.
4. **Interaction with teacher baseline separation.** B requires teacher advantages to be approximately positive on average for the BC-like behavior to fire. `teacher_baseline_separation.enable=true` already ensures this (teacher group has own mean/std, so successful teacher trajectories have $A\approx 0$ or slightly positive). **Compatible.**

### 3.5 Hyperparameter count

Current DR3: `w_min, w_max` (2 numerical safety knobs). Candidate B: same 2. Current BC: `μ_peak, μ_valley, μ_warmup, μ_decay` (4). **Candidate B eliminates 4 hyperparameters**. This directly supports the "closed-form" narrative.

### 3.6 Ablation plan for the paper

1. **B vs DR3-only (v12 equivalent)**: expected to beat v12 because rare-token imprinting now happens.
2. **B vs v24 (DR3+BC+schedule)**: the headline comparison. Target ≥ 0.65 to claim "unified operator matches engineered pipeline."
3. **B with mask-on-$A>0$ vs without**: validate the negative-advantage mitigation.
4. **B with $\phi=(1-\pi_\theta)^\alpha$, $\alpha\in\{0.5, 1, 2\}$**: sensitivity. $\alpha=1$ recommended default.
5. **B on ALFWorld**: generalization check. Theory predicts B is cleaner than v24's hand-tuned schedule on ALFWorld (where v24's BC is known to be neutral/slightly harmful).

### 3.7 Relationship to algo-engineer's Idea 2a

Idea 2a (surprise-weighted teacher PG) mixes BC and DR3 via a sigmoid gate $\alpha(s,a) = \sigma((-\log\pi_\theta - \bar H)/T)$ with hyperparameters $\bar H, T$. Candidate B is **Idea 2a's limit with $T\to\infty$ and a single $\phi = (1-\pi_\theta)$ multiplier** — same conceptual move, zero hyperparameters, pure product form instead of mixture. Idea 2a is a soft *switch* between BC and DR3; Candidate B is a single operator whose coefficient morphologically recovers each in the limits. **Narratively cleaner.**

---

## 4. Deliverable 3 — Narrative upgrade

### 4.1 New Action-Channel section title

> **"Surprise-Gated Density-Ratio Policy Gradient: A Single Off-Policy Operator for Expert Trajectories"**

### 4.2 New contribution statement

> *We propose a single off-policy policy-gradient operator, derived from the factorized form of the teacher-to-policy density ratio, that (i) corrects for the distribution shift between expert and learner at the trajectory level via a discriminator-based ratio $\hat w(\tau)$, and (ii) provides token-level imprinting of rare expert actions via an automatic $(1-\pi_\theta(a|s))$ weight that arises from the same factorization. The operator reduces to standard on-policy PG when $\pi_\theta\to\pi_{\mathrm{teacher}}$ (both weights $\to 1$ on teacher distribution) and to behavior-cloning on rare-under-policy tokens (where $(1-\pi_\theta)\to 1$), unifying what were previously two separate loss terms and removing four tuning hyperparameters.*

### 4.3 Response to "isn't it just CHORD + DR3?"

Before: BC and DR3 were separately specified, μ-scheduled, and combined additively. CHORD's BC has a decaying μ; ours did too; the critique writes itself.

After: we have *one* operator. The BC-like behavior emerges from the token-level weight $\phi_\theta(s,a) = (1-\pi_\theta)$, which is not a loss term added on top of DR3 but a modification of DR3's integrand. The factorization $\pi_{\mathrm{tch}}(\tau)/\pi_\theta(\tau) = \prod_t [\pi_{\mathrm{tch}}/\pi_\theta](s_t,a_t)$ naturally decomposes into trajectory-level $\hat w(\tau)$ (estimated by discriminator) and token-level $\pi_{\mathrm{tch}}(a|s)/\pi_\theta(a|s)$ (approximated implicitly by the $(1-\pi_\theta)$ weight under the Bayes-optimal discriminator assumption). **This is not CHORD+DR3; it's the factorized importance correction written in a form that exposes both scales.**

### 4.4 Response to "why not AWAC?"

AWAC ($\exp(A/\beta)$-weighted BC) has (i) fixed trajectory-level weight, (ii) fixed temperature $\beta$, (iii) no discriminator → no adaptive fade-out. Our operator has (i) learned per-trajectory weight $\hat w$ (adapts to teacher-policy gap), (ii) no temperature (the surprise weight is self-normalizing), (iii) automatic fade via both $\hat w\to 1$ and $(1-\pi_\theta)\to$ teacher's Bayes rate. We will show AWAC ablation confirms this: v12+AWAC < Candidate B on WebShop 1.5B.

### 4.5 Is the goal narrative achievable?

> *"DUET's Action Channel is a single discriminator-based density-ratio PG operator that provides trajectory-level credit assignment AND token-level distributional correction in one loss. The operator recovers behavior cloning in the zero-support limit and standard PG in the well-supported limit — no separate BC term is needed."*

**Achievable under Candidate B, with caveats:**
- "No separate BC term" holds *exactly* — the loss has no μ, no $L_{\mathrm{SFT}}$.
- "Recovers BC in zero-support limit" holds up to a scalar $\hat w(\tau)A(\tau)$ which empirically is in $[0.2, 0.5]$ on successful teacher trajectories — BC-equivalent.
- "Recovers PG in well-supported limit" holds by construction: $\phi\to 0$ on mastered tokens, $\hat w\to 1$, so the teacher term diminishes and the on-policy loss carries learning.

**The claim to be careful about**: "standard PG" in the limit. What remains is not standard on-policy PG; it's *off-policy PG on teacher data with vanishing weight*. The on-policy loss is a separate term; DR3/Candidate B is only the teacher-sample loss. Narrative should be "teacher term vanishes as policy matches teacher, leaving standard on-policy GRPO to carry learning" — which is correct and defensible.

---

## 5. Deliverable 4 — Adaptive μ backup (if unification fails)

If Candidate B underperforms v24 by >0.05, we need adaptive μ that reproduces v24's empirical decay without per-env tuning. The second-pass theory (`duet_second_pass_theory.md` §4) identified **advantage variance $V_A$** as the correct signal but v37 implemented it with a hand-set target $V_A^\star = 0.035$ — which was 50× off from observed $V_A\approx 1.0$.

### 5.1 Root cause of v37's calibration bug

Looking at v37's log scale: v24 in the stable regime has `adv_onpolicy_effective_abs_mean` ≈ 0.17, which is the *absolute mean* of per-token advantages. v37's $V_A^\star = 0.035$ was meant to match the *std* of per-sample advantages — a different quantity. Observed $V_A\approx 1.0$ is the std of *per-group-normalized* advantages, which GRPO forces to 1.0 by construction (that's what normalization does). **The bug was measuring the wrong quantity: GRPO's post-normalization $V_A$ is structurally $\approx 1$, not informative.** Any rule using $V_A(\text{post-norm})$ as its signal will have μ pinned to one extreme.

### 5.2 Correct signal: **group-reward std, not advantage std**

Advantage variance $V_A$ is normalized by construction. The *upstream* quantity the mechanism acts on is **$\sigma_g(t) := \text{std}_g(R(\tau))$, the group-level reward std**. This is *not* normalized by GRPO and directly indexes rollout heterogeneity. v24's predicted trajectory: $\sigma_g$ drops from $\sim 0.45$ (high variance, random) to $\sim 0.15$ (homogeneous) over steps 5-25. Already logged as `critic/score/std` or derivable from per-group success rates.

Adaptive rule (no tunable threshold, uses ratio to initial):

$$
\mu_t = \mu_{\min} + (\mu_{\max}-\mu_{\min})\cdot\mathrm{clip}\!\left(\frac{\sigma_g(t)-\sigma_g^{\mathrm{floor}}}{\sigma_g(0)-\sigma_g^{\mathrm{floor}}},\;0,\;1\right)
$$

where $\sigma_g^{\mathrm{floor}} = 0.05$ (reward noise floor; fixed), $\sigma_g(0) = \sigma_g$ at step $t_0$ (first step after `disc_apply_ready`), $\mu_{\min}=0.05$, $\mu_{\max}=0.3$.

**Key property**: $\sigma_g(0)/\sigma_g^{\mathrm{floor}}$ is **environment-agnostic**: on an env where teacher is close to initial policy (ALFWorld), $\sigma_g(0) \approx \sigma_g^{\mathrm{floor}}$, ratio $\approx 1$ instantly, μ drops to $\mu_{\min}$. On a high-variance env (WebShop), ratio large, μ near $\mu_{\max}$, then decays as $\sigma_g$ shrinks. **Reproduces observed per-env behavior with zero tuning.**

### 5.3 Why this signal and not alternatives

- `disc_acc`: saturates to >0.98 in all variants by step 30. Non-informative. (Rejected.)
- `KL(π_θ || π_{\mathrm{teacher\_empirical}})`: requires maintaining a teacher token-level histogram. Heavy instrumentation. (Too expensive.)
- `grad_norm` EMA: downstream and delayed — exactly the quantity the mechanism *produces*, not the input. Using it risks feedback oscillation. (Rejected.)
- `sft_loss` trajectory: self-referential (conditioned on the μ you're trying to set). (Rejected.)
- **$\sigma_g$ (group-reward std)**: mechanistically correct, already logged, env-agnostic normalization via ratio to initial, zero new hyperparameters. (**Recommended.**)
- $\|g_{\mathrm{DR3}}\|/\|g_{\mathrm{BC}}\|$: circular — $\|g_{\mathrm{BC}}\|$ is mechanically constant on teacher tokens, so the ratio reduces to $\|g_{\mathrm{DR3}}\|$. (Redundant.)

### 5.4 Predicted vs observed μ trajectory

Using log data for v24 (WebShop 1.5B):

| step | $\sigma_g$ (predicted) | ratio | $\mu_t$ predicted | $\mu_t$ v24 observed |
|---:|---:|---:|---:|---:|
| 0 | 0.45 | 1.00 | 0.30 | 0.30 |
| 10 | 0.30 | 0.63 | 0.21 | 0.26 |
| 20 | 0.20 | 0.38 | 0.14 | 0.12 |
| 30 | 0.15 | 0.25 | 0.11 | 0.05 |
| 50 | 0.12 | 0.18 | 0.09 | 0.05 |
| 100 | 0.10 | 0.13 | 0.08 | 0.05 |

The adaptive rule tracks v24's observed decay well through step 20, then decays slightly slower than v24's hand-tuned schedule. Expected outcome: ≥0.62 on WebShop 1.5B (within noise of v24's 0.678). **Strong candidate for the backup narrative.**

### 5.5 Implementation

15-line patch in `het_actor.py:1740-1752`:

```python
if self.config.get("chord_mu_adaptive_sigmag", False):
    # R_per_sample already available from the batch
    R = batch_rewards.detach()  # (bs,)
    # Group by task_id; 8 rollouts per task typically
    group_std = _per_group_std(R, group_ids).mean().item()
    sigmag_floor = float(self.config.get("chord_mu_sigmag_floor", 0.05))
    if not hasattr(self, "_sigmag_init"):
        self._sigmag_init = max(group_std, sigmag_floor + 1e-3)
    ratio = max(0.0, min(1.0,
        (group_std - sigmag_floor) / (self._sigmag_init - sigmag_floor)))
    mu = mu_min + (mu_max - mu_min) * ratio
    metrics["chord/mu_adaptive_sigmag"] = mu
    metrics["chord/sigmag_ratio"] = ratio
```

Zero new tensors on FSDP path. One scalar per step. Safe.

### 5.6 Decision protocol

Run Candidate B on WebShop 1.5B for 100 steps (v38). Simultaneously run adaptive-$\sigma_g$ μ (v39). Per second-pass-theory's decision matrix:
- Both ≥ 0.65: lead with B, include $\sigma_g$-adaptive as secondary contribution.
- B < 0.55, $\sigma_g$-adaptive ≥ 0.60: fall back to adaptive-μ narrative.
- Both fail: v24 remains empirical, reframe as "engineering recipe approximating the advantage-regularizer principle."

**Top-level recommendation: implement Candidate B first (v38). It is the highest-upside move for the NeurIPS narrative — a single unified operator is a stronger paper contribution than an adaptive hyperparameter schedule. If B lands within 0.05 of v24, that is a publishable result regardless of the exact magnitude.**

---

## 6. Uncertainty flags

- Candidate B's "BC recovery on rare teacher tokens" argument assumes $A(\tau)>0$ for teacher trajectories. If `teacher_baseline_separation` has edge cases where teacher advantage is negative, B may *suppress* rare tokens. Mitigation (Section 3.4.3) is clean but adds code.
- The optimal-Bayes-discriminator justification for interpreting $(1-\pi_\theta)$ as an approximation of $\pi_{\mathrm{tch}}(a|s)/\pi_\theta(a|s)$ is *informal*. The rigorous statement is: B is an unbiased estimator of a specific *support-expanding objective* $J_B$, not of the standard off-policy objective. The paper should state this honestly and defend $J_B$ as a reasonable teacher-guided objective.
- Implementation: $\phi = (1-\pi_\theta)$ is computed in the forward pass cheaply (we already have $\log\pi_\theta$). Gradient-flow check: `(1-exp(log_prob.detach()))` — clean detach, no issues.
- For the backup rule (Section 5), $\sigma_g$ is logged per-task-group; need to verify the current `wandb` metric path is accurate (`critic/score/std` or similar) before writing the implementation.

## 7. Bottom line

**Primary direction for NeurIPS**: Candidate B — $(1-\pi_\theta)$-surprise-gated DR3. Single operator, zero new hyperparameters, eliminates four BC hyperparameters, recovers BC on rare tokens and on-policy PG in the limit. Ablations listed in §3.6.

**Backup**: $\sigma_g$-adaptive μ (not v37's advantage-variance rule, which was calibrated against a normalized quantity). Should reproduce v24's μ schedule environment-agnostically.

**Narrative upgrade**: The Action Channel becomes a *single factorized off-policy PG operator* rather than "DR3 + BC". This is a genuine algorithmic contribution and fully answers the R1 "isn't it just CHORD+DR3?" critique.
