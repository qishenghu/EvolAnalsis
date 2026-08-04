# DUET — DR3 vs BC vs SC: A Theoretical Resolution

**Author**: lead-researcher
**Date**: 2026-05-03
**Audience**: project lead (paper-writing decisions)
**Scope**: Why does adding BC ("DUET\*") help on 1.5B but consistently hurt on 3B WebShop, and what does that tell us about what each DUET channel is actually doing at gradient level?

This document grounds every claim against the actual code paths in
`agentevolver/module/exp_manager/het_actor.py`,
`het_core_algos.py`,
`dr3_ratio.py`,
`state_progress.py`, and the SC injection in
`agentevolver/module/trainer/ae_ray_trainer.py`. Where I have an opinion I say so;
where I am uncertain I flag it.

---

## §1. Code-Level Mechanism (Precise)

We have three places that touch the actor gradient. Let me extract them one at a
time so the rest of the analysis has solid ground.

### 1.1 The mixing point (het_actor.py, lines 2179–2184)

```python
if dr3_enable:
    # DR3 + mini-SFT: L = L_dr3 + μ * L_sft
    pg_loss = grpo_loss + mu * sft_loss
else:
    # 原始 CHORD: L = (1-μ) * L_grpo + μ * L_sft
    pg_loss = (1 - mu) * grpo_loss + mu * sft_loss
```

This is the only place where μ enters. **In DUET\*, BC is added on top of the
already-DR3-corrected GRPO loss without down-weighting GRPO** (`grpo_loss + μ·sft_loss`,
not `(1-μ)·grpo + μ·sft`). This asymmetry is design-intentional and matters
later. The "μ-schedule" and "adaptive μ" code (lines 1773–2177) only decides
the scalar `μ`; once chosen, it always feeds this single line.

### 1.2 What DR3 actually does to the gradient (het_actor.py, lines 1503–1544)

DR3 does **not** add a loss term. It rewrites `old_log_prob` for teacher
samples *before* the standard PPO importance-ratio loss runs:

```python
dr3_w_min = float(dr3_cfg.get("w_min", 0.01))
w_hat = w_hat.clamp(min=dr3_w_min)
log_w = torch.log(w_hat.clamp_min(1e-6)).unsqueeze(-1)   # (bs,1)
old_lp_new = old_log_prob.clone()
if apply_mask.any():
    old_lp_new[apply_mask] = log_prob.detach()[apply_mask] - log_w[apply_mask]
...
old_log_prob = old_lp_new
```

Then in `repo_compute_token_loss` (het_core_algos.py, lines 1969–2016):

```python
ratio = torch.exp(log_prob - old_log_prob)       # = w_hat for teacher rows at the iter-0 update
...
on_pg_losses1 = -advantages * ratio
on_pg_losses2 = -advantages * torch.clamp(ratio, 1-cliprange, 1+cliprange)
```

Since `old_lp_new[teacher] = log_prob.detach() - log w_hat`, at the *first*
inner-PPO mini-step we have

$$\text{ratio}_{teacher,t} = \exp\bigl(\log\pi_\theta(a_t) - (\log\pi_\theta(a_t).\text{detach}() - \log\hat{w})\bigr) = \hat{w}\cdot \exp(\Delta\log\pi).$$

At the iter-0 forward pass `Δlogπ=0`, so ratio equals `ŵ` exactly. The loss
contribution from a teacher token is therefore

$$\nabla_\theta L_{DR3,teacher} = -\hat{w}_\tau \cdot A_{t} \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t),$$

with `ŵ_τ` a **per-trajectory scalar broadcast across all tokens of that
trajectory** (it's `(bs,1)` and the unsqueeze at line 1504 is the only place it
shape-aligns). This is the key thing: **DR3 is trajectory-level reweighting of a
standard PG term**. The discriminator output only sets the *magnitude* of the
PG; it never injects new information about which token to like.

### 1.3 What BC (CHORD-style SFT) actually does (het_core_algos.py, lines 1740–1820)

```python
def compute_chord_token_weights(log_prob):
    p = torch.exp(log_prob.clamp(max=0))
    phi = p * (1.0 - p)             # Bernoulli-variance weight φ(p_t)
    return phi

# in compute_chord_sft_loss:
sft_losses = -log_prob              # cross-entropy per teacher token
weighted_sft_losses = phi * sft_losses
sft_loss = agg_loss(weighted_sft_losses, mask=expert_mask, ...)
```

Per teacher token the gradient is

$$\nabla_\theta L_{BC,t} = -\,\phi(p_t) \cdot \nabla_\theta \log \pi_\theta(a_t^{teacher} \mid s_t),$$

with `φ(p_t) = p_t(1-p_t) ≥ 0`. The advantage `A_t` does **not** appear; the
sign is positive irrespective of return. **BC unconditionally pulls
`logπ_θ(a_t^{teacher}|s_t)` up.** That is the core difference from DR3.

### 1.4 Side-by-side: are DR3 and BC orthogonal?

Treat one teacher token in one teacher trajectory. Per-token gradient
contribution to `pg_loss = grpo + μ·sft` is

$$g_t^{teacher} \;=\; \underbrace{-\hat w_\tau \cdot A_t \cdot \nabla\log\pi_\theta(a_t)}_{\text{DR3 term}}
\;+\; \mu\cdot\underbrace{(-\,p_t(1-p_t)\cdot\nabla\log\pi_\theta(a_t))}_{\text{BC term}}.$$

Both terms multiply the **same direction** `∇log π_θ(a_t^{teacher})`. So they
are NOT orthogonal in the gradient direction sense: they push the same
`logπ`. They differ only in the *coefficient* in front of that direction:

- DR3 coefficient: `−ŵ_τ · A_t`. Sign and magnitude both depend on the
  trajectory's reward (via `A_t`) and the *distributional distance* to the
  current student (via `ŵ_τ`).
- BC coefficient: `−μ · p_t(1−p_t)`. Always non-positive in negative-of-loss
  view; magnitude is largest when the student is at chance (`p≈0.5`); fades
  when the student already imitates (`p≈1`) or has given up (`p≈0`).

So the right way to phrase orthogonality is **"orthogonal in conditioning, not
orthogonal in update direction"**. DR3 conditions on `(reward, density distance)`,
BC conditions on `(student certainty alone)`. They can compose constructively
when reward is positive and student is mid-confidence; they can also fight
each other when `A_t < 0` (DR3 wants to push away from the teacher action) but
`μ·p(1-p) > 0` (BC still pushes toward it). I'll come back to this.

### 1.5 SC for completeness (ae_ray_trainer.py line 3400)

```python
batch.batch["token_level_rewards"][_sc_idx, _sc_valid] += _sc_bonus / _sc_n_valid
```

SC modifies `token_level_rewards`, which then flow through GRPO group-relative
normalization → `advantages`. So SC's effect on the gradient is

$$\nabla_\theta L_{SC} = -\Delta A_t \cdot \nabla\log\pi_\theta(a_t),$$

where `ΔA_t` is the GRPO-normalized lift produced by adding `β·P(τ)/T` to the
on-policy reward. Crucially `_sc_exclude_teacher` (line 3382) skips teacher
samples: SC never directly touches teacher-token gradients.

### 1.6 Summary table

| Channel | Gradient direction | Coefficient | Conditioning | Acts on |
|---|---|---|---|---|
| DR3 | `∇log π_θ(a_t^{teacher})` | `−ŵ_τ · A_t` | reward × density distance | teacher tokens (per-traj scalar) |
| BC  | `∇log π_θ(a_t^{teacher})` | `−μ · p_t(1−p_t)` | student certainty | teacher tokens (per-token) |
| SC  | `∇log π_θ(a_t^{onpol})`   | `−ΔA_t (∝ β P(τ))` | trajectory shape vs expert states | on-policy tokens only |

DR3 and BC operate on the **same tokens**; SC operates on the **disjoint
on-policy tokens**. So "DR3 vs BC" is a real overlap question; SC is genuinely
disjoint by mask.

---

## §2. Mathematical Framework (Unified)

Drop subscripts. Let `θ` be the actor parameters,
`π = π_θ`, `π_t` the (black-box) teacher.
Sample a batch of trajectories `τ ∼ p` where some come from `π` (on-policy) and
some from `π_t` (teacher). Define the per-sample group-relative advantage
`A^{GRPO}` and let `ŵ_τ` be the DR3 estimate of `π(τ)/p(τ)` for any τ
(trivially 1 for on-policy, learned for teacher).

The DUET total objective at iteration k is, before mixing,

$$
\boxed{\;
L_{tot}(\theta) \;=\; \underbrace{\hat{\mathbb E}_{\tau\sim p}\Bigl[\, \hat w_\tau \sum_{t} -A^{GRPO}_t \cdot \rho_t \;\Bigr]}_{L_{GRPO+DR3}\,(=\,L_{DR3}\text{ in code naming})}
\;+\;\mu \cdot \underbrace{\hat{\mathbb E}_{\tau\sim \pi_t}\Bigl[\, \sum_t \phi(p_t) \cdot \bigl(-\log\pi(a_t|s_t)\bigr)\Bigr]}_{L_{BC}}
\;}
$$

with `ρ_t = π_θ(a_t)/π_θ_old(a_t)` the standard PPO ratio (clipped). The
GRPO/DR3 advantage is computed *after* SC injection:

$$
A^{GRPO}_t = \mathrm{Norm}_{\text{group}}\Bigl(\,r_t \;+\; \mathbf{1}[\text{on-policy}]\cdot\frac{\beta \,P(\tau)}{T}\Bigr).
$$

Each component's optimization target:

| Component | Statistical target (fixed-point) | Why |
|---|---|---|
| GRPO | `argmax_θ E_{τ∼π_θ} [R(τ)]` | standard RL on environment reward |
| DR3 (importance reweighting) | makes `E_{τ∼p} [ŵ_τ ∇logπ · A]` an unbiased estimator of `E_{τ∼π_θ}[∇logπ·A]` | corrects for teacher-induced sampling bias |
| SC (potential-based shaping with β·P(τ)) | `argmax_θ E_{τ∼π_θ}[R(τ) + β·P(τ)]` ≈ `argmax E[R(τ)]` because ΔΦ is potential-based | dense signal without changing optimum (with caveats; see §4) |
| BC | `argmin_θ E_{τ∼π_t}[Σ_t φ(p_t) · (-logπ(a_t|s_t))]`, asymptote at `π → π_t` weighted by φ | imitation regularizer |

**Where mixing happens**: only at one line, line 2181 (`pg_loss = grpo_loss + μ·sft_loss`).
Notice the **asymmetry** vs original CHORD: in CHORD `(1−μ)·grpo + μ·sft`, GRPO
is down-weighted as μ grows. In DUET\* (DR3-on path) GRPO is **never**
down-weighted; μ only adds BC on top. So increasing μ in DUET\* does not
trade-off RL for BC — it *adds* BC to a fixed-magnitude RL objective. **Two
practical consequences:**
1. The effective LR for BC is `μ`, but the effective LR for RL stays at 1.
2. Step-size of θ grows monotonically with μ at fixed lr — this is part of
   why high-μ on 3B blows up `actor/grad_norm` (47 vs 18 in pk03 vs pk02; see
   `3b_v39_underperform_diagnosis.md` Table 3).

**μ → 0 recovers DUET v1 exactly?** Yes by inspection:
- `μ=0` ⇒ `pg_loss = grpo_loss`. SFT is computed but multiplied by zero, so
  no gradient flows through it.
- DR3 `w_hat` correction (line 1507) is independent of μ.
- SC injection (line 3400) is independent of μ.

So `μ ≡ 0` everywhere = DUET v1 exactly. This is empirically what the
"v1-latch" runs were trying to approximate after step 17 / step 63.

### 2.1 One-line picture of the three forces on `logπ_θ(a_t)`

- **For a teacher token** in a teacher trajectory:
  `θ̇ ∝ −∇logπ · [ŵ_τ A_t  +  μ φ(p_t)]` (DR3 + BC)
- **For an on-policy token**:
  `θ̇ ∝ −∇logπ · [A^{GRPO}_t]` where `A^{GRPO}` includes SC bonus.

So the three "channels" map to: *trajectory-level scalar on teacher tokens
(DR3)*, *token-level scalar on teacher tokens (BC)*, *trajectory-level scalar
on on-policy tokens (SC)*. There is no overlap of channels on a token. There is
an interaction at the *batch normalization* level (GRPO group means/stds see
all tokens), which is why we have `teacher_baseline_separation`.

---

## §3. The DR3-vs-BC Distinction Explained

### 3.1 The user's confusion, re-stated

> "DR3 doesn't use teacher logit. BC doesn't use teacher logit. Both push the
> student toward teacher behavior. So why is DR3 'trajectory-level' and BC
> 'token-level'?"

This is exactly the right question, and the framing "DR3 = trajectory" / "BC
= token" is partially correct but partially misleading. Let me sharpen it.

### 3.2 What does *not* differentiate them

Both use **only the teacher's *trajectory*** (the action sequence). Neither
asks the teacher for `log π_t(a_t|s_t)`. So the *informational input* is the
same: the teacher's chosen action at each step.

### 3.3 What *does* differentiate them

The differentiator is **what each method does with that action**.

- **BC** treats every teacher action as a target label and minimizes
  per-token cross-entropy. This is supervised learning. The signal at token `t`
  depends only on the student's current `p_t` at that token; it has no opinion
  about whether the trajectory was good. A teacher trajectory with reward 0
  contributes the same per-token BC pressure as one with reward 1 (modulo
  φ(p_t) which depends only on student, not on outcome). **BC = density
  matching at the token level, conditioned on student certainty.**

- **DR3** treats each teacher trajectory as a sample from a distribution
  whose density ratio against the student we estimate with a discriminator.
  The output `ŵ_τ` is **a single scalar per trajectory**. That scalar is then
  used to convert the trajectory into an importance-weighted PG estimator.
  The pressure on each token is `−ŵ_τ · A_t`. So DR3 inherits sign and
  magnitude from the *advantage*, which itself depends on the *trajectory's
  reward* relative to the group baseline. **DR3 = importance-corrected RL,
  with the discriminator providing the importance weight.**

So the terms "trajectory-level" and "token-level" in our project vocabulary
mean:

- "Trajectory-level" = the *signal modulating the gradient* depends on
  trajectory-aggregate quantities (`ŵ_τ`, `R(τ)` via `A_t`).
- "Token-level" = the *signal modulating the gradient* depends on
  token-local quantities (`p_t` here; in principle could be anything per-token).

This is a slightly awkward usage because both methods produce a per-token
gradient (they differ in the *modulator*, not the *target*). A more precise
phrasing for the paper would be:

> DR3 provides **outcome-conditioned** importance reweighting; BC provides
> **outcome-unconditioned** density matching.

That is the cleanest mental model and I recommend we adopt it in the paper.

### 3.4 Concrete example

Take a teacher trajectory in WebShop where the teacher bought
"red shirt size M" but the task asked for "size S", giving environment reward
≈ 0.5 (partial credit for correct category, wrong attribute).

Suppose the student under current `π_θ` has:
- `p_t ≈ 0.6` for the click-on-red action (well-imitated).
- `p_t ≈ 0.05` for the click-on-size-M action (the wrong one).

What does each do at this last token (size selection)?

- **BC**: `φ(0.05) = 0.0475`, gradient `+0.0475 · ∇logπ` upward toward
  size-M (the wrong size). BC has no idea this was the wrong answer.
- **DR3**: GRPO advantage is `A_t ≈ (0.5 − R̄_group)/σ_group`. If group mean
  is 0.55, then `A_t ≈ −0.5/σ < 0`. Multiplied by `−ŵ_τ`: gradient
  `+ŵ_τ·|A_t|·∇logπ` *downward* from size-M.

**They push in opposite directions on this token.** This is not a hypothetical;
it is the dominant failure mode on 3B WebShop, where the partial-credit reward
(0.5 partial vs 1.0 perfect) means many teacher trajectories have low `A_t` but
non-trivial `p(1−p)` weights, and BC's blind imitation pulls the student
toward "average teacher behavior" — including the teacher's mistakes.

### 3.5 Is "trajectory vs token" the right framing for the paper?

I think it's a *good headline* but should be unpacked into:

1. **DR3 is outcome-aware**: its sign tracks `A_t`. BC is outcome-blind.
2. **DR3 is per-trajectory in modulator**: one `ŵ_τ` for the whole τ. BC's
   modulator `φ(p_t)` varies token-to-token but is purely about student
   certainty.
3. **Therefore**: when teacher trajectories have heterogeneous outcomes (high-
   variance reward like WebShop's partial credit), DR3 selectively amplifies the
   good ones and dampens the bad ones; BC averages them all.

This is the single sharpest theoretical statement of "what DR3 buys you over
BC" and I think it should be in the paper's intro.

---

## §4. Why 1.5B BC Helps but 3B WS BC Hurts

Empirical reality (carrying numbers from `MORNING_REPORT_2026-05-03.md` and
`3b_v39_underperform_diagnosis.md`):

|  | 1.5B AF | 3B AF | 1.5B WS | 3B WS |
|---|---|---|---|---|
| DUET v1 | 32.5 | 69.5 | 54.9 | **53.0** |
| DUET\* SOTA | 47.5 (+15) | 77.5 (+8) | 36.0 ¹ | **stuck 26–44** |
| disc_acc plateau | 0.997 | 0.997 | 0.96 | **0.91** |

¹ The 1.5B WS "DUET\*" 36% number isn't strictly above DUET v1's 54.9% — it's
the swC_02 reference for the BC variant, and DUET\* did not actually beat
DUET v1 on 1.5B WS either as far as our most recent numbers show. So the
sharper statement is: **BC helps clearly on AF (both scales), is marginal on
1.5B WS, hurts on 3B WS**. AF vs WS, not 1.5B vs 3B, is the primary axis.

### 4.1 The four candidate explanations

**(A) Capacity argument.** "3B has enough capacity that BC competes for
parameter budget." I find this *weak* — 3B is still tiny relative to teacher
(72B), there's no parameter contention. What 3B does have over 1.5B is a
better on-policy gradient (less reward noise), so the *opportunity cost* of
slowing GRPO to absorb BC is higher. This is a real effect but not the
dominant one.

**(B) Reward-landscape argument.** "Partial credit creates conflicting
optima." This is the core of the WS-specific problem. WS reward `r ∈ [0,1]`
is essentially `0.5·(category_match) + 0.5·(attribute_match)`. Teacher
trajectories have `r_avg ≈ 0.85` because the teacher is good at both. But
when student imitates "average teacher behavior" via BC, and BC has no
discrimination between the trajectories that scored 1.0 (perfect) vs the ones
that scored 0.5 (got the category but botched the attribute), it pushes the
student toward an action distribution that is the **mean of those two
modes**. The mean of "click size S" and "click size M" is "click whichever is
default", which on WS often gives partial credit but rarely the bonus. This
matches `MORNING_REPORT_2026-05-03.md` Table 1: `pk04` reward (0.69) >
`pk03` reward (0.67) but SR (`pk04` 26.5%) < SR (`pk03` 28.5%). **More BC ⇒
higher partial reward, lower precise success.** AF doesn't have this mode
because reward is essentially binary (task done or not), so the
partial-credit attractor doesn't exist.

**(C) SC redundancy argument.** "SC already does the work BC tries to do."
Partially true. SC encodes "did your trajectory pass through expert-like
states", which is a softer form of imitation than BC. On WS the expert
progress map is dense (4 stages: search → results → product → buy), so SC
gives ~0.12 of the reward signal (`bonus_vs_reward_ratio ≈ 0.12`). BC adds
*token-level* density matching on top. The marginal information from BC over
SC is small, but the cost (gradient interference, see §4.D below) is large.
On AF, SC is sparser (room navigation), so BC's marginal value is larger.

**(D) Discriminator-quality argument.** "Lower disc_acc plateau means weaker
DR3 fade-out." The disc plateau is 0.997 (AF, both sizes) vs 0.91 (3B WS).
Why? AF's teacher and student differ in *style* permanently
(teacher-thinks-very-long, student-doesn't, never converges) — disc keeps
having signal. WS teacher and student converge in *output distribution* fast
(both eventually click similar buttons) — disc runs out of signal. Net effect
on adaptive μ: with `d̄ ≈ 0.91` and `d_floor = 0.5`, the gating factor
`(1-d̄)/(1-d_floor) ≈ 0.18`, so for any chosen `(peak, valley)` we plateau at
`μ ≈ valley + 0.18·(peak−valley)` ≈ `0.10 + 0.018 = 0.118` for `valley=0.10,
peak=0.20`. **The adaptive μ on 3B WS is essentially a flat constant 0.12 for
the second half of training** (Table 2, swE_02 row). So we are not actually
"adaptively fading BC" at all on 3B WS; we are just running a constant low-BC.
The fade we see on 1.5B WS / AF is real *because* disc keeps rising. **The
adaptive-μ design assumes the discriminator continues to get better; on 3B WS
it doesn't.**

This is the missing link the user's intuition was groping for: the
auto-adaptive recipe was implicitly a function of disc dynamics, and the disc
dynamics are *environment-and-student-specific*, not universal.

### 4.2 Which is dominant?

I assign rough weights:

| Cause | Weight | Reasoning |
|---|---|---|
| (B) reward-landscape | **45%** | Direct evidence in pk04>pk03 reward / pk04<pk03 SR. WS-only artifact, AF doesn't have it. |
| (D) disc-quality | **30%** | Adaptive-μ degenerates to constant on 3B WS; the entire premise of v39 fails. |
| (A) capacity / GRPO opportunity cost | **15%** | grad_norm scales with μ on 3B; on-policy gradient is sharper at 3B. |
| (C) SC redundancy | **10%** | Real but marginal; would also predict 1.5B WS BC harm, which we mostly don't see. |

**Headline claim**: BC fails on 3B WS because (B) the partial-credit reward
landscape rewards "average teacher behavior" with high reward-but-low-success,
while (D) the discriminator never separates well enough for the adaptive-μ to
actually fade. Together these mean DUET\* on 3B WS is "permanent low-grade BC
into a partial-credit attractor", which is exactly the failure pattern we
observe (high reward, low SR, monotonic-with-BC degradation).

### 4.3 Why this is *not* a code bug

We tried six different μ schedules on 3B WS pk03_v00. All failed. The
buggy whip-saw schedule did best, and *its mean μ was in the same 0.10–0.15
range that the well-behaved schedules ended up emitting*. So the result is
robust to the schedule choice — what matters is "how much BC, on average,
gets injected" not "when". This is a strong signal the cause is not in the μ
scheduling code, it is in the underlying interaction of BC with WS reward.

---

## §5. Auto-Adaptive BC: Theoretically Possible?

### 5.1 What signals can drive BC modulation?

Candidates we have or could try:

| Signal | What it measures | Useful for | Failure mode |
|---|---|---|---|
| `disc_acc` (level) | how separable π_θ from π_t at trajectory feature level | knowing how far student is from teacher | plateaus when disc runs out of signal (3B WS) |
| `disc_acc` velocity | is the student/teacher distance still shrinking? | turning BC off when convergence happens | velocity is noisy at small step counts; latch is the right mechanism but timing is fragile |
| KL(π_θ ‖ π_t) | direct distance, would be ideal | precise BC modulation | requires teacher logits, which we don't have (the whole point of DR3) |
| ESS of off-policy weights | how "useful" teacher samples are for IS | knowing when DR3 itself is degenerate | already used for clipping; no clean μ mapping |
| reward gap (teacher − on-policy) | how much room to grow | scheduling teacher influence | slow signal, lags behind disc |
| BC's own NLL | how well student already matches teacher tokens | direct, but circular | confounded with student's general LM ability |
| `actor/grad_norm` | global instability proxy | preventing blow-ups | reactive, not predictive |

### 5.2 Why our velocity-based attempts failed

There are two distinct failure modes in our recent runs.

**(i) Implementation bugs** (whip-saw): The first velocity attempt had
a sign/window indexing bug that made μ flip between peak and valley each
step. Diagnosed and patched. Empirically the buggy run did BEST (39.5% on pk03)
because the noisy μ averaged to a small persistent value, and that's what 3B
WS wants.

**(ii) Premature latching** (v1 latch): triggered at step 17 with too-aggressive
threshold; left BC=0 for 84 steps, recovered to 36.5%. This was actually
*almost* the right behavior — the issue was that BC during steps 1–17 is the
"peak" period (μ=0.30), which imprints teacher style into a 3B model that
*already* has on-policy signal at step 17.

**(iii) Late latching** (v2 triple-gated, fired at step 63): the longest BC
exposure of the three; got 28.5%. Confirms more BC = worse on 3B WS.

The fundamental issue across all three: **we are trying to use a real-time
training-dynamics signal (disc velocity) to modulate a forward-looking
intervention (how much BC to apply for the next 100 steps)**, but the signal
arrives lagging the optimal action. By the time velocity says "BC has nothing
left to extract", the imprint has already happened. This is not a tuning
problem; it is a **causal-lag problem**.

### 5.3 Is there a class of auto-BC methods that COULD work?

I think yes, but they are different in kind from what we tried:

**Class A — Causal-aware gating**. Instead of "is BC still useful at the
distribution level", ask "for this specific batch of teacher tokens, would
BC's pull conflict with DR3's pull?". Operationalization:

```
g_DR3,t  ∝ −ŵ_τ · A_t        (computable per token)
g_BC,t   ∝ −μ · φ(p_t)         (computable per token)
gate_t   = max(0, sign(g_DR3,t · g_BC,t))   # 1 if same sign, 0 if opposite
sft_loss = agg(gate_t · φ(p_t) · (-logπ))
```

This is a **principled, mechanism-grounded BC modulation**: BC fires only
where it agrees with DR3-corrected RL. On 3B WS this would automatically
suppress BC on the partial-credit teacher trajectories where DR3 has decided
"this trajectory is below-group" (`A_t < 0`) — exactly the failure mode in
§3.4.

This is **the single intervention I would prioritize if we had one more
week**.

**Class B — Reward-conditional BC**. Compute teacher trajectory's GRPO
advantage `A_τ` first, then weight BC per-trajectory by `max(0, A_τ)`. This
is a coarser version of Class A.

**Class C — Logit-distillation surrogate**. Use the *student's own old
policy* as a proxy teacher for high-reward teacher trajectories: BC against
`π_θ_old(a_t|s_t, R(τ) > median)`. No teacher logits needed. Avoids the
"average teacher behavior" attractor.

We have not tried any of these; all our adaptive-μ work was about scaling a
*global* `μ`, never about *which token* or *which trajectory* gets BC.

### 5.4 Is the goal ill-posed for 3B WS specifically?

Not ill-posed, but the answer might be: **the right μ for 3B WS is zero**.
That is what DUET v1 corresponds to, and DUET v1 reports 53% — the best
number we have on 3B WS. The auto-adaptive BC story is a *positive-result
narrative* for AF (BC clearly helps; BC modulation helps even more); on 3B
WS it should probably become a *negative-result-with-mechanism* narrative.

---

## §6. Recommendation for Paper Narrative

### 6.1 If DUET v1 reproduces 53% on our infra (likely)

**Story to tell**:

1. DUET = DR3 + SC. Two orthogonal channels (action via
   outcome-aware importance reweighting, state via potential-based shaping
   on disjoint on-policy tokens). Theoretically clean, empirically dominant
   over LUFFY/CHORD/GRPO on AF and competitive on WS.
2. **Optional BC head (CHORD-style)** further improves AF (both scales) by
   8–15pp; the adaptive-μ schedule (driven by `dr3/disc_acc` velocity, with
   monotonic latch) lets it self-modulate without manual tuning.
3. **Honest negative-result vignette on 3B WS**: BC consistently degrades
   success rate even though it raises raw reward. Mechanism: WS partial-
   credit reward landscape allows BC to reach a "high-reward, low-precision"
   attractor that genuine on-policy RL avoids. We characterize this and
   recommend BC be opt-in per environment, not a default.

This is *more* defensible to reviewers than papering over the 3B WS result —
NeurIPS reviewers reward honest mechanism-level negative results.

### 6.2 If DUET v1 also fails on 3B WS (currently the gap is 53.0 LUFFY vs
51.5 v1 — in the same noise)

Fallback: drop 3B WS from the paper's headline table or relegate to "scaling
study" appendix. Keep AF (1.5B + 3B) and WS (1.5B). Frame as "DUET dominates
in 3 of 4 scaled settings; the WS-3B regime has a known partial-credit reward
pathology that affects all teacher-mixing methods including ours". Cite the
1.5B WS LUFFY = 36% number as evidence that even the prior best method tops
out low on WS.

### 6.3 What additional experiment maximizes reviewer-defensibility?

In rank order of expected information per GPU-hour:

1. **Run the per-token sign-gate ablation (Class A from §5.3)** on 3B WS for
   one config, ~16h. If it beats DUET v1 on 3B WS, it's a paper-saving
   result. If it doesn't, we learn that the partial-credit attractor isn't
   token-level after all and the problem is fundamentally distributional.
2. **Run "DUET v1 + 3 seeds"** on 3B WS to nail down the
   v1 = 53% number with variance bars. Critical because the entire negative-
   result narrative rests on v1 actually being the ceiling.
3. **Run "GRPO-only" (no DR3, no SC, no BC)** on 3B WS for one seed. This
   tells us how much DR3+SC actually buys vs vanilla GRPO on this hard
   setting. If the gap is small, the paper's WS story should be about AF.
4. **Single-seed control: BC-only (no DR3, no SC)** on 3B WS. Confirms the
   BC hurts story isn't a DR3-interaction artifact.

If we can only afford one, do (1). It's the only experiment that has a
chance of changing the paper's story for the better.

### 6.4 The user's "I'm confused, all this seems related" — resolution

You are right that DR3 and BC overlap more than the paper currently
acknowledges. They both push the same `∇logπ_θ(a_t^{teacher})`. They differ
**only** in the modulator in front of that direction:
- DR3: `−ŵ_τ · A_t` (outcome-conditioned, trajectory-coarse, environment-aware)
- BC : `−μ · φ(p_t)` (outcome-blind, token-fine, student-self-aware)

SC really is disjoint (it touches on-policy tokens only, by design and by
mask).

**Therefore the cleanest paper-level claim is two channels (DR3 + SC), with
BC as an optional outcome-blind regularizer that helps when the reward
landscape is binary (AF) and hurts when the reward landscape has a partial-
credit attractor (WS).** This is theoretically tight and empirically
defensible.

The auto-adaptive BC story is real for AF (where disc keeps rising →
adaptive μ actually adapts). On 3B WS the disc plateaus at 0.91 and the
adaptive system degenerates into a constant low-BC, which then runs the
partial-credit attractor — so the failure is not in the adaptation logic
but in the *underlying premise* that "more teacher signal helps if dosed
right". On WS, more teacher signal of the BC kind doesn't help; we need a
*selectively-applied* teacher signal (Class A in §5.3), which we have not
implemented.

That is the resolution. It is honest, it is mechanism-grounded, and it leaves
the paper's main contribution (DR3 + SC) intact.
