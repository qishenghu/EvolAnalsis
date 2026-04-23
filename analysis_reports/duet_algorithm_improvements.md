1. DUET Algorithm Improvements — Post-v24 Proposal Set

Basis:
- Winner = v24 = DR3 + SC + CHORD-style BC with linear-warm/cosine-decay μ (peak 0.3, valley 0.05, decay 25 steps). Val@100 = 0.678.
- Losers: all stability-only rescues (0.46-0.52), and v36 constant small-μ BC (0.389) < v12 no-BC (0.431).
- Theory diagnosis: BC is the ONLY per-token, teacher-specific, surprise-proportional, unconditionally positive gradient channel; it installs rare teacher tokens (click[option]) that w_hat-scaled trajectory PG cannot. Decay matters because constant BC suppresses discriminator training, entropy, and SC progress once tokens are already imprinted.

All line/file references are to the exact code paths:
- het_actor.py:1722-1759 (where `grpo_loss`, `sft_loss`, `mu` compose `pg_loss` when `use_chord=True`)
- het_core_algos.py:1627-1693 (`chord_mu_scheduler`)
- het_core_algos.py:1723-1818 (`compute_chord_sft_loss`)
- dr3_ratio.py:751 (disc_acc_val), 749-751 (w_hat via `_get_disc_for_inference()`)
- het_actor.py:1292, 1383, 1610 (`dr3_metrics` dict already available in the μ-selection scope)

--------------------------------------------------------------------------------

## Idea 1a — Discriminator-adaptive μ (score: high priority)

Formulation:
```
disc_acc_ema_t = 0.9 * disc_acc_ema_{t-1} + 0.1 * disc_acc_t
r_t           = clamp((1 - disc_acc_ema_t) / (1 - acc_target), 0, 1)
mu_t          = mu_valley + (mu_peak - mu_valley) * r_t
```
with `acc_target = 0.90` (target "teacher is installed"), `mu_peak=0.3`, `mu_valley=0.05`.

Fit to theory: Yes. disc_acc ≈ 0.5 means student and teacher are indistinguishable → rare tokens already imprinted → BC pressure should drop. disc_acc ≈ 1.0 means student is visibly different → BC pressure high. This matches the "install, then release" narrative precisely.

Implementation sketch (het_actor.py around line 1740-1752):
```python
# Replace fixed-schedule mu with disc_acc-gated mu
disc_acc_val = float(dr3_metrics.get("dr3/disc_acc", 0.0)) if isinstance(dr3_metrics, dict) else 0.0
if not hasattr(self, "_disc_acc_ema"):
    self._disc_acc_ema = disc_acc_val
else:
    beta = 0.9
    self._disc_acc_ema = beta * self._disc_acc_ema + (1 - beta) * disc_acc_val
acc_target = self.config.get("chord_mu_acc_target", 0.90)
mu_range = chord_mu_peak - chord_mu_valley
r = max(0.0, min(1.0, (1.0 - self._disc_acc_ema) / max(1e-3, 1.0 - acc_target)))
mu = chord_mu_valley + mu_range * r
metrics["chord/mu_mode"] = 1.0  # adaptive
metrics["chord/disc_acc_ema"] = self._disc_acc_ema
```
Add a config flag `chord_mu_mode: {"schedule","disc_acc","w_hat"}` defaulting to `schedule` for back-compat.

Caveats:
- disc_acc is non-monotone early (0 until buffer hits `disc_train_min_buf_size=256`). Need to gate `r=1` (full BC pressure) during the pre-training-ready window — easily done by checking `dr3_apply_ready` from het_actor.py:1384.
- acc_target=0.90 is aggressive; EMA ensures μ doesn't flicker if disc oscillates.

Likelihood of beating v24: **likely matches, might beat**. Removes the 2 hyperparameters (decay_steps, warmup_steps) that vary per env/size, and auto-adapts to larger models where disc trains faster.

Implementation cost: ~20 LOC in het_actor.py, 5 LOC config. Risk: low — falls back to schedule if `chord_mu_mode="schedule"`.

Narrative benefit: Strong. One story: "BC fades as discriminator becomes confident." Eliminates 2 hyperparameters.

--------------------------------------------------------------------------------

## Idea 1b — w_hat-adaptive μ (score: medium priority)

Formulation:
```
wbar_t  = EMA_0.9( median(w_hat_t[teacher_mask]) )
mu_t    = mu_valley + (mu_peak - mu_valley) * max(0, 1 - wbar_t)
```

Fit: softer than 1a because w_hat is a sequence-level statistic and noisier (EMA essential). But uniquely attractive: `w_hat → 1` IS the formal condition for "policy ≈ teacher on the feature manifold." That's the cleanest possible story for BC fade.

Concern: w_hat has a hard floor at `dr3_w_min=0.01` (het_actor.py:1495) but no natural upper fade. In practice median(w_hat) drops from ~0.5 early to ~0.15 late (per v24 logs); it doesn't cross 1.0 cleanly. So `1 - wbar` would be ~0.85 throughout, giving ~constant μ close to peak. That's exactly v36's failure mode.

Correction: use a different transfer function:
```
mu_t = mu_peak * (median(w_hat_t) / wbar_init)
```
where `wbar_init = median(w_hat)` recorded at step 10 (first step DR3 applies). Ratio ≤ 1 throughout; as teacher influence fades (w_hat shrinks), μ fades proportionally. But this is fragile: `wbar_init` is a single measurement.

Implementation sketch (het_actor.py, same location as 1a):
```python
# compute per-sample median w_hat over teacher samples
if w_hat is not None and teacher_sample.any():
    wh_teacher = w_hat[teacher_sample].detach()
    wh_med = float(wh_teacher.median().item()) if wh_teacher.numel() else 1.0
    if not hasattr(self, "_wbar_ema"):
        self._wbar_ema = wh_med
        self._wbar_init = wh_med
    else:
        self._wbar_ema = 0.9 * self._wbar_ema + 0.1 * wh_med
    r = max(0.0, min(1.0, self._wbar_ema / max(1e-3, self._wbar_init)))
    mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * r
```

Likelihood of beating v24: **might match**. Story is cleaner than 1a but noisier signal and wbar_init is brittle.

Implementation cost: ~25 LOC. Risk: medium (wbar_init sensitive to step 10 conditions).

Narrative benefit: High (uses an existing DR3 quantity) but requires explaining why we use median(w_hat) not E[w_hat].

--------------------------------------------------------------------------------

## Idea 2a — Surprise-weighted Teacher PG (single operator; high novelty)

Formulation. For teacher tokens only, replace `(1 - mu) * grpo_loss + mu * sft_loss` with:
```
alpha_t(s_t, a_t) = sigmoid( (-log pi_theta(a_t|s_t) - H_bar) / T_alpha )
L_teacher = - E_{(s,a) ~ teacher}[
    alpha_t * log pi_theta(a_t|s_t)                           # BC-like on high-surprise tokens
  + (1 - alpha_t) * w_hat * A_hat * clip(pi_theta/pi_theta_old, 1-eps, 1+eps)  # DR3-like PG on familiar tokens
]
```
where `H_bar` is an EMA of response-token entropy (or a fixed nominal like 2.0 nats) and `T_alpha ≈ 1.0`.

Properties:
- Unit gradient on rare teacher tokens (they get α≈1, pure BC)
- Advantage-weighted PG on learned tokens (α≈0, pure DR3)
- No explicit BC coefficient μ; alpha is data-dependent
- Single operator, clean narrative: "surprise-weighted teacher imitation"

Implementation sketch:
- New function in het_core_algos.py:
```python
def compute_surprise_weighted_teacher_loss(
    log_prob, old_log_prob, advantages, teacher_mask,
    w_hat, cliprange, H_bar_ema, T_alpha=1.0, loss_agg_mode="token-mean",
):
    surprise = -log_prob.detach()   # (bs, resp_len)
    alpha = torch.sigmoid((surprise - H_bar_ema) / T_alpha).clamp(0.05, 0.95)
    bc_term = - log_prob                                  # (bs, resp_len)
    ratio = torch.exp(log_prob - old_log_prob)
    ratio = ratio.clamp(1 - cliprange, 1 + cliprange)
    pg_term = - w_hat.unsqueeze(-1) * advantages * ratio  # (bs, resp_len)
    loss_mat = alpha * bc_term + (1 - alpha) * pg_term
    loss = agg_loss(loss_mat, teacher_mask, loss_agg_mode)
    return {"pg_loss": loss, "alpha_mean": alpha.mean()}
```
- Call site: het_actor.py:1754-1759 would become a branch on `use_spw_teacher` replacing both grpo_loss for teacher and sft_loss.

Fit to theory: exactly captures the per-token surprise modulation story. When pi_theta already assigns high probability to a token, surprise is low → α≈0 → token gets advantage-weighted PG (learned; don't over-push). When token is rare under pi_theta, surprise is high → α≈1 → pure BC (install it).

Likelihood of beating v24: **likely matches, plausibly beats**. Removes μ entirely and the separate BC/DR3 machinery for teacher samples. Risk: the bell-shaped sigmoid may mis-cut when response has many reasoning tokens (long CoT); `H_bar` calibration matters.

Implementation cost: ~60 LOC (new loss fn + dispatch in het_actor.py). Risk: medium (new loss, needs careful gradient-flow validation, need H_bar EMA bookkeeping).

Narrative benefit: **Major**. The paper can now say "DR3+SC, no BC" and describe the teacher channel as a single surprise-gated operator. Eliminates 4 hyperparameters (mu_warmup, mu_decay, mu_peak, mu_valley) at the cost of 2 (H_bar, T_alpha), both of which can be EMA'd.

--------------------------------------------------------------------------------

## Idea 3a — Handoff to DR3 activation (simple schedule rationale)

Formulation:
```
mu_t = mu_peak                     for t < apply_warmup_steps
mu_t = mu_peak * exp(-(t - apply_warmup_steps)/tau)   otherwise
```
with `tau = apply_warmup_steps` (default 10), so μ halves every 10 steps after DR3 comes online.

Fit: v24 empirically has μ decay finishing at step ~25 and DR3 apply_warmup at step 10. This formalization makes the handoff the principle. No new free parameters if tau=apply_warmup_steps.

Implementation sketch (tiny, change in chord_mu_scheduler or a new `chord_mu_handoff_scheduler`):
```python
def chord_mu_handoff_scheduler(global_step, dr3_apply_warmup_steps, mu_peak, mu_valley):
    if global_step < dr3_apply_warmup_steps:
        return mu_peak
    return max(mu_valley, mu_peak * math.exp(-(global_step - dr3_apply_warmup_steps) / max(1, dr3_apply_warmup_steps)))
```

Likelihood of beating v24: **likely equivalent**. v24 already effectively does this via cosine decay ending near DR3 maturity.

Implementation cost: ~10 LOC. Risk: near-zero.

Narrative benefit: Medium. Removes `chord_mu_decay_steps` and `chord_mu_warmup_steps` from the config by tying them to `dr3.apply_warmup_steps`. Good "clean story" win.

--------------------------------------------------------------------------------

## Idea 3b — Two-phase explicit curriculum (empirical anchor; low novelty)

Formulation:
```
Phase 1 (t < T_DR3):   mu_t = mu_peak,   DR3 weight = 0
Phase 2 (t >= T_DR3):  mu_t = mu_valley, DR3 weight = 1
T_DR3 = first t where disc_acc_ema_t >= 0.80
```

Likelihood of beating v24: **likely equivalent, probably worse by <1%**. Discrete switching often underperforms smooth decay.

Implementation cost: ~15 LOC. Risk: low, but removes a free knob only to add a threshold.

Narrative benefit: Weak. "Two phases" is a worse story than "disc-adaptive" because the threshold is still arbitrary.

--------------------------------------------------------------------------------

## Idea 4a — Token-importance-weighted BC (orthogonal enhancement)

Formulation:
```
w(a_t|s_t) = exp( -log pi_base(a_t|s_t) / T_w )             # base = ref model
w(a_t|s_t) = w / mean_teacher(w)                             # normalize
L_BC = - mu_t * E[ w(a_t|s_t) * log pi_theta(a_t|s_t) ]
```
This replaces the Bernoulli variance weighting φ(p)=p(1-p) in `compute_chord_token_weights` (het_core_algos.py:1696-1720) with a base-model rarity weighting. Rare-under-base tokens (domain-specific like `click[option]`) get large w; generic grammar tokens get small w.

Fit: directly operationalizes the theory's claim that BC's unique value is imprinting rare tokens. Bernoulli φ weights peak at p=0.5 which is unrelated to teacher-specific rarity.

Concern: we need pi_base logprobs. Currently only `old_log_prob` (= previous policy iterate) is available for teacher tokens; the true base model log_prob is not carried end-to-end for teacher samples (teacher_use_log_prob=false in v24). Two options:
  (A) Compute π_base once at training start and pipe through DataProto.
  (B) Approximate with initial log_prob (cached at step 0).

Option B: ~40 LOC — at step 0, snapshot `log_prob.detach()` for teacher tokens and cache in `self._base_teacher_logp`. Use it for all subsequent BC weighting.

Implementation sketch (het_core_algos.py, new function):
```python
def compute_chord_rarity_weights(base_log_prob, T_w=1.0):
    # high weight for low-probability-under-base tokens
    w = torch.exp(-base_log_prob / T_w)
    return w / (w.mean().clamp_min(1e-6))
```

Likelihood of beating v24: **might match**. Theory-aligned but the Bernoulli φ already acts similarly on low-probability tokens (low-p means low φ, so actually the OPPOSITE direction). Replacing φ with rarity weighting is worth a direct ablation.

Implementation cost: ~60 LOC (including base_log_prob snapshotting pipeline). Risk: medium (need to thread base_log_prob through DataProto, FSDP serialization).

Narrative benefit: Moderate. Strengthens the "rare-token installation" story but is only a refinement of existing BC.

--------------------------------------------------------------------------------

## Idea 5 — AWAC single-operator baseline (defensive; required for reviewer response)

Formulation:
```
L_awac = - E_{(s,a) ~ teacher}[ clamp(exp(A(tau)/beta_awac), 0, M) * log pi_theta(a|s) ]
```
One exponent, one clamp. No w_hat. Fixed beta_awac ∈ {0.5, 1.0}.

Implementation sketch:
```python
def compute_awac_teacher_loss(log_prob, advantages, teacher_mask, beta=1.0, M=20.0):
    weight = torch.exp(advantages / beta).clamp(max=M)           # (bs,)
    loss_mat = - weight.unsqueeze(-1) * log_prob
    return agg_loss(loss_mat, teacher_mask, "token-mean")
```
- Drop-in replacement at het_actor.py:1754-1759 when `use_awac=true`.

Predicted result: worse than v24. Lacks (i) density ratio correction, (ii) natural fade-out from w_hat→small, (iii) per-token BC pressure (the weight is sequence-level constant).

Why we still implement: reviewer defense. "Why not single operator?" → cite AWAC ablation showing v24 > AWAC by some margin.

Implementation cost: ~25 LOC. Risk: zero.

Narrative benefit: None for the main result; essential for rebuttal.

--------------------------------------------------------------------------------

## Priority Pick — ONE to implement next

**Idea 2a — Surprise-weighted teacher PG.**

Reasoning:
1. The biggest unresolved criticism of v24 is "dual channel with a decaying hyperparameter smells like mixture tuning." 2a collapses the teacher loss into ONE operator with physically grounded per-token routing, and the routing criterion (pi_theta's own surprise) is exactly what the theory identified as BC's unique contribution.
2. If 2a lands at or above v24, the paper story transforms from "DR3 + BC with schedule" to "surprise-gated teacher PG." This is a first-class algorithmic contribution — no longer "CHORD + density ratio."
3. 1a is the safer pick (lower risk, mid reward) and should be implemented as a fallback if 2a underperforms.
4. 3a is nearly free and should be shipped as a secondary ablation regardless, because it removes 2 hyperparameters from the final config with zero narrative loss.

Implementation order if two slots: 2a (primary), then 1a (fallback + ablation baseline).

--------------------------------------------------------------------------------

## Hypotheses NOT yet tested that could beat v24

### H1 (highest upside) — CHORD-warm-start then DR3-only (phase transition via checkpoint)

Train 25 steps with pure CHORD (μ=0.3→0.05), then at step 25 load the checkpoint and **disable BC entirely** (μ=0), continue with DR3-only for steps 25-100.

Logic: v24 succeeds because BC installs rare tokens EARLY. If BC's entire contribution is finished by step 25, BC after that is pure noise (v36 evidence). Explicit phase switching via checkpoint should match v24 and possibly exceed it because late-stage entropy/SC dynamics are not damped by residual BC at μ_valley=0.05.

Test cost: low — train v24 for 25 steps, fork a config with `chord_mu_peak=0, chord_mu_valley=0` loaded from that ckpt.

Rank: **#1**. Direct empirical test of the core theoretical claim; cheap; beats v24 is plausible.

### H2 (structural upside) — SC-gated BC (let SC progress drive μ)

`mu_t = mu_peak * (1 - progress_ema_t)` where `progress_ema_t` is an EMA of SC's state-channel progress Φ for on-policy samples.

Logic: SC progress is the most direct "is the student solving tasks?" signal. When progress plateaus near teacher's, BC is done. This ties the teacher channel's decay to the state channel's saturation — a unified DUET story ("Action Channel fades as State Channel matures").

Rank: **#2**. High narrative value (ties the two DUET channels). Requires surfacing SC progress to het_actor (not currently done — it lives in trainer).

### H3 (defensive upside) — Teacher-gradient-only BC subset

Restrict BC to the 10% rarest-under-policy teacher tokens, give them strong BC (μ=1.0), give rest μ=0. This is a hard-threshold version of 2a.

Rank: **#3**. More interpretable than 2a (explicit masking) but discrete cutoff typically underperforms continuous weighting. Worth trying only if 2a's continuous version fails.

Final recommendation: implement **H1** in parallel with Idea 2a. H1 is the cheapest, most direct empirical probe of the theory; 2a is the highest-novelty algorithmic contribution. If both beat v24, the paper has both a simpler algorithm (2a) AND a theoretical confirmation (H1).
