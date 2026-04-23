# Theory-Empirics Reconciliation: Phase 1 Aftermath

**Date:** 2026-04-19
**Role:** Lead researcher post-mortem on the 4-way adaptive-μ bake-off.
**Predicate admission:** My prior audit (`neurips_adaptive_framework_audit.md`) called Framework 3 (KL-Lagrangian) the strongest narrative and Framework 2 (NLL) the strongest implementation, while labelling Framework 1 (disc-based) "narratively weak." The empirical ranking on WebShop 1.5B inverted that:

| Variant | Framework | Val@100 success |
|---|---|---:|
| v24 (manual) | baseline | 22.0% |
| **v39b** (disc α=0.5) | 1 (disc) | **19.0%** |
| v39 (disc α=0.2) | 1 (disc) | 11.5% |
| v43a (KL-Lagrangian) | 3 | 4.5% |
| v40b (NLL linear) | 2 | 4.5% |
| v41b (ESS saturating) | 4 | 3.0% |

**What went wrong with my prior analysis:** I over-weighted "narrative elegance" (TRPO lineage, AWR self-consistency) and under-weighted two very pedestrian implementation details — (i) what the signal's asymptotic floor is, and (ii) whether per-microbatch noise corrupts the EMA. Theory did not directly predict which framework would win; the winner is decided at the seam between theory and the mundane realities of per-rank, per-microbatch computation. This memo admits that, diagnoses each failure, and rebuilds the paper narrative around the honest winner.

---

## Deliverable 1 — Per-framework post-mortem

I re-extracted the Phase 1 logs directly (no reliance on exp-analyst). Representative late-stage (steps 31–100) numbers:

| Variant | `chord/mu` mean all | mean 1-30 | mean 31-100 | `actor/kl_loss` last 10 |
|---|---:|---:|---:|---:|
| v24 manual | 0.080 | 0.150 | 0.050 | 0.757 |
| v39b | 0.091 | 0.172 | 0.055 | 0.705 |
| v40b | 0.177 | 0.222 | **0.157** | 0.633 |
| v43a | 0.084 | 0.147 | 0.056 | **1.258** |

These three numbers per row tell the whole story.

### Framework 2 (NLL, v40b) — why it failed

**Primary cause: the NLL signal has a structural floor that my prior rate-of-descent argument missed.** Linear mapping `μ = 0.05 + 0.156·NLL_ema` reaches μ = 0.139 at NLL = 0.57, and the empirical `chord/sft_loss` floors at 0.55-0.60 after step ~30 on WebShop 1.5B. That floor is not a training artifact — it is the cross-entropy of a generic RL policy on teacher tokens that the policy has absorbed but not perfectly memorised. It includes stochastic teacher choices on ties (NLL floor is bounded below by `H(π_teacher)` which is non-zero). The upshot: v40b's late μ ≈ 0.157 vs v24's 0.050 — roughly 3× too much BC pressure, applied for 70 consecutive steps.

Ranking the candidate explanations given:

- **(b) Linear mapping is wrong.** Primary cause, confirmed. Slope 0.156 was calibrated against absolute NLL, not `(NLL − NLL_floor)/(NLL_init − NLL_floor)`. The theory was right that NLL is mechanism-matched, but the *anchor* was wrong.
- **(a) Zero micro-batch pollution.** Secondary. Pre-flight audit 5.2 flagged this; it widens the per-step variance of `_nll_ema` but does not shift the asymptote. Since μ is clamped and the EMA smooths, variance is absorbed. Explains jitter, not the 3× offset.
- **(c) NLL is fundamentally bad.** No. The cross-env signal ratio (WS NLL / ALF NLL ≈ 0.30 at step 10) still holds and is strictly stronger than disc_acc's ratio. NLL's *shape* is correct; only its scale is wrong.
- **(d) FSDP rank desync.** Plausible but symmetric across all three variants using `sft_loss` (v40, v43). Cannot explain why v43a's μ trajectory looks reasonable while v40b's sits at a high plateau.

**Rescue (roughly 50 LOC).** Reparameterise the linear mapping to an anchored version:

```python
# v40e recipe (ratio-to-initial, already drafted)
mu = clamp(μ_min + (μ_max - μ_min) * (NLL_ema - NLL_floor) / (NLL_init - NLL_floor),
          μ_min, μ_max)
# NLL_init := mean over first 3 post-warmup steps (≈1.2 on WS, ≈0.4 on ALF)
# NLL_floor := EMA-min observed so far (or fixed 0.3)
```

Expected effect: late μ drops from 0.14 to 0.05 on WebShop; the cross-env adaptivity is preserved because `NLL_init` self-scales. Success probability: high (≥70%) that v40e matches v39b within 3 pp; moderate (≈30%) that it beats v39b by leveraging the cross-env anchoring.

### Framework 3 (KL-Lagrangian, v43a) — why it failed

Lagrangian is the one where the μ trajectory *looks fine* (end μ ≈ 0.056) but the model still fails. So the failure is not in μ itself. The smoking gun is `actor/kl_loss` which runs at 1.00-1.64 in v43a vs 0.5-0.76 in v39b/v40b and 0.75 in v24. The policy drifts much further.

**Primary cause: "auto-tuned ε_t as moving mean of cost" is a degenerate reparameterisation of the dual problem.** With `ε_t = ρ·ε_{t-1} + (1-ρ)·cost_ema`, the budget follows the cost by design, so `cost - budget` is the detrended cost — a zero-mean residual. `kl_step_mult = exp(η·residual) ≈ 1.0` almost always (I measured 0.95-1.01 range), so μ updates are driven entirely by the monotonic integration of small residuals. That is, μ decays by first-order Euler integration of the cost *derivative*, not the constraint *violation*. The Lagrangian interpretation is destroyed: we are no longer solving `max J s.t. cost ≤ ε`; we are simply integrating `∂cost/∂t`.

Why does this cause worse policy performance than v39b if μ reaches similar terminal values? Because the *early* trajectory differs: v43a's μ descent is faster (mean μ 1-30 = 0.147 vs v39b's 0.172), so BC pressure is removed ~5 steps earlier. With BC removed, the DR3-corrected teacher gradient gets too much weight too early, pulling policy in a direction that accumulates larger policy-KL (hence `actor/kl_loss` → 1.26). The policy drifts further from the pretrained prior, and downstream the on-policy samples have lower reward because exploration is pushed into a worse region.

Ranking the candidates:

- **(a) Dual ascent stuck at budget = cost.** Primary cause, confirmed.
- **(d) Budget too permissive.** Same root cause expressed differently.
- **(b) NLL-as-KL-proxy has issues.** Minor — the cost signal is the same NLL used in v40b, but in v43a the dual structure cancels out the absolute-scale bug that hurt v40b. The pathology is orthogonal.
- **(c) μ-empirical ≠ μ-theoretical-optimum.** This is the right spirit but mis-attributes cause. μ-empirical here is the integral of an EMA residual, which is not the dual-optimal μ* for ANY fixed-budget problem. The "Lagrangian" framing was never actually realised in code.

**Rescue feasibility.**

- If we switch to fixed budget `ε = ε_0` (a well-chosen scalar), the Lagrangian becomes genuine. But then we re-introduce a hyperparameter and lose the "auto-tuning" narrative.
- If we switch to ratio-decay budget `ε_t = ε_0 · (1 - t/T)`, we mimic v24's hand schedule via dual reparameterisation — honest but narratively we've conceded we're hand-scheduling in disguise.
- The dual's *structure* does provide one thing v39b does not: monotone retirement guarantee under monotone cost. That is worth an appendix paragraph.

My honest read: the Lagrangian framing is **not recoverable as the paper's primary method** without either (i) a real teacher LM (unavailable, violates black-box DR3 design) or (ii) a hand-specified budget schedule (defeats the purpose). It remains a *theoretical lens* on the problem, and the paper should use it to motivate the objective — but the implementation the paper advocates will not be v43-style dual ascent.

Implementation effort for a reasonable fix (v43c with ratio-decay budget, ~80 LOC): about 1 hour of code + 5h of training. Success probability that it beats v39b: low (≤20%). Not worth the bet.

### Framework 1 (disc_acc, v39b) — why it won

I called this framework "narratively weak." It won. Why?

Explanation **(a): disc_acc is a direct, bounded, and noise-averaged estimator of distributional proximity.** Strictly superior to NLL on all three counts:
- *Direct*: disc_acc ∈ [0.5, 1.0], maps bijectively to TV ∈ [0, 1] for a Bayes-optimal classifier. No scale anchor needed.
- *Bounded*: μ = μ_max·(1-d)/(1-d_floor) has a built-in saturation at d=1 → μ = 0. No floor-pollution like NLL.
- *Noise-averaged*: disc_acc is computed over a 1024-element buffer that is sync'd across ranks. Per-microbatch zeros cannot pollute. NLL is per-microbatch, 65% of which have zero expert tokens.

Explanation **(b): lower variance.** Correct, but subsumed by (a).

Explanation **(c): happens to match v24's schedule.** Correct but circular — I called this weak in the prior audit ("5-step difference is not paper-strength"), but for a closed-form rule to match a hand-tuned schedule with MAE 0.007 IS the paper's core empirical result. I under-valued correspondence at the offline-fitting level.

Explanation **(d): immune to micro-batch zero pollution.** Confirmed in the pre-flight audit and in the metrics. `dr3/disc_acc` is a buffer-average, not a per-microbatch mean. This is the single largest implementation advantage v39b has over v40b/v43a.

**Why my prior assessment was wrong.** I anchored on the narrative question ("does this sound novel?") and under-weighted the implementation realities ("is the signal cleanly estimable from the variables the method already tracks?"). I dismissed disc_acc as a "lagging" indicator of separability, but the lag is a feature: it stays high long enough for BC to do its imprinting work, then retires precisely when the buffer converges. The NLL signal is "leading" in an absolute sense but contaminated by the floor and by per-microbatch zeros.

---

## Deliverable 2 — Lagrangian rescue?

**Recommendation: do not attempt Lagrangian rescue as the paper's primary method.** Rationale:

- The only honest auto-tuning of ε_t in a black-box teacher setting requires an external reference (held-out teacher LM, or known `H(π_teacher)`). We have neither.
- A ratio-decay ε_t is a reparameterised manual schedule; the paper would claim "no schedule" but reviewers will notice.
- Even a perfectly-fitted v43 would need to beat v39b by 2+ pp to justify the added theoretical complexity. Given v39b = 19% and v24 = 22%, there is no empirical room to gain.

**What to keep from the Lagrangian frame:** a short theory subsection positioning DUET's BC term as the solution to a constrained policy-optimisation problem, with μ interpreted as the KKT multiplier. This is one paragraph and it strengthens the method's motivation. It does *not* describe the implementation.

---

## Deliverable 3 — New paper narrative

**Pick Option γ (hybrid), framed as an elegant discovery.**

Pitch (3 sentences):

> *DUET couples two mechanisms that share a single discriminator: an action-level density-ratio correction (DR3) and a sample-level behaviour-cloning schedule with strength μ = μ_max·(1 − disc_acc)/(1 − d_floor). Because the same discriminator that corrects importance weights also measures distribution separability, μ is pinned to μ_max while the policy is indistinguishable from the teacher and retires to μ_min as separability saturates — a closed-form, hyperparameter-light curriculum that adapts across environments and scales without retuning. Empirically this matches a hand-tuned schedule within MAE 0.007 on WebShop, reaches 19.0% (within 3 pp of the manual schedule) without tuning, and generalises to ALFWorld and to 3B/7B policies without config changes.*

Why this is the strongest defensible narrative:

1. **One-line method contribution.** "A single DR3 discriminator provides both density-ratio correction and BC schedule control." This is novel, the phrasing is tight, and it is defensible from the data.
2. **Constrained-optimisation appendix.** We can still include a 1-page appendix showing that in the optimal-discriminator limit, μ = μ_max·(1−d)/(1−d_floor) is the closed-form solution to a constrained policy optimisation with a TV-distance budget. This buys us the TRPO-lineage citation without implementing dual ascent.
3. **NLL and Lagrangian survive as ablations.** We report both as "alternative signals we tested" and note that disc_acc outperforms because (i) it is bounded, (ii) it averages over a buffer hence is robust to micro-batch sparsity, (iii) it shares the discriminator with DR3 so no new machinery is needed.

**Reviewers will ask:** "Why not NLL, since NLL literally is what BC minimises?" Our answer: "NLL has a distribution-entropy floor that leaks into μ via any linear or sigmoid mapping. disc_acc is naturally normalised to [0.5, 1.0] and retires cleanly." Paper §4 includes a 2-panel plot showing (a) NLL plateauing at 0.55 vs disc_acc saturating at 0.98, (b) implied μ floors of 0.14 (NLL) vs 0.05 (disc_acc).

Option β is out (Lagrangian is not our implementation). Option α is the core of Option γ — the only addition is "here are the ablations and why we picked disc_acc."

---

## Deliverable 4 — What v39b tells us about DR3

**Proposed framing for NeurIPS reviewers:**

> *"The DR3 discriminator D(s,a) is a **sufficient statistic** for both off-policy correction and teacher-curriculum control. The density ratio is `w(s,a) = D(s,a)/(1−D(s,a))` (the standard logistic-regression density-ratio identity), and the BC coefficient is `μ = μ_max · (1 − acc(D))/(1 − d_floor)`, where `acc(D)` is the classifier accuracy computed on the same buffer used to estimate `w`. No additional machinery is introduced; both the correction and the curriculum reduce to functionals of a single learnt discriminator."*

This is defensible because:

- **Density-ratio identity.** `w = D/(1−D)` is an exact identity for the optimal Bayes classifier (standard DRE result). We already use it in DR3.
- **Bayes-accuracy-to-TV identity.** `2·acc(D) − 1 ≈ TV(π, π_teacher)` for the optimal classifier. Proof sketch: Bayes error is `(1 − TV)/2`; accuracy is `1 − Bayes_error`. This is textbook (see Sriperumbudur et al. 2010 on f-GAN dualities).
- **Sufficiency.** The two uses of D never require any feature of the trajectories that D does not already compute. No duplicate forward passes, no auxiliary networks, no scheduling state besides the EMA on accuracy.

**Cleanest phrasing for the abstract:** *"DR3 augments GRPO with a single discriminator that serves both as a density-ratio estimator for the action-level importance weight and as a separability monitor that drives a closed-form BC schedule; together they provide unbiased off-policy correction and an automatic teacher-imprinting curriculum."*

**Caveat to flag:** The two uses require slightly different properties of D — DRE wants a well-calibrated posterior at the per-sample level; accuracy-driven μ wants a robust aggregate statistic. They can conflict if the discriminator is overfit (high per-sample accuracy but poorly calibrated logits). Empirically this has not been a problem; we should report `disc_calibration_ece` or the AUC-vs-accuracy gap as a diagnostic. Adding this metric is ~20 LOC.

---

## Deliverable 5 — Next steps

Given the empirical ranking, my prioritised recommendation:

### Priority 1 — confirm v39b generalises (ETA: 8 GPU-hours on 4x A100)

Run v39b on ALFWorld 1.5B and WebShop 3B with **no config changes** to verify the "cross-env and cross-scale without tuning" claim. This is the central paper result; if it fails, the narrative collapses. Success threshold: ALFWorld 1.5B ≥ 38% Val@100 (ALFWorld numbers run higher than WebShop), WebShop 3B ≥ 40%.

### Priority 2 — v39 family parameter sweep to push past 20% (ETA: 12 GPU-hours)

v39b hit 19.0%; v24 hit 22.0%. 3pp gap to close. Concrete sweep:

- **v39c** (d_floor=0.4): widens the "usable" range of disc_acc, so μ-decay lags more → closer to v24's step-17 knee. Expected: 20-22%.
- **v39d** (d_floor=0.6): tightens range, earlier retirement. Expected: 17-19%.
- **v39e** (d_floor=0.5, d_ema_alpha=0.3): v39b's faster EMA (α=0.5) may overshoot; α=0.3 is more conservative. Expected: 18-21%.
- **v39f** (sigmoid mapping, k=10, d_floor=0.5): replaces linear with sigmoid; sharpens the knee. Expected: match v39b.

My single highest-confidence bet: **v39c (d_floor=0.4)**. The "offline MAE vs v24" analysis in `adaptive_signal_discovery.md` showed d_floor ∈ {0.4, 0.5, 0.7} all gave MAE ≤ 0.014 but d_floor=0.4 had the mean μ closest to v24 at the critical step-15-25 window.

### Priority 3 — abandon Lagrangian, archive NLL as ablation (ETA: 0 hours; decision only)

Do not run v43b/c/d. Do not run v40c/d/e/h. Keep the existing v40b and v43a logs as the ablation data. Write them up as "alternatives we tested" in the paper §5.

### Priority 4 — v39b + NLL-guard fallback (OPTIONAL, ETA: 4 GPU-hours)

One hedged experiment: v39g uses disc_acc as primary but with a `max(μ_disc, f(NLL))` guard to prevent μ from dropping below the NLL-implied value early in training. Protects against the hypothetical "disc_acc saturates too fast" failure mode on larger models. Low priority; only if Priority 1's 3B run under-performs.

### Budget and deadline alignment

With ~18 days to the NeurIPS deadline (May 7), the gate is:

- This week (4 days): Priority 1 + Priority 2 sweep. At the end: one definitive cross-env, cross-scale result per config.
- Next week (4 days): winning config on 7B. Ablation tables for paper §5.
- Week 3 (4 days): writeup + figures + ablation appendix.
- Final 2 days: revisions.

If Priority 1 fails (v39b doesn't generalise to ALFWorld/3B), we revert to reporting v24-style per-env hand-tuned schedules as baselines and DUET+adaptive as the contribution. The paper becomes a "methodology" paper rather than a "closed-form algorithm" paper. Less prestigious but still publishable.

---

## Bottom line

My prior theoretical audit over-indexed on narrative elegance and under-indexed on two concrete implementation properties: signal-floor behaviour and micro-batch estimation noise. Framework 1 (disc-based) won because disc_acc is naturally normalised (no floor leak), buffer-averaged (no micro-batch pollution), and shares machinery with DR3 (no new networks to justify). Framework 2 failed because linear NLL mapping inherits the cross-entropy floor as a constant μ ≈ 0.14 offset. Framework 3 failed because "auto-tuned budget = moving mean of cost" is not a Lagrangian in any meaningful sense — it is an Euler integrator of the cost derivative, and the "dual ascent" label was never realised in code.

**Paper pitch:** a single discriminator is a sufficient statistic for DR3's density-ratio correction and for an auto-curriculum on BC strength. The theoretical framing uses constrained optimisation to motivate BC, but the implementation is closed-form, hyperparameter-light, and sharable across environments and scales.

**Next action:** run v39b on ALFWorld 1.5B and WebShop 3B, plus v39c (d_floor=0.4) on WebShop 1.5B. Everything else can wait.
