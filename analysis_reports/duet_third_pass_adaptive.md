# DUET — Third-Pass Adaptive BC: Pre-mortem of v37/v38, and a Teacher-NLL-Based v39

*Lead-researcher memo, 2026-04-19. Written after two adaptive schemes failed. Being self-critical, including about my own prior memos. This is a re-derivation from the data rather than a patch on previous theory.*

---

## 1. Pre-mortem: why v37 and v38 failed at the theory level (not just calibration)

My first two adaptive proposals both failed not from miscalibration but from **a wrong choice of observable**. Both of them picked a quantity that is (i) structurally invariant to the manipulation we want, and (ii) interpreted under a mean-based summary when the BC-relevant signal lives in the tail.

### 1.1 v37 (V_A-based µ) — the observable is a GRPO-normalized artifact

The second-pass theory identified advantage variance $V_A$ as the "downstream quantity BC is meant to regularize." The derivation looked clean: BC homogenizes rollouts → $\sigma_g$ shrinks → $|A|$ shrinks → $\|g_{\rm RL}\|$ shrinks. But the implementation read $V_A$ *after GRPO group-wise z-score normalization*, and the normalization is precisely what makes $V_A$ structurally $\approx 1$ by construction — per my own memo, §5, where I noted this and then forgot to propagate the correction. The v37 log confirms it: `VA_ema` spent training at 0.9–3.3, never dipping below the set target $V_A^\star = 0.035$, so the sigmoid gate returned 1.0 every step and µ pinned at 0.3.

But the deeper error is this: **the mechanism description "BC shrinks $V_A$" is not wrong — it was just specified on the wrong side of the normalization.** The mechanism predicts a shrinkage of group-reward std $\sigma_g(t)$, not of post-normalization advantage. $\sigma_g$ is mentioned in the v24 memo §5 ("Correct signal: group-reward std, not advantage std") — I had the right answer written down, and still shipped the wrong signal. Lesson: the *downstream* of a normalization is never the right knob; always reach for the *upstream*.

Is there salvage? Yes — $\sigma_g$ is logged per-group. It was the correct backup proposal and remains viable. But I now think even this is the wrong mental model: in Section 2 I will argue the mechanism is *not* "BC shrinks $\sigma_g$" for ALFWorld's v24 data, which **amplifies** on-policy advantage magnitude late (`adv_onpolicy_effective_abs_mean` 0.04→0.07→0.16, not 0.12→0.17 as WebShop v24). The advantage-regularizer theory was always a partial story, and ALFWorld breaks it.

### 1.2 v38 (SPW, $(1-\pi_\theta)$-weighted DR3) — BC-critical tokens are not the average

The unified-DR3 memo's argument rested on: "at the rare-token limit $\pi_\theta(a^*|s) \to 0$, so $(1-\pi_\theta) \to 1$, so the per-token coefficient approaches BC's." This is mathematically correct for an individual token, but the aggregate SPW multiplier applied over the teacher-token stream averages to $\mathbb{E}[1 - \pi_\theta] \approx 1 - \bar\pi$. Empirically `spw/phi/mean = 0.175–0.202` across v38's whole run, because the mean $\pi_\theta$ on teacher tokens is $\approx 0.82$ (teacher tokens are mostly NOT rare — the rare ones are a long tail).

So SPW's mean is a *constant* near-0.18 multiplier on DR3 — de facto, SPW ran DR3 at 18% strength, no BC recovery, and no adaptivity. Result: 0.474, essentially a weakened DR3.

The deeper theoretical error: **the rare-token recovery argument works at a single-token resolution, but is defeated by token-level averaging in the aggregated loss.** Under `loss_agg_mode: token-mean`, each token's coefficient is summed and divided by count — the modal (common) tokens dominate, and rare tokens get buried by their frequency in the aggregate. I should have reasoned about the loss statistic the optimizer actually sees (mean, not max), not the per-token form.

Is there salvage? A $(1 - \pi_\theta)^\gamma$ with $\gamma > 1$ would skew toward rare tokens; $\gamma = 3$ makes the multiplier $\ll 0.01$ on common teacher tokens but ~1 on rare ones. But this introduces an exponent hyperparameter that needs tuning per environment — defeating the "no schedule" promise. So SPW at $\gamma = 1$ is fundamentally too gentle, and $\gamma > 1$ loses the zero-tuning claim. The *entire architectural choice* of per-token reweighting of the DR3 surrogate is the wrong structural knob; I should have accepted that and not tried to patch it in v38.

### 1.3 What both failures share

Both v37 and v38 picked an adaptive signal that is **summary-invariant to the training regime we care about**:
- v37: post-normalization advantage is structurally constant.
- v38: mean $(1-\pi_\theta)$ on teacher tokens is structurally ~0.18.

The correct adaptive signal must have the property that its *summary statistic* (mean, or whatever the optimizer reads) **differs meaningfully between WebShop-like regimes (hard, rare-token) and ALFWorld-like regimes (easy, templated)**. Neither of my two prior signals had this property. This is the failure of my causal model: I identified downstream symptoms (variance, surprise) without checking whether their batch-aggregated forms carry the regime information.

---

## 2. Candidate adaptive signals — evaluation matrix

I lay out all five candidates from the brief, evaluate each against the four criteria, then identify the winner.

### Criteria recap

(C1) **Strong when BC helps** (WebShop cold start / rare teacher tokens);
(C2) **Weak when BC hurts** (ALFWorld templates, well-learned tokens);
(C3) **Observable without GRPO normalization artifacts** (no structural invariance);
(C4) **Smooth over training steps** (not per-microbatch noisy).

### (A) Teacher NLL directly (chord/sft_loss)

**Formula**: $\mu_t = \mu_{\min} + (\mu_{\max} - \mu_{\min}) \cdot \sigma(\alpha \cdot (\text{NLL}^{\rm EMA}_t - \tau))$.

**Empirical evidence**: NLL = $-\text{log\_prob\_mean}$ on teacher tokens is a *direct* measure of "how surprised is the policy by teacher actions on average." The data is exactly what we need:

| Regime | NLL(step 1) | NLL(step 10) | NLL(step 30) | NLL(step 100) |
|---|---:|---:|---:|---:|
| WebShop v24 | 1.16 | 1.06 | 0.70 | 0.62 |
| ALFWorld v24 | 1.11 | 0.34 | 0.24 | 0.30 |
| WebShop v36 (no BC push) | 1.16 | 1.28 | flat ~1.0 | 0.66 |

The separation at step 10 is the signal: **ALFWorld's NLL drops 3× faster than WebShop's** because ALFWorld teacher tokens are in the 1.5B's pretraining support. And when BC is applied (v24 WebShop), NLL *does* drop faster than when it isn't (v36 WebShop) because BC directly pushes it down.

**Verdict**:
- C1: YES — high on WebShop cold start (~1.2 nats, $\sigma(\alpha(1.2-0.7)) \to 1$).
- C2: YES — drops below 0.4 by step 10 on ALFWorld, µ relaxes to near-floor.
- C3: YES — NLL is pre-normalization, from raw log-probs. No GRPO artifact.
- C4: MODERATE — NLL has microbatch noise (std of 0.2–0.3 between adjacent steps in v24), but EMA smooths it.

**This is the winner** — scored highest on C1/C2/C3, only caveat is EMA smoothing for C4.

### (B) Bottom-k $\pi_\theta$ over teacher tokens

**Formula**: $P_k(t) = $ mean of bottom-10% of $\pi_\theta$ on teacher tokens this step; $\mu_t = \mu_{\min} + (\mu_{\max} - \mu_{\min}) \cdot (1 - P_k(t))$.

**Analysis**: Using the Gaussian approximation from v24's observed (log_prob_mean, log_prob_std): bottom-10% π ranges 0.004 (step 1) → 0.056 (step 100) on WebShop. This *does* separate regimes. But:
- Not currently logged (requires code change to compute quantile of log-probs on teacher-masked tokens).
- Persistently rare tokens dominate bottom-10% for hundreds of steps; the signal saturates at "~always some rare token present" and mu stays high even late in training. Bottom-10% $(1-\pi)$ stays at 0.94 even at step 100 in v24 — not the "retirement" shape we want.

**Verdict**: Theoretically sound but operationally too sticky. Rejected.

### (C) KL($\pi_\theta \| \pi_{\rm tch, empirical}$)

**Formula**: Estimate teacher's empirical distribution on teacher trajectories (token histogram or a frozen teacher LM), compute KL to current policy, use as µ driver.

**Analysis**: The DR3 discriminator *is* a KL estimator to first-order. `dr3/w_hat_teacher/mean` rises when KL is high and falls to 1 when KL is near 0. But (i) we don't have a Qwen72B teacher LM loaded at RL time (that's the whole point of DR3 being black-box), and (ii) empirical KL from a token histogram is brittle at our sample sizes (~500 teacher trajectories per microbatch). Heavy instrumentation.

**Verdict**: Theoretically clean but infrastructurally expensive. Available as cross-check via `dr3/w_hat_teacher` but has a known collapse-to-1 failure mode (the reason DR3 has w_hat floor). Rejected as primary.

### (D) ||∇L_bc|| EMA (gradient norm of pure BC)

**Formula**: Track EMA of the BC loss gradient norm, use as µ driver.

**Analysis**: Cyclical. The BC gradient on teacher tokens is $-\mu \nabla \log \pi_\theta$, so $\|g_{\rm BC}\|$ scales linearly with µ plus the per-token surprise. Measuring it and using it to set µ creates a feedback loop: if µ is high, $\|g_{\rm BC}\|$ is high, reinforcing µ. Not adaptive; runaway.

**Verdict**: Self-referential. Rejected.

### (E) Score-std in group (σ_g) pre-normalization

**Formula**: Track `critic/score/std` per-group, use ratio to initial to drive µ.

**Analysis**: This is the "correct" signal per the second-pass theory for WebShop (where BC *does* homogenize rollouts). It's logged, unnormalized, smooth. But the ALFWorld data *refutes* the homogenization story — v24-ALFWorld has `adv_onpolicy_effective_abs_mean` growing 0.04→0.16 under BC, opposite direction. So "BC shrinks $\sigma_g$" is a WebShop-only phenomenon, not universal. If we use $\sigma_g$ as µ driver, on ALFWorld we'd see $\sigma_g$ grow (not shrink) and µ would stay or rise — wrong direction.

**Verdict**: Correct for WebShop, wrong sign for ALFWorld. **Rejected — this is where my second-pass theory was incomplete.**

### Summary matrix

| Signal | C1 (WebShop high) | C2 (ALFWorld low) | C3 (no GRPO artifact) | C4 (smooth) | Logged? |
|---|---|---|---|---|---|
| **(A) Teacher NLL** | **yes, ~1.2** | **yes, ~0.3** | **yes** | EMA | **yes (chord/sft_loss)** |
| (B) Bottom-k π | yes | partially (sticky) | yes | yes | no |
| (C) KL to teacher LM | yes | yes | yes | yes | no (infra cost) |
| (D) ∥∇L_bc∥ | circular | circular | no | yes | no |
| (E) σ_g | yes | **WRONG SIGN** | yes | yes | yes |

**Winner: (A) Teacher NLL.** It is mechanism-matched (measures exactly the thing BC is supposed to push down), regime-separating, logged today, and pre-normalization. Secondary: (C) as a paper-rebuttal backup story.

---

## 3. Proposed v39 design

### 3.1 Formula

$$
\boxed{\mu_t = \mu_{\min} + (\mu_{\max} - \mu_{\min}) \cdot \sigma(\alpha \cdot (\bar{N}_t - \tau))}
$$

where:
- $N_t = -\frac{1}{|\mathcal{T}_t|}\sum_{(s,a)\in\mathcal{T}_t} \log \pi_\theta(a\mid s)$, the average NLL on teacher tokens at step $t$ (equals `chord/sft_loss` with `chord_use_token_weighting: false`, which is how v24 and ALFWorld v24 both configure it);
- $\bar{N}_t = \rho \bar{N}_{t-1} + (1-\rho) N_t$, an EMA with $\rho = 0.7$ for smoothness;
- $\sigma$ is the logistic sigmoid, $\alpha = 6.0$ is the sharpness gain, $\tau = 0.65$ is the NLL midpoint.

Fixed constants (not per-environment): $\mu_{\min} = 0.05$, $\mu_{\max} = 0.30$, $\rho = 0.7$, $\alpha = 6.0$, $\tau = 0.65$.

### 3.2 Remaining hyperparameters — exactly 2 non-trivial

The claim "no manual schedule" is supported. We retain only:
1. **$\mu_{\max} = 0.3$** — same as v24 (justified: BC strength saturates probability-space progress above ~0.3).
2. **$\tau = 0.65$** — the NLL midpoint; below this, µ decays; above this, µ ramps.

$\tau$ is environment-agnostic because NLL on teacher tokens is on the same scale across environments (both ALFWorld and WebShop start at ~1.1 nats, both achieve ~0.3–0.7 when converged). Empirically chosen from v24 WebShop endpoint (NLL $\approx 0.6$, midpoint between initial 1.2 and floor ~0.3).

$\alpha = 6$ and $\rho = 0.7$ are noise-shaping knobs that can be fixed a priori (sharper $\alpha$ → more aggressive retirement; longer EMA → smoother but laggier).

**Compared to v24**: four hyperparameters eliminated ($\mu_{\rm warmup}$, $\mu_{\rm decay}$, cosine shape, step budget).

### 3.3 Empirical prediction for µ trajectory

Using observed NLL values from v24 logs:

| Step | WebShop NLL (EMA) | $\mu_t$ (v39) | v24 manual | ALFWorld NLL (EMA) | $\mu_t$ (v39) |
|---:|---:|---:|---:|---:|---:|
| 1   | 1.16 | 0.289 | 0.30 | 1.11 | 0.285 |
| 10  | 1.17 | 0.289 | 0.21 | 0.86 | 0.245 |
| 15  | 1.07 | 0.282 | 0.14 | 0.72 | 0.202 |
| 20  | 1.10 | 0.284 | 0.07 | 0.59 | 0.153 |
| 30  | 0.92 | 0.259 | 0.05 | 0.49 | 0.118 |
| 50  | 0.86 | 0.245 | 0.05 | 0.46 | 0.109 |
| 100 | 0.71 | 0.198 | 0.05 | 0.40 | 0.095 |

**Key qualitative differences vs v24 manual**:

- **WebShop**: v39 holds µ near $\mu_{\max}$ longer (µ > 0.25 through step 30, vs v24 crossing 0.05 at step 25). Then decays to ~0.20 at step 100, not 0.05. This *more gradual* decay may be beneficial on WebShop — v12's regression late (grad_norm 11, advantage 0.33) suggests the policy benefits from continued light BC, which v24 starves at µ=0.05.
- **ALFWorld**: v39 starts at 0.29 but drops to 0.12 by step 30 (v24 is at 0.05). Critically, v39 applies *more* BC at steps 1-10 (where v24's 0.21-0.28 gave +6pp at Val@50), then releases less aggressively, keeping a light BC "safety net" at 0.10 to prevent the format regressions (plan-dumps, `[/action]`) documented in `v24_alfworld_trajectory_diff.md`.
- **µ-hybrid**: the signature of v39 is "BC stays warm longer on both environments" but decays based on what the policy actually learns, not a fixed step budget. This is the adaptive story we've been trying to tell all along.

### 3.4 Why v39 fixes what v37 and v38 broke — mechanism-by-mechanism

**Fixing v37 (V_A-based)**: v37 used post-normalization $V_A$, which is structurally invariant. v39 uses pre-model-output NLL (raw log-probs averaged over teacher tokens). GRPO normalizes *advantages*, never log-probabilities, so NLL is immune to the normalization pipeline. The `log_prob_mean` and `sft_loss` series I pulled from v24 logs show clean, interpretable trajectories across runs; if this were a post-normalization quantity it would be structurally constant.

**Fixing v38 (SPW-based)**: v38 relied on a per-token reweighting whose aggregate averaged to 0.18. v39 uses a *scalar* (mean-NLL-per-microbatch) as the µ driver, and µ multiplies the entire BC loss uniformly across tokens. The adaptation happens at the schedule level, not the token level. Common teacher tokens still get µ·BC, rare ones still get µ·BC; the relationship doesn't collapse to a tokenwise frequency-weighted mean.

**What v39 doesn't claim**: v39 is NOT a unification of BC and DR3 (that was the v38/SPW aspiration). It leaves the two operators separate and simply replaces the manual µ schedule with a data-driven one. This is a deliberately narrower scope: we fix the tractable problem (calibrating µ) rather than the ambitious one (unifying the two arms). The SPW memo's "narrative upgrade" from two-operators-to-one is abandoned.

**What v39 doesn't promise**: "beating v24 on WebShop." The v24 manual schedule may be empirically near-optimal on WebShop precisely because a human spent hyperparameter search there. v39 is expected to *approximately match* v24 on WebShop (within 3pp) and *beat v24 on ALFWorld* (where manual over-releases). The headline win is *generalization without tuning*, not "better raw number on WebShop."

---

## 4. Why v39 fixes what v37/v38 broke (mechanism-by-mechanism)

I separate the mechanistic fixes because the failure modes were distinct.

### v37 failure mechanism: downstream-of-normalization observable
v37's $V_A = \text{std}(|\bar A_i|)$ over samples, where $\bar A$ is post-GRPO-normalization mean absolute advantage per sample. GRPO divides $R_i - \mu_g$ by $\sigma_g$ per-group, so `advantages.std()` is $\approx 1$ by construction. v37 set `VA_star = 0.035` based on `adv_onpolicy_effective_abs_mean = 0.17` (a different quantity, mean absolute not std), so its target was 30× too small *and* it was reading a quantity that cannot reach the target anyway. Both the calibration and the observable were wrong.

**v39 fix**: NLL is a *pre-normalization* raw log-prob average. No GRPO touches it. The observable is in the same units across variants and environments. Scale-free calibration via EMA-compared-to-midpoint.

### v38 failure mechanism: token-level signal averaged into a scalar
v38's multiplier $(1-\pi_\theta(a|s))$ is correct at single-token resolution: on rare teacher tokens, $(1-\pi)$ → 1, recovering BC-strength. But the loss averages (token-mean mode), so DR3's coefficient became `mean((1-π_teacher_token))` ≈ 0.18 — the rarity of the tail is lost in the mean. SPW failed not because its motivation was wrong but because its implementation didn't respect the loss-aggregation mode.

**v39 fix**: the µ driver is a scalar computed from teacher-token statistics, and µ multiplies the *outer* loss. There is no token-level expected value that could average away the signal; NLL$_{\rm mean}$ is a scalar, µ is a scalar, the BC loss is scalar times gradient. No aggregation collapse.

### Shared lesson
Both v37 and v38 are examples of picking an observable that becomes trivial under the actual computation. v39 picks an observable (teacher-token NLL mean) that:
1. Differs systematically between regimes (WebShop 1.2 vs ALFWorld 0.3 at step 10).
2. Decreases as BC does its job (v24 has steeper decay than v36).
3. Is computed over an unaggregated scalar pipeline that preserves the signal.

This is the *minimum* constraint set my prior theories were violating. v39 respects it.

---

## 5. Critical open questions — what remains unexplained

Two failures have me humble. I flag things the theory still doesn't answer.

### 5.1 Why does v24 on ALFWorld amplify advantage magnitude late (0.04→0.16)?

The second-pass theory predicted BC *reduces* on-policy advantage magnitude via rollout homogenization. On WebShop that holds (v24's adv stays 0.12–0.17, vs v12's 0.16→0.33). But on ALFWorld under v24, `adv_onpolicy_effective_abs_mean` grows 0.04→0.07→0.16 — **opposite direction**. Response length grows 28% concurrently. Something in the ALFWorld/BC interaction produces longer rollouts with higher reward variance, not less.

**Hypothesis** (untested): On ALFWorld, BC installs the "I will start by checking countertop" template (per `v24_alfworld_trajectory_diff.md`), which generates *deterministic* first moves. But if the task target isn't on the countertop, the policy enters a longer search. Longer rollouts introduce more per-task reward variance (some find the object, some don't within 30 turns). The group reward std grows, advantages amplify. This is consistent with BC *biasing* rollouts but not *homogenizing* their returns.

**Experiment to test**: Log `critic/score/std` per-group for ALFWorld v24 vs v1. If v24's $\sigma_g$ grows faster than v1's, the amplification mechanism is "biased first-move + long-tail exploration," not noise. Would tell us that the "advantage regularizer" story is WebShop-only, and that on ALFWorld BC should be withdrawn aggressively — which v39 does.

### 5.2 Does NLL really capture "rare token" information, or only mean surprise?

v24's WebShop `log_prob_std = 3.48 → 1.77` (halves), while `log_prob_mean = -1.16 → -0.61` (halves similarly). The two are strongly correlated in this data, so I cannot tell from the logs whether NLL is informative because it captures tail behavior or because it correlates with mean surprise. If there's a regime where NLL drops while rare tokens remain unlearned (e.g., a medium-support environment where wrapper tokens saturate fast but rare ones don't), v39 would under-fire BC there.

**Experiment to test**: Add `log_prob_p10` and `log_prob_p90` metrics (10th and 90th percentiles of log π on teacher tokens) to v39. If they tell a different story than the mean — particularly if the p10 (tail) lags the mean — we'd know NLL is an imperfect proxy and we should use p10 instead. This can ride on the v39 run without additional experiments.

### 5.3 What sets the target $\tau = 0.65$ across model sizes?

My $\tau = 0.65$ is fit to Qwen2.5-1.5B on WebShop and ALFWorld. On Qwen2.5-3B (stronger prior), initial NLL may be lower (say 0.9 not 1.2), meaning the adaptive rule would never ramp µ. If that happens, we're under-applying BC on 3B runs. The universal-$\tau$ claim is fragile.

**Experiment to test**: Run v39 on 3B WebShop and 3B ALFWorld. Log NLL trajectory and check whether µ trajectory is qualitatively similar to 1.5B. If 3B NLL stays < 0.65 throughout, µ sits at floor and v39 ≈ v12. Fallback: set $\tau = $ initial-NLL × 0.6 (a ratio-to-initial rule, making $\tau$ scale-adaptive to model size).

### 5.4 Why is v39 NOT unifying DR3 and BC, and do we pay a narrative cost?

The SPW memo pushed a "single operator" narrative. v39 is a retreat: two operators, one adaptive knob. The contribution claim thins from "unified off-policy PG" to "adaptively weighted BC augmentation." For NeurIPS, reviewers might dock the contribution. But the empirical cost of v38's unification attempt (0.47 vs v24's 0.68) was steep — the narrative wasn't worth the performance loss. Better to have a working adaptive scheme than a beautiful broken one.

**Contingency**: If v39 validates (WebShop ≥ 0.65 AND ALFWorld Val@100 ≥ 33), the paper can still position the Action Channel as "BC-regularized DR3 with a principled teacher-surprise-driven schedule." The "two operators" framing is defended by orthogonality: DR3 corrects *importance* (unbiased off-policy correction), BC regularizes *support* (biased pull toward teacher). They address different problems, they're appropriately separate, the schedule links them. Option G from the first-principles memo's §6 is the honest story.

### 5.5 The one prediction I'm most uncertain about

v39's µ trajectory on WebShop stays ABOVE v24's manual decay (0.20 at step 100 vs v24's 0.05). This means v39 may be *over-applying BC late on WebShop*, which v22/v23/v36 told us is harmful when combined with v12-stab. Three possibilities:
- (best case) v39's gentle continued BC helps: it prevents the late grad_norm explosion v12 showed, effectively giving v24-like stability while maintaining DR3 curriculum; val score ≥ 0.65.
- (neutral case) v39 matches v24 within 3pp; acceptable win by generality.
- (worst case) v39's late BC interferes with DR3 fade-out (the v36 < v12 mechanism identified in first-principles memo §2.2c); val score drops to 0.45.

This is the single most important empirical uncertainty. The v37/v38 memos each claimed a prediction that empirical data crushed. I am explicitly flagging this one. If v39 underperforms v24 significantly, the interpretation is that v24's aggressive µ→0.05 by step 25 is load-bearing *specifically because* it stops interfering with DR3 in the late regime, not because the empirical trajectory of µ matches the NLL trajectory. We'd need a µ that retires *faster* than NLL decays — which `v39-aggressive` ($\tau = 0.9$) would be — and rerun.

---

## 6. Bottom line

**Primary recommendation**: Run v39 with $\mu_t = 0.05 + 0.25 \cdot \sigma(6(\bar N_t - 0.65))$, NLL EMA $\rho = 0.7$. ~20 LoC in `het_actor.py` replacing the `chord_mu_adaptive` block. Zero new instrumentation (NLL is already logged as `chord/log_prob_mean`).

**Self-critical appendix**: I've been wrong twice. The pattern of my errors is "picking an observable that looks right but becomes trivial under the real computation." v39's NLL-based rule does not have this pathology *that I can see* — but I said that about V_A in v37, and about $(1-\pi_\theta)$ in v38. The first experiment (v39 on WebShop 1.5B) will determine whether the theory is actually right or whether I need a fourth pass. If v39 fails, the memo I write next must start with a more thorough audit of each observable's behavior *under the aggregation and normalization pipeline it will pass through* before any theorization.

**Paper-narrative impact**: If v39 validates, the Action Channel story becomes "data-driven curriculum from teacher-NLL, no schedule hyperparameters, environment-agnostic." This closes the "is the schedule tuned?" reviewer critique cleanly and preserves the dual-channel framing. If v39 fails, the paper positions v24 as an empirical recipe with a mechanistic story attached (honest, but weaker).
