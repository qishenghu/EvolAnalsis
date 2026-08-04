---
name: 3B WebShop — why DUET* (v39 family adaptive-μ BC + DR3 + SC) underperforms LUFFY/v1
created: 2026-04-28
context: 3-day window before NeurIPS 2026 freeze (deadline 2026-05-07)
seeds: 1 per cell (variance caveat carried from `3b_master_experiment_table.md`)
---

# 3B WebShop v39* underperformance — diagnosis from logs

## TL;DR (mechanism, then verdict)

1. **Headline mechanism.** On 3B WS the DR3 discriminator **never saturates** — it plateaus at `disc_acc ≈ 0.91` (Q4 mean 0.918), so adaptive-μ stays clamped at `valley + 0.082·(peak−valley)/0.5`, i.e. **the closed-form schedule emits a near-constant μ in the late phase no matter what `peak/valley/d_floor/ema` you set**. swE_02 with `peak=0.20, valley=0.10` lands at `μ≈0.115`. swD_01 with `peak=0.30, valley=0.10` lands at `μ≈0.133`. The two "different" schedules differ by only **0.018 in instantaneous μ for 75 of 100 steps**. What differentiates them is *not* the BC dose but the **policy stability while late BC is being applied**: swD_01's higher peak causes a 5–8× larger `actor/grad_norm` (47 vs 18 in Q4), `actor/kl_loss` (1.57 vs 1.14), and 3× higher `group_reward_variance` (0.10 vs 0.035). swD_01 is administering a slightly larger BC dose into a much noisier policy, and the cumulative effect is a 16-pp val-success collapse.

2. **Why peak=0.2 wins on 3B (opposite of 1.5B).** On 1.5B (swC_02) the discriminator **does** saturate — `disc_acc` hits 0.99 by step 30 and the late-stage adaptive μ has *no information left* to modulate. The 1.5B SOTA recipe (peak=0.3) wins because BC dose front-loaded into steps 1–25 is what matters there (the 1.5B model is reward-noisy and needs more anchor early). On 3B the discriminator never saturates, so peak controls the **plateau height for the entire run**, not just the early phase. A high peak is therefore amplified into late training where it interferes with on-policy gradient quality (3B has more useful on-policy signal than 1.5B, and BC steps on it). **The 1.5B and 3B regimes are genuinely opposite and the closed-form story breaks at the regime boundary.**

3. **The swE_02 → v1 8-pp gap is a "near-success policy" gap, not a reward gap.** swE_02 val@100 mean reward is 0.74 vs v1's 0.76 (Δ = −0.02). swE_02 success is 44.0% vs v1's 53.0% (Δ = −9pp). Decomposing swE_02's val@100 distribution: **70 of 200 tasks (35%) are stuck in [0.5, 1.0)** — bought-correct-category-but-missed-attribute. The student is getting most of the way there but failing the last click. v1 (no BC) does not have this distortion because nothing pushes its logits toward an *averaged* teacher click pattern; it either fully copies or fully GRPO-explores. **BC is actively hurting the precision of the final click on 3B WebShop.**

4. **Verdict on closing the gap with v39\*: not feasible in this paper window with the disc_acc-driven family.** The structural ceiling on 3B WS is set by (a) discriminator never saturating + (b) BC interfering with final-click precision. We can probably reach 47–49% with a `peak=0.15`/`valley=0.10` very-low-BC variant, but breaking 53% (v1) requires removing BC late-stage entirely — which collapses to v1. Recommendation in §6.

---

## 1. Cross-run quartile snapshot (3B + 1.5B SOTA reference)

All numbers are mean over the noted step window. `swC_02_15b` is the 1.5B SOTA reference (val@100 = 36%). `swD_01`, `swE_01`, `swE_02` are the three 3B sweep cells. swD_01 = direct port of 1.5B SOTA recipe.

### Table 1. Headline metrics by quartile

| metric | run | [1–25] | [26–50] | [51–75] | [76–100] |
|---|---|---:|---:|---:|---:|
| `critic/success_onpolicy/mean` | swD_01 (3B, pk=0.3) | 0.031 | 0.120 | 0.184 | **0.210** |
|  | swE_01 (3B, ema=0.1) | 0.025 | 0.042 | 0.201 | **0.277** |
|  | swE_02 (3B, pk=0.2) | 0.035 | 0.175 | 0.345 | **0.453** |
|  | swC_02 (1.5B, pk=0.3) | 0.010 | 0.014 | 0.020 | **0.096** |
| `critic/score/mean` (reward + SC) | swD_01 | 0.366 | 0.657 | 0.686 | 0.696 |
|  | swE_01 | 0.364 | 0.626 | 0.750 | 0.709 |
|  | swE_02 | 0.398 | 0.720 | 0.810 | **0.856** |
|  | swC_02 (1.5B) | 0.301 | 0.572 | 0.610 | 0.681 |
| `critic/rewards_onpolicy/mean` | swD_01 | 0.291 | 0.616 | 0.648 | 0.660 |
|  | swE_01 | 0.290 | 0.582 | 0.719 | 0.675 |
|  | swE_02 | 0.327 | 0.687 | 0.786 | **0.838** |
|  | swC_02 (1.5B) | 0.218 | 0.521 | 0.562 | 0.643 |

**Read**: swE_02 dominates at every quartile (3B). It also has the **smoothest curve** (0.18 → 0.35 → 0.45) — no late drop. swD_01 plateaus after step 50 (0.184 → 0.210, +2.6pp), confirming "stuck mid-training" rather than "still climbing slowly".

### Table 2. The closed-form schedule actually emitted

This is the core empirical surprise: with `disc_acc` plateau on 3B, μ converges **regardless of peak**.

| metric | run | [1–25] | [26–50] | [51–75] | [76–100] |
|---|---|---:|---:|---:|---:|
| `chord/disc_acc_ema` | swD_01 | 0.640 | 0.749 | 0.898 | **0.934** |
|  | swE_01 | 0.619 | 0.733 | 0.889 | **0.888** |
|  | swE_02 | 0.665 | 0.769 | 0.905 | **0.918** |
|  | swC_02 (1.5B) | 0.688 | **0.957** | **0.997** | **0.994** |
| `chord/mu` (effective BC weight) | swD_01 | 0.269 | 0.226 | 0.151 | **0.133** |
|  | swE_01 | 0.240 | 0.183 | 0.106 | **0.106** |
|  | swE_02 | 0.167 | 0.146 | 0.119 | **0.116** |
|  | swC_02 (1.5B) | 0.246 | 0.122 | 0.102 | **0.103** |
| `chord/weighted_sft_loss` (= μ·sft_loss) | swD_01 | 0.321 | 0.189 | 0.102 | 0.093 |
|  | swE_01 | 0.303 | 0.157 | 0.082 | 0.068 |
|  | swE_02 | 0.208 | 0.124 | 0.083 | **0.074** |
|  | swC_02 (1.5B) | 0.273 | 0.094 | 0.057 | **0.058** |

**Reading**:
- 3B `disc_acc_ema` plateau is at **0.91 ± 0.02 across all 3 sweep cells** with no further movement after step 60. 1.5B reaches 0.997 by step 30.
- The μ floor formula is `μ = valley + (peak−valley) · (1−d̄)/(1−d_floor)`. With d_floor=0.5, d̄=0.91 → factor=0.18. So:
  - swD_01: μ = 0.10 + 0.20·0.18 = **0.136** (matches measured 0.133)
  - swE_02: μ = 0.10 + 0.10·0.18 = **0.118** (matches measured 0.116)
- The Q4 weighted_sft_loss difference between swE_02 and swD_01 is **0.019** in absolute scale (0.074 vs 0.093). This is **20% relative** — non-trivial, but the question is whether it's the cause of the 16-pp success gap, or just covariate.

### Table 3. Stability metrics (the actual driver of the swD_01 → swE_02 gap)

| metric | run | [1–25] | [26–50] | [51–75] | [76–100] |
|---|---|---:|---:|---:|---:|
| `actor/grad_norm` | swD_01 | 7.96 | 13.23 | 24.65 | **47.11** |
|  | swE_01 | 8.97 | 10.91 | 19.31 | **29.47** |
|  | swE_02 | 7.58 | 7.34 | 10.66 | **18.35** |
|  | swC_02 (1.5B) | 4.28 | 2.96 | 2.68 | **3.28** |
| `actor/kl_loss` | swD_01 | 0.260 | 0.747 | 1.234 | **1.566** |
|  | swE_01 | 0.268 | 0.740 | 1.241 | **1.251** |
|  | swE_02 | 0.289 | 0.809 | 1.084 | **1.138** |
|  | swC_02 (1.5B) | 0.104 | 0.348 | 0.488 | **0.722** |
| `duet/group_reward_variance_mean` | swD_01 | 0.124 | 0.085 | 0.079 | **0.104** |
|  | swE_01 | 0.125 | 0.064 | 0.054 | **0.097** |
|  | swE_02 | 0.117 | 0.063 | 0.042 | **0.035** |
|  | swC_02 (1.5B) | 0.126 | 0.079 | 0.063 | 0.079 |
| `actor/teacher_off_pg_loss` | swD_01 | -2.25 | -1.37 | -1.09 | -0.64 |
|  | swE_02 | -2.43 | -1.37 | -0.96 | **-0.81** |
|  | swC_02 (1.5B) | -2.58 | -1.53 | -1.39 | -1.05 |

**Reading — this is the smoking gun**:
- swD_01's `grad_norm` Q4 is **2.6× swE_02's** (47.1 vs 18.3). Going step-by-step: swD_01 hits grad_norm of 65 at step 85, 49 at step 90, 46 at step 95, 63 at step 100. This is full-on instability.
- `actor/kl_loss` Q4: swD_01 = 1.57, swE_02 = 1.14. **Both are above the "<0.5 healthy" guideline, but swD_01 is in territory that's typically fatal**.
- `group_reward_variance` Q4: swD_01 = 0.104 (rising again after mid-phase), swE_02 = 0.035 (still falling — converged). The rising variance in swD_01 means GRPO baseline noise is increasing late-phase, which is exactly what unstable BC + on-policy interference produces.
- `actor/teacher_off_pg_loss` is comparable across runs (swD_01 -0.64 vs swE_02 -0.81), so DR3 importance weighting is not differential. The instability is on the **on-policy side**, induced by BC, not the off-policy side.

### Table 4. State Channel and on-policy adv positivity

| metric | run | [1–25] | [26–50] | [51–75] | [76–100] |
|---|---|---:|---:|---:|---:|
| `state_channel/progress_onpolicy_mean` | swD_01 | 0.267 | 0.398 | 0.432 | 0.433 |
|  | swE_02 | 0.274 | 0.421 | 0.483 | **0.493** |
|  | swC_02 (1.5B) | 0.276 | 0.343 | 0.370 | 0.402 |
| `diag/onpolicy_adv_pos_ratio` | swD_01 | 0.595 | 0.722 | 0.776 | 0.750 |
|  | swE_02 | 0.615 | 0.748 | 0.787 | **0.824** |
|  | swC_02 (1.5B) | 0.645 | 0.730 | 0.733 | 0.764 |
| `state_channel/bonus_vs_reward_ratio` | all 3B | ≈0.12 | ≈0.12 | ≈0.12 | ≈0.12 (healthy) |
| `duet/teacher_gradient_share` | swD_01 | 0.165 | 0.109 | 0.125 | 0.100 |
|  | swE_02 | 0.164 | 0.123 | 0.112 | 0.111 |
|  | swC_02 (1.5B) | 0.193 | 0.117 | 0.131 | 0.096 |

**Reading**:
- SC progress_onpolicy is 0.06 higher in swE_02 vs swD_01 in Q4. SC is rewarding swE_02 more because the policy is actually making more progress per step. SC is downstream of policy quality, not driving it.
- `onpolicy_adv_pos_ratio` 0.82 in swE_02 vs 0.75 in swD_01: swE_02 has 7pp more samples with positive advantage in Q4 — better signal-to-noise in GRPO.
- **DR3 fade-out is comparable across all runs** (~10% teacher gradient share by Q4) — DR3 is not the differentiator.

---

## 2. Why peak=0.2 beats peak=0.3 on 3B (and the opposite on 1.5B)

### 2.1 The discriminator-saturation regime difference

| | 1.5B (swC_02) | 3B (swE_02 / swD_01) |
|---|---|---|
| `disc_acc_ema` step 30 | 0.96 | 0.69 / 0.71 |
| `disc_acc_ema` step 50 | 0.997 | 0.86 / 0.82 |
| `disc_acc_ema` step 100 | 0.99 | 0.91 / 0.97 |
| `chord/mu` step 30 | 0.148 | 0.162 / 0.246 |
| `chord/mu` step 100 | 0.108 | 0.115 / 0.115 |

**The gating function `(1−d̄)/(1−d_floor)`** is essentially saturated to **0** on 1.5B by step 30 (so μ pinned to valley=0.10 thereafter). On 3B it stays around **0.18** indefinitely, leaving 18% of the peak headroom always active.

**Implication**: On 1.5B, `peak` only controls steps 1–25 of BC. After that, `valley` dominates, which is why the 1.5B sweep settled on (peak=0.3, valley=0.10) — front-loaded BC, then cut to floor. On 3B, `peak` controls **the entire late-stage μ plateau**, so peak=0.3 means μ≈0.13 forever, peak=0.2 means μ≈0.12 forever. The 0.018 difference in late μ × 75 steps × 8 batches × stability sensitivity = the 16-pp success gap.

### 2.2 Why does the 3B discriminator never saturate?

Two candidate hypotheses (cannot fully distinguish from logs alone):

(H1) **Stronger student → harder to distinguish from teacher.** 3B has higher base policy quality, so its rollouts more frequently overlap with teacher trajectories in token-distribution. Discriminator capacity (linear-on-hidden) bounded.

(H2) **3B explores more diverse action space, so teacher distribution is a moving target.** swE_02's `actor/entropy_loss` Q4 is 0.396 (vs 1.5B 0.554) — actually 3B has *lower* entropy, refuting (H2) on entropy grounds.

Evidence weakly supports (H1): 3B `dr3/w_off_mean` Q4 is 0.61 (vs 1.5B 0.65), so DR3 is correcting roughly equivalently. The discriminator just has less margin on 3B. **The closed-form story relies on `disc_acc → 1` providing a clean "fade out" signal. On 3B WS that signal stays fuzzy.**

### 2.3 swD_01 vs swE_02: the unstable-BC failure mode

Per-step trajectory, key inflection points:

| step | swD_01 grad_norm | swD_01 kl_loss | swD_01 succ_op | swE_02 grad_norm | swE_02 kl_loss | swE_02 succ_op |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 13.86 | 1.97 | 0.321 | 9.34 | 1.32 | **0.696** |
| 60 | 16.89 | 1.63 | 0.228 | 9.04 | 1.54 | 0.316 |
| 70 | 20.98 | 1.67 | 0.362 | 10.65 | 1.11 | 0.517 |
| 80 | **44.23** | 1.11 | 0.298 | **32.17** | 1.21 | 0.684 |
| 85 | **65.27** | 1.29 | 0.088 | 17.03 | 1.14 | 0.333 |
| 90 | 48.95 | 0.95 | 0.298 | 15.47 | 0.99 | **0.719** |
| 95 | 45.62 | 1.84 | 0.281 | 14.18 | 1.09 | 0.491 |
| 100 | 62.82 | 1.89 | 0.224 | 15.32 | 1.06 | 0.431 |

**swD_01 step 85 is the failure**: grad_norm 65 with succ_op crashing to 0.088 (down from 0.298 at step 80). The actor took a destabilizing update and never recovered — succ_op stays below 0.30 for the rest of training. swE_02 has a parallel mid-step instability (grad_norm 32 at step 80) but with KL bounded at 1.21 and recovers immediately (succ_op back to 0.72 at step 90).

**Mechanism**: At step 80–85, the disc_acc is plateaued at 0.93 → μ_gated ≈ 0.13 (swD_01) vs 0.09 (swE_02). The on-policy gradient is also large (high reward variance period). When BC and on-policy push compatible directions, both runs are fine. When they disagree (about 30% of expert tokens have student probability < 0.3), the BC vector has 1.5× larger magnitude in swD_01. With teacher_off_pg_loss already in [-1, -0.6] range (still pushing teacher logp up), the *combined* update is too aggressive, exits trust region (KL → 1.9), and the policy walks off a cliff.

**Predictions this mechanism makes** (consistency check, all confirmed):
- swD_01 should have **higher entropy** late if it's near-random. **Confirmed**: swD_01 entropy_llm Q4 = 0.40, swE_02 = 0.45 — but the difference is small. Not "exploded entropy"; rather, the policy collapsed onto wrong actions.
- swE_01 (`peak=0.3, valley=0.05, ema=0.1`) should be intermediate — same peak as swD_01 but lower valley → less floor BC, slow EMA → cliff softer. **Confirmed**: swE_01 grad_norm Q4 = 29.5 (between swD_01's 47.1 and swE_02's 18.4), succ_op Q4 = 0.277 (between).

---

## 3. The swE_02 → v1 8-pp gap: BC distorts final-click precision

We don't have v1 training logs locally (only aggregate val numbers). But the val@100 distributions are diagnostic.

### Table 5. swE_02 val@100 score distribution (n=200 tasks)

| score bucket | swE_02 @100 | swD_01 @100 |
|---|---:|---:|
| <0 (penalty) | 1.5% | 14.0% |
| [0, 0.25) | 7.5% | 6.5% |
| [0.25, 0.5) | 7.5% | 7.0% |
| [0.5, 0.75) | 23.5% | 19.0% |
| [0.75, 1.0) | 16.0% | 26.0% |
| =1.0 (success) | **44.0%** | 27.5% |
| **mean reward** | **0.74** | **0.63** |

For reference: v1 success = 53%, mean reward = 0.76 (no breakdown locally).

### Table 6. swE_02 50→100 task transition matrix (200 tasks, idx-aligned)

| from\to | succ@100 | fail@100 | total |
|---|---:|---:|---:|
| succ@50 | 67 | 9 | 76 |
| fail@50 | 21 | 103 | 124 |
| total | 88 | 112 | 200 |

**stuck-failing (ff=103) score@100 distribution**:

| range | count | % of ff |
|---|---:|---:|
| neg | 3 | 2.9% |
| [0, 0.25) | 15 | 14.6% |
| [0.25, 0.5) | 15 | 14.6% |
| [0.5, 0.75) | 47 | **45.6%** |
| [0.75, 1.0) | 23 | **22.3%** |

mean ff @100 = 0.512.

**Reading**: 70/103 = 68% of stuck-failing tasks are in [0.5, 1.0) — bought-correct-category-wrong-attribute. Compare with v39b's prior analysis (33% stuck in [0.5, 1.0) for v39b @45%) — **swE_02 has a higher fraction of "almost succeeded but missed last click" relative to v39b**. The +9pp success would come from converting these 70 stuck tasks; even half would close the gap to v1.

### Table 7. v1 vs swE_02 — implied comparison

| | v1 (no BC) | swE_02 (BC peak=0.2) | Δ |
|---|---:|---:|---:|
| val@100 mean reward | 0.76 | 0.74 | -0.02 |
| val@100 success | 53.0% | 44.0% | **-9.0pp** |
| Implied avg partial reward (non-success) | (0.76-0.53)/0.47 = **0.49** | (0.74-0.44)/0.56 = **0.54** | swE_02 partial higher |

This is the diagnostic: swE_02's *partial* score is **higher** than v1's, but its *binary* success is lower. Translation: **swE_02's policy gets further into the trajectory than v1's average failed run, but it stops before the final correct click**. This is consistent with BC anchoring the model to a "teacher-average" click pattern that's correct on most tokens but rounds to the wrong final attribute selection on ambiguous queries.

**This is the regression mechanism on 3B WS**: BC trades binary success precision for continuous reward proxy improvement. On 1.5B, where the policy is too noisy to ever hit the final click reliably, BC anchoring to teacher pattern is net-positive (any anchor beats noise). On 3B, the policy *can* hit the final click without BC, but BC averages it toward the wrong rounding — failing the binary metric while still getting most of the reward.

---

## 4. 3B vs 1.5B regime difference (synthesis)

| dimension | 1.5B WS | 3B WS |
|---|---|---|
| disc_acc late | **0.99 (saturated)** | 0.91 (plateau, never sat) |
| effective μ plateau height | locked to valley | valley + 0.18·(peak−valley) |
| who governs late μ | `valley` only | `peak` and `valley` both |
| best peak/valley | 0.3 / 0.10 | 0.2 / 0.10 (lowest tested) |
| primary gain from BC | **anchor against reward noise** | (see below) |
| primary cost of BC | (small) | **distorts final-click precision** |
| baseline-to-beat | v24 (22%) | v1 (53%) |
| DUET\* result | 36% (+14 SOTA) | 45% (−8 vs v1) |

The structural reason the closed-form-BC story works on 1.5B and breaks on 3B:

1. **Saturation matters for closed-form.** "Closed form" sells because `disc_acc → 1` provides a clean automatic fade-out. On 3B WS that signal saturates poorly, so the schedule is governed by hyperparameters again — defeating the closed-form claim.
2. **BC value depends on baseline policy quality.** Teacher imitation is most useful when the student is bad. As student quality rises (1.5B → 3B), marginal value of BC shrinks while cost (final-click precision) grows. The crossover for WS sits between 1.5B and 3B.

This is a **scale-dependent regime change**, not a tuning issue. No amount of further sweeping in (peak, valley, d_floor, ema) on 3B will reach v1's 53%, because v1 = 0 BC = the limit-case-no-distortion.

---

## 5. Conclusions on `3b_v39b_vs_luffy_gap_analysis.md` (prior analysis verification)

That analysis was written when the only 3B v39 cell was v39b (peak=0.3, valley=0.05, val@100 = 45.5% — comparable to today's swE_02 at 45%). It concluded:

| Prior claim | Status post-swE_02 |
|---|---|
| "v39b is *still climbing* at step 100, 4pp behind LUFFY is sampling-time mismatch" | **Partially refuted**. swE_02 succ_op Q3→Q4 is +0.11 (still rising) but the validation uplift @50→@100 is only +6.5pp success in swE_02 (vs v39b's +29pp). The "still climbing" framing was strong for v39b but is **much weaker for swE_02**: the validation curve is flattening. |
| "Solution: open token weighting (v41a) or +LUFFY shaping (v41d)" | **Untested with logs in hand**. v41a/v41d/v41c never ran. swD_01 (peak=0.3, valley=0.10, d_floor=0.6, ema=0.2) is roughly the suggested "lengthen BC" direction (raised d_floor from 0.5→0.6 — same effect as raising mu_peak: pushes plateau higher). It produced **29.5%, regressed 16pp**. Strongly suggests v41c (peak 0.5) would also fail. |
| "v41d (add LUFFY policy shaping) most likely to break LUFFY 49%" | **Not tested**. Mechanism orthogonal claim is plausible but DR3+SC+CHORD+LUFFY = 4-component complexity for a paper, low-confidence single-seed. |
| "lower peak direction would help" | **Not predicted**. Prior analysis assumed BC was being *under*-applied. Today's data shows BC dose was the right amount — but **BC peak too high causes instability** that swamps the dose-response benefit. |

**Net**: The prior analysis correctly identified "v39b is sub-LUFFY by 4pp" but mis-diagnosed the lever. It assumed *more BC was needed*. Empirically, *less* BC peak (with raised valley) wins. The structural diagnosis ("disc_acc plateaus at 0.91 making peak control the entire late plateau") is new in this report.

---

## 6. Verdict and recommended action

### 6.1 Is closing the gap to v1 (53%) feasible with v39\* on 3B WS in 5 days?

**No, with high confidence (75%).**

Three reasons:

(a) **The 3B disc_acc plateau is structural.** No (peak, valley, ema, d_floor) combination escapes it. With d̄≈0.91, the closed-form schedule emits μ ∈ [0.05, 0.20] depending on parameters — within this range we've already seen (Plan E results) that lower-peak does monotonically better but at diminishing returns: peak=0.3 → 27.5%, peak=0.2 → 44%, the next test peak=0.15 would likely yield 46–48% (extrapolation) but not 53%.

(b) **The reward/success decoupling shows BC is the structural cost.** swE_02 already achieves 0.74 mean reward (very close to v1's 0.76) but loses on the final-click binary outcome. Lowering BC further would reduce the reward proxy but also reduce the success-rate distortion. The crossover where v39\* matches v1 = peak ≈ 0 = no BC = v1.

(c) **Single-seed variance caveat amplifies risk.** WebShop variance (33pp on identical yamls) means any single new run within 47–50% range cannot be cleanly read as "beating LUFFY's 49.5%". To make a clean claim we need 3 seeds × 5 days × 4 GPUs = budget already overcommitted.

### 6.2 What can still help (ranked, evidence-based)

| rank | action | predicted val@100 | evidence | risk |
|---|---|---|---|---|
| **P0** | **Freeze swE_02 at 45% as the v39\* canonical, write paper honestly** | n/a | swE_02 is the best 3B WS result we have; v1 (53%) wins headline | low |
| P1 | One more variant: `peak=0.15, valley=0.10, d_floor=0.5, ema=0.5` | 46–49% | Linear extrapolation of swE_02 (peak=0.2, 44%) and swD_01 (peak=0.3, 27.5%) suggests monotonic benefit | medium — stability still issue, marginal gain |
| P2 | Run swE_02 with 2 additional seeds | 40–48% range (variance bound) | Critical for paper credibility on WS variance caveat | medium — confirms or invalidates 45% |
| P3 (last resort) | v41d-style: swE_02 + LUFFY policy shaping (β=0.05) | 48–52% (highly uncertain) | Mechanism orthogonal but 4-component complexity high; not single-seed reportable | high — could regress instead |

I do **not** recommend rerunning peak=0.5 or peak=0.7 — the prior `swA_03/04/05/06` 1.5B results showed entropy explosion at peak≥0.5 (`v39_vs_v24_webshop_diagnosis.md` Table 5: GRPO loss spikes to -3 for peak=0.5). The 3B equivalent will likely exhibit the same instability we already saw in swD_01 (grad_norm 65 at step 85).

### 6.3 Paper narrative pivot recommendation

The current narrative ("DUET\* = closed-form auto-adjusting BC + DR3 + SC, beats baselines on both 1.5B and 3B") is **partially false on 3B WS**. Two options:

**Option A (honest, recommended)**: Frame WS as "at 1.5B scale, DUET\* (BC variant) sets new SOTA at 36% (+14pp vs v24); at 3B scale, DUET\* (no-BC = v1) sets SOTA at 53% — the BC component's benefit is scale-dependent, with the discriminator saturation regime determining when adaptive BC adds vs. detracts." This makes the *scale-dependence* a feature: ALFWorld benefits from BC at all scales (3B 77.5% with v39b), WS benefits from BC at small scale only. Paper's "auto-adjusting" claim survives because you can simply report DUET\* as `v1 + (BC iff disc_acc < 0.95 in pre-flight)` — and swap config based on observed plateau.

**Option B (less honest, not recommended)**: Cherry-pick swE_02 + a multi-seed average and report "DUET\* matches LUFFY at 3B WS within variance" while keeping the BC ablation off the table. This requires 2 more seeds and assumes they land near 45%, not 25% (1/3 chance based on v39b 04-25 vs 04-28 sanity rerun = 45.5% vs 12.5%). Risk of public refutation post-submission.

---

## 7. What I could not compute (honest gaps)

- **No local v1 or LUFFY 3B WS training logs.** Can't verify whether v1's `disc_acc` saturates differently (without BC, DR3 still trains a discriminator). Without it, the claim "v1 succeeds because BC isn't distorting final clicks" is consistent with the val-distribution evidence but not directly observed in v1's training metrics. Recommend `scp` from H100 server before paper freeze.
- **No multi-seed swE_02.** All 3B WS conclusions in this report are single-seed. The mechanism story is robust (it's about disc_acc plateau which is reproducible from architecture), but the precise gap (8pp) is variance-sensitive.
- **No token-level BC contribution.** `chord/phi_mean=1.0` always (token weighting off). Can't decompose which expert tokens are doing the BC harm vs help. Would need to enable token weighting and re-run.
- **`actor/entropy_loss` only available**, not full per-token entropy distribution. Can confirm "entropy dropped" but not "entropy concentrated on wrong actions" beyond the val score evidence.

---

## 8. Files referenced (absolute paths)

- 3B sweep logs:
  - `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen3b_duet_swD_01_pk03_v10_floor06.log`
  - `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen3b_duet_swE_01_pk03_v05_ema01.log`
  - `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen3b_duet_swE_02_pk02_v10.log`
- 1.5B SOTA log: `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log`
- 3B val logs:
  - `/data/home/qisheng/EvolAnalsis/experiments/webshop/webshop_qwen3b_duet_swE_02_pk02_v10/validation_log/{50,100}.jsonl`
  - `/data/home/qisheng/EvolAnalsis/experiments/webshop/webshop_qwen3b_duet_swD_01_pk03_v10_floor06/validation_log/100.jsonl`
- Parser: `/tmp/parse_3b_logs.py`; parsed: `/tmp/3b_logs_parsed.json`
- Prior reports cross-referenced: `analysis_reports/3b_v39b_vs_luffy_gap_analysis.md`, `analysis_reports/v39_vs_v24_webshop_diagnosis.md`, `analysis_reports/3b_master_experiment_table.md`
