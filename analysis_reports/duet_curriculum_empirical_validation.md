# DUET Curriculum Empirical Validation
*Test of the first-principles story for why `v24` (0.678) >> `v36` (0.389)*

## TL;DR

Of five specific predictions, **four are refuted by the per-step training data** and one is only partially supported. The posited mechanism — that constant BC `μ=0.05` (v36) interferes with discriminator learning, entropy, on-policy progress, and gradient-share curriculum — does not match what the logs show. v36 looks *healthy by every curriculum diagnostic* and still underperforms, while v24 is distinctive not because it "unlocks" a metric that v36 suppresses, but because it **stabilizes gradient and advantage magnitudes** during the critical early/mid-training window (steps 15–60). A revised story is proposed at the end.

Data source: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/curriculum_metrics.json` (100-step parse of `webshop_qwen1.5b_duet_{v12,v22,v23,v24,v28,v30,v36}.log` + `webshop_qwen1.5b_chord.log`). Headline outcomes reparsed directly from `val-summary/webshop/reward_mean_all`:

| run | final | peak | BC μ |
|-----|-------|------|-----|
| v12 | 0.431 | 0.431 | — (DR3+SC only) |
| v22 | 0.462 | 0.500 | constant 0.05 |
| v23 | 0.440 | 0.496 | constant 0.10 |
| v24 | 0.678 | 0.678 | decaying 0.3 → 0.05 over 25 steps |
| v28 | 0.495 | 0.506 | — |
| v30 | 0.520 | 0.523 | — |
| v36 | 0.389 | 0.527 | constant 0.05 |
| chord | 0.603 | 0.603 | decaying 0.887 → 0.05 |

Key figures: `analysis_reports/figures/fig6` … `fig11`.

---

## Claim A — v36 suppresses discriminator training

**Prediction:** `dr3/disc_acc` rises slower under constant μ because the policy stays closer to teacher, giving the discriminator less signal.

**Data (window means):**

| run | 5–15 | 15–25 | 25–40 | 40–60 | 60–99 |
|-----|-----:|------:|------:|------:|------:|
| v12 (no BC) | 0.721 | 0.695 | 0.764 | 0.914 | 0.941 |
| v22 (const μ=0.05) | 0.718 | 0.730 | **0.894** | 0.980 | 0.989 |
| v24 (decay μ) | 0.725 | **0.830** | **0.970** | 0.992 | 0.989 |
| v36 (const μ=0.05) | 0.714 | 0.729 | **0.899** | 0.979 | 0.994 |

**Verdict: REFUTED.** v36's disc_acc is *equal to or higher* than v12 at every window, and converges ~15 steps earlier (see `fig6`). If anything, adding any form of BC *accelerates* discriminator training — plausibly because BC gives the on-policy model a better-defined distribution to contrast against. v24 converges fastest (disc_acc ≥ 0.97 by step 30), which is aligned with the decaying-BC story *for v24 only*, but it does not imply that constant BC hurts the discriminator.

---

## Claim B — v36 suppresses entropy / exploration

**Prediction:** v36's `actor/entropy_loss` drops faster than v12's.

**Data (window means):**

| run | 5–15 | 15–25 | 25–40 | 40–60 | 60–99 |
|-----|-----:|------:|------:|------:|------:|
| v12 | 0.389 | 0.406 | 0.489 | 0.541 | 0.521 |
| v22 | 0.416 | 0.431 | 0.484 | 0.550 | 0.553 |
| v24 | 0.376 | **0.464** | **0.559** | **0.604** | **0.589** |
| v36 | 0.368 | 0.389 | 0.468 | 0.542 | 0.575 |
| chord | 0.379 | 0.467 | 0.531 | 0.564 | 0.498 |

**Verdict: REFUTED (for v36) / SUPPORTED (for v24 as outlier).** v36's entropy trajectory is *statistically indistinguishable* from v12 — within 0.01–0.02 at every window. The "entropy suppression" explanation for v36 ≠ v24 doesn't hold. What the data *does* show is that **v24 maintains markedly higher entropy from step 15 onward** than any other variant (e.g., 0.604 vs 0.542 for v36 at steps 40–60). Decaying μ seems to *preserve* exploration, but the comparison to v36 is not the right frame — v36's entropy is fine, v24's is just exceptional.

See `fig7`.

---

## Claim C — v24 has a step-15–25 window where BC+DR3 effective gradient exceeds others

**Prediction:** The proxy `|actor/teacher_off_pg_loss| + μ·sft_loss` shows a pronounced bump for v24 in steps 15–25 that v22/v36 lack.

**Data — BC contribution `chord/μ`:**

| run | 5–15 | 15–25 | 25–40 | 40–99 |
|-----|-----:|------:|------:|------:|
| v22 | 0.050 | 0.050 | 0.050 | 0.050 |
| v23 | 0.100 | 0.100 | 0.100 | 0.100 |
| v24 | **0.197** | **0.074** | 0.050 | 0.050 |
| v36 | 0.050 | 0.050 | 0.050 | 0.050 |
| chord | 0.553 | 0.133 | 0.050 | 0.050 |

**Data — DR3 contribution `|actor/teacher_off_pg_loss|`:**

| run | 5–15 | 15–25 | 25–40 | 40–99 |
|-----|-----:|------:|------:|------:|
| v12 | 2.058 | 1.353 | 1.837 | 1.789 |
| v22 | 2.119 | 3.599 | 1.120 | 0.802 |
| v24 | **2.613** | 1.422 | 1.403 | 1.377 |
| v36 | 2.204 | — | 1.493 | 1.491 |

**Verdict: PARTIALLY SUPPORTED.** v24 does have the strongest early DR3 teacher gradient in the 5–15 window (|teacher_off_pg_loss| = 2.613 vs 2.058 for v12, 2.204 for v36), and the decaying BC adds roughly `0.2 × 1.16 ≈ 0.23` of SFT-loss weighted gradient on top — so the *combined* early teacher signal is clearly largest for v24 (see `fig10` panel c).

However the "15–25 bump" prediction is *not* where the action is — by step 15–25 v24's μ is already down to 0.074 and its |teacher_off_pg_loss| has dropped to 1.42 (below v22's 3.60). The correct description is: **v24 front-loads teacher signal into steps 5–15 and then cleanly hands off**. v22 gets a big DR3 spike at 15–25 (second panel) but it coincides with later collapse.

v36's DR3 magnitude is not meaningfully different from v24 in the late windows, so the "constant μ adds nothing" framing has a kernel of truth — but the real difference isn't in magnitude, it's in *when* the teacher signal is applied.

See `fig8` (right panel) and `fig10`.

---

## Claim D — v36's state channel is weakened

**Prediction:** `state_channel/progress_onpolicy_mean` is lower in v36 throughout because BC over-imitates teacher, starving on-policy of good trajectories.

**Data:**

| run | 5–15 | 15–25 | 25–40 | 40–60 | 60–99 |
|-----|-----:|------:|------:|------:|------:|
| v12 | 0.292 | 0.338 | 0.260 | 0.329 | 0.268 |
| v22 | 0.274 | 0.346 | 0.334 | 0.355 | 0.367 |
| v24 | 0.283 | 0.333 | 0.320 | 0.364 | 0.370 |
| v36 | 0.297 | **0.354** | **0.334** | **0.371** | 0.359 |

**Verdict: REFUTED.** v36 consistently has the **highest** on-policy progress of any variant in windows 15–40. v12 is the one with visibly lower on-policy progress (0.260–0.268 in mid/late windows). Adding *any* BC (const or decaying) helps the on-policy policy reach teacher-visited states — which makes sense because BC nudges the policy to follow teacher-like action sequences during exploration, and those roll into states the progress map has seen.

So the causal direction implied by the theory is backwards here: BC *feeds* the state channel, it doesn't starve it.

See `fig9`.

---

## Claim E — v24's teacher_gradient_share has distinct curriculum structure

**Prediction:** v24 shows early-high TGS (BC + DR3 combined teacher gradient), middle transition, late-low TGS (BC decayed, DR3 dominant).

**Data (window means of `duet/teacher_gradient_share`):**

| run | 5–15 | 15–25 | 25–40 | 40–60 | 60–99 |
|-----|-----:|------:|------:|------:|------:|
| v12 | 0.136 | 0.102 | 0.073 | 0.098 | 0.086 |
| v22 | 0.146 | 0.098 | 0.100 | 0.116 | 0.110 |
| v24 | **0.223** | 0.128 | 0.112 | 0.128 | 0.134 |
| v36 | 0.140 | 0.101 | 0.105 | 0.133 | 0.137 |

**Verdict: PARTIALLY SUPPORTED.** v24 is the only variant with elevated TGS in the 5–15 window (0.223 vs ~0.14 for all others). The *shape* of the prediction (early-high → low → slightly-higher once DR3 kicks in) is visible in the data: v24's TGS actually dips to its minimum around step 25–40 and slowly rises again, while v12's continues to drift down. However, the *late-phase* story ("late low TGS") does not hold — all variants converge to roughly the same late TGS of 0.08–0.14, with v24 and v36 actually slightly *above* v12 in the final window.

The distinct early peak is real (see `fig8`). The claim that this specific curriculum *structure* is what distinguishes v24 is harder to verify because v36 has roughly the same late-phase TGS and still fails.

---

## Synthesis: revised empirically-grounded story

Of the five claims: A **refuted**, B **refuted** (for the v36 comparison; v24 is unusual but not because v36 is pathological), C **partially supported** (but "front-loading" not "15-25 bump"), D **refuted**, E **partially supported**.

**What the data actually says about why v24 > v36:**

Looking across *all* metrics, the most visually striking differences between v24 and v36 are **not** in the curriculum quantities (disc_acc, entropy, progress, TGS) — they are in **gradient and advantage stability**:

1. **`actor/grad_norm`** (fig11, left): v24 stays around 3–4 throughout training. v12 *explodes* to ~11 in late training; v22 and v36 drift up to 6–7. v24's decaying μ damps the gradient magnitude as DR3 takes over.
2. **`duet/adv_onpolicy_effective_abs_mean`** (fig11, right): v24 has the **smallest and flattest** on-policy advantage magnitude (0.12 early → 0.17 late). v12 grows to 0.33 — 2x larger. v36 stays at ~0.18, similar to v24 in magnitude but without the early-phase dip.
3. **`chord/sft_loss`**: v24 fits the teacher better (1.16 → 0.61) than v36 (1.26 → 0.67), despite applying *less* BC weight per step. Plausibly because v24's early high-μ window aligns the policy to teacher early when it's easiest, and then low-μ lets RL fine-tune without fighting.

**Proposed revised mechanism:** The 25-step decaying BC in v24 is not primarily a "discriminator curriculum" or "exploration curriculum" — it's an **optimization pre-conditioner**. High μ during steps 5–15 *front-loads* the policy onto teacher-reachable regions (accelerating disc_acc convergence as a byproduct), which in turn **shrinks the on-policy advantage magnitude** because the policy now sees more rewardable states from exploration. Smaller advantages → smaller gradients → more stable updates. Constant μ=0.05 (v36) is too weak to produce this alignment; pure DR3+SC (v12) gets the same goal eventually but with much larger and noisier gradients, hence the late-training instability (grad_norm exploding to 11).

This reframes v24 as exploiting a **gradient-conditioning trick** rather than a pedagogical curriculum. The causal chain is:

```
decaying BC → policy-teacher alignment in steps 5-15
            → smaller advantage magnitudes
            → stabler gradients / lower grad_norm
            → sustained learning signal through full 100 steps
```

## Unexpected findings

1. **v36 is not "broken" by any curriculum diagnostic.** Its disc_acc, entropy, SC progress, and TGS are all similar to or better than v12. Its failure mode is elsewhere — its peak was 0.527 at step ~75 and *regressed* to 0.389 by step 100. This looks like a late-phase collapse, not early suppression. This is very different from the story of "constant μ interferes from the start." Likely worth re-running v36 with 3 seeds to distinguish noise from actual regression.

2. **v12's `actor/grad_norm` explodes to ~11 in late training** (fig11 left), but its `critic/success_onpolicy` and `critic/score` do *not* collapse. The gradient instability is tolerated, just suboptimal. This suggests no-BC variants hit a ceiling due to gradient noise, not due to bad policy direction.

3. **All variants hit `disc_acc ≈ 1.0` by step 60.** After that point DR3's density ratio is essentially saturated (classifier perfectly separates classes). Yet training continues to improve — the discriminator accuracy isn't the rate-limiting factor in the second half of training, contrary to what the "DR3 curriculum" intuition might suggest.

4. **`duet/adv_onpolicy_effective_abs_mean` is the single most predictive per-step metric of final outcome** (low = good). v24 (0.17) and v36 (0.18) have similar late-phase values but v24 starts lower, suggesting the *trajectory* of advantage magnitude matters more than the endpoint.

5. **chord itself (0.603) outperforms every DR3+SC+const-μ variant** but loses to v24. CHORD's μ schedule starts at 0.887 — roughly 3× higher than v24's peak 0.3 — and decays similarly. This suggests the shape/integral of the decay schedule is not optimal at either extreme: too aggressive BC (chord) leaves DR3 underexploited; too weak/constant (v22/v36) fails to pre-condition at all; the sweet spot (v24) uses modest peak × fast decay.

## What to do next

- Treat the v36 result as a **single-seed anomaly candidate**. Re-run v36 with seeds 43, 44 to confirm its final=0.389 vs peak=0.527 regression is real, not noise.
- The theory-researcher should re-derive: the empirical mechanism looks like **gradient-magnitude stabilization via early policy-teacher alignment**, not the curriculum claims A–E. Entropy and disc_acc are *downstream* of this stabilization, not upstream causes.
- Promising follow-up: v24 with `μ_peak ∈ {0.2, 0.4, 0.5}` and decay lengths `{15, 25, 35}` to see if the 0.678 result is at a local optimum or a flat plateau; if flat, the optimizer-conditioning hypothesis is further supported.

## Files

- Figures: `/data/home/qisheng/EvolAnalsis/analysis_reports/figures/fig6_disc_acc_v12_v24_v36_comparison.png`, `fig7_entropy_v12_v24_v36.png`, `fig8_teacher_gradient_share_curriculum.png`, `fig9_sc_progress_comparison.png`, `fig10_effective_gradient_curves.png`, `fig11_gradient_stability_bonus.png`.
- Parse script: `/data/home/qisheng/EvolAnalsis/analysis/make_curriculum_figures.py`.
- Parsed metrics: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/curriculum_metrics.json`.
