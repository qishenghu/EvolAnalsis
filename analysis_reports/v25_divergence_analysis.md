# v25 Catastrophic Divergence — Root-Cause Analysis

**TL;DR**: v25 was stable and on-trend with v24 through step 97. The collapse is a **two-step cliff** (steps 98→100), not gradual drift. The actual kill vector is **not** the widened `off_cliprange_high` (that clip never binds — `off_pg_cliphit_rate = 0` throughout, same as v12/v24). The kill vector is **format-collapse feedback loop**: a late-training entropy drop plus a saturated discriminator (`disc_acc → 1.000`) combined with a ~5x teacher advantage produced one toxic gradient step that broke output formatting (step 98 emits `<story>open</story>` garbage), after which length collapsed (2053 → 959) and the onpolicy reward signal collapsed with it, yielding a grad_norm spike to 48.4. Widening the clip was a necessary enabler but not the direct trigger.

## 1. Per-step comparison table (key metrics, every 10 steps)

| step | metric                            | v12 (stable, 0.431) | v24 (stable, 0.678) | v25 (DIVERGED) |
|-----:|-----------------------------------|:-------------------:|:-------------------:|:--------------:|
|  10  | rewards_onpolicy                  | 0.22                | 0.04                | 0.16           |
|  30  | rewards_onpolicy                  | 0.26                | 0.45                | 0.37           |
|  50  | rewards_onpolicy                  | **0.72**            | 0.69                | 0.62           |
|  80  | rewards_onpolicy                  | 0.24                | **0.67**            | **0.69**       |
|  95  | rewards_onpolicy                  | 0.15                | 0.63                | 0.62           |
|  99  | rewards_onpolicy                  | 0.42                | 0.70                | **0.008**      |
| 100  | rewards_onpolicy                  | 0.51                | 0.76                | **-0.022**     |
|  50  | actor/grad_norm                   | 4.1                 | 2.9                 | 7.9            |
|  80  | actor/grad_norm                   | 8.3                 | 2.9                 | 1.5            |
|  95  | actor/grad_norm                   | 14.2                | 5.0                 | 3.6            |
|  99  | actor/grad_norm                   | 12.9                | 4.4                 | 6.7            |
| 100  | actor/grad_norm                   | 14.6                | 4.1                 | **48.4**       |
|  50  | dr3/disc_acc                      | 0.91                | 0.99                | 0.94           |
|  80  | dr3/disc_acc                      | 0.97                | 0.99                | 0.99           |
|  95  | dr3/disc_acc                      | 0.90                | 0.99                | **0.998**      |
| 100  | dr3/disc_acc                      | 0.91                | 0.99                | **1.000**      |
|  50  | actor/entropy_loss                | 0.52                | 0.57                | 0.49           |
|  95  | actor/entropy_loss                | 0.47                | 0.57                | 0.57           |
|  99  | actor/entropy_loss                | 0.49                | 0.52                | **0.41**       |
| 100  | actor/entropy_loss                | 0.49                | 0.50                | **0.39**       |
|  95  | response_length/mean              | 1377                | 2010                | 2053           |
|  99  | response_length/mean              | 1702                | 2033                | **993**        |
| 100  | response_length/mean              | 1866                | 2206                | **959**        |
| 100  | teacher_off_pg_loss               | -0.91               | -1.40               | **-2.36**      |
| 100  | on_pg_loss                        | -0.04               | +0.03               | **+0.13**      |
| 100  | progress_onpolicy_mean            | 0.28                | 0.41                | **0.04**       |
| 100  | actor/off_pg_cliphit_rate         | **0.000**           | **0.000**           | **0.000**      |

## 2. Divergence onset

The crash is **not gradual**. Clean decomposition:

- **Steps 1–97**: v25 is statistically indistinguishable from v24. Rollout jsonl inspection confirms mid-training rollouts look fine (step 50 mean_score 0.67, step 90 mean_score 0.67, similar to v24).
- **Step 98**: first visible fracture. `response_length/mean` drops 2053 → 1080 (−47%), `rewards_onpolicy` 0.62 → 0.04, `entropy_loss` 0.57 → 0.53. The policy starts emitting malformed outputs (`<story>open</story>` in rollout 99.jsonl).
- **Step 99**: propagation. Length 1080 → 993, `progress_onpolicy` 0.391 → 0.056 (state channel signal dies because the policy isn't visiting recognizable teacher states anymore).
- **Step 100**: catastrophic gradient. `grad_norm` 6.7 → **48.4** (7x overnight). `on_pg_loss` flips **positive** (+0.13) — the onpolicy gradient is now pushing *away* from the best samples because after length collapse nearly all samples are equally bad so the group-relative advantage signal inverts. `disc_acc` hits exactly 1.000 — the discriminator is perfectly certain onpolicy ≠ teacher because onpolicy is now noise.

**Single "breaking step" = 98.** Everything after is downstream.

## 3. Smoking-gun metric

The metric that most cleanly separates v25-destined-to-crash from v24-will-survive in the **pre-collapse** window (steps 80–97) is **`duet/adv_teacher_effective_mean`**:

- v24 steps 80–95: 0.10–0.23, steady, slightly declining (curriculum fading)
- v25 steps 80–95: 0.28–**0.36** — systematically 30–70% higher and *increasing*

This is the canary: even with identical DR3 clip-upper (~1.12), v25's mean teacher effective advantage is higher because the widened `off_cliprange_high=2.0` lets more of the raw teacher advantage flow through the PPO ratio clip when the ratio drifts upward. Pair that with `disc_acc` climbing toward 1.0 (v25 hits 0.998 at step 95 vs v24's 0.986) and you get a gradually-sharpening teacher gradient that the on-policy samples can't stabilize around.

Secondary canary: **`actor/teacher_off_pg_loss`** magnitude. v25 touches −3.39 at step 95 (vs v24 −1.64) — a teacher-gradient spike of 2x in the step immediately preceding collapse.

## 4. Clip activity — the surprising finding

**The PPO off-policy clip does not fire in either v12 or v25.** `actor/off_pg_cliphit_rate`, `self_off_pg_cliphit_rate`, and `teacher_off_pg_cliphit_rate` are all exactly 0.000 throughout both runs (checked at steps 1, 5, 20, 50, 80, 100). Similarly `dr3/w_clipfrac_off = 0`.

This **invalidates the stated v25 hypothesis**: "BC term's role is to escape PPO trust-region constraints on rare teacher tokens". If the clip at 0.6 is already not binding, widening it to 2.0 cannot be "releasing trapped gradient". The reported cliphit metric may be measuring sample-level clip activation (a whole micro-batch of 8 is very rarely fully outside the range), while token-level effects still differ. But from the logged signal we have, the widened clip is **not doing what was hypothesized**.

What widening *does* do: it increases the **variance** of the teacher importance weight `ratio = π/π_old` that is finally applied (because the surrogate `min(r·A, clip(r)·A)` equals `r·A` more often at high `r`). That matches the `duet/adv_teacher_effective_mean` divergence observed above.

## 5. Discriminator saturation interaction

v25's `disc_acc` reaches **0.998 by step 95** and hits **1.000 at step 100** — meaning the density ratio `w_hat` becomes a delta function. This is the known failure mode: a saturated discriminator produces w_hat that is either clipped to the upper/lower bound or a degenerate zero-variance estimate. Because v25 runs with `w_hat_ema_alpha=0.3` (less smoothing than v29's 0.1), any single extreme batch propagates immediately. The `dr3.clip_max=2.0` doesn't clamp `w_off_max` in practice (it stays at ~0.7–0.9) because the closed-form `w_clip_upper=1.13` is the actual binding constraint — but the sharpening still shows up as micro-level asymmetries between teacher and on-policy micro-batches.

So the answer to "which combination killed v25" is: **sharp w_hat (disc_acc→1) × mildly-elevated teacher effective advantage (clip=2.0 letting more r·A through) × late-training entropy drop (0.57→0.41 in two steps) → one unlucky gradient → format collapse → gradient explosion.** No single signal exceeded a hard threshold; it was a confluence.

## 6. Teacher advantage magnitude

`teacher_diag/adv/teacher/max` has been pinned at 5.0 since step 1 in all three runs (this is a clip from the GRPO advantage computation upstream). So the "5.0" number itself is not what killed v25 — v12 and v24 also show 5.0 throughout and were fine. What differs is the **token-weighted** teacher contribution after DR3 reweighting, which is captured in `duet/adv_teacher_effective_mean` (not the raw max). v25 runs 30–70% hotter on that metric in the pre-crash window (see §3).

## 7. Predictions for v26–v29

Based on the v25 failure mechanism (disc_acc saturation + widened clip + entropy drop compound → format break):

| run | change vs v25            | prediction                                                                                                     |
|-----|--------------------------|----------------------------------------------------------------------------------------------------------------|
| v26 | off_cliprange_high=**5.0** | **Will collapse — probably earlier** (~step 70–85). Strictly more permissive than v25; teacher effective advantage magnitude will grow faster. No countervailing stabilizer. **Recommend pre-emptive kill** if grad_norm > 10 before step 60 or entropy_loss < 0.45 before step 80. |
| v27 | on-policy clip_high=**1.0** (also widened) | **Will collapse, likely different mode**. Widening on-policy clip releases on-policy gradient variance too. Could crash earlier (onpolicy has most tokens) or paradoxically survive because onpolicy gradient drowns out teacher. ~40% survival odds. |
| v28 | off_cliprange_high=**0.6** (baseline) + w_hat_ema=0.1 (smoother) | **Will survive** — this is essentially v12 with better DR3 smoothing. Expected final score ≈ 0.43–0.55 (matches v12 baseline with marginal uplift from smoother `w_hat`). This is the correct "variance-only path" control. Do NOT kill. |
| v29 | off_cliprange_high=2.0 + dr3.clip_max=5.0 + w_hat_ema=0.1 | **Coin-flip (~50/50)**. Smoother `w_hat_ema=0.1` buffers the discriminator saturation that killed v25, but `dr3.clip_max=5.0` lets individual teacher tokens get 5x weights. If it survives past step 90 it may produce the paper's best score; if it crashes it'll look like v25 with bigger grad spike. |

## 8. Concrete recommendation

1. **Kill v26 now** unless `grad_norm` stays < 5.0 and `entropy_loss` > 0.50 through step 80. v26 is strictly more unstable than v25 and the hypothesis it's testing (cliprange=5.0 works) is already falsified by v25's cliphit_rate=0 observation: if the clip wasn't binding at 0.6 or 2.0, pushing it to 5.0 cannot help — it can only hurt.

2. **Let v28 run to completion.** v28 is the cleanest control for "does smoother w_hat_ema alone rescue v12?". Its result is the key ablation data point for the paper: it tells us whether the CHORD-BC term in v24 is necessary, or just substituting for the variance-reduction role that `w_hat_ema=0.1` can provide directly.

3. **Let v29 run but watch** `duet/adv_teacher_effective_mean` (kill if > 0.45 for three consecutive steps) and `dr3/disc_acc` (kill if > 0.999 for two consecutive steps before step 80). If v29 survives, it is the correct "variance-without-BC" story for the paper.

4. **Predicted paper narrative if v28 survives at ~0.50 and v29 survives at ~0.70**: the BC term in v24 is doing *variance control on teacher gradients*, not *trust-region escape*. The right framing is "smooth w_hat + mild clip widening" replaces CHORD-BC with a principled DR3-native mechanism. If v28 lands at ≥0.60, BC is entirely unnecessary. If v28 lands at ~0.43 (v12 level) and v29 at ~0.70, it's specifically the *combination* of smoother EMA and wider clip that matters, which is a clean, publishable ablation story.

5. **Do not run more `off_cliprange_high`-only variants.** The metric says the clip is not the knob. Future ablations should target `w_hat_ema_alpha` (0.05 / 0.1 / 0.3), `dr3.disc_temperature` (to prevent disc_acc→1 saturation), and `actor/entropy_coef` (to prevent the late entropy drop that precipitated step-98 collapse).
