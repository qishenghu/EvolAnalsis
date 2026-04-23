# v24 on ALFWorld 1.5B: Why DUET+BC Regressed -2pp

**Date**: 2026-04-19
**Run**: `alfworld_qwen1.5b_duet_v24` vs baseline `alfworld_qwen1.5b_duet`
**Headline numbers** (200 validation tasks, single seed):

| Checkpoint | DUET-v1 (no BC) | DUET-v24 (+BC) | delta |
|------------|-----------------|----------------|-------|
| Val@50     | 27.5%           | **33.5%**      | **+6.0pp** |
| Val@100    | **32.5%**       | 30.5%          | -2.0pp |

The regression is real at step 100 but inverts at step 50 — v24 is not uniformly worse. This is a **peak-then-regress** shape, matching the v36/WebShop pattern that led to v24 in the first place. The paper narrative implication is nuanced: BC doesn't "fail" on ALFWorld, it delivers gains on a different schedule than on WebShop, and the hand-tuned v24 schedule (25-step mu decay) stops helping too early for ALFWorld's slower learning curve.

---

## Headline finding (30 words)

v24 trades late-phase on-policy gradient flow for early BC-driven lift: it wins decisively at val@50 (+6.0pp) but BC-induced advantage amplification and response-length drift erode the lead by val@100 (-2.0pp). The BC intervention is correctly specified; the decay schedule is mis-calibrated for ALFWorld's slower dynamics.

---

## Q1. Was there early gain then late regression?

**Yes, and the validation data makes it unambiguous.**

- Val@50: v24 = 33.5%, v1 = 27.5% → v24 is **ahead by +6.0pp** at mid-training. This is a real mid-training gain, not noise.
- Val@100: v24 = 30.5%, v1 = 32.5% → v24 has **fallen back by -2.0pp** at end-of-training. v24 regressed -3.0pp from its own val@50 peak.
- Training-side rolling success (10-step window) shows the same shape: v24 leads or matches v1 between steps 45-65 (rolling 0.21-0.32), then v1 opens a gap between steps 70-95 (rolling 0.31-0.33 for v1 vs 0.27-0.31 for v24).

Late-phase regression signals:
- `actor/grad_norm`: v24 rises to 4.38 late (61-100) vs v1 at 9.56. v1 has **larger** gradient flow late. Despite the larger grad_norm in v1, v1 is still improving, suggesting v24 is under-utilizing late-phase signal.
- `response_length/mean`: v24 = 7309 late vs v1 = 5681 late. v24's trajectories are **28% longer** late in training — the policy explores more (or loops more), and this correlates with the regression.
- `duet/adv_onpolicy_effective_abs_mean`: v24 = 0.162 late vs v1 = 0.089 late. v24's effective advantage magnitude is **~1.8x larger** late, but it is not being converted into success.

Interpretation: v24's advantages are amplified (BC residual effect even at mu=0.05, plus longer trajectories giving more reward-sparse tokens), but the amplification is directed toward lower-quality trajectories.

Figure: `fig_v24_alfworld_val_curve.png` shows the peak-regression shape for both training-side rolling success and validation@50/@100.

## Q2. Did BC over-imitate on ALFWorld?

**Partial yes — the BC teacher fit is fast and strong, but mu hits the floor before this becomes catastrophic.**

- `chord/mu` trajectory: 0.299 (step 1) → 0.198 (step 11) → 0.065 (step 21) → **0.05 (step 30+ forever)**. The schedule is "fire early, then idle at floor". Config sets `chord_mu_decay_steps: 25, chord_mu_valley: 0.05`, so after step 25 mu=0.05.
- `chord/sft_loss` drops from 1.109 (step 1) to ~0.19 (step 31) and stays in the 0.16-0.44 band afterward. Fit is tight but not collapsed (never approaches 0). This is what we'd expect for templated ALFWorld actions — teacher distribution is narrow and easy to match.
- `chord/phi_mean = 1.0` throughout. `chord_use_token_weighting: false` in config means phi is trivially identity; no token re-weighting is applied. So the BC signal is uniform per-token teacher cross-entropy, scaled by mu.
- Entropy is essentially unchanged between v1 and v24: v1 = 0.075/0.103/0.092 (early/mid/late), v24 = 0.061/0.099/0.083. Differences are within 0.02 and don't indicate mode collapse.

No smoking-gun over-imitation. What BC *does* appear to be doing is shrinking KL: `actor/kl_loss` on v24 tops out at ~0.6 vs v1 at ~0.8, and mean over late phase is 0.25 (v24) vs 0.57 (v1). This is the expected regularizer effect. The regression is not from over-imitation of action tokens — it's from the dynamics downstream.

Figure: `fig_v24_alfworld_bc_diag.png` shows mu and sft_loss trajectories.

## Q3. Did State Channel interact badly with BC?

**No direct harmful interaction, but SC activity is essentially over before BC influence ends.**

- `state_channel/beta_effective` is non-zero only for the first ~20 steps on *both* v1 and v24. Config uses a decay schedule that fades SC bonus early. Late-phase success (61-100) occurs entirely without SC bonus contribution.
- `state_channel/progress_onpolicy_mean`: v24 = 0.34 late vs v1 = 0.32 late. v24 is slightly *ahead* on SC progress late, consistent with BC nudging the policy toward teacher-adjacent states. This is not "SC broken" — if anything, v24 reaches higher-progress states more often than v1.
- `state_channel/bonus_vs_reward_ratio`: identical zero behavior across both variants after step 20. Not a meaningful discriminator.

The SC mechanism is orthogonal to BC on this config because the SC bonus schedule expires before BC's residual effect matters. So the failure mode is not SC+BC redundancy.

Figure: `fig_v24_alfworld_sc_interaction.png` panels 1-2 confirm SC progress and bonus ratio similarity.

## Q4. Did `teacher_gradient_share` curriculum fire on ALFWorld?

**Yes — it decays cleanly on both variants, faster on v24.**

- v1 TGS: 0.398 (early 1-30) → 0.330 (mid 31-60) → 0.273 (late 61-100). Standard monotone fade.
- v24 TGS: 0.440 (early) → 0.301 (mid) → 0.224 (late). Steeper fade — teacher contribution drops faster.
- End-of-training TGS: v24 reaches 0.12-0.21 range in final 10 steps, v1 stays at 0.26-0.41.

**Interpretation**: BC effectively accelerates the curriculum. The DR3 discriminator reaches `disc_acc = 1.0` on v24 by step 26 (vs step 30+ on v1), because BC pulls the student toward teacher behavior early, making the density ratio collapse the teacher contribution faster. This may be *too* aggressive: v24 loses teacher-gradient-assisted exploration late, while v1 still has teacher share in the 0.26-0.41 range. In a regime where on-policy success is the bottleneck (ALFWorld late-phase), premature teacher fade-out hurts.

This is the most mechanistically interesting finding of the run. It's the same mechanism v24 was designed to exploit for WebShop (rapid teacher-to-student handoff), but on ALFWorld the student isn't ready to run solo as early.

Figure: `fig_v24_alfworld_bc_diag.png` panel 3.

## Q5. Does `adv_onpolicy_effective_abs_mean` look different?

**Yes — markedly.**

- v1: 0.039 (early) → 0.061 (mid) → 0.089 (late). Steady, moderate growth consistent with the policy getting better and rewards becoming more discriminative across the group.
- v24: 0.036 (early) → 0.073 (mid) → **0.162 (late)**. Near-doubling from mid to late, and ~2x v1's late magnitude.

Second-pass theory posits BC acts as an advantage regularizer. Here we see the *opposite*: v24's effective advantages are larger than v1's. Two mechanisms can produce this:
1. Longer trajectories (v24's response_length 7309 vs v1's 5681) spread token_level_rewards over more tokens with higher variance.
2. Teacher baseline separation with a faster-shrinking teacher share means the on-policy group variance shrinks *differently* in the two regimes, and the z-score normalization can amplify residuals.

Either way, the "advantage regularizer" prediction from the second-pass memo does not hold on ALFWorld. BC here **increases** effective advantage magnitude late, and that additional gradient signal appears to be misdirected (longer rollouts with lower success).

Figure: `fig_v24_alfworld_sc_interaction.png` panel 3 (adv) and panel 4 (length) together.

---

## Data noise caveat

This analysis rests on a **single seed** for both v1 and v24. A 200-task validation set gives per-checkpoint SE around 3.3pp (sqrt(0.3*0.7/200)). So:

- Val@50 gap of +6.0pp is ~1.8 SE — likely real but not overwhelming.
- Val@100 gap of -2.0pp is ~0.6 SE — **within noise**. It is plausible that the "regression" is partially a noisy validation draw at step 100.

The training-side rolling curves and auxiliary metrics (grad_norm, response_length, adv_abs_mean) are *consistent* with v24 being weaker late, so we don't think it's pure noise, but a second seed would lock this down. **I would not present "v24 worse than v1 on ALFWorld" as a headline claim in the paper from this evidence alone.** What we can claim confidently: (i) BC gives a real mid-training lift, (ii) BC shifts the training dynamics (grad_norm, length, TGS curriculum) significantly, (iii) the fixed 25-step mu schedule is not tuned for ALFWorld's slower dynamics.

## Paper narrative implication

This strongly supports the "adaptive BC" story, not the "BC always helps/hurts" story:

1. **Mid-training BC lift is real**: +6.0pp at val@50 on ALFWorld is not marginal. The low-rare-token-gap, low-format-fragility prediction of the second-pass memo said "near zero effect" — this is wrong at mid-training. BC helps more broadly than the gap theory predicts, likely via the KL-regularization / curriculum-acceleration channel rather than the format-correction channel.
2. **Schedule transfer is the problem**: v24's WebShop-tuned 25-step mu decay fires, then releases the policy before on-policy gradient flow can take over on ALFWorld. WebShop saturates training faster, so 25 steps is enough for the handoff; on ALFWorld it's not.
3. **The DR3 curriculum accelerates under BC**: v24's teacher_gradient_share decays faster (late: 0.22 vs 0.27). This was intentional for WebShop but is premature on ALFWorld, matching the "teacher fade-out too fast" failure mode we've seen on 1.5B-WebShop-v12.
4. **Advantage-regularizer story needs a caveat**: second-pass theory said BC calibrates advantages to be smaller and cleaner. On ALFWorld we see the opposite direction late — |adv| grows faster under BC. The theory likely needs a term for "BC shifts policy into longer-trajectory regimes where advantage normalization behaves differently".

For the paper: present v24 on ALFWorld as evidence that **BC schedules need to be per-environment or adaptive**, not that BC hurts ALFWorld. The val@50 win is the positive signal; the val@100 drop is the schedule-mismatch cost.

## Recommendation if we run again

1. **Lengthen mu decay**: `chord_mu_decay_steps: 60` (from 25) for ALFWorld. Keep peak 0.3, valley 0.05. Goal: maintain BC throughout the mid-training bump that v24 shows, avoid premature release.
2. **Or: adaptive mu tied to TGS**: slow mu decay while TGS is still >0.3; accelerate only when TGS drops below 0.2. This would self-calibrate across environments.
3. **Add a second seed**: minimally needed to turn "val@100 -2pp" from "within noise" to a defensible claim (or confirm the opposite — that v24 matches v1 late with better seed variance).
4. **Log more frequently near val@50**: save extra checkpoints at 40/45/55/60 so we can see the peak location precisely and verify the peak-then-regress pattern.
5. **Consider `chord_use_token_weighting: true`**: v24 ran with phi=1.0 (no weighting). If rare-token calibration is a significant channel, enabling token weighting and comparing could distinguish "BC helps via weight on rare tokens" vs "BC helps via overall KL regularization".

---

## Figure index

- `figures/fig_v24_alfworld_val_curve.png` — Training rolling success + val@50/100 bars showing peak-regression.
- `figures/fig_v24_alfworld_train_metrics.png` — 4-panel: reward, grad_norm, entropy, kl.
- `figures/fig_v24_alfworld_bc_diag.png` — BC-specific: mu schedule, sft_loss, teacher_gradient_share.
- `figures/fig_v24_alfworld_sc_interaction.png` — SC progress, SC bonus ratio, adv_onpolicy |adv| magnitude, response length (late-phase drift).

## Data sources used

- Parsed training metrics: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/alfworld_metrics.json`
- Raw training log v24: `/data/home/qisheng/EvolAnalsis/logs/alfworld_qwen1.5b_duet_v24.log`
- Raw training log v1: `/data/home/qisheng/EvolAnalsis/logs/alfworld_qwen1.5b_duet.log`
- v24 validation jsonl: `/data/home/qisheng/EvolAnalsis/experiments/alfworld/alfworld_qwen1.5b_duet_v24/validation_log/{50,100}.jsonl`
- v1 validation jsonl: `/data/home/qisheng/EvolAnalsis/experiments/alfworld/alfworld_qwen1.5b_duet/validation_log/{50,100}.jsonl`
- v24 config: `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v24.yaml`
- v1 config: `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet.yaml`
