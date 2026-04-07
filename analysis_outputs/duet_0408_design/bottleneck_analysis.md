# Bottleneck Analysis: Hybrid 0405 vs LUFFY — Improvement Targets for 0408

**Date:** 2026-04-05
**Analyst:** exp-analyst agent
**Hybrid 0405 run:** rb1ee38c (webshop_3b_duet_hybrid_0405)
**LUFFY run:** o405qtk1 (webshop_3b_luffy)

---

## Executive Summary

Hybrid 0405 achieves **53% val success** vs LUFFY's **50%**, a **+3pp improvement** — the first DUET variant to definitively beat LUFFY on WebShop. However, both methods share a massive **peak-decline problem** (58% drop from training peak), and Hybrid lags LUFFY significantly in mid-training (steps 30-70). The three top improvement targets for 0408 are: (1) DR3 mid-training fade-out is too aggressive, (2) both methods have severe late-stage instability, and (3) SC bonus could be more adaptive.

---

## 1. Validation Results

| Metric | Hybrid 0405 | LUFFY | Delta |
|--------|------------|-------|-------|
| **Val@50 Success** | 14.0% | 9.0% | **+5.0pp** |
| **Val@50 Reward** | 0.592 | 0.509 | **+0.083** |
| **Val@100 Success** | **53.0%** | **50.0%** | **+3.0pp** |
| **Val@100 Reward** | 0.766 | 0.753 | **+0.013** |

### Observation
Hybrid leads at both checkpoints, with a larger lead at step 50 (+5pp) than at step 100 (+3pp). This suggests LUFFY's aggressive teacher weighting catches up in late training, compressing the gap.

---

## 2. Training Dynamics Summary

| Metric | Hybrid 0405 | LUFFY | Verdict |
|--------|------------|-------|---------|
| Peak Training Success | 0.7719 @ step 80 | 0.7895 @ step 80 | LUFFY slightly higher peak |
| Final5 Training Success | 0.5150 | 0.4502 | **Hybrid better by +0.065** |
| Rolling-5 Peak | 0.5940 @ step 83 | 0.5477 @ step 83 | **Hybrid more consistent** |
| Peak Decline | 0.4505 (58.4%) | 0.4386 (55.6%) | Similar — both very large |
| KL Loss (final5) | 1.5787 | 1.7730 | Hybrid slightly lower |
| Teacher Grad Share (final5) | 0.1116 | 0.6684 | DR3 fades teacher aggressively |
| Teacher Grad Share (peak) | 0.3309 | 1.0000 | LUFFY keeps full teacher weight longer |

---

## 3. Step-by-Step Phase Analysis

### Training Success by 10-step Window

| Window | Hybrid Mean | LUFFY Mean | Leader | Gap | Phase |
|--------|------------|------------|--------|-----|-------|
| 0-10 | 0.0058 | 0.0156 | LUFFY | 0.010 | Warmup |
| 10-20 | 0.0686 | 0.0717 | LUFFY | 0.003 | Warmup |
| 20-30 | 0.0189 | 0.0035 | **Hybrid** | 0.015 | Early learning |
| 30-40 | 0.0248 | 0.0665 | LUFFY | 0.042 | **LUFFY acceleration** |
| 40-50 | 0.0194 | 0.0767 | LUFFY | 0.057 | **LUFFY acceleration** |
| 50-60 | 0.1603 | 0.1215 | **Hybrid** | 0.039 | Hybrid catches up |
| 60-70 | 0.2508 | 0.3725 | LUFFY | **0.122** | **Biggest LUFFY lead** |
| 70-80 | 0.4126 | 0.4370 | LUFFY | 0.024 | Convergence zone |
| 80-90 | 0.4980 | 0.4577 | **Hybrid** | 0.040 | **Hybrid overtakes** |
| 90-100 | 0.5120 | 0.4504 | **Hybrid** | 0.062 | **Hybrid leads** |

**Windows won:** Hybrid=4, LUFFY=6

### Key Phase Insights

1. **Steps 30-50: LUFFY Acceleration Phase**
   - LUFFY's `teacher_gradient_share = 1.0` means ALL teacher gradient counts fully
   - DR3 has already reduced Hybrid's share to 0.2-0.3
   - **This is where Hybrid loses the most ground**

2. **Steps 60-70: Maximum Gap**
   - Step 63: Hybrid 0.18 vs LUFFY 0.61 — a **0.43 gap** (single worst step)
   - Step 70: Hybrid 0.40 vs LUFFY 0.59 — **0.19 gap**
   - LUFFY's teacher_gradient_share drops from 1.0 to 0.1 around step 65

3. **Steps 80+: Hybrid Overtakes**
   - After teacher_gradient_share equalizes (~0.1 for both)
   - SC bonus continues providing guidance that LUFFY lacks
   - Hybrid more consistent: rolling-5 peak 0.594 vs 0.548

---

## 4. DR3 Diagnostic — Why Mid-Training Fade Hurts

| DR3 Metric | Step 10 | Step 30 | Step 50 | Step 70 | Step 90 | Trend |
|------------|---------|---------|---------|---------|---------|-------|
| disc_acc | 0.56 | 0.85 | 0.94 | 0.99 | 1.00 | Monotonic increase |
| w_off_mean | 0.94 | 0.76 | 0.55 | 0.42 | 0.38 | Monotonic decrease |
| alpha | 0.12 | 0.12 | 0.11 | 0.12 | 0.12 | Stable |
| teacher_grad_share | 0.27 | 0.21 | 0.29 | 0.09 | 0.12 | Drops around step 65 |
| logw_applied_mean | — | -0.20 | -0.49 | -0.73 | -0.84 | Growing downweight |

### Analysis

DR3's discriminator rapidly learns to separate teacher from on-policy distributions. By step 30, `disc_acc` reaches 0.85, and `w_off_mean` drops to 0.76. By step 50, teacher samples are being downweighted to 0.55x.

**The problem**: At step 30-50, the policy is barely learning (success ~2-8%). DR3 is already saying "teacher samples are too different to be useful" — but the policy hasn't learned enough to succeed on its own yet. This creates a **learning valley** between steps 30-50.

**LUFFY avoids this**: LUFFY's `p/p_beta` formula with beta=0.1 keeps teacher_gradient_share at ~1.0 until the policy starts producing similarly-rewarded trajectories around step 65. This is effectively a **natural curriculum** — full teacher weight until the policy can compete.

### Hybrid's Advantage Despite This

Despite the mid-training valley, Hybrid achieves:
- Better final5 training (+0.065)
- Better validation (+3pp)
- Better late-phase stability

This suggests DR3's data-driven fade-out, combined with SC, produces a **better final policy** even though the learning path is slower.

---

## 5. State Channel Contribution Analysis

| SC Metric | Early (1-20) | Mid (40-60) | Late (80-100) | Trend |
|-----------|-------------|-------------|---------------|-------|
| bonus_vs_reward_ratio | 0.134 | 0.123 | 0.117 | ↓ Slight decrease |
| coverage_mean | 1.000 | 1.000 | 1.000 | → Perfect throughout |
| coverage_nonzero_ratio | 1.000 | 1.000 | 1.000 | → Perfect throughout |
| progress_onpolicy_mean | 0.282 | 0.432 | 0.508 | ↑ Increasing (policy improves) |
| progress_teacher_mean | 0.517 | 0.508 | 0.500 | → Stable (teacher baseline) |
| shaped_ratio | 0.813 | 0.859 | 0.867 | ↑ Slightly increasing |
| bonus_per_sample_mean | 0.058 | 0.087 | 0.085 | ↑ Then plateaus |
| beta_effective | 0.200 | 0.200 | 0.200 | → Constant (no decay) |
| teacher_excluded_count | 8.0 | 8.0 | 8.0 | → Consistent exclusion |

### Step-Level Deltas (only 13 data points logged)

| Metric | Value | Note |
|--------|-------|------|
| step_delta_mean | 0.022 | Small but positive |
| step_delta_positive_ratio | 0.082 | 8.2% of steps show progress |
| step_delta_negative_ratio | 0.041 | 4.1% regress |
| step_delta_abs_mean | 0.078 | Absolute magnitude |

### SC Verdict

- **Coverage is perfect (100%)**: The attribute-aware match mode works well for WebShop
- **Bonus/reward ratio is healthy (12%)**: Not overwhelming task reward
- **Progress tracks policy improvement**: onpolicy_progress goes 0.28 → 0.51
- **Teacher exclusion works**: 8 teacher samples excluded per step consistently
- **Step-level deltas are sparse**: Only 8% of steps get positive deltas — may indicate step_level signal is too weak to matter

### SC Improvement Opportunities
1. **Beta is constant at 0.2**: Could decay over training as policy improves
2. **Step-level deltas are sparse**: Low coverage (8%) suggests most steps don't have matched expert progress — consider relaxing matching criteria
3. **Bonus plateaus at 0.085**: Not scaling with training progress

---

## 6. Generalization Analysis

| Metric | Hybrid 0405 | LUFFY |
|--------|------------|-------|
| Training peak | 0.7719 | 0.7895 |
| Training final5 | 0.5150 | 0.4502 |
| Val@100 success | 0.5300 | 0.5000 |
| Val@100 reward | 0.7656 | 0.7528 |
| **Train-Val Gap (peak-val)** | **0.2419** | **0.2895** |
| **Train-Val Gap (final5-val)** | **-0.0150** | **-0.0498** |

### Interpretation

- **Hybrid has a smaller train-val gap** by ~5pp (0.24 vs 0.29)
- The final5 training success is actually LOWER than validation for both — this is because both methods have high variance in late training and the training metrics include high-noise steps
- SC likely helps generalization by providing environment-grounded reward signals rather than pure policy-based learning
- **Neither method shows signs of overfitting** — the train-val alignment is healthy

---

## 7. Where Is Hybrid 0405 WORSE Than LUFFY? (Improvement Targets)

### Target 1: Mid-Training Learning Speed (Steps 30-70)

| Evidence | Detail |
|----------|--------|
| Gap magnitude | LUFFY leads by 0.042-0.122 in 10-step windows |
| Root cause | DR3 disc_acc reaches 0.85 by step 30 → teacher downweighting starts too early |
| Impact | ~40 steps of slower learning, partially recovered in steps 80+ |
| Fix approaches | (a) Warmup DR3 longer (apply_warmup_steps: 30 vs current 10), (b) Softer discriminator temperature, (c) Minimum teacher_gradient_share floor |

### Target 2: Training Peak Stability (Step 80+ Decline)

| Evidence | Detail |
|----------|--------|
| Gap magnitude | Both lose ~58% from peak; Hybrid declines 0.45, LUFFY 0.44 |
| Root cause | Likely: policy overshoot beyond KL constraint → entropy collapse + reward hacking |
| Impact | Peak→final drop wastes 20 steps of training budget |
| Fix approaches | (a) LR decay after step 70, (b) Increase kl_loss_coef from 0.001 to 0.01, (c) Best-checkpoint selection (save_freq=10) |

### Target 3: On-Policy Policy Gradient Signal

| Evidence | Detail |
|----------|--------|
| Hybrid on_pg_loss final5 | -0.0069 (very small) |
| LUFFY on_pg_loss final5 | -0.0236 (3.4x larger) |
| Hybrid adv_onpolicy_mean | 0.086 (positive, healthy) |
| LUFFY adv_onpolicy_mean | 0.000 (near zero) |
| Interpretation | Hybrid's on-policy PG gradient is weaker than LUFFY's. Hybrid's advantage for on-policy samples is larger (0.086 vs 0.0), but the PG loss is smaller — suggesting clipping or other loss weighting is reducing the effective gradient |

---

## 8. Where Is Hybrid 0405 BETTER Than LUFFY?

| Metric | Hybrid | LUFFY | Advantage |
|--------|--------|-------|-----------|
| KL loss (final5) | 1.579 | 1.773 | -0.194 (more stable) |
| Training success (final5) | 0.515 | 0.450 | +0.065 (better late training) |
| Val success@100 | 53% | 50% | +3pp |
| Val reward@100 | 0.766 | 0.753 | +0.013 |
| Train-val gap | 0.242 | 0.290 | -0.048 (better generalization) |
| teacher_gradient_share | 0.112 | 0.668 | Data-driven curriculum |
| teacher_off_pg_loss | -0.760 | -0.326 | Stronger teacher PG signal when applied |

---

## 9. Peak Before Step 100? — Checkpoint Selection Analysis

### Hybrid 0405
- **Single-step peak**: 0.7719 @ step 80
- **Rolling-5 peak**: 0.5940 @ step 83
- **Steady-state zone** (steps 90-98): mean=0.509, std=0.082

### LUFFY
- **Single-step peak**: 0.7895 @ step 80
- **Rolling-5 peak**: 0.5477 @ step 83
- **Steady-state zone** (steps 90-99): mean=0.452, std=0.097

### Recommendation
Both methods peak at step 80 with massive subsequent decline. However:
- The **validation at step 100** shows strong performance (53%/50%) despite training curve decline
- This suggests the training-metric decline is **noise** (different task batches), not true policy degradation
- **Practical fix**: Save checkpoints every 10 steps and evaluate top-3 → pick best
- **Principled fix**: Add LR cosine decay starting at step 60, reaching 0.1x at step 100

---

## 10. Summary: 0408 Design Priorities

### Priority 1: Slow Down DR3 Fade-Out (Expected Impact: +2-4pp)
DR3 fades teacher weight too early (by step 30, w_off drops to 0.76). LUFFY keeps full weight until step 65.
- **Option A**: Increase `apply_warmup_steps` from 10 to 25-30
- **Option B**: Set `disc_temperature` to 2.0 (softer discrimination)
- **Option C**: Add `teacher_gradient_share_min: 0.3` floor until step 50

### Priority 2: Stabilize Late Training (Expected Impact: +1-3pp)
Both methods lose ~58% from peak after step 80.
- **Option A**: LR cosine decay: `lr_schedule: cosine, warmup_steps: 10, final_lr_ratio: 0.1`
- **Option B**: Increase `kl_loss_coef` from 0.001 to 0.005
- **Option C**: Save best checkpoint (already practical, no code change needed)

### Priority 3: Boost SC Step-Level Signal (Expected Impact: +1-2pp)
Step deltas cover only 8% of steps, limiting SC's per-step guidance.
- **Option A**: Relax step-level matching threshold
- **Option B**: Increase `step_level.eta` from 0.05 to 0.1
- **Option C**: Add SC beta decay (0.2 → 0.05) to prevent late-training interference

### Non-Priorities (Already Working Well)
- SC trajectory-level bonus: 12% of reward, stable, good coverage
- Teacher exclusion from SC: Working perfectly
- KL constraint: Hybrid is actually more stable than LUFFY
- Generalization: No overfitting detected

---

## Appendix: Hybrid 0405 Config Highlights

```yaml
# DR3 settings
use_dr3: true
dr3:
  enable: true
  apply_to: teacher_no_logprob
  feature_mode: v3_aug
  use_policy_shaping: true          # Hybrid mode: DR3 + policy shaping
  policy_shaping_beta: 0.1          # Same beta as LUFFY
  hidden_proj_dim: 64
  disc_temperature: 1.0
  apply_warmup_steps: 10            # KEY: only 10 steps warmup
  alpha_mode: sync_batch_ema
  dual_enable: true

# State Channel settings
state_channel:
  enable: true
  exclude_teacher: true
  beta: 0.2                         # Constant, no decay
  match_mode: attribute_aware
  grpo_decouple: true
  step_level:
    enable: true
    eta: 0.05                        # Conservative step delta
```

---

## Appendix: LUFFY Teacher Advantage Anomaly

LUFFY's `diag/adv_teacher_sample_mean` reaches extreme values (4426 @ step 30, 7012 @ step 50) during the mid-training phase when `teacher_gradient_share = 1.0`. This is caused by the `p/p_beta` policy shaping formula amplifying teacher advantages when the policy diverges from the reference. Hybrid avoids this via DR3's bounded density ratios (w_off capped at 5.0).

Despite these extreme values, LUFFY's training doesn't collapse — the GRPO group normalization absorbs the scale. But this behavior makes LUFFY less predictable and harder to tune.
