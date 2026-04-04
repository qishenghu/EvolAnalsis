# WebShop Deep Experiment Analysis: LUFFY+SC and Hybrid
**Date**: 2026-04-03
**Analyst**: Experiment Analyst (DUET Project)

---

## 1. Cross-Experiment Comparison Table

### Validation Results (200 tasks per evaluation)

| Method               | val@50 mean | val@50 perf | val@100 mean | val@100 perf | delta     | Trend     |
|----------------------|-------------|-------------|--------------|--------------|-----------|-----------|
| **LUFFY**            | 0.5086      | 17/200      | **0.7528**   | **99/200**   | **+0.244**| IMPROVING |
| DUET_orig (DR3 only) | 0.5992      | 45/200      | 0.7251       | 65/200       | +0.126    | Improving |
| DUET_0402            | 0.4835      | 13/200      | 0.7353       | 71/200       | +0.252    | Improving |
| DUET_0403            | 0.6456      | 61/200      | 0.6790       | 66/200       | +0.033    | Flat      |
| DUET_0404            | 0.4974      | 4/200       | 0.6463       | 47/200       | +0.149    | Improving |
| DUET_0401            | 0.5170      | 24/200      | 0.5649       | 36/200       | +0.048    | Flat      |
| **Hybrid (NEW)**     | **0.6402**  | **34/200**  | 0.5121       | 31/200       | -0.128    | DEGRADING |
| OnPolicy             | 0.2759      | 2/200       | 0.4019       | 4/200        | +0.126    | Improving |
| GRPO 3B (800 tasks)  | 0.3725      | 5/200       | 0.3592       | 6/200        | -0.013    | Flat      |
| **LUFFY+SC (NEW)**   | 0.5175      | 11/200      | 0.2211       | 1/200        | **-0.296**| COLLAPSED |
| CHORD                | 0.2667      | 1/200       | -0.1000      | 0/200        | -0.367    | COLLAPSED |
| DUET_0402_v2         | -0.1000     | 0/200       | -0.1000      | 0/200        | 0.000     | DEAD      |

### Key Observations
- **LUFFY is the clear winner at val@100**: 0.7528 avg reward, 99/200 perfect scores
- **DUET_orig (DR3-only, no effective SC) is second**: 0.7251, 65/200 perfect -- note: SC was technically enabled but `grpo_decouple` was NOT set, so bonus was never injected into advantages
- **Hybrid had the strongest val@50** (0.6402) but degraded to 0.5121 by val@100
- **LUFFY+SC catastrophically collapsed**: from 0.5175 at val@50 to 0.2211 at val@100

---

## 2. LUFFY+SC Collapse Analysis

### 2.1 The Collapse Timeline

The collapse is precise and dramatic, occurring between steps 84-90:

| Step | CJK % | FmtErr % | Rollout Mean | KL Loss |
|------|--------|----------|--------------|---------|
| 80   | 0%     | 5%       | 0.7604       | 0.743   |
| 81   | 0%     | 5%       | 0.7074       | 0.923   |
| 82   | 2%     | 3%       | 0.5929       | 1.003   |
| 83   | 0%     | 9%       | 0.6294       | 0.692   |
| 84   | 2%     | 28%      | 0.6762       | 0.732   |
| 85   | 6%     | 31%      | 0.6516       | 0.853   |
| 86   | 17%    | 39%      | 0.5948       | 0.575   |
| 87   | 16%    | 56%      | 0.5936       | 1.902   |
| 88   | 25%    | 55%      | 0.4623       | 2.225   |
| 89   | 41%    | 70%      | 0.4929       | 1.405   |
| 90   | 52%    | 75%      | 0.4704       | 2.543   |
| 95   | 56%    | 77%      | 0.3939       | 0.893   |
| 100  | 72%    | 80%      | 0.4239       | 2.639   |

**At val@100**: 89.5% of trajectories contain CJK (Chinese) characters, 91.5% have format errors.

### 2.2 Root Cause: Language Collapse via KL Explosion

The model undergoes **language collapse** -- it starts generating Chinese text instead of English. This is a well-known failure mode of Qwen2.5 models under RL training, where the policy drifts so far from the reference that it falls into a different language mode.

**KL comparison** (LUFFY vs LUFFY+SC):
- LUFFY: KL oscillates around 1.0-1.8 throughout training (never collapses)
- LUFFY+SC: KL stays low (0.3-1.0) until step 87, then spikes to 2.0-2.6

**Why does SC cause this?** The SC bonus alters the advantage landscape:
- At step 1: SC bonus mean = 0.038, adv_on mean = 0.016. **SC bonus is 2.4x the advantage magnitude**
- This means SC bonus dominates the gradient signal early on, pushing the policy toward states that match teacher observations
- Over time, as the model improves, the bonus_vs_reward_ratio drops (0.11 -> 0.06), but the damage accumulates
- The progress_onpolicy metric drops from 0.376 (step 50) to 0.148 (step 100), showing the model is moving AWAY from expert states despite the SC reward
- This creates a negative feedback loop: model diverges from expert -> SC signal becomes noisy -> policy instability -> KL explosion -> language collapse

### 2.3 Why LUFFY+SC is Worse than Plain LUFFY

| Metric                          | LUFFY (step 80) | LUFFY+SC (step 80) |
|---------------------------------|-----------------|---------------------|
| Rollout mean reward             | 0.876           | 0.760               |
| val@100 mean reward             | 0.753           | 0.221               |
| KL divergence                   | 1.431           | 0.743               |
| Entropy                         | 0.422           | 0.465               |
| CJK in outputs                  | 0%              | 0%                  |
| Format errors                   | 0%              | 5%                  |
| On-policy reward                | 0.861           | 0.731               |

At step 80, LUFFY+SC looks mostly fine -- slightly worse than LUFFY but still functional. The catastrophe happens in steps 85-90 when accumulated policy drift crosses a tipping point.

**Critical finding**: LUFFY has HIGHER KL than LUFFY+SC throughout most of training (LUFFY reaches 2.0+ at step 50 already), yet LUFFY never collapses. This suggests that SC creates a different, more dangerous KIND of policy drift -- not just magnitude, but direction. SC pushes the policy toward intermediate states that may not be compatible with the language structure that LUFFY's p/(p+beta) shaping preserves.

---

## 3. Hybrid Degradation Analysis

### 3.1 Failure Mode: Repetition Loops

Unlike LUFFY+SC's language collapse, Hybrid exhibits **repetition loops** -- the model gets stuck generating `</think></think></think>...` endlessly.

| Metric at val@100     | LUFFY | Hybrid | LUFFY+SC |
|-----------------------|-------|--------|----------|
| Negative rewards      | 5     | 38     | 109      |
| Repetition loops      | 0     | 64     | 14       |
| CJK contamination     | 0     | 9      | 179      |
| Format errors         | 0     | 6      | 183      |
| Avg resp len (neg)    | 3603  | 9735   | 3779     |
| Avg resp len (pos)    | 4968  | 5976   | 5487     |
| Avg turns (neg)       | 4.8   | 13.9   | 7.4      |
| Avg turns (pos)       | 5.1   | 7.6    | 8.4      |

**Key observation**: Failed Hybrid trajectories are extremely long (9735 chars avg) with high turn counts (13.9), indicating the model gets trapped in loops rather than terminating cleanly.

### 3.2 Repetition Onset Timeline

| Step | Repetition % | Rollout Mean |
|------|-------------|--------------|
| 65   | 0.0%        | 0.4913       |
| 70   | 7.8%        | 0.6969       |
| 75   | 43.8%       | 0.5231       |
| 80   | 14.1%       | 0.7087       |
| 85   | 4.7%        | 0.6996       |
| 90   | 0.0%        | 0.6367       |
| 95   | 7.8%        | 0.5968       |
| 100  | 28.1%       | 0.5422       |

The repetitions are intermittent, spiking at steps 75 and 100. This suggests instability rather than irreversible collapse.

### 3.3 DR3 Analysis in Hybrid

| Metric                   | Value Range    | Assessment           |
|--------------------------|----------------|----------------------|
| disc_acc                 | 0.76-0.91      | Decent but not great |
| teacher_gradient_share   | 0.07-0.34      | Fluctuates widely    |
| w_mean                   | 0.99-1.09      | Near 1.0 throughout  |
| w_std                    | 0.00           | No variance!         |
| dual_lambda              | 0.00           | Never activated      |
| ess_off_window           | ~31            | Saturated            |

**Critical finding**: DR3 w_hat values are clustered extremely tightly around 1.0 (w_mean=1.0-1.09, w_std=0.0). This means **DR3 is not actually doing any meaningful density ratio correction**. The discriminator reaches decent accuracy (0.8-0.9) but the resulting weights are all ~1.0.

Compare with DUET_orig (which works better):
- DUET_orig disc_acc: 0.90-0.96 (higher discrimination)
- DUET_orig teacher_gradient_share: fluctuates between 0.07-1.0 (wider dynamic range)

### 3.4 Why Hybrid Underperforms LUFFY

The Hybrid uses DR3 w_hat as teacher_loss_scale (not LUFFY policy_shaping). Since w_hat ~= 1.0 throughout, teachers get full weight always -- there is no effective fade-out mechanism.

Meanwhile:
- LUFFY's p/(p+beta) naturally downweights teachers as the policy diverges from teacher actions (beta=0.1, aggressive decay)
- This allows LUFFY to "graduate" from teachers more cleanly
- Hybrid keeps teacher influence constant, leading to the model oscillating between teacher-like and self-discovered behaviors

**The SC component in Hybrid adds additional reward shaping noise** (bonus_ratio ~0.06-0.11) on top of the already-confused teacher weighting, further destabilizing training in the second half.

---

## 4. Why State Channel Fails on WebShop (Structural Diagnosis)

### 4.1 The SC-WebShop Mismatch

SC uses `stage` match mode, which matches on-policy observations against teacher trajectories at corresponding stages. This works on ALFWorld because:
- ALFWorld has clear sequential stages (search -> take -> go to -> put)
- Teacher and student visit similar rooms/objects even on different tasks
- Progress is well-defined and monotonic

On WebShop, the mismatch is fundamental:
- Progress values are LOW: on-policy progress averages only 0.15-0.38 (vs teacher's 0.40-0.46)
- Teacher trajectories explore different products, pages, and option combinations
- Two completely different correct solutions may share zero intermediate states
- The `unique_states_matched_total` drops from ~618 (step 1) to ~248 (step 100), showing matches become sparser as the model learns its own search patterns

### 4.2 Quantitative Evidence

**DUET_orig (working DUET)**: SC was technically enabled but `grpo_decouple` was NOT set, so:
- shaped_count = 0 (no samples received bonus)
- bonus = 0.000 throughout
- It effectively ran as DR3-only, and achieved 0.7251 at val@100

**Hybrid and LUFFY+SC** both set `grpo_decouple: true`:
- shaped_count = 35-56 per step (bonus actively injected)
- bonus_vs_reward_ratio = 0.06-0.14

This is the strongest evidence yet: **the ONLY difference between working DUET (0.725) and problematic DUET variants is whether SC bonus is actually injected into advantages**.

### 4.3 The SC Bonus is Not Small Enough to be Harmless

Early in training:
- SC bonus mean: 0.038
- On-policy advantage mean: 0.016
- **Ratio: SC bonus is 2.4x the GRPO advantage signal**

Even at its lowest (step 100):
- SC bonus mean: 0.027
- On-policy advantage mean: -0.086 (already negative due to drift)

The bonus never becomes negligible relative to the policy gradient signal.

---

## 5. Degeneration Patterns Across ALL WebShop Methods

| Method       | val@100 | CJK | Repetitions | Format Errors | Clean? |
|--------------|---------|-----|-------------|---------------|--------|
| LUFFY        | 0.753   | 0   | 0           | 0             | YES    |
| DUET_orig    | 0.725   | 0   | 0           | 0             | YES    |
| DUET_0402    | 0.735   | 0   | 0           | 0             | YES    |
| DUET_0404    | 0.646   | 2   | 1           | 0             | Mostly |
| Hybrid       | 0.512   | 9   | 64          | 6             | NO     |
| OnPolicy     | 0.402   | 10  | 6           | 14            | NO     |
| LUFFY+SC     | 0.221   | 179 | 14          | 183           | NO     |
| CHORD        | -0.100  | 0   | 200         | 200           | NO     |

**Pattern**: Methods that maintain a clean text generation mode (no CJK, no repetitions) all score above 0.64. Any method that develops degeneration drops sharply. This suggests that **policy stability is the primary determinant of WebShop performance**, more important than the specific teacher utilization method.

---

## 6. Key Insights for Next Iteration

### Insight 1: SC is net-negative on WebShop
Every method with active SC injection (grpo_decouple=true) degrades on WebShop:
- LUFFY+SC: catastrophic collapse
- Hybrid: moderate degradation
- DUET_orig (SC accidentally inactive): best DUET variant

**Recommendation**: Disable SC for WebShop entirely, or fundamentally redesign the match mode for shopping environments.

### Insight 2: DR3 w_hat ~= 1.0 in Hybrid means DR3 is not functional
The discriminator learns to distinguish (disc_acc=0.9) but the resulting density ratios cluster at 1.0. This likely means:
- Feature representation (`v3_aug`) maps teacher and student to similar embeddings for weight computation despite discriminator success
- The disc_temperature=1.5 may be smoothing the sigmoid output too much
- Or the alpha calibration is counteracting the discriminator signal

Compare: DUET_orig achieves teacher_gradient_share variations from 0.07 to 1.0 (full dynamic range), while Hybrid stays at 0.07-0.34 (compressed range).

### Insight 3: LUFFY's p/(p+beta) is the gold standard for WebShop
LUFFY achieves the highest score with the simplest mechanism. Its advantages:
- Natural fade-out as policy diverges from teacher (no discriminator needed)
- No additional reward shaping noise
- Maintains policy stability (no language collapse, no repetitions)
- Higher KL is tolerable as long as it doesn't cross the language-collapse threshold

### Insight 4: Policy stability determines the ceiling on WebShop
The correlation between "clean generation" and high performance is near-perfect. Any mechanism that introduces gradient noise (SC bonus, unstable DR3 weights) risks pushing the policy past the Qwen2.5 language-stability boundary.

### Insight 5: For the paper, environment-specific results are justified
DUET (DR3 + SC) beats LUFFY on ALFWorld (69.5% vs 61.5%) where SC's stage matching is appropriate. On WebShop, DUET should use DR3-only (no SC). This is a legitimate architectural insight: reward shaping from expert progress maps requires structural compatibility between the expert's solution path and the student's exploration space.

---

## 7. Recommended Next Steps (Analysis Only -- Not Implementation)

1. **Run DUET (DR3-only, no SC) vs LUFFY on WebShop** -- DUET_orig already shows 0.725 which is close to LUFFY's 0.753. A fair comparison with matched hyperparameters could narrow or close this gap.

2. **Investigate the DR3 w_hat=1.0 problem**: The Hybrid's DR3 is not producing useful density ratios. Check whether DUET_orig's DR3 (which does work) uses different feature_mode or temperature settings.

3. **Consider adaptive SC activation**: Rather than always-on SC, enable SC only when stage-match coverage exceeds a threshold (e.g., coverage > 0.8 AND unique_matched > 500).

4. **Run LUFFY with stronger KL regularization** (kl_coef > 0.001) to see if the performance advantage holds with more stable training.

---

## Appendix: Config Differences Summary

| Config Key                       | LUFFY          | LUFFY+SC       | Hybrid         |
|----------------------------------|----------------|----------------|----------------|
| teacher_policy_shaping_enable    | true           | true           | false          |
| teacher_policy_shaping_beta      | 0.1            | 0.1            | N/A            |
| use_dr3                          | false          | false          | true           |
| dr3.feature_mode                 | N/A            | N/A            | v3_aug         |
| dr3.disc_temperature             | N/A            | N/A            | 1.5            |
| dr3.use_policy_shaping           | N/A            | N/A            | true           |
| dr3.policy_shaping_beta          | N/A            | N/A            | 0.1            |
| state_channel.enable             | false          | true           | true           |
| state_channel.grpo_decouple      | N/A            | true           | true           |
| state_channel.beta               | N/A            | 0.2            | 0.2            |
| state_channel.match_mode         | N/A            | stage          | stage          |
| state_channel.step_level.enable  | N/A            | true           | true           |
| state_channel.step_level.eta     | N/A            | 0.05           | 0.05           |

## Appendix: Wandb Run IDs

| Experiment   | Run ID    | Run Dir Timestamp   |
|-------------|-----------|---------------------|
| LUFFY       | o405qtk1  | 20260331_234059     |
| LUFFY+SC    | m4wx6gwh  | 20260403_000352     |
| Hybrid      | lgdzhqmu  | 20260403_035001     |
| DUET_orig   | bgokw3m6  | 20260401_062323     |
| CHORD       | i7406ada  | 20260401_024132     |
| OnPolicy    | 0wq1m98g  | 20260331_025725     |
