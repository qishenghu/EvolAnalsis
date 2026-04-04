# DUET 0403 Collapse Analysis Report
Generated: 2026-04-02

## Executive Summary

**The "collapse" at steps 80-100 is NOT a catastrophic failure — it is GRPO advantage saturation followed by mean regression.** DUET 0403 trains too fast, reaching near-perfect performance (0.99 reward at step 79, 80.7% samples ≥0.95 at step 80), which collapses within-group reward variance to near-zero. Without gradient signal, the policy drifts back to the LUFFY-level baseline.

**The 0403 changes contribute but are not the sole cause:**
- Teacher adv clip ±5: **NEVER activates** (max teacher adv ~0.7, nowhere near ±5)
- gap_gate OFF: Removes self-correction; teacher_loss_scale stays at 0.0
- disc_temperature=2.5: **Flattens DR3 w_hat to ~1.0 always** (effectively disables importance correction)
- SC-GRPO decouple: Working correctly, SC bonus stable at ~0.07

**Dual root cause:**
1. **GRPO advantage collapse at high performance.** At step 80, 75% of GRPO groups have within-group std < 0.05 (100% at step 76). Near-zero advantages → no learning signal.
2. **DR3 discriminator degradation after step 80.** disc_acc drops from 0.99 (step 79) to 0.79 (step 100) as on-policy converges to teacher distribution. Combined with disc_temperature=2.5 (which already flattened w_hat to ~1.0), DR3 provides zero useful signal during the critical regression phase.

---

## 1. Pinpointing the Collapse

### Reward Trajectory
```
 Phase       | DUET Avg | LUFFY Avg | Gap      | Verdict
--------------------------------------------------------------
 Steps 41-50 |   0.6933 |    0.5338 |  +0.1595 | DUET leads
 Steps 51-60 |   0.7316 |    0.5603 |  +0.1713 | DUET leads (peak gap)
 Steps 61-70 |   0.8014 |    0.6715 |  +0.1299 | DUET leads
 Steps 71-80 |   0.8418 |    0.7386 |  +0.1033 | DUET leads
 Steps 81-90 |   0.7751 |    0.7571 |  +0.0180 | Gap vanishes
 Steps 91-100|   0.7033 |    0.7179 |  -0.0147 | LUFFY overtakes
```

**Peak**: reward_onpolicy=0.9896 at step 79 (orig reward 0.9045 + SC bonus 0.085)
**Collapse onset**: step 81 (reward drops to 0.8033, 10% drop from peak)

### Critical Transition (Steps 79→81)
| Metric | Step 79 | Step 80 | Step 81 |
|--------|---------|---------|---------|
| reward_onpolicy | 0.990 | 0.971 | 0.803 |
| orig_reward (no SC) | 0.905 | 0.900 | 0.744 |
| success_rate | 100% | 100% | 89.7% |
| %samples ≥0.95 | 62.5% | **80.7%** | 36.2% |
| %zero_reward | 0% | 0% | 10.3% |
| resp_len_mean | 1779 | 1997 | **2688** |
| grp_std (mean) | 0.056 | **0.025** | 0.191 |
| %flat_groups | 62.5% | **75.0%** | 25.0% |
| adv_pos_frac | 0.75 | **0.89** | 0.60 |

**Step 80 is the saturation point**: 75% of GRPO groups are "flat" (within-group std < 0.05), 80.7% of samples score ≥0.95, advantages are near-zero (mean=-0.0005). The model has essentially solved most tasks, but GRPO has no signal to maintain this level.

---

## 2. Teacher Advantage Clip Analysis (±5)

**Result: The clip is IRRELEVANT for this run.**

```
Steps 40-100: Teacher advantage range is [0.000, 1.093]
Max observed: 1.093 (step 77)
0% of teacher samples near ±5 boundary at any step
```

Teacher sample-level advantages are consistently in [0.0, 0.7] range. The ±5 clip has zero effect. This is because teacher baseline separation keeps teacher advantages in a reasonable range — teacher rewards are always 1.0, so with separation, the advantage normalization is tight.

---

## 3. KL Divergence Analysis

**Result: KL is NOT diverging. DUET actually drifts LESS than LUFFY.**

```
old_log_prob_mean (on-policy):
  Step 60: DUET=-1.529, LUFFY=-1.482
  Step 80: DUET=-1.551, LUFFY=-1.516
  Step 100: DUET=-1.454, LUFFY=-1.385

  DUET drift (60→100): 0.074
  LUFFY drift (60→100): 0.096
  DUET/LUFFY drift ratio: 0.77x (DUET is MORE stable)
```

No KL explosion. The 0403 changes (gap_gate OFF, no adaptive_weight) do not cause excessive policy drift.

---

## 4. SC-GRPO Decoupling Verification

**Result: Working correctly. SC bonus is stable and properly excluded from teacher samples.**

```
SC Metrics (on-policy only, steps 40-100):
  SC bonus: stable 0.05-0.085 (avg ~0.068)
  SC coverage: 59-100% (avg ~85%)
  Bonus/reward ratio: 7-15% (healthy)

Teacher SC contamination check:
  Steps 60,70,80,90,100: 0 teacher samples with nonzero SC bonus ✓
```

The SC-GRPO decoupling is functioning as designed. SC adds ~0.07 bonus to on-policy samples post-GRPO normalization. This is proportional (~9% of reward) and not causing inflation.

---

## 5. Entropy and Degeneracy

**Result: No entropy collapse. No response length explosion.**

```
Entropy (on-policy token mean):
  Step 60: 0.401 → Step 80: 0.440 → Step 100: 0.411
  No decline — slightly INCREASING at the saturation point

Response length (on-policy mean):
  Steps 60-70: ~1500-2200 (tightening, std ~500)
  Steps 71-80: ~1600-2000 (stable)
  Steps 81-90: ~2000-2700 (expanding)
  Steps 91-100: ~1800-2300 (recovering)

  Note: Step 81 has a length spike (2688, std=1699) coinciding with reward drop.
  This is a CONSEQUENCE, not a cause — the model starts generating longer
  but lower-quality responses when it loses gradient direction.
```

---

## 6. DR3 Behavior (from wandb: run-20260402_162739-j2rle81i)

**DR3 discriminator degrades after step 80, confirming exp-analyst's finding.**

```
Phase         | disc_acc | disc_loss | w_mean | w_max  | tch_grad_share | reward
----------------------------------------------------------------------------------
Steps 1-10    |  0.4315  |   0.3918  | 1.0013 | 1.0139 |     0.2734     | 0.1945
Steps 11-30   |  0.8247  |   0.3700  | 1.0440 | 1.0864 |     0.4076     | 0.4995
Steps 31-50   |  0.9802  |   0.2438  | 1.0464 | 1.0983 |     0.4139     | 0.6695
Steps 51-70   |  0.9441  |   0.3029  | 1.0442 | 1.1029 |     0.2892     | 0.7664
Steps 71-80   |  0.9810  |   0.2205  | 1.0458 | 1.0976 |     0.3099     | 0.8420
Steps 81-90   |  0.9426  |   0.3760  | 1.0266 | 1.0971 |     0.1887     | 0.7753  ← degradation starts
Steps 91-100  |  0.8233  |   0.5207  | 1.0100 | 1.0958 |     0.1175     | 0.7034  ← disc_acc back to step-8 level
```

**Key findings:**
1. **disc_acc degrades**: 0.99 (step 79) → 0.79 (step 100). The discriminator loses ability to distinguish on-policy from teacher as policy improves.
2. **w_mean always ~1.0**: disc_temperature=2.5 flattens all importance weights to ~1.0 (w_std=0.000 at every step). DR3 effectively provides NO importance correction throughout the entire run.
3. **w_max always 1.05-1.13**: Even the most extreme weight is barely different from 1.0. Teacher samples are treated identically to on-policy.
4. **teacher_gradient_share DECLINES**: 0.41 (steps 31-50) → 0.31 (steps 71-80) → 0.12 (steps 91-100). Teacher is NOT dominating in late training.
5. **teacher_loss_scale = 0.0 always**: Confirms gap_gate OFF / adaptive_weight OFF.

**Implication**: disc_temperature=2.5 is too high — it neutralizes DR3's core mechanism (importance weight correction). The discriminator trains and achieves high accuracy but its signal is flattened before being used. When disc_acc later degrades, there's no fallback because the weights were already at 1.0.

---

## 7. GRPO Saturation: The Smoking Gun

### Group Variance Collapse
```
DUET %Flat Groups (within-group std < 0.05):
  Step 67: 75.0%   (reward=0.835)
  Step 68: 87.5%   (reward=0.921)
  Step 72: 75.0%   (reward=0.819)
  Step 76: 100.0%  (reward=0.942)  ← ALL GROUPS FLAT
  Step 79: 62.5%   (reward=0.990)
  Step 80: 75.0%   (reward=0.971)  ← SATURATION PEAK
  Step 81: 25.0%   (reward=0.803)  ← Variance returns as performance drops
  Step 95: 0.0%    (reward=0.680)  ← Back to normal variance

LUFFY %Flat Groups (same metric):
  Steps 60-80: 0-50% (avg ~25%)   ← Much more variance throughout
  Steps 80-100: 0-38% (avg ~20%)  ← Stable learning signal maintained
```

**This is the core mechanism**: When DUET gets very good (reward>0.85), most GRPO groups converge to similar reward values. GRPO normalizes within-group, so low variance → near-zero advantages → no learning signal. The policy then drifts randomly, eventually regressing.

LUFFY doesn't have this problem because (a) it doesn't reach as high performance, and (b) its IS-corrected teacher advantages can be enormous (observed values up to 23,000), providing strong gradient anchors that prevent drift even when on-policy variance is low.

### Advantage Signal Quality
```
Step 80 (peak):  adv_mean=-0.0005, adv_std=0.045, pos_frac=0.89
Step 85 (post):  adv_mean=-0.0026, adv_std=0.085, pos_frac=0.68
Step 90:         adv_mean=-0.0113, adv_std=0.????, pos_frac=0.46
Step 95:         adv_mean=-0.0369, adv_std=0.????, pos_frac=0.51
```

At step 80, 89% of advantages are positive — everything gets uniformly encouraged, which is equivalent to no signal. By step 90, advantages become slightly negative on average, actively pushing the policy away from its peak.

---

## 8. DUET vs LUFFY Stability Comparison

### Why LUFFY Doesn't Collapse
1. **Lower peak performance** → Groups maintain variance → GRPO signal persists
2. **IS-corrected teacher advantages** → Massive teacher gradient (10,000x) serves as anchor
3. **Policy shaping** → LUFFY's KL-based teacher influence is inherently bounded and stable

### Why DUET Overshoots and Regresses
1. **SC bonus** accelerates early training (good!) → reaches high performance faster
2. **DR3-corrected advantages** are well-behaved (max ~0.7) → no strong teacher anchor when on-policy saturates
3. **Teacher baseline separation** keeps teacher advantages small → can't provide gradient anchor
4. **No mechanism to maintain performance once reached** → GRPO variant of the "forget what you learned" problem

---

## 9. Verdict on Competing Theories

### theory-researcher's hypothesis: "Teacher adv fixed at +5, DR3 fade-out weakened"
| Claim | Verdict | Evidence |
|-------|---------|----------|
| Teacher adv fixed at +5 from step 40 | **WRONG** | Max teacher adv = 1.09 (step 77). 0% clip activation. |
| disc_temp=2.5 keeps w_hat at 0.5-0.8 | **WRONG** (reversed) | w_hat stays at ~1.0 (not 0.5-0.8). High temp FLATTENS weights. |
| teacher_gradient_share constant ~25-30% | **WRONG** | Declines from 0.41 → 0.12 over training. |
| 3B over-imitates 72B teacher | **PARTIALLY RIGHT** | Model converges toward teacher behavior at peak, but teacher gradient share is LOW (12%) during collapse. |

### exp-analyst's hypothesis: "DR3 discriminator degrades"
| Claim | Verdict | Evidence |
|-------|---------|----------|
| disc_acc degrades 0.99→0.78 after step 80 | **CORRECT** | disc_acc: 0.99 (step 79) → 0.79 (step 100) |
| w_off → 1.0 (importance correction fails) | **TRUE but misleading** | w was ALWAYS ~1.0 due to disc_temp=2.5 |
| Teacher adv clip ±5 not the cause | **CORRECT** | Max teacher adv = 1.09, 0% clip |

### This analysis: "GRPO advantage variance collapse + disc degradation"
| Finding | Evidence |
|---------|----------|
| 75-100% of GRPO groups "flat" at steps 76-80 | Group std < 0.05 for 75% of groups at step 80 |
| Advantages near-zero at peak | adv_mean = -0.0005, 89% positive (uniform encouragement = no signal) |
| disc_acc degrades as policy converges to teacher | disc_acc: 0.99 → 0.79 (steps 80-100) |
| DR3 effectively disabled by disc_temp=2.5 | w_mean ≈ 1.0 at all steps, w_std = 0.000 |
| Policy regresses to LUFFY level, not below | Steps 91-100 avg: DUET 0.70, LUFFY 0.72 |

## 10. Recommendations

### Immediate (for next experiment run)
1. **Lower disc_temperature to 1.5** (was 2.5). This restores meaningful DR3 importance weights so w_hat actually varies with the discriminator's predictions. The previous 0402 run used 1.5 and had much more variation in w.
2. **Re-enable gap_gate or adaptive_weight**. With both OFF and disc_temp=2.5, there is ZERO mechanism to modulate teacher influence. At least one self-correction mechanism must be active.
3. **Learning rate decay at high reward**: Reduce LR when reward_onpolicy_mean > 0.85 to slow drift during the saturation phase.
4. **EMA checkpoint**: Save exponential moving average of model weights to recover from drift.

### Architectural (for DUET design)
5. **Advantage variance floor**: When within-group std < 0.05, inject noise or increase sampling temperature. This prevents the GRPO signal collapse that initiates the regression.
6. **Variance-adaptive SC**: Scale SC bonus inversely with within-group variance so it becomes the primary signal when GRPO collapses.
7. **DR3 teacher anchor**: Set minimum teacher_gradient_share (e.g., 10%) to maintain gradient anchor even at high performance. Currently it drops to 6% at step 100.
8. **Discriminator reset or replay**: When disc_acc drops below 0.90, retrain discriminator with fresh buffer to prevent cascading degradation.
