# DUET 0401 Regression Diagnosis: WebShop 3B

## Executive Summary

**DUET 0401 underperforms original DUET on WebShop due to the std floor (0.1) killing beneficial teacher advantage explosions.** Stage-based SC is a secondary issue: it's nearly uniform within GRPO groups, so it cancels in normalization and effectively acts as a no-op.

**Recommendation: Revert the std floor. SC stage matching needs redesign but is not the primary cause of regression.**

---

## Two Changes Investigated

| Change | File | Impact |
|--------|------|--------|
| 1. Std floor at 0.1 | `ae_ray_trainer.py:502-508` | **PRIMARY CULPRIT** - kills beneficial teacher advantage explosions |
| 2. Stage-based SC | `state_progress.py` (match_mode: stage) | Secondary issue - nearly uniform within groups, cancels in GRPO |

---

## Finding 1: Std Floor Kills Teacher Gradient Signal (PRIMARY)

### The mechanism

When on-policy rewards converge within a GRPO group (all samples get similar reward), the group std collapses toward 0. In the GRPO advantage formula `adv = (reward - mean) / std`, this causes:

- **Original DUET (no floor):** Teacher advantage = (1.0 - converged_mean) / tiny_std -> MASSIVE positive value (1000-22000x). PPO clipping bounds the actual gradient, but the signal maximally pushes toward teacher actions.
- **0401 (floor=0.1):** Teacher advantage = (1.0 - converged_mean) / 0.1 -> moderate positive (0.05-0.45). The gradient signal from teacher is severely attenuated.

### Evidence: 33/100 steps have teacher_adv > 100 in original

The original DUET experiences std collapse in **33% of all training steps**, with teacher advantages reaching:

| Step | Teacher Advantage |
|------|------------------|
| 33 | 17,563 |
| 27 | 15,680 |
| 23 | 14,048 |
| 46 | 11,533 |
| 65 | 11,168 |
| 88 | 10,648 |
| 57 | 10,001 |
| 54 | 9,528 |
| 4 | 8,493 |
| 28 | 6,596 |
| 41 | 6,572 |
| 31 | 5,760 |
| 98 | 4,840 |

0401's teacher advantages NEVER exceed 0.45 (std floor caps them).

### Concrete example: Group 3555 at step 30

**Original:** All 7 on-policy samples scored 0.7 (identical). std = 0.0. Teacher = 1.0.
-> Teacher advantage = **22,132** (explosive signal toward teacher)

**0401:** On-policy rewards: [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.88] (SC adds ~0.07 each). std = 0.042.
-> Teacher advantage = **0.168** (weak signal toward teacher, 131,000x smaller!)

### Per-group std distributions confirm floor triggers frequently

| Step | Groups with std < 0.1 (0401) | Groups with std < 0.1 (Orig) |
|------|------------------------------|------------------------------|
| 10 | 3/8 (37.5%) | 1/8 (12.5%) |
| 30 | 4/8 (50.0%) | 5/8 (62.5%) |
| 50 | 2/8 (25.0%) | 2/8 (25.0%) |
| 70 | 4/8 (50.0%) | 1/8 (12.5%) |
| 90 | 3/8 (37.5%) | 6/8 (75.0%) |

At step 90, the original has 6/8 groups with std < 0.1 (natural convergence). Without the floor, these all produce explosive teacher advantages. With the floor, all are capped.

### Why explosions don't destabilize the original

PPO clipping: `clip(ratio, 1-eps, 1+eps)`. Even with advantage = 22,132, the ratio is clipped to [1-0.28, 1+0.6] (per config: `off_cliprange_high: 0.6`, `clip_ratio_high: 0.28`). The actual gradient magnitude is bounded. The explosion simply means the policy update **maximally favors teacher actions** in those groups.

### Timeline shows 0401 falls behind in late training

| Step Range | 0401 Reward (avg) | Orig Reward (avg) | Delta |
|-----------|-------------------|-------------------|-------|
| 1-10 | 0.166 | 0.135 | +0.031 (0401 slightly ahead) |
| 11-20 | 0.353 | 0.327 | +0.026 (0401 slightly ahead) |
| 21-30 | 0.557 | 0.513 | +0.044 (0401 ahead) |
| 31-40 | 0.601 | 0.538 | +0.063 (0401 ahead) |
| 41-50 | 0.633 | 0.608 | +0.025 (0401 slightly ahead) |
| 51-60 | 0.619 | 0.624 | -0.005 (CROSSOVER) |
| 61-70 | 0.666 | 0.686 | -0.020 (Orig pulls ahead) |
| 81-90 | 0.686 | 0.733 | -0.047 (Orig dominates) |

The std floor actually HELPS early (prevents destabilizing huge gradients before PPO clipping "warms up"). But in late training, when on-policy converges and teacher demonstrations become the primary learning signal, the floor kills the mechanism that drives continued improvement.

### Success rates at late steps

| Step | 0401 Success | Orig Success |
|------|-------------|-------------|
| 50 | 96.4% | 83.9% |
| 70 | 91.4% | 94.8% |
| 90 | 84.2% | 98.2% |
| 100 | 87.9% | 98.3% |

0401 peaks around step 50 then regresses. Original steadily improves to 98%.

---

## Finding 2: Stage SC Is Nearly a No-Op for GRPO (SECONDARY)

### The mechanism

Stage-based SC assigns progress based on WebShop page type:
- search_home: 0.0
- search_results: 0.2
- product_detail: 0.5
- purchase_complete: 1.0

SC bonus = beta * mean(progress_across_steps). Since all on-policy samples in a group follow similar trajectories (search -> results -> detail -> buy), they all reach similar stages -> SC bonus is nearly uniform within each group.

### Evidence: SC bonus within-group std is negligible

| Step | SC bonus mean | SC bonus within-group std | SC bonus nonzero |
|------|-------------|--------------------------|------------------|
| 1 | 0.041 | 0.011 | 49/57 (86%) |
| 10 | 0.037 | 0.011 | 42/58 (72%) |
| 30 | 0.042 | 0.006 | 35/59 (59%) |
| 50 | 0.077 | 0.008 | 56/56 (100%) |
| 90 | 0.062 | 0.011 | 49/57 (86%) |

The within-group std (0.006-0.011) is tiny compared to the bonus mean (0.04-0.08). This means SC adds roughly the same value to all samples in a group.

### GRPO normalization cancels uniform offsets

GRPO: `adv_i = (reward_i - group_mean) / group_std`

If SC adds constant `c` to all rewards in a group:
- New mean = old_mean + c
- New adv_i = (reward_i + c - (old_mean + c)) / std = (reward_i - old_mean) / std = old adv_i

**SC bonus cancels out in GRPO normalization.** It doesn't change the learning signal.

### SC also reduces teacher-on gap slightly

SC bonus shifts on-policy rewards up by ~0.04-0.08, reducing the teacher-onpolicy gap. With teacher baseline separation (`teacher_baseline: all_mean`), the teacher baseline also shifts up slightly, reducing teacher advantages. This is a minor negative effect.

### The original had SC enabled but hash matching failed

Original config also had `state_channel.enable: true` but used hash-based matching (the default before 0401). Hash matching produced **zero matches** for WebShop (sc_matched_states = 0, sc_bonus = 0 across all 100 steps). The original effectively ran without SC.

---

## Root Cause Analysis

| Factor | Direction | Magnitude | Explanation |
|--------|-----------|-----------|-------------|
| Std floor caps teacher advantage | Negative | **LARGE** | Kills beneficial 1000-22000x teacher gradient boosts in 33% of steps |
| Std floor stabilizes early training | Positive | Small | Slightly better reward in steps 1-50 |
| Stage SC uniform within groups | Neutral/Slight Negative | Small | GRPO cancels uniform offsets; minor teacher gap reduction |
| Stage SC nonzero bonus | Neutral | Negligible | Doesn't change relative ordering |

**Net effect: Std floor dominates. It trades early stability for late-training stagnation.**

---

## Recommendations

### Immediate: Revert std floor

Remove lines 503-508 of `ae_ray_trainer.py`. The std collapse -> teacher advantage explosion is a FEATURE, not a bug. PPO clipping ensures numerical safety. The original's 33/100 steps with explosions don't destabilize training; they accelerate learning from teacher demonstrations.

### Alternative: Adaptive floor

If concerned about early-training instability, use a decaying floor:
```python
# Floor decays from 0.1 -> 0.0 over warmup_steps
floor = max(0.0, 0.1 * (1 - step / warmup_steps))
if std.item() < floor:
    std = torch.tensor(floor, device=scores.device)
```

### SC needs redesign

Stage-based SC fails because WebShop stages are too coarse (4 values) and all samples in a group follow similar trajectories. For SC to help GRPO, it needs **within-group discriminative power** -- different SC values for different quality levels within the same task group. Options:
1. Finer-grained progress (e.g., product match quality, search relevance)
2. Step-level deltas with sufficient observation diversity
3. Hash matching with better observation normalization

---

## Raw Data

### Key metric comparison at selected steps

| Step | Metric | 0401 | Orig | Diff |
|------|--------|------|------|------|
| 30 | reward_onpolicy_mean | 0.534 | 0.592 | -0.059 |
| 30 | adv_teacher_sample_mean | 0.174 | 4426.7 | -4426.5 |
| 30 | onpolicy_adv_pos_ratio | 0.373 | 0.610 | -0.237 |
| 50 | reward_onpolicy_mean | 0.866 | 0.740 | +0.126 |
| 50 | adv_teacher_sample_mean | 0.053 | 0.069 | -0.016 |
| 90 | reward_onpolicy_mean | 0.645 | 0.806 | -0.160 |
| 90 | adv_teacher_sample_mean | 0.154 | 0.049 | +0.105 |
| 100 | reward_onpolicy_mean | 0.734 | 0.776 | -0.042 |
| 100 | success_rate | 87.9% | 98.3% | -10.4% |
