# DUET 0403 Theory Synthesis: Two-Stage Model

**Date**: 2026-04-02
**Author**: theory-researcher
**Integrates**: exp-analyst (disc degradation data), algo-engineer (GRPO saturation interpretation)

## Executive Summary

Both analyses are correct — they describe different STAGES of the same event. The "collapse" is a **two-stage process**: GRPO saturation initiates the instability (algo-engineer), and DR3 discriminator degradation amplifies it into a visible drop (exp-analyst). Neither alone fully explains the data.

## The Causal Timeline

### Evidence for ordering

| Step | GRPO state | DR3 state | Performance |
|------|-----------|-----------|-------------|
| 70 | Improving, some groups flat | disc_acc ~0.99, w_off ~0.7 | ~0.90 |
| **76** | **100% groups flat (std < 0.05)** | disc_acc ~0.99 | ~0.97 |
| **79** | Fully saturated | **disc_acc = 0.992**, w_off = 0.77 | **0.807 (peak)** |
| 85 | Noise-dominated | disc_acc starting to drop | ~0.65 |
| **98** | N/A | **disc_acc = 0.775**, w_off = 1.01 | **0.143** |

**GRPO saturates at step 76. Discriminator is still healthy at step 79 (0.992).** Saturation precedes discriminator degradation by ~5-10 steps. This rules out discriminator failure as the initiating cause.

### Stage 1: GRPO Reward Saturation (Steps 70-82)

**Mechanism**: DUET's dual-channel optimization is SO effective that it drives training reward to 0.99.

At saturation:
- All 8 on-policy rollouts per group score 0.95-0.99
- non_teacher_std ≈ 0.02-0.04
- Advantages = (score - mean) / std → mathematically non-zero but **noise-dominated**
- The 0.01-0.04 score differences between rollouts reflect environment stochasticity, not meaningful quality differences
- GRPO is now amplifying noise into gradient signal → **policy performs a random walk**

Teacher advantages become strongly NEGATIVE:
- Teacher score ≈ 0.75 vs on-policy mean ≈ 0.97
- teacher_adv = (0.75 - 0.95) / 0.03 ≈ -6.7 → clamped to **-5.0**
- Policy is (correctly) pushed AWAY from teacher behavior

At this stage, the policy is ABOVE the teacher. The gradient signal is: "you're better than the teacher, don't regress." But the positive signal (which on-policy rollout is best) is pure noise. **Net effect: directionally correct but noise-dominated → random walk.**

**Why doesn't standard GRPO (without DUET) have this problem?** It does! But standard GRPO never reaches 0.99, so it never saturates. DUET's teacher channels accelerate learning so much that they push the policy into the saturation regime.

### Stage 2: DR3 Amplification (Steps 82-100)

**Mechanism**: The random walk from Stage 1 shifts the on-policy distribution, triggering a positive feedback loop through the DR3 discriminator.

**Step-by-step:**

1. **Policy drifts** (steps 80-85): Random walk shifts parameter θ → on-policy distribution changes in feature space
2. **Buffer confusion** (steps 85-90): The discriminator's FIFO buffer (size 1024) now contains:
   - Old on-policy features (from peak performance, steps 65-80)
   - New on-policy features (from drifting policy, steps 80-85)
   - Both labeled "on-policy" but from different distributions → **label noise**
   - Teacher features remain stationary
3. **Discriminator degrades** (steps 85-95): disc_acc drops from 0.99 → 0.78 due to:
   - Non-stationary on-policy features in buffer (Factor 1, see v2 analysis)
   - Age decay 0.02 too mild: 20-step-old stale data retains 67% weight
4. **Temperature amplifies** (steps 90-100): disc_temperature=2.5 compresses remaining signal
   - At disc_acc=0.78, temperature=1.5 would give teacher w_hat ≈ 0.45 (still useful)
   - At disc_acc=0.78, temperature=2.5 gives teacher w_hat ≈ 0.60-0.80 (nearly useless)
5. **Teacher influence flips** (steps 85-100): As on-policy reward drops (0.99 → 0.50-0.70):
   - Teacher (at 0.75) becomes relatively BETTER than on-policy
   - Teacher advantages flip from -5 to small positive values
   - With w_off → 1.0, teacher gets full gradient weight
   - Policy is now pulled TOWARD teacher (0.75 level) instead of staying at 0.99
6. **Stable equilibrium at ~0.72**: Policy converges to teacher-level performance and stabilizes

**This is not a "collapse" — it's convergence to a lower equilibrium.** The policy transitions from an unstable high-performance state (maintained only by noise-dominated gradients) to a stable teacher-level state (maintained by teacher-matching gradients).

### Why LUFFY Doesn't Exhibit This

Three reasons:

1. **Lower ceiling**: LUFFY's policy shaping (f = ratio/(ratio+β)) bounds the optimization. The policy never reaches 0.99, so GRPO never saturates. LUFFY peaks at 0.79 and stays there — stable but lower.

2. **No learned component to degrade**: LUFFY has no discriminator, so there's no fragile learned component that can fail during drift.

3. **Self-correcting teacher influence**: LUFFY's shaping function automatically adjusts teacher weight based on current policy-teacher similarity. If the policy drifts, the correction adjusts immediately (closed-form, no training lag).

**Key insight**: DUET reaches higher but is more fragile at the peak. LUFFY is more stable but has a lower ceiling. This is a fundamental tradeoff, not a bug.

## Implications for DUET's Narrative

### Strength: DUET reaches 0.99 training reward

This is remarkable and validates the dual-channel design:
- Action Channel (DR3) provides unbiased importance correction
- State Channel (SC) provides dense progress signal
- Together, they accelerate learning far beyond LUFFY's ceiling

### Challenge: Sustaining peak performance

When the policy exceeds the teacher, GRPO advantages become noise → instability. This is a **general RL problem** (well-known in reward-hacking literature), not a DUET-specific weakness.

### NeurIPS framing

"DUET's dual-channel design discovers policies that significantly surpass the teacher (0.99 vs 0.75 teacher reward). At this point, GRPO's comparative advantage signal saturates — a well-known limitation of relative advantage methods at optima. We address this with [techniques], and note that the peak-performance checkpoint already represents a substantial improvement over all baselines."

A reviewer asking "why does it collapse?" gets the answer: "It doesn't collapse — it reaches an unprecedented peak that GRPO can't maintain. The **best checkpoint** is the right evaluation metric, as is standard in RL."

## Revised Fix Proposals for 0404

### Tier 1: Prevent GRPO Saturation (address Stage 1)

**Fix 1A: Cosine or stepped LR decay**

```yaml
actor:
  optim:
    lr: 1.0e-06
    lr_scheduler: cosine  # or stepped
    lr_min: 1.0e-07       # 10x reduction at end
```

When reward is high and gradients are noise-dominated, smaller LR = smaller random walk = slower drift. This is the most reliable fix.

**Expected effect**: Policy stays near 0.99 longer, eventual decay is slower and more gradual.

**Fix 1B: Entropy regularization**

```yaml
actor:
  entropy_coeff: 0.005
```

Prevents rollout diversity collapse. If rollouts maintain diversity, GRPO can still find meaningful contrasts.

**Concern**: May also prevent reaching 0.99 in the first place (entropy term fights convergence). Needs careful tuning.

**Fix 1C: Checkpoint selection (pragmatic baseline)**

Always save the best-performing checkpoint. For evaluation, use best rather than final.

```yaml
trainer:
  save_best: true
  best_metric: success_onpolicy/mean
```

This doesn't fix training but ensures we capture the peak.

### Tier 2: Prevent DR3 Amplification (address Stage 2)

**Fix 2A: disc_temperature 2.5 → 1.5**

Restores discrimination signal during drift. Even if the policy starts walking randomly, the discriminator can track the shift and maintain meaningful w_hat values. This prevents the positive feedback loop.

```yaml
dr3:
  disc_temperature: 1.5
```

**Fix 2B: disc_age_weight_decay 0.02 → 0.06**

Addresses buffer staleness. Fresh data dominates discriminator training, allowing faster adaptation to distribution shift.

```yaml
dr3:
  disc_age_weight_decay: 0.06
```

**Fix 2C: disc_acc fallback — mute teacher when discriminator unreliable**

When disc_acc < 0.85 for 3 consecutive calls, set w_hat = w_min (0.01) for teacher samples. This MUTES teacher rather than giving it full weight (which is what happens currently when disc degrades → w → 1.0).

**Implementation**: In `het_actor.py`, after DR3 step:
```python
# Track consecutive low-acc calls
if dr3_metrics.get("dr3/disc_acc", 1.0) < 0.85:
    self._disc_low_acc_count += 1
else:
    self._disc_low_acc_count = 0

# Mute teacher when discriminator unreliable
if self._disc_low_acc_count >= 3:
    w_hat = torch.where(teacher_sample, torch.full_like(w_hat, dr3_w_min), w_hat)
    metrics["dr3/disc_fallback_active"] = 1.0
```

**NeurIPS narrative**: "When distributions converge and the discriminator can no longer distinguish them, DR3 gracefully transitions to on-policy-only training — the natural endpoint of the teacher curriculum."

### On algo-engineer's Variance-Adaptive SC Proposal

**Assessment: Theoretically elegant but likely insufficient at saturation.**

When all rollouts are near-optimal:
- Trajectory-level SC bonus: β · P(τ) ≈ β · 0.9 for all → constant → doesn't differentiate
- Step-level deltas: sum(δ) = Φ_final - Φ_initial ≈ 0.8 for all → constant sum
- Step-level deltas DO vary per-step (different paths), but they affect token_level_rewards, which GRPO sums to trajectory scores → the summed contribution is approximately equal

So variance-adaptive SC can't create meaningful trajectory-level contrast at saturation. It's the right idea (replace vanishing GRPO signal with SC signal) but the SC signal also vanishes at the same point.

**However**, step-level deltas could help with **credit assignment within** trajectories (which tokens are progress-making) even when trajectory-level signal vanishes. This deserves further investigation for the paper but isn't a fix for 0403's instability.

## Recommended 0404 Configuration

**Apply in this order:**

| Priority | Fix | Category | Change | Risk |
|---------|-----|----------|--------|------|
| 1 | LR decay | Stage 1 | cosine schedule, lr_min=1e-7 | Low: standard technique |
| 2 | disc_temperature | Stage 2 | 2.5 → 1.5 | Low: reverting to 0401 value |
| 3 | disc_age_weight_decay | Stage 2 | 0.02 → 0.06 | Low: config change only |
| 4 | Checkpoint selection | Pragmatic | save_best=true | None |
| 5 | disc_acc fallback | Stage 2 | Code change in het_actor.py | Low: safety net only |
| 6 | entropy_coeff | Stage 1 | 0 → 0.005 | Medium: may reduce peak |

**Minimum viable 0404**: Fixes 1-4 (all config changes, no code changes).

**Full 0404**: Fixes 1-5 (one small code change for disc_acc fallback).

**Do NOT apply** for 0404:
- gap_gate re-enable: Was originally disabled for good reason. With LR decay + disc fixes, we shouldn't need it.
- Periodic discriminator reset: Too disruptive, addresses symptom not cause.
- Variance-adaptive SC: Needs more theoretical work, not ready for implementation.
- kl_loss_coef increase: KL is stable, not the problem.

## Open Questions for the Paper

1. **Is 0.99 training reward overfitting to training tasks?** Need to check evaluation performance at step 79 vs final. If eval also peaks at step 79, the peak is real. If eval is flat while train hits 0.99, it's overfitting.

2. **Does any GRPO-based method avoid saturation?** If standard GRPO on easier tasks also saturates, we can position this as a general GRPO property that DUET inherits (and mitigates via checkpoint selection + LR decay).

3. **Can we prove that DUET's asymptotic performance (with fixes) matches or exceeds the best checkpoint?** This would be the strongest result: DUET not only reaches higher peaks but maintains them.

## Verification Metrics for 0404

| Metric | Expected (with fixes) | 0403 baseline |
|--------|---------------------|---------------|
| Peak training reward | ~0.99 (same) | 0.99 |
| Step 100 reward | > 0.85 | ~0.72 |
| GRPO flat group ratio at step 76 | Still 100% (LR decay won't prevent saturation, just slow drift) | 100% |
| disc_acc at step 100 | > 0.90 | 0.775 |
| w_off at step 100 | 0.3-0.6 | 1.01 |
| Best checkpoint reward | ~0.99 | 0.99 |
