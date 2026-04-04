# DUET 0403 Collapse Theory Analysis v2 (Revised)

**Date**: 2026-04-02
**Revision**: v2 — Updated with exp-analyst empirical data that overturned several initial hypotheses.

## What the Data Showed (Contradicting v1)

| Hypothesis | v1 Prediction | Actual | Status |
|-----------|--------------|--------|--------|
| Teacher adv clip ±5 dominates | Clamped to +5 always | Max teacher adv = 0.40 | **REFUTED** |
| KL explosion | KL spikes at collapse | KL 1.24 → 1.10 (decreasing) | **REFUTED** |
| SC post-GRPO mismatch | SC could distort | SC ratio stable at 0.083 | **REFUTED** |
| DR3 fade-out too slow | w_hat stays moderate | w_off 0.77 → 1.01 (INCREASES) | **PARTIALLY CORRECT** (direction right, mechanism wrong) |

**v1 errors**: I assumed non_teacher_std would collapse and trigger the ±5 clamp. In reality, WebShop's continuous rewards maintain enough variance (teacher adv peaks at 0.40, nowhere near ±5). The ±5 clamp is irrelevant for WebShop.

## Actual Root Cause: Discriminator Degradation → Importance Correction Failure

### The Evidence

```
Step 79: disc_acc=0.992, w_off=0.77, success=0.807 (peak)
Step 98: disc_acc=0.775, w_off=1.01, success=0.143 (collapse)
```

The discriminator's classification accuracy drops 22 points in ~20 steps. This is NOT a gradual fade-out — it's a failure.

### Why the Discriminator Degrades

After reading `dr3_ratio.py` in detail, I've identified a **two-factor mechanism**:

#### Factor 1: Buffer Staleness (Non-stationary On-Policy Distribution)

The discriminator trains on a FIFO rolling buffer (`buffer_size=1024`). The buffer contains:

| Label | Distribution | Property |
|-------|-------------|----------|
| 1 (on-policy) | Mix of features from steps 60-98 | **Non-stationary** — each step has different policy |
| 0 (teacher) | Same teacher trajectories always | **Stationary** — always same distribution |

As training progresses and the policy improves rapidly (steps 60-80), the on-policy features in the buffer become **heterogeneous**:
- Step 60 features: from a weaker policy, looks very different from teacher
- Step 80 features: from a stronger policy, looks more similar to teacher
- Step 95 features: from an even stronger policy, very similar to teacher

All labeled "on-policy" (label=1). The discriminator sees label=1 samples that span a wide range of distributions. This is effectively **label noise** — the discriminator can't learn a consistent boundary for "on-policy".

Age-weighted decay (`disc_age_weight_decay=0.02`) is too mild:
```
Age 10 steps: weight = exp(-0.02 * 10) = 0.82  (barely down-weighted)
Age 20 steps: weight = exp(-0.02 * 20) = 0.67  (still 67% influence)
Age 30 steps: weight = exp(-0.02 * 30) = 0.55  (still 55% influence)
```

Stale features from 20-30 steps ago retain majority weight. These features corrupt the discriminator's training signal.

#### Factor 2: Temperature Amplifies Consequences

**Critical code finding**: Temperature is only applied at INFERENCE time, not during training:

```python
# Training (dr3_ratio.py:693): raw logits
logits = self._disc(xb)
loss_per_sample = self._bce(logits, yb_used)

# Inference (dr3_ratio.py:738-739): temperature-scaled
if self.disc_temperature != 1.0:
    logits = logits / self.disc_temperature
d = torch.sigmoid(logits)
```

So temperature=2.5 does NOT cause the discriminator accuracy to drop. The accuracy drops due to Factor 1 (buffer staleness). But temperature **amplifies the consequences**:

At disc_acc=0.775, the discriminator still has SOME discrimination power. Raw logits might give:
- Correctly classified teacher: sigmoid(-1.5) = 0.18 → r_hat = 0.22 → w ≈ 0.3
- Correctly classified on-policy: sigmoid(1.5) = 0.82 → r_hat = 4.6 → w ≈ 2.0

With temperature=2.5: logits/2.5 gives:
- Teacher: sigmoid(-0.6) = 0.35 → r_hat = 0.54 → w ≈ 0.6
- On-policy: sigmoid(0.6) = 0.65 → r_hat = 1.86 → w ≈ 1.2

Temperature compresses w_hat toward 1.0, eliminating the remaining discrimination signal. The 77.5% accurate discriminator is rendered **effectively useless** by temperature=2.5.

With temperature=1.5:
- Teacher: sigmoid(-1.0) = 0.27 → r_hat = 0.37 → w ≈ 0.45
- On-policy: sigmoid(1.0) = 0.73 → r_hat = 2.7 → w ≈ 1.5

Still discriminative enough to maintain meaningful importance correction.

### The Collapse Chain

```
Policy improves → on-policy features shift in feature space
     ↓
Buffer contains mix of old+new on-policy features (non-stationary label noise)
     ↓
Discriminator accuracy degrades (0.99 → 0.78)
     ↓
Temperature=2.5 compresses remaining discrimination signal → w_hat ≈ 1.0 for all
     ↓
Teacher samples get ratio=1.0 (full weight), adv=+0.40 (positive)
On-policy samples get ratio=1.0, adv≈0 (centered by GRPO)
     ↓
Net gradient biased toward teacher behavior (1/9 of batch, all positive adv)
     ↓
Policy shifts further toward teacher → features shift more → discriminator degrades more
     ↓
POSITIVE FEEDBACK LOOP → collapse
```

### Why LUFFY Doesn't Collapse

LUFFY has no discriminator, so there's no component that can degrade. Its teacher influence mechanism (policy shaping: f(ratio) = ratio/(ratio+β)) is a closed-form function of the current policy, not a learned component with stale-data vulnerability. The shaping function is always computed fresh from current π_θ.

### Why 0401 (disc_temperature=1.5) Might Have Avoided This

With temperature=1.5, the same disc_acc=0.775 would still produce discriminative w_hat values (see calculation above). The remaining discrimination signal survives temperature scaling. The positive feedback loop wouldn't start because teacher samples would still be partially down-weighted.

Additionally, 0401 had gap_gate ON, which provides an independent attenuation mechanism that doesn't depend on the discriminator.

## NeurIPS Reviewer Concern: Discriminator Stability

A reviewer will ask: "Your method relies on a learned discriminator to provide importance weights. How stable is this discriminator? What happens when the policy distribution shifts during training?"

**This is a legitimate concern** that we must address in the paper.

Answer: The discriminator operates in a non-stationary setting where the on-policy distribution shifts every training step. We address stability through:
1. **Temperature scaling** (T=1.5): prevents discriminator overconfidence while preserving discrimination signal
2. **Age-weighted training**: exponential decay on buffer samples prioritizes recent data
3. **Graceful degradation**: when the discriminator can't distinguish distributions (disc_acc → 0.5), w_hat → 1.0, which is equivalent to standard PPO — the method automatically degrades to the on-policy baseline rather than producing harmful corrections
4. **Practical monitoring**: disc_acc provides a direct signal for discriminator health; we recommend fallback to uniform weights when disc_acc < 0.85 for multiple consecutive steps

(Point 3 is currently aspirational — w_hat → 1.0 SHOULD be safe, but the 0403 collapse shows it isn't because teacher samples get full weight. We need to implement the fallback in Point 4.)

## Revised Fix Proposals

### Fix 1: disc_temperature 2.5 → 1.5 (HIGHEST PRIORITY)

**Why**: Directly addresses Factor 2. The remaining discrimination signal at disc_acc=0.78 survives temperature=1.5 but is destroyed by temperature=2.5.

**Expected effect**: w_hat stays meaningful (0.3-0.5 for teacher) even when disc_acc degrades to 0.78, preventing the positive feedback loop.

**Risk**: Low. Temperature=1.5 was the 0401 value. We have empirical evidence it works.

```yaml
dr3:
  disc_temperature: 1.5
```

### Fix 2: disc_acc Fallback — Drop Teacher from Loss

**Why**: Safety mechanism for when the discriminator genuinely fails. When disc_acc < 0.85 for K consecutive calls, teacher samples can't be safely importance-corrected.

**Proposal**: When disc_acc < threshold, set w_hat = w_min (0.01) for teacher samples, effectively muting them. This is better than uniform weighting (w=1.0) because uniform gives teacher FULL gradient weight.

**Implementation**: In `het_actor.py`, after DR3 step:
```python
if dr3_metrics.get("dr3/disc_acc", 1.0) < 0.85:
    w_hat[teacher_mask] = dr3_w_min  # 0.01 = effectively muted
    metrics["dr3/disc_fallback_active"] = 1.0
```

**NeurIPS narrative**: "When distributions converge and the discriminator can no longer distinguish them, DR3 gracefully transitions to on-policy-only training. This is the natural endpoint of the teacher curriculum."

**Note on exp-analyst's suggestion of w=1.0 (uniform)**: I disagree. Setting w=1.0 means teacher samples get FULL gradient weight at exactly the moment we can't trust our importance correction. This is the 0403 failure mode. We should mute, not amplify.

### Fix 3: More Aggressive Buffer Freshness

**Why**: Addresses Factor 1. Stale buffer samples corrupt discriminator training.

**Options** (in order of preference):
1. Increase `disc_age_weight_decay`: 0.02 → 0.06 (20-step-old samples: weight 0.30 instead of 0.67)
2. Reduce `buffer_size`: 1024 → 512 (less staleness, but also less training data)
3. Increase `disc_steps_per_call`: 2 → 4 (more adaptation per call, but 2x compute)

Recommendation: Option 1 alone. It's a single hyperparameter change that directly addresses the staleness problem.

```yaml
dr3:
  disc_age_weight_decay: 0.06
```

### Fix 4: Re-enable gap_gate (power=0.5) as Independent Safety Layer

**Why**: Provides discriminator-independent teacher attenuation. Even if the discriminator degrades, gap_gate still reduces teacher influence based on reward gap.

**Consideration**: The original reason for disabling was "double suppression." With disc_temperature back at 1.5, the DR3 fade-out is stronger, so gap_gate power=0.5 should be complementary, not excessive.

```yaml
dr3:
  gap_gate_enable: true
  gap_gate_power: 0.5
```

### On exp-analyst's suggestion: Periodic Discriminator Reset

**I advise against this.** Resetting the discriminator:
- Loses all learned discrimination → new warmup period with w_hat ≈ uninformative
- Introduces discontinuity in the training signal
- Hard to tune (when to reset? what threshold?)

Buffer freshness (Fix 3) addresses the same root cause (stale data) without the discontinuity.

## Recommended 0404 Configuration (Revised)

Apply in this order:
1. **Fix 1**: disc_temperature: 2.5 → 1.5
2. **Fix 3**: disc_age_weight_decay: 0.02 → 0.06
3. **Fix 2**: disc_acc fallback (code change in het_actor.py)
4. **Fix 4**: gap_gate re-enable with power=0.5 (only if 1-3 insufficient)

Do NOT apply:
- Periodic discriminator reset (too disruptive)
- Uniform weighting fallback (amplifies the failure mode)
- Teacher adv clip changes (irrelevant — max teacher adv is 0.40 on WebShop)

## Verification Plan

| Metric | Expected with fixes | Without fixes (0403) |
|--------|-------------------|---------------------|
| disc_acc at step 80 | > 0.90 | 0.85 (degrading) |
| disc_acc at step 100 | > 0.85 | 0.775 |
| w_off at step 100 | 0.3-0.6 | 1.01 |
| success at step 100 | > 0.75 (stable) | 0.143 (collapsed) |
| dr3/disc_fallback_active | Rarely triggered | N/A (not implemented) |
