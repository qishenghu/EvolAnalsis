# DUET 0403 Collapse Theory Analysis

**Date**: 2026-04-02
**Observation**: DUET 0403 beats LUFFY mid-training but collapses at steps 80-100.

## 0403 Configuration Changes (vs 0401)

| Change | Old | New | Intent |
|--------|-----|-----|--------|
| SC-GRPO decoupling | SC bonus in task_reward | SC bonus post-GRPO | Fair teacher-vs-onpolicy GRPO comparison |
| Teacher adv clip | None (unbounded) | Hardcoded ±5 | Prevent advantage explosion from small std |
| gap_gate | ON | OFF | Remove double suppression |
| adaptive_weight | ON | OFF | Remove gap_gate dependency |
| disc_temperature | 1.5 | 2.5 | Softer DR3 (slower w_hat decay) |

## Root Cause Analysis

### The Critical Observation: Zero Attenuation

0403 removed **both** teacher attenuation mechanisms (gap_gate + adaptive_weight) simultaneously, leaving DR3 w_hat as the **only** mechanism to control teacher gradient influence. But disc_temperature=2.5 deliberately slows w_hat decay. The result: **strong, sustained, unattenuated teacher gradient signal with no self-correction mechanism**.

### Detailed Gradient Trace

#### Step 1: Teacher advantage computation (`ae_ray_trainer.py:524-534`)

```
adv_teacher = (teacher_score - all_mean) / non_teacher_std
adv_teacher = clamp(adv_teacher, -5, +5)
```

Key parameters:
- `teacher_baseline = "all_mean"` → base includes teacher score
- `std_source = "non_teacher"` → std computed from on-policy samples ONLY

**The std collapse positive feedback loop:**

| Training phase | On-policy scores | non_teacher_std | Teacher adv (pre-clamp) | Effective adv |
|---------------|-----------------|-----------------|------------------------|---------------|
| Early (0-30) | {0, 0.1, 0.3, 0.2, 0.4, 0.1, 0, 0.3} | ~0.15 | ~2-3 | ~2-3 |
| Mid (30-60) | {0.4, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.6} | ~0.07 | ~5-8 | **+5 (clamped)** |
| Late (60-100) | {0.6, 0.65, 0.6, 0.65, 0.7, 0.6, 0.65, 0.6} | ~0.03 | ~10-30 | **+5 (clamped)** |

**As on-policy improves and becomes consistent, teacher advantage saturates at +5. This is by design (the clip), but the problem is this +5 is CONSTANT — it provides no signal about whether teacher influence is still helpful.**

#### Step 2: DR3 correction (`het_actor.py:1486-1488`)

```python
old_lp_new[teacher] = log_prob.detach()[teacher] - log_w[teacher]
# This makes the effective PPO ratio = w_hat for teacher samples
```

With disc_temperature=2.5:
- Early: w_hat ≈ 1.0 (discriminator uninformative)
- Mid: w_hat ≈ 0.5-0.8 (discriminator learning but softened)
- Late: w_hat ≈ 0.3-0.6 (still moderate due to high temperature)

Contrast with disc_temperature=1.0: w_hat would reach 0.01-0.1 by step 60.

#### Step 3: PPO loss (`het_core_algos.py:1900-1904`)

For teacher samples with A=+5 and w_hat < 0.8:
```
losses1 = -5 * w_hat        (e.g., -5 * 0.5 = -2.5)
losses2 = -5 * 0.8          (= -4.0, from clip at 1-eps=0.8)
max(-2.5, -4.0) = -2.5      → uses unclipped branch
gradient ∝ -5 * w_hat * ∇log_prob
```

PPO clipping does NOT help here because the unclipped branch dominates when ratio < 1.

#### Step 4: Effective gradient share computation

| Phase | w_hat | Teacher grad magnitude | On-policy grad magnitude | Teacher share (1/9 batch) |
|-------|-------|----------------------|------------------------|--------------------------|
| Early | 1.0 | 5 * 1.0 = 5.0 | ~2.0 | 5/(5+16) = **24%** |
| Mid | 0.7 | 5 * 0.7 = 3.5 | ~1.0 | 3.5/(3.5+8) = **30%** |
| Late | 0.5 | 5 * 0.5 = 2.5 | ~1.0 | 2.5/(2.5+8) = **24%** |

**Teacher gradient share stays at 24-30% throughout training.** This is ~2-3x the data proportion (1/9 = 11%). With gap_gate and adaptive_weight OFF, nothing modulates this.

### The Collapse Mechanism (Hypothesis B + D compound)

**Phase 1 (Steps 0-40): Beneficial teacher learning**
- DR3 discriminator is learning; w_hat ≈ 1.0
- Teacher advantage is moderate (non_teacher_std still large)
- Policy learns genuine useful behaviors from teacher → performance rises

**Phase 2 (Steps 40-80): Over-commitment begins**
- On-policy improves → non_teacher_std shrinks → teacher adv saturates at +5
- disc_temperature=2.5 keeps w_hat at 0.5-0.8 → teacher gradient share is 25-30%
- Policy is being pulled toward teacher behavior at every gradient step with CONSTANT force
- Performance peaks because the "easy" teacher strategies transfer well to the 3B model

**Phase 3 (Steps 80-100): Collapse**
- The 3B model has over-committed to mimicking 72B teacher strategies
- Some teacher strategies require capabilities the 3B model doesn't have (complex reasoning, multi-step planning)
- The policy has concentrated probability mass on teacher-like actions → entropy collapse
- On tasks where teacher strategies don't transfer, the policy fails badly
- On-policy reward starts dropping → but teacher adv stays at +5 (it's clamped, not adaptive)
- **No self-correction**: lower on-policy reward doesn't reduce teacher influence
- Performance spirals downward

### Why LUFFY Doesn't Collapse

LUFFY has inherent self-correction through policy shaping:
- f(ratio) = ratio / (ratio + β): bounded above by 1 (when ratio → ∞, f → 1)
- As policy diverges from teacher, ratio shrinks → f(ratio) ≈ ratio/β → teacher gradient proportional to actual policy-teacher similarity
- When the policy struggles on a task, it naturally falls back to on-policy learning
- LUFFY's teacher influence is **responsive** to the policy state; DUET 0403's is **constant**

### Why 0401 (with gap_gate ON) Didn't Collapse

gap_gate provides a performance-based attenuation:
- gate = f(reward_gap) ∈ [0, 1]
- As on-policy catches teacher, gap → 0 → gate → 0 → teacher advantage scaled down
- This creates a natural curriculum: strong teacher early, fading teacher late
- 0403 removed this to avoid "double suppression" but replaced it with NOTHING

## Hypothesis Rankings

| Hypothesis | Likelihood | Evidence |
|-----------|-----------|---------|
| **B: Constant +5 teacher adv creates fixed gradient bias** | **HIGH** | Math shows teacher_adv saturates at +5 once non_teacher_std < 0.1 (happens by step 40). No adaptive mechanism remains. |
| **D: disc_temperature=2.5 prevents natural DR3 fade-out** | **HIGH** | Temperature 2.5 keeps w_hat at 0.5-0.8 when temperature 1.0 would give 0.01-0.1. The "natural fade-out" is the core DR3 thesis — softening it this much defeats the purpose. |
| **A: KL explosion** | **MEDIUM** | kl_loss_coef=0.001 is weak. Constant teacher pushing could drive KL. But KL explosion would cause rapid divergence, not gradual collapse. Need to check `actor/kl_loss` in wandb. |
| **C: SC post-GRPO scale mismatch** | **LOW** | SC bonus per token ≈ 0.2 * 0.5 / 512 ≈ 0.0002. GRPO advantages are O(1). SC contribution is negligible (~0.02%). Not the cause. |

**Root cause = B + D compound**: Fixed +5 teacher advantage combined with slowed DR3 fade-out creates sustained, unattenuated teacher gradient pressure that eventually causes mode collapse.

## Fix Proposals

### Fix 1: Relative Teacher Advantage Cap (addresses B directly)

**Rationale**: Teacher advantage should scale with the natural reward variance, not be a fixed constant.

```python
# In compute_advantage_teacher_baseline_separation, line 532-533
if is_teacher[i]:
    # Cap teacher adv to p90 of |on-policy advantages| in this group
    nt_advs = [adv[j].item() for j in id2idxs[gid] if not is_teacher[j]]
    if nt_advs:
        cap = max(np.percentile(np.abs(nt_advs), 90), 1.0)
    else:
        cap = 5.0
    adv_i = torch.clamp(adv_i, min=-cap, max=cap)
```

**Expected effect**: Early training (high variance): cap ≈ 2-5, similar to now. Late training (low variance): cap ≈ 0.5-1.0, naturally reducing teacher influence. Self-correcting.

**Complexity**: Low. Drop-in replacement for the hardcoded ±5.

### Fix 2: Restore disc_temperature to 1.5 (addresses D directly)

**Rationale**: disc_temperature=2.5 was chosen for "softer DR3" but it defeats the natural fade-out that is DUET's core mechanism. The discriminator SHOULD become sharp as training progresses — that's what drives curriculum learning.

**Expected effect**: w_hat decays to 0.01-0.1 by step 60-80, naturally suppressing teacher gradient without needing gap_gate.

**Risk**: May re-introduce early instability if discriminator is too aggressive. Mitigate with `apply_warmup_steps: 15` (currently 10).

### Fix 3: Re-enable ONE attenuation mechanism

The 0403 rationale was correct: gap_gate + adaptive_weight is double suppression. But the fix was wrong: removing BOTH leaves zero suppression.

**Option A**: Re-enable gap_gate only, with lower power (0.5 instead of 1.0)
```yaml
dr3:
  gap_gate_enable: true
  gap_gate_power: 0.5  # softer than default
```

**Option B**: Re-enable adaptive_weight only, with higher minimum
```yaml
teacher_experience:
  adaptive_weight:
    enable: true
    min: 0.2  # never fully suppress
    max: 1.0
```

Prefer Option A because gap_gate acts on advantages (closer to the gradient), while adaptive_weight acts on reward mixing (more indirect).

### Fix 4: KL safety net (addresses A)

Increase kl_loss_coef from 0.001 to 0.005. This provides a stronger anchor to the reference model without dominating the learning signal. Low-risk, orthogonal to other fixes.

```yaml
actor:
  kl_loss_coef: 0.005
```

### Fix 5: Entropy regularization (addresses mode collapse symptom)

Set entropy_coeff > 0 to maintain policy diversity:
```yaml
actor:
  entropy_coeff: 0.005
```

This doesn't fix the root cause but prevents the mode collapse endpoint.

## Recommended 0404 Configuration

Apply fixes in this priority order:

1. **Fix 1 (relative teacher adv cap)** — Most principled, addresses root cause
2. **Fix 2 (disc_temperature back to 1.5)** — Restores DR3's core mechanism
3. **Fix 4 (kl_loss_coef: 0.005)** — Safety net, low risk

If 1+2+4 are insufficient, add Fix 3A (gap_gate with power=0.5).

Do NOT apply Fix 5 unless entropy collapse is confirmed in wandb (check `actor/entropy` metric).

## Metrics to Verify This Theory

Before running 0404, check these in 0403 wandb:

| Metric | If theory is correct, expect to see... |
|--------|--------------------------------------|
| `duet/teacher_gradient_share` | Stays at 20-30% (doesn't decay to 5% like in 0401) |
| `dr3/disc_acc` | Reaches 0.9+ but w_hat stays moderate (0.3-0.8) due to temperature |
| `dr3/w_hat_mean` (teacher) | Stays at 0.3-0.8 (not decaying to 0.01 as theory requires) |
| `actor/kl_loss` | Gradually increasing, possibly spiking at step 80-100 |
| `actor/entropy` | Monotonically decreasing, possibly reaching very low values by step 80 |
| On-policy std of rewards | Shrinking over time (confirming teacher_adv clamp activation) |

## NeurIPS Reviewer Angle

A reviewer would ask: "Why not just use a fixed decay schedule for teacher influence instead of this complex DR3 machinery?"

Answer: The 0403 collapse actually demonstrates why data-driven fade-out matters. A fixed schedule can't adapt to training dynamics. The PROBLEM with 0403 is that we accidentally disabled the adaptive mechanism (gap_gate OFF, adaptive_weight OFF, disc_temperature too high). When properly configured, DR3 provides exactly the kind of responsive, data-driven curriculum that a fixed schedule cannot.

The collapse is **not a flaw in DUET's design** — it's a flaw in 0403's parameterization that removed the self-correcting properties DUET is designed to have.
