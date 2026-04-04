# Advantage Explosion Analysis: The Real Bottleneck

**Date**: 2026-04-02 (updated after 0402 wandb data)

## 1. Root Cause Chain

```
WebShop continuous [0,1] rewards
  → On-policy rewards cluster tightly as policy improves (e.g., [0.71, 0.72, 0.69, ...])
  → non_teacher_std collapses (e.g., 0.005)
  → GRPO advantage = (reward - baseline) / std → explodes
  → Teacher advantage ≈ (1.0 - 0.74) / 0.005 = 52   (or -662 in pathological cases)
  → On-policy advantage ≈ (0.72 - 0.71) / 0.005 = 2
  → teacher_gradient_share = |52| / (|52| + |2|) ≈ 96%
  → DR3 w_hat (0.77) cannot compensate: 0.77 × 52 = 40 >> 2
```

### Why ALFWorld is immune

Binary {0,1} rewards with any mix of successes/failures:
- std({0, 0, 0, 1, 0, 1, 0}) ≈ 0.49
- Teacher advantage = (1.0 - 0.4) / 0.49 ≈ 1.2
- On-policy advantage for success = (1.0 - 0.4) / 0.49 ≈ 1.2
- Bounded by nature of binary rewards

### Why LUFFY is immune

LUFFY also uses the same GRPO advantage computation with same std collapse.
But LUFFY's gradient formula is:

```
teacher_gradient ∝ -advantage × (p/(p+β))
```

Where p = π_θ(a_teacher|s) is the **per-token** probability. For most teacher tokens,
p is very small (e.g., 0.001-0.01), so:

```
teacher_gradient ∝ -662 × (0.005 / (0.005 + 0.1)) = -662 × 0.048 = -31.5
```

The token-level `p/(p+β)` acts as an **implicit advantage clamp** because tokens where the
student hasn't learned the teacher's action get near-zero weight regardless of advantage magnitude.

DR3's gradient formula:
```
teacher_gradient ∝ -advantage × clip(ratio, 1-ε, 1+ε)
```

Where ratio = π_new/π_old_corrected ≈ w_hat ≈ 0.77, clipped to [0.8, 1.2]:

```
teacher_gradient ∝ -662 × 0.8 = -529.6
```

**No token-level damping** — DR3 operates at the sample level (one w_hat per trajectory),
so the full advantage magnitude flows through.

## 2. The Safeguard Gap

`ae_ray_trainer.py:499-504`:
```python
if s_for_std.numel() <= 1:
    std = torch.tensor(1.0, device=scores.device)  # ← safeguard for 0 or 1 samples
else:
    std = s_for_std.std()
    if torch.isnan(std).item() or std.item() == 0.0:  # ← only catches exactly 0.0
        std = torch.tensor(1.0, device=scores.device)
```

**No floor for small-but-nonzero std.** A std of 0.001 passes through and creates advantages of ~300.

## 3. Evaluation of Proposed Fixes

### A. Clip teacher advantages to ±C (e.g., C=5)

**Mechanism**: After GRPO advantage computation, clamp teacher-sample advantages.

**Theory assessment**: ✅ **Recommended (primary fix)**

Truncated importance sampling is well-established in off-policy RL:
- Ionides (2008) and subsequent work show that clipping high-magnitude gradient terms
  reduces variance with controllable bias
- C=5 means we accept teacher advantages up to 5σ (in the "ideal" normalization).
  Beyond 5σ, the signal is almost certainly an artifact of std collapse, not genuine signal.
- On ALFWorld, teacher advantages are naturally ~1-2, so C=5 never activates → zero impact on existing results.

**Implementation** (`ae_ray_trainer.py:524-527`):
```python
adv_i = scores[i] - base
if norm_adv_by_std_in_grpo:
    adv_i = adv_i / (id2_std[gid] + epsilon)
if is_teacher[i]:
    adv_i = torch.clamp(adv_i, -C, C)  # ← add this
adv[i] = adv_i
```

**Pros**: Simple, targeted (teacher only), preserves on-policy learning, robust
**Cons**: Hyperparameter C needs choosing (but C=5 is well-motivated, and anything in [3,10] should work)

### B. Teacher-specific advantage normalization (separate std)

**Mechanism**: Normalize teacher advantages by teacher's own std.

**Theory assessment**: ⚠️ **Problematic**

- With n_teacher=1 per group, teacher std within a group is undefined (single sample).
- Would need cross-group teacher std pooling or running EMA → adds complexity and statefulness.
- Cross-group pooling mixes teacher rewards across different tasks, which may not be meaningful
  (teacher could score 1.0 on easy tasks and 0.6 on hard tasks).
- Not clear what the "right" teacher normalization constant is.

**Verdict**: Over-engineered for the problem. Skip.

### C. w_hat / max(1, |adv|) — adaptive weight

**Mechanism**: Scale DR3 density ratio inversely with advantage magnitude.

**Theory assessment**: ❌ **Reject**

- Conflates two theoretically distinct quantities: density ratio correction and advantage estimation.
- DR3's w_hat estimates p_θ/(αp_θ + (1-α)q_teacher) — it has a clear statistical meaning as a
  density ratio. Dividing by |adv| destroys this interpretation.
- A reviewer would rightly ask: "What estimator is w_hat/|adv|? What does it converge to?"
- There's no theoretical guarantee on the resulting gradient estimator.

**Verdict**: Ad-hoc, theoretically unsound. Skip.

### D. Change std_source from "non_teacher" to "all" (my addition)

**Mechanism**: Include teacher rewards in the std computation.

**Theory assessment**: ⚠️ **Partially helpful but insufficient**

With 7 on-policy at ~0.71 and 1 teacher at 1.0:
- std_all ≈ 0.10 (vs std_nt ≈ 0.013)
- Teacher advantage: 2.5 (vs 19)

But when on-policy approaches teacher level (both ~0.9):
- std_all also collapses → same problem returns
- And it reduces ALL advantages (teacher and on-policy) by the same factor,
  which could slow on-policy learning (this may be what caused 0401's failure)

**Verdict**: Helps early/mid training but doesn't solve the asymptotic case. Also a
global change that affects on-policy — risky.

### E. Std floor (revisiting 0401's approach with better calibration)

**Mechanism**: `std = max(s_for_std.std(), std_floor)` with std_floor configurable.

**Theory assessment**: ⚠️ **Viable but fragile**

The 0401 failure was "std floor killed learning." Likely because:
1. The floor was set too high (e.g., 0.5), making ALL advantages tiny (even on-policy)
2. The floor was applied uniformly, not teacher-specific

A std_floor of 0.05-0.1 would:
- Cap teacher advantage at (1.0 - 0.75) / 0.05 = 5.0 (equivalent to clip C=5!)
- Also cap on-policy advantages at (0.73 - 0.71) / 0.05 = 0.4 (probably too small)

**Verdict**: Same problem as D — affects on-policy learning. Teacher advantage
clipping (option A) is strictly better because it's targeted.

## 4. Recommendation: Layered Fix

### Layer 1 (immediate): Teacher advantage clipping

```python
# In compute_grpo_outcome_advantage_with_teacher_separation()
# ae_ray_trainer.py, line ~526-527
adv_i = scores[i] - base
if norm_adv_by_std_in_grpo:
    adv_i = adv_i / (id2_std[gid] + epsilon)
if is_teacher[i]:
    adv_i = torch.clamp(adv_i, min=-teacher_adv_clip, max=teacher_adv_clip)
adv[i] = adv_i
```

Config:
```yaml
algorithm:
  grpo:
    teacher_baseline_separation:
      enable: true
      teacher_baseline: all_mean
      non_teacher_baseline: non_teacher_mean
      std_source: non_teacher
      teacher_adv_clip: 5.0  # ← NEW
```

### Layer 2 (from earlier analysis): Disable gap_gate

```yaml
dr3:
  gap_gate_enable: false
adaptive_weight:
  enable: false
```

### Layer 3 (keep): disc_temperature=2.5

Already validated to help.

### Expected Combined Effect

| Component | teacher_gradient_share before | after |
|-----------|------------------------------|-------|
| Advantage clip (C=5) | 99.96% (|adv|=662) | ~70% (|adv|=5) |
| + gap_gate OFF | 70% × 0.6 gate | 70% (no gate) |
| + DR3 w_hat (T=2.5) | w=0.775 | w=0.775 |
| **Net teacher_grad_share** | **99.96%** | **~55-65%** |
| LUFFY reference | — | **40-60%** |

This puts DUET's teacher_gradient_share in the same healthy range as LUFFY.

## 5. Theoretical Justification for C=5

The GRPO advantage under proper normalization (when std is healthy) has distribution approximately:
- On-policy: mean 0, std 1 (by construction of group normalization)
- Teacher: mean = gap/std ≈ (R_teacher - μ)/σ, typically 1-3 when std is healthy

Setting C=5 means:
- We accept teacher advantages up to 5σ above or below the baseline
- Any advantage |A| > 5 corresponds to std < gap/5 ≈ 0.05 (for gap=0.25), indicating
  std has collapsed below meaningful levels
- On ALFWorld: teacher advantages are naturally ~1-2 (binary rewards, large std) → clip never fires
- On WebShop healthy phase: teacher advantages are ~2-3 → clip rarely fires
- On WebShop collapsed phase: teacher advantages are ~50-600 → clip activates, prevents explosion

This is equivalent to saying: "trust the advantage normalization when std is healthy, but
override it when std collapse produces degenerate values."

## 6. Why This Fix Is Better Than LUFFY's Implicit Solution

LUFFY's p/(p+β) implicitly clips the effective advantage via token-level damping. But:
- It clips based on student probability (low p → low weight), NOT advantage magnitude
- This means even reasonable-magnitude advantages on low-probability tokens get suppressed
- It cannot distinguish "the student should learn this token" from "the student will never learn this"

DR3 + explicit advantage clipping:
- Preserves advantage information (direction and relative magnitude within the clip range)
- Applies correction at the trajectory level (density ratio) rather than per-token
- Only activates when std actually collapses (clean theoretical trigger)

For the paper: "DR3 with advantage clipping provides more principled teacher weighting than
LUFFY's policy shaping: the density ratio w_hat corrects for distributional shift, while the
advantage clip prevents degenerate normalization — each addressing a distinct failure mode."
