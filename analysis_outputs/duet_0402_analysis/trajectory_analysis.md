# DUET 0402 Analysis: Trajectory Comparison vs LUFFY

**Date**: 2026-04-02
**Runs**: webshop_3b_duet_0402 (T=2.5), webshop_3b_duet (T=1.5), webshop_3b_luffy
**Key finding**: SC bonus inflates on-policy rewards above teacher reward, creating catastrophically negative teacher advantages via GRPO normalization.

---

## 1. Validation Performance Summary

| Method | Mean Reward (step 100) | Perfect (r=1.0) | High (r>0.5) | Zero (r=0) |
|--------|----------------------|-----------------|-------------|-----------|
| **LUFFY** | 0.7528 | 99 (49.5%) | 55 (27.5%) | 12 (6.0%) |
| **DUET 0402** | 0.7353 | 71 (35.5%) | 83 (41.5%) | 12 (6.0%) |
| Gap | -0.0175 | -28 | +28 | 0 |

Per-sample comparison (n=200, same eval set):
- LUFFY better: 44 samples (22.0%)
- DUET better: 29 samples (14.5%)
- Same (+-0.01): 127 samples (63.5%)
- LUFFY perfect but DUET not: **33 samples** (DUET mean=0.751 on these)
- DUET perfect but LUFFY not: 5 samples

**Interpretation**: The gap is concentrated in LUFFY getting perfect scores where DUET gets high-partial. DUET 0402 is converting fewer "almost perfect" into "perfect" — likely from losing teacher guidance in late training.

---

## 2. ROOT CAUSE: SC Bonus Creates Negative Teacher Advantages

### The mechanism

1. SC bonus pushes successful on-policy `reward_sum` above teacher's 1.0:
   - `reward_sum = reward_original + sc_bonus`
   - For a successful sample: `1.0 + 0.2*progress ≈ 1.0 + 0.08 = 1.08`

2. GRPO teacher_baseline_separation computes teacher advantage as:
   ```
   teacher_adv = (teacher_reward - all_mean) / non_teacher_std
   ```

3. When most on-policy samples in a group succeed (reward_sum ≈ 1.08):
   - `all_mean ≈ (1.08*7 + 1.0*1)/8 = 1.07`
   - `teacher_adv_raw = 1.0 - 1.07 = -0.07` (NEGATIVE!)
   - `non_teacher_std ≈ 1e-7` (all on-policy have same reward)
   - **`teacher_adv_norm = -0.07 / 1e-7 = -700,000`**

4. The std floor in the code only catches `std == 0.0` exactly, not near-zero values.

### Direct evidence (step 90, 4 of 7 teacher groups are negative):

| Group | On-policy rewards | All mean | Non-teacher std | Teacher adv normalized |
|-------|------------------|----------|-----------------|----------------------|
| 2 | [1.08]*7 | 1.070 | 1.0e-7 | **-63,791** |
| 4 | [1.085]*7 | 1.074 | 1.0e-7 | **-67,778** |
| 5 | [1.085]*7 | 1.074 | 1.5e-7 | **-64,900** |
| 7 | [1.085]*7 | 1.074 | 1.0e-7 | **-67,778** |

These astronomical negative advantages mean **the policy is being trained to do the OPPOSITE of teacher actions**.

### Progression over training:

| Step | On-policy reward_sum > 1.0 | Negative teacher adv groups | Batch adv_teacher_sample_mean |
|------|---------------------------|---------------------------|------------------------------|
| 50 | 3/56 (5%) | 0 | 0.14 |
| 75 | 19/59 (32%) | 4/5 (80%) | 3.32 |
| 90 | 34/57 (60%) | 4/7 (57%) | **-3,450** |
| 100 | 10/58 (17%) | 1/6 (17%) | 1,803 |

As the policy improves and more on-policy samples succeed, the SC bonus pushes more groups into negative-teacher-advantage territory. This creates a destructive feedback loop:
- Policy improves → more successes → SC inflates rewards → teacher penalized → policy loses teacher guidance → performance plateaus/drops

### Why LUFFY and DUET orig don't have this problem:
- **LUFFY**: No SC bonus. On-policy rewards never exceed 1.0.
- **DUET orig**: SC was NOT enabled (sc_bonus=0 at all steps).

---

## 3. Teacher Advantage Oscillation (all runs)

All three runs show GRPO teacher advantage oscillation (huge spikes). This is a universal GRPO issue from per-group normalization with small std, NOT specific to DR3:

| Run | Steps with |teacher_adv| > 100 | Negative spikes |
|-----|---------------------------------------|-----------------|
| DUET 0402 | 55/100 | 10 (from SC bug) |
| DUET orig | 33/100 | 0 |
| LUFFY | 47/100 | 0 |

**0402 phases**:
- Steps 1-8: Normal (mean 0.7)
- Steps 9-50: Huge positive spikes (mean 5,904) — DR3 warming up
- Steps 51-64: Calm (mean 198) — DR3 stabilized
- Steps 65-100: Wild oscillation WITH negatives (mean 641, 9 negative) — SC bug kicks in

---

## 4. Teacher vs On-Policy Advantage Magnitude

The fundamental difference in teacher gradient signal:

| Step | DUET 0402 teacher_adv_token | LUFFY teacher_adv_token | Ratio (LUFFY/0402) |
|------|---------------------------|------------------------|-------------------|
| 1 | 6.0 | 5.4 | 0.9x |
| 10 | 7.8 | 4.3 | 0.6x |
| 50 | 1.6 | 90,450 | 58,000x |
| 75 | 38.6 | 3.6 | 0.1x |
| 100 | 33,424 | 2.95 | 0.0001x |

The teacher advantage alternates between near-zero (DR3 suppression) and enormous values (GRPO normalization spikes). This causes unstable, whiplash-like gradient dynamics.

LUFFY maintains more consistent teacher advantages through the policy shaping mechanism (pi_theta^10).

---

## 5. On-Policy Training Reward Trajectory

| Step | DUET 0402 | DUET orig | LUFFY |
|------|-----------|-----------|-------|
| 1 | 0.228 | 0.251 | 0.157 |
| 25 | 0.435 | 0.553 | 0.434 |
| 50 | 0.736 | 0.740 | 0.647 |
| 75 | 0.899 | 0.636 | 0.846 |
| 100 | 0.721 | 0.776 | 0.790 |

0402 and LUFFY track similarly until step 75, then 0402 drops back. This coincides with when SC-induced negative teacher advantages become prevalent.

---

## 6. State Channel Bonus Analysis (0402 only)

| Step | sc_bonus (mean) | sc_progress | reward_orig | reward_total | bonus/reward |
|------|----------------|------------|-------------|-------------|-------------|
| 1 | 0.042 | 0.212 | 0.185 | 0.228 | 22.9% |
| 10 | 0.039 | 0.197 | 0.134 | 0.173 | 29.5% |
| 50 | 0.073 | 0.363 | 0.663 | 0.736 | 11.0% |
| 75 | 0.049 | 0.246 | 0.850 | 0.899 | 5.8% |
| 100 | 0.058 | 0.289 | 0.663 | 0.721 | 8.7% |

SC bonus is modest (~5-30% of reward) but sufficient to push successful samples above 1.0.

---

## 7. Proxy Teacher Gradient Share

| Step | DUET 0402 | DUET orig | LUFFY |
|------|-----------|-----------|-------|
| 1 | 79.2% | 73.8% | 87.3% |
| 10 | 72.3% | 75.9% | 60.1% |
| 50 | 63.1% | 75.8% | 100% |
| 75 | 99.5% | 30.4% | 79.6% |
| 100 | 100% | 63.6% | 95.1% |

All three runs oscillate significantly. DUET orig shows the expected DR3 fade-out (75.8% → 30.4% at step 75), but 0402 doesn't fade properly due to the SC interaction.

---

## 8. Gap Gate Interaction

Config: `gap_gate_enable: true`, `adaptive_weight.mode: gap_linear`, `tau: 0.5`

The gap_gate scales teacher advantages by `gate = reward_gap / tau`:
- Step 50: gap = 0.264, gate = 0.264/0.5 = 0.53
- Step 75: gap = 0.101, gate = 0.101/0.5 = 0.20
- Step 100: gap = 0.279, gate = 0.279/0.5 = 0.56

Gap gate IS compounding with DR3, reducing teacher influence by 44-80%. However, this is a SECONDARY issue — the SC-induced negative advantage bug is far more damaging.

---

## 9. Diagnosis Summary

**Primary bug**: SC bonus + GRPO teacher_baseline_separation interaction
- SC bonus pushes on-policy reward_sum above teacher's 1.0
- GRPO group normalization creates catastrophically negative teacher advantages
- Near-zero std floor (only checks exact 0.0) amplifies to ±700,000
- **Impact**: Policy actively trained to diverge from teacher in late training

**Secondary issues**:
1. Gap gate compounds teacher fade-out on top of DR3 (40-80% additional reduction)
2. Teacher advantage oscillation from GRPO normalization (affects all runs)
3. DR3 w_hat instability in early training (temperature=2.5 didn't fully fix)

---

## 10. Proposed Fix for 0403

### Primary fix (MUST DO): Fix GRPO std floor + decouple SC from teacher baseline

**Change 1**: In `compute_grpo_outcome_advantage_teacher_baseline_separated()` at `ae_ray_trainer.py:499-504`, add a configurable std floor:
```python
# Current:
if torch.isnan(std).item() or std.item() == 0.0:
    std = torch.tensor(1.0, device=scores.device)

# Fix:
std_min = 0.1  # new config param: algorithm.grpo.std_min
std = std.clamp(min=std_min)
```

**Change 2**: Cap scores at `max_teacher_score` for teacher-containing groups when computing group statistics, OR exclude SC bonus from the scores used for advantage computation (use `reward_original` instead of `reward_sum`).

Recommended implementation: Store `reward_original` separately in the batch and use it for GRPO normalization. SC bonus still contributes through step-level deltas in token_level_rewards.

### Secondary fix (RECOMMENDED): Disable gap gate

```yaml
# config change:
dr3:
  gap_gate_enable: false  # was: true
```

Gap gate is redundant with DR3's natural fade-out and compounds the teacher signal reduction.

### Config for 0403 experiment:

```yaml
algorithm:
  grpo:
    teacher_baseline_separation:
      enable: true
      teacher_baseline: all_mean
      non_teacher_baseline: non_teacher_mean
      std_source: non_teacher
      std_min: 0.1  # NEW: prevent near-zero std explosion
    use_reward_original_for_baseline: true  # NEW: exclude SC bonus from GRPO baseline
actor_rollout_ref:
  actor:
    dr3:
      gap_gate_enable: false  # disable compounding gate
      disc_temperature: 2.5   # keep from 0402
```

### Expected impact:
- Eliminates negative teacher advantages (root cause)
- Prevents catastrophic std explosion (safety net)
- Removes gap_gate compounding (cleaner DR3 signal)
- Should close the 0.0175 gap with LUFFY and potentially exceed it (DUET has both channels while LUFFY only has action channel)
