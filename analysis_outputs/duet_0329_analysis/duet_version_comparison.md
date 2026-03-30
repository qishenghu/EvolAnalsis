# DUET(0329) vs DUET(0327): Version Comparison Analysis

**Analysis date**: 2026-03-30
**Intermediate run**: DUET(0328) also examined where data available

---

## 1. What Changed: Git Commits Between Runs

7 commits between DUET(0327) launch (2026-03-27 13:06) and DUET(0329) launch (2026-03-29 23:21):

| # | Commit | Type | Summary | Files Changed |
|---|--------|------|---------|:------------:|
| 1 | `6f5db62b` | **Bug fix + metrics** | Fix SC token distribution, beta decay normalization, step delta ordering; add 30+ wandb metrics | 3 (285 ins) |
| 2 | `5f837934` | **Bug fix** | Remove verbose logging that crashed training at step 0 | 1 (5 ins) |
| 3 | `96e5fdd2` | **Hyperparameter tune** | SC beta 0.5→0.2, DR3 temp 1.2→2.5, KL 0.001→0.005, enable step deltas | 3 (19 ins) |
| 4 | `e78d0dea` | **Bug fix** | Slice step_ids to response length (shape mismatch crash) | 1 (4 ins) |
| 5 | `fec57040` | **Bug fix** | Move non-tensor data to non_tensor_batch, cleanup temp keys before FSDP | 1 (7 ins) |
| 6 | `6f01b5f9` | **Param refinement** | DR3 temp 2.5→1.5, w_min 0.1→0.01 (embrace natural fade-out) | 3 (4 ins) |
| 7 | `3ecfcb0e` | **Feature** | Exclude teacher samples from SC bonus injection | 3 (45 ins) |

### Change Categories
- **4 bug fixes** (commits 1, 2, 4, 5): Critical stability and correctness issues
- **2 hyperparameter changes** (commits 3, 6): SC dampening + DR3 refinement
- **1 algorithmic change** (commit 7): Teacher SC exclusion — the most impactful design decision

---

## 2. Config Diff Summary

| Parameter | DUET(0327) | DUET(0328) | DUET(0329) | Rationale |
|-----------|:----------:|:----------:|:----------:|-----------|
| `dr3.clip_max` | 10.0 | **5.0** | **5.0** | Prevent extreme density ratios |
| `dr3.disc_label_smoothing` | 0.05 | **0.1** | **0.1** | Soften discriminator confidence |
| `dr3.disc_temperature` | 1.2 | **2.5** | **1.5** | 0328 over-corrected; 0329 moderate |
| `actor.kl_loss_coef` | 0.001 | **0.005** | **0.005** | Reduce policy drift |
| `state_channel.beta` | 0.5 | **0.2** | **0.2** | SC bonus was 30-50% of task reward |
| `state_channel.beta_decay_target` | 0.5 | **0.3** | **0.3** | Ensure beta actually decays |
| `state_channel.exclude_teacher` | *(absent)* | *(absent)* | **true** | Key 0329 change |
| `state_channel.step_level.enable` | false | **true** | **true** | Finer credit assignment |
| `state_channel.step_level.eta` | 0.1 | **0.05** | **0.05** | Conservative step deltas |

**The 0328→0329 diff is minimal**: only `disc_temperature` (2.5→1.5) and `exclude_teacher` (new). This isolates the impact of teacher SC exclusion.

---

## 3. Validation Results

| Version | Step 50 | Step 100 | Delta 50→100 |
|---------|:-------:|:--------:|:------------:|
| DUET(0327) | 53.0% (106/200) | 66.0% (132/200) | +13.0pp |
| DUET(0328) | N/A | N/A | (run appears incomplete) |
| **DUET(0329)** | 48.0% (96/200) | **69.5%** (139/200) | **+21.5pp** |

**Key finding**: DUET(0329) starts slower at step 50 (-5pp vs 0327) but accelerates much faster, surpassing 0327 by +3.5pp at step 100.

---

## 4. Training Dynamics Comparison

### On-Policy Reward Trajectory
| Step | DUET(0327) | DUET(0329) | Gap |
|:----:|:----------:|:----------:|:---:|
| 1 | 0.525 | 0.298 | -0.227 |
| 10 | 0.649 | 0.491 | -0.158 |
| 30 | 0.710 | 0.556 | -0.154 |
| 50 | 0.848 | 0.704 | -0.144 |
| 70 | 0.898 | 0.649 | -0.249 |
| 80 | 0.650 | 0.640 | -0.010 |
| 100 | **0.939** | **0.939** | **0.000** |

**0327 learns faster in-batch but both converge to identical 0.939 on-policy reward at step 100.** The early 0327 advantage comes from higher SC beta (0.5 vs 0.2) providing stronger dense reward — but this advantage doesn't translate to better validation.

### Teacher Gap Closure (most revealing metric)
| Step | DUET(0327) gap | DUET(0329) gap |
|:----:|:--------------:|:--------------:|
| 1 | 0.718 | 0.653 |
| 50 | 0.492 | 0.296 |
| 100 | **0.394** | **0.061** |

**Critical finding**: DUET(0327)'s teacher gap stalls at 0.39 while DUET(0329) closes to 0.06. The 0327 gap cannot fully close because teacher rewards are inflated by SC bonus (~1.33 vs natural 1.0).

---

## 5. Root Cause: Teacher SC Bonus Inflation

### The Bug in DUET(0327)
In DUET(0327), SC bonus was applied to ALL samples including teacher:

| Step | Teacher SC Bonus | Teacher reward_sum | On-policy SC Bonus |
|:----:|:----------------:|:------------------:|:------------------:|
| 1 | +0.329 | **1.329** | +0.192 |
| 50 | +0.340 | **1.340** | +0.223 |
| 100 | +0.333 | **1.333** | +0.278 |

Teacher samples received **even more SC bonus than on-policy samples** because teachers traverse more expert states (high progress ~0.85 vs on-policy ~0.35-0.58).

### Impact on GRPO Advantages
| Step | 0327 Teacher Adv | 0329 Teacher Adv | 0327 range | 0329 range |
|:----:|:-----------------:|:-----------------:|:----------:|:----------:|
| 1 | +0.910 | +2.751 | [+0.04, +5.73] | [+0.05, +16.98] |
| 50 | +0.329 | **-0.216** | [+0.07, +1.19] | [-1.69, +0.17] |
| 100 | **+0.103** | **-0.207** | [-0.15, +0.40] | [-0.96, +0.09] |

**In DUET(0327)**: Teacher advantages remain positive throughout training (even at step 100: +0.10), continuously pulling the policy toward teacher behavior.

**In DUET(0329)**: Teacher advantages go negative by step 50 (-0.22), meaning the agent's own successful trajectories now generate higher GRPO advantages than teacher demonstrations. This is exactly the DR3 fade-out behavior the design intended.

### Why 0327 Teacher Advantages Stay Positive
GRPO normalizes by group: `adv = (reward - mean) / std`. With teacher rewards inflated to ~1.33 and on-policy successes at ~1.0, teachers always appear "above average" even when the agent performs well. The SC bonus effectively locks teacher influence on — **fighting DR3's natural fade-out**.

---

## 6. Response Efficiency Comparison

| Metric | DUET(0327) step 100 | DUET(0329) step 100 |
|--------|:-------------------:|:-------------------:|
| On-policy success | 66.1% (37/56) | **82.1% (46/56)** |
| Mean response tokens | 4,778 | **3,155** (-34%) |
| Median response tokens | 2,724 | 2,810 |
| P90 response tokens | 12,827 | **6,325** (-51%) |
| Mean msg count | 36.3 | **31.6** (-13%) |

DUET(0329) is notably more efficient at step 100 — higher success rate with shorter responses. The P90 reduction from 12.8K to 6.3K tokens is particularly significant: fewer long, wandering trajectories.

---

## 7. SC Behavior Comparison

| Step | 0327 On-Policy SC | 0329 On-Policy SC | 0327 SC Progress | 0329 SC Progress |
|:----:|:-----------------:|:-----------------:|:----------------:|:----------------:|
| 1 | 0.192 | 0.070 | 0.384 | 0.350 |
| 50 | 0.223 | 0.097 | 0.447 | 0.486 |
| 100 | 0.278 | 0.117 | 0.556 | 0.584 |

- **SC bonus magnitude**: 0329 is ~2.5x smaller due to beta 0.2 vs 0.5
- **SC progress**: Nearly identical (slightly higher in 0329 at step 100), confirming the progress computation itself is consistent
- **SC as fraction of reward**: 0327 SC bonus was ~30-50% of task reward; 0329 is ~12-15% (much more proportionate)

---

## 8. Intermediate Run: DUET(0328)

DUET(0328) has trajectory data for steps 1-40 only (run appears incomplete — may have crashed or been terminated). It shares all config changes from commit `96e5fdd2` but lacks the teacher SC exclusion and the disc_temperature correction.

| Step | 0327 | 0328 | 0329 |
|:----:|:----:|:----:|:----:|
| 1 | 0.525 | 0.355 | 0.298 |
| 10 | 0.649 | 0.539 | 0.491 |
| 20 | 0.701 | 0.580 | 0.594 |
| 30 | 0.710 | 0.540 | 0.556 |
| 40 | 0.763 | 0.614 | 0.501 |

0328 and 0329 show similar early-training patterns (both slower than 0327 due to reduced SC beta). 0328's incomplete run prevents full comparison.

---

## 9. Summary: What Changed and Did It Help?

### Changes Ranked by Impact

| Rank | Change | Impact |
|------|--------|--------|
| **1** | **SC teacher exclusion** (`exclude_teacher: true`) | **Highest**. Fixed teacher reward inflation (+0.33), enabled DR3 fade-out to work as designed. Teacher advantages properly go negative by step 50. |
| **2** | **SC beta reduction** (0.5→0.2) | **High**. Reduced SC from 30-50% to 12-15% of task reward. Prevents SC from dominating the reward signal. |
| **3** | **Bug fixes** (step_ids slice, non-tensor batch, logging crash) | **Medium**. Enabled step-level deltas to actually work. Prevented FSDP serialization errors. |
| **4** | **DR3 parameter tuning** (temp, clip_max, label_smoothing) | **Medium**. Moderate smoothing (temp 1.5) balances between 0327's aggressive (1.2) and 0328's over-smooth (2.5). |
| **5** | **KL coef increase** (0.001→0.005) | **Low-Medium**. Better policy stability, but entropy levels are similar between versions. |
| **6** | **Step-level deltas** (enabled, eta=0.05) | **Low**. Provides finer credit assignment but requires bug fix (commit 4) to work at all. |

### Bottom Line

| Metric | DUET(0327) | DUET(0329) | Verdict |
|--------|:----------:|:----------:|---------|
| Validation step 100 | 66.0% | **69.5%** | +3.5pp improvement |
| Teacher gap closure | 0.394 (stalled) | **0.061** (closed) | 6.5x better convergence |
| Response efficiency | 4,778 tokens | **3,155 tokens** | 34% shorter |
| Training batch success | 66.1% | **82.1%** | +16pp improvement |
| Theoretical correctness | SC fights DR3 | **SC + DR3 aligned** | Design invariants now hold |

**DUET(0329) is the superior configuration.** The primary driver is the SC teacher exclusion fix, which aligns the two channels instead of having them fight each other. The hyperparameter changes (SC beta, DR3 params) amplify this by keeping SC proportionate and DR3 well-calibrated.

The +3.5pp validation gain understates the improvement — the training dynamics are qualitatively different. In DUET(0327), teacher influence is permanently locked on by SC bonus inflation. In DUET(0329), DR3 provides a proper closed-form curriculum: high teacher influence early (advantage +2.75), natural fade-out mid-training (advantage → 0), and eventual hand-off (advantage negative by step 50).
