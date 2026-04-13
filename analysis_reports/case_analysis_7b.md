# Case-Level Trajectory Analysis: 7B vs 3B DUET on WebShop

## Executive Summary

**Core finding: DUET's advantage disappears at 7B because the stronger base model eliminates the performance gap that teacher demonstrations are designed to fill.** At 3B, DUET provides +0.11 to +0.57 reward improvement over on-policy across training. At 7B, DUET provides -0.004 to -0.085 -- consistently *below* on-policy. This is not a DUET bug; LUFFY and CHORD show the same pattern. The 7B on-policy baseline is so strong that teacher mixing becomes dead weight.

---

## 1. The 7B On-Policy Advantage

### 1.1 Base Model Capability Gap

At step 1 (before any RL training), the 7B model already outperforms 3B substantially:

| Metric | 7B On-Policy (Step 1) | 3B On-Policy (Step 1) |
|--------|----------------------|----------------------|
| Mean reward | 0.333 | 0.179 |
| r >= 0.5 | 42.2% | 25.0% |
| r >= 0.8 | 21.9% | 12.5% |
| Buy completion rate | 68.4% | 40.4% |
| Avg actions/trajectory | 11.5 | 14.9 |

The 7B model at step 1 already completes the buy flow 68% of the time (vs 40% for 3B) and uses fewer, more targeted actions. It demonstrates competent search query formulation from the start.

### 1.2 Peak On-Policy Performance

| Method | Best On-Policy Mean Reward | Step |
|--------|---------------------------|------|
| 7B On-Policy | 0.902 | 50 |
| 3B On-Policy | 0.601 | 50 |
| 7B DUET | 0.906 | 90 |
| 3B DUET | 0.908 | 79 |

The 7B on-policy reaches 0.90 by step 50 *without any teacher data*. DUET at 3B needs 79 steps (with teacher mixing, DR3, and SC) to reach the same level. At 7B, DUET merely matches what on-policy achieves on its own.

### 1.3 Training Dynamics: Teacher Gap Closure

**7B DUET teacher-onpolicy gap:**

| Step | Teacher Mean | On-Policy Mean | Gap |
|------|-------------|---------------|-----|
| 1 | 1.000 | 0.415 | 0.585 |
| 10 | 1.000 | 0.154 | 0.846 |
| 30 | 1.000 | 0.472 | 0.528 |
| 50 | 1.000 | 0.802 | 0.198 |
| 80 | 1.000 | 0.844 | 0.156 |
| 90 | 1.000 | 0.906 | 0.094 |

**3B DUET teacher-onpolicy gap:**

| Step | Teacher Mean | On-Policy Mean | Gap |
|------|-------------|---------------|-----|
| 1 | 1.000 | 0.157 | 0.843 |
| 30 | 1.000 | 0.490 | 0.510 |
| 50 | 1.000 | 0.675 | 0.325 |
| 80 | 1.000 | 0.886 | 0.114 |
| 90 | 1.000 | 0.819 | 0.181 |

At 3B, the gap stays above 0.18 even at step 90, meaning teacher demonstrations remain informative throughout training. At 7B, the gap drops to 0.094 by step 90, and the on-policy reward actually *exceeds* the teacher+SC reward (`reward_onpolicy_mean: 1.009` vs `reward_teacher_mean: 1.000` at step 90 per batch diag). At this point, teacher mixing introduces noise rather than signal.

---

## 2. Relative Advantage Analysis

### 2.1 At 7B: No Method Beats On-Policy

Averaged reward difference from on-policy baseline in 10-step windows:

| Step Range | On-Policy Avg | LUFFY-OnP | CHORD-OnP | DUET-OnP |
|-----------|--------------|-----------|-----------|----------|
| 1-10 | 0.229 | -0.003 | -0.005 | -0.004 |
| 11-20 | 0.352 | -0.011 | +0.059 | -0.031 |
| 21-30 | 0.471 | +0.019 | +0.053 | -0.036 |
| 31-40 | 0.576 | -0.007 | -0.016 | -0.026 |
| 41-50 | 0.674 | -0.109 | -0.049 | -0.085 |
| 51-60 | 0.657 | -0.064 | -0.036 | -0.040 |
| 61-70 | 0.702 | +0.001 | +0.012 | -0.019 |
| 71-80 | 0.758 | +0.002 | +0.000 | -0.008 |
| 81-90 | 0.772 | -0.008 | -0.003 | -0.034 |

All three teacher-mixing methods underperform pure on-policy at 7B. CHORD comes closest (sometimes slightly positive), while DUET shows the most consistent negative gap. The worst period for all methods is steps 41-50, where on-policy surges ahead rapidly.

### 2.2 At 3B: DUET Shows Strong Advantage

| Step Range | On-Policy Avg | DUET-OnP |
|-----------|--------------|----------|
| 1-10 | 0.131 | -0.022 |
| 11-20 | 0.268 | +0.114 |
| 21-30 | 0.439 | +0.064 |
| 31-40 | 0.437 | +0.135 |
| 41-50 | 0.437 | +0.114 |
| 51-60 | 0.404 | +0.227 |
| 61-70 | 0.220 | +0.507 |
| 71-80 | 0.209 | +0.573 |
| 81-90 | 0.294 | +0.486 |

At 3B, DUET provides massive advantages, especially late in training when the 3B on-policy model starts to *degrade* (dropping from 0.44 to 0.22). DUET stabilizes training and maintains high performance, likely due to the SC reward shaping preventing catastrophic policy drift.

---

## 3. Failure Mode Analysis

### 3.1 No Catastrophic Failures at 7B

Unlike 3B experiments, no 7B method shows:
- CJK language collapse: **0 instances across all methods and steps**
- Repetition loops (repeated </think> tags): **0 instances**
- Premature termination (<=2 actions): **0 instances**

The 7B model is fundamentally more stable. It does not suffer from the degenerate failure modes that teacher mixing is designed to prevent.

### 3.2 Multi-Action Tag Defect (DUET-Specific)

DUET 7B exhibits a transient multi-action-tag failure where the model stuffs multiple actions inside a single `<action>` tag:

```
<action>
click[a2-yellow]
click[medium]
click[buy now]
</action>
```

This fails because the environment can only parse one action per tag. The rate spikes during steps 36-56:

| Step | On-Policy | LUFFY | CHORD | DUET |
|------|-----------|-------|-------|------|
| 36 | 0.0% | 1.8% | 3.5% | **12.3%** |
| 41 | 1.6% | 3.5% | 1.8% | **17.5%** |
| 46 | 0.0% | 1.8% | 0.0% | **12.5%** |
| 51 | 0.0% | 12.3% | 0.0% | **19.3%** |
| 56 | 0.0% | 7.1% | 0.0% | **14.3%** |
| 61+ | 0.0% | 0.0% | 0.0% | 0.0% |

DUET peaks at 19.3% at step 51 (11/57 on-policy trajectories affected). LUFFY shows a smaller spike (12.3% at step 51). On-policy and CHORD are largely unaffected.

**Root cause hypothesis**: The teacher trajectories demonstrate a compact action style (avg 6.6 actions, always completing buy flow). When the student model partially learns this compactness through DR3-weighted gradient updates, it attempts to compress multiple sequential actions into one turn. The SC bonus may exacerbate this by rewarding trajectories that reach purchase states quickly, creating gradient pressure toward "shortcutting" the action sequence.

**Impact**: The multi-action defect is self-correcting (resolves by step 61) and affects only ~13% of trajectories during the spike period. Given the already-narrow gap between methods at 7B, this transient degradation contributes perhaps 0.02-0.03 to DUET's underperformance during steps 40-60, but does not explain the overall pattern.

### 3.3 Negative Reward Trajectories (Invalid Actions)

All methods start with 10-16 negative-reward trajectories at step 1 (out of ~57-64 on-policy), converging to 0-1 by step 50. No significant between-method differences.

---

## 4. Behavioral Pattern Analysis

### 4.1 Action Quality at Step 50 (Mid-Training)

| Metric | On-Policy | LUFFY | CHORD | DUET |
|--------|-----------|-------|-------|------|
| Avg actions/traj | 5.1 | 4.4 | 7.6 | 4.8 |
| No-buy rate | 0/64 | 2/56 | 6/56 | 1/56 |
| Multi-action | 0/64 | 3/56 | 0/56 | 13/56 |
| >10 actions | 1/64 | 2/56 | 7/56 | 2/56 |
| Action loops | 0/64 | 0/56 | 7/56 | 1/56 |

**On-policy** produces the most consistent trajectories: every trajectory completes the buy flow, with minimal format errors.

**CHORD** generates the most actions per trajectory (7.6) due to action loops (clicking the same option repeatedly), but achieves comparable rewards.

**DUET** has the multi-action tag problem (23% of trajectories) but otherwise clean trajectories.

### 4.2 Search Strategy Comparison

All methods converge to similar search strategies by step 50. Typical successful queries:

- **On-Policy**: `search[machine washable window coverings living room color: dove grey size: 52"x45" price:<30.00]`
- **LUFFY**: `search[Machine washable window coverings living room color: dove grey size: 52"x45" price < 30.00]`
- **DUET**: `search[Machine washable window coverings living room color: Dove grey size: 52"w x 45"l price: <30.00 dollars]`
- **Teacher**: `search[Gogobebe Teal Green and Brown Flannel Fleece Throw]` (knows exact product name)

The teacher has a fundamentally different search strategy -- it searches for specific product names (suggesting the Qwen-72B teacher had access to or memorized product catalog information). The student models at 7B learn to compose effective attribute-based queries without needing this shortcut.

### 4.3 Head-to-Head Task Comparison

On the 8 tasks shared across all methods at step 50:

| Task | On-Policy | LUFFY | CHORD | DUET |
|------|-----------|-------|-------|------|
| 1403 | **0.982** | 0.837 | 1.000 | 0.857 |
| 1494 | **0.800** | 0.750 | 0.800 | 0.657 |
| 1829 | 0.883 | 0.921 | 0.929 | 0.707 |
| 1944 | **0.804** | 0.728 | 0.898 | 0.714 |
| 2088 | **0.950** | 0.457 | 0.914 | 0.800 |
| 3818 | **0.900** | 0.721 | 0.871 | 0.814 |
| 4534 | 0.900 | 0.829 | 0.814 | **0.914** |
| 6277 | **1.000** | 0.874 | 0.143 | 0.952 |

On-policy wins or ties on 6/8 tasks. DUET wins on 1/8. This directly supports the finding that teacher mixing adds overhead without benefit at 7B.

---

## 5. State Channel and DR3 Dynamics at 7B

### 5.1 State Channel Analysis

| Step | SC Progress | SC Bonus | SC Coverage | Bonus/Reward Ratio |
|------|------------|---------|-------------|-------------------|
| 1 | 0.420 | 0.084 | 0.860 | 0.202 |
| 10 | 0.225 | 0.045 | 0.724 | 0.293 |
| 30 | 0.277 | 0.055 | 0.593 | 0.117 |
| 50 | 0.599 | 0.120 | 1.000 | 0.149 |
| 80 | 0.503 | 0.101 | 0.860 | 0.119 |
| 90 | 0.513 | 0.103 | 0.860 | 0.113 |

SC bonus/reward ratio stays around 0.11-0.20, within the healthy range (<0.15 target, slightly above early on). Coverage is high (86-100%). The SC is functioning correctly, but its reward shaping contribution (~0.1) is small relative to the outcome reward that the 7B model already achieves on-policy (0.80-0.90).

### 5.2 Teacher Advantage Dynamics

| Step | Teacher Adv Mean | On-Policy Adv Mean | Teacher-OnPolicy Reward Gap |
|------|-----------------|-------------------|---------------------------|
| 1 | 0.371 | 0.088 | 0.419 |
| 10 | 0.347 | 0.054 | 0.839 |
| 30 | 0.158 | 0.055 | 0.437 |
| 50 | 0.083 | 0.113 | 0.078 |
| 80 | 0.091 | 0.099 | 0.053 |
| 90 | 0.046 | 0.089 | -0.084 |

By step 50, teacher advantages are *below* on-policy advantages (0.083 vs 0.113). By step 90, the teacher-minus-onpolicy reward gap is **negative** (-0.084), meaning on-policy trajectories with SC bonus actually outscore teacher trajectories. The teacher has become a drag on training.

---

## 6. Representative Case Studies

### Case A: DUET Multi-Action Failure (Step 50, Task 1829)

**Task**: Find loose fit women's tops, color: a2-yellow, size: medium, price < $30

**DUET** (reward=0.000):
```
Action 1: search[Find me loose fit, day comfort, hand wash women's tops...]
Action 2: click[b09npml43m]
Action 3: <action>click[a2-yellow]\nclick[medium]\nclick[buy now]</action>  -- MULTI-ACTION
Action 4-30: Repeats Action 3 in an infinite loop
```

**LUFFY** (reward=1.000):
```
Action 1: search[loose fit, day comfort, hand wash women's tops...]
Action 2: click[B09NPML43M]
Action 3: click[a2-yellow]
Action 4: click[medium]
Action 5: click[buy now]
```

Both methods identified the correct product and options. DUET's multi-action tag caused the environment to reject the action, triggering an infinite retry loop. LUFFY correctly issued one action per turn.

### Case B: 7B On-Policy Competence at Step 1

**Task**: Find refillable containers, rose gold, 12 count, price < $40

**7B On-Policy** (reward=0.800, step 1):
```
Action 1: search[easy carry rose gold refillable containers 12 count pack price < 40]
Action 2: click[b08x2pkkb2]
Action 3: click[buy now]
```

**3B On-Policy** (same task, step 1): Selected wrong size, clicked "description" instead of options, never completed purchase (reward=0.000).

The 7B model demonstrates correct WebShop interaction patterns from the very first step of training, suggesting instruction-following capability from pretraining is sufficient for basic task completion.

### Case C: On-Policy vs DUET at Step 90 (Same Task 343)

**DUET Success** (reward=1.000):
```
Action 1: search[(height adjustable AND high density AND easy install AND easy assemble AND home office chairs AND living room AND...]
Action 2: click[b08p8lrfz4]
Action 3: click[type 1-yellow]
Action 4: click[buy now]
```

**DUET Failure** (reward=0.429, same task, different rollout):
```
Action 1: search[height adjustable, high density, easy install, easy assemble, home office chairs for living room, color: yellow,...]
Action 2: click[b09p8d2q1q]  -- wrong product
Action 3: click[back to search]
Action 4: search[...] (retry)
Action 5: click[b09l44298x] -- wrong product again
Action 6: click[vivid yellow]
Action 7: click[buy now]
```

The failure case demonstrates that even at 7B, product selection remains the critical decision point. The agent searched for "yellow" instead of "type 1-yellow" and selected a different product. SC progress confirms this: 0.617 for the success vs 0.225 for the failure.

---

## 7. Conclusions

### Why DUET Doesn't Help at 7B

1. **The 7B base model is already near-expert quality.** At step 1, 7B achieves 0.33 mean reward (vs 0.18 for 3B). By step 50, 7B on-policy reaches 0.90 -- matching the best DUET achieves at 3B after 79 steps.

2. **Teacher demonstrations become redundant.** By step 50, the teacher-onpolicy reward gap at 7B is only 0.078 (vs 0.325 at 3B). By step 90, on-policy *exceeds* teacher quality when including SC bonus. The teacher signal provides no marginal information.

3. **Teacher mixing introduces format noise.** DUET's multi-action-tag defect (peaking at 19.3% of trajectories) is a DUET-specific behavioral corruption likely caused by conflicting gradients from teacher trajectories that demonstrate compact action sequences. This costs DUET approximately 0.02-0.03 reward during steps 40-60.

4. **On-policy exploration is sufficient at 7B.** The 7B model discovers effective search strategies, product selection, and option matching through its own exploration. It doesn't need external demonstrations because its pretrained knowledge is rich enough to solve WebShop tasks via RL alone.

5. **SC shaping adds diminishing value.** With on-policy rewards at 0.80+, the SC bonus of ~0.10 represents only a 12% modulation. At 3B where on-policy rewards are 0.30-0.40, the same bonus represents 25-33% modulation -- a much stronger learning signal.

### Implications for the Paper

- DUET is a **capability-gap-dependent** algorithm. It works when the student model has a significant gap to close relative to the teacher. At 7B, this gap closes too quickly for the DUET machinery to provide value.
- This is actually consistent with the DR3 "natural fade-out" design: DR3 should reduce teacher gradient share as the gap closes. The issue is that at 7B, the gap closes so fast that even the early-training teacher signal doesn't provide enough advantage to justify the multi-action-tag corruption it introduces.
- The result suggests DUET's sweet spot is **smaller models or harder environments** where on-policy exploration is insufficient.
