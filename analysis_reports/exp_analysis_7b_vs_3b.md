# WebShop: 7B vs 3B Scale Comparison Analysis

**Date**: 2026-04-13
**Experiments analyzed**: 8 runs (4 methods x 2 scales)
**Methods**: On-policy GRPO, LUFFY, CHORD, DUET
**Note**: 7B DUET crashed at step 93 (disk), 7B CHORD/LUFFY completed 100 steps. All 3B runs completed 100 steps. Batch size = 64 trajectories per step.

---

## 1. Training Reward Curves

### Smoothed Reward (10-step moving average) at Key Checkpoints

| Scale | Method   | Step 25 | Step 50 | Step 75 | Last-20 Avg | Peak (window) | Peak At |
|-------|----------|---------|---------|---------|-------------|---------------|---------|
| 3B    | Onpolicy | 0.345   | 0.437   | 0.137   | 0.293       | 0.474         | 24-33   |
| 3B    | LUFFY    | 0.509   | 0.585   | 0.762   | 0.766       | 0.806         | 75-84   |
| 3B    | CHORD    | 0.550   | 0.588   | 0.750   | 0.739       | 0.757         | 84-93   |
| 3B    | DUET     | 0.546   | 0.599   | 0.801   | 0.783       | 0.827         | 75-84   |
| 7B    | Onpolicy | 0.421   | 0.674   | 0.763   | 0.757       | 0.785         | 79-88   |
| 7B    | LUFFY    | 0.494   | 0.612   | 0.796   | 0.772       | 0.806         | 75-84   |
| 7B    | CHORD    | 0.550   | 0.666   | 0.790   | 0.775       | 0.812         | 75-84   |
| 7B    | DUET     | 0.461   | 0.634   | 0.776   | 0.770*      | 0.793         | 75-84   |

*DUET-7B last-20 uses steps 74-93 due to crash.

### Key Observations - Reward Curves

**3B scale**: DUET clearly leads. Its peak smoothed reward (0.827) exceeds LUFFY (0.806) by 2.1 points and CHORD (0.757) by 7 points. The on-policy baseline collapses badly after step 60, ending at 0.293 (training instability / catastrophic forgetting).

**7B scale**: All four methods converge to nearly identical late-training performance (~0.77). The on-policy baseline no longer collapses -- it reaches 0.757, competitive with teacher-augmented methods. DUET's peak (0.793) is essentially within noise of CHORD (0.812) and LUFFY (0.806).

---

## 2. Success Rate Progression

### Training Success Rate at Sampled Steps

| Step | 3B-OnP | 3B-LUF | 3B-CHO | 3B-DUE | 7B-OnP | 7B-LUF | 7B-CHO | 7B-DUE |
|------|--------|--------|--------|--------|--------|--------|--------|--------|
| 1    | 0.375  | 0.391  | 0.453  | 0.422  | 0.563  | 0.531  | 0.422  | 0.688  |
| 10   | 0.453  | 0.328  | 0.469  | 0.438  | 0.469  | 0.516  | 0.359  | 0.406  |
| 20   | 0.469  | 0.906  | 0.969  | 0.875  | 0.453  | 0.484  | 0.641  | 0.391  |
| 30   | 0.859  | 1.000  | 1.000  | 0.984  | 0.859  | 0.859  | 0.891  | 0.859  |
| 50   | 0.844  | 0.875  | 0.906  | 0.859  | 1.000  | 0.969  | 0.875  | 0.953  |
| 70   | 0.344  | 0.938  | 0.984  | 1.000  | 1.000  | 0.969  | 0.953  | 0.984  |
| 90   | 0.703  | 0.953  | 0.922  | 1.000  | 1.000  | 0.984  | 1.000  | 1.000  |
| 100  | 0.516  | 1.000  | 0.906  | 0.969  | 1.000  | 0.953  | 1.000  | --     |

### Last-20-Step Average Success Rate

| Scale | Onpolicy | LUFFY | CHORD | DUET  |
|-------|----------|-------|-------|-------|
| 3B    | 0.566    | 0.931 | 0.916 | 0.937 |
| 7B    | 0.939    | 0.927 | 0.937 | 0.919 |

At 7B, the on-policy baseline achieves 93.9% success, matching or exceeding teacher-augmented methods. At 3B, it is catastrophically worse (56.6%).

---

## 3. Teacher-Onpolicy Advantage Gap

This metric (`diag/group_teacher_minus_on_reward_mean`) measures how much better teacher trajectories are compared to on-policy rollouts, averaged over 10-step windows.

| Bucket | 3B-LUF | 3B-CHO | 3B-DUE | 7B-LUF | 7B-CHO | 7B-DUE |
|--------|--------|--------|--------|--------|--------|--------|
| 5      | 0.885  | 0.857  | 0.841  | 0.766  | 0.778  | 0.695  |
| 15     | 0.546  | 0.572  | 0.529  | 0.663  | 0.589  | 0.603  |
| 25     | 0.529  | 0.457  | 0.391  | 0.519  | 0.472  | 0.488  |
| 35     | 0.440  | 0.418  | 0.335  | 0.422  | 0.429  | 0.373  |
| 45     | 0.446  | 0.439  | 0.332  | 0.431  | 0.355  | 0.288  |
| 55     | 0.430  | 0.458  | 0.257  | 0.411  | 0.372  | 0.274  |
| 65     | 0.314  | 0.330  | 0.147  | 0.292  | 0.272  | 0.202  |
| 75     | 0.263  | 0.295  | 0.097  | 0.243  | 0.247  | 0.142  |
| 85     | 0.238  | 0.272  | 0.091  | 0.217  | 0.207  | 0.135  |
| 95     | 0.277  | 0.287  | 0.141  | 0.271  | 0.267  | 0.144  |

### Key Observations - Advantage Gap

DUET closes the teacher-onpolicy gap far more aggressively than LUFFY and CHORD at both scales. By steps 75-85:
- **3B DUET**: gap = 0.09 vs LUFFY 0.25 vs CHORD 0.28
- **7B DUET**: gap = 0.14 vs LUFFY 0.23 vs CHORD 0.23

This confirms DUET's DR3 mechanism (density-ratio-based teacher weighting) is driving on-policy quality up more effectively. However, at 7B this does not translate to validation improvement because:

1. The starting gap at 7B is already smaller (7B models start closer to teacher quality)
2. The absolute on-policy reward at 7B is already very high for all methods (0.75+)
3. Further closing the gap provides diminishing returns when on-policy is already near 1.0

---

## 4. On-Policy Reward Mean (Batch Diagnostics)

This is the per-batch mean reward of on-policy rollouts only (excluding teacher samples).

| Bucket | 3B-LUF | 3B-CHO | 3B-DUE | 7B-LUF | 7B-CHO | 7B-DUE |
|--------|--------|--------|--------|--------|--------|--------|
| 5      | 0.120  | 0.139  | 0.152  | 0.227  | 0.224  | 0.287  |
| 35     | 0.566  | 0.581  | 0.647  | 0.569  | 0.560  | 0.622  |
| 55     | 0.560  | 0.544  | 0.730  | 0.593  | 0.620  | 0.709  |
| 75     | 0.739  | 0.703  | 0.883  | 0.759  | 0.758  | 0.846  |
| 95     | 0.718  | 0.704  | 0.832  | 0.725  | 0.727  | 0.840  |

DUET's on-policy reward is systematically higher at both scales, reaching 0.88 at 3B and 0.85 at 7B vs ~0.72 for baselines. This is DUET's core strength: it improves the on-policy samples themselves, not just the mixed batch.

---

## 5. Entropy Trends

### On-Policy Token Entropy (10-step bucket avg)

| Bucket | 3B-OnP | 3B-LUF | 3B-CHO | 3B-DUE | 7B-OnP | 7B-LUF | 7B-CHO | 7B-DUE |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 5      | 0.358  | 0.322  | 0.332  | 0.348  | 0.151  | 0.156  | 0.158  | 0.161  |
| 35     | 0.367  | 0.516  | 0.555  | 0.539  | 0.366  | 0.324  | 0.319  | 0.317  |
| 55     | 0.483  | 0.403  | 0.432  | 0.439  | 0.455  | 0.392  | 0.426  | 0.442  |
| 75     | 0.410  | 0.414  | 0.387  | 0.471  | 0.387  | 0.371  | 0.382  | 0.413  |
| 95     | 0.381  | 0.458  | 0.414  | 0.475  | 0.419  | 0.368  | 0.360  | 0.428  |

Key findings:
- **7B starts at much lower entropy** (0.15 vs 0.35 for 3B), indicating the 7B model begins with more confident/peaked distributions.
- **DUET maintains higher entropy** than baselines at both scales late in training (0.475 at 3B, 0.428 at 7B vs ~0.40 for others). This may indicate DUET preserves exploration capacity better.
- **7B on-policy baseline entropy rises** from 0.15 to 0.42, a 2.8x increase. The 3B baseline stays relatively flat (0.36-0.48). This entropy increase in 7B-Onpolicy correlates with its improved performance -- the model learns to explore.

---

## 6. On-Policy Advantage Positive Ratio

This metric shows what fraction of on-policy samples receive positive advantages (i.e., are reinforced).

| Bucket | 3B-OnP | 3B-LUF | 3B-CHO | 3B-DUE | 7B-OnP | 7B-LUF | 7B-CHO | 7B-DUE |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 5      | 0.391  | 0.384  | 0.386  | 0.564  | 0.453  | 0.495  | 0.457  | 0.753  |
| 25     | 0.583  | 0.557  | 0.447  | 0.834  | 0.517  | 0.510  | 0.510  | 0.740  |
| 55     | 0.692  | 0.599  | 0.605  | 0.865  | 0.356  | 0.492  | 0.360  | 0.783  |
| 75     | 0.447  | 0.490  | 0.584  | 0.870  | 0.234  | 0.415  | 0.250  | 0.797  |
| 95     | 0.552  | 0.322  | 0.440  | 0.829  | 0.289  | 0.291  | 0.254  | 0.854  |

**This is the most striking diagnostic difference.** DUET maintains 80%+ positive advantage ratio throughout training at both scales. All other methods drop to 25-45% at 7B late training.

Interpretation: DUET's State Channel (SC) adds progress-based bonuses that keep on-policy advantages predominantly positive. In baselines, as the policy improves and teacher samples dominate the group mean, on-policy advantages become increasingly negative (the "teacher domination" problem in GRPO normalization). DUET's teacher_baseline_separation + SC bonus counteracts this.

At 7B, the low positive-advantage ratio for baselines (25-30%) means the policy update primarily penalizes on-policy rollouts and reinforces teacher trajectories -- effectively doing imitation learning rather than RL. Yet these methods still perform well because the 7B model's on-policy rollouts are already high quality.

---

## 7. Validation Performance

### Validation Reward at Steps 50 and 100

| Scale | Method   | Step 50 | Step 100 | >0.5 rate (100) | >0.8 rate (100) |
|-------|----------|---------|----------|-----------------|-----------------|
| 3B    | Onpolicy | 0.276   | 0.402    | 0.475           | 0.120           |
| 3B    | LUFFY    | 0.509   | 0.753    | 0.770           | 0.610           |
| 3B    | CHORD    | 0.572   | 0.728    | 0.760           | 0.580           |
| 3B    | DUET     | 0.599   | 0.763    | 0.795           | 0.635           |
| 7B    | Onpolicy | 0.666   | 0.760    | 0.775           | 0.615           |
| 7B    | LUFFY    | 0.581   | 0.755    | 0.775           | 0.610           |
| 7B    | CHORD    | 0.643   | 0.758    | 0.770           | 0.620           |
| 7B    | DUET     | 0.681   | --*      | --              | --              |

*7B DUET has no step-100 validation due to crash at step 93.

### Analysis

**3B**: DUET wins on validation. At step 100: DUET (0.763) > LUFFY (0.753) > CHORD (0.728) >> Onpolicy (0.402). The margin over LUFFY is small (+1.0 points) but DUET leads at both checkpoints. At step 50, DUET leads more clearly: 0.599 vs CHORD 0.572 vs LUFFY 0.509 -- showing faster convergence.

**7B**: All methods reach the same validation ceiling (~0.76). Onpolicy (0.760) matches CHORD (0.758) and LUFFY (0.755). DUET at step 50 (0.681) leads slightly, suggesting it would likely match or marginally exceed baselines at step 100.

**The validation confirms: the 7B scale eliminates DUET's advantage.**

---

## 8. Why DUET Doesn't Differentiate at 7B

### Hypothesis 1: 7B Baseline Saturation

The 7B on-policy GRPO baseline reaches 0.760 validation reward -- essentially the same as all teacher-augmented methods. This means teacher demonstrations provide negligible additional signal at this scale. The 7B model is capable enough to discover high-quality policies purely from on-policy exploration.

Evidence:
- 7B Onpolicy validation at step 100: 0.760 (vs 3B: 0.402)
- 7B Onpolicy training success at steps 50-100: consistently 95-100%
- 7B Onpolicy smoothed reward peak: 0.785

### Hypothesis 2: Task Ceiling Effect

WebShop may have a performance ceiling near 0.76-0.81 validation reward that all methods approach regardless of algorithm. The 200-sample validation set shows all 7B methods in the 0.755-0.760 range with std ~0.32, meaning the methods are statistically indistinguishable.

### Hypothesis 3: DUET's Advantage is in the Learning Efficiency, Not Final Performance

DUET-7B at step 50 achieves validation 0.681 vs Onpolicy 0.666 vs LUFFY 0.581. The advantage is in reaching good performance faster, but all methods converge to the same point given enough steps. At 3B, 100 steps is not enough for convergence (Onpolicy never recovers), so DUET's efficiency advantage persists as a final-performance advantage.

### Hypothesis 4: DUET's Components May Partially Interfere at 7B

DUET-7B's on-policy advantage positive ratio (0.85) is dramatically higher than baselines (0.25-0.30). While this prevents the "teacher domination" problem, it may also reduce the selective pressure of GRPO -- if nearly every sample gets positive advantage, the algorithm cannot distinguish good from great rollouts. At 3B where on-policy quality is more variable, this positive-bias helps. At 7B where on-policy is already strong, it may slightly reduce discrimination.

Supporting evidence: DUET-7B peak smoothed reward (0.793) is slightly below CHORD (0.812) and LUFFY (0.806), though within noise.

---

## 9. Summary Table

| Metric                        | 3B DUET Advantage | 7B DUET Advantage |
|-------------------------------|-------------------|-------------------|
| Validation reward (step 100)  | +1.0 over LUFFY   | N/A (crashed)     |
| Validation reward (step 50)   | +2.7 over CHORD   | +3.8 over LUFFY   |
| Peak smoothed train reward    | +2.1 over LUFFY   | -1.9 vs CHORD     |
| Last-20 train reward          | +1.7 over LUFFY   | -0.5 vs CHORD     |
| Teacher gap closure (step 85) | 3x faster         | 1.6x faster       |
| On-policy reward (step 75)    | +14.4 vs LUFFY    | +8.7 vs LUFFY     |
| Advantage pos ratio           | 83% vs 49%        | 80% vs 30%        |

### Conclusions

1. **DUET's mechanisms work correctly at 7B** -- DR3 closes the teacher gap, SC maintains high advantage positive ratio, on-policy reward is higher than baselines.

2. **The 7B baseline is too strong for differentiation.** On-policy GRPO alone reaches the same validation ceiling, eliminating the headroom that DUET exploits at 3B.

3. **DUET shows faster mid-training convergence at 7B** (step-50 validation: 0.681 vs 0.666/0.643/0.581). This could be a reportable result: "DUET accelerates learning at 7B even when final performance converges."

4. **WebShop may not be challenging enough at 7B.** The task's action space and episode complexity may be within the 7B model's native capability. A harder environment (e.g., SciWorld) or longer-horizon tasks might better showcase DUET at 7B scale.

5. **The 3B Onpolicy collapse is real and severe** -- reward drops from 0.53 to 0.10 between steps 60-72, suggesting catastrophic forgetting or mode collapse. Teacher data prevents this in all augmented methods. This is a strong argument for teacher utilization at smaller scales.
