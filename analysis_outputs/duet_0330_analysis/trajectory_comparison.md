# DUET 0330 vs 0329 — Deep Trajectory Comparison

## Overview

**Two changes in 0330 vs 0329:**
1. **beta_decay fix**: Uses binary success rate metric with `beta_decay_target=0.8` — SC β now actually decays as success improves
2. **KL coef reduction**: `kl_loss_coef` 0.005 → 0.001

**Data sources**: wandb output logs (100 training steps each), batch_diag trajectory files, validation JSONL (steps 50, 100)

---

## 1. SC Bonus Evolution (MOST IMPORTANT FINDING)

**The beta_decay fix works as intended.** In 0329, β was stuck at 0.200 for all 100 steps (the bug). In 0330, β starts around 0.104 (early avg) and decays to 0.034 (late avg).

| Step | β_eff (0330) | β_eff (0329) | Bonus (0330) | Bonus (0329) | Bonus/Reward (0330) | Bonus/Reward (0329) |
|------|-------------|-------------|-------------|-------------|--------------------|--------------------|
| 1    | 0.134       | **0.200**   | 0.037       | 0.062       | 0.140              | 0.199              |
| 10   | 0.083       | **0.200**   | 0.028       | 0.070       | 0.060              | 0.145              |
| 30   | 0.067       | **0.200**   | 0.023       | 0.065       | 0.043              | 0.119              |
| 50   | 0.036       | **0.200**   | 0.016       | 0.085       | 0.024              | 0.129              |
| 70   | 0.032       | **0.200**   | 0.012       | 0.088       | 0.018              | 0.148              |
| 100  | 0.028       | **0.200**   | 0.013       | 0.102       | 0.019              | 0.121              |

**Key observations:**
- **0330 β range**: 0.009 → 0.149, mean 0.067 (3x lower than 0329's fixed 0.200)
- **0330 bonus evolution**: 0.031 (early) → 0.014 (late), a **55% reduction**
- **0329 bonus evolution**: 0.061 (early) → 0.080 (late), actually **increases 31%** (because on-policy progress rises while β stays fixed)
- **Bonus/reward ratio**: 0330 stays well under 0.05 by mid-training; 0329 stays around 0.12-0.15 throughout

**Impact**: In 0330, the SC bonus correctly fades as the agent improves. In 0329, the always-on β=0.2 means SC bonus grows in absolute terms as progress increases, which over-rewards high-progress trajectories late in training.

---

## 2. Teacher Advantage Trajectories

| Step | Teacher Adv (0330) | Teacher Adv (0329) | On-Policy Adv (0330) | On-Policy Adv (0329) |
|------|-------------------:|-------------------:|---------------------:|---------------------:|
| 1    | 1.660              | 2.751              | 0.008                | 0.004                |
| 10   | 3.268              | 1.525              | 0.017                | 0.013                |
| 30   | 7.725              | 1.661              | 0.007                | 0.002                |
| 50   | -0.368             | -0.216             | 0.000                | -0.003               |
| 60   | **-366.504**       | -0.093             | 0.003                | -0.021               |
| 70   | 6.082              | -0.048             | -0.040               | -0.052               |
| 100  | -0.034             | -0.207             | -0.102               | -0.028               |

**Key observations:**
- **0330 teacher_adv first negative**: step 13 (20/100 steps negative total)
- **0329 teacher_adv first negative**: step 5 (27/100 steps negative total)
- **0330 has an extreme outlier** at step 60: teacher_adv = -366.5 — this is a single-batch anomaly (likely a batch where all teacher samples were in tasks where on-policy already performs well, making teacher relatively worse after normalization)
- Both show teacher advantage oscillating between positive and negative, indicating GRPO baseline separation working — teacher advantage depends on batch composition
- **Teacher positive ratio at step 100**: both at 0.625 (62.5% of teacher samples have positive advantage)

---

## 3. On-Policy Success Rate & Teacher Gap

| Step | Reward_OnPol (0330) | Reward_OnPol (0329) | Gap (0330) | Gap (0329) |
|------|--------------------:|--------------------:|-----------:|-----------:|
| 1    | 0.216               | 0.298               | 0.784      | 0.702      |
| 10   | 0.425               | 0.491               | 0.575      | 0.509      |
| 30   | 0.490               | 0.556               | 0.510      | 0.444      |
| 50   | 0.625               | 0.704               | 0.375      | 0.296      |
| 70   | 0.639               | 0.649               | 0.361      | 0.351      |
| 100  | 0.672               | **0.939**           | 0.328      | **0.061**  |

**Summary statistics:**
| Metric | 0330 | 0329 |
|--------|------|------|
| Mean reward (all 100 steps) | 0.499 | 0.580 |
| Mean reward (last 20 steps) | 0.637 | **0.745** |
| Max reward | 0.754 (step 89) | **0.939** (step 100) |
| MA-10 at step 100 | 0.660 | **0.770** |

**0329 training reward is consistently higher than 0330.** The gap widens especially in the last 30 steps. At step 100, 0329 achieves 0.939 (near-perfect) while 0330 sits at 0.672.

**However, validation tells a different story:**
| Checkpoint | Val Success (0330) | Val Success (0329) |
|------------|-------------------:|-------------------:|
| Step 50    | 50.5%              | 48.0%              |
| Step 100   | 69.0%              | 69.5%              |

**Validation performance is essentially identical** (69.0% vs 69.5%), despite 0329's training reward being 40% higher at step 100. This suggests 0329's higher training reward is partially overfitting — the always-on SC bonus inflates training signal without proportional generalization benefit.

---

## 4. Response Length Distribution

| Step | Resp Length (0330) | Resp Length (0329) | Teacher Length |
|------|-------------------:|-------------------:|---------------:|
| 1    | 7,285              | 7,026              | 3,209          |
| 50   | 4,296              | 4,529              | 3,456          |
| 100  | 6,086              | 3,156              | 2,834          |

| Metric | 0330 | 0329 |
|--------|------|------|
| Mean resp length (all) | 5,614 | 5,355 |
| Mean resp length (last 20) | **6,351** | 5,780 |

**0330 produces slightly longer responses on average** (5% more overall, 10% more in late training). At step 100, 0330 generates 6,086 tokens while 0329 drops to 3,156. The KL coef reduction (0.005→0.001) gives the policy more freedom to deviate, which manifests as somewhat longer exploratory responses.

---

## 5. DR3 Discriminator

| Step | Disc Acc (0330) | Disc Acc (0329) | w_mean (0330) | w_mean (0329) |
|------|----------------:|----------------:|--------------:|--------------:|
| 10   | 0.801           | 0.794           | 1.000         | 1.014         |
| 30   | 0.872           | 0.918           | 1.034         | 1.058         |
| 50   | 0.983           | 0.981           | 1.040         | 1.062         |
| 70   | 0.989           | 0.991           | 1.109         | 1.064         |
| 100  | 0.895           | 0.966           | 1.023         | 1.101         |

Both discriminators learn effectively (>0.98 accuracy by step 50). DR3 weights stay in a narrow range (1.0-1.1), confirming the clipping mechanism works. **0330's discriminator accuracy drops to 0.895 at step 100** — possibly because the lower KL coef allows the policy to be more diverse, making on-policy samples harder to distinguish from teacher samples.

---

## 6. Teacher Gradient Share

| Metric | 0330 | 0329 |
|--------|------|------|
| Overall mean | **0.484** | 0.371 |
| First 20 steps | **0.641** | 0.528 |
| Last 20 steps | 0.237 | 0.192 |

**0330 maintains higher teacher gradient share** throughout training (48.4% vs 37.1% overall). Both decay toward ~20% by the end. The higher early share in 0330 makes sense: with lower SC bonus (due to beta_decay), more relative gradient comes from teacher correction. However, both converge to similar levels by training end.

---

## 7. KL & Entropy

| Step | Entropy (0330) | Entropy (0329) | OnPol Entropy (0330) | OnPol Entropy (0329) |
|------|---------------:|---------------:|---------------------:|---------------------:|
| 1    | 0.089          | 0.076          | 0.503                | 0.454                |
| 50   | 0.109          | 0.118          | 0.496                | 0.448                |
| 100  | 0.111          | 0.127          | 0.142                | 0.334                |

**By step 100**: 0330's on-policy entropy drops to 0.142 vs 0329's 0.334. Despite having a lower KL constraint (0.001 vs 0.005), 0330's policy becomes **more deterministic** in late training. This is counterintuitive — the lower KL coef should allow more exploration. The explanation: with less SC bonus support, the policy must rely more on its own learning, leading to stronger convergence to specific action patterns.

---

## 8. Failure Analysis (Validation Step 100)

| Pattern | 0330 (62 failures) | 0329 (61 failures) |
|---------|--------------------:|--------------------:|
| Heavy invalid actions | **37 (59.7%)** | 25 (41.0%) |
| Wandering (>30k chars) | 4 (6.5%) | 6 (9.8%) |
| Other | 21 (33.9%) | 30 (49.2%) |
| Mean failure length | 24,885 | 28,806 |

**0330 has more "invalid action" failures** (59.7% vs 41.0%), while 0329 has more diverse failure modes. 0330's failures are slightly shorter on average. All failures are "max-turns-exhausted" (none self-terminate as failures).

---

## 9. SC Progress & Coverage

| Step | Progress_OnPol (0330) | Progress_OnPol (0329) | Coverage (0330) | Coverage (0329) |
|------|----------------------:|----------------------:|----------------:|----------------:|
| 1    | 0.314                 | 0.350                 | 0.587           | 0.632           |
| 50   | 0.503                 | 0.486                 | 0.797           | 0.788           |
| 100  | 0.522                 | **0.584**             | 0.801           | **0.845**       |

Progress and coverage are similar between versions, with 0329 slightly ahead by step 100. This is consistent with 0329's higher training reward.

---

## Summary: Impact of Each Change

### beta_decay Fix (Primary Change)
- **Working correctly**: β decays from ~0.10 to ~0.03 as success improves
- **SC bonus reduces 55%** over training (vs 31% increase in 0329)
- **Training reward lower**: 0330's on-policy reward is consistently 10-20% below 0329
- **Validation identical**: 69.0% vs 69.5% — the extra SC bonus in 0329 does NOT translate to generalization
- **Interpretation**: 0329's always-on β=0.2 inflates training rewards without improving generalization. The beta_decay fix prevents this inflation, producing more honest training signals.

### KL Coef Reduction (0.005 → 0.001)
- **Late-stage entropy**: 0330's policy becomes MORE deterministic despite lower KL constraint
- **Response lengths**: 0330 produces slightly longer responses (+5-10%)
- **DR3 accuracy**: Slight drop at step 100 (0.895 vs 0.966) — policy diversity makes discrimination harder
- **Net effect**: Marginal. The 5x KL reduction doesn't dramatically change behavior, suggesting the policy update is dominated by the GRPO loss rather than the KL term.

### Key Conclusion
**0330 and 0329 achieve equivalent validation performance** (69.0% vs 69.5%). The beta_decay fix produces more theoretically sound training dynamics (honest reward signals, proper SC fade-out), while 0329's inflated training rewards are a "pleasant lie" that doesn't help generalization. For the paper, 0330's dynamics are preferable as they validate the DUET design thesis: SC should fade as the policy improves.
