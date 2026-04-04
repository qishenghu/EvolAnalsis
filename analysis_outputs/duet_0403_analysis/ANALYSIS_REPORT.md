# DUET 0403 Collapse Analysis Report

**Date:** 2026-04-02
**Wandb Run:** j2rle81i (DUET 0403), o405qtk1 (LUFFY baseline)
**Config changes vs 0402:** SC-GRPO decoupling, teacher advantage clip +-5, gap_gate OFF, adaptive_weight OFF, disc_temperature=2.5

---

## 1. Validation Score (Step 50)

| Metric | Value |
|--------|-------|
| Average reward | 0.6456 |
| Average success | **30.5%** |

This is mid-training (step 50 of ~100). The collapse happens after this checkpoint.

---

## 2. Collapse Timeline

| Phase | Steps | DUET Success (smoothed) | LUFFY Success (smoothed) |
|-------|-------|------------------------|-------------------------|
| Early training | 1-30 | 0.05 -> 0.15 | 0.01 -> 0.05 |
| DUET leads | 30-60 | 0.15 -> 0.40 | 0.05 -> 0.30 |
| Pre-collapse peak | 60-80 | 0.40 -> **0.807 (step 80)** | 0.30 -> 0.789 (step 80) |
| **COLLAPSE** | **80-100** | **0.807 -> 0.143 (-82%)** | 0.789 -> stable ~0.45 |
| Final 5-step mean | 94-98 | **0.238** | **0.455** |

**Key observation:** Both methods peak at step 80 (batch composition effect), but LUFFY recovers and DUET does not. LUFFY ends at 0.455 vs DUET at 0.238.

---

## 3. Root Cause: DR3 Discriminator Degradation

The collapse follows a clear causal chain:

### Step 1: Discriminator Accuracy Degrades (steps 79-98)
```
Step 79: disc_acc = 0.992  (peak)
Step 85: disc_acc = 0.945
Step 90: disc_acc = 0.891
Step 95: disc_acc = 0.789
Step 98: disc_acc = 0.775
```
**Drop: 0.992 -> 0.775 (-22%)**

The discriminator loses its ability to distinguish on-policy from teacher samples. This is NOT normal — as training progresses, the distributions should become MORE distinguishable (policy evolves, teacher is fixed).

### Step 2: Importance Weights Collapse to 1.0
```
Step 79: w_off_mean = 0.768
Step 85: w_off_mean = 0.945
Step 90: w_off_mean = 1.025
Step 95: w_off_mean = 0.990
Step 98: w_off_mean = 1.009
```
**Drift: 0.77 -> 1.0 (+30%)**

When w_off -> 1.0, teacher samples get equal weight as on-policy samples. DR3's correction mechanism is effectively disabled — it's as if we're running without importance weighting at all.

### Step 3: Teacher Gradient Share Drops (Misleading Signal)
```
Step 70-80: teacher_gradient_share = 0.303
Step 80-100: teacher_gradient_share = 0.160
```
This looks like healthy fade-out, but it's actually meaningless when the weights themselves are wrong. The gradient share drops because the discriminator's outputs are unreliable, not because the policy has learned to be self-sufficient.

### Step 4: On-Policy Advantages Go Negative
```
Step 60-80: adv_onpolicy_mean = -0.009
Step 80-100: adv_onpolicy_mean = -0.016  (79% more negative)
```
With corrupted importance weights and declining policy quality, on-policy samples consistently get negative advantages, pushing the policy away from its own (now-degraded) behavior without a clear direction.

---

## 4. What Does NOT Cause the Collapse

| Suspect | Evidence | Verdict |
|---------|----------|---------|
| KL explosion | KL avg 1.24 pre -> 1.10 post (decreases!) | **Cleared** |
| Teacher advantage clipping | Max adv_teacher = 0.40, well under +-5 clip | **Cleared** |
| SC bonus overwhelming | bonus_vs_reward_ratio = 0.083 (stable, <0.15) | **Cleared** |
| Entropy collapse | No entropy metric logged | Inconclusive |
| Teacher ratio change | Stable at ~0.11 throughout | **Cleared** |

---

## 5. LUFFY Stability Comparison

| Metric | LUFFY (80-100) | DUET (80-100) | Interpretation |
|--------|---------------|---------------|----------------|
| Success mean | 0.454 (+4.9% from 60-80) | 0.345 (-21.9% from 60-80) | LUFFY improves, DUET degrades |
| KL loss | 1.685 (rising) | 1.099 (falling) | LUFFY explores more, DUET contracts |
| Teacher grad share | 0.160 (declining) | 0.160 (declining) | Same decay rate, different outcomes |

**LUFFY is completely stable at steps 80-100.** It doesn't have a discriminator that can degrade, so teacher mixing via LUFFY's simpler mechanism remains robust throughout training.

---

## 6. Hypotheses for DR3 Discriminator Failure

### H1: Feature Space Collapse (Most Likely)
As the policy improves, its generated token distributions move toward teacher-like patterns. The discriminator's features can't separate them anymore. This is exacerbated by disc_temperature=2.5 which may smooth the logits too much.

### H2: Discriminator Overfitting then Forgetting
The discriminator reaches 0.99 accuracy by step 79, possibly overfitting to specific distributional artifacts that shift as the policy evolves. When the distribution shifts, the discriminator's learned features become stale.

### H3: Training Instability from SC-GRPO Decoupling
The 0403 change decouples SC bonus from GRPO scores. This changes the advantage landscape. The policy may enter a region where DR3's discriminator was not trained, causing garbage importance weights.

---

## 7. Recommendations

1. **Discriminator reset/warm-restart**: Periodically retrain the discriminator from scratch (every ~20 steps) to prevent stale features from accumulating.

2. **Discriminator accuracy floor**: If disc_acc < 0.85, fall back to LUFFY-style uniform weighting. The importance weights are unreliable below this threshold.

3. **Lower disc_temperature**: Try 1.0 or 1.5 (from current 2.5). High temperature softens discriminator logits, making it harder to maintain separation as distributions converge.

4. **ESS-based early warning**: Monitor ESS (effective sample size) — if ESS approaches batch size, the weights are becoming uniform and DR3 is failing.

5. **Clip w_off away from 1.0**: Instead of just floor (w_min=0.01), add a ceiling check. If mean(w_off) > 0.95 for multiple consecutive steps, the discriminator needs retraining.

---

## 8. Files Generated

| File | Description |
|------|-------------|
| `wandb_data.json` | Raw wandb data for both runs |
| `plot1_training_curve_collapse.png` | Training curve with collapse region |
| `plot2_collapse_diagnostics.png` | KL + teacher_grad_share around collapse |
| `plot3_full_diagnostics.png` | 12-panel full metric dashboard |
| `plot4_luffy_comparison.png` | DUET vs LUFFY stability at steps 80-100 |
| `plot5_collapse_mechanism.png` | 6-panel causal mechanism analysis |
| `plot6_head_to_head.png` | Full training head-to-head with smoothing |
| `plot7_dr3_degradation.png` | DR3 degradation chain detail |
| `plot8_correlation.png` | disc_acc vs success scatter with phase coloring |
| `analysis_summary.txt` | Numeric summary of all metrics |
