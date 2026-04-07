# DR3 Discriminator Improvement Proposals

## Executive Summary

DR3 provides trajectory-level off-policy correction for teacher samples via a discriminator-based density ratio estimator. The current implementation uses a 12-dim feature vector (v3_aug: log-prob statistics) fed into a 5K-parameter MLP. Analysis reveals the feature space is the primary bottleneck — not the discriminator architecture or training procedure.

**Core narrative**: DR3 = multi-scale off-policy correction (trajectory-level, complementing LUFFY's token-level p/(p+beta)).

**Three highest-priority improvements** (addressing the most likely NeurIPS reviewer critiques):
1. **Action-token masking** — remove observation token noise from features
2. **Step-level feature segments** — capture temporal structure of policy mismatch
3. **Platt scaling calibration** — ensure w_hat values are meaningful, not just separable

---

## Current System Profile

### Feature Extraction (v3_aug, 12 dimensions)

| # | Feature | What it captures |
|---|---------|-----------------|
| 1-4 | lp_mean, lp_std, lp_min, lp_max | Central tendency and extremes of log-prob distribution |
| 5-7 | lp_low_ratio_10, lp_low_ratio_20, lp_high_ratio_1 | Tail behavior (how surprised is the student by this trajectory?) |
| 8 | resp_len | Response length (coarse task complexity proxy) |
| 9-12 | kl_ref_mean, kl_ref_std, kl_ref_abs_mean, kl_ref_pos_ratio | Divergence from reference policy |

### What's Lost
- **Sequential structure**: All features are trajectory-level aggregates. Cannot distinguish "uniformly medium surprise" from "low surprise prefix + high surprise suffix"
- **Action vs observation tokens**: Features mix action tokens (policy-informative) with observation tokens (environment-generated, uninformative for distribution comparison)
- **Step-level granularity**: Agent trajectories have natural step boundaries (step_ids available in trainer but NOT passed to actor update)
- **Cross-token dependencies**: Only global statistics survive

### Discriminator Architecture
- MLP: Linear(12,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,1)
- **5,057 parameters**, < 1MB total memory (including buffer)
- < 1ms overhead per micro-batch (vs 100-500ms for LLM forward pass)
- Massive headroom for richer features without OOM risk

### Available But Unused Information

| Data | Currently Used? | Effort to Add | Value |
|------|----------------|---------------|-------|
| `step_ids` (step boundaries) | NO (not in select_keys) | 1 line | HIGH — enables step-level features |
| `old_log_probs` (rollout-time policy) | NO | Trivial (add arg) | MEDIUM — captures policy shift |
| Token-level entropy | NO (entropy_coeff=0) | Small | MEDIUM — uncertainty signal |
| `teacher_loss_scale` (reward-gap gate) | NO | 1-2 lines | LOW — leaks reward info |
| Multi-layer hidden states | NO (only last layer hook exists) | Medium | MEDIUM — richer representation |
| Attention patterns | NO | Large (output_attentions=True) | LOW — too expensive |
| Reference model hidden states | NO | Large (cross-worker) | NOT RECOMMENDED |

---

## Improvement Proposals

### Proposal 1: Action-Token Masking [PRIORITY: HIGH, EFFORT: LOW]

**Problem**: Current features compute statistics over ALL response tokens. In agent tasks, response = interleaved (action_tokens, observation_tokens). Observation tokens are environment-generated — their log-prob under the student reflects language model priors, not policy differences. Including them dilutes the signal.

**Change**: When computing v3_aug features, restrict `response_mask` to action tokens only (exclude environment observation tokens within the response).

**Implementation**: Use `step_ids` + message role markers (or `loss_mask` from multi-turn) to identify action vs observation tokens. Compute features only over action tokens.

**Expected benefit**: Sharper discriminator signal — teacher and student action tokens have very different log-prob distributions, but observation tokens are similar for both. Removing observation noise should improve discriminator calibration.

**Compute cost**: Zero — same tensor operations with a different mask.

**Risk**: Requires `step_ids` or equivalent in actor update (see "Available But Unused" table — 1-line fix to add to `select_keys`). In some environments, action/observation boundary may be blurry.

---

### Proposal 2: Step-Level Feature Segments [PRIORITY: HIGH, EFFORT: MEDIUM]

**Problem**: A single 12-dim vector per trajectory loses temporal structure. A teacher trajectory might be similar to the student's policy in early steps (navigation) but very different in later steps (experiment execution). Averaging masks this.

**Change**: Partition response tokens by `step_ids`. Compute v3_aug statistics per step. Then aggregate step-level features before the discriminator using one of:
- (A) Mean-pool step features → same discriminator: `feat = mean([feat_step_k])`
- (B) Per-step discriminator + mean logit: `logit = mean([D(feat_step_k)])`
- (C) Concatenate [mean, std, max] of step-level features → slightly larger discriminator

**Expected benefit**: Captures the temporal pattern of policy mismatch. The discriminator can learn that teacher trajectories have "low mismatch in early steps, high mismatch in late steps" — a signal invisible to trajectory-level statistics. This directly addresses the "lossy convergence" failure mode where trajectory-level features become similar even though per-step differences remain.

**Compute cost**: Low. `step_ids` already exist. Per-step masked statistics are cheap. Discriminator input grows from 12 to ~36 (option C), requiring hidden_dim increase to 96-128.

**Risk**: Variable step counts require padding/truncation. Short trajectories (1-3 steps) may not benefit.

---

### Proposal 3: Platt Scaling Calibration [PRIORITY: HIGH, EFFORT: LOW]

**Problem**: The discriminator outputs `D(x) ≈ P(on-policy | features)`, but label smoothing (eps=0.05) and temperature scaling (T=1.0-1.5) deliberately miscalibrate the outputs. The density ratio `r = D/(1-D)` depends critically on calibrated probabilities — even small calibration errors cause large ratio errors near D=0 or D=1.

**Change**: After each discriminator training round, fit a 2-parameter affine calibration on a held-out portion (20%) of the buffer:
```
d_calibrated = sigmoid(a * logit + b)
```
where (a, b) are fit by minimizing NLL on the held-out set. This replaces the fixed temperature parameter.

**Expected benefit**: Ensures w_hat values are meaningful posterior probability ratios, not just "separable but miscalibrated" scores. This is critical because the whole off-policy correction theory assumes D(x) is a good posterior estimate.

**Compute cost**: Negligible — fitting 2 parameters on ~200 held-out samples.

**Risk**: Reduces effective training data by 20%. Alternative: use the newest buffer entries (not yet trained on) as calibration set.

---

### Proposal 4: Smooth Warmup Ramp [PRIORITY: MEDIUM-HIGH, EFFORT: LOW]

**Problem**: The current `apply_warmup_steps=10` uses a hard cutoff: w_hat=1.0 for steps 1-10, then switches to discriminator-estimated w_hat. With `use_relative_ratio=false` (0406 config), the direct ratio can be extreme early in training when distributions are maximally different. The hard transition can cause a sudden gradient shock.

**Change**: Replace hard cutoff with exponential ramp:
```
w_effective = (1-gamma) * 1.0 + gamma * w_hat
gamma = min(1.0, (step - warmup_start) / ramp_length)
```

**Expected benefit**: Smoother transition prevents gradient shock. The discriminator gets time to stabilize before its output affects training.

**Compute cost**: Zero — purely scaling existing w_hat values.

**Risk**: Adds one hyperparameter (ramp_length). Could delay the benefit of off-policy correction if ramp is too slow.

---

### Proposal 5: Percentile-Based Features [PRIORITY: MEDIUM, EFFORT: LOW]

**Problem**: Fixed thresholds (lp_low_ratio_10, lp_low_ratio_20, lp_high_ratio_1) use hard-coded log-prob boundaries (-10, -20, -1) that may not be optimal across environments, model sizes, or training stages.

**Change**: Replace/augment with quantile features: 10th, 25th, 50th (median), 75th, 90th percentile of the per-token log-prob distribution.

**Expected benefit**: Scale-invariant and captures full distributional shape without hard-coded thresholds. More informative about tails than mean/std alone.

**Compute cost**: Minimal — `torch.quantile` on 1D tensor per trajectory.

**Risk**: Increases feature dim from 12 to ~15. Minimal discriminator capacity concern.

---

### Proposal 6: Log-Prob Histogram Features [PRIORITY: MEDIUM, EFFORT: LOW]

**Problem**: Summary statistics (mean, std, quantiles) cannot distinguish distributions with the same moments but different shapes (e.g., bimodal vs uniform).

**Change**: Compute fixed-bin histogram (e.g., 16 bins spanning [-30, 0]) of the per-token log-prob distribution. Each bin value = fraction of tokens in that range.

**Expected benefit**: Captures full distributional shape. Two trajectories with identical mean/std but different log-prob distributions will produce different histograms.

**Compute cost**: O(n) binning per trajectory. Adds ~16 features.

**Risk**: Fixed bin edges may not be optimal. Adaptive (quantile-based) binning could address this.

---

### Proposal 7: MC Dropout Uncertainty [PRIORITY: LOW-MEDIUM, EFFORT: MEDIUM]

**Problem**: w_hat is a point estimate with no uncertainty quantification. When the discriminator is uncertain (overlap region of distributions), the point estimate can be arbitrary.

**Change**: Run discriminator K=3-5 times with dropout enabled. Compute mean and variance of w_hat. When variance is high, shrink w_hat toward 1.0 (no correction):
```
w_final = w_mean * (1 - uncertainty_shrinkage) + 1.0 * uncertainty_shrinkage
```

**Expected benefit**: Prevents harmful corrections where discriminator is guessing. Also provides a diagnostic metric (`dr3/w_hat_uncertainty`).

**Compute cost**: K × discriminator inference — still negligible (MLP is 5K params).

**Risk**: MC Dropout uncertainty can be poorly calibrated for small networks. Add dropout to MLP first (currently no dropout on MLP layers).

---

## Theoretical Considerations

### The Hybrid Composition: w_hat × p/(p+beta)

The product of trajectory-level density ratio and token-level heuristic shaping is NOT a single well-defined importance weight. This is a real theoretical weakness that reviewers may target.

**Recommended framing**: Hybrid is a variance-reduction factorization, not a single-formula derivation:
- w_hat removes coarse trajectory-level distribution mismatch
- p/(p+beta) handles fine-grained token-level credit assignment
- Neither alone is sufficient; their combination empirically dominates either alone
- The product is bounded (w_hat ∈ [0, clip_max], p/(p+beta) ∈ [0, 1]), so no divergence risk

**Precedent**: PPO itself composes heuristics (clipping + advantage normalization) without a single convergence proof.

### The Feature-Space Bottleneck

The discriminator estimates `P(on-policy | features(x))`, not `P(on-policy | x)`. These are equal only if features are sufficient statistics — they are not.

**Recommended framing**: DR3 estimates a "coarsened trajectory-level density ratio." For the purpose of trajectory-level reweighting, coarsened ratios are sufficient — we need relative ordering (which teacher trajectories are more compatible with current policy), not exact importance weights.

### The old_log_prob Construction Trick

Setting `old_log_prob_eff = log_prob.detach() - log(w_hat)` makes the PPO ratio for teacher samples equal to w_hat. This is elegant but has a subtle implication: PPO clipping now clips the density ratio (not the standard policy ratio). This is actually desirable — it provides additional safety against extreme density ratios.

---

## NeurIPS Reviewer Defenses

| Critique | Defense |
|----------|---------|
| "12 features can't estimate density ratios" | DR3 estimates a *coarsened* trajectory-level ratio. For reweighting, relative ordering suffices. Proposals 1-2 enrich features. |
| "1 training step can't track non-stationarity" | Age-weighted loss + EMA alpha handle drift. Policy changes slowly (PPO clip). Ablation: vary disc_steps 1-5. |
| "Hybrid has no convergence guarantee" | Variance-reduction factorization. Bounded product. Empirical ablation shows combination > parts. |
| "use_relative_ratio switch contradicts theory" | Relative ratio has insufficient dynamic range at alpha=0.125 (max 1.143). Direct ratio + clipping provides both range and safety. |
| "Discriminator learns success predictor, not density ratio" | v3_aug excludes reward/advantage by design. Proposal 1 (action masking) further isolates policy signal. |

---

## Recommended Implementation Order

1. **Add `step_ids` to actor `select_keys`** — 1 line, unlocks proposals 1 and 2
2. **Proposal 1: Action-token masking** — low effort, high impact
3. **Proposal 3: Platt scaling** — low effort, high impact on calibration
4. **Proposal 4: Smooth warmup ramp** — config-level change, addresses early-training instability
5. **Proposal 2: Step-level features** — medium effort, highest theoretical impact
6. **Proposals 5-6: Percentile + histogram** — low effort quality-of-life improvements
7. **Proposal 7: Uncertainty** — only if calibration (Proposal 3) proves insufficient

Proposals 1-4 can be implemented with minimal code changes and no OOM risk.
Proposal 2 requires moderate refactoring of `compute_sequence_features()` and discriminator input handling.
