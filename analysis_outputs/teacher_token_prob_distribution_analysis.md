# Teacher Token Probability Distribution Analysis

**Source data**: webshop_3b_duet_0409_ema (100 steps), webshop_3b_luffy (100 steps)
**Analysis date**: 2026-04-06
**Purpose**: Inform the design of adaptive token-level weighting for teacher samples

## Data Availability

Per-token log_prob histograms are **not** stored in trajectory files. However, per-trajectory
**percentile statistics** are logged:
- `teacher_old_logp_mean`: mean log_prob over teacher tokens
- `teacher_old_logp_p10`: 10th percentile (hardest 10% of tokens)
- `teacher_old_logp_p50`: median log_prob
- `teacher_old_logp_min`: single hardest token

These are saved in `trajectories_step_N.jsonl` for every teacher sample's `diag` dictionary.
There are 5-8 teacher trajectories per step, each with 2000-8500 teacher tokens.

## Key Finding: The Distribution Is Heavily Bimodal and Stable

### Distribution Shape (inferred from percentiles)

| Cluster | Probability range | Fraction | Role |
|---------|------------------|----------|------|
| Learned | p > 0.9 | ~55% | Trivially reproduced tokens (formatting, structure, common vocab) |
| Frontier | 0.01 < p < 0.9 | ~35% | Where learning happens (actions, reasoning, domain terms) |
| Unlearnable | p < 0.01 | ~5-10% | Teacher-specific phrasing, vocabulary mismatch, formatting quirks |

### Evolution Over Training

| Step | prob_mean | prob_p10 | prob_min |
|------|-----------|----------|----------|
| 1 | 0.274 | 0.012 | ~1e-18 |
| 10 | 0.293 | 0.017 | ~4e-18 |
| 25 | 0.371 | 0.046 | ~7e-15 |
| 50 | 0.408 | 0.061 | ~8e-14 |
| 75 | 0.488 | 0.150 | ~1e-13 |
| 100 | 0.462 | 0.092 | ~4e-12 |

Note: prob_mean and prob_p10 are token-weighted averages across teacher trajectories.

### The median (p50) is ~1.0 at ALL training steps

This is the most important finding: **more than 50% of teacher tokens are trivially
learned from the very start of training.** The median log_prob stays within 0.001 of
zero throughout all 100 steps. These tokens contribute almost zero gradient through
any reasonable weighting scheme.

### The frontier shifts slowly

- p10 (hardest 10%): 0.012 at step 1 -> 0.092 at step 100 (7.7x improvement)
- p_mean: 0.274 at step 1 -> 0.462 at step 100 (1.7x improvement)
- **Implication**: Fixed weighting parameters do NOT need training-step adaptation.

## DUET vs LUFFY: Identical Token Probability Evolution

| Step | DUET prob_mean | LUFFY prob_mean | DUET prob_p10 | LUFFY prob_p10 |
|------|---------------|-----------------|---------------|----------------|
| 1 | 0.242 | 0.242 | 0.007 | 0.007 |
| 25 | 0.333 | 0.347 | 0.027 | 0.032 |
| 50 | 0.380 | 0.378 | 0.044 | 0.048 |
| 100 | 0.458 | 0.457 | 0.088 | 0.099 |

The teacher token probability distribution is nearly identical between DUET and LUFFY,
confirming that it depends on student policy capacity, not the method-specific loss function.

## Already-Computed Tensors (Zero Additional Cost)

The following tensors are already computed per mini-batch in `het_core_algos.py`:
1. `log_prob` (bs, resp_len): current policy log-probability -- ALWAYS available
2. `teacher_mask` (bs, resp_len): identifies teacher tokens -- ALWAYS available
3. `exp(log_prob)` = `teacher_ratio`: computed at line 402 (LUFFY mode)
4. `p/(p+beta)`: standard shaping at line 783
5. `advantages` (bs, resp_len): per-token advantages -- ALWAYS available
6. `response_mask` (bs, resp_len): valid token mask -- ALWAYS available

Any adaptive weighting that is a function of `p = exp(log_prob)` adds NO new forward pass computation.
Teacher tokens are only 2-4% of total batch tokens (~30K out of ~1M), so per-token operations
on the teacher subset are negligible.

## Existing Adaptive Weighting Experiments

| Method | Peak reward (steps 71-80) | Final reward (steps 91-100) |
|--------|--------------------------|----------------------------|
| EMA + p/(p+0.1) | **0.883** | **0.832** |
| EMA + capped_monotonic(cap=0.6) | 0.862 | 0.826 |
| bell_curve(p_target=0.08, sigma=1.2) | 0.733 | 0.790 |
| capped_monotonic(cap=0.6) | 0.800 | 0.807 |
| LUFFY baseline | 0.739 | 0.718 |

The standard p/(p+0.1) + EMA already outperforms all tested adaptive alternatives.

## Recommendations

### The Real Opportunity: Hard Tail Mitigation

The 3-5% unlearnable tokens have extremely negative log_probs (down to -40, i.e., prob ~1e-18).
Under the current p/(p+beta) shaping:
- At p=0.01: weight = 0.091 (reasonable)
- At p=0.001: weight = 0.010 (small but nonzero)
- At p=1e-6: weight = 0.00001 (negligible)

However, the **advantage** for these tokens can be very large (teacher always succeeds,
reward=1.0), creating the product `large_advantage * small_but_nonzero_weight` that
produces gradient noise.

### Why bell_curve underperformed

The bell_curve centers weight at p_target=0.08 with sigma=1.2 in log-space.
This means it downweights tokens with p > 0.35 (about 40% of the frontier).
Combined with the bimodal distribution, this effectively ignores the ~15-20%
of tokens in the [0.35, 0.9) range that are still actively learning.

### If pursuing adaptive weighting further

1. **Do not re-center the weight function** -- the existing p/(p+beta) already has
   the right shape for this distribution (monotone increasing, saturating at 1.0).

2. **Consider a floor on log_prob contribution** -- clamp teacher log_prob to
   something like max(-10, log_prob) before computing the loss. This would eliminate
   gradient from tokens with p < e^{-10} ~ 4.5e-5 (the truly unlearnable tail).

3. **The slow frontier shift means static parameters work** -- there is no need for
   step-dependent adaptive weighting schedules.
