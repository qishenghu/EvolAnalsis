# WebShop Trajectory Diagnosis: Why DUET Underperforms LUFFY

## Executive Summary

DUET underperforms LUFFY on WebShop (validation mean reward: 0.7251 vs 0.7528; perfect scores: 32.5% vs 49.5%) despite outperforming it on ALFWorld. The root cause is **State Channel (SC) is completely dead on WebShop** — 0% observation coverage across all 100 training steps. This eliminates DUET's key advantage, leaving only DR3 (Action Channel), which adds complexity without compensating benefit.

---

## 1. Reward Distribution Analysis

WebShop has **continuous rewards** [−0.1, 1.0], fundamentally different from ALFWorld's binary {0, 1}.

### Training Reward Trajectories (on-policy only)

| Step | DUET mean | LUFFY mean | GRPO mean | DUET r=1 | LUFFY r=1 | GRPO r=1 |
|------|-----------|------------|-----------|----------|-----------|----------|
| 1    | 0.2505    | 0.1569     | 0.1788    | 1.8%     | 14.1%     | 1.6%     |
| 10   | 0.0913    | 0.0943     | 0.1042    | 0%       | 9.4%      | 0%       |
| 25   | 0.5533    | 0.4343     | 0.4873    | 25.0%    | 14.1%     | 0%       |
| 50   | 0.7401    | 0.6467     | 0.6015    | 37.5%    | 18.8%     | 0%       |
| 75   | 0.6359    | 0.8463     | 0.2179    | 10.2%    | 51.6%     | 0%       |
| 100  | 0.7758    | 0.7903     | 0.2977    | 41.4%    | 51.7%     | 0%       |

**Key observations:**
- Both DUET and LUFFY vastly outperform GRPO (which collapses at step 75)
- LUFFY surpasses DUET after step 50, especially in perfect scores (r=1.0)
- GRPO NEVER achieves r=1.0 on WebShop — continuous reward makes GRPO advantage normalization less effective
- Massive partial reward regime: typically 40-70% of samples have reward in [0.5, 1.0)

### Partial Reward Histogram at Step 100 (on-policy)

| Reward Range | DUET | LUFFY | GRPO |
|-------------|------|-------|------|
| [-0.1, 0)  | 1.7% | 0%    | 28.1% |
| [0, 0.1)   | 0%   | 1.7%  | 20.3% |
| [0.3, 0.5) | 12.1% | 10.3% | 14.1% |
| [0.5, 0.6) | 24.1% | 24.1% | 3.1% |
| [0.8, 0.9) | 13.8% | 1.7%  | 4.7% |
| [1.0, 1.1) | 41.4% | 51.7% | 0% |

LUFFY has a stronger peak at r=1.0. DUET has more samples in the high-partial range [0.8, 0.9) — close but not perfect.

### Validation Results (Step 100)

| Method | Mean Reward | r=0 | r in (0,0.5) | r in [0.5,1) | r=1.0 |
|--------|------------|-----|-------------|-------------|-------|
| DUET   | 0.7251     | 4.5% | 12.5%      | 49.5%       | 32.5% |
| LUFFY  | 0.7528     | 3.5% | 10.0%      | 34.5%       | 49.5% |

LUFFY has **17 percentage points more perfect scores** on validation.

---

## 2. State Channel (SC) Effectiveness — THE SMOKING GUN

### SC Coverage on WebShop: ZERO

| Step | sc_progress | sc_bonus | sc_coverage | sc_matched_states | step_deltas non-zero |
|------|-------------|----------|-------------|-------------------|---------------------|
| 1    | 0.000000    | 0.000000 | 0.000000    | 0.00              | 0/331 (0.0%)        |
| 10   | 0.000000    | 0.000000 | 0.000000    | 0.00              | 0/606 (0.0%)        |
| 25   | 0.000000    | 0.000000 | 0.000000    | 0.00              | 0/161 (0.0%)        |
| 50   | 0.000000    | 0.000000 | 0.000000    | 0.00              | 0/285 (0.0%)        |
| 100  | 0.000000    | 0.000000 | 0.000000    | 0.00              | 0/120 (0.0%)        |

**SC is completely inert on WebShop.** Zero matched states, zero bonus, zero progress, zero step deltas across ALL 100 training steps for ALL on-policy trajectories.

Meanwhile, teacher samples DO match (sc_coverage=1.0, sc_matched=7, sc_progress=0.71) because the progress map is built from teacher trajectories — but `exclude_teacher: true` correctly prevents these from receiving SC bonus.

### Why SC Fails on WebShop

**Root cause: Non-deterministic search results.** WebShop's search engine returns products in different orders across sessions. Even for the SAME task with the SAME search query:

```
Teacher obs:  "...[SEP] $43.59 [SEP] B09NDS8F4V [SEP] AODONG..."
On-policy obs: "...[SEP] $43.59 [SEP] B09QW2HQRK [SEP] CandyM..."
```

The first 526 characters match, then product IDs diverge. Since SC uses **exact hash matching**, a single character difference = no match.

### State Space Comparison

| Metric | ALFWorld | WebShop |
|--------|----------|---------|
| Teacher trajectories | 19,497 | 26,178 |
| Unique tasks | 2,348 | 5,691 |
| Total observations | 219,492 | 191,951 |
| Unique observations | 12,198 | 43,799 |
| **Obs reuse rate** | **17.99x** | **4.38x** |
| Teacher-teacher overlap (same task) | High | ~4/6 (67%) |
| Teacher-onpolicy overlap | 31% coverage | **0% coverage** |

ALFWorld rooms are deterministic — "The fridge 1 is closed" is always "The fridge 1 is closed". WebShop product pages contain dynamic product IDs, prices, and descriptions that change between sessions.

### SC on ALFWorld (for reference)

ALFWorld achieves 31% on-policy coverage at step 50 (sc_matched=9/29 observations, sc_progress=0.194). This provides meaningful dense reward shaping that helps break reward sparsity. On WebShop, this mechanism contributes nothing.

---

## 3. DR3 (Action Channel) Behavior

DR3 metrics are not stored in batch_diag files (logged only to wandb). However, from trajectory data:

### Teacher Advantage Dynamics

| Step | DUET teacher adv mean | LUFFY teacher adv mean |
|------|----------------------|----------------------|
| 1    | 2.02                 | 5.44                 |
| 10   | 6.41                 | 4.34                 |
| 25   | **40,549.52**        | **40,550.19**        |
| 50   | 0.74                 | 90,449.91            |
| 75   | 1.03                 | 3.62                 |
| 100  | 0.69                 | 2.95                 |

**Both methods suffer from teacher advantage explosions** at step 25. Per-trajectory analysis shows one teacher sample with adv_mean=26,768 — a single extreme outlier dominates the batch mean.

DR3's w_hat correction in DUET helps stabilize this (DUET recovers to 0.74 by step 50, while LUFFY still has 90,449). This suggests DR3 IS providing some stabilization benefit, but it's not enough to overcome the loss of SC.

### What DUET becomes on WebShop

Without SC, DUET = LUFFY + DR3. The question is whether DR3 adds net value:
- **DR3 benefit**: Stabilizes teacher advantage computation, provides principled teacher fade-out
- **DR3 cost**: Additional complexity, discriminator training overhead, potential to suppress useful teacher signal too early

The data suggests DR3 provides marginal stabilization but doesn't compensate for the complete loss of SC's dense reward shaping.

---

## 4. Teacher Trajectory Analysis

### Teacher Data Quality

| Metric | WebShop | ALFWorld |
|--------|---------|----------|
| Trajectories | 26,178 | 19,497 |
| Mean reward | 1.0000 | 1.0000 |
| All r=1.0 | Yes | Yes |
| Mean messages | 17.7 | 26.5 |
| Std messages | 3.4 | 14.6 |

Both use high-quality filtered teacher data (all r=1.0). WebShop teachers are shorter (17.7 msgs vs 26.5) and more consistent (std 3.4 vs 14.6).

### Teacher Mix Ratio

Both DUET and LUFFY maintain ~9-12.5% teacher sample ratio throughout training, consistent with `n_teacher_rollouts_per_task: 1` and `n: 8` (rollouts per task).

### Teacher-to-On-Policy Reward Gap Over Training

| Step | DUET gap | LUFFY gap |
|------|----------|-----------|
| 1    | 0.74     | 0.81      |
| 10   | 0.95     | 0.95      |
| 25   | 0.45     | 0.57      |
| 50   | 0.26     | 0.35      |
| 75   | 0.33     | 0.14      |
| 100  | 0.18     | 0.18      |

Both methods converge to similar teacher-onpolicy gaps by step 100 (~0.18). LUFFY closes the gap faster after step 50.

---

## 5. Response Pattern Analysis

### Action Distributions at Step 100

| Metric | DUET | LUFFY | GRPO |
|--------|------|-------|------|
| Mean response length (chars) | 714 | 502 | 2,251 |
| Actions/trajectory | 6.1 | 6.6 | 5.9 |
| search actions | 70 | 70 | 84 |
| click actions | 259 | 286 | 218 |
| buy actions | 64 | 64 | 74 |

- LUFFY produces **shorter responses** (502 vs 714 chars) with **more click actions** (286 vs 259) — more efficient navigation
- GRPO produces very long responses (2,251 chars) — likely repetitive or unfocused
- Both DUET and LUFFY complete purchases at similar rates (64 buy actions each)
- LUFFY's higher click count suggests more exploration of product options before buying

---

## 6. Failure Analysis

### Failure Categories (Validation Step 100)

| Category | DUET | LUFFY |
|----------|------|-------|
| success (r=1.0) | 67 | 104 |
| high_partial (r in [0.5,1)) | 99 | 69 |
| low_partial (r in (0,0.5)) | 25 | 20 |
| bought_wrong_product (r=0, did buy) | 8 | 7 |
| browsed_no_buy (r=0, no purchase) | 1 | 0 |

**Both methods have the same failure MODE** — "bought_wrong_product" (right category, wrong attributes/options). The key difference is DUET has **more high-partial** results (99 vs 69) while LUFFY converts more to **perfect** (104 vs 67).

### Typical Failure Examples

**DUET failures:**
- Bought shoes without selecting correct size
- Bought shirts without matching fit type or color
- Bought candles without verifying all specifications

**LUFFY failures:**
- Nearly identical failure patterns — same types of attribute mismatches
- Slightly fewer total failures (7 vs 9 zero-reward)

The failures are qualitatively similar — the difference is LUFFY is slightly better at attribute matching, likely due to better credit assignment from policy shaping.

---

## 7. Conclusions and Hypotheses

### Primary Finding: SC is Dead on WebShop

DUET's State Channel (SC) provides **zero contribution** on WebShop due to hash-based observation matching failing on a non-deterministic state space. This is DUET's primary mechanism for dense reward shaping, and its absence removes DUET's key advantage over LUFFY.

### Why LUFFY Outperforms DUET on WebShop

1. **LUFFY's advantage is simpler**: Policy shaping (`π/π_β`) directly regularizes toward teacher behavior without requiring state matching
2. **DR3 adds complexity without SC payoff**: On ALFWorld, DR3 + SC work together (DR3 fades teachers while SC provides growing dense signal). Without SC, DR3 only suppresses teacher influence without a compensating dense reward mechanism
3. **WebShop's continuous rewards partially solve reward sparsity**: Unlike ALFWorld (binary {0,1}), WebShop provides continuous feedback [0,1]. SC's dense reward shaping is less critical when the environment already provides graded rewards
4. **DUET has more high-partial results**: DUET struggles to convert partial matches (r≈0.8) into perfect ones (r=1.0), suggesting DR3 may suppress teacher signal on the fine-grained attribute selection that matters for the last 20% of reward

### Why DUET Still Beats GRPO

Even without SC, DUET (= LUFFY + DR3 effectively) benefits from teacher trajectory mixing and teacher baseline separation. GRPO collapses entirely on WebShop (0.30 reward at step 100, 0% perfect scores).

### Recommendations

1. **For WebShop**: Consider disabling SC and running DUET with DR3 only to reduce overhead
2. **For the paper**: Present WebShop as a case study for when SC's hash-based matching fails, motivating future work on approximate/embedding-based state matching
3. **Potential fix**: Implement embedding-based state matching (e.g., sentence embeddings of observations) instead of exact hash for environments with non-deterministic observations
4. **Advantage explosion**: Investigate the teacher advantage explosion at step 25 (adv=40,549) — this affects both methods but may interact differently with DR3

---

## Raw Data Summary

- Analysis date: 2026-04-01
- DUET config: `config/duet_paper_experiments_configs/webshop/webshop_3b_duet.yaml`
- Training steps: 100
- Batch size: 64 trajectories/step (8 tasks × 8 rollouts, minus teacher swaps)
- Teacher source: `webshop_qwen72b_filtered.pkl` (26,178 trajectories, all r=1.0)
- SC settings: `enable: true, beta: 0.2, match_mode: hash, exclude_teacher: true`
- DR3 settings: `enable: true, apply_to: teacher_no_logprob, feature_mode: v3_aug`
