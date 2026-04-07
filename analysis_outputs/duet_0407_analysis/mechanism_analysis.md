# DUET 0407 Mechanism Analysis: Why DUET vs LUFFY on WebShop

**Date**: 2026-04-05
**Analyst**: algo-engineer

---

## Executive Summary

**The premise "DUET fails to beat LUFFY on WebShop" is INCORRECT for the best DUET variant.**

| Experiment | Val@50 | Val@100 | Delta vs LUFFY@100 |
|-----------|--------|---------|---------------------|
| hybrid_0405 (DR3+SC) | 0.5916 | **0.7656** | **+1.3pp** |
| duet_0405 | 0.6680 | **0.7613** | **+0.85pp** |
| **luffy** | **0.5086** | **0.7528** | baseline |
| 0407_sc | 0.5911 | 0.7391 | -1.4pp |
| luffy_sc_0405 | 0.5167 | 0.7087 | -4.4pp |
| 0407_alpha | 0.5374 | 0.5218 | -23.1pp |
| onpolicy | 0.2759 | 0.4019 | -35.1pp |
| chord | 0.2667 | -0.1000 | catastrophic |
| luffy_sc (original) | 0.5175 | 0.2211 | catastrophic |

**Hybrid 0405 is the best WebShop run at 0.7656, beating LUFFY by +1.3pp.**
The 0407 experiments were attempted improvements that REGRESSED performance.

---

## 1. Config Diff: What Changed in 0407

### 0407_SC vs Hybrid 0405 (the winning config)

| Parameter | Hybrid 0405 | 0407 SC | Change Rationale |
|-----------|-------------|---------|------------------|
| `state_channel.beta` | 0.2 | 0.15 | "Compensate for progress_agg=last giving ~50% higher P(tau)" |
| `state_channel.progress_agg` | (default=mean) | last | "last has 0.82-0.96 correlation vs mean's 0.77-0.83" |
| `state_channel.step_level.enable` | true (eta=0.05) | **false** | "step_level is broken in multi-turn WebShop (sliding_window)" |

### 0407_Alpha vs Hybrid 0405

| Parameter | Hybrid 0405 | 0407 Alpha |
|-----------|-------------|------------|
| `dr3.alpha_prior` | (none, uses sync_batch_ema) | **0.3** |
| All else | identical | identical |

**Diagnosis**: Both 0407 changes HURT. 0407_SC's SC modifications regressed by -2.65pp. 0407_Alpha's alpha_prior=0.3 was catastrophic (-24pp).

---

## 2. Code Path Trace: DUET Hybrid vs LUFFY

### LUFFY Code Path (`use_dr3: false, teacher_policy_shaping_enable: true`)
File: `het_actor.py:1750-1830`
```
het_compute_teacher_aware_loss(
    teacher_use_log_prob=False,          # ratio = exp(log_prob), NOT exp(log_prob - old_log_prob)
    teacher_policy_shaping_enable=True,  # applies p/(p+β) shaping
    teacher_policy_shaping_beta=0.1,
    teacher_use_clip=False,
    teacher_loss_scale=None,             # NO trajectory-level scaling
)
```

Effective teacher gradient per token:
```
L_teacher = -advantage * p/(p+β)     where p = exp(log_prob), β=0.1
```

### DUET Hybrid Code Path (`use_dr3: true, dr3.use_policy_shaping: true`)
File: `het_actor.py:1340-1559`

1. DR3 discriminator observes features, trains, produces `w_hat` (per-trajectory scalar)
2. DR3 modifies `old_log_prob`: `old_lp_new[teacher] = log_prob.detach() - log(w_hat)`
3. **CRITICAL**: This old_log_prob modification is DEAD CODE because:
   - In `het_compute_teacher_aware_loss` with `teacher_use_log_prob=False` (line 396-399):
   - `teacher_ratio = torch.exp(log_prob)` — **ignores old_log_prob entirely**
4. DR3 computes hybrid scale: `_hybrid_scale = w_hat.unsqueeze(-1)`
5. Calls `het_compute_teacher_aware_loss` with `teacher_loss_scale = _hybrid_scale`

```python
# het_actor.py:1532-1559  — Hybrid mode
_hybrid_w = w_hat.unsqueeze(-1)   # (bs, 1)
_hybrid_scale = _hybrid_w         # or teacher_loss_scale * _hybrid_w if exists

het_compute_teacher_aware_loss(
    teacher_use_log_prob=False,
    teacher_policy_shaping_enable=True,
    teacher_policy_shaping_beta=0.1,    # same β as LUFFY
    teacher_use_clip=False,
    teacher_loss_scale=_hybrid_scale,   # THIS IS THE ONLY DIFFERENCE
)
```

Then in `het_core_algos.py:562-566`:
```python
if teacher_loss_scale is not None:
    teacher_ratio = teacher_ratio * teacher_loss_scale   # scales by w_hat
```

Effective teacher gradient per token:
```
L_teacher = -advantage * p/(p+β) * w_hat    where w_hat ∈ [0.01, 5.0]
```

### The ONLY Mechanism Difference

| Aspect | LUFFY | DUET Hybrid |
|--------|-------|-------------|
| On-policy loss | Standard GRPO | Standard GRPO + SC bonus in reward |
| Teacher token-level | `p/(p+β)` | `p/(p+β)` (identical) |
| Teacher trajectory-level | 1.0 (uniform) | w_hat (DR3 discriminator) |
| Teacher gradient formula | `-adv * p/(p+β)` | `-adv * p/(p+β) * w_hat` |
| DR3 old_log_prob modification | N/A | **DEAD CODE** (ignored by teacher_use_log_prob=False) |

**The DR3 old_log_prob correction at lines 1482-1525 has ZERO effect in Hybrid mode.**
It only matters when `teacher_use_log_prob=True`, which these configs don't use.

### Warmup Behavior (first 10 steps)
During `apply_warmup_steps: 10` and before `apply_min_buf_size: 512` is met, DUET falls back to LUFFY-style loss BUT with `teacher_policy_shaping_enable=self.config.get(...)`. Since the 0407 configs set `teacher_policy_shaping_enable: false` at the actor level, the warmup fallback uses **NO policy shaping** (raw `exp(log_prob)` as teacher ratio). This is likely harmless since it only affects ~10 steps and raw probabilities are very small for teacher tokens.

---

## 3. Trajectory Data Comparison: 0407_SC vs LUFFY

### Training Reward (batch_diag)

**0407_SC outperforms LUFFY on training reward at nearly every step:**

| Steps | 0407_SC (10-step avg) | LUFFY (10-step avg) | Delta |
|-------|----------------------|---------------------|-------|
| 1-10 | 0.254 | 0.237 | +0.017 |
| 11-20 | 0.517 | 0.447 | +0.070 |
| 21-30 | 0.631 | 0.525 | +0.106 |
| 31-40 | 0.651 | 0.579 | +0.072 |
| 41-50 | 0.704 | 0.530 | +0.175 |
| 51-60 | 0.749 | 0.574 | +0.176 |
| 61-70 | 0.844 | 0.734 | +0.111 |
| 71-80 | 0.825 | 0.767 | +0.058 |
| 81-90 | 0.836 | 0.760 | +0.077 |
| 91-100 | 0.794 | 0.670 | +0.125 |

### On-policy Advantage Positive Ratio

0407_SC: 0.49 → 0.74 (step 1→100)
LUFFY: 0.39 → 0.36 (step 1→100)

**SC bonus inflates on-policy advantages to be consistently positive.** LUFFY maintains ~40% positive ratio (healthy for exploration). 0407_SC pushes to 70-98% positive, which may cause over-exploitation.

### LUFFY's Advantage Explosion (steps 30-50)

LUFFY shows catastrophic `adv_teacher_token_mean` explosion:
- Step 30: 52,585
- Step 40: 93,936
- Step 50: 90,450

Then crashes back to normal by step 60 (1.87). This is a severe numerical instability in LUFFY that somehow self-corrects. 0407_SC stays stable at 0.9-4.3 throughout.

### Teacher-OnPolicy Reward Gap

| Step | 0407_SC | LUFFY |
|------|---------|-------|
| 1 | 0.77 | 0.81 |
| 30 | 0.30 | 0.39 |
| 60 | 0.22 | 0.41 |
| 100 | 0.06 | 0.18 |

0407_SC closes the gap 3x faster, reaching near-zero by step 100 (teacher reward = on-policy reward). This suggests the model is learning effectively on the training tasks.

---

## 4. The Fundamental Question: Why Val@100 Regresses

### 0407_SC vs Hybrid 0405

| Metric | Hybrid 0405 | 0407_SC | Interpretation |
|--------|-------------|---------|----------------|
| Val@100 | **0.7656** | 0.7391 | -2.65pp regression |
| SC beta | 0.2 | 0.15 | Weaker SC bonus |
| progress_agg | mean | last | Different aggregation |
| step_level | enabled (η=0.05) | disabled | Removed step deltas |

The 0407_SC "improvements" to SC actually hurt. The step_level disabling removed a signal that was apparently contributing positively in Hybrid 0405, despite the analysis claiming it was broken.

### Why DUET (Hybrid 0405) Beats LUFFY

The +1.3pp advantage of Hybrid 0405 over LUFFY comes from:

1. **DR3 w_hat trajectory scaling**: Provides adaptive curriculum — upweights informative teacher trajectories, downweights already-learned ones. This is a more nuanced signal than LUFFY's uniform treatment of all teacher samples.

2. **SC bonus**: Dense on-policy reward signal that helps on-policy samples learn faster (as seen in training reward curves). However, SC alone hurts LUFFY (-4.4pp for luffy_sc_0405), suggesting SC needs DR3's corrective mechanism to be beneficial.

3. **Synergy**: SC inflates on-policy advantages → policy learns faster. DR3 naturally fades teacher influence as policy improves. Together, these create a self-correcting loop. Without DR3, SC's advantage inflation causes instability (as seen in luffy_sc collapse at 0.2211).

---

## 5. What Would It Take to EXACTLY Match LUFFY's Mechanism?

Config changes to strip DUET back to pure LUFFY:
```yaml
actor_rollout_ref:
  actor:
    use_dr3: false                          # disable DR3 entirely
    teacher_policy_shaping_enable: true     # enable LUFFY's p/(p+β) shaping
    teacher_policy_shaping_mode: p_div_p_beta
    teacher_policy_shaping_beta: 0.1

exp_manager:
  state_channel:
    enable: false                           # disable SC
  teacher_experience:
    policy_shaping:
      enable: false                         # exp_manager-level shaping (redundant but keep clean)
```

To get **LUFFY + SC** (no DR3):
```yaml
actor_rollout_ref:
  actor:
    use_dr3: false
    teacher_policy_shaping_enable: true
    teacher_policy_shaping_mode: p_div_p_beta
    teacher_policy_shaping_beta: 0.1

exp_manager:
  state_channel:
    enable: true
    # Use Hybrid 0405 SC settings, NOT 0407's
    exclude_teacher: true
    beta: 0.2
    beta_decay: false
    match_mode: attribute_aware
    grpo_decouple: true
    step_level:
      enable: true
      eta: 0.05
```

**WARNING**: Previous experiments show LUFFY+SC performs WORSE than LUFFY alone:
- luffy_sc: 0.2211 (catastrophic)
- luffy_sc_0405: 0.7087 (-4.4pp vs LUFFY)

SC needs DR3's trajectory weighting to work properly. Without DR3, SC's advantage inflation causes the policy to over-exploit training tasks.

---

## 6. Recommendations

### Don't Fix What Isn't Broken
**Hybrid 0405 already beats LUFFY (0.7656 vs 0.7528).** The 0407 "improvements" regressed performance. Recommendation: **revert to Hybrid 0405 SC settings**.

### The Real Bottleneck
The gap between training reward (0.87 at step 100) and validation (0.77) suggests overfitting to training tasks, not a mechanism failure. Focus on:
1. More training tasks (currently 800)
2. Task diversity / curriculum
3. Regularization (KL coefficient)

### If We Want to Push Higher
1. **Don't touch SC settings** — Hybrid 0405's SC config works; 0407's changes hurt
2. **Investigate LUFFY's advantage explosion** at steps 30-50 — understanding why it self-corrects might reveal a robustness mechanism DUET could borrow
3. **Longer training**: Hybrid 0405 may still be improving at step 100
4. **alpha_prior**: 0.3 was too aggressive; if revisiting, try 0.15-0.2 range

### Dead Code Alert
The DR3 `old_log_prob` correction (het_actor.py:1482-1525) has NO EFFECT when:
- `teacher_use_log_prob=False` (all current configs)
- `dr3.use_policy_shaping=True` (Hybrid mode)

It's ~45 lines of dead code in the Hybrid path. The correction only matters for the non-Hybrid DR3 path (`use_policy_shaping=False`), which uses `repo_compute_token_loss` where the ratio IS `exp(log_prob - old_log_prob)`.

---

## Appendix: Full Validation Leaderboard (WebShop 3B)

| Rank | Experiment | Val@100 |
|------|-----------|---------|
| 1 | hybrid_0405 | **0.7656** |
| 2 | duet_0405 | 0.7613 |
| 3 | luffy | 0.7528 |
| 4 | duet_0406_v3 | 0.7445 |
| 5 | duet_0407_sc | 0.7391 |
| 6 | duet_0402 | 0.7353 |
| 7 | duet (original) | 0.7251 |
| 8 | luffy_sc_0405 | 0.7087 |
| 9 | duet_0406_v1 | 0.6819 |
| 10 | duet_0403 | 0.6790 |
| 11 | duet_0404 | 0.6463 |
| 12 | duet_0401 | 0.5649 |
| 13 | duet_0407_alpha | 0.5218 |
| 14 | hybrid (original) | 0.5121 |
| 15 | onpolicy | 0.4019 |
| 16 | luffy_sc | 0.2211 |
| 17 | chord | -0.1000 |
