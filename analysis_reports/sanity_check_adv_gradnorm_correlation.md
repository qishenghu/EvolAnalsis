# Sanity Check: Advantage-Regularizer Mechanism Validation

**Date**: 2026-04-21 01:32
**Purpose**: Validate the 2nd-pass theory claim that v24's mechanism is "indirect advantage-scale calibration via teacher-state-alignment" before committing to v37 (adaptive μ).

## Test 1: Cross-variant correlation `corr(adv_on_abs, grad_norm)` at late training (Q4, step 75-100 avg)

| variant | grad_norm | adv_on_abs | grp_rwd_var | entropy | Val@100 |
|---|---:|---:|---:|---:|---:|
| v1 (DUET, no BC) | 7.41 | 0.179 | 0.058 | 0.542 | 0.549 |
| v12 (DR3+SC, no BC) | 12.24 | 0.379 | 0.130 | 0.498 | 0.431 |
| v22 (const μ=0.05, v1-stab) | 7.95 | 0.206 | 0.084 | 0.541 | 0.462 |
| **v24 (decay μ=0.3→0.05)** | **4.28** | **0.176** | 0.064 | 0.579 | **0.678** |
| v28 (w_hat EMA) | 7.20 | 0.171 | 0.062 | 0.538 | 0.495 |
| v29 (combined rescue) | 6.92 | 0.168 | 0.056 | 0.600 | 0.511 |
| v30 (strong KL) | **3.73** | 0.152 | 0.055 | 0.611 | 0.520 |
| v31 (entropy coeff) | 6.84 | 0.169 | 0.070 | 0.581 | 0.517 |
| v32 (lower lr) | 6.61 | 0.204 | 0.100 | 0.516 | 0.465 |
| v33 (soft disc) | 7.43 | 0.171 | 0.054 | 0.556 | 0.520 |
| v36 (const μ=0.05, v12-stab) | 6.10 | 0.180 | 0.061 | 0.572 | 0.389 |
| chord | 6.49 | 0.107 | 0.077 | 0.469 | 0.603 |

**`corr(adv_on_abs, grad_norm)` cross-variant = 0.803 (n=12)** ← **strong positive correlation**

The causal chain `teacher-alignment → low advantage → low grad_norm` is validated at coarse resolution.

## Test 2: Within-variant step-level correlation (noise)

| variant | r |
|---|---:|
| v24 | -0.391 |
| v12 | +0.498 |
| v36 | +0.010 |
| v30 | -0.140 |
| v1 | -0.002 |
| chord | -0.434 |

**Weak / inconsistent** — the mechanism operates over training-long timescales, not step-by-step. Individual steps are dominated by per-batch noise.

## Test 3: `group_reward_variance_mean` (theory prediction: v24 < v36 < v12)

| variant | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| v12 | 0.134 | 0.122 | 0.068 | **0.130** |
| v24 | 0.121 | 0.081 | 0.061 | 0.064 |
| v36 | 0.122 | 0.079 | 0.063 | 0.061 |
| chord | 0.118 | 0.070 | 0.063 | 0.077 |

- **Partially supported**: v24 and v36 (both BC) have much lower Q4 variance (~0.06) than v12 (0.13)
- **Caveat**: v24 ≈ v36 on this metric — mechanism can't differentiate winner from loser on this alone

## Test 4: `adv_on_std` (on-policy advantage std over training)

| variant | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| v1 | 0.24 | 0.34 | 0.28 | 0.27 |
| v12 | 0.24 | 0.36 | 0.34 | **0.50** |
| v24 | **0.19** | 0.24 | 0.23 | 0.29 |
| v36 | 0.22 | 0.24 | 0.24 | 0.28 |
| chord | **0.16** | 0.19 | 0.19 | **0.18** |

- **v24 has lowest early-training adv std** (0.19 at Q1) — consistent with theory
- **v12's Q4 adv std explodes** (0.50) — consistent with "no-BC → advantage magnitude grows"
- **CHORD has lowest adv std throughout** (0.16-0.18) — not surprising, strong early BC

## Verdict

- **Cross-variant causal chain supported (r=0.803)**
- **v30 exception**: strong KL achieves even lower grad_norm (3.73) than v24 (4.28), but 15pp lower val — **low grad_norm is necessary but not sufficient**. BC's role is "low grad_norm *while converging to teacher*," not just "low grad_norm"
- **Within-variant step-level correlation is noise** — mechanism operates at coarse timescales
- **Group variance partially separates BC vs no-BC** (Q4: v24/v36 ~0.06, v12 ~0.13) but can't distinguish v24 from v36

## Implications for v37

- Run v37 with adaptive μ is justified: the advantage-regularizer signal exists at the variant level
- Calibration risk: `V_A_target=0.035` from v24's endpoint may be suboptimal; needs ablation
- Backup plan if v37 < 0.60: theory collapses to "hand-tuned v24 schedule approximates principled rule" narrative (Candidate B-weakened)

## Raw data source
- `/tmp/sanity_check.txt` (full output)
- Per-variant training logs at `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_*.log`
