# Round 7: v39 (Adaptive μ via disc_acc) — Run Plan + Expected Results

**Started**: 2026-04-22 02:29
**Expected completion**: ~10:30 (8h)
**User returning**: ~10:30

## What's running

**Sequential orchestrator** `run_duet_round7_v39.sh`:
1. v39 on WebShop 1.5B (~3h) → expected done ~05:30
2. v39 on ALFWorld 1.5B (~5h) → expected done ~10:30

## v39 mechanism

Replaces v24's hand-tuned μ schedule (μ=0.3→0.05 over 25 steps) with:
```python
d = EMA(dr3/disc_acc, alpha=0.2)
μ = clamp(μ_max × max(0, (1 − d) / (1 − d_floor)), μ_min, μ_max)
# μ_max=0.3, μ_min=0.05, d_floor=0.5
```

**Intuition**: disc_acc=0.5 (indistinguishable) → μ=μ_max (need teacher); disc_acc=1.0 (fully separable, policy learned) → μ=μ_min.

## Signal validation (pre-experiment, from `adaptive_signal_discovery.md`)

**Offline reproduction of v24 schedule from v24's own disc_acc trace**:
| step | disc_acc rule μ | v24 actual μ |
|---|---:|---:|
| 1 | 0.30 | 0.30 |
| 10 | 0.18 | 0.20 |
| 25 | 0.08 | 0.07 |
| 50+ | 0.05 | 0.05 |

Correlation r = **0.97**, MAE = **0.007**. Nearly byte-perfect match.

## Success criteria

| Env | v24 result | v39 target | Interpretation |
|---|---:|---:|---|
| WebShop 1.5B | 0.678 | ≥ 0.65 | match v24 → narrative upgrade works |
| ALFWorld 1.5B | 30.5% (regression) | **≥ 32.5%** (DUET-v1 baseline) | fix v24's -2pp regression |

## Failure modes + fallbacks

If v39 WebShop < 0.60:
- disc_acc signal hypothesis is wrong
- Next: v40 with NLL-based (theory-researcher's alternative, `chord/sft_loss` signal)

If v39 WebShop ≥ 0.65 but v39 ALFWorld < 32.5%:
- disc_acc rule retires BC too fast on ALFWorld (we wanted early retire but may have over-shot)
- Tune d_floor: lower to 0.3 or 0.4 delays retirement

If v39 ALFWorld ≥ 32.5%:
- **Full narrative win**: adaptive μ replaces hand-tuned schedule + generalizes to second env
- Paper can claim: "closed-form adaptive schedule subsumes CHORD's empirical decay"

## Prior results for comparison

| Variant | WebShop 1.5B | ALFWorld 1.5B | Signal |
|---|---:|---:|---|
| v1 (DUET no BC) | 0.549 | 32.5% (V@100) | - |
| v12 (DR3+SC no BC, stab) | 0.431 | - | - |
| **v24 (hand-tuned BC)** | **0.678** | 30.5% (regression from 33.5% @V50) | 4 hyperparams |
| v36 (const μ=0.05) | 0.389 | - | 1 hyperparam |
| v37 (V_A adaptive, BUG) | 0.532 (de-facto const μ=0.3) | - | - |
| v38 (SPW multiplier) | 0.474 | - | 0 hyperparams |
| **v39 (disc_acc adaptive)** | **RUNNING** | **RUNNING** | **2 hyperparams** |

## Code changes

Modified `agentevolver/module/exp_manager/het_actor.py:1757-1828`:
- Added `chord_mu_adaptive_mode: "disc_acc"` branch
- Reads `dr3_metrics["dr3/disc_acc"]`, EMA it, apply clamp-linear rule
- Metrics logged: `chord/disc_acc_ema`, `chord/mu_adaptive_gated`, `chord/mu_mode=3.0`
- Legacy "va" mode (v37) preserved for back-compat

## Monitoring checklist for resume

When I resume, check:
1. `grep 'val-summary/webshop/reward_mean_all' logs/webshop_qwen1.5b_duet_v39.log | tail -2`
2. `grep 'val-summary/alfworld' logs/alfworld_qwen1.5b_duet_v39.log | tail -2`
3. μ trajectory: `grep 'chord/mu_adaptive_gated' logs/*_v39.log | head -20`
4. disc_acc EMA: `grep 'chord/disc_acc_ema' logs/*_v39.log | head -20`
5. Sanity: grad_norm, kl_loss, response_length stable

## Reports to read if v39 succeeds

- `adaptive_signal_discovery.md` — why disc_acc
- `duet_third_pass_adaptive.md` — why v37/v38 failed
- `v24_alfworld_dynamics_analysis.md` — v24-ALFWorld regression diagnosis
