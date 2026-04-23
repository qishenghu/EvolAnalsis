# User Return Status — 2026-04-22

**Launched**: 02:29
**Orchestrator**: `run_duet_round7_v39.sh` (PID 2528884)

## Running now

**Round 7: v39 (disc_acc adaptive μ)** on GPU 0-3
1. v39 WebShop 1.5B → expected done ~05:30
2. v39 ALFWorld 1.5B → expected done ~10:30

## What v39 does

Replaces v24's hand-tuned μ schedule (μ=0.3→0.05 over 25 steps) with:
```python
d = EMA(dr3/disc_acc, alpha=0.2)
μ = clamp(0.3 · max(0, (1-d)/(1-0.5)), 0.05, 0.30)
```
**Intuition**: disc_acc=0.5 (indistinguishable) → μ=0.3; disc_acc=1.0 → μ=0.05.

**Offline validation**: v39's rule applied to v24's recorded disc_acc trace produces μ={0.30, 0.18, 0.08, 0.05}, near-byte-identical to v24's hand-tuned {0.30, 0.20, 0.07, 0.05}. r=0.97.

## 8-hour work completed (autonomous)

1. ✅ **Two agents diagnosed v37/v38 failures**:
   - v37 V_A failed because GRPO normalization forces std≈1 (structural, not calibration)
   - v38 SPW failed because token-mean aggregation averages out rare-token signal (π_θ mean≈0.82, multiplier ≈0.18 constant)
2. ✅ **Agent team found correct adaptive signal**: `dr3/disc_acc`
   - Cross-variant correlation analysis across 9 candidate signals
   - disc_acc wins on r (0.97) + knee position (step 25) + ALFWorld self-adjust
3. ✅ **Implemented v39 in `het_actor.py:1757-1828`** (~35 LOC)
4. ✅ **Implemented v40 NLL backup in same block** (~25 LOC)
5. ✅ **Created v39 configs** WebShop + ALFWorld
6. ✅ **Created v40 backup config** (NLL-based, ready if disc_acc fails)
7. ✅ **Smoke test** passes (16/16)

## Resume instructions for user

### Check v39 results first
```bash
# v39 WebShop Val
grep 'val-summary/webshop/reward_mean_all' logs/webshop_qwen1.5b_duet_v39.log | tail -2
# v39 ALFWorld Val
grep 'val-summary/alfworld/reward_mean_all' logs/alfworld_qwen1.5b_duet_v39.log | tail -2

# μ trajectory (should decay from 0.3 to 0.05 automatically)
grep 'chord/mu_adaptive_gated\|chord/disc_acc_ema' logs/webshop_qwen1.5b_duet_v39.log | head -20
```

### Decision tree

| v39 WebShop | v39 ALFWorld | Next step |
|---|---|---|
| ≥ 0.65 | ≥ 32.5% | **🏆 WIN: adaptive schedule generalizes cross-env → paper-ready** |
| ≥ 0.65 | < 32.5% | Partial: adaptive WebShop works but over-retires on ALFWorld. Try `chord_mu_d_floor: 0.3` (longer μ_max regime) |
| 0.55-0.65 | — | Mixed: adaptive close but not matching v24. Try v40 (NLL-based, backup ready) |
| < 0.55 | — | v39 fails. Pre-ready v40 NLL config: `nohup python launcher.py --conf config/.../webshop_qwen1.5b_duet_v40.yaml ...` |

### Key baselines to compare against

| Variant | WebShop 1.5B | ALFWorld 1.5B | μ |
|---|---:|---:|---|
| v1 (no BC) | 0.549 | 32.5% | - |
| v12 (DR3+SC, stab) | 0.431 | - | - |
| v24 (hand-tuned decay) | **0.678** | 30.5% (regress) | 4 params |
| v36 (const μ=0.05) | 0.389 | - | 1 param |
| v37 (V_A adaptive, BUG) | 0.532 | - | bug |
| v38 (SPW) | 0.474 | - | 0 params |
| **v39 (disc_acc adaptive)** | **???** | **???** | **2 params** |

## Critical files

Reports:
- `analysis_reports/adaptive_signal_discovery.md` — empirical signal ranking
- `analysis_reports/duet_third_pass_adaptive.md` — theory post-mortem + alternative
- `analysis_reports/v24_alfworld_dynamics_analysis.md` — why v24-ALFWorld regresses
- `analysis_reports/v24_alfworld_trajectory_diff.md` — template overfit case studies
- `analysis_reports/v39_run_plan.md` — full v39 experimental plan
- **This file** — resume state

Code:
- `agentevolver/module/exp_manager/het_actor.py:1757-1828` — adaptive μ branches (disc_acc, nll, va)

Configs:
- `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v39.yaml` — disc_acc adaptive
- `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39.yaml` — disc_acc adaptive
- `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v40.yaml` — NLL adaptive backup

## If v39 succeeds: next experiments to queue

1. v39 on WebShop 3B (scaling)
2. v39 on ALFWorld 3B
3. ablation: v39 with d_floor in {0.3, 0.5, 0.7} (robustness)

## If v39 fails: fallback plan

1. Launch v40 (NLL-based, ~30 LOC change in het_actor.py already applied)
2. If v40 also fails: accept v24 as empirical recipe, write paper with adaptive in future work

## Health monitoring during run

Key metrics to watch:
- `chord/mu_mode:3.0` should appear (indicates disc_acc mode active)
- `chord/disc_acc_ema` should rise from ~0.5 to ~1.0 over training
- `chord/mu_adaptive_gated` should decrease from ~1.0 to ~0.0
- Computed μ (chord/mu in logs) should follow 0.3 → 0.05 shape

If any of these looks wrong at step 10-20 (e.g., μ stays at 0.3 constant like v37's bug), abort and debug V_A_target... wait, not V_A. For disc_acc: check that dr3/disc_acc is actually being read correctly.
