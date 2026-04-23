# Round 8 Phase 1 Preflight Audit

**Date:** 2026-04-19
**Scope:** 4 WebShop 1.5B DUET experiments with 4 adaptive-μ modes
**Auditor:** Algorithm engineer (pre-flight review before ~12h sequential run)

---

## 1. Verdict

**GREEN LIGHT — proceed to run.**

All four adaptive-μ branches (disc_acc, nll/linear, ess_ratio/saturating, kl_lagrangian) are syntactically valid, numerically stable under synthetic stress tests, and produce μ values that stay within `[chord_mu_valley, chord_mu_peak]` = `[0.05, 0.30]`. All config keys read in het_actor.py are present in the four target YAMLs.

Zero code fixes applied. Three issues flagged for human awareness (none are blockers).

---

## 2. Files audited

- `agentevolver/module/exp_manager/het_actor.py` lines 1757–1985 (adaptive μ dispatch)
- `agentevolver/module/exp_manager/het_core_algos.py` lines 580–610 (SPW; inactive in Phase 1), 1767–1862 (`compute_chord_sft_loss`)
- `agentevolver/module/exp_manager/dr3_ratio.py` lines 700–900 (disc_acc / ess / disc_trained_steps emission)
- `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_{v43a,v40b,v39b,v41b}.yaml`

---

## 3. Smoke test results

Harness: `tmp_scripts/smoke_test_round8_modes.py` — extracts each branch into a callable function, feeds it synthetic `dr3_metrics` and `sft_loss` scalars matching realistic ranges, asserts type/range/finiteness and state evolution.

| Mode | Test scenarios | Result |
|---|---|---|
| `disc_acc` (v39b, α=0.5) | warmup→acc=0.5→acc=0.9 x10→acc=1.0 | **PASS** — μ: 0.30 → 0.30 → 0.10 → 0.075 (clean decay) |
| `nll/linear` (v40b, slope=0.156) | nll=0→nll=1 x20→nll=5 x30 | **PASS** — μ: 0.05 → 0.206 → 0.30 (clamped) |
| `ess_ratio/saturating` (v41b, pow=0.5) | warmup→anchor=ess 10→drop to 2→rise to 100 | **PASS** — μ: 0.05 → 0.05 → 0.13 → 0.05 |
| `ess_ratio/sigmoid` (additional coverage) | anchor=10→drop to 1 x20 | **PASS** — μ = 0.289 (near peak, correct) |
| `kl_lagrangian` (v43a, η=0.3, ρ=0.9) | cost=1→3 x10→0.5 x30→100 spike→0.01 x100 | **PASS** — μ: 0.30 → 0.30 → 0.05 → 0.30 → 0.05 |

All five tests pass. μ stays in [0.05, 0.30] across all scenarios. No NaN/Inf. State evolves correctly across calls (EMAs update, `_mu_lagrange_state` mutates, anchors capture).

Overflow stress: `math.exp(0.3 * (100 - 0)) ≈ 1.14e13` resolves to `step_mult = 3.08e3` after the cost-budget EMA dampening, which is well below Python float infinity. μ is then clamped to peak=0.3. No overflow in practice.

---

## 4. Checklist coverage

### A. Syntax / imports — PASS
- `ast.parse(het_actor.py)` succeeds (verified).
- `math` module imported at top of `het_actor.py` (line 23, confirmed via AST walk). Only `math.exp` is used (no `math.log`, no division by zero paths).
- All branches reference `dr3_metrics`, `sft_loss`, `chord_mu_peak`, `chord_mu_valley`, `chord_global_step` which are in scope at the dispatch site (verified by context read lines 1193, 1293, 1743, 1755).

### B. Control flow — PASS
- Dispatch chain uses `elif` cascade on `adaptive_mode ∈ {disc_acc, nll, va, ess_ratio, kl_lagrangian}` then `else: scheduler`. Mutually exclusive, always assigns `mu` and `adaptive_metrics`.
- `isinstance(dr3_metrics, dict)` guards on lines 1779, 1891 handle the rare `dr3_metrics = None` fallback from line 1343 (ExceptionHandler in DR3 observe block).
- `adaptive_mode` is lowercased+stripped at line 1766, matching the config strings (all 4 configs use lowercase quoted strings).
- The CHORD block entry condition at line 1709 is `use_chord and has_teacher_data and (ret_dict is None or (dr3_enable and use_chord))` — with DR3 enabled in all 4 configs, this runs the hybrid path after the DR3 loss is computed.

### C. Numerical / stability — PASS (with one design concern, see §5.2)
- `kl_lagrangian`: `math.exp(η·diff)` — η=0.3, diff bounded by [-|NLL_range|, +|NLL_range|] ≈ [-5, 5] in realistic settings, exp in [~0.22, ~4.5]. Extreme spike (cost=100) tested — 3.08e3 mult, μ clamps to peak. Safe.
- `disc_acc` sigmoid branch (not used by v39b which uses linear mapping): `math.exp(d_k · (ema - d_floor))` with d_k=10, diff bounded [-0.5, 0.5], exp in [e^-5, e^5] ≈ [0.007, 148]. Safe.
- `ess_ratio` saturating: `ratio^0.5` with `ratio = max(0, ratio)` — no negative-to-fractional exponent issue. Safe.
- `nll/linear`: `mu_raw = intercept + slope * nll_ema` then clamp. No division, no exp. Safe.
- `nll/ratio` (NOT used by v40b, but shared code): `ratio_cur = max(_nll_now, 1e-3)` if anchor unset, `max(1e-6, _nll_anchor_cur)` always. No div-by-zero.
- EMA seeding: all four modes use `if not hasattr(self, "_foo"): self._foo = current_value` — correctly initialises to first observation, not 0. Verified at lines 1783, 1818, 1834, 1894, 1944, 1954, 1960.
- Clamp order: `max(chord_mu_valley, min(chord_mu_peak, mu_raw))` — valley < peak always (0.05 < 0.30). Correct.
- Floating-point clamp in `kl_lagrangian`: `_new_mu = max(chord_mu_valley, min(chord_mu_peak, _new_mu))` — correct order.

### D. State persistence — PASS
- `_disc_acc_ema`, `_nll_ema`, `_ess_ema`, `_mu_lagrange_state`, `_kl_cost_ema`, `_kl_budget_ema`, `_ess_anchor`, `_nll_anchor` all stored as attributes on `self` (the FSDP actor worker object) via `setattr` under `hasattr` guards. Persists across micro-batches, mini-batches, PPO epochs, and global steps for the lifetime of the Ray actor.
- **Not persisted across checkpoint restart.** Acceptable for Phase 1 since `save_freq=100` but `total_epochs=1` and no resume is planned.
- Per-rank: each FSDP rank maintains its own state. Not synchronised across ranks. Implication discussed in §5.1.

### E. Metric logging — PASS
- `chord/mu` logged (line 1999) as Python float.
- `chord/mu_mode` uses unique floats: 1.0=scheduler, 2.0=va, 3.0=disc_acc, 4.0=nll, 5.0=ess_ratio, 6.0=kl_lagrangian.
- Per-mode diagnostics: `chord/disc_acc_ema`, `chord/nll_ema`, `chord/ess_anchor`, `chord/kl_cost_ema`, etc. all present and typed as float.
- `append_to_dict` accumulates per-micro-batch values into lists; `reduce_metrics` (verl utility) then takes the mean before wandb log. Confirmed behavior.

### F. Config-to-code binding — PASS
Each of the 4 YAMLs only sets keys actually read by the matching branch (cross-checked via grep):

**v43a (kl_lagrangian):** `chord_mu_adaptive: true`, `chord_mu_adaptive_mode: "kl_lagrangian"`, `chord_mu_kl_eta: 0.3`, `chord_mu_kl_budget_rho: 0.9`, `chord_mu_kl_cost_ema_alpha: 0.3`. Matches keys at lines 1947, 1953, 1959. `chord_mu_kl_eps_fixed` absent → correct (moving budget used).

**v40b (nll/linear):** `chord_mu_adaptive_mode: "nll"`, `chord_mu_nll_mapping: "linear"`, `chord_mu_nll_slope: 0.156`, `chord_mu_nll_intercept: 0.05`, `chord_mu_nll_ema_alpha: 0.3`. Matches keys at lines 1817, 1823, 1825, 1826. `chord_mu_nll_target/k/ratio_*` absent → correct (linear branch only uses slope/intercept).

**v39b (disc_acc):** `chord_mu_adaptive_mode: "disc_acc"`, `chord_mu_d_floor: 0.5`, `chord_mu_d_ema_alpha: 0.5`. Matches keys at 1782, 1787. `chord_mu_d_mapping` absent → defaults to "linear" (correct for v39 family).

**v41b (ess_ratio/saturating):** `chord_mu_adaptive_mode: "ess_ratio"`, `chord_mu_ess_mapping: "saturating"`, `chord_mu_ess_saturating_pow: 0.5`, `chord_mu_ess_ema_alpha: 0.2`, `chord_mu_ess_anchor_min_window: 8.0`. Matches keys at 1893, 1899, 1906, 1923. `chord_mu_ess_tau/sigmoid_k/velocity_beta` absent → correct (saturating branch only).

All 4 configs share base settings: `chord_mu_peak=0.3`, `chord_mu_valley=0.05`, `chord_mu_warmup_steps=0`, `chord_mu_decay_steps=25`. Valid ranges. Default `.get()` fallbacks cover any unspecified keys.

### G. Bug risks (from retrospective)
- **SPW (v38, deprecated) code path** (`het_core_algos.py:580-610`): gated on `use_spw_teacher` flag. Verified all 4 configs do NOT set `use_spw_teacher` → SPW code is inert. No accidental near-zero multiplier risk.
- **V_A miscalibration (v37, legacy)**: `va` mode remains in the cascade (lines 1860-1882) for back-compat but NOT enabled by any Phase 1 config.
- **disc_acc=0 warmup bug**: Fix applied in committed-uncommitted diff at lines 1780-1781 — uses `_disc_ready > 0` check, replacing raw 0.0 with 0.5 during no-signal periods. This fix is present in THIS code path. ✓
- **EMA init from 0**: All six EMAs (`_disc_acc_ema`, `_nll_ema`, `_ess_ema`, `_kl_cost_ema`, `_kl_budget_ema`, `_ess_vel_ema`) use `hasattr` guard to init from first observation, never from 0. ✓

---

## 5. Issues flagged for human awareness (not blockers)

### 5.1 Cross-rank μ desynchronization (structural; affects all adaptive modes)

**Description:** Each FSDP rank independently maintains `_disc_acc_ema`, `_nll_ema`, `_ess_ema`, `_mu_lagrange_state`. With 4 GPUs, each rank sees a **different** subset of micro-batches and a different signal value → different μ.

For `disc_acc` specifically the effect is acute: `dr3/disc_acc` is only computed on rank0 (inside `if self.disc_steps_per_call > 0 and self._can_train() and can_optimize`, where `can_optimize = (not self.broadcast_params) or (rank == 0)`, and all 4 Phase 1 configs set `broadcast_params: true`). On non-rank0 workers, `dr3/disc_acc = 0.0` and `dr3/disc_trained_steps = 0.0` always. The `_disc_ready > 0` fallback at line 1781 then sets `_disc_acc_now = 0.5` permanently on non-rank0 → `_disc_acc_ema = 0.5` → `_gated = 1.0` → μ = μ_peak = 0.3 for the lifetime of the run on ranks 1/2/3.

With FSDP gradient averaging, effective μ ≈ (μ_rank0 + 3·μ_peak)/4. Late-stage rank0 μ=0.05 gives effective μ ≈ 0.24, not 0.05 as "intended" from the rank0-only view.

**Impact on Phase 1:**
- v39b (disc_acc): Adaptive decay exists but is DAMPENED. rank0 signal drops, others pegged at peak → average μ ∈ [0.24, 0.30] instead of intended [0.05, 0.30].
- v40b (nll) and v43a (kl_lagrangian): use `sft_loss.detach().item()` which IS computed per-rank (no broadcast gating). Each rank has its own `_nll_ema` / `_kl_cost_ema` driven by that rank's micro-batch sft_loss. Still desynced but for a different reason (different samples per rank), not pegged at peak.
- v41b (ess_ratio): `dr3/ess_off_window` is the ESS of each rank's own `_w_off_hist` — per-rank signal. Similarly desynced but not pegged.

**Why not fix now:** (a) v39a already ran with this issue and completed with WebShop Val ≈ 0.65. Not catastrophic. (b) The fix requires an `all_reduce` on disc_acc scalar across ranks, plus equivalent for sft_loss/ess signals — a non-trivial multi-line change with risk of deadlock if executed only on some ranks. Flagging for post-round fix.

**Monitoring recommendation during Phase 1:** Log and visualize `chord/mu` per-rank in wandb (if possible) or at least watch whether the aggregated `chord/mu` settles at a value HIGHER than rank0's intended signal would predict. For v39b this will show up as `chord/mu` plateauing around ~0.2 after the disc learns, not reaching μ_valley=0.05.

### 5.2 Per-micro-batch EMA pollution from teacher-sparse micro-batches (design concern for v40b, v43a)

**Description:** `compute_chord_sft_loss` returns `sft_loss = agg_loss(weighted_losses, expert_mask)` with `expert_mask = exp_mask * response_mask`. On micro-batches with **zero expert tokens**, `agg_loss` returns NaN → replaced with 0.0 (line 1823-1824 of het_core_algos.py). With Phase 1 settings (`ppo_micro_batch_size_per_gpu=2`, 12.5% teacher fraction), roughly 6 of 8 micro-batches per rank have `expert_mask.sum()==0` → `sft_loss=0`.

The per-micro-batch update `_nll_ema = 0.7 * _nll_ema + 0.3 * sft_loss_now` therefore oscillates: +0.3*real_NLL on teacher batches, decay-by-0.7 on non-teacher batches. With α=0.3 the half-life is ~2 micro-batches. Result: μ from linear mapping (v40b) oscillates wildly within a single training step between ~0.13 and ~0.30 depending on whether the current micro-batch had a teacher.

**Impact on v40b:** The slope=0.156 was presumably calibrated against the aggregated `chord/sft_loss` value reported in wandb (mean across micro-batches per step, ~0.5-1.2 range for WebShop 1.5B early steps). The per-micro-batch EMA reaches a narrower, lower range. Expected μ trajectory could differ from the offline calibration.

**Impact on v43a:** `_kl_cost_ema` and `_kl_budget_ema` track the same zero-polluted signal. Because `_kl_budget_ema` uses ρ=0.9 (half-life ~7 samples), it lags `_kl_cost_ema` (α=0.3, half-life ~2). Dual-ascent behavior is preserved in shape but operates on a noisier signal than assumed.

**Why not fix now:** This is a design choice inherited from v40 (which was the predecessor and hasn't been run). The fix would be to skip EMA updates on teacher-empty micro-batches (`if expert_mask.sum() > 0:` guard) — a 2-line change. Recommend validating v40b mid-run (after step ~30) that `chord/nll_ema` trajectory looks sensible; if it pins near 0, this bug is biting.

### 5.3 ess_ratio saturating warmup behavior (intentional, documenting only)

During the first ~8 micro-batches before `_ess_anchor` is captured (gated by `ess_window_len >= 8`), `_ess_anchor_cur = max(_ess_now, 1.0)` — approximately the raw current ESS. Ratio ≈ 1, saturating gives `gated = max(0, 1 - 1^0.5) = 0` → μ = μ_valley = 0.05.

This is CORRECT behavior for the saturating mapping (which triggers only when ESS DROPS below anchor), but users should expect v41b to start at μ=0.05, not μ=0.30 as μ_peak might suggest.

---

## 6. Fixes applied

**None.** All issues found are either pre-existing structural properties (5.1), calibration concerns requiring empirical validation (5.2), or intentional design choices (5.3). Per audit protocol, complex/judgment-requiring fixes are flagged, not silently applied.

Smoke test was run before and after (zero edits) — results identical, confirming no accidental code mutation during the audit.

---

## 7. Final go/no-go recommendation

**GO.** Proceed with Round 8 Phase 1 sequential execution.

Justification:
- All branches produce valid μ ∈ [0.05, 0.30] across a broad range of synthetic signals (smoke test PASS × 5).
- No syntax errors, imports correct, config-to-code binding verified for all 4 target YAMLs.
- No numerical blow-ups under extreme stress (cost=100, ess=0.01, etc.).
- Non-blocking concerns (5.1, 5.2) are either pre-existing and already known-tolerable (v39a completed) or are calibration questions best answered empirically.

Watch conditions during Phase 1:
1. **v39b**: `chord/mu` should decay from 0.30 toward but possibly not reaching 0.05 (cross-rank averaging dampens). If `chord/disc_acc_ema` rises past 0.9 by step 30 but `chord/mu` stays above 0.2, confirm this is due to §5.1 and consider an all_reduce fix before Round 9.
2. **v40b**: `chord/nll_ema` should track in ~[0.2, 1.0]. If it pins near 0 (< 0.1) throughout, §5.2 is biting — consider skipping EMA updates on teacher-empty micro-batches.
3. **v43a**: `chord/mu_lagrange_state` (== `chord/mu`) should mutate step-to-step; `chord/kl_step_mult` should hover near 1.0 with excursions. A stuck `kl_step_mult ≈ 1.0` for 50+ steps would indicate cost≈budget collapse (EMAs synchronized).
4. **v41b**: `chord/ess_anchor` should be captured once `ess_window_len ≥ 8` (usually step 2-3) and then stay fixed. If `chord/ess_ratio` stays ≥ 1 throughout, μ will stay at valley — acceptable behavior if ESS is consistently healthy.

Baseline expectation: all four variants should produce WebShop Val@last ≥ 0.60 (a 0.08-point window below v24's 0.678 benchmark, which is the threshold below which an adaptive rule is clearly worse than the fixed schedule).

---

## Appendix: Smoke test invocation

```
python3 /data/home/qisheng/EvolAnalsis/tmp_scripts/smoke_test_round8_modes.py
```

Output: `ALL 5 TESTS PASSED` (disc_acc, nll_linear, ess_ratio_saturating, ess_ratio_sigmoid, kl_lagrangian).
