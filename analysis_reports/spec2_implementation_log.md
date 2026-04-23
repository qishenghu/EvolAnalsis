# Spec 2 / Candidate B — Surprise-Weighted DR3 implementation log

Status: **IMPLEMENTED + VALIDATED (16/16 smoke checks PASS)**. Awaiting human review before v38 launch.

---

## 1. Summary of exact changes

### 1.1 `agentevolver/module/exp_manager/het_core_algos.py`

Added three new keyword-only parameters to `het_compute_teacher_aware_loss` (end of signature, around the old L297):

```python
# ⭐ Spec 2 (Candidate B): Surprise-Weighted DR3
use_spw_teacher: bool = False,
spw_phi_formula: str = "1_minus_pi",
spw_mask_on_positive_A: bool = True,
```

Inserted the SPW multiplier right AFTER the existing `teacher_loss_scale` block (around original L571) and BEFORE `teacher_off_pg_losses_raw = -advantages * teacher_ratio`:

```python
# ⭐ Spec 2 / Candidate B: Surprise-Weighted DR3
# φ(s,a) = (1 - π_θ(a|s)) detached, applied ONLY on teacher tokens.
spw_phi_raw = None
spw_effective_coef = None
if use_spw_teacher:
    pi_theta = torch.exp(log_prob).detach().clamp(0.0, 1.0)
    spw_phi_raw = (1.0 - pi_theta).clamp(0.0, 1.0)  # φ ∈ [0, 1], detached

    tm = teacher_mask_float
    if spw_mask_on_positive_A:
        apply_mask = tm * (advantages > 0).float()
    else:
        apply_mask = tm
    # phi_factor = 1 where apply_mask=0; = φ where apply_mask=1
    phi_factor = 1.0 - apply_mask * (1.0 - spw_phi_raw)
    teacher_ratio = teacher_ratio * phi_factor
    spw_effective_coef = spw_phi_raw * advantages.detach()
```

Added diagnostics right after the `teacher_prob` stats block (around original L714):

```python
if spw_phi_raw is not None and torch.is_tensor(spw_phi_raw):
    ratio_stats.update(_masked_stats(spw_phi_raw, teacher_token_mask, "spw/phi"))
if spw_effective_coef is not None and torch.is_tensor(spw_effective_coef):
    ratio_stats.update(_masked_stats(spw_effective_coef, teacher_token_mask, "spw/effective_teacher_coef"))
```

This emits wandb metrics: `spw/phi/{mean,std,min,max,p50,p90,p99,count}` and `spw/effective_teacher_coef/{mean,std,min,max,...}`.

### 1.2 `agentevolver/module/exp_manager/het_actor.py`

Plumbed the three config flags to BOTH call sites of `het_compute_teacher_aware_loss` that are reachable from v38:

**Site A — DR3 + hybrid policy shaping (L1560-1591, the path v38 takes):**

```python
_use_spw = bool(self.config.get("use_spw_teacher", False))
_spw_formula = str(self.config.get("spw_phi_formula", "1_minus_pi"))
_spw_mask_posA = bool(self.config.get("spw_mask_on_positive_A", True))

ret_dict = het_compute_teacher_aware_loss(
    ...  # existing args
    use_spw_teacher=_use_spw,
    spw_phi_formula=_spw_formula,
    spw_mask_on_positive_A=_spw_mask_posA,
)
metrics["dr3/hybrid_policy_shaping"] = 1.0
metrics["dr3/hybrid_beta"] = _hybrid_beta
if _use_spw:
    metrics["spw/enabled"] = 1.0
    metrics["spw/mask_on_positive_A"] = 1.0 if _spw_mask_posA else 0.0
```

**Site B — pure LUFFY fallback (L1864 call, future-compat):**
Added SPW kwargs with `self.config.get(..., False/default)` so `use_spw_teacher=true` works even with DR3 disabled.

Other call sites (L1354, L1398 — the `teacher_use_log_prob=True` ExGRPO path and the DR3-not-ready fallback) were intentionally NOT modified. They are defensive fallbacks, and adding SPW there risks subtle differences when the hybrid path should be taken. For v38 (`use_policy_shaping=true`, `apply_min_buf_size=512`), the active path is unambiguously Site A once the buffer warms up.

### 1.3 `config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v38.yaml` (NEW)

Copied v12 (DR3+SC, no BC) and added only:
- `actor_rollout_ref.actor.use_spw_teacher: true`
- `actor_rollout_ref.actor.spw_phi_formula: "1_minus_pi"`
- `actor_rollout_ref.actor.spw_mask_on_positive_A: true`
- Renamed experiment_name and workspace_id to `webshop_qwen1.5b_duet_v38`

CHORD is already off in v12 (`use_chord: false`); no `chord_mu_*` params present. State Channel remains on with `exclude_teacher: true`.

---

## 2. Smoke test output

Ran `/data/home/qisheng/miniconda3/envs/duet/bin/python3 tmp_scripts/smoke_test_spw.py`:

```
============================================================
Spec 2 / Candidate B — Surprise-Weighted DR3 smoke test
============================================================
  [PASS] no NaN/Inf in baseline loss
  [PASS] no NaN/Inf in SPW (posA) loss
  [PASS] no NaN/Inf in SPW (all) loss
  [PASS] no NaN/Inf in baseline grad
  [PASS] no NaN/Inf in SPW grad
  [PASS] φ min >= 0 (got 0.7190890312194824)
  [PASS] φ max <= 1 (got 0.9658815264701843)
  [PASS] φ mean in (0,1) (got 0.8746725916862488)
  [PASS] SPW(posA) loss differs from baseline (|Δ|=1.270601e-02)
  [PASS] SPW(all) loss differs from baseline (|Δ|=4.551087e-03)
  [PASS] grad non-zero on teacher tokens (SPW)
  [PASS] grad differs from baseline (SPW effect reaches optimizer)
  [PASS] φ is detached (no grad flows through 1-π factor)
  [PASS] mask_on_positive_A=True skips A<0 teacher tokens (|Δ|=0.00e+00)
  [PASS] back-compat: no-SPW-kwarg matches use_spw_teacher=False
  [PASS] spw/effective_teacher_coef/mean diagnostic present (value=0.5497705340385437)
============================================================
OVERALL: PASS
============================================================
```

16/16 checks pass. Highlights:

- **φ range**: mean 0.87, min 0.72, max 0.97 (high because simulated log_probs were in [-3.5, -0.5]; real training will span more). Confirms φ ∈ [0,1].
- **Effect on loss**: non-trivial (Δ ≈ 1.3% for posA mode, 0.45% for all-tokens mode). Large enough to bias gradients, small enough to not blow up.
- **Gradient separation**: baseline vs SPW grads differ by >1e-8 at max. Confirms SPW actually reaches the optimizer.
- **φ detached**: direct autograd check shows ∂φ/∂log_prob = 0 under the detach/clamp path.
- **mask_on_positive_A=True short-circuit**: when ALL advantages are negative, SPW is a no-op (|Δ|=0.0) — meaning failed teacher trajectories are NOT affected. Critical safety invariant.
- **Back-compat**: omitting all SPW kwargs yields identical loss to `use_spw_teacher=False`.

---

## 3. Config validation output

```
{
  "experiment_name": "webshop_qwen1.5b_duet_v38",
  "use_spw_teacher": true,
  "spw_phi_formula": "1_minus_pi",
  "spw_mask_on_positive_A": true,
  "use_dr3": true,
  "use_chord": false,
  "dr3.use_policy_shaping": true,
  "sc.enable": true,
  "sc.exclude_teacher": true,
  "workspace_id": "webshop_qwen1.5b_duet_v38"
}
```

Python AST parse of `het_core_algos.py` and `het_actor.py`: both OK.

---

## 4. Correctness reasoning (the "double-count" check)

The user correctly flagged a risk: DR3 already shifts `old_log_prob ← log_prob.detach() - log(w_hat)` at `het_actor.py:1500`. After that shift, in `het_compute_teacher_aware_loss`:

- `ratio = exp(log_prob - old_log_prob) = exp(log_prob - (log_prob.detach() - log(w_hat))) ≈ w_hat × exp(Δ)` where Δ is the fresh-vs-stored log-prob difference.
- But v38 has `dr3.use_policy_shaping=true`, so the code enters the **hybrid branch** (L1544), which uses `teacher_use_log_prob=False`. That means `teacher_ratio = exp(log_prob) × shaping(...)` — the `old_log_prob`-based `ratio` is NOT used for the teacher branch.
- DR3's `w_hat` contribution in this path comes through `teacher_loss_scale = teacher_loss_scale × w_hat` (set at L1547-1549), which multiplies `teacher_ratio`.
- My SPW patch runs AFTER both `teacher_loss_scale` scaling and policy shaping: `teacher_ratio *= phi_factor`.
- Final teacher loss per token: `L_teacher = − A × φ × w_hat × shaping(exp(log_prob))` — exactly the spec's `φ × w_hat × A × clip(ratio)` form (where clip ≡ shaping here).

No double-count. Ordering is: shaping → teacher_loss_scale (= w_hat) → SPW → loss.

### Mask semantics confirmation

In both `het_compute_teacher_aware_loss` and `compute_chord_sft_loss`, "teacher tokens" are `teacher_mask * exp_mask * response_mask`. My patch uses `teacher_mask_float` alone as the gate for φ, because `teacher_ratio` is only consumed via `torch.where(teacher_mask_float.bool(), teacher_off_pg_losses, self_off_pg_losses)` (L590-594). Non-teacher tokens never see `teacher_ratio`, so applying φ only there is equivalent. Correct.

---

## 5. Modified files

Absolute paths:
- `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py`
- `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py`
- `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v38.yaml` (NEW)
- `/data/home/qisheng/EvolAnalsis/tmp_scripts/smoke_test_spw.py` (NEW)

---

## 6. Unexpected findings

1. **Two distinct DR3 code paths** exist in `het_actor.py`: `dr3.use_policy_shaping=true` routes to hybrid LUFFY mode (Site A), while `use_policy_shaping=false` routes to `repo_compute_token_loss` (a different function that I did NOT modify). If someone flips `use_policy_shaping=false` in future experiments AND wants SPW, they'd need a separate patch into `repo_compute_token_loss`. v12/v38 both use `use_policy_shaping=true`, so this is not a v38 blocker.

2. **Two fallback call sites** (L1354 ExGRPO and L1398 DR3-not-ready) were left unchanged. In v12/v38 with `apply_min_buf_size: 512`, these fallbacks run only for the first ~64 steps (until buffer is warm). SPW not being applied during warmup is arguably desirable — it gives the discriminator time to stabilize before adding a new multiplicative factor.

3. **φ is very high at init** (~0.87 mean in the smoke test). Real data will likely show φ ~ 0.95-0.99 early and decay as the policy learns teacher actions. The metric `spw/phi/mean` should be a clean "how much teacher info remains useful" indicator.

4. **mask_on_positive_A=True is numerically safer than the alternative.** In the smoke test, when all advantages were forced negative, SPW = no-op. This avoids a subtle pathology: on failed teacher trajectories (A<0), SPW would otherwise down-weight the gradient that pushes teacher tokens DOWN, which is the opposite of what we want (we want strong gradient AGAINST rare teacher tokens that led to failure). With posA masking, failed teacher tokens keep the full `w_hat × A` weight.

---

## 7. Human review checklist before launching v38

Please review the following before kicking off v38:

- [ ] **Intent check**: SPW is meant to *reduce* teacher gradient on known tokens. Confirm this aligns with the research question (e.g., "does concentrating teacher gradient on surprising tokens accelerate learning vs. v12's uniform DR3?").
- [ ] **Interaction with `teacher_baseline_separation`**: enabled in v38. SPW only affects the teacher surrogate, not advantage computation, so baseline-separation math is unchanged. But if teacher gradient effectively shrinks, the teacher baseline may diverge faster — watch `critic/*_teacher_*` wandb metrics.
- [ ] **φ detached but advantages.detach() in diagnostic**: `spw_effective_coef` uses `advantages.detach()` purely for logging — it is NOT part of the loss. Loss uses the non-detached `advantages`. Double-check this is acceptable (my read: yes, because the loss computation `teacher_off_pg_losses_raw = -advantages * teacher_ratio` uses live `advantages`).
- [ ] **Warmup interaction**: for the first ~64 steps, v38 falls back to LUFFY (no SPW). If early-training behavior matters for comparison with v12, note this in the experiment report.
- [ ] **`exclude_teacher=true` for SC**: already true in v12/v38. SPW changes teacher *gradient weighting*, not SC bonus application. SC still bonus-applies only to on-policy samples. No interaction.
- [ ] **Metric naming**: `spw/phi/mean`, `spw/effective_teacher_coef/mean`, `spw/enabled`, `spw/mask_on_positive_A`. Confirm these fit your wandb dashboard conventions.
- [ ] **`spw_phi_formula`** currently only supports `"1_minus_pi"`. If you want alternatives (e.g., `"entropy"`, `"-log_pi"`), they need explicit implementation — the current code silently falls back to `1_minus_pi` for unknown strings. Acceptable for now.
- [ ] **No kills / no concurrent runs**: v24-ALFWorld still running on GPU 0-3 per user note. v38 launch must wait for v24 completion.
