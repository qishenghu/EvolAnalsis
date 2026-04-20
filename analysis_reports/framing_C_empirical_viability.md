# Framing C — Empirical Viability Report

**Target claim (Framing C):** "BC and DR3 operators are automatically specialized by π_θ: BC dominates the gradient on rare-teacher-tokens (low p_θ); DR3 dominates on common ones (high p_θ). No explicit gating needed."

**Verdict: YELLOW, trending RED.**
The current v24 instrumentation can *loosely* support a weaker version of this story (population-level), but the load-bearing **per-token crossover figure** is NOT producible from existing logs. It also appears empirically unlikely to produce a clean crossover: in v24 the μ-decay schedule, not p_θ, is the dominant modulator of BC. A brief re-run with targeted logging (~5h on WebShop-1.5B) is required to settle the question — and there is a real risk that the answer kills Framing C.

---

## 1. Data availability audit (Task 1)

v24 run: `wandb/run-20260419_155709-9h7vyhkq` (100 training steps, complete).
Relevant source paths:
- Actor: `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_actor.py`
- Losses: `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/het_core_algos.py` (`compute_chord_sft_loss` L1723; `het_compute_teacher_aware_loss` L237)
- DR3: `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/dr3_ratio.py`
- Log: `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_v24.log` (6.1 MB, 100 steps)

What we **have**:

| Quantity | Granularity | Source |
|---|---|---|
| `p_θ(a_teacher\|s_teacher)` **mean, std** | per-step, aggregated over teacher tokens | `chord/log_prob_mean`, `chord/log_prob_std` |
| Shaped teacher ratio `p/(p+β)` mean/std/p50/p90/p99 | per-step | `teacher_diag/teacher_ratio/*` |
| Teacher advantage mean/std/p50/p99 | per-step | `teacher_diag/adv/teacher/*` |
| `w_hat` mean/std/p50/p90/p99/max | per-step, teacher only | `dr3/w_off_*` |
| DR3 gradient share (|A| based) | per-step | `duet/teacher_gradient_share` |
| `μ(t)` CHORD schedule | per-step | `chord/mu` |
| Aggregated SFT loss | per-step | `chord/sft_loss`, `chord/weighted_sft_loss` |
| Discriminator accuracy | per-step | `dr3/disc_acc` |

What we **don't** have (the critical gap):

- **Per-token** `p_θ`, `w_hat`, advantage, BC and DR3 contribution.
- **Binned** `p_θ` histogram with conditional BC/DR3 losses — i.e. nothing that lets us say "at `p_θ ∈ [1e-3, 1e-2]` BC contributes X% of grad, DR3 Y%".
- Any record of teacher-token `log_prob` quantiles (only mean + std).
- No per-token tensors are saved to disk in v24.

The **only** usable per-token tensor is `log_prob` during `update_actor`, but it is discarded after the backward pass. Teacher-token `w_hat` is sequence-level (one value per trajectory), so the DR3 side is intrinsically not per-token in the current design.

## 2. Can we produce the crossover figure from existing data? (Task 2)

**No.** The x-axis of the target figure is per-token `p_θ` in bins. What we have is per-step summary moments. I cannot reconstruct the distribution, let alone conditional BC / DR3 contributions.

**What we can plot today (weaker evidence):**
- `log_prob_mean` vs `μ(t)` vs step — shows the population drift of teacher-token `p_θ`.
- `teacher_gradient_share` vs `disc_acc` vs `success` — DR3 fade-out trend.
- `chord/weighted_sft_loss` vs `chord/grpo_loss` — relative magnitude of the two operators.

None of these shows a per-token crossover; they show time-series population trends.

## 3. Theory prediction sanity-check (Task 3, from v24 log)

Correlations across 99 steps (v24):

| Pair | Pearson r | Interpretation |
|---|---|---|
| `disc_acc ↔ success_onpolicy` | **+0.16** | very weak; theory predicted disc_acc rises with training — it does, but success in v24 only reaches 0.12 by step 100 |
| `disc_acc ↔ teacher_gradient_share` | **−0.49** | consistent with DR3 fade-out |
| `μ ↔ teacher_gradient_share` | **+0.64** | **concerning — μ decay explains tgs decay better than p_θ does** |
| `μ ↔ success` | −0.17 | μ decays monotonically while success barely moves |
| `success ↔ teacher_gradient_share` | −0.09 | essentially none |

Population trajectory of key quantities:

| step | μ | p_mean=e^log_p | BC∝μ(1−p) | DR3∝r·A | disc_acc | tgs | success |
|---|---|---|---|---|---|---|---|
| 1 | 0.30 | 0.31 | 0.21 | 4.0 | 0.00 | 0.31 | 0.00 |
| 10 | 0.21 | 0.35 | 0.14 | n/a | 0.72 | 0.32 | 0.00 |
| 25 | 0.05 | 0.45 | 0.03 | 1.3 | 0.90 | 0.15 | 0.00 |
| 50 | 0.05 | 0.49 | 0.03 | 0.52 | 0.99 | 0.08 | 0.00 |
| 100 | 0.05 | 0.54 | 0.02 | 1.6 | 0.99 | 0.08 | 0.12 |

Two red flags emerge:

1. **μ is doing the work, not π_θ-specialization.** μ collapses from 0.30→0.05 over 25 steps (CHORD schedule); teacher-token p_θ drifts only from 0.31→0.54. The BC operator's decline is ~90% attributable to μ, not to p_θ rising. A static μ (or μ=const ablation) would reveal whether "specialization by π_θ" is really operative — *it probably is not in v24 as configured.*

2. **v24 ran with `chord_use_token_weighting: false`.** That means `φ(p)=p(1−p)` is forced to 1, so the per-token BC gradient is `μ · (1 − p_θ)` scaled uniformly — there is no p-dependent re-weighting built into the BC operator at all in v24. The crossover Framing C envisions would need `chord_use_token_weighting: true` at minimum. In v24 logs, `chord/phi_mean=1.000, phi_min=1.000, phi_max=1.000` (step 50) confirms this.

## 4. What's missing & concrete logging patch (Task 4)

To produce a defensible crossover figure we must log per-step **binned** statistics over teacher tokens. Proposed patch in `compute_chord_sft_loss` and `het_compute_teacher_aware_loss`:

```python
# Inside compute_chord_sft_loss, after "expert_mask = exp_mask * response_mask":
with torch.no_grad():
    p_expert = torch.exp(log_prob.clamp(max=0))[expert_mask.bool()]
    sft_expert = sft_losses[expert_mask.bool()]
    # Bin edges in log-space
    edges = torch.tensor([0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0], device=p_expert.device)
    bin_idx = torch.bucketize(p_expert, edges)
    for b in range(1, len(edges)):
        m = (bin_idx == b)
        if m.any():
            chord_diag[f"pbin/{b}/count"] = int(m.sum().item())
            chord_diag[f"pbin/{b}/p_mean"] = float(p_expert[m].mean().item())
            chord_diag[f"pbin/{b}/sft_loss_mean"] = float(sft_expert[m].mean().item())
            # Effective BC contribution proxy: -d/dθ(-μ log p) ~ μ (1-p)
            chord_diag[f"pbin/{b}/bc_grad_mag"] = float(((1 - p_expert[m])).mean().item())
```

And in `het_compute_teacher_aware_loss`, compute and log per-token DR3 contribution in the same bins by broadcasting `w_hat` × `teacher_ratio` × `advantages` at teacher tokens. The `w_hat` is currently sequence-level; we need to broadcast it onto each teacher token first (already done for the loss; just expose it in diagnostics).

This is **~40 lines of new code**, all inside existing functions, all in the already-covered backward-no-grad block. Performance impact negligible.

**Re-run spec:**
- Config: clone `webshop_qwen1.5b_duet_v24.yaml` → `..._v25_binlog.yaml`, no other changes.
- Additionally flip `chord_use_token_weighting: true` for v26 to test the "real" CHORD variant that Framing C implicitly assumes.
- Control run: re-run v12 (DR3-only, `use_chord: false`) with the same DR3 per-bin logging — this isolates DR3's p-dependence from BC.
- Time: WebShop 1.5B ~5h/run × 3 runs (v25, v26, v12-relog) = ~15h on 4 GPUs.

## 5. Red-flag assessment (Task 5)

Based on existing v24 aggregates, **all three red flags are already live**:

1. **Crossover likely monotonic.** `teacher_ratio_mean` (DR3 side, after p/(p+β) shaping) stays at 0.44–0.81 across the whole run while the dominant modulator of BC is μ. If anything, DR3 is *always* larger than BC in magnitude (see `DR3∝r·A` vs `BC∝μ(1−p)` table), contradicting the "BC dominates at low p_θ" story. There is no step at which the ordering reverses.
2. **DR3 contribution is flat-to-increasing, but dominated by `A`, not `p_θ`.** `r_mean ≈ 0.44-0.81`; the variance in "DR3 effective" is mostly driven by the teacher advantage `A_mean` fluctuating from 1.2 to 5.0. Not a clean "rises with p_θ" picture.
3. **μ dominates over p_θ.** corr(μ, tgs) = +0.64 > |corr(disc_acc, tgs)| = 0.49. The "automatic specialization" story is largely a restatement of the μ curriculum.

## 6. Deliverables summary

- **Verdict:** YELLOW, leaning RED. Framing C is not empirically supportable from v24 logs today, and preliminary aggregates suggest the per-token crossover may not exist at all — the effect appears dominated by the μ schedule.
- **To salvage YELLOW:** run the logging patch above on v25 and re-examine. If a crossover exists per-token even though the means hide it, we'll see it in the bins. Expected decision by +20h of compute.
- **If RED confirmed:** abandon Framing C. Re-frame paper around "DR3 + BC are complementary operators acting on disjoint signals (importance-weighted RL vs. imitation), whose mixing is governed by the dual-ESS dual variable and the μ schedule" — i.e. a *mechanism* claim rather than an *automatic-specialization* claim. This is still publishable and is what the code actually does.
- **Figures the paper needs**:
  1. *Per-bin BC vs DR3 gradient magnitude vs p_θ* — MISSING, requires re-run.
  2. *Time-series of tgs, μ, disc_acc, success* — available now from v24.
  3. *Ablation: no-μ-decay vs no-DR3 vs full DUET* — currently v12 (no-BC) exists; need a "no-DR3, only μ-scheduled BC" control.
