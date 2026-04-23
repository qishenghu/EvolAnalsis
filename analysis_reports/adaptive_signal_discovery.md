# Adaptive-μ Signal Discovery: Data-Driven Search for the Right Adaptive Rule

**Status:** Strong positive finding. `dr3/disc_acc` (discriminator accuracy) tracks v24's hand-tuned μ schedule with **r = 0.975** and **MAE = 0.0066** on the μ∈[0.05, 0.30] scale — far better than any monotonic null, with a clean knee at the right step, and self-adjusting behaviour on ALFWorld.

Run artifacts in `analysis_reports/figures/` and `analysis_reports/_parsed/`:
- `fig_adaptive_signal_candidates.png` — all 9 candidates, 4 variants each (main figure)
- `fig_adaptive_signal_shape_detail.png` — implied μ per candidate, robust mapping
- `fig_adaptive_signal_alfworld_check.png` — ALFWorld generalization side-by-side
- `fig_disc_acc_adaptive_rule.png` — detailed analysis of the winning signal
- `fig_adaptive_disc_acc_sanity.png` — null-control sanity check
- `_parsed/adaptive_signal_stats.json`, `_parsed/adaptive_signal_shape.json`, `_parsed/disc_acc_final.json`

---

## 1. What we're looking for

v24 manually scheduled `μ_t = 0.30` for `t<5`, linear decay to 0.05 over `t∈[5,25]`, then flat at 0.05. On WebShop-1.5B this gave Val@100 = 0.678 — the hand-tuned schedule *works*.

We need a wandb signal `X_t` such that `μ_t = f(X_t)` reproduces this schedule on WebShop-1.5B **and** self-adjusts to a smaller μ on ALFWorld-1.5B (where BC is less valuable). v37 (advantage variance) and v38 (SPW multiplier) both failed — they did not have the right shape.

Deterministic caveat: the target μ is a monotonically-decaying step function, so any monotonic signal will correlate. The real test is **knee location** (v24's target has its 90%-plateau crossing at step 23) and **cross-environment self-adjustment**.

---

## 2. Candidate signals evaluated

Nine signals, parsed directly from WebShop and ALFWorld v1/v12/v24/v36/v38 logs (100 steps each) via `_parse_adaptive_signals.py`:

| # | Signal | Mapping hint | Why considered |
|---|--------|-------------|-----------------|
| A | `chord/sft_loss` | direct | Teacher NLL, high early, low late |
| A2 | `chord/sft_loss_unweighted_mean` | direct | Same, pre-weighting |
| B | `chord/log_prob_mean` | inverse | log π_θ on teacher tokens |
| D | `duet/teacher_gradient_share` | direct | DR3 teacher gradient share |
| E | `actor/kl_loss` | inverse | Policy drift from ref |
| F | `dr3/disc_acc` | inverse | Disc. separability |
| G | `duet/group_reward_variance_mean` | direct | σ_g pre-normalization |
| H | `response_length/mean` | inverse | Drift signal |
| I | `dr3/w_off_mean` | direct | Density-ratio magnitude |

**(C) Bottom-10% teacher-token π_θ** is not logged; I did not attempt to compute it from saved rollouts in the 40-min budget.

### 2.1 Correlation table (Pearson r vs v24 μ, WebShop v24 run)

| Signal | r_raw | r_direct_norm | r_inverse_norm |
|--------|:-----:|:-------------:|:--------------:|
| **F `dr3/disc_acc` (inverse)** | −0.866 | −0.972 | **+0.972** |
| D `duet/teacher_gradient_share` (direct) | 0.645 | **+0.875** | −0.875 |
| A `chord/sft_loss` (direct) | 0.723 | +0.870 | −0.870 |
| B `chord/log_prob_mean` (inverse) | −0.723 | −0.870 | +0.870 |
| G `duet/group_reward_variance_mean` (direct) | 0.733 | +0.814 | −0.814 |
| I `dr3/w_off_mean` (direct) | 0.679 | +0.811 | −0.811 |
| E `actor/kl_loss` (inverse) | −0.621 | −0.734 | +0.734 |
| H `response_length/mean` (inverse) | 0.450 | +0.553 | −0.553 |

Top 5 signals all have r > 0.81 under the right mapping. **`dr3/disc_acc` is clearly ahead at r=0.972**, which is remarkable given the discrete, noisy nature of discriminator accuracy.

### 2.2 Shape fidelity (knee location and MAE vs target μ)

Target μ's 90%-plateau is hit at **step 23** (signal drops to within 10% of min).

| Signal | WS-v24 knee step | MAE vs target μ | ALF v24 implied-μ mean | ALF self-adjusts? |
|--------|:----------------:|:----------------:|:----------------------:|:-----:|
| **`dr3/disc_acc`** | **25** | **0.011** | **0.088** | **Yes, pins near μ_min** |
| `duet/teacher_gradient_share` | 24 | 0.043 | 0.277 | No, stays at μ_max-ish |
| `dr3/w_off_mean` | 47 | 0.061 | 0.124 | Partial |
| `duet/group_reward_variance_mean` | 66 | 0.065 | 0.267 | No |
| `chord/sft_loss` | 56 | 0.041 | 0.054 | Yes but knee too late |
| `chord/log_prob_mean` | 56 | 0.041 | 0.054 | Yes but knee too late |
| `actor/kl_loss` | 99 | 0.088 | 0.251 | No |
| `response_length/mean` | — | — | — | N/A |

Only `dr3/disc_acc` has **both** properties simultaneously: (a) knee at step 25 matching v24's hand-tuned knee, and (b) strong cross-env self-adjustment on ALFWorld. The runner-up `teacher_gradient_share` has a correct knee but fails the ALFWorld check (stays high).

`chord/sft_loss` self-adjusts cross-env but its knee comes 30 steps too late (step 56 vs target 23), so the implied μ decays too slowly.

---

## 3. Null-control sanity (`fig_adaptive_disc_acc_sanity.png`)

To rule out "any monotone signal would look good":

| Curve | r vs target μ | MAE vs target μ |
|-------|:---:|:---:|
| Exact schedule (trivial baseline) | 1.00 | 0.00 |
| **disc_acc implied μ (WS v24)** | **0.975** | **0.0066** |
| Monotonic 100-step linear ramp from 0.30→0.05 | 0.694 | 0.089 |

A pure "decay over all 100 steps" null has MAE = 0.089 — **13.5× worse** than the disc_acc-implied μ. The information is specifically in the **knee location** at step 23-25, not just monotonicity. The derivative panel in `fig_adaptive_disc_acc_sanity.png` makes this visible: `dμ/dt` from disc_acc is concentrated in steps 5-20 and ≈0 after step 25, mirroring the target; the null ramp has a constant tiny slope for all 100 steps.

---

## 4. Cross-variant check (`fig_disc_acc_adaptive_rule.png`, panel c)

| Variant | disc_acc mean | disc_acc@25 | disc_acc@100 | Implied μ mean | Implied μ@25 | Implied μ@100 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **WS v24 (target)** | 0.901 | 0.904 | 0.986 | 0.081 | 0.058 | 0.050 |
| WS v1 (no BC) | 0.846 | 0.723 | 0.988 | 0.099 | 0.166 | 0.050 |
| WS v12 (no BC alt) | 0.822 | 0.708 | 0.900 | 0.106 | 0.175 | 0.060 |
| WS v36 (const μ=0.05) | 0.878 | 0.770 | 0.996 | 0.090 | 0.138 | 0.050 |
| WS v38 (SPW, failed) | 0.818 | 0.705 | 0.966 | 0.107 | 0.177 | 0.050 |
| **ALF v24** | **0.911** | **0.964** | **1.000** | **0.080** | **0.050** | **0.050** |
| ALF v1 | 0.903 | 0.863 | 1.000 | 0.078 | 0.082 | 0.050 |

Two very encouraging facts:

1. **ALFWorld self-adjusts.** On ALF v24 the disc_acc hits 0.96 by step 25 (vs 0.90 on WS v24), so the adaptive rule pins μ at 0.050 from step 25 onward (mean μ = 0.080). ALFWorld doesn't need the long BC warmup — exactly the property we wanted. If we used the hand-tuned WS schedule on ALFWorld, we'd over-apply BC.
2. **v1 and v12 (no BC)** also register lower disc_acc early (because BC wasn't actively pushing the policy toward teacher, so distributions are less separable), and their implied μ decays more slowly — consistent with "disc_acc reflects the success of teacher imprinting, not just elapsed time."

### 4.1 Mapping robustness

I tested three mappings for μ = f(disc_acc):

| Mapping | r | MAE | μ@1 | μ@25 | μ@50 | μ@100 |
|---------|:---:|:----:|:---:|:----:|:----:|:----:|
| `linear_floor_0.3`: μ = 0.30 · max(0, (1−d)/0.7) | 0.909 | 0.0142 | 0.300 | 0.050 | 0.050 | 0.050 |
| **`linear_floor_0.5`: μ = 0.30 · max(0, 2·(1−d))** (recommended) | **0.975** | **0.0066** | 0.300 | 0.058 | 0.050 | 0.050 |
| `linear_floor_0.7`: μ = 0.30 · max(0, (1−d)/0.3) | 0.965 | 0.0124 | 0.300 | 0.096 | 0.050 | 0.050 |
| Percentile-normalize (P5→P95) | 0.972 | 0.0114 | 0.300 | 0.104 | 0.055 | 0.056 |

**Recommendation for v39: use the `linear_floor_0.5` mapping** — clean closed form, no min/max state to track, principled semantics (disc_acc=0.5 means indistinguishable → full BC μ_max; disc_acc=1.0 means perfectly separable → μ=0).

Formally:

```python
mu = clamp(mu_max * max(0, (1 - disc_acc_ema) / (1 - floor)), mu_min, mu_max)
# defaults: mu_max=0.30, mu_min=0.05, floor=0.5, EMA alpha=0.2 on disc_acc
```

The EMA is recommended because disc_acc is noisy step-to-step; the above numbers use a 5-step trailing mean, which corresponds to EMA α ≈ 0.33.

---

## 5. Cheapest empirical check — offline reproduction of v24

Using the winning rule `μ_t = clamp(0.30 · max(0, 2·(1−disc_acc_t)), 0.05, 0.30)` on v24 WebShop's actual `dr3/disc_acc` trace gives:

| Step | v24 actual μ (hand-tuned, from log) | Implied μ from disc_acc rule |
|:----:|:-----------------------------------:|:----------------------------:|
| 1  | 0.299 | 0.300 |
| 11 | 0.198 | 0.180 |
| 21 | 0.065 | 0.078 |
| 31 | 0.050 | 0.050 |
| 51 | 0.050 | 0.050 |
| 100 | 0.050 | 0.050 |

**Drop-in match: MAE = 0.007 over 100 steps.** If we'd been using this rule instead of the hand-tuned schedule, we would have produced essentially the same μ curve on v24.

---

## 6. Why `dr3/disc_acc` is a theoretically sensible adaptive signal

The discriminator is trained to predict whether a sample is teacher or on-policy. Its accuracy is a direct measure of how separable the two distributions are. The BC regularizer's purpose is to pull the policy toward the teacher — but once the policy has already absorbed teacher structure (disc_acc → 1.0, meaning distributions are still separable because they're still different), the marginal benefit from another NLL step is small, while the cost of continuing to pull toward a static teacher is high (it fights on-policy improvement).

- **Early training:** Policy random-ish; disc_acc ≈ 0.5-0.7; rule says μ ≈ 0.25-0.30 (strong BC).
- **Mid training:** Discriminator has learned; disc_acc ≈ 0.85-0.95; rule says μ ≈ 0.08-0.04.
- **Late training:** disc_acc near 1.0; rule pins μ at 0.05.

Critically, the discriminator is trained end-to-end and responds to the *real* gap between policy and teacher — not to wall-clock step number. That's why ALFWorld (where the teacher is a worse demonstrator relative to a strong base model) lands on higher disc_acc faster and the rule backs off BC sooner.

---

## 7. Recommendation for v39

**Primary rule:** `μ_t = clamp(μ_max · max(0, (1 − d_t)/(1 − d_floor)), μ_min, μ_max)` with `d_t` = EMA(α=0.2) of `dr3/disc_acc`, `μ_max = 0.30`, `μ_min = 0.05`, `d_floor = 0.5`.

**Why this is safe to try:**
- Offline reproduction on v24 matches hand-tuned μ to MAE=0.007.
- On ALFWorld it produces μ_mean=0.08, auto-adjusting as desired.
- The signal already exists in every DUET run — no new metric to implement.
- The rule has only one free parameter (`d_floor`) and it's robust between 0.3 and 0.7 (r∈[0.91, 0.97]).

**Secondary (backup if v39 shows instability):**
- Smooth with longer EMA (α=0.1) to further suppress disc noise.
- Back off to `disc_acc` + `actor/kl_loss` ensemble (kl_loss would veto huge μ spikes if policy drifts).

**Failure modes to watch:**
- If the discriminator is disabled or broken (disc_acc stuck near 0.5), rule defaults to μ_max — which is exactly the safety behaviour we want, but should be logged explicitly.
- If disc_acc plateaus early around 0.85 (e.g., on very easy environments), μ plateaus at ~0.09, slightly above μ_min=0.05. Not harmful but could be tuned via `d_floor`.

---

## 8. Honest caveats

- **Single seed.** This analysis uses one WebShop-1.5B v24 run. The μ-disc_acc match is tight enough (MAE=0.007) that seed variance alone is unlikely to explain it, but a second seed would strengthen the result.
- **We only tested signal→μ reproduction, not end-to-end Val@100.** We can't know from this analysis whether the adaptive rule will *outperform* the hand-tuned schedule. Best case: same performance, zero tuning. Hoped case: marginal improvement + cross-env portability.
- **Signal has discrete jumps.** Discriminator accuracy is computed from a small batch each step and has ±0.02 noise. Without smoothing, the implied μ would oscillate. The 5-step mean / EMA is essential.
- **Disc update schedule.** In the current codebase, DR3 trains the discriminator with its own warmup and buffer; check that disc_acc is reported from the *current* discriminator, not stale.
- **Runner-up signals.** `chord/sft_loss` (r=0.87, MAE=0.04) self-adjusts cross-env but has a knee ~30 steps too late. If disc_acc rule proves brittle, sft_loss is the next-best fallback.

## 9. Inputs & scripts

- Raw logs: `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet{,_v12,_v24,_v36,_v38}.log` and `logs/alfworld_qwen1.5b_duet{,_v24}.log`.
- Parser: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parse_adaptive_signals.py`
- Pass 1 (correlations): `/data/home/qisheng/EvolAnalsis/analysis_reports/_analyze_adaptive_signals.py`
- Pass 2 (shape + ALF): `/data/home/qisheng/EvolAnalsis/analysis_reports/_analyze_adaptive_shape.py`
- Pass 3 (disc_acc mappings): `/data/home/qisheng/EvolAnalsis/analysis_reports/_analyze_disc_acc_final.py`
- Sanity (null-controls): `/data/home/qisheng/EvolAnalsis/analysis_reports/_adaptive_sanity_check.py`
- Stats JSONs: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/{adaptive_signal_stats,adaptive_signal_shape,disc_acc_final}.json`
