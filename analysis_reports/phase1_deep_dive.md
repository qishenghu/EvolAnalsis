# Phase 1 Deep Dive — Why disc_acc Wins and Lagrangian/NLL/ESS Fail

Deep empirical diagnosis of why the theoretically strongest adaptive-μ variant (v43a KL-Lagrangian) ties the theoretically weakest (v40b NLL) at ~4.5% success, while the supposedly pragmatic disc_acc family wins with v39b at 19%. Based on per-step metrics from webshop_qwen1.5b_duet {v24, v39, v39b, v40b, v41b, v43a} logs (100 training steps each).

## 0. Anchor numbers

Training-log (last-10-step moving average) numbers that match user's ranking:

| Variant | Mode | μ s5-30 above 0.10 (/25) | Late KL>1 frac | Last-10 success | Last-10 reward | Best success |
|---|---|---:|---:|---:|---:|---:|
| v24 (hand) | sched | 13 | 4% | 0.088 | 0.649 | 0.175 |
| v39 (α=0.2) | disc_acc | 18 | 4% | 0.067 | 0.660 | 0.211 |
| v39b (α=0.5) | disc_acc | 18 | 4% | 0.109 | 0.639 | 0.246 |
| v40b | NLL | 25 | 2% | 0.065 | 0.601 | 0.211 |
| v41b | ESS | 0 | 14% | 0.026 | 0.592 | 0.143 |
| v43a | Lagrange | 15 | 46% | 0.044 | 0.637 | 0.175 |

Ranking by last-10 success preserves the user's eval ranking: **v39b > v24 ≈ v39 ≈ v40b > v43a > v41b**.

MAE of each variant's μ-trajectory vs v24's implied hand schedule (peak=0.30, valley=0.05, decay_steps=25, linear 5→25):

| Variant | MAE s1-25 | MAE s26-50 | MAE s51-100 | Full MAE |
|---|---:|---:|---:|---:|
| v39 (α=0.2) | 0.031 | 0.015 | 0.003 | **0.013** |
| v39b (α=0.5) | 0.037 | 0.014 | 0.002 | **0.013** |
| v40b (NLL) | 0.055 | **0.132** | **0.095** | **0.094** |
| v41b (ESS) | **0.130** | 0.000 | 0.000 | **0.031** |
| v43a (Lagr.) | 0.019 | 0.013 | 0.003 | **0.009** |

Important note: **shape similarity to v24 does NOT predict performance.** v43a matches v24 _most_ closely (MAE 0.009) yet under-performs v39b (MAE 0.013) by a wide margin. Performance is driven by second-order properties (KL stability, grad_norm behavior), not μ shape alone.

---

## 1. Per-variant diagnosis

### v43a — KL-Lagrangian (Hypothesis A CONFIRMED)

**What μ actually did** (fig `fig_v43a_lagrangian_diagnostic.png`, `fig_phase1_mu_trajectories.png`):
μ starts at 0.300, decays smoothly to 0.084 at step 25, reaches the 0.05 floor at step 43, stays there for 46% of training. Shape tracks v24 closely (MAE 0.009), but the decay is driven almost entirely by the base schedule — the Lagrangian's step multiplier contributes at most 3%.

**What the adaptive signal did** (fig 3 panel 3, fig 4):
- `kl_step_mult` range: **0.935 to 1.034** (std=0.022); only 5% of steps have |mult−1| > 0.05; **0%** exceed |mult−1| > 0.20.
- `kl_cost_ema - kl_budget_ema` stays near zero (median |diff| = 0.062): auto-tuned budget tracks a running mean of actual cost, so dual ascent sees near-zero gradient.
- `mu_lagrange_state` trajectory matches `chord/mu` exactly: the Lagrangian contributes _no_ schedule modulation.

**Why it underperformed** (catastrophic late-training divergence):
- 23/50 late-training steps have actor/kl_loss > 1.0 (vs v39b's 2, v24's 2).
- At worst-KL steps (s85–100), actual KL averages 1.16 while `kl_cost_ema` = 0.613 — the EMA **lags actual KL by 2× in magnitude**. The budget tracks the (lagged) cost, so they stay in balance; step_mult stays at 0.99.
- Counter-intuitively, when KL momentarily spikes and cost_ema starts rising, the auto-tuned budget also rises, so the gap `cost − budget` stays near zero: **moving target problem**. The Lagrangian is fundamentally incapable of raising μ when the actual KL crisis occurs because its measurements don't reflect current KL, and its target follows the measurement.
- With μ pinned at 0.05, the BC ceiling removed as the training progresses, the model drifts in the PPO direction unchecked → late KL explosion.

**Verdict**: Implementation failure, not concept failure. The Lagrangian control loop was disabled by (a) auto-tuning the budget from the same EMA stream (creates moving target), (b) EMA β too aggressive (smooths out signal), and (c) step_mult magnitude caps limiting reaction. The μ floor at 0.05 prevented late-training rescue.

---

### v40b — NLL linear (Hypothesis B partially refuted, different bug found)

**What μ actually did** (fig 5):
μ starts at 0.300, decays only to 0.190 at step 25, **0.181 at step 50**, and **0.127 at step 100** — stays well above v24's 0.05 floor throughout. Full-trajectory MAE is 0.094 — **the largest of any variant**, driven by persistent over-imitation in the mid-late phase.

**What the adaptive signal did** (fig 5 panels 1-2):
- `chord/sft_loss` is **not** contaminated by zero micro-batches at the logged granularity: zero-fraction=0%, min=0.29, max=1.89, mean=0.79. So the pre-flight audit 5.2 warning about 65% zero-valued micro-batches is either averaged away at the rank/global level or was a misdiagnosis.
- `nll_ema` decays slowly: 1.59 (s1) → 1.21 (s10) → 0.90 (s25) → 0.49 (s100). Never crosses below τ=0.65 until ~step 60.
- Applying the linear mapping μ = 0.02 + 0.156·NLL: at s25 μ ≈ 0.160 (matches observed 0.190); at s100 μ ≈ 0.096 (matches observed 0.127).
- Net effect: μ_eff follows NLL_ema with ~5 step lag. **NLL_ema decays far more slowly than v24's 25-step hand schedule**.

**Why it underperformed** (late-training over-imitation):
- Cumulative weighted_sft over 100 steps = **14.13** (vs v39b's 7.92 — 1.8× more BC pressure).
- Late-training weighted_sft = 0.088/step (vs v39b's 0.028 — 3× higher).
- Late grad_norm mean = 7.20 (vs v39b's 3.01), with a spike to 16.06 at step 75. Large conflicting gradients between SFT (pulling toward teacher) and PPO (pulling toward current-policy reward).
- Final reward 0.620 (vs v39b 0.697) and best success 0.211 (vs v39b 0.246). Stable (only 2% KL>1) but performance-limited.

**Verdict**: Implementation works; design is wrong. The linear map `0.02 + 0.156·NLL` maps NLL=0.65 → μ=0.121 (above valley), so μ never reaches v24's valley=0.05. Over-imitation pulls the policy away from reward. Fixable with a steeper map (`μ = μ_min + (μ_max - μ_min) · clip((NLL - 0.5)/(1.8 - 0.5), 0, 1)`) or an additional floor-anchoring term.

---

### v39b — disc_acc EMA α=0.5 (Hypothesis C: best match in shape AND best performance)

**What μ actually did** (fig 1):
μ: 0.300 (s1) → 0.202 (s10) → 0.108 (s25) → 0.057 (s50) → 0.060 (s100). Shape tracks v24 closely (MAE 0.013). Notably **higher early-mid than v43a** (μ=0.108 at s25 vs v43a's 0.084), giving more BC pressure during the critical consolidation window.

**What the adaptive signal did** (fig 3 panel 1):
- `chord/disc_acc_ema`: starts low (discriminator untrained), ramps to 0.855 by s25 (v39) / 0.884 (v39b).
- α=0.5 (v39b) responds 2.5× faster than α=0.2 (v39), so μ decays more sharply early. Observed: μ at s15 = 0.185 (v39b) vs 0.224 (v39) — v39b arrives at moderate μ earlier, matching v24's schedule better in the ramp phase.

**Why it outperformed everyone else**:
- Highest time-above-μ=0.10 in window s5–30 **among well-performing variants**: 18 steps (tied with v39, above v24's 13 and v43a's 15). This gives strong early BC.
- Stable late training: only 2 KL>1.0 spikes (vs v43a's 23). Late mean grad_norm = 3.01 (lowest).
- Best-smoothed success = 0.118 (best across variants). Last-10 success = 0.109.
- Crucially: μ decays _based on discriminator state_, i.e., on observable evidence that the policy has actually become distinguishable from teacher. This is self-gating — if the policy is still close to teacher (disc_acc low), μ stays high; once policy diverges (disc_acc → 1), μ drops. It's a _correct_ feedback loop.

**Verdict**: Works as designed. The disc_acc signal is well-calibrated (goes from ~0.5 at init to 0.85+ by step 25, matching the implied window of v24's schedule), and α=0.5 is fast enough to track the discriminator's training dynamics without over-smoothing. Hypothesis C is **confirmed**.

---

### v41b — ESS saturating (Hypothesis D CONFIRMED with catastrophic severity)

**What μ actually did** (fig 1):
μ at s1 = **0.050** (already at floor!), briefly jumps to 0.113–0.120 at s2–s3, then collapses back to 0.050 for **94% of all steps**. This is the worst mu trajectory by a large margin.

**What the adaptive signal did** (fig 3 panel 4):
- `ess_ratio` starts at 1.000 (cold-start), climbs to 6.5 by step 25 and stays there.
- `ess_ema` starts at 1.0, reaches 30+ by step 25.
- The saturating map clearly interprets "high ESS = healthy sampling" as "no need to imitate", collapsing μ to the valley immediately.
- The audit 5.3 warning is empirically confirmed: ESS responds to _degradation_ (when ESS drops, BC increases), which is a _corrective_ framing. But in practice the ESS stays high throughout normal training, so the saturating map never fires — the variant sees no reason to imitate from step 1.

**Why it underperformed**:
- **Zero** steps above μ=0.10 in s5-30 — essentially ran as pure GRPO.
- 14% of late steps have KL>1 (second-worst after v43a), indicating the lack of BC anchoring let the policy drift substantially.
- Best success 0.143, last-10 success 0.026, last-10 reward 0.592 (worst of all variants).

**Verdict**: Design failure, not implementation. The ESS map semantics are wrong: ESS measures off-policy sampling health, not imitation needs. Early in training the buffer is full of similar-to-teacher samples (ESS is high because teacher and policy agree), which the map misreads as "no BC needed". In reality, that's exactly when BC is most useful. Rescue would require inverting the polarity: `μ = μ_max - (μ_max - μ_min) · g(ess_ratio)` where `g` saturates at 1 for high ESS. But that's still a crude proxy — disc_acc directly measures what we care about (policy-teacher separability).

---

## 2. Cross-variant comparison — what does v39b share with success?

**What the winners have in common** (v24, v39b):
1. **Sufficient early BC**: μ stays above 0.10 for ≥13 of steps 5–30. This lets the policy acquire teacher-like patterns before PPO drift kicks in.
2. **Rapid convergence to a low floor**: μ reaches 0.05–0.06 by step 50. Late-training BC pressure is minimal, so PPO dominates gradient updates.
3. **Late-training KL stability**: <5% of late steps have KL>1. The policy update is contained, gradient magnitudes stay bounded.
4. **Effective `weighted_sft` late = 0.028–0.030**. This is the target "dosage" for the late phase.

**What the failures have in common** (v40b, v41b, v43a):
- v40b: μ too high late (0.13–0.18), effective `weighted_sft` = 0.088 (3× target). Over-imitation → grad conflicts → stunted reward. Stable but under-performing.
- v41b: μ too low everywhere (0.05 throughout), zero early BC. Under-imitation → KL drift, failed exploration.
- v43a: μ shape correct but KL control loop broken. 46% of late steps have KL>1. Correct BC pressure but no safety net when policy diverges.

**Why v39b beats v43a despite similar μ**: The two variants have μ trajectories that track v24 to within 0.013 and 0.009 MAE respectively. But v39b's disc_acc signal is an _orthogonal_ diagnostic that also validates the discriminator is learning — if it wasn't, μ wouldn't decay; the signal is intrinsically self-correcting. v43a's Lagrangian is trying to control KL via μ, but (1) its cost measurement lags actual KL, (2) its budget moves with the cost, and (3) its step_mult is capped. So when actual KL diverges, μ does nothing. In other words: **v39b degrades gracefully if the signal is wrong; v43a doesn't**.

---

## 3. Theory vs empirics — what went wrong in the predictions

**Theory said**: Lagrangian > NLL > ESS > disc_acc, because:
- Lagrangian directly bounds the quantity we care about (KL drift)
- NLL is a direct measure of how teacher-like the policy currently is
- ESS captures off-policy sample quality
- disc_acc is a _derivative_ quantity (how separable are the two distributions), theoretically less principled

**Empirics said**: disc_acc > {Lagrangian, NLL} > ESS.

**What went wrong**:

1. **Lagrangian was implementation-disabled.** Three design decisions collectively nullified dual ascent: (a) EMA-smoothed cost creates ~2-step-window lag, so cost_ema doesn't reflect current KL; (b) auto-tuned budget from the same EMA stream creates a moving target (cost ≈ budget perpetually); (c) step_mult clamped to [0.95, 1.035]. Any one fix alone wouldn't save it — all three are needed. A _correctly_ implemented Lagrangian with hard budget and faster EMA might outperform disc_acc, but it's a non-trivial engineering lift.

2. **NLL's linear map had wrong calibration.** The choice μ = 0.02 + 0.156·NLL was derived to produce μ_peak=0.30 at NLL=1.8. But 1.8 was too conservative for the cold-start signal (observed nll_ema starts at 1.59 — already "in-range" for the linear map), so μ never escaped the upper plateau. NLL semantics are right; calibration is wrong.

3. **ESS's semantics are wrong.** High ESS = low divergence between policy and teacher = "easy" off-policy learning. But the adaptive μ needs to respond to _how much BC we want_, not how _easy_ the current learning is. These are anti-correlated: when ESS is high, policy is close to teacher and BC could safely be high; when ESS is low, policy has drifted and BC is risky. The saturating map has the polarity wrong for the early-training regime where ESS is already high.

4. **disc_acc lucked into the right semantics.** The discriminator's confusion (acc = 0.5) means "policy looks like teacher" — we want lots of BC. Certainty (acc = 1.0) means "policy has diverged" — we want less BC. The signal is a direct, observable measure of the _current_ distance between the two distributions. No lag (discriminator is re-trained every step from fresh data), no calibration drift (acc is a bounded probability). And the α=0.5 EMA is aggressive enough to track the discriminator's ramp-up but smooth enough to avoid single-batch noise.

**Is it fixable?** Lagrangian is fixable but requires fundamental redesign (separate faster cost tracker for control, fixed target budget rather than auto-tuned, asymmetric response — aggressive when cost > budget, gentle when below). NLL is fixable with a 2-parameter re-calibration. ESS requires polarity inversion, which might work but doesn't match v39b's principled semantics. **disc_acc with α=0.5 is the shortest path to v24-level performance.**

---

## 4. Recommendations

**Can v43a / v40b be rescued with bug fixes?**

v43a (Lagrangian), minimum viable fix (est. 1 day):
1. Use `raw kl_loss` directly (not EMA) as the cost signal, or EMA with α≥0.5.
2. Fix budget at a hyperparameter (e.g., 0.4) rather than auto-tuning. Remove `kl_budget_ema`.
3. Remove step_mult cap at [0.95, 1.035]; allow up to [0.5, 2.0] to get meaningful response.
4. Asymmetric response: expand eta when cost > budget, shrink more slowly when below (so drift corrections are aggressive).
Expected outcome: KL stability similar to v39b (≤5% late KL>1), μ should self-regulate into v24-like schedule when training is healthy.

v40b (NLL), minimum viable fix (est. 0.5 days):
1. Re-calibrate linear map to saturate early: `μ = μ_min + (μ_max - μ_min) · clip((NLL - 0.5)/(1.8 - 0.5), 0, 1)` with τ=0.5, ceil=1.8.
2. Alternative: log-map `μ = μ_min · (μ_max/μ_min)^clip((NLL - 0.5)/(1.8 - 0.5), 0, 1)` for smoother decay.
Expected outcome: μ at s50 should hit 0.05 instead of 0.18, removing over-imitation.

v41b (ESS), minimum viable fix (est. 1 day):
Invert polarity is a band-aid, not a fix. Recommend deprecating this variant unless we find a semantic interpretation for "low ESS → more BC" that doesn't cause perverse feedback (high BC → lower ESS → more BC → runaway).

**Fastest path to v39b-on-ALFWorld**:

Given v39b's apparent robustness to hyperparameter choices (α∈{0.2, 0.5} both produce reasonable μ shapes; α=0.5 is slightly better), port exactly:
- Copy v39b's full config to `alfworld_qwen1.5b_duet_v39b.yaml`.
- Change env/data paths only (model, teacher data, tasks). Keep all CHORD/DR3/SC parameters identical.
- Validation checkpoint: verify `chord/disc_acc_ema` reaches 0.85+ by step 25 on ALFWorld (this is the _necessary_ signal sanity check). If not, the discriminator is not learning → no adaptive μ schedule. This is the single most important pre-flight check.
- Expected behavior: μ 0.30 → 0.05 by step ~40–50, last-10 success > 0.10, late KL <0.5.

If ALFWorld's discriminator converges more slowly (e.g., disc_acc still at 0.70 by step 25), that's a sign of easier separation between teacher/policy on that env, and α=0.3 (between v39 and v39b) might be preferable.

Do **not** run v43a-style Lagrangian or v41b-style ESS on ALFWorld without the fixes above. v40b-style NLL could be informative but is an ablation, not a production candidate.

---

## Figures generated

All saved under `/data/home/qisheng/EvolAnalsis/analysis_reports/figures/`:

- `fig_phase1_mu_trajectories.png` — μ overlay for all 6 variants + v24 implied schedule. Shows v39b matches v24 well, v40b stays too high, v41b collapses to floor.
- `fig_phase1_performance_metrics.png` — grad_norm, kl_loss, entropy_loss, rewards_onpolicy. Shows v43a's late KL explosion and v40b's grad_norm spikes.
- `fig_phase1_signals.png` — each variant's own adaptive signal trajectory.
- `fig_v43a_lagrangian_diagnostic.png` — cost_ema vs budget (near-identical), Lagrange state, applied μ. Visually confirms dual ascent is inert.
- `fig_v40b_nll_pollution.png` — sft_loss raw, nll_ema, applied μ. Confirms no zero-pollution but slow NLL decay drives over-imitation.

Raw numeric data: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/phase1_summary.json`, `phase1_rows.json`, `phase1_mae_vs_v24.json`.
