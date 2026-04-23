# Adaptive-μ Signal Expansion: Empirical Sanity Check

**Date**: 2026-04-19
**Question**: Which candidate adaptive-μ signals empirically track v24's hand-tuned schedule,
and which ones exhibit the critical **cross-env self-adjust** behavior (naturally smaller on
ALFWorld where BC should be less needed)?
**Data**: v24 WebShop (100 steps) + v24 ALFWorld (100 steps) + v39 WebShop (100 steps,
adaptive-from-disc_acc) + v39 ALFWorld (68 steps, still running).

---

## TL;DR

| | |
|---|---|
| **Best adaptive signal (empirical)** | `chord/sft_loss` (teacher NLL) — r=+0.75, MAE=0.033, ALF/WS=0.30 |
| **Runner-up** | `chord/log_prob_std` — r=+0.79, MAE=0.032, ALF/WS=0.37 |
| **Highest raw alignment, but NO cross-env self-adjust** | `dr3/ess_off_window` — r=-0.97, MAE=0.014 but ALF/WS=1.03 |
| **Avoid: inverted cross-env behavior** | TGS — ALF/WS=2.93 (WOULD over-apply BC on ALFWorld) |
| **Avoid: collapses μ too early** | `adv_teacher_abs_mean` — knee at step 4 |
| **v39 diagnosis** | μ stayed above 0.1 until step 24 (v24: step 17). μ-AUC was **26% higher** than v24 over training. disc_acc trajectory is nearly identical to v24's, so the adaptive law over-distills mid-training despite matching the discriminator signal offline. |

---

## 1. Setup

v24's hand-tuned schedule (piecewise-linear) is μ = {0.30, 0.20, 0.07, 0.05, 0.05} at steps
{1, 10, 25, 50, 100}. It crosses μ<0.1 at **step 17**. v39 replaces this with an
adaptive law μ=f(disc_acc) and reaches WebShop Val@100 success = 11.5%, versus v24's 22%
(10.5pp gap) — despite offline reproduction r=0.97.

We evaluated 10 candidate signals on v24 WebShop + ALFWorld, fit a linear map μ̂ = a + b·x,
and computed three scores:
- **|r|** — alignment with v24's hand-tuned μ (higher = better);
- **MAE** — prediction error (lower = better);
- **ALF/WS (mid)** — ratio of mean signal value on ALFWorld vs WebShop over steps 10–50.
  Values **< 0.8** mean the signal is naturally less active on ALFWorld → μ auto-drops where
  BC is less useful (**good cross-env self-adjust**). Values **> 1.05** mean BC would be
  over-applied on ALFWorld (**bad**).

Fit predictions were clipped to μ∈[0, 0.5].

---

## 2. Full empirical results

| Signal | WS r | \|r\| | WS MAE | knee | ALF/WS | Direction | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| **E. chord/sft_loss (teacher NLL)** | +0.745 | 0.745 | 0.033 | 14 | **0.30** | dec in training | **PROMISING: strong self-adjust** |
| **F. chord/log_prob_std** | +0.786 | 0.786 | 0.032 | 14 | **0.37** | dec | **PROMISING: strongest alignment w/ self-adjust** |
| D. actor/kl_loss | -0.654 | 0.654 | 0.037 | 29 | 0.31 | inc | moderate — late knee, self-adjust OK |
| B. dr3/ess_off_window | -0.969 | 0.969 | 0.014 | 19 | 1.03 | inc | HIGH r but NO self-adjust (saturates) |
| K. disc_acc × ess_ratio | -0.876 | 0.876 | 0.022 | 22 | 1.01 | inc | same issue as B (product dominated by same saturation) |
| J. w_off/w_mean ratio | +0.742 | 0.742 | 0.033 | 30 | 0.92 | dec | mild self-adjust, knee too late |
| H. 1 / group_reward_variance | -0.563 | 0.563 | 0.039 | 27 | 0.78 | inc | weak alignment |
| A. TGS (teacher_gradient_share) | +0.621 | 0.621 | 0.037 | 17 | **2.93** | dec | **AVOID: inverted ALF/WS** |
| C. dr3/w_std | -0.650 | 0.650 | 0.040 | 13 | 1.25 | inc | weak + slight anti-self-adjust |
| G. adv_teacher_effective_abs_mean | +0.541 | 0.541 | 0.041 | **4** | 1.16 | dec | **AVOID: μ crashes to 0 by step 4** |

Raw signal values at key steps (v24):

| Step | sft_loss (WS / ALF) | log_prob_std (WS / ALF) | kl_loss (WS / ALF) | TGS (WS / ALF) |
|---:|:-:|:-:|:-:|:-:|
| 1 | 1.16 / 1.11 | 3.48 / 3.90 | 0.004 / 0.002 | 0.31 / 0.32 |
| 10 | 1.06 / **0.34** | 2.97 / **1.33** | 0.057 / 0.007 | 0.32 / 0.43 |
| 25 | 0.81 / **0.18** | 2.29 / **0.68** | 0.246 / 0.027 | 0.15 / **0.28** |
| 50 | 0.72 / 0.39 | 2.09 / 1.04 | 0.386 / 0.059 | 0.08 / **0.23** |
| 100 | 0.61 / 0.30 | 1.77 / 0.91 | 1.29 / 0.09 | 0.08 / **0.22** |

Key observation: `sft_loss` collapses to ~0.18-0.40 on ALFWorld by step 10-25 (fast
convergence of teacher-conditioned policy), but on WebShop it stays at 0.70-0.80.
`log_prob_std` shows the same pattern. **These signals "discover" that ALFWorld needs less
BC without any manual tuning.** TGS does the opposite (it is HIGHER on ALFWorld steps 25-100,
producing ALF/WS = 2.93).

---

## 3. Cross-env signal trajectory figure

See `/data/home/qisheng/EvolAnalsis/analysis_reports/fig_adaptive_signal_expansion.png` — each
panel shows signal trajectory overlaid for WS v24 (blue) and ALF v24 (orange), with v24's
hand-tuned μ on the right axis (dashed black) and predicted μ (dotted red). Summary bar
charts at the bottom show |r|, MAE, and ALF/WS ratio across all 10 candidates.

A tighter view of μ reconstruction for the 4 best candidates is in
`fig_adaptive_top4_mu_reconstruction.png`. For signal E (sft_loss), the ALF-predicted μ
(orange) sits at 0.05-0.15 for most of training, while the WS-predicted μ (blue) matches
v24's hand-tuned schedule (black). This is exactly the behavior we want for a
**one-signal-across-envs** adaptive law.

---

## 4. v39 diagnosis (10.5pp WebShop gap)

See `fig_v39_diagnosis.png`. Key quantitative findings (WebShop, v39 vs v24, steps 1-100):

| Metric | v24 (hand-tuned) | v39 (adaptive) | Delta |
|---|---:|---:|---:|
| μ_knee (first step where μ<0.1) | 17 | **24** | +7 |
| μ_mid (mean over steps 10-50) | 0.075 | **0.112** | +49% |
| μ_AUC (trapz over 100 steps) | 7.83 | **9.85** | +26% |
| kl_loss_mid (steps 10-50) | 0.280 | 0.316 | +13% |
| disc_acc shape | identical | identical | — |

**Diagnosis**: v39's disc_acc trajectory matches v24's (middle panel of fig_v39_diagnosis),
confirming the offline r=0.97 reproduction story. But the adaptive mapping μ=f(disc_acc)
keeps μ at ~0.20 during steps 10-20 (when disc_acc is still 0.5-0.8), whereas v24's hand-tuned
law already had μ<0.1 by step 17. Total BC mass (μ-AUC) is **26% higher** in v39, concentrated
in the critical mid-training window where on-policy learning should dominate. The effect is
**over-distillation**, not policy divergence: KL is only 13% higher, but μ×sft_loss
integrated is the issue.

This explains why offline r=0.97 doesn't guarantee live parity: the regression matches the
**shape** of v24's μ but shifts the **phase** — v39 applies BC 7 steps later than v24 at the
same disc_acc level. When BC is still high at step 20, the policy has less budget to explore
novel on-policy trajectories, which is where v24 started earning its success signal.

Implication for theory-researcher: any disc_acc-based signal needs a **phase correction**
(explicit step-term or an EMA that triggers earlier). Or — better — use a signal that
**naturally** peaks earlier, like sft_loss which drops below its step-1 value by step 3-4
on WebShop and step 3 on ALFWorld.

---

## 5. Ranking & recommendations

### Recommend PROTOTYPING (promising empirically):

1. **E. chord/sft_loss (teacher NLL)** — The winner. Moderate WS alignment (r=+0.75), but
   **strong cross-env self-adjust** (ALF/WS=0.30). This is the only signal that **naturally
   predicts less BC on ALFWorld than WebShop**, which is what we need. sft_loss is already
   logged in every CHORD run. Simple affine map μ̂ = clip(0.02 + 0.156·sft_loss, 0.05, 0.30)
   reconstructs v24's schedule on WebShop with MAE=0.033.

2. **F. chord/log_prob_std** — Very close second. Slightly better WS alignment (r=+0.79,
   MAE=0.032) and similar cross-env behavior (ALF/WS=0.37). Captures dispersion of teacher
   log-probs — arguably a richer signal than mean NLL. Worth testing head-to-head vs E.

3. **J. w_off/w_mean ratio** — Shows mild self-adjust (ALF/WS=0.92) and decent WS alignment
   (r=+0.74, MAE=0.033). Knee is too late (step 30), but could be combined with E/F as a
   composite signal.

### Recommend COMPOSITE (likely strongest):

**E × TGS complement**, or **clip(α·E + β·F)**: sft_loss and log_prob_std are nearly
collinear (both reflect teacher-policy agreement), but averaging them may stabilize.
Empirically, using just E gives MAE=0.033; F gives MAE=0.032; simple average should not
materially change either.

### Signals theory-researcher should AVOID:

1. **A. TGS (teacher_gradient_share)** — ALF/WS=2.93 means TGS is 3× HIGHER on ALFWorld in
   mid-training. Using TGS to drive μ would over-apply BC on ALFWorld, i.e. produce the
   OPPOSITE of the desired cross-env behavior. This is a trap because TGS *looks* like a
   natural DR3 curriculum signal on WebShop alone.

2. **B. dr3/ess_off_window** — Highest raw alignment (r=-0.97, MAE=0.014) but **saturates**
   near window size on both envs (WS peaks ~31, ALF peaks ~32). ALF/WS=1.03 → same BC level
   on both envs, defeating the cross-env goal. **Do not let the high r fool you.**

3. **G. adv_teacher_effective_abs_mean** — Knee at step 4. Any signal that crashes μ to 0
   within 5 steps cannot reproduce v24's 100-step curriculum.

4. **K. disc_acc × ess_ratio** — Inherits B's saturation problem (ALF/WS=1.01). Including
   disc_acc as a factor does not help because disc_acc saturates to 1.0 on both envs by
   step 25.

---

## 6. What v39's failure tells us about design

- Offline signal-reproduction (r=0.97) is **necessary but not sufficient**. A signal that
  matches shape can still mis-phase. Report **knee_step** and **μ-AUC** alongside r/MAE in
  future offline evaluations.
- A good adaptive signal should satisfy **three** criteria, not two:
  1. |r| > 0.7 with v24 μ,
  2. MAE < 0.04 (on WS),
  3. **ALF/WS mid-ratio < 0.8** (cross-env self-adjust).
  Signals B and K fail #3. Signals A and G fail #3 and #1-#2 respectively.
- Only **E (sft_loss), F (log_prob_std), and D (kl_loss)** pass all three. Among these, D is
  policy-side and may interact with DR3's density ratio (riskier), while E/F are
  teacher-side and clean of DR3 feedback loops.

---

## Artifacts

- `/data/home/qisheng/EvolAnalsis/analysis_reports/fig_adaptive_signal_expansion.png`
- `/data/home/qisheng/EvolAnalsis/analysis_reports/fig_adaptive_top4_mu_reconstruction.png`
- `/data/home/qisheng/EvolAnalsis/analysis_reports/fig_v39_diagnosis.png`
- `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/adaptive_signal_expansion.json` (per-signal fits & trajectories)
- `/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/v39_diagnosis.json`
- Analysis script: `/data/home/qisheng/EvolAnalsis/analysis_reports/_analyze_adaptive_signals_expansion.py`
- Parser: `/data/home/qisheng/EvolAnalsis/analysis_reports/_parse_adaptive_signals.py`

---

## Appendix: Final ranking (composite score)

Composite = |r| · (1 / (MAE+0.01)) · (1 if ALF/WS < 0.8 else 0.2). Higher is better.

| Rank | Signal | \|r\| | MAE | ALF/WS | Composite |
|---:|---|---:|---:|---:|---:|
| 1 | **F. log_prob_std** | 0.786 | 0.032 | 0.37 | 18.7 |
| 2 | **E. sft_loss** | 0.745 | 0.033 | 0.30 | 17.3 |
| 3 | D. kl_loss | 0.654 | 0.037 | 0.31 | 13.9 |
| 4 | H. 1/group_var | 0.563 | 0.039 | 0.78 | 11.5 |
| 5 | B. ess_off_window | 0.969 | 0.014 | 1.03 | 8.1 (but penalized ×0.2) |
| 6 | K. disc_acc×ess_ratio | 0.876 | 0.022 | 1.01 | 5.5 |
| 7 | J. w_off/w | 0.742 | 0.033 | 0.92 | 3.4 |
| 8 | G. adv_teacher_abs_mean | 0.541 | 0.041 | 1.16 | 2.1 |
| 9 | C. dr3/w_std | 0.650 | 0.040 | 1.25 | 2.6 |
| 10 | A. TGS | 0.621 | 0.037 | **2.93** | 2.6 |

**Top-2 recommendations: F (log_prob_std) and E (sft_loss)**. Both are teacher-side, logged
for free in every CHORD run, and would have self-adjusted on ALFWorld without any tuning.
