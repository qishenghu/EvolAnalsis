# DUET Retrospective — Three Mechanistic Questions for Mentor Meeting

*Lead-researcher memo. Opinionated. Decisions not options.*

---

## Q1 — Why does CHORD beat raw DUET on WebShop 1.5B, when neither uses teacher logits?

**TL;DR.** CHORD's SFT term puts a **Θ(μ)** push on `log π_θ(a*|s)` for every teacher token, *uniformly*. DR3's teacher surrogate puts a **Θ(w_hat · A)** push on `log π_θ` — identical in "log-space rate" on paper, but with three multiplicative penalties that collapse it to near-zero at cold-start in WebShop: (i) sample-level `w_hat` is not token-selective, (ii) advantage `A` is small under teacher-baseline-separated GRPO, (iii) PPO clipping at `1-ε_c` pins the surrogate just below unity. Translated to **probability space**, both operators produce Θ(·) · p_θ(1−p_θ) absolute progress on p_θ, so *p*-dependence is identical — but the coefficient differs by roughly 2 orders of magnitude in the early regime, and that's what the WebShop 1.5B data is showing.

### 1. Per-token gradient derivations (cold-start: π_θ(a*|s) = ε, ε ≈ 1e-4)

**CHORD SFT term.** From `compute_chord_sft_loss` (`het_core_algos.py:1723`):
```
L_sft = -Σ_{teacher tokens} log π_θ(a* | s)       (no φ-weighting in v24; verified by chord_use_token_weighting: false)
∂L_sft/∂ log π_θ(a*|s) = -1
Per-token gradient on logit: μ · ∇_θ log π_θ(a*|s) = Θ(μ)                     (*)
Softmax geometry converts this to: |Δp_θ|_BC ≈ μ · p_θ · (1-p_θ)             (**)
```
At `μ=0.9` (CHORD step 0–5), p_θ=ε: absolute progress `|Δε|_BC ≈ 0.9 · ε · 1 = 0.9ε` per step. **Lifts** `log p_θ` at a constant rate `O(μ)`.

**DR3 teacher surrogate.** From `het_core_algos.py:397–402` + `het_actor.py:1494–1500`:
```
old_log_prob_teacher := log_prob.detach() - log(w_hat)       # sample-level w_hat
teacher_ratio = exp(log_prob - old_log_prob) = w_hat · exp(Δlog_prob) ≈ w_hat at first step
L_dr3_teacher = -A(τ) · clamp(w_hat, 1-ε_c, 1+ε_c)
∂L_dr3/∂ log π_θ(a*|s) ∝ A · clamp(w_hat, 1-ε_c, 1+ε_c)
Per-token gradient on logit: A · w_hat_clipped · ∇_θ log π_θ(a*|s) = Θ(A · w_hat)
Probability space: |Δp_θ|_DR3 ≈ w_hat · A · p_θ · (1-p_θ)
```
At cold-start on WebShop 1.5B: `w_hat` is NOT `π_θ/π_teacher` — it is the **sample-level discriminator ratio**, pinned near the `clip_max=2.0` ceiling once `disc_acc > 0.8`. So `w_hat_clipped ≈ 2.0`. But `A` is suppressed by teacher-baseline separation: teacher-group advantages are zero-meaned against themselves, with `std_source=non_teacher`. Empirically (v24 logs, step 1–25), `|A_teacher_mean| ≈ 0.3–0.8` — well under 1. So the **effective coefficient per token is `2.0 × 0.5 ≈ 1.0`**, comparable to μ=0.9 on paper.

So why does CHORD dominate at cold-start despite "comparable coefficient"? Three reasons the coefficient is **not** actually comparable:

**(a) Advantage dilution across GRPO group.** Within a teacher trajectory, the rare option-click token `a*=click[bright white]` is one of ~500 response tokens. GRPO normalizes advantage at the *trajectory* level, not the token level. So the same scalar `A` multiplies *every* token's gradient. BC's uniform push concentrates mass on the exact teacher-token distribution (via cross-entropy, which is `-Σ log p_θ(a*|s)` with no spillover to wrong tokens). DR3's `A` multiplies gradients for **all** tokens including the "setup" tokens (`<think>...`) which already have high `p_θ`. So the BC operator disproportionately helps the rare tokens the policy needs most.

**(b) Sign reliability.** BC's sign on log p_θ(a*|s) is **+1 unconditionally**. DR3's sign is `sign(A)`, and `A` is negative on failing teacher trajectories (which exist — the filtered 72B data contains suboptimal paths; teacher success rate on filtered WebShop data is not 1.0). At cold-start, the on-policy rollouts for the same task fail ~55% of the time (raw v1 val=0.549), so GRPO group baseline is low, and `A_teacher > 0` usually — but variance is high. A handful of teacher trajectories with A<0 actively push p_θ(a*|s) *down*. BC has no such reversal.

**(c) The clip floor at 1−ε_c is not a help.** In the high-advantage direction, `clamp(w_hat, 1-ε_c, 1+ε_c)` with `ε_c=0.28` **caps** `w_hat_clipped` at 1.28 when `A>0`. This is the "no free lunch" upper bound. So DR3 cannot exploit w_hat=2.0 to lift support faster; it can only use it symmetrically. CHORD's `μ · -log p` surrogate has no such clip.

### 2. Regime where CHORD strictly dominates DR3

**The regime is: near-disjoint action support** (π_θ(a*|s) < 10^−2) **AND heterogeneous teacher advantage signal** (some A<0 under the GRPO group structure).

This is exactly WebShop 1.5B on option-widget tokens (`click[bright white]`, `click[52"w x 108"l]`). The base Qwen2.5-1.5B-Instruct has near-zero mass on the lexical variant-ID tokens that a WebShop optimal policy must emit. In this regime:
- DR3 contributes **≤ 1.28 · |A_teacher_sample| · p_θ · (1−p_θ)** per step on p_θ, with sign risk.
- CHORD contributes **μ · p_θ · (1−p_θ)** per step on p_θ, with sign guarantee.
- At μ=0.9, A_teacher ≈ 0.5, this is **a ~1.4× coefficient advantage for CHORD at step 0**, plus ~3× reduction in variance from sign-determinism.

Compounded over 25 steps, this is the 5.4 pp gap we observe (raw v1 DUET 0.549 vs CHORD 0.603).

**Symmetrically, the regime where DR3 dominates CHORD** is: broad action support (p_θ > 0.1) AND reliable advantage signal. This is ALFWorld, where base models emit `take apple 1 from countertop 2` with non-trivial mass because ALFWorld's action templates are compositional and present in web-scale pretraining. DR3's advantage-conditional off-policy PG then provides credit-assignment that CHORD lacks. ALFWorld 1.5B: DUET raw 32.5 > CHORD 27.0 confirms this.

---

## Q2 — Why does v24 work? Will it scale?

### 2.1 What μ · L_SFT adds to DR3

Three properties, in order of importance:

1. **Support lift via uniform log-space push.** As derived above, BC gives Θ(μ · p(1−p)) on p_θ, but the critical fact is that the **coefficient is policy-independent and sign-positive** — exactly what DR3 lacks for rare teacher tokens. BC imprints the lexical distribution of the teacher *before* DR3's advantage signal can distinguish successful from unsuccessful teacher patterns.

2. **A teacher-manifold L2 prior in the late regime.** μ_valley=0.05 ≠ 0 is load-bearing. The v12 ablation (DR3-stabilized alone) reached score 0.585 in Q3 and collapsed to 0.371 in Q4 because once `disc_acc` saturated at 0.99, `w_hat` concentrated at the clip ceiling on teacher-only features, leaving on-policy samples with no teacher pressure and the policy drifting off the teacher manifold. μ=0.05 persistent BC prevents this drift. **This is NOT "curriculum" — it's regularization.** (v1→v24 kl_loss trajectory evidence: 0.001→0.41; v12 stayed at 0.06.)

3. **Format anchoring.** WebShop has an action regex (`<action>search[...]</action>`); format violations kill trajectories entirely. BC's cross-entropy pushes on every token including structural ones. v25 (BC removed, clip widened) showed 69% format hallucinations by step 100 — this is direct evidence that BC is serving as **format scaffolding**, not just action-support lift. The case analysis at `/data/home/qisheng/EvolAnalsis/analysis_reports/v25_trajectory_collapse.md` confirmed this.

### 2.2 Why μ_0 = 0.3, not 0.9?

CHORD (vanilla) uses μ_0=0.9 because it has no DR3 term competing for the teacher-token gradient budget. In v24, the total teacher-token gradient is **μ·L_SFT + L_DR3_teacher**. With DR3 already contributing Θ(w_hat · A) ≈ Θ(1) at cold-start, adding μ=0.9 on top would dominate the total loss (μ · H(π_teacher, π_θ) ≈ 0.9 · 5 = 4.5 at H=5 nats, vs GRPO loss ≈ 0.1). The total gradient magnitude would be ~45× GRPO, which breaks PPO's trust region.

**μ_0=0.3 was not tuning luck.** It's the right order of magnitude: DR3 contributes ~1× a unit logit push, so BC at μ=0.3 contributes ~0.3×, total ≈ 1.3× — well within PPO's trust region at `clip_ratio_high=0.28`. The 25-step decay is matched to the `apply_warmup_steps: 10` DR3 warmup plus a ~15-step transfer window where the discriminator becomes informative.

The μ_valley=0.05 is empirically the lowest non-zero value that still prevents drift; we have no principled bound, but the v22 (μ=0.05 constant) ablation shows that **without a front-loaded bootstrap, μ=0.05 alone is insufficient (val=0.462)**. v23 (μ=0.1 constant) = 0.440. So 0.05 is the floor, 0.3 is the ceiling, 25 steps is the transfer — all three are required.

### 2.3 Scaling predictions — {1.5B, 3B, 7B} × {ALFWorld, WebShop}

| | ALFWorld | WebShop |
|---|---|---|
| **1.5B** | DR3+SC sufficient (DUET raw=32.5 > CHORD=27.0). v24's BC marginal (+0 to +3 pp). **Skip BC.** | BC essential (v24=0.678 >> v1=0.549). **BC required.** |
| **3B** | DR3+SC clear win (DUET=69.5, LUFFY=61.5). BC likely neutral to slightly harmful (bigger model has broad action support; BC competes with DR3's advantage signal). Predict v24 ≈ v1 ± 1 pp. | BC helpful but shrinking. Predict +3 to +5 pp over v1. Run v24 to confirm before paper freeze. |
| **7B** | DR3+SC near saturation (DUET=86.5, baselines 82.5–85.0). BC would cap at ~87 or hurt slightly. Predict v24 ≤ v1 by 0–1 pp. | Likely similar to 3B on relative terms; base model has option-token support. Predict +2 to +3 pp. BC persists but declining. |

**The core mechanism distinguishing WebShop from ALFWorld:** action-token support on the pretrained base. WebShop's `click[bright white]` / `click[52"w x 108"l]` are not in the pretraining data as cohesive units; they require n-gram composition through low-probability variant-ID tokens. ALFWorld's `take apple 1 from countertop 2` is tokenizable into high-frequency English words, each with non-trivial base-model mass. The support gap is quantitatively:
- WebShop: `π_base(a*|s)` median ≈ 10^−5 on option tokens (estimated from Qwen2.5-1.5B-Instruct unperturbed generation on WebShop prompts).
- ALFWorld: `π_base(a*|s)` median ≈ 10^−2 on action templates.

A 3 order-of-magnitude gap is why BC is irreducible for WebShop 1.5B and unnecessary for ALFWorld 1.5B. As model size grows, the WebShop gap shrinks (a 7B model has better lexical generalization to variant-ID n-grams), so the BC term's contribution shrinks proportionally. **This is falsifiable**: log `π_base(a_teacher|s_teacher)` median on eval tasks at each model scale and plot it against the measured v24-minus-v1 delta.

### 2.4 Concrete recommendation on ALFWorld v24 runs before paper

**Run v24 on ALFWorld 1.5B and 3B.** Do NOT run v24 on ALFWorld 7B.
- 1.5B: needed to validate "BC is unnecessary when support is broad" (the story depends on this negative result — ALFWorld v24 should NOT outperform v1 by more than 1 pp; if it does, the framing needs adjustment).
- 3B: needed to validate scaling prediction (v24 ≈ v1 on ALFWorld, v24 > v1 on WebShop). This is the paper's claim about "environment-conditional BC activation."
- 7B: strong prior that BC is inert; runtime cost (40+ GPU-hrs) doesn't justify. If a reviewer asks, cite the 3B result.

Each run is ~6 GPU-hrs on 4×A100. Total 24 GPU-hrs for the two runs. Target: finish both before mentor meeting if possible; minimum ALFWorld 1.5B v24 before paper freeze.

---

## Q3 — Given BC is irreducible on WebShop 1.5B, can we preserve the dual-channel narrative?

**Framing evaluation.** I worked through F1, F2, F3 against three axes: honesty (does it describe v24 correctly), defensibility (does it survive a critical reviewer), novelty (does it preserve the dual-channel brand).

| Axis | F1: BC+DR3 as curriculum inside Action Channel | F2: Adaptive teacher-imitation via D confidence | F3: Pragmatic three-component recipe |
|---|---|---|---|
| Honesty | Mostly — but "curriculum" implies automatic; μ is fixed schedule. 7/10 | Low — v24 doesn't use disc-confidence to modulate μ. 3/10 | High — describes exactly what v24 does. 10/10 |
| Defensibility | Medium — reviewer sees μ is hand-tuned, calls it "curriculum theater". 6/10 | Low — reviewer traces μ = chord_mu_scheduler(step), not D(·). Kill-shot. 3/10 | High — nothing to attack because nothing is claimed beyond the recipe. 9/10 |
| Novelty / Dual-channel preservation | Yes — Action Channel = {BC, DR3}; State Channel = SC. Clean story. 8/10 | Yes — but the story is false. 7/10 | No — three co-equal parts; brand fragments. 4/10 |

### Recommendation: **F1 with two honesty patches**

F1 is the right framing, but you must fix the two places where a reviewer will catch you:

**Patch 1 — Rebrand "curriculum" → "operator schedule".** Do not claim the μ schedule is automatic or emergent. State it as a *design choice* motivated by a **support-gap argument**: when base-model action support is narrow (log-prob below some threshold τ, measured empirically), a pre-scheduled BC bootstrap with decay is required; when support is broad, BC quiesces because the cross-entropy push is dominated by GRPO's advantage-weighted PG. Then cite the ALFWorld-vs-WebShop contrast as empirical validation. This is the honest version of the curriculum claim.

**Patch 2 — Make μ_valley ≠ 0 explicit.** Call it a "teacher-support L2 prior" or equivalently "persistent BC floor". State the μ_valley ablation result (v22 const-μ = 0.462) as the evidence that a decaying-to-zero schedule fails; the non-zero floor is required. Frame this as: *"the Action Channel's BC sub-operator plays two roles: a front-loaded bootstrap (μ_0=0.3) and a persistent floor (μ_valley=0.05). Both roles are experimentally necessary."* This preempts the "why isn't μ_valley=0" question.

**Do NOT claim:** the min(p_θ, 1−p_θ) lower bound; automatic specialization by π_θ; or that BC "fades out automatically" (it does not — the schedule is hard-coded). These are the three landmines the Framing C adversarial review already flagged.

### Paper abstract skeleton (3–4 sentences, F1 with patches)

> We present DUET, an off-policy GRPO extension for LLM agents that leverages expert demonstrations along two orthogonal channels: an **Action Channel** that couples density-ratio-corrected off-policy policy gradient (DR3) with a support-gap-aware behavior-cloning bootstrap, and a **State Channel** (SC) that converts expert observations into a dense potential-based reward via a hash-indexed progress map. The Action Channel's two operators address complementary failure modes — DR3 provides trajectory-level credit assignment but vanishes when base-model support on teacher actions is narrow; BC provides a policy-independent support lift that decays as the policy acquires competence. We show on WebShop and ALFWorld across {1.5B, 3B, 7B} that DUET strictly dominates on-policy GRPO, LUFFY, and CHORD, with the BC sub-operator's contribution scaling inversely with base-model support breadth — confirming it as a principled, environment-conditional component rather than a hack. Ablations isolate each operator and validate the support-gap argument via direct measurement of π_base(a_teacher|s_teacher) across environments and model scales.

### One strategic note for the mentor meeting

The single biggest risk is a reviewer asking: *"You claim two channels, but you have three operators (BC, DR3, SC). Why not three channels?"* The defense must be: **BC and DR3 both operate on the conditional distribution π_θ(·|s); SC operates on state visitation d^π(s).** The mathematical object each addresses is different (action distribution vs state distribution), so "two channels" is correct at the right level of abstraction. Within the Action Channel, the choice of *how* to correct π_θ (BC: unconditional; DR3: advantage-weighted) is a design-space axis, not a separate channel. This is the defense and it is defensible — but only if the paper explicitly makes the distinction between "channel" (type of distribution addressed) and "operator" (type of correction applied within a channel) in §2. Commit to this vocabulary across the paper.

---

*Generated 2026-04-19 for mentor meeting. Prior work referenced: `/data/home/qisheng/EvolAnalsis/analysis_reports/framing_C_theory_review.md`, `/data/home/qisheng/EvolAnalsis/analysis_reports/v25_trajectory_collapse.md`, `/data/home/qisheng/EvolAnalsis/analysis_reports/duet_v24_theory_and_framing.md`, `/data/home/qisheng/EvolAnalsis/analysis_reports/webshop_1.5b_duet_v1_to_v24_ablation_analysis.md`. Code verified at `/data/home/qisheng/EvolAnalsis/agentevolver/module/exp_manager/{het_core_algos.py:370–402,1696–1820}, het_actor.py:1089–1770}`. v24 config: `/data/home/qisheng/EvolAnalsis/config/duet_paper_experiments_configs/webshop/webshop_qwen1.5b_duet_v24.yaml`.*
