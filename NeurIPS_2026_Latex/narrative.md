# DUET — Paper Narrative & Framing Decisions

**Last updated**: 2026-05-04
**Status**: Locked-in narrative for NeurIPS 2026 submission (deadline 2026-05-07)
**Purpose**: Reference document for the writing process. Captures the agreed-upon story arc, framing, and the rationale for key decisions so subsequent sessions stay consistent.

---

## Title

**DUET: Principled Experience Replay for LLM Agent Reinforcement Learning**

Alternatives considered and rejected:
- *"DUET: Self-Calibrating Teacher Utilization in LLM Agent RL"* — too narrow on auto-fade; loses the "fix LUFFY" framing.
- *"DUET: Two-Channel Experience Replay for LLM Agent RL"* — too architectural; misses the principled angle.
- *"Beyond Heuristic Schedules: Principled Off-Policy Correction..."* — too aggressive against CHORD authors who may end up reviewing.

**Why "principled" is the load-bearing word**: ML reviewers use *heuristic* (negative) vs *principled* (positive) as opposing labels. CHORD was criticized for its heuristic μ schedule. DUET answers with: every mechanism has a stated theoretical basis (importance sampling, potential-based shaping, baseline variance analysis); none is a manual schedule.

---

## Narrative Arc

The paper is framed as **a principled fix to LUFFY-style experience replay**, not as "yet another teacher-mixing method".

### Problem
On-policy RL (GRPO) on weak LLM agents collapses (1.0% / 0.5% on 1.5B AlfWorld / WebShop). Mixing teacher trajectories (LUFFY) helps the cold-start, but the existing instantiations have known shortcomings.

### Diagnosis (the paper's intellectual contribution)

We identify **two systematic biases** in LUFFY-style methods. This is the heart of the paper.

1. **Baseline contamination.** Teacher samples (always reward = 1) enter the GRPO group → group baseline μ_g is inflated and σ_g shrinks → successful on-policy rollouts receive small/negative advantage → exploration is systematically punished. This contradicts the very purpose of on-policy RL.

2. **Unaddressed off-policy mismatch.** Teacher trajectories are drawn from π_β ≠ π_θ. The GRPO importance ratio ρ = π_θ/π_old does not correct for the teacher–student gap. Teacher gradients continue to dominate even after the student catches up, damaging asymptotic performance and forcing CHORD-style manual decay schedules.

Both biases compound on weak base models (where the gap is largest and longest), explaining why prior teacher-mixing methods underperform there.

### DUET = principled fixes + extensions

| # | Mechanism | Role | Auto-calibration signal |
|---|---|---|---|
| 1 | **Baseline separation** (Fix 1) | Compute GRPO baseline separately for teacher vs on-policy sub-groups | (always on; not subject to fade) |
| 2 | **DR3** density-ratio correction (Fix 2) | Discriminator estimates ŵ = π_θ/π_β; replaces ρ for teacher samples | ŵ → 1 as student catches up |
| 3 | **BC** adaptive imitation (Extension 1) | Token-level CE on teacher tokens, weighted by μ(t). **Cold-start safety net while DR3 stabilizes.** | μ(t) ↓ as disc_acc ↑ |
| 4 | **SC** state progress shaping (Extension 2) | Hash-based progress map P(s); per-step reward bonus β·ΔP, on-policy only | hash hit rate ↓ for OOD states |

### Two-Channel Architecture (orthogonal axis)

- **Action Channel** = DR3 + BC (both leverage teacher action distributions; modify policy gradient)
- **State Channel** = SC (leverages teacher state visitation; modifies reward)

The two channels are orthogonal axes of teacher utilization. The 4 mechanisms map to (Fix 1: orthogonal to channels; Fix 2 + Ext 1: Action; Ext 2: State).

---

## Key Framing Decisions (locked 2026-05-04)

### "Principled" — what each mechanism cashes out as

| Mechanism | Principled justification | Reference |
|---|---|---|
| Baseline separation | GRPO advantage variance argument: teacher inclusion biases group baseline; separating restores zero-mean advantage in on-policy sub-group | Self-derived (one paragraph in method) |
| DR3 | ŵ = D/(1−D) is the standard discriminator-based density-ratio estimator; ŵ → 1 is a fixed-point as π_θ → π_β | Goodfellow et al. 2014; Sugiyama et al. density-ratio estimation |
| BC adaptive μ | Cold-start safety net design: high μ when DR3 unreliable (low disc_acc), fades as DR3 stabilizes — design rationale, not external theorem | Self-derived |
| SC | Potential-based reward shaping; policy invariance theorem | **Ng, Harada & Russell 1999** |

**Important**: BC's μ schedule is **not** framed as JSD-driven distillation pressure. See below.

### Why μ is NOT framed as JSD-driven

The earlier draft proposed μ ∝ disc_acc as JSD-driven distillation pressure. This was **discarded** because:

- Empirically, our disc_acc is **monotonically increasing** across training (it reflects discriminator capacity growth as much as student–teacher gap).
- The JSD interpretation requires disc_acc to track current gap; reviewers can falsify this with our own logs.
- Better, more honest framing: **μ is a cold-start safety mechanism complementing DR3.** Early disc_acc is low → ŵ noisy → DR3 alone is risky → BC provides token-level dense supervision as a safety net. As disc_acc rises → DR3 becomes reliable → BC fades, transferring control to DR3.

This re-framing also clarifies the BC/DR3 relationship in the paper: not two parallel mechanisms, but **main + safety-net**.

### LUFFY reproducibility — honest reporting

| Source | 3B WebShop strict success rate |
|---|---|
| Original LUFFY paper | 49.5% |
| Our L20X-144G server reproduction | 38.0% |
| Our 4×A100 reproduction | 3.5% strict / 11.5% lenient (≥0.9) |

**Decision**: Main table reports **L20X 38.0%** (most conservative against ourselves; we don't want to overstate DUET's lead). Appendix gives a 3-way comparison + reproducibility discussion as a separate contribution.

### Single-seed caveat

All numbers are single-seed (no time for multi-seed by 2026-05-07). Mitigation:
- Use binomial 95% CI on val@200 success rate as proxy stability estimate (e.g., 47.5% vs 30.0% on n=200 has non-overlapping CIs).
- Discuss limitation explicitly in §Limitations.

---

## Story Arc for Introduction (5 paragraphs)

1. **Problem**: cold-start trap on weak LLM agents; concrete numbers.
2. **Existing**: LUFFY/CHORD as natural mitigations; CHORD's reviewer-criticized heuristic schedule.
3. **Diagnosis**: 2 biases (paper's intellectual contribution; this is the paragraph that buys reviewer respect).
4. **DUET**: principled framework with 4 self-calibrating mechanisms; introduce the two channels.
5. **Results & contributions**: 4/4 SOTA, +13pp avg, +17.5pp on weak; bullet contribution list.

---

## Notes for Method Section Structure

- §3.1 Problem setup (GRPO recap + LUFFY mixing recap)
- §3.2 Diagnosis: two biases (formal restatement with notation)
- §3.3 DUET overview (channels + mechanism map)
- §3.4 Baseline separation (Fix 1) — short, ~½ page
- §3.5 DR3 density-ratio correction (Fix 2) — longest, ~1 page (discriminator training, ESS clipping, fade-out fixed point)
- §3.6 BC adaptive cold-start imitation (Extension 1) — ~½ page (μ schedule as safety net, decoupled from DR3 reliability)
- §3.7 SC potential-based shaping (Extension 2) — ~½ page (hash map P(s), Ng 1999 invariance, teacher-exclusion rationale)
- §3.8 Combined update — ¼ page

**Crucial writing rule**: each subsection ends with a "principled justification" sentence stating what theorem/argument makes this not a heuristic.

---

## Anticipated Reviewer Objections (and where we address them)

| Objection | Response | Section |
|---|---|---|
| "Single seed" | Binomial CI; honest §Limitations | §Limitations |
| "DR3/BC/SC each = known idea" | Paper's contribution is the *diagnosis* and the *combination*, not invention of any single mechanism | §Intro ¶3 + §Method overview |
| "LUFFY 38% vs 49.5%—did you cripple baselines?" | 3-way reproduction transparency | §Appendix Reproducibility |
| "μ has 4 hyperparams" | Robustness mini-sweep table; framing as safety-net (not precise gap measure) | §Method §3.6 + §Appendix sensitivity |
| "Discriminator collapses?" | ESS clipping + ŵ histogram over time | §Method §3.5 + §Appendix |
| "WebShop ablation absent?" | Frame AF as standard ablation testbed; WS as cross-domain check | §Experiments framing |

---

## Files in this folder

- `abstract.tex` — abstract (180–220 words)
- `sections/01_introduction.tex` — 5-paragraph intro
- `sections/03_method.tex` — method opening (problem setup + overview)
- `data/raw_data.md` — main numbers, source attribution
- `tables/main_results.tex` — Table 1
- `tables/main_results_with_reward.tex` — extended SR + RM
- `tables/README.md` — preamble + usage
- `narrative.md` — this file
