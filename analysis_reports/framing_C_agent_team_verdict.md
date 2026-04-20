---
name: Framing C agent team verdict
description: Synthesis of 3-agent review on proposed NeurIPS Framing C for DUET/v24
type: analysis
---

# Framing C — Agent Team Verdict

**Date:** 2026-04-19
**Proposal under review:** "Support-gap + minimal two-operator Action Channel" framing, with SC downgraded to reward shaping.

## Verdict: **RED — Framing C does not pass review.**

Two independent reviewers (theory-researcher, exp-analyst) returned red-light. Partial trajectory data (case-analyst, 15/120 samples) further weakens the support-gap narrative. Three independent failure axes — the framing cannot be rescued by fixing just one.

---

## Axis 1: Math is wrong (theory-researcher)

- DR3's PPO ratio is `w_hat` (sample-level scalar from discriminator), **not** token-level `π_θ/π_teacher`. Memo §1 Regime-A derivation is about the wrong quantity. Conclusion directionally right, derivation wrong.
- Proposed Theorem 1's lower bound `c · min(p_θ, 1−p_θ)` **vanishes at p_θ = ε**, the exact regime the story depends on. Vacuous at cold-start.
- AWAC-style single-operator designs cover both regimes — "two operators required" is false. At best we can claim "two operators are minimal and independently analyzable."
- μ_valley = 0.05 serves **two distinct roles** (cold-start bootstrap vs teacher-support prior). Curriculum framing collapses them dishonestly.

## Axis 2: Empirical crossover doesn't exist (exp-analyst)

- v24 was run with `chord_use_token_weighting: false`. **BC is uniform SFT, not p(1−p)-weighted**. The mechanism Framing C implicitly relies on is off.
- `corr(μ, tgs) = +0.64`  vs `corr(disc_acc, tgs) = −0.49`: **μ schedule explains behavior better than p_θ specialization does.** The "automatic specialization by p_θ" claim has no empirical support in wandb.
- DR3 gradient magnitude dominates BC by 20-100× throughout training. No crossover.
- Producing the crossover figure requires 40-line logging patch + 3 re-runs (~15h GPU) — even then, not guaranteed.

## Axis 3: Trajectory data refutes "support gap" (case-analyst, partial)

Case-analyst crashed at 15/120 samples but the partial data is informative:

| Task | v12 p_click_option | v12 p_teacher_action | v12 p_buy_now |
|------|---:|---:|---:|
| webshop_218 | 0.19 | 0.00 | 0.625 |
| webshop_74 | 0.56 | 0.375 | 0.375 |
| webshop_49 | 0.44 | 0.06 | 0.50 |

**v12 has moderate support on `click_option`** (0.19–0.56), not near-zero. What it lacks is **specificity** — it clicks some option, just the wrong one. This is not a "support gap"; it is a **specificity gap**.

Implication: BC doesn't lift support (v12 already had it). BC teaches **which specific option** to click via token-level imitation. The theoretical story changes from "BC is needed to reach zero-support regions" to "BC is needed for teacher-specific option identity" — a weaker, less universal claim.

---

## Why this is fatal for Framing C

Framing C's appeal was **elegance through a single theoretical primitive** (support gap → minimal operator pair). Each axis attacks a different pillar:

- Math axis: the primitive is wrong
- Empirics axis: the mechanism isn't happening in v24 anyway
- Trajectory axis: the phenomenon isn't what the primitive describes

You cannot patch any single one without undermining the others. Example: fix the math → still no empirical crossover. Fix logging + re-run → still doesn't change that v12 had support. Fix trajectory narrative (specificity gap) → now the math doesn't map, because "specificity" is a different mathematical problem than "support."

---

## What the data actually supports (candidate new framings)

### Option F: "Two views, one channel" (weaker but honest)
Frame Action Channel as containing two operators serving **complementary purposes**:
- BC: teacher-specific token identity (the "which option" problem)
- DR3: trajectory-level credit assignment via density ratio (the "was this trajectory good" problem)

These are orthogonal purposes, not automatically-specialized by p_θ. μ_t decay is presented as "BC's role retires as policy imitation converges; DR3's role grows as discriminator converges."

- **Pros:** matches v24's actual behavior; defensible; requires no new experiments
- **Cons:** softer novelty story; reviewer will ask "why these two specifically?"

### Option G: "DR3 is the contribution; BC is inherited from CHORD, curriculum is ours"
- Contribution: DR3 density-ratio estimator + adaptive curriculum schedule that automatically trades off BC and DR3
- BC is explicitly CHORD's (cite and credit)
- SC stays in main paper
- Novelty concentrates on DR3 + curriculum coupling

- **Pros:** very defensible; no post-hoc theorems
- **Cons:** contribution surface looks smaller; "DR3 + adaptive μ" is a narrow claim

### Option H: "Method paper, not theory paper"
Drop the Theorem 1 attempt entirely. Present DUET as an **empirical recipe** with strong ablations:
- 24-variant ablation grid (already done)
- Per-environment scaling (1.5B/3B/7B)
- Component-wise attribution (SC, DR3, BC each)
- Behavioral case studies (option-specificity gap)

- **Pros:** zero theoretical risk; strong ablations are NeurIPS-compliant
- **Cons:** reviewers who want theory will score down; competes on benchmark numbers alone

---

## Recommendation

**Option F + Option H combined.** Don't overclaim theory. Present DUET as a dual-channel method where:
- Each channel is motivated by a clear empirical failure mode (documented in v1–v24 ablation + trajectory case studies)
- BC and DR3 within Action Channel are described as solving orthogonal sub-problems (specificity vs credit assignment)
- SC remains a core contribution (v4 ablation is load-bearing evidence)
- Theoretical claims limited to **what we can actually prove**: convergence of w_hat under discriminator consistency, bounded variance under dual-ESS clipping

This gives us:
- No wrong math
- No contradictions between claims and implementation
- Strongest empirical package in the submission (24 variants + scaling + case studies)
- Honest novelty: DR3 + SC + adaptive curriculum, each with its own failure-mode motivation

**Do not commit resources to Framing C.** Don't run the logging patch / 3 re-runs — they would validate a story that the math kills anyway.

## Action items

1. **Abandon Framing C.** Do not cite it; do not let it propagate into drafts.
2. **Choose replacement narrative** (F+H recommended; discuss with user).
3. **Still needed regardless of framing:**
   - v24 on ALFWorld 1.5B/3B — confirms cross-environment generalization
   - DR3 + BC no-SC ablation — quantifies SC's contribution (answers R2 reviewer question)
   - 3B CHORD ALFWorld rerun (already in progress)
4. **Do not need:** per-bin p_θ logging, AWAC baseline, Theorem 1 rewrite.

## Source files

- `/data/home/qisheng/EvolAnalsis/analysis_reports/framing_C_theory_review.md` — adversarial math audit
- `/data/home/qisheng/EvolAnalsis/analysis_reports/framing_C_empirical_viability.md` — wandb + logging audit
- `/data/home/qisheng/EvolAnalsis/tmp_scripts/rollout_v12_opt_shard*.jsonl` — partial trajectory evidence (15/120 samples)
- `/data/home/qisheng/EvolAnalsis/analysis_reports/duet_v24_theory_and_framing.md` — source memo (supersede)
