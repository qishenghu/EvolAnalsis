# DUET Paper Main Figure — Text-to-Image Prompts

This document contains prompt drafts for generating the main architecture figure for the DUET paper (NeurIPS 2026 submission).

Paper title: **DUET: Principled Experience Replay for LLM Agent Reinforcement Learning**

---

## Conceptual content the figure must convey

DUET is a **principled experience-replay framework** for GRPO-style RL of LLM agents.
Inserting teacher trajectories into the same rollout batch as student trajectories
(LUFFY-style mixing) induces two systematic biases. DUET **first corrects** these
biases, then **extracts** complementary teacher signal through two orthogonal
channels.

### Two input streams
1. On-policy rollouts from the current student policy (a group of `n` rollouts per task).
2. Cached teacher trajectories from a stronger LLM (e.g. Qwen2.5-72B), mixed at the rollout level (1 of `n` per task).

### Stage 1 — Bias correction (the "principled" part)

Two biases of naive teacher mixing, and DUET's fix for each:

- **Baseline contamination → Baseline Separation.** Successful teacher rollouts pull
  up the GRPO group mean, suppressing the advantages of successful *student* rollouts
  and discouraging exploration. DUET computes the GRPO baseline separately for
  teacher and on-policy samples.

- **Off-policy mismatch → Density-Ratio Correction (DR3).** Teacher trajectories are
  off-policy w.r.t. the student, but exact teacher–student importance correction is
  infeasible when teacher likelihoods are unavailable. A small discriminator
  estimates the trajectory-level density ratio `ŵ_τ = π_θ / π_β`, which fills in for
  the missing importance ratio in the policy-gradient update for teacher samples.

### Stage 2 — Signal extraction (two channels)

- **🟧 Action Channel — token-level Behavior Cloning (BC).** A cross-entropy term on
  teacher tokens, weighted by an adaptive coefficient `μ(t)` that fades as the
  discriminator separates student / teacher distributions.

- **🟩 State Channel — potential-based progress shaping (SC).** A hash-based expert
  progress map `P(s)` adds dense reward shaping on on-policy samples (`r' = r + β·P`).
  Teacher samples are excluded by design; the bonus is potential-based, so it
  preserves the optimal policy (Ng et al., 1999).

### Combined update
```
L  =  L_PG[baseline-sep, DR3, SC]   +   μ(t) · L_BC
```
Each mechanism auto-attenuates as the student approaches the teacher's capability.

---

## Prompt Version 1 — Concept diagram (NeurIPS aesthetic, RECOMMENDED)

> A clean, professional architecture diagram for a machine-learning paper, in flat
> vector style with a white background. Wide horizontal layout. Three main visual
> zones from left to right.
>
> **Left zone (Input streams)**: A "Student π_θ" box on top and a "Teacher π_β
> (frozen, cached)" box below, both as simple rounded rectangles with small
> neural-network icons. Arrows from each emit short trajectory lines (7 from the
> student, 1 from the teacher) merging into a "Mixed Batch (n=8: 7 on-policy + 1
> teacher)" stack. The single teacher trajectory is tinted lightly to make it
> visually distinct.
>
> **Center zone (DUET architecture)**: One large container labeled at the top
> "DUET — Principled Experience Replay", divided into two horizontal stages with
> a thin vertical separator:
>
> - **Stage 1 — Bias Correction** (left half, label tinted neutral gray): two
>   small stacked sub-blocks.
>   - Top sub-block: "Baseline Separation" — a tiny visual showing two histogram
>     curves (one teacher, one on-policy) being normalized independently into
>     advantage values. Subscript: "fixes group-baseline contamination".
>   - Bottom sub-block: "Density-Ratio Correction (DR3)" — a tiny discriminator
>     MLP icon outputting a scalar `ŵ_τ = π_θ / π_β`, with a small arrow into a
>     "PG ratio (teacher samples)" box. Subscript: "fixes off-policy mismatch".
>   These two outputs flow into the "Policy Gradient" block on the right edge of
>   the stage, indicating that they shape the corrected GRPO update.
>
> - **Stage 2 — Signal Extraction** (right half, label tinted neutral gray): two
>   stacked panels, each with a colored header bar.
>   - **🟧 Action Channel** (orange header #E07A5F): a token strip with teacher
>     tokens highlighted in orange and a small `μ(t)` adaptive-weight curve
>     beside it. Inside the panel: simply the label "Token-level BC, weighted
>     μ(t)". Subscript: "applied to teacher tokens; fades as discriminator
>     separates".
>   - **🟩 State Channel** (green header #81B29A): a small hash-table icon with
>     cells colored by progress value, label "Progress map P(s) → r' = r + β·P".
>     Subscript: "on-policy samples only; potential-based, policy-invariant".
>   These two channel outputs each emit one arrow into the right zone.
>
> **Right zone (Update)**: The corrected Policy Gradient (with DR3 + Baseline
> Separation + SC reward shaping baked in) and the BC loss (weighted by μ(t))
> converge into a single "GRPO Update" panel showing the combined loss
> `L = L_PG[corrected] + μ(t)·L_BC`. A circular feedback arrow loops back to the
> Student π_θ on the left, completing the training loop.
>
> Use a minimalist palette: orange (#E07A5F) for Action Channel, green (#81B29A)
> for State Channel, neutral grays (#6B7280, #2D3748) for the bias-correction
> stage and for text/arrows. Light gray (#F7FAFC) backgrounds for sub-blocks,
> white panel backgrounds with colored header bars. Clean LaTeX-style serif font
> for symbols. Arrows are subtle dark gray with arrowheads. NO photorealistic
> elements, NO 3D, NO clipart. The figure should look like a top-tier conference
> paper figure — clean, balanced, information-dense but readable.
> Title at top center: "DUET: Principled Experience Replay for LLM Agent
> Reinforcement Learning". Small caption-line under title:
> "Stage 1 corrects two biases of teacher mixing; Stage 2 extracts complementary
> teacher signal through two channels."

---

## Prompt Version 2 — Minimalist iconographic (alternative aesthetic)

> A minimalist architecture figure for a NeurIPS / ICLR paper. Horizontal flow
> diagram on a clean white background, using a restrained palette: deep navy
> (#1B3A57), warm coral (#E76F51), sage green (#84A98C), and neutral gray
> (#6B7280) for connecting lines.
>
> **Left**: Two stacked icons — a small humanoid "student" silhouette next to
> `π_θ` and a slightly larger silhouette next to `π_β (frozen)`. An arrow merges
> them into a small grid of 8 trajectory tiles, with one tile tinted coral to
> indicate the teacher rollout.
>
> **Center**: a 2 × 2 grid of small iconographic blocks under one umbrella
> heading "DUET". The two columns are labeled at the top with thin headers:
> "Correct" (left column, neutral gray) and "Extract" (right column, neutral
> gray).
>
> - Top-left cell — **Baseline Separation**: two small histogram curves being
>   normalized independently. One-word tag: "baseline".
> - Bottom-left cell — **DR3**: a tiny discriminator network with output `ŵ`.
>   One-word tag: "ratio".
> - Top-right cell — **🟧 Action Channel (BC)** (coral background tint): a token
>   strip with a small `μ(t)` decay curve beside it. One-word tag: "imitate".
> - Bottom-right cell — **🟩 State Channel (SC)** (sage green background tint):
>   a small heat-map of progress values. One-word tag: "shape".
>
> **Right**: A single "GRPO Update" oval that all four cells feed into, with
> `θ ← θ − η∇L` beside it.
>
> No drop shadows. No gradients other than the heat-map. Pure flat design. Clean
> sans-serif typography (Inter or IBM Plex Sans). Lowercase labels. Small
> key/legend at bottom showing the column headers ("Correct" = bias fixes,
> "Extract" = teacher-signal channels). The overall feel is like a Distill.pub
> diagram or an Anthropic technical figure: minimal, elegant, information-rich.
>
> Title: "DUET: Principled Experience Replay for LLM Agent Reinforcement Learning".

---

## Prompt Version 3 — Schematic (technical, for the Method section)

> Generate a technical schematic for a research paper, drawn as a labeled block
> diagram with clear data-flow arrows. White background; black-and-white-and-one-
> accent-color (a single saturated teal #14B8A6 highlights the adaptive feedback
> signals).
>
> Top-left: An LLM block labeled "Student π_θ" produces 8 rollouts (small
> horizontal lines bundled). Top-right: A separate LLM block labeled
> "Teacher π_β (offline cache)" provides a single trajectory shown as a database
> cylinder.
>
> Both sources merge into a "Mixed batch (n=8: 7 on-policy + 1 teacher)" block.
>
> From the mixed batch, FOUR DUET mechanisms branch out, visually grouped into
> two stages with rounded enclosing borders:
>
> ### Stage 1 — Bias Correction (top group)
>
> - **Baseline Separation**: a small block computing GRPO mean and std separately
>   over teacher and non-teacher subgroups, emitting unbiased advantages.
>   Annotation: "removes teacher contamination of group baseline".
>
> - **DR3 — Density-Ratio Correction**: a small MLP discriminator block consumes
>   `(state, action)` features and outputs `ŵ_τ`. This scalar replaces the missing
>   teacher importance ratio in the PG update. Equation shown:
>   `ratio_τ^teacher = ŵ_τ`. Annotation:
>   "applied to teacher trajectories only; auto-fades as ŵ → 0".
>
> Both Stage-1 outputs feed into the Policy-Gradient block.
>
> ### Stage 2 — Signal Extraction (bottom group)
>
> - **🟧 Action Channel (BC)**: a "Cross-entropy on teacher tokens" block,
>   weighted by `μ(t)`. The weight is computed from the discriminator's
>   accuracy signal, shown as a small dashed teal feedback arrow indicating
>   adaptation. (Do NOT spell out the BC formula in detail — a single
>   `L_BC` label is enough.)
>
> - **🟩 State Channel (SC)**: a "Progress hash-map P(s)" block annotates each
>   on-policy state with a scalar progress value. Reward is shaped as
>   `r' = r + β · ΔP(s_t → s_{t+1})`. Annotation:
>   "on-policy samples only (teacher excluded); potential-based, policy-
>   invariant".
>
> The Action-Channel BC loss and the (corrected) Policy-Gradient loss combine
> in a "GRPO Combined Loss" block: `L = L_PG[corrected] + μ(t)·L_BC`. The loss
> flows into "Optimizer" → `θ ← θ − η∇L` → loops back to update Student π_θ.
>
> Use thin black arrows for primary data flow, dashed teal arrows for adaptive
> / feedback signals (μ(t) adaptation, ŵ correction, progress-map lookup).
> Labels in clear sans-serif (Helvetica or Arial). The figure should look like
> a top-tier ICML paper Figure 2 — pedagogical, dense but readable, with no
> decorative elements.
>
> Title: "DUET: Principled Experience Replay for LLM Agent Reinforcement Learning".

---

## Suggested usage

1. **First pass with Version 1** (concept diagram) — gives a publication-ready
   overview suitable for the abstract figure on page 1.
2. **Iterate with Version 2** if Version 1 feels too busy — the 2×2 grid is
   the most compact way to show all four mechanisms; good for slides / talks.
3. **Use Version 3** in the Method section if you want a more technical,
   data-flow-oriented look.

## Notes for human polishing

After image generation:
- Replace any garbled equations with proper LaTeX rendering (most text-to-image
  models butcher math).
- Verify variable names are consistent with the paper notation
  (`π_θ`, `π_β`, `ŵ_τ`, `P(s)`, `μ(t)`, `β`).
- Crop to 16:9 or 4:3 depending on whether you want full-page-width or
  column-width.
- Make sure the figure visually communicates the **two-stage structure**
  (correct → extract). The four DUET mechanisms should not appear as a flat
  list of siblings; the Stage-1 corrections should look like preprocessing
  applied to the GRPO update, while the Stage-2 channels should look like
  parallel signal contributions.

---

## One-line summary (for the figure caption)

> **Figure 1**: Overview of DUET. Naive teacher mixing in GRPO induces two
> systematic biases. DUET first **corrects** these via (i) baseline separation
> — independent GRPO normalization for teacher and on-policy samples, and
> (ii) DR3 — a discriminator-based density-ratio correction filling in for the
> missing teacher importance ratio. DUET then **extracts** complementary
> teacher signal through two channels: an **Action Channel** that performs
> token-level behavior cloning with an adaptive weight `μ(t)`, and a **State
> Channel** that supplies potential-based reward shaping via an expert progress
> map `P(s)`. All signals are integrated into a single GRPO update, and each
> mechanism auto-attenuates as the student approaches the teacher's capability.
