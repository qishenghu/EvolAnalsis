# Figure 1 — DUET architecture diagram (prompt brief, v2)

**Output target**: `NeurIPS_2026_Latex/figures/fig1_method.pdf` (or `.png`)
**Caption (already in `sections/03_method.tex` §3.3)**: emphasizes that DUET's
four mechanisms fall into **two categories** (estimator corrections + signal
channels), applied **simultaneously** every training step. The figure must
not imply temporal pipelining (no "Stage 1 → Stage 2" arrows).
**Aspect ratio**: 16:7 to 16:6 (wide horizontal).

---

## Hard rules for this figure

These are non-negotiable; everything else is a stylistic choice.

1. **No "Stage 1 / Stage 2" labels, no left-to-right pipeline arrows
   between the two categories.** The two categories operate in parallel.
   Layout must read as "two side-by-side groups", not "first this, then
   that".
2. **No inset attenuation curves / training-dynamics plots.** Those live in
   Figure 3 and would clutter Figure 1.
3. **No tiny visual hints inside each mechanism box** (no token strips, no
   histograms, no progress heatmaps, no MLP icons). Just clean geometric
   cards with labels. T2I models render these badly and they fight the
   minimalist aesthetic anyway.
4. **No 3D, no drop shadows, no gradients (other than possibly very subtle
   card elevation), no clipart, no skeuomorphism.**
5. **Math symbols (π_θ, π_β, ŵ, μ(t), Φ(s), P(τ), ℒ) must be readable.** Plan
   to re-render text in vector tooling after generation if T2I garbles it.

6. **CRITICAL: SC formula is trajectory-level, not step-wise.** The SC card
   must show `r(τ) ← r(τ) + β P(τ)` (or "progress bonus", with `P(τ)` defined
   as the trajectory-mean of the teacher progress potential). Do **not** show
   `r' = r + β(Φ(s') − Φ(s))` — that is the strict Ng99 step-wise form, which
   is *not* what DUET implements. Reviewers look at the figure before the
   text; if the formula in the figure contradicts the text in §3.7, that's a
   review-killer.

---

## What the figure must convey

A reader should pick up four facts within ~5 seconds:

1. DUET takes two inputs: student rollouts and a fixed teacher cache.
2. DUET applies four mechanisms organized into two categories.
3. The categories are *Bias Correction* (BS, DR3) and *Signal Channels*
   (BC for actions, SC for states).
4. All four feed a single combined objective that updates the student.

---

## Recommended composition (Layout A — "Two-column card panel")

```
                       Student π_θ      Teacher π_β  (cache, frozen)
                            \              /
                             ↘            ↙
                       Mixed group  (n on-policy + 1 teacher)
                                  │
            ┌─────────────────────┴──────────────────────┐
            │                  D U E T                   │
            │                                            │
            │   ── Bias Correction ──   ── Signal ──     │
            │                                            │
            │   ┌──────────┐ ┌──────┐  ┌────┐ ┌────┐     │
            │   │ Baseline │ │ DR3  │  │ BC │ │ SC │     │
            │   │ Separat. │ │  ŵ   │  │μ(t)│ │P(τ)│     │
            │   └──────────┘ └──────┘  └────┘ └────┘     │
            │       Bias 1       Bias 2  action   state  │
            │                                            │
            └────────────────────┬───────────────────────┘
                                 │
                                 ↓
                      DUET update  ℒ = ℒ_PG + μ(t) ℒ_BC
                                 │
                                 ↺  (back to Student π_θ)
```

Two **horizontally adjacent** label bars sit above the four cards: the left
two cards (BS, DR3) are under the bar "Bias Correction" (cool slate gray
header); the right two cards (BC, SC) are under the bar "Signal Channels"
(warm coral header). The bars are visually equal in weight — no implication
that one comes before the other.

---

## Prompt v1 — Editorial, two-column (recommended)

```
A NeurIPS-quality architecture diagram for a machine-learning paper.
Wide horizontal aspect ratio 16:7. Pure white background. Editorial,
minimalist style — like a Distill.pub diagram or a magazine technical
illustration. Generous whitespace. Clean Helvetica or Inter sans-serif
for labels; LaTeX-style serif for math symbols.

Composition top to bottom in three vertical zones:

ZONE 1 — INPUTS (top, ~15% of height).
Two small rounded rectangle cards side by side, labeled "Student π_θ"
(left) and "Teacher π_β   cache, frozen" (right). Two thin arrows from
both cards converge downward into a single horizontal bar labeled
"Mixed group   n on-policy rollouts + 1 teacher rollout". The two arrows
should be subtly distinguished — student arrow is solid charcoal, teacher
arrow is the same width but a softer gray, signaling "off-policy / cached".

ZONE 2 — DUET BODY (middle, ~60% of height; the visual focus).
One large rounded rectangle container labeled "DUET" in light, restrained
typography at the top center. Inside the container, two equal-width
horizontally adjacent regions, separated only by a hair-thin vertical
divider line:

  LEFT REGION — header label "Bias Correction" in slate gray (#374151)
  small caps. Below the header, two cards side by side:
    • Card 1: "Baseline Separation". One short subtitle line: "removes
      Bias 1: contamination". No icons inside.
    • Card 2: "DR3" with a small subscript "ŵ = π_θ / π_β". Subtitle:
      "removes Bias 2: off-policy mismatch".
  Both cards have a soft slate-gray top border (3px), white interior, and
  very subtle card elevation (a single 1-pixel light gray bottom edge).

  RIGHT REGION — header label "Signal Channels" in warm coral (#E07A5F)
  small caps. Below the header, two cards side by side:
    • Card 3: "BC   action channel". One subtitle: "adaptive token
      imitation, weight μ(t)". Coral top border (3px).
    • Card 4: "SC   state channel". Subtitle: "progress bonus
      r(τ) ← r(τ) + β P(τ)" where P(τ) is defined in tiny text as
      "trajectory-mean teacher progress". Sage green top border (3px,
      color #84A98C). The two channel cards may use different border
      colors (coral for action, sage for state) to subtly reinforce
      the action / state axis. **Do NOT use the step-wise potential
      form `r' = r + β(Φ(s') − Φ(s))` — that is the strict Ng99
      reference, not what DUET implements.**

Single thin convergence arrow from the bottom-center of the DUET container
flows down into Zone 3. No internal arrows between the four cards —
they operate in parallel.

ZONE 3 — UPDATE (bottom, ~25% of height).
A single softly-rounded pill labeled "DUET update" with a small math line
beneath: "ℒ = ℒ_PG[BS, DR3, SC]  +  μ(t) ℒ_BC". A curving feedback arrow
loops from the right edge of the pill back up to the Student π_θ card,
implying iterative training.

Style anchors: Swiss design, magazine-quality typography, restrained
2-color accent palette (slate gray + coral with sage green as a tertiary
accent on the state channel only), pure white background, no shadows,
no 3D effects, no clipart, no decorative flourishes. The figure should
feel like it was drawn in Adobe Illustrator by a designer who reads
Distill.pub, not generated by an image model. Title at top center, in
slightly smaller weight than the body labels: "DUET: Principled Teacher
Experience Replay".
```

---

## Prompt v2 — Concentric / hub-and-spokes (alternative, more iconic)

Use this if v1 looks busy. v2 emphasizes that all four mechanisms operate
on the same update simultaneously, which directly answers the "no temporal
ordering" requirement.

```
A NeurIPS architecture figure, minimalist editorial style, white background,
aspect ratio 16:8. The composition is a single centered hub-and-spokes
diagram with very generous whitespace.

CENTER: a single rounded square labeled "DUET update" with a tiny math line
"ℒ = ℒ_PG + μ(t) ℒ_BC" inside. Use restrained typography.

INPUTS: above the center hub, two small cards "Student π_θ" and "Teacher
π_β (cache)" feed inward via two thin arrows — student arrow charcoal solid,
teacher arrow soft gray. They merge into a small "mixed group" badge that
points down into the center hub.

FOUR SPOKES — four equally-weighted rounded cards arranged at the cardinal
NW, NE, SW, SE positions around the center hub, each connected to the hub
by a thin gray arc. The four cards are grouped visually by category:

  TOP ROW (NW, NE) — "Bias Correction" group, slate gray (#374151) accent
  borders:
    NW: "Baseline Separation   removes Bias 1"
    NE: "DR3   ŵ = π_θ / π_β   removes Bias 2"

  BOTTOM ROW (SW, SE) — "Signal Channels" group, warm accents:
    SW: "BC   action channel   weight μ(t)"   (coral #E07A5F border)
    SE: "SC   state channel    progress bonus β P(τ)"  (sage #84A98C border)

Two thin curved labels run above the top row and below the bottom row in
small-caps gray text: "Bias Correction" (above NW–NE) and "Signal Channels"
(below SW–SE). These labels are decorative grouping cues, not arrows.

A subtle gray feedback arrow loops from the right side of the center hub
back up to the Student π_θ input card, implying iterative training.

Style: Swiss editorial design, Distill.pub aesthetic, very restrained
palette, generous whitespace, no shadows, no gradients, no 3D. Title at
top center: "DUET: Principled Teacher Experience Replay".
```

---

## Aesthetic anchors that nanobanana / similar T2I models honor well

(Use one or two of these phrases per prompt to nudge the style.)

- "editorial illustration"
- "Distill.pub diagram aesthetic"
- "Swiss design, magazine technical illustration"
- "vector schematic, generous whitespace, restrained palette"
- "Apple keynote technical figure"
- "drawn in Adobe Illustrator by a senior designer"
- "single composed scene, not collage"

---

## After generation — text labels are usually wrong

T2I models reliably get the **layout** right and reliably get the
**text** wrong (especially math symbols). Recommended workflow:

1. Generate at the highest resolution available (≥ 4K).
2. Pick the version with the cleanest layout, best card geometry, and
   correct color grouping.
3. **Re-typeset all text in Inkscape / Figma** — load the raster as a
   background layer, place real text on top with a clean sans-serif
   (Inter, IBM Plex Sans) and a math-aware font for symbols (STIX or
   Latin Modern Math).
4. Export as PDF.
5. Save as `NeurIPS_2026_Latex/figures/fig1_method.pdf` — the placeholder
   `\label{fig:method}` and caption are already in §3.3 and resolve
   automatically.

If the generated layout is bad, regenerate. If only the text is bad, fix
the text — never re-generate just to chase typography.

---

## What changed from the previous prompt file

- Removed all "Stage 1 / Stage 2" language; replaced with categorical
  "Bias Correction" / "Signal Channels".
- Removed Layout v3 (it included inset attenuation curves which duplicate
  Figure 3 and clutter the figure).
- Removed all "tiny visual hints inside each mechanism" (histograms,
  token strips, MLP icons, heatmaps) — the new aesthetic is clean cards
  with labels only.
- Tightened color palette (slate + coral, sage as tertiary accent only).
- Added explicit aesthetic anchors and a workflow section for fixing
  T2I text rendering after generation.
