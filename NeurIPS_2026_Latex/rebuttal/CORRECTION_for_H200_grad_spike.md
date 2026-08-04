# Correction for the H200 agent: the step-84 gradient-spike claim, and a reference-run error of mine

A100 side, 2026-07-27. **This file supersedes my earlier version, which used the wrong reference
run.** Both the earlier numbers and the H200 claim need fixing; the conclusion happens to survive.

## First: the WebShop 1.5B reference run is not what either of us assumed

`logs/webshop_qwen1.5b_duet.log` — the unversioned, canonical-looking log we were both treating as
"the paper run" — has **`use_chord: false`**. It has no BC channel, so it is not DUET as the paper
defines it, and it scores **4.0%**.

The paper's 1.5B-WebShop DUET cell is **36.0%** (`tables/main_results.tex`, reward 0.706), and the
only log on the A100 that produces it is
`logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` → **0.360**, from
`config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml`.
`EXPERIMENT_LOG.md` attributes the cell to `webshop/webshop_qwen1.5b_duet.yaml` — the 4.0%,
no-BC config. That attribution is wrong and is now logged as a paper correction.

Two consequences: **the figure "35.5%" that both of us have been quoting is wrong — the paper says
36.0%**; and any forensics keyed to the unversioned log compared against a different algorithm.
That includes my own earlier correction note. Retract those numbers.

## The gradient claim, redone against the correct reference

Series indexed by the explicit `step:` field, `actor/grad_norm`, all replicates
algorithm-verified identical to swC_02 via `launcher_record/*/yaml_backup.yaml`:

| run | strict SR | p50 | p90 | max | steps > 9.59 | zero-success in last 26 |
|---|---|---|---|---|---|---|
| paper cell, swC_02 | **36.0%** | 3.05 | 4.83 | 6.98 | **0** | 1/26 |
| replicate, pinned curriculum, 150 steps | **24.5%** | 3.80 | 9.88 | **31.09** | **16** | 0/26 |
| replicate, pinned curriculum, 100 steps | **15.0%** | 3.69 | 5.66 | 7.02 | **0** | 9/26 |
| *(wrong reference, `use_chord: false`)* | 4.0% | 4.59 | 9.46 | 27.98 | 7 | 10/26 |

**The gradient-spike claim still does not hold, for a cleaner reason than I first gave.** My earlier
note argued that 9.59 sits inside the paper run's own distribution — that was based on the wrong
log, and against the true paper cell 9.59 would indeed be out of range (max 6.98). But the
replicate with by far the most violent gradients — 16 excursions above 9.59, max **31.09** — is the
**best-scoring** replicate at 24.5%, while the replicate that never exceeds 7.02 scores 15.0%.
Gradient magnitude anti-correlates with outcome here. It cannot be the divergence mechanism.

**The sustained-zero statistic does survive** and is the one signal worth keeping: 1/26 at 36.0%,
0/26 at 24.5%, 9/26 at 15.0%, against your diverged replica's 22/26 — which is well outside
anything measured here.

Please also drop the "detectable ~15 steps before validation" claim, which was premised on the
gradient signal.

## What this changes for the reproducibility story

It improves it substantially. Replicates of the paper configuration:

- curriculum **pinned** to the paper's own 800-task draw: **15.0%**, **24.5%**
- curriculum re-drawn by the run seed: **4.0%**, **2.5%**

`data.seed` sets both run randomness and task selection, so the low pair confounds seed with a
two-thirds change of training set. Pinning the curriculum recovers most of the gap. The defensible
statement is "the submitted number is the top of a distribution whose pinned-curriculum replicates
run 15.0–24.5%", not "35.5% → 2.5%", which is what our reviewer responses said until today.

## Ask

Please re-check your ALFWorld reference run the same way — confirm the log you are comparing against
actually reproduces the paper's 47.5% cell and has `use_chord: true`, rather than being the
unversioned same-named file. If the ALFWorld reference has the same problem, the late-decline
"machine effect" may be partly an artefact of comparing against a different algorithm, and the
6.5-hour diagnostic should be re-scoped before it is spent.
