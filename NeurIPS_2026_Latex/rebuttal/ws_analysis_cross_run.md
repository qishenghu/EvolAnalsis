# WebShop: does the option-clicking story explain the whole run population?

**Question.** Three 1.5B WebShop DUET runs with identical code and identical config (seed only)
land at val@100 strict success of 35.5% / 3.5% / 2.5%. A mechanism was proposed from saved
rollouts: strict WebShop success requires clicking every requested option button before
`click[buy now]`; the winning run learned to do this inside the 100-step budget, the others did
not. This note asks whether that mechanism generalises across the ~70-run population on disk.

**Answer: yes, and more strongly than the three-run comparison suggested.** Across 71 1.5B
WebShop runs, option-clicking predicts strict success at r = +0.91 while predicting mean reward
at only r = +0.49; buy-completion rate does the exact opposite (r = +0.85 with mean reward,
r = +0.11 with strict success). At the episode level the gate is essentially hard: of 6,928
greedy validation episodes that bought an item without clicking a single option on a task that
requires at least one, **zero** scored 1.0. Item/search quality is *not* the discriminator —
the three seeds buy the known-correct product at 77% / 66% / 70%, statistically similar, but
satisfy the option requirement at 72% / 14% / 6%.

Everything below is measured from files on disk; nothing was re-run and no GPU was touched.

---

## 0. Setup, and what was verified

Run identification (the established facts name runs by seed, the on-disk dirs by sweep tag):

| label in the brief | on-disk run | val@50 strict | val@100 strict | val@100 mean reward |
|---|---|---|---|---|
| paper / seed2026 | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` | 1.0% | 35.50% | 0.706 |
| seed2025 | `webshop_qwen1.5b_duet_a100_seed2025` | 1.5% | 3.50% | 0.521 |
| seed2027 | `webshop_qwen1.5b_duet_a100_seed2027` | 1.0% | 2.50% | 0.543 |

These match the brief exactly (1.0/1.5/1.0 at step 50; 35.5/3.5/2.5 and 0.706/0.521/0.543 at
step 100), so the run mapping is confirmed. Mean on-policy training reward at steps 81–100 also
reproduces exactly (paper 0.567, seed2027 0.477).

**Population.** 82 WebShop runs have `experiments/webshop/<run>/validation_log/100.jsonl`
(200 episodes each, greedy). 71 of them are 1.5B; the rest are 3B/7B. 81 also have on-policy
training dumps at `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_{1..100}.jsonl`
(64 rows/step, teacher rows flagged by `diag.is_teacher`).

**Validation set is shared.** All 82 runs evaluate the *same* 200 tasks. 79 of them present the
tasks in the same order; the two `a100_seed*` runs use the same task set in a **different order**
(this matters: an earlier index-paired comparison of those two against the paper run is invalid;
everything below is keyed on the instruction text instead).

**Metric definition.** For each episode I recover the action sequence, take the prefix before the
first `click[buy now]`, and count *distinct* option clicks in it — a `click[x]` where `x` is not
`buy now` / `back to search` / `< prev` / `next >` / `description` / `features` / `reviews` and
is not an ASIN (`^[bB][0-9a-zA-Z]{9}$`). Teacher rows are excluded from all training-rollout
numbers.

> **Definitional caveat, stated up front.** With this definition I get paper = 0.95 mean options
> and 28.0% of episodes with ≥2 at training steps 81–100, where the brief reports 2.31 and 70.7%.
> The *ordering and the ratio* reproduce (paper ≫ seed2027 on both), and the reward numbers match
> to 3 decimals, so I am confident it is the same data with a looser counting rule in the earlier
> pass — counting *all* `click[...]` actions before buy, including the item click itself, gives
> paper 2.38 / seed2027 1.46, close to the reported means. I could not reproduce the reported
> ≥2 fractions under any variant I tried. **All numbers in this note use my stricter definition
> consistently across all runs**, so cross-run comparisons are internally valid, but the absolute
> option counts here are *not* interchangeable with the ones in the brief.

---

## 1. Cross-run correlation: option-clicking predicts strict success, not mean reward

71 1.5B runs, all measured at step 100. `v_opt` = mean distinct options before buy per greedy
validation episode; `v_ge2` = fraction of episodes with ≥2; `v_bought` = fraction that reach
`click[buy now]` at all.

| predictor (validation, step 100) | vs **strict success** | vs **mean reward** |
|---|---|---|
| mean options before buy (`v_opt`) | **r = +0.914**, ρ = +0.895 | r = +0.486, ρ = +0.470 |
| frac. episodes ≥2 options (`v_ge2`) | **r = +0.964**, ρ = +0.839 | r = +0.444, ρ = +0.380 |
| frac. episodes ≥1 option | r = +0.745, ρ = +0.901 | r = +0.475, ρ = +0.496 |
| buy-completion rate (`v_bought`) | r = +0.110, ρ = +0.043 | **r = +0.854**, ρ = +0.627 |
| mean actions per episode | r = +0.413, ρ = +0.757 | r = +0.371, ρ = +0.234 |

This is a clean **double dissociation**, and it is exactly the shape the local-optimum story
predicts: *finishing the episode* (search → click item → buy) buys you partial credit and drives
mean reward; *clicking the options* is what converts partial credit into a 1.0.

Robustness:

- **Non-degenerate subset** (drop 6 runs with mean reward < 0.30 or buy rate < 50%, N = 65):
  `v_opt` vs strict **r = +0.913**, ρ = +0.924; `v_bought` vs strict **r = −0.152**. The
  dissociation gets *cleaner*, not weaker.
- **Drop the top-5 runs** (N = 66, max strict = 22%): `v_opt` vs strict r = +0.866,
  `v_ge2` r = +0.901. Not an outlier artifact.
- **Partial correlation** r(`v_opt`, strict | mean reward) = **+0.886**. Option-clicking is not a
  proxy for "this run is just better overall".
- Pooling all 82 runs (adding 3B/7B) barely moves anything: `v_opt` vs strict r = +0.897,
  vs mean reward r = +0.571.

### 1b. The non-circular version

The correlations above are measured on the same episodes as the outcome, so they are partly a
decomposition rather than a prediction. The genuinely predictive version uses **training**
rollouts — different tasks, different (stochastic) sampling, teacher rows excluded — to predict
*validation* strict success at step 100:

| predictor (on-policy training rollouts) | → val@100 strict | → val@100 mean reward |
|---|---|---|
| mean options, steps 81–100 | **r = +0.833**, ρ = +0.855 | r = +0.395, ρ = +0.399 |
| frac ≥2 options, steps 81–100 | **r = +0.846**, ρ = +0.800 | r = +0.357, ρ = +0.328 |
| mean options, steps 61–80 | r = +0.733, ρ = +0.753 | r = +0.393, ρ = +0.390 |
| mean options, steps 41–60 | r = +0.640, ρ = +0.540 | r = +0.299, ρ = +0.223 |
| mean options, **validation at step 50** | r = +0.717, ρ = +0.367 | r = +0.316, ρ = +0.253 |

The dissociation survives (≈2× stronger against strict than against mean reward at every window),
and the signal is already visible at steps 41–60 — half-way through the budget you can predict
final strict success from how often the policy touches option buttons.

---

## 2. Where the threshold is

### Episode level: the threshold is exactly 2 options

All 82 runs pooled, 13,538 greedy validation episodes that bought something:

| distinct options before buy | n | strict success | mean reward |
|---|---|---|---|
| 0 | 7,462 | 2.1% | 0.537 |
| 1 | 3,432 | 8.7% | 0.657 |
| **2** | 2,067 | **41.2%** | 0.765 |
| 3 | 484 | 38.4% | 0.786 |
| 4 | 63 | 14.3% | 0.604 |
| ≥5 | 30 | 0.0% | 0.447 |

A 4.7× jump from 1 → 2 options, then a plateau, then decay (≥4 options means the policy is
flailing). Mean reward, by contrast, rises smoothly by ~0.11 per option — partial credit is
graded, strict success is a step function.

Controlling for both task and item choice: restrict to (task, purchased-ASIN) cells that appear
with at least two different option counts across the 1.5B runs (290 cells), and center scores
within cell:

| options | n episodes | score deviation from cell mean | strict success |
|---|---|---|---|
| 0 | 6,021 | **−0.055** | **0.0%** |
| 1 | 3,026 | +0.047 | 7.8% |
| 2 | 1,160 | **+0.136** | **35.8%** |
| 3 | 214 | +0.141 | 31.3% |
| 4 | 62 | +0.057 | 6.5% |

Same task, same product, only the option clicks differ — and strict success goes 0% → 36%.

### Why 2? Because that is what the tasks require

Taking each task's required option count as the minimum options any run needed to score 1.0 on it
(131 of 200 tasks have a known 1.0 solution somewhere in the population):

| required options | # tasks |
|---|---|
| 0 | 4 |
| 1 | 13 |
| **2** | **85** |
| 3 | 29 |

The modal WebShop validation task needs exactly 2 option clicks. That is the whole explanation of
the episode-level threshold.

**The gate is effectively hard.** Split episodes by whether the task can be solved with zero
options:

| options clicked | task solvable with 0 options? | n | strict success |
|---|---|---|---|
| 0 | no (196 tasks) | **6,928** | **0.0%** |
| 0 | yes (4 tasks) | 252 | 54.4% |
| 1 | no | 3,051 | 7.7% |
| 2 | no | 1,164 | 35.7% |
| 3 | no | 277 | 25.6% |

Zero strict successes out of 6,928 buy-without-clicking episodes on tasks that need an option.
The 2.1% "strict at 0 options" in the pooled table is entirely those 4 option-free tasks.

Per-run confirmation across all three seeds: fraction of episodes that met the task's required
option count and still scored 1.0 = 75.5% (paper) / 38.9% (seed2025) / 62.5% (seed2027); fraction
that did **not** meet it and scored 1.0 = **0.0% / 0.0% / 0.0%**.

### Run level: the jump is around ≥2-options ≈ 40% of episodes

Exhaustive changepoint scan over the 71 1.5B runs (minimising within-group SS of strict success):

- best split on `v_ge2` at **0.385**: below (n = 62) mean strict **3.48%**; at/above (n = 9) mean
  strict **26.83%**.
- best split on `v_opt` at **1.355**: below (n = 63) mean strict **3.77%**; at/above (n = 8) mean
  strict **27.44%**.

Binned view (1.5B runs):

| `v_opt` bin | #runs | mean strict | median strict | mean reward |
|---|---|---|---|---|
| [0.00, 0.10) | 20 | 0.85% | 1.0% | 0.389 |
| [0.10, 0.30) | 7 | 2.57% | 3.0% | 0.520 |
| [0.30, 0.60) | 16 | 2.94% | 2.5% | 0.462 |
| [0.60, 0.90) | 11 | 4.91% | 4.5% | 0.532 |
| [0.90, 1.30) | 8 | 11.00% | 11.0% | 0.560 |
| [1.30, ∞) | 9 | **25.89%** | 26.5% | 0.625 |

| `v_ge2` bin | #runs | mean strict | median strict | mean reward |
|---|---|---|---|---|
| [0.00, 0.02) | 26 | 1.33% | 1.0% | 0.427 |
| [0.02, 0.05) | 11 | 3.36% | 3.5% | 0.472 |
| [0.05, 0.12) | 12 | 3.46% | 3.5% | 0.513 |
| [0.12, 0.25) | 7 | 6.21% | 5.5% | 0.459 |
| [0.25, 0.45) | 10 | 13.40% | 13.5% | 0.559 |
| [0.45, 1.00] | 5 | **33.30%** | 35.5% | 0.701 |

Note how mean reward moves over a range of 0.39 → 0.70 (a factor of 1.8) while strict success
moves 0.85% → 33% (a factor of 39) across the same bins.

Only **9 of 71** 1.5B runs cross the ≥2-options-40%-of-the-time line. Five of those nine are the
`swC_*` / `ws_swC_v_*` family, i.e. the paper run and its close variants; the others are
`ws_swC_v_pk04_v00`, and near-misses `duet_minus_bc` (41.5%) and `duet_v37` (38.5%). The paper run
is not unique — it is a member of a small, identifiable cluster.

The one substantive exception is `webshop_qwen1.5b_duet_v31`: `v_ge2` = 26.0% but strict = 1.5%.
Its buy-completion rate is 50.5%, the lowest of the 65 non-degenerate runs — it clicks options
and then fails to finish the episode. It is the mirror image of seed2027 and it fits the
two-gates model in §4 rather than contradicting it.

---

## 3. Validation (greedy, temperature 0) for the three seeds

Same 200 tasks, keyed by instruction text.

| run | strict | mean R | buy% | mean options | ≥1 opt | ≥2 opt | mean actions |
|---|---|---|---|---|---|---|---|
| paper (swC_02) | 35.50% | 0.706 | 97.0% | **1.80** | 92.0% | **71.0%** | 5.30 |
| seed2025 | 3.50% | 0.521 | 88.5% | 0.84 | 76.5% | 6.5% | 4.68 |
| seed2027 | 2.50% | 0.543 | 93.5% | 0.32 | 30.0% | 1.0% | 4.33 |

The greedy read separates the runs far more sharply than the stochastic training rollouts do.
This is the single most important refinement to the story: **in training rollouts seed2025 clicks
options about as often as the paper run** (mean 0.98 vs 0.95 at steps 81–100, ≥2 in 19.5% vs
28.0%), yet under greedy decoding it almost never does (≥2 in 6.5% vs 71.0%). Sampling noise
produces option clicks in all three runs; only the paper run's *mode* clicks options. The
training-rollout view understates the gap; validation is the clean read.

The action-length signature makes it concrete:

| run | episodes that are exactly search → click item → buy (3 actions) | ≥5 actions | mean actions |
|---|---|---|---|
| paper | **7.5%** | 73.5% | 5.30 |
| seed2025 | 13.5% | 29.0% | 4.68 |
| seed2027 | **67.5%** | 10.0% | 4.33 |

Two-thirds of seed2027's greedy validation episodes are the minimal three-action shortcut. That is
the local optimum, visible directly in the policy.

Spot-checked example (`seed2027`, score 0.0): actions are
`search[your query]` → `click[B07CKTXS1J]` → `click[buy now]` — the search string is literally the
placeholder from the prompt template, the item that comes back is a shampoo, the task asked for a
512 GB USB flash drive, and the policy buys it immediately.

---

## 4. Is it really options, or is it item selection (search quality)?

This is the alternative I was asked to falsify. Method: an ASIN is "correct" for a task if some
run in the population purchased it and scored exactly 1.0. To avoid circularity the goal set for
the three seeds is built from the **79 other runs only**; for the population correlations it is
built leave-one-run-out. 131 of 200 tasks have a known correct ASIN.

| run | bought the correct item | met the required option count | both | strict | strict given **both** |
|---|---|---|---|---|---|
| paper (swC_02) | 77.1% | **71.8%** | 57.3% | 54.2% | 94.7% |
| seed2025 | 65.9% | **14.3%** | 11.1% | 5.6% | 50.0% |
| seed2027 | 70.2% | **6.1%** | 3.8% | 3.8% | 100.0% |

(percentages over the 131 known-goal tasks; the paper run's strict rate is higher here than the
all-200 figure because the 69 tasks with no known solution are excluded from all three runs.)

**Item selection is not the discriminator.** The three runs find the correct product within
11 points of each other (77 / 66 / 70). They differ by **12×** on the option gate (72% vs 6%).
And conditional on passing *both* gates, all three convert at 50–100%.

Population-level, over the 71 1.5B runs (leave-one-out goal set):

| predictor | vs strict success | vs mean reward |
|---|---|---|
| fraction of tasks where the **correct item** was bought | **r = +0.067** | r = +0.654 |
| fraction of tasks where the **required options** were clicked | **r = +0.970** | r = +0.481 |
| both | r = +0.963 | r = +0.515 |

Partial correlations: r(option-gate, strict | item-gate) = **+0.914**;
r(item-gate, strict | option-gate) = **+0.134**. On the non-degenerate subset the item-gate
correlation with strict success is actually slightly *negative* (r = −0.061).

### Score-loss decomposition (131 known-goal tasks, loss = 131 − Σ score)

| run | total loss | never bought | wrong item | **right item, wrong options** |
|---|---|---|---|---|
| paper | 20.24 | 2.20 (10.9%) | 13.73 (67.8%) | 4.31 (21.3%) |
| seed2025 | 46.98 | 10.90 (23.2%) | 20.14 (42.9%) | 15.94 (33.9%) |
| seed2027 | 50.95 | 9.65 (18.9%) | 17.86 (35.1%) | **23.44 (46.0%)** |

Reading the *gap* rather than the level, paper → seed2027 loses 30.71 points of score:
**19.13 (62%) is "right item, wrong options"**, 7.45 (24%) is failing to buy at all, and only
4.13 (13%) is picking a worse item. For paper → seed2025 the split is 11.63 (44%) options,
8.70 (33%) no-buy, 6.41 (24%) item.

### Direct paired comparison, restricted to episodes where **both runs bought the same known-correct item**

This removes item choice entirely — same task, same product, greedy decoding:

| comparison | n | paper score / strict / options | other score / strict / options |
|---|---|---|---|
| paper vs seed2027 | 89 | 0.957 / **69.7%** / 1.82 | 0.745 / **5.6%** / 0.31 |
| paper vs seed2025 | 75 | 0.957 / **72.0%** / 1.81 | 0.813 / **9.3%** / 0.97 |

Identical task, identical purchased product, and strict success goes 70% → 6%. The residual score
(0.745, 0.813) is precisely the partial credit the brief describes: the item is right, the
attributes are unconfigured.

---

## 5. Verdict and what it implies

The option-clicking mechanism explains the population, not just the three seeds. Specifically:

1. It is a **necessary condition** with essentially no exceptions: 0 / 6,928 zero-option purchases
   on option-requiring tasks scored 1.0.
2. It is **nearly sufficient** once the right item is found: 94.7% / 50% / 100% strict conditional
   on both gates for the three seeds; +0.136 within-(task, item) score lift at 2 options.
3. It **selectively predicts strict success** (r ≈ +0.91–0.97) and not mean reward (r ≈ +0.44–0.49)
   across 71 runs, with the reverse pattern for buy-completion (+0.85 / +0.11). Mean reward is a
   different quantity, gated mostly by episode completion and item quality.
4. **Search / item selection is not the discriminator** between the seeds (77 / 66 / 70% correct
   item), and across the population it does not predict strict success at all (r = +0.07).
5. The threshold is **2 options per episode**, because 85 of the 200 validation tasks require
   exactly 2; at the run level, the discontinuity is at roughly **40% of episodes clicking ≥2
   options**, and only 9 of 71 1.5B runs are above it.

There is a real two-gate structure: `buy-completion` gates mean reward, `option-clicking` gates
strict success, and `item selection` sits mostly with mean reward. `duet_v31` (options yes, buy
no) and `seed2027` (buy yes, options no) are the two failure modes. Any claim of the form "DUET
improves WebShop" should be read as "DUET's winning configurations cross the option gate"; the
paper's headline WebShop number is the product of crossing a threshold, not of a smooth
improvement.

---

## 6. What this implies for the paper / rebuttal

- **Report both metrics and name the gate.** Strict success and mean reward measure different
  things on WebShop (they correlate at only ρ ≈ 0.5 across the run population). Presenting strict
  success alone makes the WebShop result look bimodal and seed-fragile; presenting both, with the
  option-gate explanation, converts "lucky seed" into "identified a threshold mechanism".
- **The seed variance is explainable, not noise.** Say explicitly that the three seeds differ by
  12× on the option gate and by <11 points on item selection, and that conditional on the gate all
  three convert at 50–100%. That is a much stronger reviewer answer than a variance band.
- **Add "fraction of validation episodes clicking ≥2 options" as a diagnostic** to the WebShop
  table or appendix. It predicts final strict success at r = +0.96 in-sample and r = +0.85 from
  training rollouts 20 steps out, so it is a cheap early-stopping / run-health signal.
- **The 100-step budget is the binding constraint.** Option-clicking is U-shaped in time and the
  late rise only starts around step 80. Runs that have not begun climbing by step 60–80 do not
  cross the gate. Either extend the budget or say plainly that WebShop strict success at 100 steps
  is a threshold-crossing event.
- **A targeted fix is available and testable.** Because the gate is a specific action class,
  anything that raises option-click frequency — an option-coverage term in the State Channel
  progress map, an auxiliary bonus for clicking a button that appears in the instruction, or
  simply keeping the teacher weight up longer (teacher demos click 2.03 options before buying) —
  should move strict success directly. `duet_minus_bc` at 41.5% ≥2-options and 16.5% strict shows
  the gate is reachable by more than one configuration.
- **Do not claim causality from this analysis.** These 71 runs differ in μ-schedule, floors, EMA
  and seed simultaneously; the evidence is observational. The within-(task, item) contrast and the
  same-item paired contrast are the closest thing to a controlled comparison here, and both point
  the same way.

---

## 7. Gaps and caveats

- **The absolute option counts here disagree with the ones in the brief** (paper 0.95 vs 2.31 mean
  at steps 81–100). Direction, ordering and all reward numbers reproduce; the counting rule
  differs. Cross-run comparisons in this note are internally consistent but should not be mixed
  with the earlier figures.
- 69 of 200 validation tasks have no known 1.0 solution anywhere in the population, so "correct
  item" is undefined there. All item-quality analyses are restricted to the other 131. If the
  unknown-solution tasks are systematically harder in item selection, item quality is
  under-measured.
- "Correct item" is inferred from other runs' perfect scores, not from WebShop ground truth. The
  goal set was built leave-trio-out / leave-one-out to remove the obvious circularity, but a
  product that no run ever solved perfectly cannot be credited.
- The run-level validation correlations (§1) are partly mechanical, since the option count and the
  strict success are computed from the same episodes. §1b (training rollouts → validation) and §2
  (within-task-within-item) are the non-circular evidence; they are somewhat weaker (r ≈ 0.83)
  but still strongly dissociated.
- Everything is at step 100 only. Runs with fewer than 100 steps, or without step-100 validation,
  are excluded (10 dirs under `experiments/webshop/` have rollouts but no `validation_log/100`).
- Several runs have episodes whose instruction cannot be parsed (the policy's first search
  returned nothing usable); these are dropped from task-keyed analyses. Worst cases:
  `duet_v5` 200/200, `duet_v25` 182/200, `duet_v21` 121/200, `onpolicy` 93/200. These are
  degenerate runs and are excluded by the non-degenerate robustness check anyway.
- 3B and 7B runs were only pooled, not analysed separately; the 7B baselines (50% strict) sit at
  the top of the option-clicking scale too but n is too small to say more.
- I did not verify that all 71 runs used identical decoding settings at validation; I assumed
  greedy for all based on the trio.

---

## Appendix: per-run scatter table (71 1.5B runs, sorted by val@100 strict success)

`vOpt` = mean distinct options before buy (greedy val, step 100); `v≥2%` = frac. episodes with ≥2;
`vBuy%` = frac. reaching `click[buy now]`; `tOpt` / `t≥2%` = same from on-policy training rollouts
at steps 81–100 (teacher rows excluded).

| run | strict% | meanR | vOpt | v≥2% | v≥1% | vBuy% | vAct | tOpt | t≥2% |
|---|---|---|---|---|---|---|---|---|---|
| ws_swC_v_pk03_v00.BUGGY_1717 | 39.50 | 0.713 | 2.64 | 86.5 | 95.0 | 57.5 | 6.99 | – | – |
| ws_swC_v_pk03_v00.LATCH_V1_2234 | 36.50 | 0.722 | 1.96 | 80.0 | 89.5 | 91.0 | 5.10 | 1.64 | 59.0 |
| **webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06 (paper)** | **35.50** | 0.706 | 1.80 | 71.0 | 92.0 | 97.0 | 5.30 | 0.95 | 28.0 |
| ws_swC_v_pk03_v00 | 28.50 | 0.674 | 2.04 | 71.0 | 89.5 | 93.0 | 5.62 | 1.79 | 61.8 |
| ws_swC_v_pk04_v00 | 26.50 | 0.691 | 1.85 | 77.0 | 91.5 | 94.5 | 5.25 | 1.86 | 74.9 |
| webshop_qwen1.5b_duet_v24 | 22.00 | 0.678 | 1.22 | 38.5 | 78.0 | 99.0 | 4.30 | 0.52 | 11.5 |
| webshop_qwen1.5b_duet_swB_01_pk03_v10_ema02 | 20.50 | 0.502 | 1.35 | 43.5 | 78.5 | 72.0 | 8.26 | 1.17 | 31.6 |
| webshop_qwen1.5b_duet_minus_bc | 16.50 | 0.537 | 1.52 | 41.5 | 91.5 | 78.5 | 7.53 | 0.90 | 20.1 |
| webshop_qwen1.5b_duet_v37 | 16.00 | 0.532 | 1.39 | 38.5 | 89.5 | 78.5 | 9.79 | 1.07 | 27.8 |
| webshop_qwen1.5b_duet_swA_02_peak02 | 13.50 | 0.545 | 1.31 | 34.0 | 84.0 | 80.0 | 7.85 | 0.77 | 15.6 |
| webshop_qwen1.5b_duet_swC_03_pk03_v12_ema02 | 13.50 | 0.589 | 1.18 | 28.5 | 86.5 | 88.0 | 6.25 | 0.78 | 16.0 |
| webshop_qwen1.5b_chord | 11.50 | 0.603 | 1.21 | 26.5 | 92.0 | 92.0 | 6.57 | 1.20 | 33.0 |
| webshop_qwen1.5b_duet_v39 | 11.50 | 0.605 | 0.72 | 18.0 | 51.5 | 94.0 | 4.53 | 0.73 | 14.5 |
| webshop_qwen1.5b_duet_swA_11_pk05_v10 | 11.00 | 0.534 | 1.19 | 29.0 | 84.5 | 79.5 | 7.75 | 1.17 | 32.3 |
| webshop_qwen1.5b_duet_minus_dr3 | 9.50 | 0.502 | 1.11 | 19.5 | 80.5 | 77.0 | 9.29 | 0.83 | 19.1 |
| webshop_qwen1.5b_duet_swA_05_peak06 | 8.00 | 0.521 | 1.06 | 15.5 | 77.5 | 81.0 | 7.80 | 0.75 | 14.2 |
| webshop_qwen1.5b_duet_swA_12_pk05_ema02_v10 | 8.00 | 0.555 | 1.22 | 27.5 | 77.0 | 86.0 | 6.55 | 0.48 | 10.0 |
| webshop_qwen1.5b_duet_v10 | 7.00 | 0.571 | 0.41 | 10.0 | 30.5 | 96.0 | 3.60 | 0.45 | 7.7 |
| webshop_qwen1.5b_duet_swA_03_peak04 | 5.50 | 0.639 | 0.84 | 2.5 | 82.0 | 100.0 | 3.88 | 0.47 | 0.7 |
| webshop_qwen1.5b_duet_v3 | 5.50 | 0.446 | 0.87 | 14.0 | 68.0 | 72.0 | 8.99 | 0.49 | 8.9 |
| webshop_qwen1.5b_duet_v39_postfix | 5.50 | 0.607 | 0.67 | 2.0 | 64.5 | 97.0 | 6.04 | 0.26 | 2.4 |
| webshop_qwen1.5b_duet_v11 | 5.00 | 0.388 | 0.67 | 12.5 | 51.5 | 67.5 | 5.76 | 0.45 | 8.8 |
| webshop_qwen1.5b_duet_v40b | 4.50 | 0.496 | 0.90 | 10.5 | 75.0 | 82.0 | 7.25 | 0.74 | 14.2 |
| webshop_qwen1.5b_duet_v43a | 4.50 | 0.569 | 0.58 | 1.5 | 57.0 | 96.5 | 4.10 | 0.64 | 10.1 |
| webshop_qwen1.5b_luffy | 4.50 | 0.573 | 0.65 | 4.5 | 60.5 | 95.5 | 4.26 | 0.33 | 4.7 |
| ws_1_5b_gap_pk03_v10_NOtw | 4.50 | 0.549 | 0.40 | 3.0 | 36.5 | 93.0 | 3.50 | 0.41 | 5.8 |
| webshop_qwen1.5b_duet | 4.00 | 0.549 | 0.49 | 8.5 | 39.5 | 94.5 | 4.67 | 0.37 | 7.1 |
| webshop_qwen1.5b_duet_swA_10_pk05_ema02 | 4.00 | 0.529 | 0.27 | 2.5 | 24.5 | 93.5 | 5.20 | 0.17 | 2.4 |
| webshop_qwen1.5b_duet_swB_03_pk03_v10_ema01 | 4.00 | 0.568 | 0.62 | 7.5 | 49.5 | 95.5 | 5.29 | 0.46 | 9.4 |
| webshop_qwen1.5b_duet_v23 | 4.00 | 0.440 | 0.55 | 10.5 | 42.0 | 74.5 | 7.05 | 0.50 | 9.4 |
| webshop_qwen1.5b_duet_v8 | 4.00 | 0.574 | 0.61 | 1.0 | 60.0 | 93.5 | 3.60 | 0.63 | 10.7 |
| **webshop_qwen1.5b_duet_a100_seed2025** | **3.50** | 0.521 | 0.84 | 6.5 | 76.5 | 88.5 | 4.68 | 0.98 | 19.5 |
| webshop_qwen1.5b_duet_swA_08_ema08 | 3.50 | 0.409 | 0.74 | 15.5 | 50.5 | 69.5 | 9.63 | 0.42 | 7.9 |
| webshop_qwen1.5b_duet_v22 | 3.50 | 0.462 | 0.38 | 6.5 | 29.5 | 81.5 | 6.76 | 0.46 | 10.1 |
| webshop_qwen1.5b_duet_v38 | 3.50 | 0.474 | 0.40 | 4.5 | 33.5 | 82.0 | 7.82 | 0.32 | 6.6 |
| webshop_qwen1.5b_duet_v15 | 3.00 | 0.556 | 0.28 | 5.5 | 22.5 | 97.5 | 3.48 | 0.20 | 3.7 |
| webshop_qwen1.5b_duet_v16 | 3.00 | 0.542 | 0.24 | 4.5 | 17.5 | 98.0 | 3.62 | 0.20 | 4.6 |
| webshop_qwen1.5b_duet_v41b | 3.00 | 0.543 | 0.14 | 1.5 | 12.5 | 96.0 | 3.20 | 0.11 | 1.7 |
| **webshop_qwen1.5b_duet_a100_seed2027** | **2.50** | 0.543 | 0.32 | 1.0 | 30.0 | 93.5 | 4.33 | 0.27 | 3.8 |
| webshop_qwen1.5b_duet_v17 | 2.50 | 0.508 | 0.35 | 6.0 | 26.5 | 90.0 | 5.25 | 0.19 | 4.4 |
| webshop_qwen1.5b_duet_v20 | 2.50 | 0.477 | 0.38 | 8.0 | 28.5 | 85.5 | 4.09 | 0.14 | 2.6 |
| webshop_qwen1.5b_duet_swB_02_pk03_v15_ema02 | 2.00 | 0.539 | 0.15 | 1.5 | 14.0 | 95.5 | 4.85 | 0.12 | 2.0 |
| webshop_qwen1.5b_duet_v13 | 2.00 | 0.477 | 0.37 | 3.5 | 32.0 | 82.0 | 6.88 | 0.15 | 3.0 |
| webshop_qwen1.5b_duet_v28 | 2.00 | 0.495 | 0.41 | 8.5 | 31.5 | 86.5 | 5.53 | 0.37 | 7.8 |
| webshop_qwen1.5b_duet_v39b | 2.00 | 0.320 | 0.34 | 3.0 | 31.0 | 88.0 | 3.85 | 0.57 | 11.9 |
| webshop_qwen1.5b_duet_swA_04_peak05 | 1.50 | 0.548 | 0.01 | 0.0 | 1.5 | 99.0 | 3.28 | 0.00 | 0.0 |
| webshop_qwen1.5b_duet_swA_06_peak07 | 1.50 | 0.549 | 0.00 | 0.0 | 0.0 | 100.0 | 3.02 | 0.01 | 0.1 |
| webshop_qwen1.5b_duet_swC_01_pk03_v10_floor04 | 1.50 | 0.542 | 0.17 | 1.0 | 15.5 | 97.0 | 3.23 | 0.09 | 1.7 |
| webshop_qwen1.5b_duet_v31 | 1.50 | 0.517 | 0.72 | 26.0 | 46.5 | 50.5 | 3.01 | 0.31 | 8.5 |
| webshop_qwen1.5b_duet_v36 | 1.50 | 0.389 | 0.18 | 4.5 | 13.0 | 72.0 | 7.11 | 0.22 | 5.2 |
| webshop_qwen1.5b_duet_minus_sc | 1.00 | 0.450 | 0.03 | 0.0 | 2.5 | 84.5 | 2.88 | 0.14 | 2.9 |
| webshop_qwen1.5b_duet_v12 | 1.00 | 0.431 | 0.06 | 0.5 | 5.0 | 84.5 | 2.92 | 0.06 | 0.4 |
| webshop_qwen1.5b_duet_v14 | 1.00 | 0.528 | 0.01 | 0.0 | 1.0 | 99.5 | 3.13 | 0.02 | 0.4 |
| webshop_qwen1.5b_duet_v18 | 1.00 | 0.501 | 0.00 | 0.0 | 0.0 | 100.0 | 3.79 | 0.00 | 0.1 |
| webshop_qwen1.5b_duet_v19 | 1.00 | 0.469 | 0.02 | 0.0 | 2.0 | 95.5 | 3.10 | 0.09 | 1.7 |
| webshop_qwen1.5b_duet_v2 | 1.00 | 0.521 | 0.01 | 0.0 | 0.5 | 96.5 | 3.20 | 0.04 | 0.5 |
| webshop_qwen1.5b_duet_v21 | 1.00 | 0.095 | 0.42 | 2.0 | 38.0 | 36.0 | 2.39 | 0.63 | 15.7 |
| webshop_qwen1.5b_duet_v29 | 1.00 | 0.511 | 0.03 | 0.0 | 3.0 | 94.0 | 3.36 | 0.02 | 0.1 |
| webshop_qwen1.5b_duet_v30 | 1.00 | 0.520 | 0.00 | 0.0 | 0.0 | 96.5 | 3.00 | 0.02 | 0.3 |
| webshop_qwen1.5b_duet_v32 | 1.00 | 0.465 | 0.06 | 1.0 | 5.0 | 88.5 | 3.54 | 0.12 | 2.1 |
| webshop_qwen1.5b_duet_v33 | 1.00 | 0.520 | 0.09 | 1.0 | 8.0 | 96.5 | 3.23 | 0.07 | 0.9 |
| webshop_qwen1.5b_duet_v39c_postfix | 1.00 | 0.511 | 0.40 | 7.5 | 30.5 | 91.5 | 5.41 | 0.21 | 3.1 |
| webshop_qwen1.5b_duet_v7 | 1.00 | 0.473 | 0.01 | 0.0 | 1.5 | 89.0 | 2.98 | 0.02 | 0.1 |
| webshop_qwen1.5b_duet_v9 | 1.00 | 0.533 | 0.06 | 0.5 | 5.0 | 97.0 | 3.19 | 0.15 | 2.5 |
| ws_1_5b_swC02_da | 1.00 | 0.548 | 0.00 | 0.0 | 0.0 | 100.0 | 3.02 | 0.03 | 0.3 |
| webshop_qwen1.5b_duet_v4 | 0.50 | 0.343 | 0.40 | 12.5 | 25.5 | 39.0 | 2.06 | 0.23 | 5.1 |
| webshop_qwen1.5b_duet_v6 | 0.50 | 0.305 | 0.01 | 0.0 | 1.5 | 62.0 | 2.50 | 0.04 | 0.5 |
| webshop_qwen1.5b_onpolicy | 0.50 | 0.152 | 0.06 | 0.0 | 5.5 | 35.5 | 1.84 | 0.02 | 0.1 |
| webshop_qwen1.5b_duet_minus_baseline_sep | 0.00 | −0.100 | 0.01 | 0.0 | 1.0 | 0.0 | 1.02 | 0.00 | 0.0 |
| webshop_qwen1.5b_duet_v25 | 0.00 | −0.041 | 0.04 | 1.0 | 3.0 | 8.5 | 2.18 | 0.13 | 3.1 |
| webshop_qwen1.5b_duet_v5 | 0.00 | −0.100 | 0.00 | 0.0 | 0.0 | 0.0 | 1.00 | 0.00 | 0.0 |

### Reproduction

Analysis scripts (read-only) are in the session scratchpad
`/tmp/claude-1000/-data-home-qisheng-EvolAnalsis/a5d90f98-198d-42a4-aeb3-820cd312fa72/scratchpad/`:
`ws_lib.py` (action parsing + episode stats), `run_cross.py` (§1), `run_robust.py` (§1, §4),
`run_thresh.py` (§1b, §2), `run_paired2.py` (§3, §4). Run with
`PYTHONPATH=<scratchpad> /data/home/qisheng/miniconda3/envs/duet/bin/python <script>` from
`/data/home/qisheng/EvolAnalsis`.
