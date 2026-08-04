# WebShop 1.5B: the local optimum, measured from the reward data itself

**Question.** The behavioural story (short path "search → click item → buy" vs. long path
"… → click each option → buy") is established. This note asks whether the *reward function itself*,
as seen through the on-policy rollouts, makes the short path a genuine attractor for a
policy-gradient objective — and whether any of DUET's shaping terms push against it.

**Answer: yes, and no.** On 88.4% of the 800 training tasks the short path has a hard reward
ceiling below 1.0 — 13,657 episodes with ≤1 option click on such tasks produced **zero** exact
successes — while the marginal *mean* reward of the click that crosses that ceiling is zero or
negative during the exact window in which the policy collapses onto the short path. The State
Channel is *exactly* neutral to option clicking: over 16,647 clean option clicks, ΔΦ = 0 in
**100.00%** of cases.

---

## 0. Provenance and method

**Runs** (on-policy rollout dumps, `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_{1..100}.jsonl`,
64 rows/step, 5,714–5,732 on-policy episodes per run after dropping `diag.is_teacher`):

| label | directory |
|---|---|
| paper seed2026 | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` |
| seed2025 | `webshop_qwen1.5b_duet_a100_seed2025` |
| seed2027 | `webshop_qwen1.5b_duet_a100_seed2027` |

⚠️ **Landmine.** The paper's run is `..._swC_02_pk03_v10_floor06`, per
`evidence_factual_bdey.md:300` and `DECISION_webshop_1p5b_cell.md:21`. There is a *different*
run on disk literally named `webshop_qwen1.5b_duet`; it is not the paper run and its behaviour
is different (its steps 81–100 mean reward is 0.525, not 0.567). I analysed the wrong one first.

**Cross-check that the extraction is sound.** Two independent checks, both exact:
mean `reward.outcome` over training steps 81–100 = **0.567** (paper) / **0.546** (seed2025), and
from `experiments/webshop/<run>/validation_log/100.jsonl`, mean score / P(score=1) =
**0.7057 / 35.5%** (paper), 0.5212 / 3.5% (seed2025), 0.5435 / 2.5% (seed2027) — all matching
`DECISION_webshop_1p5b_cell.md` exactly.

**Definitions.**
- `k` = number of **distinct** option clicks strictly *before* the first `click[Buy Now]`.
  An "option click" is `click[X]` where X is not an ASIN (`^[bB][0-9a-zA-Z]{9}$`), not
  `buy now`, and not a navigation element (`back to search`, `< prev`, `next >`,
  `description`, `features`, `reviews`).
- `reward` = `reward.outcome`, the raw environment score. This **excludes** the SC bonus
  (`diag.reward_original` == `reward.outcome`; `diag.reward_sum` is the one that adds `sc_bonus`).
- "requested attributes" `r` = count of `color` / `size` / `fit type` in the instruction, parsed
  with the repo's own `extract_instruction_attributes()` (`state_progress.py:93`). Price is
  excluded from `r` because satisfying it requires no click.

> **Note on a definitional mismatch with the earlier behavioural pass.** The established facts
> quote "≥2 options in 70.7% (paper) / 13.0% (seed2027)" for steps 81–100 of *training*. On
> training rollouts I get 20.6% / 5.9% for that window. But on the **validation** set at step 100
> I get **71.0% / 2.0%** — so the earlier pass's headline behavioural numbers are almost certainly
> validation-side, not training-side (§7a). The shape reproduces either way. **All numbers below
> are training-side unless explicitly marked validation**, under the definition stated above.

---

## 1. Reward as a function of options clicked before Buy Now

Pooled over all three runs, on-policy only:

| window | k | n | share | mean r | median r | **P(r = 1.0)** |
|---|---|---|---|---|---|---|
| **early (1–40)** | 0 | 5211 | 75.8% | 0.312 | 0.429 | 0.77% |
| | 1 | 978 | 14.2% | 0.154 | −0.100 | 2.15% |
| | 2 | 395 | 5.7% | 0.082 | −0.100 | 3.29% |
| | 3+ | 291 | 4.2% | 0.041 | −0.100 | 0.00% |
| **mid (41–60)** | 0 | 3038 | 88.8% | 0.446 | 0.500 | 1.42% |
| | 1 | 298 | 8.7% | 0.440 | 0.571 | 4.36% |
| | 2 | 60 | 1.8% | 0.518 | 0.667 | 10.00% |
| | 3+ | 27 | 0.8% | 0.413 | 0.450 | 7.41% |
| **late (61–100)** | 0 | 4374 | 63.7% | 0.459 | 0.571 | 1.74% |
| | 1 | 1713 | 25.0% | 0.575 | 0.667 | 5.20% |
| | 2 | 602 | 8.8% | 0.641 | 0.754 | **20.76%** |
| | 3+ | 173 | 2.5% | 0.587 | 0.750 | 14.45% |

Per run, late (61–100) — the divergence is visible in the *share* column, not in the reward column:

| run | k=0 share | k=1 | k=2 | k≥3 | P(r=1) at k=2 |
|---|---|---|---|---|---|
| paper seed2026 | 57.9% | 24.2% | **13.8%** | 4.0% | **26.03%** |
| seed2025 | 53.0% | 35.1% | 9.4% | 2.5% | 18.14% |
| seed2027 | 80.3% | 15.6% | **3.1%** | 1.0% | 5.56% |

Two facts to hold together:
1. **Early in training, mean reward is monotonically *decreasing* in k** (0.312 → 0.154 → 0.082 → 0.041).
   Extra option clicks at that stage are noise: unguided, often invalid, and they displace the
   actions that actually earn credit.
2. **The entire strict-success mass lives at k ≥ 2** (late: 20.8% at k=2 vs 1.7% at k=0), and
   mean reward becomes *increasing* in k only once a run has already started clicking options.
   The reward landscape the policy sees is not the landscape it needs to see.

---

## 2. The hard ceiling — why this is a genuine local optimum, not a soft preference

Splitting by the number of option attributes the instruction actually requests (all 100 steps,
all three runs pooled):

| tasks | policy | n episodes | **exact successes** | mean r | **max r observed** |
|---|---|---|---|---|---|
| r = 0 (3.2%) | k = 0 | 486 | 158 (32.5%) | 0.525 | 1.000 |
| r = 1 (8.4%) | k = 0 | 1157 | **1** (0.09%) | 0.397 | 1.000 |
| r = 1 | k = 1 | (see §4) | 118 total at k ≤ 1 | 0.429 | 1.000 |
| **r = 2 (69.6%)** | **k = 0** | **8310** | **0** (0.000%) | 0.377 | **0.833** |
| **r = 2** | **k ≤ 1** | **10673** | **0** (0.000%) | 0.389 | **0.917** |
| r = 2 | any k | 11969 | 162 (1.35%) | 0.389 | 1.000 |
| **r = 3 (18.8%)** | **k ≤ 1** | **2984** | **0** (0.000%) | 0.411 | **0.818** |
| r = 3 | any k | 3162 | 5 (0.16%) | 0.405 | 1.000 |

**13,657 episodes on tasks requiring ≥2 option clicks, played with ≤1 option click, produced
exactly zero strict successes**, with an observed reward ceiling of 0.83–0.92. The short path is
not merely worse — on 88.4% of the training set it is *incapable* of the reported metric,
no matter how well it is executed. Everything the policy can learn while staying on it is
progress toward a wall.

---

## 3. Marginal return: is there a flat or negative region?

**Yes — and it sits exactly at the step that crosses the ceiling.**

Restricting to r = 2 tasks (69.6% of the 800), pooled across runs:

| window | E[r\|0] | E[r\|1] | Δ | E[r\|2] | Δ | P(r=1) at k=0 → 1 → 2 |
|---|---|---|---|---|---|---|
| early 1–40 | 0.288 | 0.153 | **−0.134** | 0.095 | **−0.059** | 0.00% → 0.00% → 3.60% |
| mid 41–60 | 0.429 | 0.427 | **−0.002** | 0.551 | +0.124 | 0.00% → 0.00% → 10.91% |
| late 61–100 | 0.449 | 0.584 | +0.135 | 0.666 | +0.082 | 0.00% → 0.00% → **24.60%** |

and for r = 3 tasks (18.8%), the first click is *negative* in every window
(early −0.272, mid −0.202, late −0.082), with P(r=1) = 0.00% at k ≤ 2 throughout.

The structural point: **the transition 1 → 2 is the only one that unlocks any probability of
r = 1 on the majority task type, and it is precisely the transition with no mean-reward signal
during the collapse window (steps 21–60).** Blocks 21–60 are where all three runs drop to
k ≈ 0.1–0.4 and 1.2–7.7% of episodes reach k ≥ 2 (§6 table).

### The quantity GRPO actually differentiates

Mean reward across tasks is not what the optimizer sees; GRPO normalizes *within* the rollout
group for one task. Two direct measurements:

**(a) Logged advantage `diag.adv_mean`, by k** (this is the value that entered the loss):

| window | k=0 | k=1 | k=2 | k≥3 |
|---|---|---|---|---|
| early 1–40 | **+0.062** | +0.035 | **+0.011** | +0.042 |
| mid 41–60 | +0.048 | +0.059 | +0.091 | +0.036 |
| late 61–100 | **−0.008** | +0.047 | **+0.111** | +0.092 |

Early, the advantage *decreases* in k: the update is actively pulling probability mass off the
long path. The sign flips only after take-off.

**(b) Within-group paired reward difference** (same task, same step, same group of 8; averaged
over groups containing both arms):

| window | k=0 vs k≥2 | P(Δ>0) | P(r=1): k=0 → k≥2 |
|---|---|---|---|
| early 1–40 | **−0.167** | 22.0% | 0.33% → 1.96% |
| mid 41–60 | +0.071 | 67.1% | 0.00% → 9.20% |
| late 61–100 | +0.193 | 76.4% | 0.00% → **18.38%** |

In the early phase, within the *same task*, a rollout that clicks ≥2 options scores 0.167 lower
than a sibling that does not, and does so 78% of the time. That is a strong, correctly-signed
gradient *away* from the behaviour required for success. Per run, late 61–100 on `k=1 vs k≥2`:
paper +0.106 (P(Δ>0) 69.9%), seed2025 +0.069 (56.6%), seed2027 **+0.002 (37.0%, median −0.004)** —
seed2027 never left the flat region.

### Cost asymmetry makes the flat region slightly downhill

`env_params` (verified in the run configs): `invalid_action_penalty: -0.05`,
`invalid_action_penalty_cap: -0.1`, `invalid_action_final_reward: -0.1`.

| | option-click invalid rate |
|---|---|
| paper seed2026 | 6.1% (381 / 6267) |
| seed2025 | 6.6% (562 / 8522) |
| seed2027 | 6.4% (357 / 5620) |
| **teacher trajectories** | **0.0% (0 / 1438)** |

An exploratory extra option click therefore carries ≈ 6.4% × (−0.05) ≈ **−0.003** in expected
penalty on top of a marginal mean return of ≈ 0.000 (r=2, mid window). Under the objective, the
step that unlocks success has *strictly negative* expected value at the moment the policy is
deciding whether to take it. That is the definition of an attractor.

---

## 4. What the instructions actually ask for

`multi_turn.max_steps: 30`; parsed with the repo's own `extract_instruction_attributes()`.
All 800 training tasks were seen in the dumps for each run.

| clickable option attrs requested | paper / seed2027 (identical 800) | seed2025 (different 800) |
|---|---|---|
| 0 | 26 (3.2%) | 25 (3.1%) |
| 1 | 67 (8.4%) | 74 (9.2%) |
| **2** | **557 (69.6%)** | 555 (69.4%) |
| **3** | **150 (18.8%)** | 146 (18.2%) |
| **≥ 1** | **96.8%** | 96.9% |
| **≥ 2** | **88.4%** | 87.6% |

The reward is well-behaved when the requirement matches what the policy does — r = 1 tasks,
late window: k = 0 → E[r] 0.461, P(r=1) 0.25%; k = 1 → E[r] **0.760**, P(r=1) **58.11%**. A
+0.30 mean jump and +58pp of success for one click. That signal is unmissable, and all three runs
did learn to click roughly one option late in training.

**It is the wrong lesson for 88.4% of the tasks.** The gradient teaches "click one option"
because that is where the mean reward gradient is, and there is no comparable gradient for
"click *all* of them" until the policy is already doing it. Note also the mirror image: on
r = 1 tasks, clicking a *second* option costs −0.311 mean and −46pp of success — so the policy
is simultaneously being taught "one is enough".

---

## 5. Does the step cap punish long paths? No.

Cap is `multi_turn.max_steps: 30`; the empirical maximum is exactly 30 actions in all three runs.

| bucket | paper: n (share) | mean r | P(r=1) | bought |
|---|---|---|---|---|
| 0–2 actions | 265 (4.6%) | −0.097 | 0.00% | 0.4% |
| **3 (search/click/buy)** | **3577 (62.6%)** | +0.471 | 1.37% | 94.1% |
| 4–5 | 1199 (21.0%) | +0.459 | 6.92% | 79.6% |
| **6–8 (full option path)** | 322 (5.6%) | +0.452 | **13.35%** | 75.5% |
| 9–14 | 142 (2.5%) | +0.179 | 4.23% | 43.0% |
| 15–29 | 117 (2.0%) | +0.015 | 0.85% | 16.2% |
| **30 (at cap)** | 92 (1.6%) | −0.047 | 0.00% | **0.0%** |

The minimal success path for r = 2 needs 5 actions (search, click item, 2 options, buy); for
r = 3, six. The cap of 30 leaves 5–6× headroom and **is not binding**: only 1.6–2.4% of episodes
reach it, and those are degenerate flailing (0–3% ever reach Buy Now, mean reward ≈ −0.03, the
invalid-action floor). Length 6–8 — the correct-path bucket — has the *highest* strict success
rate in every run.

So the cap does not create the local optimum. What penalises long paths is (i) the −0.1 floor for
episodes that wander and never buy (18.7–22.3% of all episodes, mean reward −0.09), and (ii) the
−0.05 per invalid click. Both are consequences of *bad* exploration, not of path length as such.

---

## 6. Is the State Channel neutral to option clicking? It is exactly neutral.

Config: `state_channel.match_mode: attribute_aware`, `beta: 0.2`. Recomputing the repo's own
Φ (`webshop_attribute_aware_potential`, `state_progress.py:252`) on the saved observations:

| transition | n | mean ΔΦ | P(ΔΦ>0) | P(ΔΦ=0) | P(ΔΦ<0) |
|---|---|---|---|---|---|
| after `search[...]` | 19146 | +0.105 | 88.1% | 5.0% | 7.0% |
| after `click[<ASIN>]` | 16460 | **+0.525** | 95.4% | 0.1% | 4.5% |
| after `click[<option>]` | 20409 | **−0.006** | 4.2% | **90.4%** | 5.4% |
| after `click[Buy Now]` | 457 | +0.086 | 39.0% | 35.2% | 25.8% |

Restricting to the clean case — an option click made *on* a product-detail page that *stays* on
a product-detail page (i.e. a valid option selection, not a misfire):

> **n = 16,647 clean option clicks. mean ΔΦ = +0.000000. P(ΔΦ = 0) = 100.00%.
> max |ΔΦ| = 0.000000.** (88.9% of these observations are byte-identical before and after.)

**Not one of 16,647 correct option clicks produced any potential-based step reward.** The reason
is structural, not a bug: WebShop's observation after selecting an option is the same product
page, and `webshop_attribute_aware_potential` scores a product page by *which attributes the page
offers* (`compute_attribute_match_score`, `state_progress.py:198`) — availability, not selection.
Since availability does not change when you click, Φ cannot change. The two terms that *do* move
Φ (reaching search results, +0.105; opening a product, +0.525) are exactly the actions the short
path already performs.

The logged per-episode diagnostics are consistent with this: `sc_bonus` at k=2 exceeds k=0 by
only 0.022–0.044 (e.g. paper late: 0.0706 → 0.0924), and that residual is a *correlation* —
episodes that click options also tend to land on better-matching product pages — not a reward
for the click. `sc_coverage` is flat across k (0.78–0.91) in all three runs.

**Consequence for the paper's own framing.** The State Channel is designed to densify a sparse
outcome reward. On WebShop it densifies the part of the trajectory that was never the bottleneck
and is blind to the part that is. It cannot help escape this local optimum, and the ablation
`−SC` should not be expected to show a WebShop effect through this pathway.

---

## 7. How the escape actually happened: jackpots, not gradients

Fraction of rollout groups containing at least one exact success (left) and fraction of
individual rollouts with r = 1 (right):

| steps | paper seed2026 | seed2025 | seed2027 |
|---|---|---|---|
| 41–50 | 3.8% / 0.53% | 7.5% / 3.14% | 3.8% / 1.05% |
| 51–60 | 6.2% / 2.64% | 6.2% / 1.92% | 2.5% / 1.93% |
| 61–70 | 2.5% / 0.35% | 3.8% / 1.92% | 7.5% / 4.20% |
| 71–80 | **15.0%** / 4.38% | 8.8% / 4.57% | 7.5% / 2.27% |
| 81–90 | **33.8%** / 7.85% | 16.2% / 5.24% | 6.2% / 1.75% |
| 91–100 | **43.8%** / 11.38% | 21.2% / 7.84% | 3.8% / 3.32% |

and the corresponding behaviour (mean k / share k ≥ 2 / mean reward):

| steps | paper | seed2025 | seed2027 |
|---|---|---|---|
| 11–20 | 0.84 / 20.3% / 0.175 | 0.79 / 20.5% / 0.181 | 0.60 / 13.8% / 0.233 |
| 21–30 | 0.17 / 3.3% / 0.410 | 0.38 / 7.7% / 0.380 | 0.32 / 6.1% / 0.400 |
| 51–60 | 0.11 / 1.8% / 0.472 | 0.25 / 4.2% / 0.426 | 0.07 / 1.2% / 0.478 |
| 71–80 | 0.36 / 6.5% / 0.530 | 0.32 / 5.3% / 0.492 | 0.22 / 3.5% / 0.505 |
| 81–90 | 0.75 / 20.6% / 0.561 | 0.94 / 18.4% / 0.519 | 0.31 / 5.9% / 0.438 |
| 91–100 | **1.31 / 41.5% / 0.573** | 1.04 / 21.3% / 0.573 | **0.31 / 3.1% / 0.516** |

Because GRPO normalizes within the group, a single r = 1 in a group whose mean is ≈ 0.55 confers
a large positive advantage on that one rollout. The escape is therefore **variance-driven**:
it requires a group to *stumble* onto a complete option sequence, after which the within-group
normalization amplifies it. Note that mean reward is nearly identical across the three runs at
step 91–100 (0.573 / 0.573 / 0.516) while the group-level jackpot rate differs 11× (43.8% vs
3.8%). This is exactly why val@100 strict is a 10× spread while mean reward is a 1.2× spread.

### 7a. The same landscape on the held-out validation set (N = 200)

| run | step | mean k | k ≥ 2 | mean score | P(score=1) | E[r] and P(r=1) by k |
|---|---|---|---|---|---|---|
| paper | 50 | 0.02 | 0.5% | 0.5219 | 1.0% | k=0: 0.523 / 1.0% (n=198) |
| **paper** | **100** | **1.81** | **71.0%** | **0.7057** | **35.5%** | k=0: 0.485 / 12.5% (n=16) · k=1: 0.663 / 23.8% (n=42) · **k≥2: 0.743 / 41.5% (n=142)** |
| seed2025 | 50 | 0.12 | 2.5% | 0.4390 | 1.5% | k=0: 0.437 / 0.6% (n=181) |
| seed2025 | 100 | 0.84 | 6.5% | 0.5212 | 3.5% | k=0: 0.341 / 2.2% (n=46) · k=1: 0.573 / 3.5% (n=141) · k≥2: 0.593 / 7.7% (n=13) |
| seed2027 | 50 | 0.01 | 0.0% | 0.4591 | 1.0% | k=0: 0.464 / 1.0% (n=198) |
| **seed2027** | **100** | **0.34** | **2.0%** | 0.5435 | 2.5% | k=0: 0.549 / 1.5% (n=136) · k=1: 0.541 / 5.0% (n=60) · k≥2: 0.385 / 0.0% (n=4) |

This is the cleanest single view of the whole story. **At step 50 all three runs are behaviourally
identical** (mean k = 0.02 / 0.12 / 0.01, k ≥ 2 in ≤2.5%) and score 1.0–1.5% strict — matching the
established "indistinguishable at val@50". By step 100 the paper run clicks ≥2 options on **71.0%**
of validation episodes and seed2027 on **2.0%**, a 35× behavioural gap that maps almost one-to-one
onto the 35.5% vs 2.5% strict-success gap. Mean score moves far less (0.706 vs 0.544) because the
short path still collects 0.45–0.55 of partial credit. **The reported metric is a direct readout of
whether the run left the local optimum.**

The teacher demonstrations are the intended source of that stumble, and they are unambiguous:
686 teacher rows per run, 7.25 actions, mean k = 2.10, 89.5% with k ≥ 2, reward 1.000 for
100% of them, and **0/1438 invalid option clicks**. The information is present in the batch
every step; what is missing is a term in the objective that pays for imitating that specific
part of it before the jackpot arrives.

---

## 8. What this implies

1. **The WebShop 1.5B cell is not a seed lottery in a vacuum** — it is a lottery *because* the
   objective has a measured flat/negative marginal region at the decisive action, on 88.4% of
   the training tasks, with a hard ceiling behind it. This is a defensible, quantitative reason
   for the volatility, and it is a property of the benchmark + reward, not of DUET.
2. **The State Channel provably cannot address it** on WebShop (ΔΦ = 0, 16,647/16,647). If the
   paper claims SC provides dense guidance in WebShop, that claim needs qualifying. A one-line
   fix to `compute_attribute_match_score` — score *selected* options, not *available* ones —
   would put a non-zero potential on exactly the missing behaviour; this is a concrete,
   cheap ablation for the rebuttal.
3. **A 100-step budget is inside the take-off**, and the take-off is jackpot-gated. Reporting
   mean reward alongside strict success (already in `main_results_with_reward.tex`) is the
   honest framing; the jackpot-rate table above is the mechanism behind the gap.
4. **The teacher channel is the only term with the right information**, and its behavioural
   signal (k ≥ 2 in 89.5%, zero invalid clicks) is far cleaner than anything the on-policy
   reward provides. This is an argument *for* the paper's thesis — off-policy expert data is
   valuable precisely where the on-policy gradient is flat — and it is measurable rather than
   asserted.

---

## 9. Gaps and caveats

- **Definition mismatch (§0)** with the earlier behavioural pass: its "70.7% / 13.0% click ≥2"
  is labelled *training* steps 81–100 but matches my *validation* step-100 figures (71.0% / 2.0%)
  far better than my training figures (20.6% / 5.9%). Worth confirming which before either set of
  numbers goes into text — they support the same conclusion but should not be mixed.
- **The "options requested" parse** uses the repo's own regex, which handles the standard
  `with color: X, and size: Y, and price lower than Z dollars` template. 1/800 instructions
  yielded no attributes at all. Style/flavor/material options exist in `_WEBSHOP_OPTION_TYPES`
  but are not extracted from instructions, so `r` is a *lower bound* — the true fraction needing
  ≥2 clicks is ≥ 88.4%.
- **Causality.** Everything here is observational: episodes with k = 2 are not randomly assigned.
  The within-group paired comparison (§3b) controls for task and policy-step but not for the
  fact that a rollout which clicks two options is also a rollout that happened to reach a good
  product page. The hard-ceiling result (§2) is immune to this — it is a support constraint,
  not a conditional mean.
- **Φ recomputation** uses the current `state_progress.py`. I did not diff it against the paper
  run's `launcher_record` backup, so the ΔΦ = 0 result is a statement about the code as it
  stands today. Given `classify_webshop_page` and `compute_attribute_match_score` are both
  availability-based, the conclusion is unlikely to be version-sensitive, but it is unverified.
- **`sc_step_deltas`** is present in only a subset of dumps (paper run: absent in the late
  window), so the per-episode step-delta sums could not be compared across k. The direct Φ
  recomputation supersedes this, but the logged field would be an independent confirmation.
- **Validation N = 200 per run**, so the §7a per-k cells are thin at the tails (paper k=0 has
  n=16; seed2027 k≥2 has n=4). The headline columns (mean k, k≥2 share, P(score=1)) are over the
  full 200 and are solid; the per-k breakdown within a run is indicative only.
- **Small-n cells in §3.** Some `req=3` and `k≥3` cells rest on n < 30 and are marked with their
  n throughout. The load-bearing claims (§2 hard ceiling, §3 r=2 marginals, §6 ΔΦ) all rest on
  n > 300.

---

*Scripts: `ws_reward_landscape.py`, `ws_grpo_gradient.py`, `ws_supp.py` in this session's
scratchpad. Read-only; no training, GPU, or experiment state was touched.*
