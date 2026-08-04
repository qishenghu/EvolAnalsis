# Anatomy of the escape event — WebShop 1.5B DUET, paper run vs. two replicates

Read-only analysis of saved rollouts and training logs. No process, GPU, checkpoint or experiment
directory was touched.

**Sources.** Rollouts: `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_{1..100}.jsonl`
(64 rows/step; teacher rows dropped via `diag.is_teacher`). Metrics: the `step:N - k:v - k:v ...`
lines in `logs/<run>.log` (100 steps, 382 distinct keys). Validation: `experiments/webshop/<run>/validation_log/{50,100}.jsonl`.

| short name | run directory |
|---|---|
| `paper2026` | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` |
| `seed2025`  | `webshop_qwen1.5b_duet_a100_seed2025` |
| `seed2027`  | `webshop_qwen1.5b_duet_a100_seed2027` |

## 0. Metric definition, and how it was validated

An **option click** is `click[v]` where `v` is not `buy now`, not `back to search` / `< prev` /
`next >` / `description` / `features` / `reviews` / `attributes`, not the literal `search` button,
and does not match the item-ASIN pattern `^[bB][0-9a-zA-Z]{9}$`. `opt_u` = number of *distinct*
option clicks emitted **before the first `click[buy now]`**.

Two independent checks that this is the right quantity:

- On teacher rows it returns **2.10 distinct options, 89.5 % with ≥2, 7.25 actions/trajectory**
  (all three runs, identical to 2 d.p.), against the previously established teacher figures of
  2.03 / 84.2 % / 7.31.
- Mean on-policy episode score over steps 81–100 reproduces the established numbers **exactly**:
  0.567 (paper), 0.477 (seed2027).

The previously circulated per-block figures (paper 2.61, 4.86, 1.36 …) additionally count the
item-page click, so their trough sits at ≈1.1 (= one item click + ~0.1 options) rather than ≈0.1.
The column `nonnav` below is that variant, kept for continuity; `opt_u` is the cleaner signal and is
used for every claim in this document.

---

## 1. When the paper run's option-clicking rises — and it is a **ramp, not a step**

Per-step, on-policy only. `f≥1` = fraction of episodes clicking at least one option before buying,
`f≥2` = at least two. Values are 5-step rolling means; onset = first step where the rolling mean
crosses baseline + 3 sd (baseline = steps 45–70, the quiescent plateau) and stays above for 4 steps.

| run | signal | baseline (45–70) | sd | first sustained z>2 | first sustained z>3 | z at step 100 |
|---|---|---|---|---|---|---|
| paper2026 | f≥1 | 0.079 | 0.051 | **step 73** | **step 75** | 14.2 |
| paper2026 | f≥2 | 0.018 | 0.018 | step 77 | **step 80** | 23.5 |
| seed2025 | f≥1 | 0.161 | 0.072 | step 81 | **step 82** | 8.1 |
| seed2025 | f≥2 | 0.031 | 0.025 | step 82 | **step 82** | 5.0 |
| seed2027 | f≥1 | 0.089 | 0.068 | step 89 | **never** | 1.3 |
| seed2027 | f≥2 | 0.024 | 0.027 | never | **never** | −0.2 |

**It is gradual.** A linear fit to paper2026's per-step `f≥1` over steps 70–100 gives slope
**+0.0222 per step, R² = 0.87** — a 25-step ramp from 8 % to 86 %, not a discontinuity. The
rolling-5 series makes this plain (paper2026 `f≥1`):

```
step  70   72   74   76   78   80   82   84   86   88   90   92   94   96   98  100
      .11  .15  .22  .24  .27  .32  .39  .40  .45  .47  .52  .51  .65  .73  .76  .80
```

A single-changepoint fit maximising the Welch t places the break at step 93–95, but that is an
artefact of fitting one step to a ramp; the onset test above is the correct read.

**The escape has two stages.** `f≥1` lifts off at step 73–75; `f≥2` lifts off 5–7 steps later
(step 77–80) and is still climbing at step 100. The model first learns to click *an* option, then
learns to click a *second* one.

---

## 2. What else changes at step 75 — nothing in the mechanism

Every one of the 382 logged keys with complete coverage was passed through the identical onset test
on the paper run. Ranked by onset step (behavioural series added for comparison; timing/`perf`/
`global_seqlen`/`grad_dir` bookkeeping keys suppressed):

| onset | dir | metric | base (45–70) | sd | mean (95–100) |
|---|---|---|---|---|---|
| **75** | UP | **behaviour: f≥1 option** | 0.0789 | 0.0505 | 0.778 |
| 77 | UP | behaviour: mean distinct options | 0.105 | 0.074 | 1.296 |
| **80** | UP | **behaviour: f≥2 options** | 0.0183 | 0.0176 | 0.424 |
| 80 | UP | `adaptive_weight/onpolicy_success_ema` | 0.0164 | 0.0052 | 0.0955 |
| 80 | DOWN | `dr3/ess_off_window` | 31.09 | 0.16 | 31.08 |
| 82 | UP | `diag/response_len_onpolicy_mean` | 1364 | 60.7 | 1820 |
| 89 | DOWN | `diag/llm_token_ratio_in_response` | 0.165 | 0.012 | 0.134 |
| 96 | UP | `response_length/mean` | 1717 | 102 | 2140 |
| 96 | UP | `state_channel/unique_states_matched_total` | 163.3 | 20.6 | 256.2 |
| **97** | UP | **`critic/success_onpolicy/mean`** | 0.0163 | 0.0248 | 0.161 |
| 98 | UP | `dr3/disc_loss` | 0.232 | 0.006 | 0.260 |
| 100 | DOWN | `dr3/disc_acc` | 0.9967 | 0.003 | 0.9867 |

**The behaviour moves first.** The only logged quantities that move at or before step 82 are
`adaptive_weight/onpolicy_success_ema` (an EMA of success — a *lagging* readout of the same event,
and it is 5 steps behind `f≥1`), a 0.03-unit wobble in `dr3/ess_off_window` that is numerically
irrelevant (31.09 → 31.08), and on-policy response length, which is the mechanical consequence of
emitting one more env step per episode. `dr3/ess_off_window` is flagged only because its sd is
0.16 on a base of 31.09; it is noise, not signal.

**No DUET mechanism metric announces the escape.** Window means for the paper run:

| metric | 20–44 | 45–70 | 71–85 | 86–100 |
|---|---|---|---|---|
| `chord/mu` | 0.144 | 0.102 | 0.101 | 0.104 |
| `chord/n_expert_tokens` | 360 | 376 | 349 | 350 |
| `dr3/disc_acc` (ema) | 0.912 | 0.997 | 0.998 | 0.992 |
| `dr3/w_off_mean` | 0.796 | 0.625 | 0.647 | 0.661 |
| `dr3/alpha` | 0.106 | 0.112 | 0.105 | 0.108 |
| `duet/teacher_gradient_share` | 0.119 | 0.130 | 0.110 | 0.089 |
| `state_channel/progress_onpolicy_mean` | 0.333 | 0.372 | 0.383 | 0.410 |
| `state_channel/bonus_vs_reward_ratio` | 0.116 | 0.124 | 0.117 | 0.115 |
| `actor/entropy_loss` | 0.541 | 0.608 | 0.589 | 0.534 |
| `actor/kl_loss` | 0.300 | 0.471 | 0.586 | 0.778 |
| `actor/grad_norm` | 2.97 | 2.64 | 3.06 | 3.36 |
| `critic/success_onpolicy/mean` | 0.012 | 0.016 | 0.050 | **0.119** |
| `critic/rewards_onpolicy/mean` | 0.502 | 0.545 | 0.608 | 0.660 |

`actor/pg_loss`, `actor/ppo_kl`, `critic/advantages/mean`, `diag/onpolicy_adv_pos_ratio` and
`duet/group_reward_variance_mean` produce **no onset at either the 2-sd or the 3-sd threshold**;
they oscillate about a flat mean across steps 65–100 (|z| at step 100: 0.06, 0.14, 1.61, 0.53,
1.70). Three optimiser statistics do eventually move, all *after* the behavioural onset and only at
the looser 2-sd threshold: `actor/kl_loss` up at step 87, `actor/grad_norm` up at step 89,
`actor/entropy_loss` down at step 95. Both are slow drifts that begin inside the baseline window;
neither has a break at step 73–80.

(`dr3/w_off_mean` and `chord/mu` are logged at 90–97 of the 100 steps rather than all 100, so they
were excluded from the completeness-filtered onset ranking. Their window means, tabulated below,
are flat to within 0.05 and 0.003 respectively.)

**Caveat on "which moves first".** `critic/success_onpolicy/mean` is a far noisier readout of the
same underlying transition: baseline 0.016 ± 0.025 (sd larger than the mean) versus 0.079 ± 0.051
for `f≥1`. The option signal detects the transition ~20 steps earlier mainly because its
signal-to-noise is ~2.4× better, not necessarily because it is causally upstream. A cross-correlation
of the two rolling-5 series over steps 50–95 is broad and flat (r = 0.83–0.87 for every lag in
−5…+4), so **the lead–lag cannot be resolved from 46 noisy points**; the honest statement is that
option-clicking is the earliest *detectable* manifestation, not that it provably precedes success.

---

## 3. Breadth — the escape is broad, and memorisation is architecturally impossible

Each run trains one epoch: **800 distinct task_ids, 8 tasks per step, 8 rollouts per task, 100
steps.** Verified for all three runs: 800 distinct task_ids total, exactly 8 distinct tasks per
step, and the maximum number of times any task_id appears across the whole run is 8 — i.e. its own
single group. `training/epoch` is 0 at step 100.

**No task is ever revisited.** Every step's 8 tasks are fresh. The escape therefore cannot be
memorisation of the 800; it is measured entirely on tasks the policy has never seen.

Distinct tasks (out of the 8 × window-length seen in that window) that produced at least one
≥2-option episode:

| run | steps 45–70 (208 tasks) | steps 71–85 (120 tasks) | steps 86–100 (120 tasks) |
|---|---|---|---|
| paper2026 | 24 (11.5 %) | 54 (45.0 %) | **98 (81.7 %)** |
| seed2025 | 41 (19.7 %) | 47 (39.2 %) | 86 (71.7 %) |
| seed2027 | 29 (13.9 %) | 26 (21.7 %) | 24 (20.0 %) |

Concentration falls as the escape proceeds: in paper2026 the five tasks contributing the most
≥2-option episodes account for 30 % of them in steps 45–70, 19 % in 71–85 and **11 %** in 86–100
(the floor for 120 tasks × 8 rollouts is ≈ 4 %). The escape spreads across the task distribution
rather than concentrating.

**It is a decision change, not a navigation change.** Separating "reached a page offering ≥2 option
buttons" from "clicked one once there":

| run | window | reached an option page | of those, clicked ≥1 | of those, clicked ≥2 |
|---|---|---|---|---|
| paper2026 | 45–70 | 86.5 % | 8.1 % | 2.0 % |
| paper2026 | 71–85 | 87.9 % | 36.7 % | 10.6 % |
| paper2026 | 86–100 | 85.1 % | **75.9 %** | **37.3 %** |
| seed2025 | 45–70 | 81.1 % | 18.3 % | 3.6 % |
| seed2025 | 71–85 | 83.3 % | 42.2 % | 11.4 % |
| seed2025 | 86–100 | 86.5 % | **88.3 %** | 22.6 % |
| seed2027 | 45–70 | 81.9 % | 10.0 % | 3.0 % |
| seed2027 | 71–85 | 82.6 % | 18.8 % | 4.9 % |
| seed2027 | 86–100 | 81.9 % | 24.3 % | 4.0 % |

**Reach is flat at 81–88 % in every run and every window.** All three policies stand in front of the
option buttons about equally often throughout training. The entire divergence is in whether they
press them.

---

## 4. seed2025 is mid-escape — it completed stage 1 and stalled at stage 2

This is the sharpest result in the analysis, and it reframes the "aborted escape" question.

The right statistic is **P(clicks a 2nd option | clicked a 1st)**. Training rollouts:

| run | 45–70 | 71–85 | 86–100 |
|---|---|---|---|
| paper2026 | 0.231 | 0.287 | **0.488** (rising) |
| seed2025 | 0.192 | 0.266 | 0.254 (stalled) |
| seed2027 | 0.273 | 0.255 | 0.141 (decaying) |
| *teacher rows* | *0.922* | *0.920* | *0.922* |

seed2025's `f≥1` rolling-5 actually **overtakes** the paper run — it peaks at 0.85 (step 97) versus
the paper's 0.80 — while its `f≥2` plateaus around 0.22–0.27 from step 86 and then *declines* to
0.15 by step 100. seed2025 learned "click one option, then buy". The paper run learned "keep
clicking options until they are all set".

The held-out validation rollouts at step 100 make it unambiguous (n = 200 each; the strict-success
column reproduces the reported 35.5 / 3.5 / 2.5 % exactly):

| run | opt = 0 | opt = 1 | opt ≥ 2 | **P(≥2 \| ≥1)** | val strict | val mean reward |
|---|---|---|---|---|---|---|
| paper2026 | 8.0 % | 21.0 % | **71.0 %** | **0.772** | 35.5 % | 0.706 |
| seed2025 | 23.5 % | **70.0 %** | 6.5 % | **0.085** | 3.5 % | 0.521 |
| seed2027 | **70.0 %** | 29.0 % | 1.0 % | **0.033** | 2.5 % | 0.543 |
| *teacher rows (training)* | *2.9 %* | *7.6 %* | *89.5 %* | *0.922* | — | *1.0* |

At val@50 all three are indistinguishable on the same axis (`f≥1` = 1.0 / 9.5 / 1.0 %,
`f≥2` = 0.5 / 2.5 / 0.0 %), matching the indistinguishable val@50 strict scores.

So: **yes, seed2025 is mid-escape.** It is 70 % of the way through stage 1 by the end of the budget
and has barely started stage 2. Its block curve ending 2.59, 2.18 is one option plus one item click,
not two options. seed2027 never left the plateau on either stage.

**Option count is necessary but not sufficient.** Conditional strict success on validation:

| run | opt = 0 | opt = 1 | opt ≥ 2 |
|---|---|---|---|
| paper2026 | 0.125 (n=16) | 0.238 (n=42) | **0.415** (n=142) |
| seed2025 | 0.021 (n=47) | 0.036 (n=140) | 0.077 (n=13) |
| seed2027 | 0.014 (n=140) | 0.052 (n=58) | 0.000 (n=2) |

Even the paper run converts only 41.5 % of its ≥2-option episodes into a strict success — it often
clicks two options of the *wrong* values. A two-way decomposition of the 32.0 pp paper-vs-seed2025
gap is strongly order-dependent (composition-first attributes 13.2 pp to the option mix and 18.8 pp
to conditional accuracy; skill-first attributes 29.1 pp and 2.9 pp), i.e. the interaction term
dominates: the paper run both clicks more options *and* picks better ones, and the two cannot be
cleanly separated. seed2025's `opt≥2` cell has n = 13, so its 0.077 carries very wide error bars and
should not be quoted alone.

---

## 5. The teacher channel does not move — and cannot explain the escape

**Teacher demonstration content is stationary.** Teacher rows, by window:

| run | 45–70 | 71–85 | 86–100 |
|---|---|---|---|
| paper2026 | 2.08 opts, 88.6 % ≥2, 7.32 acts | 2.08, 89.2 %, 7.11 | 2.09, 90.3 %, 7.03 |
| seed2025 | 2.06, 89.7 %, 7.22 | 2.21, 93.4 %, 7.35 | 2.03, 88.9 %, 7.18 |
| seed2027 | 2.01, 88.9 %, 7.07 | 2.11, 87.3 %, 7.33 | 2.02, 84.3 %, 7.21 |

No shift at the escape, and no systematic difference between the run that escaped and the two that
did not. `luffy/total_teacher_rollouts` is 6.8–7.1 per step throughout in all three runs.

**The BC weight on option-click tokens is a constant, by construction.** Across all 100 steps of all
three runs, `chord/phi_min = chord/phi_max = 1.000` and `chord/phi_std = 0.000`, with
`chord/mu_mode = 3` and `chord/d_floor = 0.6` fixed. The weighted-SFT term therefore applies a
uniform weight to every teacher token; there is no token-type-specific weighting that could have
been redirected onto option-click tokens, and nothing about it changes at step 75.
`chord/mu` = 0.101–0.104, `chord/n_expert_tokens` = 349–376 across every window of every run.

**The off-policy correction does not move either.** `dr3/w_off_mean` 0.63–0.68,
`teacher_diag/teacher_ratio/mean` 0.51–0.56, `duet/teacher_gradient_share` 0.089–0.167,
`diag/teacher_token_ratio` 0.017–0.024 — all flat, all overlapping between the three runs.

**Answer to Q5: no.** Nothing in the teacher channel — content, mixing rate, BC weight, or density
ratio — coincides with the escape.

### One correction to an established fact

paper2026 and seed2027 draw from the **identical 800-task pool** (re-verified: the two task_id sets
are equal as sets). But the **per-step order is not identical**: of the 100 steps, **0 share the
same 8-task set**. The epoch shuffle is driven by `seed`, not `task_seed`. So the two runs see the
same tasks — and hence the same teacher trajectories over the run as a whole — but in a different
order and in different groups of 8. This should be stated when the pair is described as a
controlled comparison; it is still far tighter than seed2025 (89 tasks shared), but it is not
step-for-step matched.

---

## 6. Summary of the escape event

1. Steps 1–20: high-entropy random clicking (`f≥1` 25–48 %), then collapse.
2. Steps 25–70: the local optimum. `f≥1` ≈ 8 %, `f≥2` ≈ 2 %, mean reward 0.50–0.55, strict success
   1.4 % (n = 1479 on-policy episodes over steps 45–70). Policy has learned "search → click item →
   buy": partial credit, cheap, stable.
3. **Steps 73–75:** `f≥1` begins a smooth 25-step ramp (+2.2 pp/step, R² 0.87). No logged metric
   moves; teacher channel, DR3, SC, CHORD-BC and every optimiser statistic are flat.
4. **Steps 77–80:** `f≥2` lifts off, five steps behind stage 1, and keeps rising to step 100.
5. **Steps 86–100:** the behaviour is broad — 82 % of the (never-before-seen) tasks in the window
   produce ≥2-option episodes, and concentration on the top-5 tasks falls to 11 %.
6. **Step ~97:** `critic/success_onpolicy/mean` finally clears its own 3-sd threshold — the escape
   becomes visible in the headline training metric roughly 20 steps after it began, and only 3 steps
   before the run ends.
7. seed2025 runs the same script 7 steps late and stops after stage 1 (`P(≥2|≥1)` = 0.085 on
   validation vs 0.772). seed2027 never leaves step 2.

The escape is a **slow, broad, two-stage behavioural transition that the logged metrics do not
anticipate**, and it finishes inside the 100-step budget in exactly one of three runs.
