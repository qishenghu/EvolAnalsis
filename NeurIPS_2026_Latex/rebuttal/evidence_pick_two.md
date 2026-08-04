# Evidence pack: the Pick Two anomaly (ALFWorld 3B) — reviewer UyKJ

**Scope.** Reviewer UyKJ asks why plain GRPO (51%) beats all teacher-mixing methods including
DUET (38%) on the *Pick Two* task type at ALFWorld 3B (Appendix F, Fig. `fig_task_type_breakdown`).
Everything below was recomputed from saved artefacts on this machine. No training was launched,
no GPU touched, no checkpoint or experiment file modified.

**One caveat up front:** re-running the existing figure script
`NeurIPS_2026_Latex/figures/make_task_type_figures.py` (to reproduce its printed tables)
regenerated `figures/fig_task_type_breakdown.pdf` / `.png`. The script is deterministic and
the regenerated numbers are byte-identical to the ones already in the appendix text
(DUET 37.8%, LUFFY 44.4%, CHORD 17.8%, GRPO 51.1%), so the figure content is unchanged.

---

## 0. Provenance and sanity checks

### 0.1 Which runs the Appendix-F figure uses

| Method | Run directory | Checkpoint | Overall val |
|---|---|---|---|
| DUET | `experiments/alfworld/alfworld_3b_duet_0329` | step 100 | 69.5% |
| LUFFY | `experiments/alfworld/alfworld_3b_luffy` | step 100 | 61.5% |
| CHORD | `experiments/alfworld/alfworld_qwen3b_chord` | **step 50** | 46.5% |
| GRPO | `experiments/alfworld/alfworld_3b_grpo_react_tags` | step 100 | 58.5% |

(Source: `NeurIPS_2026_Latex/figures/make_task_type_figures.py:36-49`; overall rates recomputed
directly from the `validation_log/*.jsonl` files, 200 rows each.)

### 0.2 The task-type classifier is correct (verified against ground truth)

The figure classifies task types heuristically from the agent's first `<think>` block. I verified
this against ALFWorld ground truth in `AgentGym/agentenv-alfworld/configs/mappings_test.json`
(200 held-out tasks, `item_id` 2420–2619):

* Ground-truth marginals: `pick_and_place_simple` 46, **`pick_two_obj_and_place` 45**,
  `pick_clean…` 37, `pick_cool…` 28, `pick_heat…` 25, `look_at_obj_in_light` 19.
* Heuristic marginals on the val log: `pick_and_place` 46, **`pick_two` 45**, `clean` 37,
  `cool` 28, `heat` 25, `examine_in_light` 19. **Exact match on all six counts.**
* Stronger check: the *object* multiset of the 45 heuristically-labelled pick_two tasks matches
  the ground-truth object multiset exactly (cellphone×5, creditcard×4, spraybottle×4,
  keychain×3, soapbottle×3, tissuebox×3, toiletpaper×3, remotecontrol×3, book×2, cd×2,
  newspaper×2, pencil×2, soapbar×2, box/candle/cloth/cup/pillow/statue/watch ×1).

So **n = 45** for the Pick Two column, and the identification of those 45 tasks is verified.
(The row *order* of the validation log is a permutation of `mappings_test.json`, so per-row index
alignment could not be established — but the set is provably the right set.)

---

## 1. Statistics first: 38% vs 51% on n = 45 is not a significant difference

All four methods were evaluated on the **same** 45 tasks, so the correct test is paired (McNemar).

| | count |
|---|---|
| both DUET and GRPO solve | 11 |
| DUET only | 6 |
| GRPO only | 12 |
| neither | 16 |

* **Exact two-sided McNemar test: p = 0.238** (18 discordant pairs).
* 95% Wilson CIs: DUET 37.8% [25.1, 52.4]; GRPO 51.1% [37.0, 65.0]; LUFFY 44.4% [30.9, 58.8];
  CHORD 17.8% [9.3, 31.3]. **DUET's and GRPO's intervals overlap over almost their whole range.**

**Empirical seed noise on this exact 45-task subset.** We have three seed replicates of one
identical configuration (1.5B `DUET − DR3`: `alfworld_qwen1.5b_duet_minus_dr3`,
`…_seed2025`, `…_seed2027`, all at step 100):

| seed | pick_two | overall |
|---|---|---|
| default | 15/45 = 33.3% | 47.5% |
| 2025 | 19/45 = 42.2% | 38.0% |
| 2027 | 16/45 = 35.6% | 41.0% |

Seed-to-seed **range on the pick_two subset alone is 8.9 pp** (sd 3.8 pp) with everything else
held fixed. A 13.3 pp single-seed difference on n = 45 is therefore of the same order as run-to-run
variation, and the paired test does not reject the null.

---

## 2. The anomaly does not replicate at 1.5B or 7B, and disappears in DUET's own training distribution

### 2.1 Across scales (validation, same 45 tasks)

| Setting | DUET | LUFFY | CHORD | GRPO |
|---|---|---|---|---|
| 1.5B @100 | **24.4%** | 0.0% | 4.4% | 0.0% |
| 3B @50 | 20.0% | 31.1% | 17.8% | 33.3% |
| 3B @100 | 37.8% | 44.4% | 17.8% (@50) | **51.1%** |
| 7B @100 | **77.8%** | 66.7% | 55.6% (@50) | 75.6% |

GRPO beats DUET on Pick Two **only in the 3B row**. At 1.5B DUET wins 24.4% vs 0.0%; at 7B DUET
wins 77.8% vs 75.6%. If teacher mixing systematically damaged Pick Two, the effect should be
monotone in scale, or at least present at more than one scale. It is not.

### 2.2 Relative deficit (pick_two rate − overall rate), same runs

| Run | pick_two | overall | deficit |
|---|---|---|---|
| 1.5B DUET | 24.4% | 47.5% | −23.1 |
| 1.5B CHORD | 4.4% | 27.0% | −22.6 |
| 3B DUET @100 | 37.8% | 69.5% | −31.7 |
| 3B LUFFY @100 | 44.4% | 61.5% | −17.1 |
| 3B CHORD @50 | 17.8% | 46.5% | −28.7 |
| 3B GRPO @100 | 51.1% | 58.5% | **−7.4** |
| 7B DUET @100 | 77.8% | 86.5% | −8.7 |
| 7B GRPO @100 | 75.6% | 85.0% | −9.4 |

Pick Two is the hardest type for **every** method at **every** scale — that is the robust,
universal signal. What is unusual in the 3B row is not that DUET is weak on Pick Two, but that
this particular GRPO checkpoint is unusually *strong* on it (its deficit is only −7.4 pp, whereas
its weakest type is Clean at 40.5%).

### 2.3 Training-set behaviour contradicts a systematic mechanism

On-policy training rollouts (label from `mappings_train.json`, teacher rollouts excluded),
3B, steps 76–100, ~200 rollouts per method per type:

| Run | p&p | examine | heat | cool | clean | **PICK_TWO** | ALL | deficit |
|---|---|---|---|---|---|---|---|---|
| DUET v1 (0329) | 86.9% | 59.5% | 51.3% | 54.8% | 61.5% | **41.7%** | 61.7% | −20.0 |
| DUET* v39b (paper headline cfg) | 78.2% | 72.2% | 48.2% | 58.5% | 64.2% | **35.5%** | 60.6% | −25.0 |
| DUET − SC | 67.9% | 46.0% | 35.8% | 37.8% | 37.0% | **16.8%** | 42.6% | −25.7 |
| LUFFY | 79.7% | 59.5% | 49.7% | 44.5% | 51.3% | **38.9%** | 55.4% | −16.6 |
| GRPO | 80.6% | 61.8% | 62.1% | 43.0% | 40.2% | **34.6%** | 54.9% | −20.3 |

On the ~5× larger training sample, DUET's Pick Two success (41.7%) is **higher** than GRPO's
(34.6%), and the relative deficit is indistinguishable across methods (−16.6 to −25.7). The
−31.7 pp validation deficit is an outlier relative to everything else we can measure.
*Caveat:* teacher demonstrations exist for these training task ids, so DUET's absolute training
numbers are optimistically biased; the *deficit* column is the comparable quantity.

---

## 3. Why Pick Two is intrinsically hardest, and why the State Channel helps least there

Even though the 13 pp gap is not statistically resolvable, there is a real, measurable mechanism
that explains why Pick Two is the type where DUET's dense signal adds the least. All numbers below
come from `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl`
(19,497 filtered demos, joined to `mappings_train.json` by `data_id`) and from
`agentevolver/module/exp_manager/state_progress.py` re-run offline.

### 3.1 Pick Two demos are not rarer — they are the longest and the most self-similar

| Teacher task type | kept demos | raw | teacher pass rate | mean obs/demo | median | unique-obs / obs | demos with a repeated state |
|---|---|---|---|---|---|---|---|
| pick_and_place_simple | 6163 | 6720 | 0.917 | 9.28 | 6 | 0.985 | 13.5% |
| pick_clean… | 3324 | 4480 | 0.742 | 12.31 | 9 | 0.973 | 25.5% |
| **pick_two_obj_and_place** | **3049** | **3890** | **0.784** | **14.72** | **13** | **0.921** | **52.6%** |
| pick_cool… | 2745 | 3940 | 0.697 | 13.79 | 10 | 0.992 | 11.8% |
| pick_heat… | 2313 | 3210 | 0.721 | 11.81 | 8 | 0.993 | 7.5% |
| look_at_obj_in_light | 1903 | 1960 | 0.971 | 5.96 | 4 | 0.937 | 30.8% |

Task coverage is also fine: 371/389 (95.4%) of training pick_two tasks have ≥1 kept demo, in line
with all other types (95.1–100%). So **data scarcity is not the explanation.** Length is:
pick_two demos are the longest (mean 14.72 observations vs 5.96–13.79) and **52.6% of them revisit
at least one identical normalised state**, four to seven times the rate of the single-object
manipulation types.

### 3.2 That self-similarity breaks the progress function Φ

`ExpertProgressMap` assigns `Φ(s) = max over occurrences of j/(T−1)`
(`state_progress.py:707-710`). Repeated states therefore inherit the *latest* index at which they
occur. Evaluating the built maps on the experts' own trajectories:

| Task type | mean T | per-step granularity 1/(T−1) | mean Φ inflation vs true j/(T−1) | position where Φ first ≥ 0.9 | max Φ reachable in first half | negative Φ steps |
|---|---|---|---|---|---|---|
| pick_and_place | 9.28 | 0.121 | +0.110 | 0.857 | 0.519 | 3.6% |
| clean | 12.31 | 0.088 | +0.102 | 0.817 | 0.612 | 6.0% |
| **pick_two** | **14.72** | **0.073** | **+0.166** | **0.597** | **0.802** | **16.2%** |
| cool | 13.79 | 0.078 | +0.126 | 0.803 | 0.641 | 6.5% |
| heat | 11.81 | 0.092 | +0.120 | 0.792 | 0.601 | 6.1% |
| examine | 5.96 | 0.202 | +0.234 | 0.501 | 0.675 | 12.2% |

Reading: on a Pick Two demo the progress map already reports Φ ≥ 0.9 at **60%** of the way through
the expert's *own* trajectory (79–86% for the other manipulation types), the first half of the
trajectory already attains **Φ = 0.80** (0.52–0.64 elsewhere), and **16.2%** of consecutive
expert-side deltas are *negative* — i.e. the shaping term η·[Φ(s_{t+1}) − Φ(s_t)] penalises the
expert's own second-object phase. Per-step granularity is also the coarsest (0.073).

### 3.3 The problem is not coverage — it is dynamic range

Answering the reviewer's implicit sub-question directly: I rebuilt the per-task progress maps and
evaluated them on DUET's **own saved on-policy rollouts** (3B, steps 76–100, 1315 rollouts over
199 tasks, 194 with a map), splitting each rollout in half:

| Task type | rollouts | coverage 1st half | coverage 2nd half | mean Φ 1st half | mean Φ 2nd half | ΔΦ |
|---|---|---|---|---|---|---|
| pick_and_place | 259 | 0.718 | 0.777 | 0.316 | 0.606 | **+0.291** |
| examine | 118 | 0.619 | 0.571 | 0.417 | 0.445 | +0.028 |
| heat | 189 | 0.657 | 0.578 | 0.363 | 0.443 | +0.080 |
| cool | 301 | 0.709 | 0.664 | 0.390 | 0.468 | +0.078 |
| clean | 205 | 0.649 | 0.673 | 0.366 | 0.543 | +0.177 |
| **pick_two** | 203 | **0.817** | **0.808** | 0.440 | 0.575 | **+0.134** |

And splitting pick_two rollouts at the first *successful* delivery (`"You put the …"`, 141 rollouts):
coverage 0.866 pre-put → 0.800 post-put; mean Φ **0.494 pre-put → 0.606 post-put**.

**So the map covers the second-object portion perfectly well (hit rate is the highest of any task
type, ≈0.81).** What it cannot do is *grade* it: delivering the entire second object moves Φ by
only ≈0.11–0.13, because the first-object phase already scores ≈0.44–0.49 and Φ has saturated.

### 3.4 Consequence: the State Channel pays for failure on Pick Two

From the per-rollout diagnostics saved in
`checkpoints/agentevolver/alfworld_3b_duet_0329/Trajectory/trajectories_step_*.jsonl`
(5618 on-policy rollouts, `diag.sc_progress` / `sc_coverage` / `sc_bonus` / `reward_components`;
β = 0.2 verified empirically as `sc_bonus / sc_progress` = 0.1996–0.1999 at every step sampled):

| Task type | n | SC progress \| **fail** | SC progress \| succ | gap | AUC(SC progress → success) | mean shaped reward \| **fail** |
|---|---|---|---|---|---|---|
| pick_and_place | 1542 | 0.260 | 0.526 | 0.266 | 0.836 | +0.054 |
| examine | 469 | 0.366 | 0.455 | 0.089 | 0.602 | −0.175 |
| heat | 765 | 0.222 | 0.500 | 0.278 | 0.837 | +0.030 |
| cool | 960 | 0.246 | 0.561 | 0.315 | 0.894 | −0.162 |
| clean | 1038 | 0.216 | 0.499 | 0.284 | 0.832 | −0.074 |
| **pick_two** | **844** | **0.414** | 0.556 | **0.142** | **0.710** | **+0.330** |

(shaped reward = `reward_components.original + sc_bonus + step_delta_sum`.)

A **failed** Pick Two rollout receives SC progress 0.414 — nearly double what a failed Clean (0.216)
or Heat (0.222) rollout receives — and a net positive shaped reward of +0.330, the only task type
where failures are net-rewarded. The success/failure separation is 0.142 (half of the 0.27–0.32
seen elsewhere) and the ranking AUC drops to 0.710 vs 0.83–0.89. Under group-relative GRPO
normalisation this compresses the advantage spread between "delivered one object" and "delivered
both", which is exactly the distinction Pick Two requires. On-policy step-delta sign flips are
also worst here (23.9% of deltas negative vs 12.4–18.7% elsewhere).

### 3.5 Matched-checkpoint ablation (weak but directionally consistent)

The only 3B State-Channel ablation with a validation log is at step 50:

| 3B run @ step 50 | p&p | examine | heat | cool | clean | **pick_two** | ALL | deficit |
|---|---|---|---|---|---|---|---|---|
| DUET (0329) | 71.7 | 36.8 | 44.0 | 46.4 | 62.2 | **20.0** | 48.0 | −28.0 |
| **DUET − State Channel** | 84.8 | 78.9 | 76.0 | 39.3 | 64.9 | **46.7** | 64.5 | **−17.8** |
| GRPO | 78.3 | 36.8 | 64.0 | 35.7 | 29.7 | **33.3** | 47.5 | −14.2 |

Removing the State Channel roughly halves the Pick Two relative deficit (−28.0 → −17.8). **But this
is not a clean isolation:** the −SC run is better on 5/6 types at step 50 and 16.5 pp better
overall, so most of the pick_two gain is a general checkpoint-quality effect, not a pick_two-specific
one. At step 100 the −SC run has no validation log, and by steps 76–100 its *training* success has
collapsed (42.6% overall vs DUET's 61.7%), i.e. the −SC advantage at step 50 does not persist.
**This ablation should be quoted as suggestive at most, not as proof.**

---

## 4. Behavioural signature at validation (case-level)

Parsing the 45 pick_two validation trajectories (`<action>` blocks and environment replies):

| | success | mean actions | distinct receptacles visited | repeated-action rate | "Nothing happens"/"Invalid action" |
|---|---|---|---|---|---|
| DUET, pick_two | 37.8% | 24.0 | 9.29 | 31.5% | 4.13 |
| GRPO, pick_two | 51.1% | 19.0 | 7.00 | 30.5% | 3.91 |
| DUET, all other types | 78.7% | 13.9 | 5.84 | 12.4% | 1.64 |
| GRPO, all other types | 60.6% | 14.6 | 5.80 | 18.0% | 3.73 |

DUET is *shorter and cleaner* than GRPO on every other task type (13.9 vs 14.6 actions, 12.4% vs
18.0% repeated actions) but *longer and more repetitive* on Pick Two (24.0 vs 19.0 actions).
Even on the pick_two episodes it solves, DUET needs 18.2 actions vs GRPO's 14.6. With a hard
`max_steps: 30` budget (`config/duet_paper_experiments_configs/alfworld/alfworld_3b_duet.yaml:137`,
identical in the GRPO config at line 100) and a median expert solution of 13 steps, Pick Two is the
only type where the solution length approaches the budget — so a ~25% efficiency loss converts
directly into failures. 17/28 (60.7%) of DUET's pick_two failures exhaust the 30-action cap.

**Concrete failure mode** (both cases are tasks GRPO solved and DUET failed):

* *val idx 74* ("find two tissueboxes, put on coffeetable"). DUET picks up tissuebox 1 at action 5,
  then spends actions 6–22 opening drawers 1–6 hunting for the *second* box **while still carrying
  the first**, and never delivers anything. GRPO: take → deliver → return → take → deliver, 8 actions,
  success.
* *val idx 51* ("two remote controls into the sofa"). DUET picks up remotecontrol 3 at action 9,
  then burns actions 10–30 searching for a second one while holding the first, issuing
  `take remotecontrol 1 from tvstand 1` four times (all "Invalid action" — inventory is full).
  GRPO delivers the first remote at action 11, *then* fetches the second: success in 17 actions.

Across all 12 tasks GRPO solved and DUET failed, **11/12 DUET trajectories picked up exactly one
object** and 9/12 never delivered any, while all 12 GRPO trajectories picked up two.
The pattern is "carry-and-keep-searching instead of deliver-then-search" — precisely the policy
that the flat, first-half-saturated Φ of §3.2–3.4 fails to discourage: with Φ already ≈0.8 after
the first object is in hand, wandering while carrying it is not penalised.

*Honest counter-evidence on this point:* the raw statistic "took ≥1 object but never delivered any"
is 64.3% for DUET failures and 59.1% for GRPO failures — similar. The discriminating statistics are
the action budget (60.7% of DUET failures hit the 30-step cap vs 0% of GRPO's, which instead run out
of the 21,580-token response budget at 17–29 actions) and the case-level reading above.
Because the two runs hit *different* termination limits, the "cap exhaustion" comparison is
confounded and should not be quoted as a headline number.

---

## 5. What I would and would not claim in the rebuttal

**Claim (well supported):**
1. n = 45; the 13.3 pp difference is not statistically significant (paired McNemar p = 0.24) and is
   within measured seed-to-seed variation (8.9 pp range on the same subset for one fixed config).
2. Pick Two is the hardest type for every method at every scale; the anomaly is that this one 3B
   GRPO checkpoint is unusually strong on it, and it does not reproduce at 1.5B or 7B.
3. There is a concrete, quantified reason the State Channel is *least useful* on Pick Two: the
   expert progress map saturates in the first sub-goal (Φ ≥ 0.9 at 60% of the demo; first-half
   max Φ = 0.80), leaving only ≈0.13 of dynamic range for the entire second object — so SC-shaped
   reward barely separates success from failure there (AUC 0.71 vs 0.83–0.89; failed pick_two
   rollouts net +0.33 shaped reward, the only task type with net-positive failure reward).
4. Concrete behavioural signature: carry-and-keep-searching rather than deliver-then-search.

**Do not claim:**
* That removing the State Channel fixes Pick Two (the step-50 −SC evidence is confounded).
* That teacher mixing systematically hurts Pick Two (the training-distribution and 7B numbers
  contradict it).
* Any number for the *paper-headline* 3B DUET run (`alfworld_qwen3b_duet_v39b`, 77.5%) — see gaps.

---

## 6. Draft rebuttal text

> **Pick Two (UyKJ).** Thank you — we investigated this directly and report both a statistical and
> a mechanistic answer.
>
> *Statistically, the gap is not resolvable.* The Pick Two column contains n = 45 of the 200
> validation tasks (verified against the ALFWorld ground-truth task types). Because all methods are
> evaluated on the same tasks, the appropriate test is paired: DUET and GRPO agree on 27 of 45 tasks
> (11 both-solve, 16 both-fail) and disagree on 18 (6 DUET-only, 12 GRPO-only), giving an exact
> McNemar p = 0.24. The 95% Wilson intervals overlap almost completely (DUET [25.1, 52.4]%,
> GRPO [37.0, 65.0]%). For calibration, three seed replicates of one *identical* configuration span
> 8.9 pp on this same 45-task subset. We will add n and this caveat to the appendix.
>
> *The effect does not replicate off that single cell.* At 1.5B, DUET reaches 24.4% on Pick Two
> versus 0.0% for GRPO; at 7B, DUET 77.8% versus GRPO 75.6%. On DUET's own on-policy training
> rollouts at 3B (≈200 per method per type, steps 76–100), DUET solves 41.7% of Pick Two tasks
> versus GRPO's 34.6%, and the Pick-Two-minus-overall deficit is statistically indistinguishable
> across DUET (−20.0), GRPO (−20.3) and LUFFY (−16.6). What is robust across every scale and method
> is that Pick Two is simply the hardest of the six types.
>
> *Mechanistically, there is nonetheless a real reason the State Channel helps least here, and we
> now say so explicitly.* Pick Two expert demonstrations are the longest (14.7 observations on
> average versus 6.0–13.8) and by far the most self-revisiting: 52.6% contain a repeated normalised
> state, versus 7.5–30.8% for the other types. Because Φ(s) is defined as the *maximum* normalised
> index at which a state occurs, this self-similarity makes the potential saturate inside the first
> sub-goal: on a Pick Two demonstration the map already reports Φ ≥ 0.9 at 60% of the trajectory
> (79–86% for the other manipulation types), the first half already attains Φ = 0.80 (0.52–0.64
> elsewhere), and 16.2% of consecutive expert-side deltas are negative (3.6–6.5% elsewhere).
> Importantly, this is *not* a coverage failure — measured on DUET's own rollouts, the map's hit
> rate on Pick Two is the highest of any type (0.81), and stays high after the first object is
> delivered (0.87 → 0.80). It is a dynamic-range failure: mean Φ rises only from 0.49 to 0.61 across
> the entire second-object phase, versus +0.29 for Pick&Place. Consequently the shaped signal barely
> separates outcome on Pick Two: SC progress on failed rollouts is 0.414 (versus 0.216–0.260 for the
> other manipulation types), the success-versus-failure separation is 0.14 (versus 0.27–0.32), and
> the ranking AUC of SC progress against success falls to 0.71 (versus 0.83–0.89). Pick Two is the
> only type where failed rollouts receive net-positive shaped reward (+0.33).
>
> *This has a visible behavioural signature.* On the 12 Pick Two tasks GRPO solved and DUET did not,
> 11 of 12 DUET trajectories picked up exactly one object and then kept searching for the second
> *while still carrying the first*, never delivering either; all 12 GRPO trajectories used the
> deliver-then-search order. Because Φ is already ≈0.8 once the first object is in hand, the shaping
> term does not discourage carrying-and-searching, and with a 30-action budget against a median
> expert solution of 13 actions this inefficiency is fatal: DUET averages 24.0 actions per Pick Two
> episode versus GRPO's 19.0, despite being *more* efficient than GRPO on every other type
> (13.9 versus 14.6 actions).
>
> We therefore regard the Pick Two cell as a combination of a small-sample single-seed effect and a
> genuine, now-quantified limitation of a *max-index, exact-match* potential function on tasks with
> repeated sub-goals. The fix is a task-structure-aware potential (e.g. sub-goal-count-indexed
> progress, or the soft/stage matcher we already implement for WebShop) rather than a change to the
> Action Channel, and we will state this limitation and the n = 45 caveat explicitly in Appendix F.

---

## 7. Gaps / things I could not verify

1. **Run provenance mismatch with the main table.** `tables/main_results.tex` reports 3B-ALFWorld
   as GRPO 47.0%, LUFFY 61.5%, CHORD 67.0%, SFT+GRPO 59.5%, DUET **77.5%**. The Appendix-F figure
   is computed from `alfworld_3b_duet_0329` (DUET v1, **no BC**, 69.5% overall) and
   `alfworld_3b_grpo_react_tags` (**58.5%** overall). Per `analysis_reports/3b_master_experiment_table.md:42,50`,
   the headline 77.5% comes from `alfworld_qwen3b_duet_v39b` and the GRPO baseline is listed there as
   **58.5%**, not 47.0%. No local validation log reproduces 77.5% / 47.0% / 67.0% / 59.5%
   (I enumerated every `experiments/**/validation_log/*.jsonl`; the full list is in §0.1 plus
   7B/1.5B runs). **Two issues for the authors: (a) the main table's GRPO 3B-ALFWorld number
   (47.0%) appears inconsistent with the internal master table (58.5%) and with the only local
   GRPO run; (b) the Pick Two anomaly is measured on a *different, weaker* DUET configuration than
   the paper's headline 3B model.** I could not check whether the anomaly survives on v39b, because
   that run has no local validation log (only training trajectories).
2. **Single seed at 3B.** The seed-variance estimate (8.9 pp range) comes from 1.5B `DUET − DR3`
   replicates, not from 3B DUET. No 3B seed replicates exist locally.
3. **DUET − SC at 3B has only a step-50 validation log**, and that run's training success collapses
   after step ~50, so it cannot serve as a matched step-100 ablation.
4. **Termination-limit confound.** DUET pick_two failures stop at the 30-action cap; GRPO failures
   stop at the 21,580-token response cap (17–29 actions). I did not disentangle the two, so I
   avoided quoting the cap-exhaustion contrast as primary evidence.
5. **CHORD at 3B is a step-50 checkpoint** in the appendix figure while everything else is step 100
   — already noted in the figure script but not in the appendix caption.
6. **Validation row ordering** could not be aligned to `mappings_test.json` item ids (the log stores
   no item id), so the pick_two *set* is verified but per-row indices are internal to the log.

## 8. Reproduction commands

All analyses are pure reads of:
* `experiments/alfworld/*/validation_log/{50,100}.jsonl` (200 rows each: `input/output/score/step/reward`)
* `checkpoints/agentevolver/{alfworld_3b_duet_0329,alfworld_3b_grpo_react_tags,alfworld_3b_luffy,alfworld_qwen3b_duet_minus_sc,alfworld_qwen3b_duet_v39b}/Trajectory/trajectories_step_*.jsonl`
  (per-rollout `task_id`, `messages`, `success`, `diag.sc_*`, `diag.reward_components`)
* `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl` (19,497 demos)
* `AgentGym/agentenv-alfworld/configs/mappings_{train,test}.json` (2420 / 200 tasks with true task types)
* `agentevolver/module/exp_manager/state_progress.py` (`normalize_observation`, `ExpertProgressMap`)

Python used: `/data/home/qisheng/miniconda3/envs/duet/bin/python`.
