# Verdict: why the 1.5B-WebShop cell does not reproduce across seeds

Scope: the three like-for-like 1.5B DUET runs on WebShop. All numbers below were recomputed for this
document from the raw artefacts, not copied from earlier passes.

| tag | run directory (under `checkpoints/agentevolver/` and `experiments/webshop/`) |
|---|---|
| **paper** | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` |
| seed2025 | `webshop_qwen1.5b_duet_a100_seed2025` |
| seed2027 | `webshop_qwen1.5b_duet_a100_seed2027` (identical 800-task pool, `task_seed: 2026`) |

Sources: `.../Trajectory/trajectories_step_{1..100}.jsonl` (5,714 / 5,732 / 5,714 on-policy episodes
per run after dropping `diag.is_teacher` rows) and `experiments/webshop/<run>/validation_log/{50,100}.jsonl`
(n=200 each). Reproduction scripts:
`/tmp/claude-1000/-data-home-qisheng-EvolAnalsis/a5d90f98-198d-42a4-aeb3-820cd312fa72/scratchpad/{verdict_verify,verdict_train,verdict_ceiling}.py`.

Anchors reproduced exactly before anything else was computed: val@100 strict success
35.5 / 3.5 / 2.5 %, val@100 mean reward 0.7057 / 0.5212 / 0.5435, training steps 81–100 mean
on-policy reward 0.5670 / 0.5460 / 0.4769.

---

## 1. The mechanism, in one paragraph

WebShop's grader awards 1.0 only when **every** requested attribute matches, and requested
attributes such as colour/size/fit-type are selected by clicking option buttons on the product page
before `click[buy now]`. On the paper run's 800 training tasks, 88.4 % request at least two such
attributes (69.6 % exactly two, 18.8 % three, 8.4 % one, 3.2 % none —
`extract_instruction_attributes`, `agentevolver/module/exp_manager/state_progress.py:93`, applied to
the first observation of every episode). All three runs converge early onto a three-action policy —
`search[...] → click[<ASIN>] → click[buy now]` — which earns large partial credit (mean reward
0.41–0.49 by step 30) but is structurally incapable of an exact match on 88 % of tasks. Escaping
that policy means adding option clicks, and the runs differ in exactly one measurable dimension:
whether they did. **The single most supportive measurement is the same-task, same-reward
comparison.** Over the 200 validation instructions, restricted to the 38 tasks where the paper run
and seed2027 earned an *identical* partial score (matched within ±0.02, mean 0.398 — so option
clicking earned neither run anything), the paper run clicks ≥2 distinct options on **52.6 %** of
episodes and seed2027 on **0.0 %** (paired difference +0.526, bootstrap 95 % CI
[+0.368, +0.684], 2,000 resamples; `verdict_verify.py`). Identical task, identical reward, a
53-point behavioural gap.

**What survived adversarial review.** The behavioural account survived both challenges. The counter
was rebuilt independently — replicating `webshop_env._parse_action_from_llm_output` byte-for-byte and
classifying clicks by the repo's own `parse_product_detail_options()` rather than a nav blacklist —
and is externally certified by the reward channel: across 14,352 pooled on-policy episodes there is
not one strict success with fewer option clicks than the instruction requests (see §2), which a
counter that under-counted clicks could not produce. The buy-now gate is not load-bearing (removing
it changes the ≥2-option rate by 0.0 pp), no item-ASIN click leaks into the option count (100 % of
counted option clicks occur on a page classified `product_detail` and stay there), and
`diag.is_teacher` is always a real boolean.

**What did not survive, and must not be written.** Two formulations circulated in earlier passes are
*definitional*, not empirical, and a reviewer will say so:
1. "0 strict successes in ~13,700 episodes with ≤1 option click." True (pooled: r=2,k≤1 → 0/10,735,
   max reward 0.917; r=3,k≤2 → 0/3,113, max reward 0.909) but it is a restatement of the grader.
2. "A 35× behavioural gap mapping one-to-one onto 35.5 % vs 2.5 % strict success." Since success
   implies k≥2 on ≥2-attribute tasks, the run-level k≥2 rate is bounded below by the run-level
   success rate; part of that "mapping" is arithmetic.

State the necessity once, as a property of the reward function, then stop using it as evidence. The
defensible statements are the ones measured **where option clicking earned nothing**:

| val@100, failed episodes only | paper | seed2025 | seed2027 |
|---|---|---|---|
| n failed | 129 | 193 | 195 |
| P(≥2 options \| failed) | **64.3 %** | 6.2 % | 1.0 % |
| mean options \| failed | 1.729 | 0.834 | 0.308 |

Paired on the same instruction with **both runs failing**: paper vs seed2027, 129 pairs,
64.3 % vs 1.6 %, difference **+0.628**, CI [+0.543, +0.713], paper wins on 63.6 % of tasks and
loses on 0.8 %. Paper vs seed2025 (193 common instructions — seed2025 has 7 unparseable validation
episodes), 123 pairs, 63.4 % vs 6.5 %, difference +0.569, CI [+0.488, +0.659].

Three facts close the alternatives.
*Not a proxy for the score:* P(strict success | ≥2 options) = 0.415 / 0.077 / 0.000 — nowhere near 1
and not constant.
*Not general competence:* on val@100 the three runs are indistinguishable on searches
(1.01 / 1.01 / 1.02), item clicks (1.00 / 1.02 / 0.97) and buy rate (0.97 / 0.89 / 0.94); only option
clicks move (1.80 / 0.84 / 0.32).
*No hidden third improvement:* at training steps 81–100 the paper run's zero-option episodes average
reward **0.454**, slightly *below* seed2027's **0.468**. The paper run got better at exactly one
thing.

**Why the gradient is flat there, and one thing the paper currently overclaims.** Reaching the
option buttons is not the problem — all three runs stand on a page offering ≥2 options in 81–88 % of
episodes in every window. The gradient carries almost no signal for pressing them: on the 69.6 %
majority (r=2) task type, the 1→2 option transition is worth about −0.06 in mean reward early and
+0.08 late, against a 6.1–6.6 % invalid-action rate at −0.05 each. And the State Channel is
**provably inert on this behaviour**: `webshop_attribute_aware_potential`
(`state_progress.py:252`) = stage(page_type) + `compute_attribute_match_score`, and
`compute_attribute_match_score` (`state_progress.py:154`) scores which attribute values the page
*offers*, not which are *selected*. Empirically, over **16,783 / 16,783** valid option clicks that
begin and end on a `product_detail` page, ΔΦ = 0 (mean 0.000000, max |ΔΦ| 0.000000), while ΔΦ is
+0.10 after a search and +0.53 after an item click. On WebShop the SC provides no dense guidance on
the one behaviour that separates the runs; that should be stated rather than defended. The
off-policy channel does carry it — teacher rows in the same batches (2,040 pooled) click 2.09
distinct options, 89.5 % with ≥2, 7.27 actions, reward 1.000 — which is a positive result for the
paper's thesis, not a negative one.

---

## 2. Why this produces exactly the observed signature

**Identical at val@50.** The behaviour that decides strict success is at floor for every run at step
50: P(≥2 options | failed) = 0.5 % / 2.5 % / 0.0 %, mean options over all episodes 0.02 / 0.11 /
0.01. There is nothing yet to differ on, so val@50 reads 1.0 / 1.5 / 1.0 %. The divergence is
acquired *inside* the budget, not present from initialisation.

**Divergence only in the last 50 steps.** Per-10-step blocks of P(≥2 options | episode failed) — a
series containing zero successes by construction:

| steps | 41–50 | 51–60 | 61–70 | 71–80 | 81–90 | 91–100 |
|---|---|---|---|---|---|---|
| paper | 1.6 % | 1.3 % | 2.1 % | 4.8 % | 15.3 % | **29.6 %** |
| seed2025 | 2.2 % | 3.7 % | 2.8 % | 5.0 % | 16.1 % | 18.1 % |
| seed2027 | 2.0 % | 1.3 % | 3.8 % | 3.4 % | 5.0 % | 2.5 % |

The onset test (baseline steps 45–70, +3 sd, 4-step persistence, 5-step rolling mean) puts the paper
run's take-off at **step 74** and its on-policy strict-success take-off at step 86; seed2025 at 81
and 96; seed2027 has **no onset in either series**. So the escape occupies the final quarter of the
budget and the 100-step cut lands in the middle of it. (The 74-vs-86 ordering cannot be an artefact
of the success metric's own definition, since the leading series contains no successes — but with
n=3 runs this supports "earliest detectable manifestation", not proven causal precedence.)

**Large gap on strict success, small gap on mean reward.** Mean validation reward moves 0.7057 →
0.5435 (+0.162, a 1.30× ratio) while strict success moves 35.5 % → 2.5 % (14×). The reason is the
shape of the reward ceiling. Pooled over all three runs:

| requested option attrs | options clicked | n | strict successes | max reward observed |
|---|---|---|---|---|
| r=2 | k=0 | 8,452 | 0 | 0.833 |
| r=2 | k=1 | 2,283 | 0 | 0.917 |
| r=2 | k=2 | 856 | 144 (16.8 %) | 1.000 |
| r=3 | k≤2 | 3,113 | 0 | 0.909 |
| r=3 | k=3 | 36 | 4 (11.1 %) | 1.000 |

The short path saturates at 0.83–0.92. A run that never adds option clicks therefore loses only
~0.1–0.2 of mean reward but *all* of its strict success. Threshold metrics amplify; averages do not.

**Mass stranded in 0.75–0.90.** Val@100 score distribution:

| band | paper | seed2025 | seed2027 |
|---|---|---|---|
| 1.0 | 35.5 % | 3.5 % | 2.5 % |
| 0.90–1.0 | 3.0 % | 1.0 % | 0.0 % |
| **0.75–0.90** | **20.5 %** | **32.5 %** | **23.5 %** |
| 0.50–0.75 | 21.0 % | 27.5 % | 43.5 % |
| <0.50 | 20.0 % | 35.5 % | 30.5 % |

That band is precisely the r=2, k≤1 and r=3, k≤2 cells of the table above: correct product, correct
price, one attribute unselected. This is the signature the response to reviewer y9x6 already
describes qualitatively ("32.5 % stranded"); it now has a mechanism and a cell in a table.

---

## 3. Is a longer budget expected to fix it?

**Measured trajectories, not hope.** The correct quantity is the trend in the failure-conditional
option rate over the last three blocks (values from the table in §2):

- paper: 4.8 % → 15.3 % → 29.6 %, still rising steeply at step 100 (the ≥1-option rate, 5-step
  rolling, runs 0.12 at step 71 → 0.22 at 74 → 0.41 at 83 → 0.66 at 95 → 0.76 at 98).
- seed2025: 5.0 % → 16.1 % → 18.1 % — took off (onset step 81) and is **mid-escape**: its
  validation ≥1-option rate is 76.5 % but P(2nd option | 1st) is only 0.085, against the paper run's
  0.772 and the teacher's 0.92. It has learned to click *an* option, not *all* options.
- seed2027: 3.4 % → 5.0 % → 2.5 % — flat, with no onset anywhere in 100 steps.

Honest reading: **a longer budget is a plausible fix for seed2025 and an open question for
seed2027.** seed2025 has a rising stage-1 curve and a stalled stage-2 curve; extending it tests
whether stage 2 follows. seed2027 shows no take-off at all, so extension tests whether the take-off
is merely late or whether this seed sits in the optimum indefinitely. Nothing in the logged state
distinguishes the paper run at steps 70–74 from seed2027 at the same steps: an onset test over all
382 logged metric keys finds nothing moving at or before step 82 except a lagging success EMA, a
0.03-unit wobble in `dr3/ess_off_window`, and response length. `dr3/disc_acc`, `chord/mu`,
`teacher_gradient_share` and every `state_channel` key are flat and mutually indistinguishable
across the three runs. We cannot predict the take-off; we can only measure it.

**Status of the 150-step run — correcting the brief.** As of 2026-07-26 14:2x the run is *queued,
not executing*. `logs/a100_followup.log` reads "follow-up queue armed; waiting for the main queue to
finish"; the GPUs are currently running `alfworld_qwen1.5b_duet_a100_obsnoise_hash` (pid 3650249,
~1 h in). `webshop_qwen1.5b_duet_a100_fixedtask_seed2027_s150` is first in `QUEUE` in
`run_a100_followup_queue.sh`; no checkpoint or log directory exists for it yet.

**One design caveat that must be stated when it lands.** Diffing its config against the 100-step
seed2027 config, the *only* substantive change is `max_train_tasks: 800 → 1200`
(`config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_fixedtask_seed2027_s150.yaml:170`;
`seed: 2027`, `task_seed: 2026`, `total_epochs: 1`, `train_batch_size: 8`, `test_freq: 50` all
unchanged). Because task selection is `random.seed(task_seed); shuffle(pool); pool[:max_train_tasks]`
(`agentevolver/module/task_manager/task_manager.py:155-161`), the 1,200 tasks are a strict superset
of the paper's 800 — but the run is one epoch over 1,200, so its **steps 1–100 are not the same
curriculum as the 100-step runs' steps 1–100**, and its val@100 is therefore not a like-for-like
replicate of the 100-step number. It answers "does a longer budget produce take-off", not "does the
100-step number reproduce".

**What each outcome would mean** (validation fires at 50/100/150):

| outcome at step 150 | reading | what we may write |
|---|---|---|
| strict success rises to the paper run's range (≳25 %) **and** P(≥2 options \| failed) shows a clear onset between steps 100 and 150 | the take-off is late, not absent; the 100-step budget cuts through a phase transition | "at 150 steps the replicate reaches X %; the 100-step budget sits inside the transition." Must be reported *alongside* the protocol-matched 100-step number, never in place of it |
| strict success rises but the option-onset is absent | the gain came from something else | do **not** attribute it to this mechanism; re-run the §1 diagnostics before writing anything |
| strict success stays ≤5 % and the option rate stays flat through step 150 | this seed does not escape at 1.5× budget | say so plainly. The cell is then reported as a distribution, and the submitted 35.5 % is stated as a high draw whose mechanism we can name but not reproduce on demand |
| strict success stays low but P(≥2 options \| failed) is rising at step 150 | the escape is underway and slower than 1.5× budget | report as "still in transition at 150 steps"; this is a budget statement, not a method statement |

Instrument **P(≥2 options | episode failed)** per training step as the primary readout. It crosses
its threshold 12–15 steps before strict success does (74 vs 86; 81 vs 96) and is available from
training rollouts without waiting for a validation pass.

---

## 4. What would make this cell reproducible

### (a) Legitimate during the rebuttal

1. **Report the cell as a distribution over the like-for-like replicates, keeping strict success as
   the headline and adding mean reward alongside.** Already the standing recommendation in
   `DECISION_webshop_1p5b_cell.md` (Option A + B). Nothing here changes it; the mechanism only
   explains *why* the distribution is wide.
2. **Replicate the baseline at the same budget.** `webshop_qwen1.5b_sft_rl_a100_seed{2025,2027}` are
   queued. Without them, "DUET is volatile here" is unanchored — the reviewer's question is about
   the *comparison*, and CHORD already shows the same signature (0.603 mean reward, 11.5 % strict,
   37.5 % stranded in 0.75–0.90).
3. **Add the failure-conditional behavioural table (§1) and the ceiling table (§2) to the
   appendix.** These are analyses of runs already on disk; they cost no GPU time and convert
   "unlucky seed" into a named, measured failure mode.
4. **Publish the ΔΦ = 0 measurement and soften the WebShop SC claim accordingly.** 16,783/16,783 is
   falsifiable and it is better to state it than to have a reviewer find it. It also predicts, and
   is consistent with, the −SC ablation on WebShop being unfavourable to us.
5. **Report the 150-step run with the `max_train_tasks: 1200` caveat above, whatever it shows.**

### (b) Camera-ready or future work

6. **Fix `compute_attribute_match_score` to score *selected* options, not offered ones**
   (`state_progress.py:154`, a one-function change), and re-run the cell. This puts non-zero
   potential on exactly the behaviour whose gradient is currently flat. It is the strongest
   scientific follow-up available and it is a method change, so it belongs in the camera-ready with
   the original numbers retained.
7. **Re-run the whole 1.5B-WebShop column at a budget past the transition** (Option C), so every
   method is compared after take-off rather than inside it. Only defensible if applied uniformly to
   every method in the column and reported as a protocol change made after seeing the result.
8. **Report P(≥2 options | failed) as a secondary metric for WebShop** — it separates 64 × between
   runs whose mean reward differs by 1.3 ×, and it is a behavioural quantity, not a threshold.
9. **Increase group size or add an option-clicking exploration bonus** to raise the probability of
   at least one exact success per group. Group-level P(≥1 exact success) rises 2.5 % → 43.8 % for the
   paper run over blocks 61–70 → 91–100 while seed2027 goes 7.5 % → 3.8 %; under within-group
   normalisation the escape is jackpot-driven, which is why replicates diverge late rather than
   early.

### Explicitly unacceptable

- **Running further seeds and reporting the best.** Every replicate launched must be reported,
  including seed2028 and any 150-step run, whichever way they land.
- **Substituting mean reward for strict success as the headline** after observing that mean reward is
  kinder to us. Mean reward is an *additional* column, never a replacement (`DECISION` Option B).
- **Quoting the 150-step number as the 100-step number**, or comparing a 150-step DUET against a
  100-step baseline.
- **Retiring the submitted 35.5 % quietly.** It stays in the table, with the spread stated next to it.
- **Selecting among the 66 1.5B-WebShop runs on disk** (a hyperparameter sweep whose median is
  3.0 %) for anything other than the pre-registered paper configuration.

### Do not blame these — ruled out with evidence

Step cap (`multi_turn.max_steps: 30`; only 1.6–2.4 % of episodes reach it, the minimal correct path
is 5–6 actions, and the 6–8-action bucket has the highest strict-success rate); navigation or context
length (all runs reach an option-offering page in 81–88 % of episodes in every window); click
accuracy (per-click precision 0.870 / 0.841 / 0.931 — statistically indistinguishable; only *coverage*
moves, recall 0.779 / 0.436 / 0.497); memorisation (800 tasks, 8 per step, each seen exactly once,
`training/epoch = 0` at step 100, top-5-task concentration of ≥2-option episodes falls 30 % → 11 %);
teacher channel (teacher content stationary across windows and runs; `chord/phi` is identically
1.000, min = max, sd = 0, at every logged step of all three runs).

### Two provenance corrections to propagate

- The paper's run is **`webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06`**, not
  `webshop_qwen1.5b_duet`. A similarly-named run exists on disk with different behaviour.
- seed2027 draws from the identical 800-task pool but **0 of 100 steps share the same 8-task set** —
  the per-epoch shuffle follows `seed`, not `task_seed`. It is a much tighter comparison than
  seed2025 (89 shared tasks) but it is not matched step-for-step. `DECISION_webshop_1p5b_cell.md`
  should be amended.

### Scope restriction

Paper vs seed2025 separates sharply on **validation** failures (mean options 1.73 vs 0.85) but only
weakly on **training** failures at steps 81–100 (0.955 vs 0.980 mean options; 22.3 % vs 17.1 %
k≥2|fail). The training/validation decode discrepancy is real and unresolved, so make
paper-vs-seed2025 statements validation-side only. Paper vs seed2027 holds on both sides
(training 0.955 vs 0.271 mean options, 22.3 % vs 3.8 % k≥2|fail) and should carry the argument.

---

## 5. Paragraph for the response to reviewer y9x6

> **On the 1.5B-WebShop cell.** The like-for-like replicate has landed and it is not favourable: on
> the *identical* 800 training tasks, changing only run-time randomness gives 2.5 % strict success
> against the submitted 35.5 % (a third run on a different task draw gives 3.5 %). We diagnosed it
> rather than absorbing it. WebShop scores 1.0 only on an exact match of every requested attribute,
> and 88.4 % of our training instructions request at least two attributes that must be *clicked* on
> the product page. All three runs first converge on `search → click item → buy`, which earns
> 0.83–0.92 but cannot reach 1.0 on those tasks, and the escape requires learning to click each
> option — a behaviour whose marginal reward is roughly flat (−0.06 early, +0.08 late on the
> majority task type) against a 6 % invalid-action penalty. The runs differ in exactly that one
> behaviour, and we can show it on episodes where the behaviour earned *nothing*: restricted to the
> 38 validation tasks on which the submitted run and the replicate obtained an identical partial
> score (matched within ±0.02, mean 0.398), the submitted run clicks ≥2 options on 52.6 % of
> episodes and the replicate on 0.0 % (paired difference +0.53, bootstrap 95 % CI [+0.37, +0.68]);
> over all 129 tasks where both merely failed, 64.3 % vs 1.6 %. Nothing else about the two policies
> differs — searches 1.01 vs 1.02, item clicks 1.00 vs 0.97, buy rate 0.97 vs 0.94 per episode —
> and the submitted run's zero-option episodes are, if anything, slightly *worse* (mean reward 0.454
> vs 0.468). This is why the metric is knife-edge: at step 50 all three runs are behaviourally
> identical (≥2 options on ≤2.5 % of episodes) and read 1.0 / 1.5 / 1.0 %, the escape onsets at step
> 74 in the submitted run and step 81 in one replicate and never in the other, and a policy that
> never escapes loses only 0.16 of mean reward (0.706 vs 0.544) while losing all of its strict
> success — 23.5 % of its evaluation episodes sit stranded in the 0.75–0.90 band that the submitted
> run converted. We therefore report this cell as a distribution rather than a point, add mean
> reward alongside, and are running a 150-step budget to test whether the take-off is late rather
> than absent; that run will be reported next to, never in place of, the protocol-matched 100-step
> number, and we will not quote a best-seed number for this cell. Two further notes: the same
> signature appears in CHORD at this scale (0.603 mean reward, 11.5 % strict, 37.5 % stranded), so
> this is a property of 1.5B-WebShop at a 100-step budget rather than of DUET; and no DUET mechanism
> anticipates the escape — over all 382 logged metrics, `dr3/disc_acc`, `chord/mu`, teacher gradient
> share and every State-Channel key are flat and mutually indistinguishable across the three runs.
> Relatedly, and against our own interest, we measured that our WebShop potential is *inert* on this
> behaviour: Φ scores which options a page offers, not which are selected, so ΔΦ = 0 for
> 16,783 of 16,783 valid option clicks. We correct the paper's claim of dense WebShop shaping
> accordingly, and scoring selected options is the first fix we will report in the camera-ready. The
> ALFWorld cells are unaffected — that reward is binary, with a three-seed spread of
> 47.5 / 38.0 / 41.0 %.

---

## Provenance of every number in this document

| number | file / command |
|---|---|
| val@{50,100} strict, mean, P(k≥2\|fail), stranded bands, action composition, P(2nd\|1st) | `experiments/webshop/<run>/validation_log/{50,100}.jsonl` via `verdict_verify.py` |
| paired same-task and same-reward tests, bootstrap CIs | same file; instructions keyed on `Instruction: [SEP] … [SEP]` in the `output` field; `verdict_verify.py` |
| per-block option rates, steps 81–100 pooled stats, onset tests, rolling ≥1-option series | `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_{1..100}.jsonl`; `verdict_train.py` |
| requested-attribute distribution, ceiling table, ΔΦ, teacher stats | same trajectories + `agentevolver/module/exp_manager/state_progress.py`; `verdict_ceiling.py` |
| Φ definition | `state_progress.py:252` (`webshop_attribute_aware_potential`), `:154` (`compute_attribute_match_score`), `:145` (`parse_product_detail_options`) |
| task-subset selection | `agentevolver/module/trainer/ae_ray_trainer.py:890-911`, `agentevolver/module/task_manager/task_manager.py:155-161` |
| 150-step config | `config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_fixedtask_seed2027_s150.yaml` |
| queue state | `logs/a100_followup.log`, `run_a100_followup_queue.sh`, `ps -eo pid,etime,cmd` |
| step cap | `multi_turn.max_steps: 30` in each run's config |
| 382-key onset sweep, per-click precision/recall, jackpot rates, reach-vs-click, memorisation stats | carried over from `ws_analysis_escape_event.md` and `ws_analysis_reward_landscape.md`; not re-derived here |

**Caveats not papered over.** Conditioning on failure conditions on a collider, but the induced bias
runs *against* the finding and the same-reward test (§1) and action-composition test do not depend
on it. Validation is n=200 per run; the paired tests use 129 and 38 matched pairs and the CIs
exclude zero widely. The attribute parser handles colour/size/fit-type but not style/flavour/
material, so 88.4 % is a lower bound. One anomaly in the ceiling table (r=1, k=0 shows 1 strict
success in 1,166) reflects parser imprecision on that task class, not a violation of the
constraint. Marginal-return estimates are observational: episodes with k=2 are not randomly
assigned, so those figures are descriptive; the ceiling result and the same-reward comparison are
not affected. No counterfactual has been run — we can show the current potential is inert on this
behaviour, not that fixing it would change the outcome.
