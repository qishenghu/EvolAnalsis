# WebShop 1.5B seed-2025 result: forensic verdict

For the DUET authors, deciding what to tell reviewer y9x6.
Every number below was re-derived from files on disk; provenance is given inline.
Date of audit: 2026-07-26. All paths relative to `/data/home/qisheng/EvolAnalsis`.

---

## 1. Verdict

**None of the three offered explanations is right. It is not an environment fault, it is not a
"genuine seed effect" in the sense of seed 2026 being better, and although the task-sampling
confound is real it is not what produced the gap. The correct verdict is: the paper's 35.5% is a
single high-variance draw at a late phase transition, selected as the maximum of a sweep, and it
does not reproduce.** The single most decisive piece of evidence is a third run already on disk,
`experiments/webshop/ws_1_5b_swC02_da`: its YAML differs from the paper run in nothing but
`rollout.gpu_memory_utilization` (0.75 -> 0.6) and names/output paths (5-hunk `diff` of
`launcher_record/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06/yaml_backup.yaml` vs
`launcher_record/ws_1_5b_swC02_da/yaml_backup.yaml`); it has `seed: 2026`; and it trained on the
**identical 8 tasks at every one of the 100 steps** as the paper run (per-step task-id sets from
`checkpoints/agentevolver/*/Trajectory/trajectories_step_*.jsonl` match 100/100). It scored
val@100 strict **1.0%**, mean reward **0.5477** — *worse than the seed-2025 run's 3.5% / 0.5212*.
Seed 2026 on the paper's exact data therefore yields 35.5% once and 1.0% once. Whatever the
35.5% is, it is not attributable to the seed and not attributable to the task draw.

---

## 2. The three runs, measured

Scored directly from `experiments/webshop/<run>/validation_log/{50,100}.jsonl`, `score` field
(strict = fraction with `score >= 1.0`):

| run | seed | val@50 mean / strict | val@100 mean / strict | val@100 median | distinct scores |
|---|---|---|---|---|---|
| `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` (paper) | 2026 | 0.5219 / 1.0% | **0.7057 / 35.5%** | 0.8333 | 42 |
| `webshop_qwen1.5b_duet_a100_seed2025` (new) | 2025 | 0.4390 / 1.5% | **0.5212 / 3.5%** | 0.6655 | 42 |
| `ws_1_5b_swC02_da` (same seed + same tasks as paper) | 2026 | 0.4793 / 0.5% | **0.5477 / 1.0%** | 0.6000 | 33 |

n = 200 in every cell. Validation decoding is **greedy and identical** in all runs:
`actor_rollout_ref.rollout.val_kwargs` in each run's `yaml_backup.yaml` sets only `n: 1` and
`stop_sequences`, inheriting `temperature: 0`, `top_p: 1.0`, `top_k: -1`, `do_sample: False`
from `external/config_fallback/ppo_trainer.yaml:436-452`. So validation adds no sampling noise:
the entire val difference is a difference in the weights the run arrived at.

Population context, from a scan of all 81 `experiments/webshop/*/validation_log/100.jsonl`
(66 of them Qwen-1.5B):

- Qwen-1.5B **median** strict@100 = **3.0%**, median mean-reward = **0.5207**. The seed-2025 run
  (3.5% / 0.5212) sits essentially exactly at the population median.
- 66.7% of 1.5B runs score strict <= 4%; interquartile range of strict is [1.0%, 5.5%].
- The paper run is the **maximum of all 66 1.5B runs on both metrics** (next best: `duet_v24` at
  22.0% / 0.6782; then `swB_01` at 20.5% / 0.5020).

---

## 3. Hypothesis by hypothesis

### H1 — Environment fault during the seed-2025 run: **REFUTED**

For:
- Nothing. Every fault signature was checked and is absent.

Against (all from `logs/webshop_agentgym.log`, parsed with the access-log regex
`^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - [\d.]+ - (\w+) (\S+) - (\d+) - ([\d.]+) seconds`):

- **100,924 requests parsed, HTTP status histogram = `{'200': 100924}`.** Zero 4xx, zero 5xx.
- Window `2026-07-26 01:12:32 -> 03:41:05`. Largest inter-request gap = **132 s**, and it occurs at
  01:12:32, i.e. server warm-up before training started; the next two largest are 26 s and 24 s.
- Route counts: `/create` 6800, `/reset` 6800, `/instruction_text` 6800, `/observation` 6800,
  `/delete` 6799, `/step` 30056, `/available_actions` 36868. **6800 = 100 steps x 64 rollouts +
  2 x 200 validation.** Nothing lost, nothing retried.
- Exactly **one** server lifecycle: one `Uvicorn running on`, `Started server process [3008701]`
  at line 15, one `Shutting down` at line 539738. No restart, no duplicate stack.
- Episode shape shows no fault: 193/200 seed-2025 val episodes reach a `Buy Now` page and mean
  assistant turns = **4.80** — *longer* than the same-seed replica `ws_1_5b_swC02_da` (3.02) and
  only slightly below the paper run (5.31). Truncation/fault signatures go the other way.
- Score distribution is not degenerate: 42 distinct score values, median 0.6655, mass concentrated
  in the partial-credit band, not at the floor. A broken grader or a broken page server cannot
  produce that.
- **Internal control that kills H1 outright:** the same server, in the same 149-minute window,
  served the 6400 *training* episodes of the same run, whose reward curve ends at the same place as
  the paper run's (see H4 table). A fault would have to be invisible to 6400 training episodes and
  visible only in the 400 interleaved validation episodes on the same server.

The 7 "unparsed" validation rows are model pathology, not environment failure. Dumping them from
`.../webshop_qwen1.5b_duet_a100_seed2025/validation_log/100.jsonl` (rows 36, 55, 81, ...): each is
**one** assistant turn, 4.4-5.0k chars, a `search[...]` query that degenerates into token repetition
("`... youth small small size small small small small ...`") and exhausts the 512-token budget
(`response_length: 512`); the episode contains no environment observation at all. All 7 score
exactly `-0.1` = `env_service.env_params.invalid_action_final_reward: -0.1` in the config. The
paper run and the replica each have exactly 1 such row at step 50 and 0 at step 100 — same failure
mode, different rate.

### H2 — Task-sampling confound: **REAL MECHANISM, NOT THE CAUSE**

The mechanism is real and must not be misdescribed in the rebuttal:

- `data.seed` has exactly two consumers in the repo (`grep -rn` over `agentevolver/` and
  `external/`): `ae_ray_trainer.py:359` (`train_dataloader_generator.manual_seed`) and
  `ae_ray_trainer.py:904-919`, which passes it to `task_manager.load_tasks_from_environment` where
  `random.seed(seed); random.shuffle(response); response = response[:max_tasks]` selects **which
  800 of the 6710 WebShop training tasks** are used.
- Empirically the two runs' 800-task sets overlap in **89 tasks (Jaccard 0.059)** — 89% disjoint.
  I reproduced both sets **exactly, offline**: `ids = [d["item_id"].replace("webshop_","") for d in
  json.load(open("env_service/environments/webshop/webshop_train.json"))]; random.seed(s);
  random.shuffle(ids); set(ids[:800])` equals the observed set for both seeds (0 extra, 0 missing).
  The mechanism is fully determined.
- 89 is an ordinary draw: hypergeometric expectation is 800·800/6710 = **95.4**.
- Because `task_manager` seeds the *global* RNG, the per-task teacher-demo draw
  (`teacher_experience.select_mode: random`) also changes. The contrast is confounded by design.

But it does not explain the result:

- Teacher coverage of the two subsets (against
  `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl`, 26,178 demos over 5,691 distinct
  task ids): **686/800 = 0.8575** (2026) vs **668/800 = 0.8350** (2025); mean demos/task 3.97 vs
  3.82; mean demo reward **1.0 for both**. Monte-Carlo over 3000 random 800-subsets: mean 0.8487,
  sd 0.0120, 95% band [0.8250, 0.8712]; the two observed values sit at the 75th and 12th
  percentiles — both inside the band.
- Equal difficulty at initialisation: on-policy train reward at steps 1-20 is **0.1051** (2026) vs
  **0.1058** (2025) (from `trajectories_step_*.jsonl`, excluding `diag.is_teacher`).
- No leakage difference: train ∩ test = **0** for both subsets, against
  `env_service/environments/webshop/webshop_test.json` (200 ids).
- The validation set is provably identical. Val is loaded with `shuffle=False` and no seed, and
  empirically the **union of regex-recovered instructions over all six validation logs is exactly
  200** — never 201. Per log: paper 199/200 unique (1/0 unparsed at step 50/100), seed-2025
  194/193 unique (6/7 unparsed), replica 199/200 (1/0 unparsed). All unparsed rows score exactly
  -0.1 and are the repetition collapses described in H1.
- **Decisive:** the replica `ws_1_5b_swC02_da` drew the *identical* 800 tasks in the identical
  per-step order as the paper run (per-step set identity 100/100) and still scored 1.0%.

### H3 — Genuine training-seed effect (seed 2026 > seed 2025): **REFUTED**

For: the two runs really do differ only in `seed` at the config level. `diff` of the two YAMLs is
exactly three semantic lines (`experiment_name`, `data.seed: 2026 -> 2025`, `workspace_id`).

Against:
- Seed 2026 produces **35.5% and 1.0%** on identical data (paper run vs `ws_1_5b_swC02_da`). A seed
  whose effect spans the entire population range is not an effect.
- `data.seed` is not a pure run-time RNG knob anyway (H2): it resamples the curriculum, so even a
  clean 2026-vs-2025 difference would not be a "seed effect" in the sense a reviewer means.
- The seed-2025 run was never disadvantaged: identical val set, clean environment, teacher coverage
  inside the MC band, equal step-1-20 difficulty, fresh initialisation, identical greedy val
  decoding, same node, same 4 GPUs.

### H4 — Run-to-run variance at a late phase transition, amplified by a knife-edge metric and by
max-of-sweep selection: **SURVIVES**

On-policy training reward by 20-step block (`trajectories_step_*.jsonl`, teacher rows excluded;
n ≈ 1145 per block):

| block | paper (2026) mean / strict | seed-2025 mean / strict | replica (2026) mean / strict |
|---|---|---|---|
| 1-20 | 0.1051 / 0.9% | 0.1058 / 1.3% | 0.1094 / 1.3% |
| 21-40 | 0.4382 / 1.5% | 0.3901 / 1.3% | 0.4200 / 1.5% |
| 41-60 | 0.4618 / 1.6% | 0.4311 / 2.5% | 0.4556 / 1.3% |
| 61-80 | 0.5107 / 2.4% | 0.4641 / 3.2% | 0.5142 / 1.0% |
| 81-100 | 0.5670 / **9.6%** | 0.5460 / **6.5%** | 0.5046 / **1.1%** |

Three facts follow.

1. **The training-side gap is 1.5x while the validation-side gap is 10x** (9.6% vs 6.5% on train,
   35.5% vs 3.5% on val). Nothing in the training signal is 10x apart. On *mean reward* — the
   metric that is not knife-edge — the val gap is only 0.706 vs 0.521, and the train gap is
   0.567 vs 0.546.
2. **Strict success is a phase transition inside the 100-step budget.** The paper run's train
   strict goes 2.4% (61-80) -> 9.6% (81-100); its val strict goes 1.0% (step 50) -> 35.5%
   (step 100). Both runs were mid-take-off when we stopped and measured. Where a run sits on that
   curve at step 100 is close to a coin flip, and strict@1.0 (every required attribute option
   clicked) amplifies it roughly 4-7x relative to mean reward.
3. **The replica shows the variance is not seed-borne.** Identical seed, identical tasks, identical
   order; only vLLM `gpu_memory_utilization` differs (0.75 vs 0.6), which changes batch scheduling
   and hence sampling nondeterminism. It never took off at all (1.1% train strict at 81-100).

Combined with the population scan — paper run is the max of 66 1.5B runs on both metrics; median
1.5B run is 3.0% / 0.5207; seed-2025 is at that median — the parsimonious reading is: **the 35.5%
is the upper tail of a wide, heavy-tailed distribution over runs, and it was selected because it
was the maximum of a sweep.**

---

## 4. What is still unresolved

1. **Whether 35.5% is ever recoverable at all.** `ws_1_5b_swC02_da` differs from the paper run in
   `gpu_memory_utilization`, so it is a strong but not bit-for-bit replicate. No true bit-for-bit
   rerun of `swC_02` exists on disk.
2. **The paper run's environment cannot be symmetrically audited.** `logs/webshop_agentgym.log` and
   `logs/webshop_envservice.log` are single non-rotated files overwritten by each queue launch; the
   July log covers only the seed-2025 run. The env evidence *clears* seed-2025 but cannot *clear or
   convict* the April paper run. (Indirect evidence that the env is unchanged: same 200 val task
   ids, same instruction strings, ~21-23 clickable options per product page in all three runs,
   `env_service/environments/webshop/webshop_{train,test}.json` unmodified since April.)
3. **Which `env_service/env_service.py` the April paper run used.** `launcher.py` backs up only
   `agentevolver/` and `config/`, and commit `ae2b8c47` (WebShop shared Ray actor) lands between the
   two runs. No evidence it changed outcomes; not provable from artifacts.
4. **Whether seed 2025 closes the gap past step 100.** Both training curves were still rising.
   `webshop_qwen1.5b_duet_a100_seed2025_long150.yaml` exists but
   `experiments/webshop/*long150*` does not.
5. **True seed-to-seed variance of DUET on WebShop-1.5B.** With two seeds and a bundled
   curriculum-resampling confound, variance cannot be decomposed.

### What would settle it

Two configs are already written and never executed:
`config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_fixedtask_seed2025.yaml`
and `..._fixedtask_seed2027.yaml`. Both set `data.task_seed: 2026` (line 175), and the code path
exists (`ae_ray_trainer.py:904-919`: `task_seed = seed if task_seed is None else int(task_seed)`),
so the 800-task draw is pinned to the paper run's while only run-time RNG varies. Neither has an
`experiments/` directory yet.

Run:
1. the two `fixedtask` configs, plus
2. a bit-for-bit rerun of `swC_02` at `gpu_memory_utilization: 0.75` (the only non-name difference
   from the replica), and
3. extend at least one to 150 steps.

Then report **mean ± sd of val MEAN REWARD** (not strict@1.0) at steps 100 and 150 over >= 3
fixed-task replicates.

Decision rule:
- If the fixed-task replicates span roughly **0.52-0.71** mean reward, the 35.5% is run-to-run
  variance at a phase transition, and the paper must report mean ± sd over seeds instead of a single
  max-of-sweep number.
- If all fixed-task replicates land near **0.70** while only the 89%-disjoint-curriculum seed-2025
  run stays at 0.52, the effect is curriculum resampling, not seed noise.

Also: add `env_service/` to `BACK_TARGETS` in `launcher.py` and give the env logs per-run filenames,
so the env code path and env health are provable for future runs.

---

## 5. Wording for the response to y9x6

### Honest version — use this now

> We take this seriously and ran a forensic audit rather than a plausibility argument. Three
> findings.
>
> **(a) The seed-2025 run was not disadvantaged.** The two runs share the same fixed 200-task
> validation set (verified: the union of recovered instructions across all validation logs is
> exactly 200) and identical greedy validation decoding (temperature 0, do_sample False). The
> environment was healthy for the entire run: 100,924/100,924 HTTP 200 responses, no timeouts, no
> retries, a single uninterrupted server lifecycle, and exactly 6,800 episodes created and released
> = 100 steps x 64 rollouts + 2 x 200 validation. Teacher coverage of the two training sets (0.858
> vs 0.835) lies inside a Monte-Carlo 95% band of [0.825, 0.871] over random subsets, teacher
> demonstration quality is identical (reward 1.0 in both), and on-policy reward over steps 1-20 is
> equal (0.1051 vs 0.1058).
>
> **(b) `data.seed` is not only a run-time seed in our codebase**, and we should have said so. It
> also selects which 800 of the 6,710 WebShop training tasks are used; the two runs' training sets
> overlap in only 89 tasks. The validation set is unaffected (loaded with `shuffle=False` and no
> seed). We have added a `data.task_seed` option that pins the task draw so replicates share one
> curriculum.
>
> **(c) The honest conclusion is that our reported WebShop-1.5B number is not robust.** We found a
> third run with the same seed (2026) and, verified step by step, the same 800 training tasks in the
> same order, differing only in a vLLM memory-utilisation setting; it reached 1.0% strict success /
> 0.548 mean reward — below the seed-2025 run. Strict success is a knife-edge metric (every required
> attribute must be selected) and both runs are mid-take-off at step 100: on the training set the
> two runs differ by only 1.5x (9.6% vs 6.5% over steps 81-100) and on validation mean reward by
> 0.706 vs 0.521, while strict success differs by 10x. Across our 66 WebShop Qwen-1.5B runs the
> median strict@100 is 3.0% and median mean reward 0.5207; the seed-2025 run sits at that median and
> the reported run is the maximum.
>
> We are therefore running three replicates with the task draw pinned and the run-time seed varied,
> extending training to 150 steps, and will replace the single WebShop-1.5B number with mean ± sd of
> validation **mean reward** over seeds. We will report those numbers in this discussion period, and
> if they do not support the original figure we will revise the claim. We note that this affects the
> WebShop-1.5B cell specifically; [state here explicitly whether the 3B/7B and ALFWorld results were
> multi-seed or are subject to the same caveat].

### Do **not** say

- "It is a seed effect / seeds vary." — Contradicted by the same-seed replica scoring 1.0%.
- "The environment was broken during the seed-2025 run." — Refuted by 100,924/100,924 HTTP 200 and
  by the run's own healthy training curve on the same server.
- "The two runs used the same training data." — False: 89/800 overlap.
- "Both runs are within noise, nothing to see here." — True in spirit but unsupported until the
  fixed-task replicates exist; asserting it now would be the third unverified claim in a row.

### Stronger version — only if the fixed-task replicates land in the 0.52-0.71 band

> Replacing the single run with N = 3 fixed-curriculum replicates gives validation mean reward
> X ± Y at step 100 and Z ± W at step 150. We have updated Table [n] to report mean ± sd and moved
> strict success to the appendix, since at a 100-step budget it measures where a run sits on a
> phase transition rather than final policy quality. We thank the reviewer — this materially
> improved the paper's reporting.

---

## Appendix: file index

| Fact | File |
|---|---|
| Config diff (3 semantic lines) | `config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml` vs `config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_seed2025.yaml` |
| Replica config diff (`gpu_memory_utilization` only) | `launcher_record/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06/yaml_backup.yaml` vs `launcher_record/ws_1_5b_swC02_da/yaml_backup.yaml` |
| Validation scores | `experiments/webshop/{swC_02...,...a100_seed2025,ws_1_5b_swC02_da}/validation_log/{50,100}.jsonl` |
| Per-step training task ids + on-policy rewards | `checkpoints/agentevolver/<run>/Trajectory/trajectories_step_*.jsonl` |
| Environment HTTP health | `logs/webshop_agentgym.log` |
| `data.seed` consumers | `agentevolver/module/trainer/ae_ray_trainer.py:359,895-919`; `agentevolver/module/task_manager/task_manager.py` (`random.seed`/`shuffle`/`[:max_tasks]`) |
| Task pool / val ids | `env_service/environments/webshop/webshop_train.json` (6710), `webshop_test.json` (200) |
| Teacher demos | `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl` (26,178 demos, 5,691 task ids) |
| Val decoding defaults | `external/config_fallback/ppo_trainer.yaml:436-452` |
| Unexecuted fixed-task configs | `config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_fixedtask_seed{2025,2027}.yaml` |
