# Forensics: was the WebShop ENVIRONMENT faulty during the seed-2025 run?

**Verdict: No. The environment was healthy end-to-end. Environment fault is ruled out.**

Scope of this document: I was assigned the "environment fault" hypothesis only. I checked it
thoroughly, found no support for it, and in the course of doing so turned up a decisive
cross-check (a *same-seed* replicate of the paper run) that also constrains the other two
hypotheses. That cross-check is reported at the end.

Runs under comparison (all on the same host, Ray node IP `172.23.11.6`):

| label | experiment dir | date | seed | val@100 strict | val@100 mean |
|---|---|---|---|---|---|
| **2026-paper** | `experiments/webshop/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` | 2026-04-30 01:00–03:12 | 2026 | 35.5% | 0.7057 |
| **2025-new** | `experiments/webshop/webshop_qwen1.5b_duet_a100_seed2025` | 2026-07-26 01:12–03:41 | 2025 | 3.5% | 0.5212 |
| **2026-replica** | `experiments/webshop/ws_1_5b_swC02_da` | 2026-05-04 (ends 11:29) | **2026** | **1.0%** | **0.5477** |

---

## 1. Env-service health during 01:12–03:41 on 2026-07-26

### 1.1 AgentGym WebShop server (`logs/webshop_agentgym.log`, port 36003)

Parsed every access-log line (`<ts> - <ip> - <METHOD> <path> - <code> - <secs> seconds`):

```
time range: 2026-07-26 01:12:32 -> 2026-07-26 03:41:05
codes: {'200': 100924}                <-- 100% success, zero non-2xx
  /available_actions n=  36868 mean=0.2134s p50=0.0500 max=4.170
  /step              n=  30056 mean=0.8836s p50=0.3600 max=13.270
  /create            n=   6800 mean=0.6090s p50=0.6500 max=1.820
  /reset             n=   6800 mean=0.4183s p50=0.0700 max=3.970
  /instruction_text  n=   6800 mean=0.2896s p50=0.0400 max=4.510
  /observation       n=   6800 mean=0.2206s p50=0.0200 max=3.590
  /delete            n=   6799 mean=1.1499s p50=0.8900 max=5.600
gaps > 2 min in request stream: []    (149 consecutive minutes covered)
minutes with < 50 requests: 1         (the final partial minute)
```

* **100,924 HTTP requests, every single one `200 OK`.** There is not one 4xx/5xx, not one
  timeout, not one connection reset in the entire run. (The `grep -c 503` hits in these files
  are false positives — they are substrings of WebShop environment ids such as
  `Created WebShop environment 12503882`.)
* **Exactly one server lifecycle.** `grep -c "Uvicorn running on"` = 1 (PID 3008701, matching
  `logs/a100_queue.log`), with a single `Shutting down` at the very end of the file. No restart,
  no re-bind, no second stack.
* **No episode was lost.** 6800 `/create` = 6800 `/reset` = 6800 `/instruction_text` =
  6800 `/observation` = 6799 `/delete` (the last delete lands after log close). 6800 =
  100 train steps x 8 tasks x 8 rollouts (6400) + 2 validation passes x 200 (400). Every
  expected episode was created exactly once; nothing was retried or dropped.

Latency drifts mildly over the run (mean `/step` 0.60 s in the 01:30 bucket to 1.33 s in the
02:45 bucket, back to 1.00 s by 03:15). This is consistent with CPU contention from the 14B
teacher-sampling job that started on GPU 6 at 01:31. Latency has no effect on reward — and no
request ever exceeded 13.3 s, far below any client timeout.

### 1.2 env_service wrapper (`logs/webshop_envservice.log`, port 8083)

Normalising every line to a template and counting:

```
6800  Creating env with instance_id: <N>
6800  Creating instance with env_type: webshop, task_id: <N>, instance_id: <N>
6800  (RemoteEnv) Deleting WebShop environment <N>
6667  (RemoteEnv) Created WebShop environment <N>
6634  (RemoteEnv) WebShop environment released.
 ~250 interleaved/concatenated variants of the two lines above (stdout interleaving artefact)
   1  Starting server on 127.0.0.1:8083
   1  Started a local Ray instance (dashboard 8265)
```

Zero `Traceback`, zero `Error`, zero `Exception`, zero `Timeout`, zero `died`, zero `restart`.
The 6667/6634 counts reconcile to 6800 once the concatenated lines are counted (e.g.
`Created WebShop environment 36054387WebShop environment released.`). One Ray instance, one
server start, no worker deaths.

### 1.3 Rotated variants

There are none. `ls logs/webshop_agentgym* logs/webshop_envservice*` returns exactly two files,
both mtime `2026-07-26 03:41`, both covering the full run window. Nothing was rotated away.

### 1.4 Trainer-side error rates (normalised per step, both runs are exactly 100 steps)

| pattern | 2025-new (42,523 lines) | 2026-paper (42,536 lines) |
|---|---|---|
| `Traceback` | 0 | 0 |
| `Exception` / `exception` | 0 | 0 |
| `ConnectionError` / `refused` / `reset by peer` | 0 | 0 |
| `TimeoutError` / `retrying` / `Retry` | 0 | 0 |
| `failed` / `Failed` / `FAILED` | 0 | 0 |
| `reconnect` | 0 | 0 |
| genuine `ERROR`-level lines | **0** | 0 |

The only two `raylet` "worker died ... SIGTERM" lines in the 2025 log are at lines 42473 and
42519 — i.e. **after** `Dumped generations to .../validation_log/100.jsonl` and after
`local_global_step_folder: .../global_step_100`. They are the normal teardown of the vLLM
workers at the end of a completed run. `logs/a100_queue.log` confirms `[07-26 03:41] rc=0`.

The 4 launcher-banner lines at the very tail of the 2025 log (`Copying ./config`,
`Running command: ...`) are stdout-buffering artefacts of the parent shell; the same banner
appears at the head of the file (line 1-5) and there is only one
`agentevolver.main_ppo` invocation.

### 1.5 The infrastructure incidents from earlier that night did NOT touch this run

* `logs/webshop_qwen1.5b_duet_a100_seed2027.log` ends with `SIGTERM received at
  time=1784998641` = `2026-07-26 00:57:21`. That is the aborted pre-restart attempt, 15 minutes
  before the run we are examining started.
* `logs/a100_queue.log` shows the queue was (re)started at `01:12:07`, stopped both env stacks,
  brought up a **fresh** WebShop stack (AgentGym PID 3008701 on 36003, env_service PID 3009124
  on 8083), ran `[07-26 01:12] RUN`, and got `[07-26 03:41] rc=0`. The next `kill_port` on 36003
  and 8083 happens at 03:41, **after** the run finished. No port was killed during the run.
* Aux ALFWorld stack for teacher sampling: `alfworld_agentgym_aux.log` -> `Uvicorn running on
  http://127.0.0.1:18011`; `alfworld_envservice_aux.log` -> `Starting server on
  127.0.0.1:18091`, Ray dashboard 8267. **Disjoint** from WebShop's 36003 / 8083 / 8265 / 8266.
  The teacher job ran on GPU 6 (`[07-26 01:31] starting 14B teacher sampling on GPU 6`), training
  on GPUs 0,1,2,4 (`run_a100_rebuttal_queue.sh:180 CUDA_VISIBLE_DEVICES=$GPUS`). No overlap.
  Its only observable effect is the mild `/step` latency drift noted above.

### 1.6 The environment served the *same* content in all three runs

If the WebShop index or product pages had degraded, the agent would see fewer clickable options.
It did not:

| run@step | mean #clickables on product pages | product pages reached |
|---|---|---|
| 2025-new@50 | 22.39 | 201 |
| 2025-new@100 | 22.99 | 343 |
| 2026-paper@50 | 21.34 | 203 |
| 2026-paper@100 | 22.11 | 384 |
| 2026-replica@100 | 21.43 | 203 |

Identical to within noise. The option buttons the 2026-paper policy clicks were equally
available to the 2025 policy — it just did not click them (Section 4).

---

## 2. The 7 unparseable validation rows are model degeneration, not env failure

All 7 rows in `webshop_qwen1.5b_duet_a100_seed2025/validation_log/100.jsonl` (indices
36, 55, 81, 101, 129, 164, 170) have the **identical** signature:

* `input` is exactly 1041 chars = the system prompt + the fixed `"OK. I'll help you find..."`
  primer, i.e. the standard prefix, byte-identical to the 193 parseable rows.
* `output` begins with a first assistant action that has collapsed into a repetition loop, e.g.

  ```
  assistant\n<action>\nsearch[yellow heather men's t-shirts youth large large size large
  large large large large large large ... (repeated to the 512-token cap)
  ```

  ```
  assistant\n<action>\nsearch[Youth Men's Heathers Heathers Cotton Heathers Men's Dress Shirts
  Heathers Heathers Cotton Heathers Men's Dress Shirts ... (repeated to the cap)
  ```

* The generation is cut at the per-turn cap (`override_generation_config:
  {'max_new_tokens': 512, 'temperature': 0.6, ...}`), so `</action>` is never emitted and the
  string never contains a closing `[SEP]` — that is the *only* reason the `Instruction:` regex
  fails. The instruction text is not missing from the environment; it is simply buried behind a
  malformed action string in the flattened log format.
* The environment responded correctly and helpfully every time:
  `user\nInvalid action. Only search[...] and click[...] are allowed.\n\nYou can use:
  search[your query]\nClickable elements: ['search']`
* All 7 score **exactly -0.10000000149011612**, which is
  `env_service.env_params.invalid_action_final_reward: -0.1` in the config — a deterministic
  reward rule for episodes that end in invalid actions, applied by the scorer, not an error code.

These are real evaluations of a policy that degenerated, not failures scored 0. Corroborating
evidence:

* The 2026-paper run has the same failure mode, just less of it: **10** episodes at negative
  scores at step 100 (7 of them exactly -0.1), and 1 unparseable row at step 50.
* Regex-detecting degenerate repetition (`(\w+)(?: \1){20,}`) finds 3 such rows in
  2025-new@100 and 0 in 2026-paper@100 — the rest of the -0.1 rows are ordinary
  format failures, not repetition loops.
* Removing every negative-score episode moves the means to 0.5980 (2025-new) vs 0.7469
  (2026-paper). The degeneracy explains ~0.077 of the 0.185 mean-reward gap, i.e. **less than
  half**, and explains **none** of the strict-success gap (3.5% vs 35.5%), which is unaffected
  by episodes scoring below 1.0 either way.

---

## 3. Episode-level shape: no signature of a broken environment

### 3.1 Training rollouts (all 100 steps, `rollout_log/{step}.jsonl`, 64 episodes/step)

| window | metric | 2025-new | 2026-paper |
|---|---|---|---|
| 1–10 | turns / invalid-ep rate / buy rate / mean score | 6.56 / 0.763 / 0.289 / 0.153 | 6.33 / 0.714 / 0.320 / 0.171 |
| 11–30 | | 7.24 / 0.371 / 0.669 / 0.421 | 6.49 / 0.361 / 0.695 / 0.426 |
| 31–60 | | 4.58 / 0.218 / 0.894 / 0.537 | 4.08 / 0.101 / 0.938 / 0.585 |
| 61–100 | | 4.93 / 0.171 / 0.930 / 0.626 | 4.34 / 0.100 / 0.953 / 0.658 |

Both runs follow the same curve. In the last 40 steps the 2025 run reaches a **buy rate of 0.930
vs 0.953** and a **mean training reward of 0.626 vs 0.658** — a few percent apart, nothing
resembling a broken environment. Episodes are not systematically short and not error-terminated.

From the trainer metrics (`step:N - ...` lines), averaged over steps 91–100:

| metric | 2025-new | 2026-paper |
|---|---|---|
| `critic/rewards_onpolicy/mean` | 0.6574 | 0.6541 |
| `critic/success_onpolicy/mean` | 0.0838 | 0.1246 |
| `diag/teacher_sample_ratio` | 0.1030 | 0.1077 |
| `duet/teacher_gradient_share` | 0.0953 | 0.0892 |
| `dr3/disc_acc` | 0.9834 | 0.9905 |
| `diag/response_len_onpolicy_mean` | 1823.0 | 1750.5 |
| `actor/kl_loss` | 0.9433 | 0.6714 |

On-policy **mean** reward is a dead heat (0.657 vs 0.654). Only the **exact-success** rate
differs (0.084 vs 0.125), and both are in a steep late take-off (2025: 0.014 -> 0.072 across
windows 1–20 -> 81–100; 2026: 0.011 -> 0.104). The only metric materially worse for 2025 is
`actor/kl_loss` (0.94 vs 0.67) — a policy-drift signal, i.e. an optimisation difference, not an
environment signal.

### 3.2 Validation episodes (step 100, 200 episodes each)

| run | Buy Now reached | episodes with any invalid action | turns mean / max | rows with no `[SEP]` obs |
|---|---|---|---|---|
| 2025-new@50 | 176/200 | 72 | 4.30 / 30 | 6 |
| 2025-new@100 | 193/200 | 28 | 4.80 / 30 | 7 |
| 2026-paper@50 | 199/200 | 4 | 3.23 / 30 | 1 |
| 2026-paper@100 | 200/200 | 18 | 5.31 / 27 | 0 |
| 2026-replica@100 | 200/200 | 3 | 3.02 / 4 | 0 |

193/200 of the 2025 validation episodes complete a purchase. The 7 that do not are exactly the
7 degenerate rows of Section 2. Episodes are *longer*, not shorter, than the replica's — the
opposite of what a broken environment produces.

### 3.3 Validation task set is identical and disjoint from training

Extracting `task_id` from every `Creating instance with env_type: webshop, task_id: N` line in
`logs/webshop_envservice.log`:

* 6800 creations total; the last 200 are 200 **distinct** task ids; every one of those 200 ids
  appears **exactly twice** in the whole stream (the step-50 and step-100 validation passes).
* `val ∩ train(seed 2025) = 0` and `val ∩ train(seed 2026) = 0` — no contamination, and the
  validation set is **not** drawn by the training seed.
* Independently confirmed by the instruction-text route: the 193 recoverable instructions from
  the 2025 log are a strict subset of the 2026 run's 200 (`set(A) <= set(B)` -> `True`,
  `|B - A| = 7`), and those 7 missing ones are precisely the 7 degenerate rows.

**So both runs were evaluated on the same 200 tasks.** On the 193 tasks both parsed:
2025 mean 0.5437 / strict 3.6%; 2026 mean 0.7003 / strict 36.3%. The gap is not a task-set
artefact.

---

## 4. What actually differs: option-selection behaviour, not the environment

Score distribution at step 100 (200 episodes each):

| band | 2025-new | 2026-paper |
|---|---|---|
| negative | 22 | 10 |
| 0 | 16 | 8 |
| (0, 0.5) | 33 | 22 |
| [0.5, 0.75) | 55 | 42 |
| [0.75, 0.9) | 65 | 41 |
| [0.9, 1.0) | 2 | 6 |
| **exactly 1.0** | **7** | **71** |

The 2025 run piles up in the 0.75–0.9 band — the WebShop signature of *buying an almost-right
item*, missing one required attribute/option. Counting non-navigational click actions (option
buttons such as colour/size, excluding `Buy Now`, `Back to Search`, `Next >`, `Description`,
`Features`, `Reviews`, `Attributes`, and product-ASIN clicks):

| run@step | option clicks / episode | fraction of episodes with >=1 option click |
|---|---|---|
| 2025-new@50 | 1.25 | 0.455 |
| 2025-new@100 | 1.52 | 0.780 |
| 2026-paper@50 | 1.22 | 0.975 |
| **2026-paper@100** | **3.31** | **1.000** |
| 2026-replica@50 | 0.94 | 0.905 |
| 2026-replica@100 | 1.01 | 1.000 |

Between step 50 and step 100 the 2026-paper policy underwent a behavioural phase transition:
option clicks per episode went 1.22 -> 3.31, and its strict success went 1.0% -> 35.5%. The 2025
policy only got to 1.52 option clicks and 3.5%. The 2026-replica never made the transition at all
(1.01 clicks, 1.0%).

This is a policy-behaviour difference on an identical environment surface (Section 1.6 shows the
same ~22 clickables were on offer in every run).

---

## 5. Decisive cross-check: a SAME-SEED replicate of the paper run also scores 1.0%

While looking for another run in the same window, I found `experiments/webshop/ws_1_5b_swC02_da`
(ran 2026-05-04, same host). Diff of its config against the paper run:

```
86c86
<     gpu_memory_utilization: 0.75      (paper)
---
>     gpu_memory_utilization: 0.6       (replica)
   ... plus experiment_name / workspace_id / rollout_data_dir / validation_data_dir
```

`seed: 2026` in **both**. And the task draw proves the seed took effect identically:

```
identical order swC02 vs swC02_da: True
steps with identical task list: 100 / 100
train task sets: 800 vs 800, overlap 800
(for contrast: swC02 vs seed2025 train overlap = 89 / 800)
```

Two runs with the **same seed**, drawing the **exact same 800 training tasks in the exact same
order**, evaluated on the **same 200 validation tasks**, produced:

* paper: strict 35.5%, mean 0.7057
* replica: strict **1.0%**, mean **0.5477**

The replica is *closer to the seed-2025 run* (3.5% / 0.5212) than to its own seed-twin.

Population context — strict@100 for every WebShop run with a saved `validation_log/100.jsonl`
(83 runs). Restricting to the ~60 Qwen-1.5B runs, the modal strict@100 is 1–4%; the top-5 1.5B
runs are `swC_02` (35.5%), `duet_v24` (22.0%), `swB_01` (20.5%), `minus_bc` (16.5%),
`duet_v37` (16.0%). Mean@100 for 1.5B runs clusters at 0.45–0.60; `swC_02`'s 0.7057 is the
single highest.

The seed-2025 run at 3.5% / 0.5212 sits at the centre of that distribution. **The paper's
35.5% is the outlier**, and it is not reproducible even at fixed seed.

---

## 6. Verdict and residual uncertainty

**Environment fault: ruled out.** 100,924 requests / 100% `200 OK`, single uninterrupted server
lifecycle, 6800 episodes created and 6800 deleted with none lost, no gaps > 2 min, no timeouts,
no reconnects, zero exceptions on either side of the wire, identical page content served, and
the aux teacher stack provably on disjoint ports and a disjoint GPU. The "7 unparsed rows" are a
model repetition collapse penalised by a config rule (`invalid_action_final_reward: -0.1`), and
the same failure mode is present (at lower rate) in the paper run.

**Task-sampling confound: real but insufficient.** The seed does fully determine the training
task draw (89/800 overlap between seeds; 800/800 within a seed). But it does **not** affect the
validation set (fixed 200 tasks, disjoint from train, identical across runs), and holding the
draw fixed still yields 35.5% vs 1.0%.

**Genuine run-to-run variance: supported, and it is larger than a seed effect.** The 1.5B WebShop
policy sits right at a late, sharp behavioural transition (learning to click all product options
before Buy Now). `critic/success_onpolicy/mean` is still climbing steeply at step 100 in both
runs. A metric that thresholds at *exactly* reward 1.0 turns that near-threshold state into a
near-binary outcome, which is why strict@100 swings 1.0% -> 35.5% between runs that are
byte-identical in configuration except `gpu_memory_utilization`.

### Caveats

* I could **not** inspect the AgentGym/env_service logs for the 2026-04-30 paper run or the
  2026-05-04 replica: `logs/webshop_agentgym.log` and `logs/webshop_envservice.log` are
  single, non-rotated files overwritten by each queue launch. My env-health evidence is
  one-sided (it establishes the 2025 run's environment was fine, which is what was asked, but I
  cannot symmetrically re-verify the paper run's environment).
* The `agentevolver` code backed up with the two runs differs (Apr 28 vs Jul 26 snapshots). I
  diffed all three changed files. The changes are additive and **inert under this config**:
  - `het_actor.py`: adds an `elif adaptive_mode == "disc_acc_velocity"` branch; this config uses
    `chord_mu_adaptive_mode: disc_acc`, so the branch is never entered.
  - `state_progress.py`: adds `match_dropout` / `obs_noise_p` / `match_mode="soft"` plumbing, all
    defaulting to 0/off, and routes exact matching through a new `_lookup()`. This config uses
    `match_mode: attribute_aware`, which short-circuits at `state_progress.py:876` **before**
    `_lookup` is reached, and takes the unchanged `matched = len(observations)` path at line 932.
  - `ae_ray_trainer.py`: adds `diag/group_teacher_minus_on_max_reward_*` metrics and a
    `chord_mu_gap_use_best_of_k` gate (default `False`, and only consulted under
    `chord_mu_adaptive_mode: gap`).
  I judge these behaviourally inert, but I verified this by reading, not by re-running.
* I did not verify the WebShop product index / `items_shuffle` data files were byte-identical
  across April and July. The indirect evidence (same 200 val task ids, same instruction strings,
  same ~22 clickables per product page, near-identical training reward curves) is strong but not
  a checksum.

### Files and commands

* Env access log parse: `logs/webshop_agentgym.log`, regex
  `^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - [\d.]+ - (\w+) (\S+) - (\d+) - ([\d.]+) seconds`
* Env service task ids: `logs/webshop_envservice.log`, regex
  `Creating instance with env_type: webshop, task_id: (\d+),`
* Validation rows: `experiments/webshop/<run>/validation_log/{50,100}.jsonl`
  (keys: `input`, `output`, `score`, `step`, `reward`)
* Training rollouts: `experiments/webshop/<run>/rollout_log/{step}.jsonl` (64 rows/step) and
  `.../rollout_log/task_{step}.jsonl` (8 rows/step, has `task_id`)
* Config diffs: `launcher_record/<run>/yaml_backup.yaml`, code snapshots under
  `launcher_record/<run>/backup/agentevolver/`
* Queue timeline: `logs/a100_queue.log`
