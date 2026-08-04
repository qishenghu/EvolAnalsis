# Evidence pack — Reviewer bDeY factual questions (Q1 group composition, Q2 SFT data/curve, Q3 teacher cache) + Table 1 underline audit

All numbers below were recomputed from files in this repo on 2026-07-26. Every claim carries a
`file:line` or a reproducible command. Anything I could not verify is listed under **GAPS** at the
bottom and is *not* asserted in the draft rebuttal text.

Reference runs (the two 1.5B SOTA runs quoted in Table 1):
- ALFWorld: `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_duet_v39c_postfix.yaml`
- WebShop: `config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml`

---

## Q1 — Group composition: what are n and m, and is m fixed?

### Configured values (both SOTA configs)

| Setting | ALFWorld config | WebShop config |
|---|---|---|
| `actor_rollout_ref.rollout.n` (group size) | `8` — line 130 | `8` — line 131 |
| `exp_manager.teacher_experience.n_teacher_rollouts_per_task` | `1` — line 228 | `1` — line 236 |
| `exp_manager.teacher_experience.mix_mode` | `rollout_level` — line 226 | `rollout_level` — line 234 |
| `exp_manager.teacher_experience.select_mode` | `random` — line 230 | `random` — line 238 |
| `exp_manager.teacher_experience.max_trajectories_per_task` | `6` — line 229 | `6` — line 237 |
| `data.train_batch_size` (prompts per update) | `8` — line 158 | `8` — line 160 |
| `data.max_train_tasks` | `800` — line 168 | `800` — line 170 |
| `trainer.total_epochs` | `1` | `1` |

So the **nominal** group is **n = 7 on-policy + m = 1 teacher = 8 total per prompt**, and a step is
8 prompts × 8 rollouts = 64 trajectories.

**Caveat on `max_trajectories_per_task: 6`**: this key is read into `self.teacher_max_per_task` at
`agentevolver/module/exp_manager/exp_manager.py:88` and is **never referenced anywhere else** in the
codebase (`grep -rn "teacher_max_per_task" agentevolver/` returns exactly that one line). It is a
dead config key under `rollout_level` mixing and has **no effect**. Do not cite it in the rebuttal as
a per-task cap — it isn't one.

### Is m fixed? Is there resampling-until-success? — **NO. Verified.**

**m is an upper bound drawn from a frozen, pre-collected, offline cache. There is no online teacher
generation and no resample-until-m-successes loop anywhere in the training path.**

Evidence chain:

1. **The cache is loaded once from disk at construction time.**
   `exp_manager.py:96-102` — if `teacher_experience.enable` is true, `load_teacher_trajectories(data_path)`
   is called in `__init__`; it reads a `.pkl`/`.jsonl` file (`exp_manager.py:669-731`) and raises
   `FileNotFoundError` if it yields 0 trajectories. The only I/O is `pickle.load` / `json.loads`.

2. **Selection is sampling without replacement from that fixed per-task list — no generation.**
   `get_teacher_rollouts_for_luffy_mixing()` at `exp_manager.py:1026-1090`:
   ```python
   if task_id not in self.teacher_task2trajectories:
       continue                                    # 1057-1059: cache miss -> task simply skipped
   teacher_trajs = self.teacher_task2trajectories[task_id]
   if len(teacher_trajs) <= n_teacher_rollouts_per_task:
       selected = teacher_trajs.copy()             # 1061-1062: take whatever exists
   elif self.teacher_select_mode == "random":
       selected = random.sample(teacher_trajs, n_teacher_rollouts_per_task)   # 1063-1065
   ```
   No environment call, no LLM call, no retry loop.

3. **Teacher rollouts *replace* on-policy rollouts; group size is held at exactly n_rollout.**
   `experience_collate.py:722`, `:726`, `:742` (`mix_trajectories`):
   ```python
   n_onpolicy_to_keep = self.n_rollout - actual_teacher_count
   kept_onpolicy_cmts = onpolicy_cmts[:n_onpolicy_to_keep]
   ...
   task_cmts = kept_onpolicy_cmts + teacher_cmts
   ```
   The full `n_rollout=8` on-policy rollouts are always generated first
   (`get_n_onpolicy_rollouts_per_task()` returns `self.n_rollout`, `experience_collate.py:397-406`),
   then up to `m` of them are dropped and replaced by cached teacher trajectories.

4. **Cache miss → the teacher slot is *back-filled with on-policy*, not dropped.**
   Docstring `experience_collate.py:426-430` states the intended behaviour and the code implements it:
   `actual_teacher_count = 0` ⇒ `n_onpolicy_to_keep = 8` ⇒ the group is 8 on-policy + 0 teacher.
   The group is never smaller than 8. Empirically confirmed: over the 100-step ALFWorld run,
   `sum(teacher) + sum(on-policy) = 782 + 5618 = 6400 = 100 steps × 8 prompts × 8 rollouts` exactly.

5. **Teacher conversion is pure tokenization.** `env_manager.convert_offpolicy_to_cmt()`
   (`agentevolver/module/env_manager/env_manager.py:591`) only re-tokenizes stored message lists into
   the same CMT container used for on-policy rollouts.

6. **The two optional gates that could reduce m are OFF in both SOTA configs.**
   `difficulty_gate_79` (skip teacher when on-policy avg reward is high, `experience_collate.py:518-529`)
   and `disable_teacher_710` (hard cut-off after step N, `experience_collate.py:492-504`) are absent
   from both YAMLs, so both default to `enable: False`. Confirmed empirically: the logged
   `luffy/tasks_with_partial_teacher` is 0.000 at every one of the 100 steps in both runs.

### Empirical realised composition (from the actual training logs)

Parsed from `logs/alfworld_qwen1.5b_duet_v39c_postfix.log` and
`logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` (100 `step:` metric lines each):

| Metric | 1.5B-ALFWorld | 1.5B-WebShop |
|---|---|---|
| steps parsed | 100 | 100 |
| prompts seen (8/step) | 800 | 800 |
| Σ teacher rollouts | **782** | **686** |
| Σ on-policy rollouts | 5618 | 5714 |
| Σ total (must be 6400) | 6400 ✓ | 6400 ✓ |
| avg teacher rollouts **per prompt** | **0.978** | **0.858** |
| prompts with 0 teacher (cache miss) | 18 / 800 (**2.25%**) | 114 / 800 (**14.25%**) |
| `luffy/tasks_with_partial_teacher` | 0 at all steps | 0 at all steps |
| `diag/teacher_sample_ratio` mean (nominal 0.125) | 0.1221 | 0.1071 |
| `duet/teacher_gradient_share` first → last | 0.195 → 0.156 (mean 0.241) | 0.521 → 0.057 (mean 0.134) |

Reproduce: parse lines matching `step:(\d+) - ` and split on ` - `, keys
`luffy/total_teacher_rollouts`, `luffy/total_onpolicy_kept`, `luffy/tasks_without_teacher`.

---

## Q2 — Does the SFT-GRPO baseline use the exact same teacher data? + SFT training curve

### What "SFT + GRPO" actually is in our codebase (two stages, 50 + 50 steps)

**Stage 1** — `config/duet_paper_experiments_configs/alfworld/alfworld_qwen1.5b_sft.yaml`
(and `webshop/webshop_qwen1.5b_sft.yaml`): the GRPO backbone with a **constant-weight SFT auxiliary
loss** on the mixed-in teacher rollout:
- `actor.use_chord: true` (line 61), `use_dr3: false` (line 60)
- `chord_mu_peak: 1.0`, `chord_mu_valley: 1.0`, `chord_mu_warmup_steps: 0`, `chord_mu_decay_steps: 0`
  (lines 70-73) ⇒ μ(t) ≡ 1.0 for all t (verified in logs: `chord/mu` = 1.000 at every step, n=45/47).
- `chord_use_token_weighting: false` (line 74) ⇒ φ ≡ 1, i.e. **plain token-level cross-entropy**, not
  CHORD's φ(p)=p(1−p) reweighting. Verified: `chord/phi_mean:1.000`, `chord/phi_std:0.000`.
- The SFT loss is applied **only to teacher tokens**: `compute_chord_sft_loss()` at
  `agentevolver/module/exp_manager/het_core_algos.py:1767`, `expert_mask = exp_mask * response_mask`
  (line 1799), `sft_losses = -log_prob` (line 1802).

**Stage 2** — `alfworld_qwen1.5b_sft_rl.yaml`: initialises from
`checkpoints/agentevolver/alfworld_qwen1.5b_sft/global_step_50/actor_hf` (line 45), sets
`use_chord: false`, and **deletes the entire `teacher_experience` block** ⇒ pure on-policy GRPO, zero
teacher data. (`diff` of the two YAMLs confirms these are the only substantive changes.)

Budget arithmetic (`total_epochs: 1`, `max_train_tasks: 400`, `train_batch_size: 8` ⇒ 400/8 = 50
steps per stage; `save_freq: 50` produced `global_step_50`). Confirmed on disk: each of the four SFT
run dirs `experiments/{alfworld,webshop}/*_sft{,_rl}/rollout_log/` contains step files `1.jsonl`…`50.jsonl`.
**50 + 50 = 100 steps, identical to DUET's budget.**

### Answer: same cache and same *per-prompt rate*, but ~half the *total* teacher volume

**Identical**: cache file, `n_teacher_rollouts_per_task: 1`, `select_mode: random`,
`mix_mode: rollout_level`, `rollout.n: 8`, group size 8.
The SFT config points at exactly the same file (`alfworld_qwen1.5b_sft.yaml:185`:
`data_path: data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl`;
`webshop_qwen1.5b_sft.yaml:192`: `webshop_qwen72b_filtered.pkl`).

**Different**: the SFT stage occupies only the first 50 of the 100 steps, so it consumes roughly half
as many teacher trajectories in total.

| | teacher trajs consumed | prompts with teacher data | steps with teacher data |
|---|---|---|---|
| DUET — ALFWorld | **782** | 782 / 800 | 100 / 100 |
| SFT+GRPO — ALFWorld | **390** | 390 / 400 (stage 1 only) | 50 / 100 |
| DUET — WebShop | **686** | 686 / 800 | 100 / 100 |
| SFT+GRPO — WebShop | **335** | 335 / 400 (stage 1 only) | 50 / 100 |

(Σ of `luffy/total_teacher_rollouts` over the run logs; SFT stage-1 avg teacher/prompt = 0.975
ALFWorld, 0.838 WebShop — statistically indistinguishable from DUET's 0.978 / 0.858, i.e. the
*sampling process* is identical, only the number of steps it runs for differs.)

**Task-set relationship (verified, worth stating pre-emptively):** both stages of SFT+GRPO use
`max_train_tasks: 400`, `shuffle: true`, `seed: 2026`; DUET uses `max_train_tasks: 800` with the same
shuffle and seed. `task_manager.py:156-161` does `random.seed(seed); random.shuffle(response);
response[:max_tasks]` — a deterministic prefix take. Therefore **DUET's 800 training tasks are a
strict superset of the SFT baseline's 400**, and the SFT baseline sees its 400 tasks twice (once per
stage) while DUET sees 800 once. Prompt-instance count (800) and update-step count (100) are
identical across the two.

### The SFT training curve **can** be fully reconstructed — no rerun needed

Source: `diag/reward_onpolicy_mean` (mean reward over the on-policy rollouts only, teacher excluded)
logged at every step in the run consoles:

| run | wandb dir | steps |
|---|---|---|
| ALFWorld SFT (stage 1) | `wandb/run-20260415_163026-ean1ymqj/files/output.log` | 1–50 |
| ALFWorld SFT+GRPO (stage 2) | `wandb/run-20260415_203728-tm5scrq5/files/output.log` | 1–50 → plot as 51–100 |
| WebShop SFT (stage 1) | `wandb/run-20260416_095646-y1mh3yk5/files/output.log` | 1–50 |
| WebShop SFT+GRPO (stage 2) | `wandb/run-20260416_132005-npb5x7pw/files/output.log` | 1–50 → plot as 51–100 |

10-step block means of `diag/reward_onpolicy_mean` (ALFWorld is binary, so this **is** on-policy
success rate; WebShop is partial-credit reward):

| steps | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 | 61-70 | 71-80 | 81-90 | 91-100 |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT+GRPO ALFWorld | 0.016 | 0.007 | 0.087 | 0.264 | 0.285 | 0.392 | 0.398 | 0.380 | 0.474 | 0.308 |
| DUET ALFWorld | 0.065 | 0.049 | 0.072 | 0.188 | 0.322 | 0.418 | 0.431 | 0.478 | 0.432 | 0.427 |
| SFT+GRPO WebShop | 0.036 | 0.152 | 0.385 | 0.457 | 0.488 | 0.508 | 0.490 | 0.590 | 0.638 | 0.609 |
| DUET WebShop | 0.074 | 0.241 | 0.476 | 0.535 | 0.522 | 0.546 | 0.565 | 0.607 | 0.640 | 0.654 |

SFT-specific diagnostics also available for the curve, if useful:
- `chord/sft_loss`: ALFWorld 0.844 → 0.153 (mean 0.354, n=45); WebShop 1.380 → 0.561 (mean 0.801, n=47)
- `chord/n_expert_tokens` per micro-batch: ALFWorld mean 354.8; WebShop mean 372.0
- `chord/mu` ≡ 1.000 throughout (constant SFT weight, no schedule)

**Honest framing point for the rebuttal:** on ALFWorld the SFT+GRPO *training* curve reaches roughly
DUET's level (0.474 vs 0.432 at steps 81-90) but its *held-out* score is far lower (30.0% vs 47.5%).
This is consistent with the 2×-repeated 400-task set; it is a generalisation gap, not a training-signal
gap. Stating this ourselves is stronger than letting the reviewer find it.

---

## Q3 — Teacher cache details

Computed by loading the two pickles directly.

| | ALFWorld | WebShop |
|---|---|---|
| file | `data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl` | `data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl` |
| on-disk size | 350,297,403 B (334 MiB) | 452,302,365 B (431 MiB) |
| teacher model | `Qwen2.5-72B-Instruct` (100%) | `Qwen2.5-72B-Instruct` (100%) |
| total trajectories | **19,497** | **26,178** |
| distinct task_ids covered | **2,348** | **5,691** |
| trajectories per covered task: mean | **8.30** | **4.60** |
| median / min / max | 10 / 1 / 10 | 5 / 1 / 5 |
| distribution (count→#tasks) | 10→1322, 9→259, 8→142, 7→144, 6→98, 5→89, 4→80, 3→85, 2→65, 1→64 | 5→4716, 4→270, 3→272, 2→269, 1→164 |
| success flag | `True` for 19,497 / 19,497 | `True` for 26,178 / 26,178 |
| reward | 1.0 for 100% (min=max=1.0) | 1.0 for 100% (min=max=1.0) |
| messages per trajectory: mean / median / max | 26.52 / 22 / 62 | 17.67 / 17 / 27 |
| stored teacher `log_probs` | present in 19,497 / 19,497 | **0 / 26,178 (absent)** |

Collection cap is visible in the data: ALFWorld was capped at 10 samples/task, WebShop at 5/task; the
caches store only the **successful** (reward = 1.0) ones, so a task with fewer than the cap simply had
fewer successes. **This is where "keep generating until m successes" lived — in the *offline collection*
script, not in the training loop.** Training itself never regenerates.

Note: both SOTA configs set `use_log_prob: false` (ALFWorld line 231), so the stored ALFWorld log-probs
are not consumed; DR3 estimates the density ratio from the discriminator instead.

Per-prompt averages during training (repeat of the Q1 table, since bDeY asks for it explicitly):
**0.978 teacher rollouts per prompt on ALFWorld, 0.858 on WebShop**, over 800 prompts each.
Cache-hit rate at training time: **97.75% / 85.75%**.

---

## Table 1 underline audit

### The reviewer is right, and the bug is in the *compiled* table

There are two copies of the main-results table in the repo:

1. `NeurIPS_2026_Latex/tables/main_results.tex` — **has the 3B underlines** (lines 25-26,
   `\underline{67.0\%}` and `\underline{39.0\%}`) — but it is **not `\input` anywhere**. Verified:
   `grep -rn "input{tables/main_results}" --include=*.tex .` returns nothing; the only table inputs in
   the whole project are `tables/app_sensitivity` and `tables/main_results_with_reward`. This file is
   dead and also carries a duplicate `\label{tab:main_results}` (line 12).
2. `NeurIPS_2026_Latex/sections/04_experiments.tex:41-66` — the **inline table that actually compiles**
   (`\label{tab:main_results}` at line 50). Its `\underline` appears on **only one line** —
   line 59, the `SFT$+$GRPO` row — so **the 3B-ALFWorld and 3B-WebShop columns have no underline at all.**

### Per-column strongest non-DUET baseline (from the table's own numbers)

| Column | OnPolicy | LUFFY | CHORD | SFT+GRPO | Strongest non-DUET | Currently underlined? |
|---|---|---|---|---|---|---|
| 1.5B-ALFWorld | 1.0 | 5.5 | 27.0 | **30.0** | SFT+GRPO 30.0% | yes ✓ |
| 1.5B-WebShop | 0.5 | 5.5 | 11.5 | **18.5** | SFT+GRPO 18.5% | yes ✓ |
| 3B-ALFWorld | 47.0 | 61.5 | **67.0** | 59.5 | **CHORD 67.0%** | **NO — missing** |
| 3B-WebShop | 2.0 | 38.0 | **39.0** | 24.0 | **CHORD 39.0%** | **NO — missing** |

**Good news to state in the rebuttal:** the `$\Delta$` row is already computed against the correct
cells — 77.5 − 67.0 = 10.5 ✓ and 45.5 − 39.0 = 6.5 ✓ (and 47.5 − 30.0 = 17.5 ✓, 36.0 − 18.5 = 17.5 ✓).
So this is purely a missing typographic mark; **no reported number changes.**

### Exact corrected LaTeX (replace lines 55-63 of `sections/04_experiments.tex`)

Only two tokens change (`\underline{}` wrapped around the two CHORD cells); everything else is byte-identical.

```latex
\midrule
OnPolicy GRPO & 1.0\% & 0.5\% & 47.0\% & 2.0\% \\
LUFFY & 5.5\% & 5.5\% & 61.5\% & 38.0\% \\
CHORD & 27.0\% & 11.5\% & \underline{67.0\%} & \underline{39.0\%} \\
SFT$+$GRPO & \underline{30.0\%} & \underline{18.5\%} & 59.5\% & 24.0\% \\
DUET (Ours) & \textbf{47.5\%} & \textbf{36.0\%} & \textbf{77.5\%} & \textbf{45.5\%} \\
\midrule
$\Delta$ vs. strongest baseline & +17.5pp & +17.5pp & +10.5pp & +6.5pp \\
\bottomrule
```

Recommended housekeeping (not required for the fix): delete or `%`-comment
`NeurIPS_2026_Latex/tables/main_results.tex`, since it is unused and its duplicate
`\label{tab:main_results}` will silently collide if anyone ever `\input`s it.

**Reminder:** per project convention, run `NeurIPS_2026_Latex/build.sh` and report the page count after
applying this edit. I did not apply it (read-only for this task) and therefore did not recompile.

---

## SEPARATE FINDING — a number a confidence-4 reviewer could check and flag

Not asked by bDeY, but found while verifying, and it is in the same table:

The official metric rule is `score >= 1.0` on the 200-row validation jsonl
(`scripts/aggregate_rebuttal_results.py:63-69`). Recomputing from the saved validation logs:

| cell | paper value | recomputed from validation_log | match? |
|---|---|---|---|
| 1.5B-ALFWorld DUET | 47.5% | 47.5% (`experiments/alfworld/alfworld_qwen1.5b_duet_v39c_postfix/validation_log/100.jsonl`) | ✓ |
| 1.5B-ALFWorld SFT+GRPO | 30.0% | 30.0% (`experiments/alfworld/alfworld_qwen1.5b_sft_rl/validation_log/50.jsonl`) | ✓ |
| 1.5B-WebShop DUET | 36.0% | **35.5%** (71/200, `experiments/webshop/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06/validation_log/100.jsonl`) | **off by +0.5pp** |
| 1.5B-WebShop SFT+GRPO | 18.5% | **18.0%** (36/200, `experiments/webshop/webshop_qwen1.5b_sft_rl/validation_log/50.jsonl`) | **off by +0.5pp** |

The reward means for those same two files are **0.7057** and **0.6409**, which match the paper's
0.706 / 0.641 in `tables/main_results_with_reward.tex` **exactly** — so these are the right runs, and the
strict-SR cells are each one task (0.5pp) high. Both cells are shifted by the same amount, so the
reported **Δ = +17.5pp is correct either way** (35.5 − 18.0 = 17.5). I scanned every
`experiments/webshop/*/validation_log/*.jsonl` with n ≥ 150 and found no run producing exactly
36.0% with reward 0.706, nor 18.5% with reward 0.641. **Recommend deciding deliberately whether to
correct these two cells to 35.5% / 18.0% before submitting the rebuttal** — the Δ claim and all
conclusions are unaffected.

---

## GAPS — what I could NOT verify

1. **Number of *distinct* teacher trajectories consumed.** Logs record only counts per step, not which
   trajectory id was drawn. With one epoch over 800 distinct tasks and 1 draw per task, repeats are
   impossible *within* a run, so 782 / 686 are almost certainly all distinct — but this is an inference
   from the config (`total_epochs: 1`, prefix-take task list), not a direct measurement.
2. **Teacher-collection procedure itself.** The 10-per-task (ALFWorld) / 5-per-task (WebShop) caps and
   the 100%-success filter are *inferred from the data distribution*, not read from a collection script.
   I did not locate the script that produced these two `qwen72b` pickles (`scripts/filter_teacher_trajectories.py`
   exists but I did not verify it was the producer). If the rebuttal wants to state "we generated up to
   K samples per task at temperature T and kept the successes", **that sampling budget K and temperature
   must be confirmed from the collection script or notes before being written down.**
3. **3B-row provenance.** I verified the 3B cells only for *internal consistency* (Δ arithmetic and
   which cell is largest). I did not locate and recompute the 3B validation logs, so I cannot confirm
   47.0 / 61.5 / 67.0 / 59.5 / 77.5 and 2.0 / 38.0 / 39.0 / 24.0 / 45.5 against files. Given finding #4
   above, **the 3B cells deserve the same recomputation before the rebuttal is filed.**
4. **Held-out SFT curve.** Only *training* on-policy reward is available per step. `test_freq: 50` means
   validation ran once per stage, so the SFT+GRPO held-out curve has only 2 points (step 50 of each
   stage). A dense held-out curve would require a rerun — do not promise one.
5. **Cache-hit rate on the validation split** — not logged; the 97.75% / 85.75% figures are training-split only.
