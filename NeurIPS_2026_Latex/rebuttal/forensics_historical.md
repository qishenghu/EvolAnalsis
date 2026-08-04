# Forensics: historical run-to-run spread of WebShop 1.5B DUET

**Question addressed (my angle):** what is the empirical run-to-run spread of 1.5B-WebShop DUET
on this machine, and where do seed 2026 (35.5% / 0.706) and seed 2025 (3.5% / 0.521) sit inside it?

**Verdict:** the 32 pp gap is **not** evidence of a seed effect and **not** evidence of an
environment fault. It is inside — in fact at the low end of — the ordinary run-to-run spread of
this exact setting, which I can bound with a **true replicate pair that shares the same seed**.
The paper's 35.5% is the maximum of a 65-run historical distribution whose median is 3.0%;
seed 2025's 3.5% is the 28th of 65 (58th percentile) and sits exactly on the historical
mean-reward -> strict-SR regression line.

All numbers below were recomputed from raw `validation_log/100.jsonl` files, not copied from
any prior report.

---

## 0. Method

* **Corpus:** every `experiments/webshop/webshop_qwen1.5b_*` (and `ws_1_5b_*`) directory with a
  `validation_log/100.jsonl`. **65 runs** total (64 named `webshop_qwen1.5b_*` + `ws_1_5b_swC02_da`).
  Three more runs (`webshop_qwen1.5b_duet_v26`, `_sft`, `_sft_rl`) only have step-50 logs and are
  excluded from the step-100 statistics.
* **Metrics:** `strict SR = fraction of the 200 validation episodes with score >= 0.999`;
  `mean reward = mean of the 200 scores`. This reproduces the two headline numbers exactly
  (swC_02 -> 35.5% / 0.706; seed2025 -> 3.5% / 0.521), so the definitions match.
* **Config knobs:** read from each run's own launch-time snapshot
  `launcher_record/<experiment_name>/yaml_backup.yaml` (not from the live `config/` tree, which
  has been edited since). Code was compared through the launch-time snapshot
  `launcher_record/<experiment_name>/backup/agentevolver/`.
* **Training curves:** parsed from `logs/<experiment_name>.log`
  (`critic/success_onpolicy/mean`, `critic/rewards_onpolicy/mean`, `chord/mu`, `chord/disc_acc_ema`).
* Scripts used are in the session scratchpad; everything is read-only.

---

## 1. The headline: a same-seed replicate reproduces the entire 32 pp gap

`ws_1_5b_swC02_da` (launched 2026-05-04) and the paper run
`webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` (launched 2026-04-30) differ in **exactly one
substantive config key**:

```
actor_rollout_ref.rollout.gpu_memory_utilization: 0.75  ->  0.6
```

That is the whole YAML diff, excluding `experiment_name` / `workspace_id` / the two output paths.
Both runs use **`data.seed: 2026`** — the seed is identical, so the training-task order,
the shuffle, and the validation set are all identical by construction. `gpu_memory_utilization`
is a vLLM KV-cache sizing parameter; it changes nothing in the loss, the data, or any
hyperparameter.

I verified the two runs' code snapshots are functionally identical for this config:

| file | diff lines (swC_02 vs swC02_da) | is it live for this config? |
|---|---|---|
| `module/exp_manager/het_actor.py` | 113 | **No** — the entire diff is one added `elif use_adaptive_mu and adaptive_mode == "disc_acc_velocity":` branch. Both configs set `chord_mu_adaptive_mode: disc_acc`, so the branch is dead code. |
| `module/trainer/ae_ray_trainer.py` | 33 | **No** — added `diag/group_teacher_minus_on_max_reward_*` metrics and a `chord_mu_gap_use_best_of_k` switch (default `False`, and only read in the `gap` mu-mode which neither run uses). |
| `module/exp_manager/state_progress.py` | 0 | identical |
| `backup/config/agentevolver.yaml`, `script_config.yaml` | 0 | identical |

**Result:**

| run | seed | val@100 strict | val@100 mean reward |
|---|---|---|---|
| `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` (paper) | 2026 | **35.5%** | 0.706 |
| `ws_1_5b_swC02_da` | **2026 (same)** | **1.0%** | 0.548 |

A **34.5 pp** strict-SR gap and a **0.158** reward gap between two runs with the *same seed* and
an algorithmically inert config difference. That is larger than the 32.0 pp / 0.185 gap between
seed 2026 and seed 2025.

This single pair is decisive for the framing of the question:

* it **rules out task sampling** — same seed, therefore same train-task order and same eval set;
* it **rules out "the seed did it"** — no seed change was needed to produce the gap;
* it shows the gap is produced by ordinary **run-to-run nondeterminism** (vLLM scheduling / batching
  order, async env interleaving, FSDP reduction order), which this pipeline has in abundance.

The second-cleanest near-replicate is `webshop_qwen1.5b_duet_v39` vs `_v39_postfix`, whose only
YAML difference is `max_env_worker: 64 -> 32` (env-service concurrency, again algorithmically
inert): **11.5% vs 5.5%** strict SR at **identical** mean reward (0.605 vs 0.607). I flag this pair
as weaker evidence because their code snapshots do differ in four core files.

---

## 2. Where the two runs sit in the historical distribution

65 runs with a step-100 validation log:

| statistic | strict SR | mean reward |
|---|---|---|
| mean | 4.9% | 0.474 |
| median | 3.0% | 0.520 |
| sd | 6.3 pp | 0.158 |
| min | 0.0% | -0.100 |
| max | **35.5%** | **0.706** |
| q25 / q75 / q90 / q95 | 1.0% / 5.5% / 11.5% / 16.0% | — |

* **seed 2026 (35.5%, 0.706): rank 1 of 65.** It is the single best run ever recorded in this
  setting, on both metrics. Only 3 runs ever exceeded 20% strict SR; only 1 exceeded 30%.
* **seed 2025 (3.5%, 0.521): rank 28 of 65, 58th percentile — i.e. slightly above the median run.**
  Its mean reward 0.521 is the 52nd percentile, essentially the corpus median (0.520).
* `ws_1_5b_swC02_da` (same seed as the paper run): rank 59 of 65.

47 of the 65 runs (72%) finished at <= 5.0% strict SR. **3.5% is not an outlier — it is the modal
outcome for this configuration on this machine.** The anomaly to be explained is 35.5%, not 3.5%.

Sampling noise in the 200-episode eval is *not* what separates them (Wilson 95% CIs:
71/200 -> [29.2%, 42.3%]; 7/200 -> [1.7%, 7.0%]; reward difference z = 5.4). The two runs really
did produce different policies at step 100. The point is that *training-level* variation of this
size is routine here.

---

## 3. The mu-schedule "sweep signal" is the same size as run-to-run noise

`swC_01 / swB_01 / swC_02` differ only in `chord_mu_d_floor` (0.4 / 0.5 / 0.6) and scored
1.5% / 20.5% / 35.5%. `d_floor` is *not* inert — it enters
`mu = valley + (peak - valley) * clamp((1 - d_ema) / (1 - d_floor), 0, 1)` in
`agentevolver/module/exp_manager/het_actor.py:1795`. But I extracted the realized `chord/mu`
trajectories from the run logs, and the actual difference it produced is tiny:

| step | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| swC_02 (floor .6, 35.5%) | 0.300 | 0.278 | 0.208 | 0.148 | 0.110 | 0.102 | 0.101 | 0.102 | 0.101 | 0.104 | 0.108 |
| swB_01 (floor .5, 20.5%) | 0.300 | 0.234 | 0.193 | — | 0.111 | 0.103 | 0.100 | 0.101 | 0.102 | — | 0.104 |
| swC_01 (floor .4, 1.5%) | — | 0.209 | 0.182 | 0.130 | 0.106 | 0.105 | 0.102 | 0.102 | 0.102 | 0.100 | 0.100 |
| **seed2025 (floor .6, 3.5%)** | 0.300 | 0.285 | 0.211 | 0.136 | 0.114 | 0.114 | 0.108 | 0.103 | 0.103 | 0.107 | 0.110 |

Two things follow.

1. The three `d_floor` values differ by at most ~0.07 in mu, only over steps 10-30, and are
   **identical from step 40 onward** (all pinned at the valley 0.10 once `disc_acc_ema` saturates
   near 1.0). A ~0.05 transient difference in mu produced a 34 pp strict-SR span. The sweep was
   therefore reading noise, not signal — exactly what the same-seed replicate in section 1 shows
   independently.
2. **seed 2025's realized mu trajectory is indistinguishable from the paper run's**
   (max deviation 0.012 across all 100 steps; `disc_acc_ema` saturates the same way, 0.980 vs
   0.984 at the end). Whatever separated the two runs, it was not the algorithm's realized
   schedule.

---

## 4. Reward -> strict SR: the slope, and how much of the 32 pp it explains

Strict SR *is* a steep function of mean reward, as hypothesised — but with enormous conditional
scatter.

OLS of `strict SR (%)` on `mean val reward`, over the 65 runs:

| subset | n | slope (%SR per unit reward) | slope per +0.01 reward | Pearson r | residual sd |
|---|---|---|---|---|---|
| all | 65 | 16.8 | 0.17 pp | 0.421 | 5.7 pp |
| reward >= 0.40 | 55 | 66.2 | 0.66 pp | 0.571 | 5.4 pp |
| reward in [0.45, 0.75] | 51 | 80.4 | 0.80 pp | 0.608 | 5.3 pp |
| **reward in [0.50, 0.75]** | **41** | **95.7** | **0.96 pp** | **0.589** | **5.8 pp** |

Spearman rho over the whole corpus = 0.61. A log-linear fit over reward >= 0.40 gives
`ln(strictSR%) = 9.05 * reward - 3.58` (r = 0.513), i.e. **each +0.01 of mean reward multiplies
strict SR by 1.095** — a ~10x swing per 0.25 of reward. Residual sd in log space is 0.85,
i.e. **one sigma is a factor of 2.3x in strict SR at fixed reward**.

**Applying the [0.50, 0.75] fit to the two runs in question:**

| | observed | predicted from reward | residual |
|---|---|---|---|
| seed 2025 (reward 0.521) | 3.5% | **3.6%** | +0.0 pp |
| seed 2026 (reward 0.706) | 35.5% | 21.3% | **+14.2 pp (+2.4 residual sd)** |

**seed 2025 lands exactly on the historical curve.** Its strict SR is precisely what a WebShop
1.5B run with mean reward 0.52 has always produced here. The reward gap alone accounts for
17.7 pp of the 32.0 pp; the remaining 14.3 pp is the paper run over-performing its own reward.

Binned, non-parametric version:

| mean reward bin | n | strict SR median | strict SR range |
|---|---|---|---|
| [-0.10, 0.30) | 5 | 0.0% | 0.0 - 1.0% |
| [0.30, 0.45) | 9 | 2.0% | 0.5 - 5.5% |
| [0.45, 0.50) | 10 | 2.0% | 1.0 - 4.5% |
| [0.49, 0.56) | 31 | **2.5%** | **1.0 - 20.5%** |
| [0.56, 0.62) | 9 | 5.5% | 4.0 - 13.5% |
| [0.62, 0.75) | 3 | 22.0% | 5.5 - 35.5% |

The `[0.49, 0.56)` bin is the one that matters: it contains **31 runs including seed 2025**, and
strict SR inside it ranges from 1.0% to 20.5%. `swB_01` reached **20.5% strict SR at mean reward
0.502** — a *lower* reward than seed 2025's 0.521. So at this reward level the strict-SR outcome
is close to a lottery, and 3.5% is just above the bin median.

---

## 5. Cross-check: training-side performance says the two runs were nearly equal

Mean on-policy training metrics over steps 81-100 (from `logs/<run>.log`), vs val@100:

| run | train success (81-100) | train reward (81-100) | val strict | val reward |
|---|---|---|---|---|
| swC_02 (paper) | 10.4% | 0.647 | 35.5% | 0.706 |
| **seed 2025** | **7.2%** | **0.630** | **3.5%** | **0.521** |
| swB_01 | 13.0% | 0.698 | 20.5% | 0.502 |
| swC_01 (floor .4) | 1.2% | 0.555 | 1.5% | 0.542 |
| ws_1_5b_swC02_da (same seed as paper) | 1.2% | 0.576 | 1.0% | 0.548 |

On the training distribution the two seeds were **nearly indistinguishable** at step 100
(success 10.4% vs 7.2%; reward 0.647 vs 0.630 — a 0.017 gap, versus a 0.185 gap on validation).
Both curves were still rising. Fitting `val strict SR ~ train success (81-100)` over the 65 runs
with logs gives `valSR% = 1.58 * trSR% - 0.49` (r = 0.779, residual sd 3.91 pp):

* **swC_02 (paper): residual +19.5 pp, z = +5.0** — the largest positive residual in the corpus by a
  wide margin (2nd is v24 at +12.8 / z = +3.3).
* **seed 2025: residual -7.4 pp, z = -1.9** — negative, but three other runs are at least as
  negative (`webshop_qwen1.5b_chord` z = -2.0, `v39b` z = -2.8).

Restricting to the 12 runs whose late-training success landed in the 5-11% band — a
matched-policy-quality cohort containing *both* seeds — val@100 strict SR ranges **2.0% to 35.5%**
(mean 13.0%, sd 8.8 pp). Conditional on equivalent training-time policy quality, the val@100
number in this pipeline has a standard deviation of roughly **9 pp** and an observed range spanning
the entire gap in dispute.

Step-to-step, on-policy training success is itself wildly volatile (steps 81-100 for swC_02:
`0.069, 0.123, 0.035, 0.034, 0.035, 0.125, 0.193, 0.034, 0.034, 0.158, 0.069, 0.071, 0.070, 0.069,
0.088, 0.088, 0.316, 0.161, 0.211, 0.103`; for seed 2025: `0.034, 0.071, 0.035, 0.155, 0.054, 0.000,
0.102, 0.089, 0.070, 0.000, 0.105, 0.053, 0.103, 0.034, 0.070, 0.211, 0.121, 0.000, 0.123, 0.018`).
Evaluating at step 98 or step 102 instead of 100 would plausibly move either number substantially.

At step 50 *every* run in the corpus is between 0.0% and 3.5% strict SR (seed 2026 itself was at
**1.0%**, worse than seed 2025's 1.5%). The strict-SR metric only "ignites" between steps 50 and
100, and which runs ignite before the step-100 snapshot is largely a matter of timing. This is
consistent with the established observation that seed 2025 was ~10 steps behind and still rising.

---

## 6. Two established claims that do not survive contact with the corpus

**(a) The 7 UNPARSED validation rows are not an anomaly.**
I re-ran the instruction-recovery regex over all 65 step-100 validation logs.
**39 of 65 runs have at least one unparsed row.** Full sorted counts:
`1,1,1,1,1,1,1,1, 2,2,2,2,2, 3,3,3,3, 4,4, 5,5,5, 6,6,6, 7(seed 2025), 8,8, 10, 13, 14, 18, 25,
43, 93, 121, 182, 190, 200`.
Seven unparsed rows is ordinary; fourteen runs have >= 7. All 7 of seed 2025's unparsed rows have
score exactly **-0.1**, which is `env_service.env_params.invalid_action_final_reward` — i.e. the
policy emitted invalid actions and the episode was penalised before any `Instruction: [SEP]`
observation was logged. That is a policy failure mode, not a missing/broken environment.
Dropping those 7 rows entirely gives seed 2025 strict SR 3.6% / reward 0.544 — still nowhere near
35.5% / 0.706.

**(b) The validation task set is fixed and seed-independent.**
Across all 65 runs the parsed instruction sets are subsets of one common 200-instruction reference
set (the only exception is `webshop_qwen1.5b_duet_minus_baseline_sep`, which parsed only 10 rows
after a total collapse to reward -0.100). Seed 2026 and 20+ other runs recover all 200; seed 2025
recovers 193 of the same 200. **Changing `data.seed` did not change which tasks are evaluated.**
Combined with section 1 (a same-seed pair reproducing the gap), the task-sampling confound is
ruled out on the eval side.

---

## 7. Full table

`val@100` / `val@50` recomputed from raw logs; knobs read from each run's own
`launcher_record/<name>/yaml_backup.yaml`. All runs use `data.seed: 2026` except
`webshop_qwen1.5b_duet_a100_seed2025`. Sorted by val@100 strict SR.

| run | val@100 strict | val@100 rew | val@50 strict | val@50 rew | seed | dr3 | chord | mu peak/valley | d_floor | ema | SC beta / step-level | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06 | 35.5% | 0.706 | 1.0% | 0.522 | 2026 | True | True | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v24 | 22.0% | 0.678 | 1.0% | 0.510 | 2026 | True | True | 0.3/0.05 | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_swB_01_pk03_v10_ema02 | 20.5% | 0.502 | 3.5% | 0.495 | 2026 | True | True | 0.3/0.1 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_minus_bc | 16.5% | 0.537 | 2.0% | 0.401 | 2026 | True | False | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v37 | 16.0% | 0.532 | 1.5% | 0.460 | 2026 | True | True | 0.3/0.05 | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_02_peak02 | 13.5% | 0.545 | 3.0% | 0.540 | 2026 | True | True | 0.2/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swC_03_pk03_v12_ema02 | 13.5% | 0.589 | 2.0% | 0.492 | 2026 | True | True | 0.3/0.12 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_chord | 11.5% | 0.603 | 3.0% | 0.558 | 2026 | False | True | 0.9/0.05 | None | None | OFF / None |  |
| webshop_qwen1.5b_duet_v39 | 11.5% | 0.605 | 2.5% | 0.482 | 2026 | True | True | 0.3/0.05 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_11_pk05_v10 | 11.0% | 0.534 | 1.5% | 0.515 | 2026 | True | True | 0.5/0.1 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_minus_dr3 | 9.5% | 0.502 | 1.5% | 0.511 | 2026 | False | True | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_05_peak06 | 8.0% | 0.521 | 1.5% | 0.530 | 2026 | True | True | 0.6/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_12_pk05_ema02_v10 | 8.0% | 0.555 | 1.0% | 0.472 | 2026 | True | True | 0.5/0.1 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v10 | 7.0% | 0.571 | 2.0% | 0.470 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_03_peak04 | 5.5% | 0.639 | 2.5% | 0.516 | 2026 | True | True | 0.4/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v3 | 5.5% | 0.446 | 1.0% | 0.524 | 2026 | True | False | None/None | None | None | 0.2 / True |  |
| webshop_qwen1.5b_duet_v39_postfix | 5.5% | 0.607 | 0.5% | 0.503 | 2026 | True | True | 0.3/0.05 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v11 | 5.0% | 0.388 | 2.0% | 0.470 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v40b | 4.5% | 0.496 | 2.0% | 0.496 | 2026 | True | True | 0.3/0.05 | None | None | 0.2 / False | mu mode=nll |
| webshop_qwen1.5b_duet_v43a | 4.5% | 0.569 | 3.0% | 0.450 | 2026 | True | True | 0.3/0.05 | None | None | 0.2 / False | mu mode=kl_lagrangian |
| webshop_qwen1.5b_luffy | 4.5% | 0.573 | 1.0% | 0.467 | 2026 | False | False | None/None | None | None | OFF / None |  |
| webshop_qwen1.5b_duet | 4.0% | 0.549 | 1.5% | 0.444 | 2026 | True | False | None/None | None | None | 0.2 / True |  |
| webshop_qwen1.5b_duet_swA_10_pk05_ema02 | 4.0% | 0.529 | 1.5% | 0.517 | 2026 | True | True | 0.5/0.05 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swB_03_pk03_v10_ema01 | 4.0% | 0.568 | 1.5% | 0.504 | 2026 | True | True | 0.3/0.1 | 0.5 | 0.1 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v23 | 4.0% | 0.440 | 2.0% | 0.496 | 2026 | True | True | 0.1/0.1 | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v8 | 4.0% | 0.574 | 1.0% | 0.442 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_a100_seed2025 | 3.5% | 0.521 | 1.5% | 0.439 | 2025 | True | True | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_08_ema08 | 3.5% | 0.409 | 0.5% | 0.449 | 2026 | True | True | 0.3/0.05 | 0.5 | 0.8 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v22 | 3.5% | 0.462 | 1.5% | 0.500 | 2026 | True | True | 0.05/0.05 | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v38 | 3.5% | 0.474 | 2.0% | 0.494 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v15 | 3.0% | 0.556 | 1.5% | 0.473 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v16 | 3.0% | 0.542 | 2.0% | 0.480 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v41b | 3.0% | 0.543 | 1.0% | 0.455 | 2026 | True | True | 0.3/0.05 | None | None | 0.2 / False | mu mode=ess_ratio |
| webshop_qwen1.5b_duet_v17 | 2.5% | 0.508 | 2.0% | 0.524 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v20 | 2.5% | 0.477 | 1.0% | 0.345 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_swB_02_pk03_v15_ema02 | 2.0% | 0.539 | 0.5% | 0.423 | 2026 | True | True | 0.3/0.15 | 0.5 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v13 | 2.0% | 0.477 | 1.5% | 0.444 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v28 | 2.0% | 0.495 | 1.5% | 0.506 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v39b | 2.0% | 0.320 | 1.5% | 0.501 | 2026 | True | True | 0.3/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_04_peak05 | 1.5% | 0.548 | 1.5% | 0.527 | 2026 | True | True | 0.5/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swA_06_peak07 | 1.5% | 0.549 | 2.5% | 0.528 | 2026 | True | True | 0.7/0.05 | 0.5 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_swC_01_pk03_v10_floor04 | 1.5% | 0.542 | 1.0% | 0.521 | 2026 | True | True | 0.3/0.1 | 0.4 | 0.2 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v31 | 1.5% | 0.517 | 2.0% | 0.428 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v36 | 1.5% | 0.389 | 1.5% | 0.527 | 2026 | True | True | 0.05/0.05 | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_minus_sc | 1.0% | 0.450 | 2.0% | 0.480 | 2026 | True | True | 0.3/0.1 | 0.6 | 0.2 | OFF / False |  |
| webshop_qwen1.5b_duet_v12 | 1.0% | 0.431 | 1.5% | 0.423 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v14 | 1.0% | 0.528 | 1.5% | 0.506 | 2026 | True | False | None/None | None | None | 0.15 / False |  |
| webshop_qwen1.5b_duet_v18 | 1.0% | 0.501 | 1.0% | 0.252 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v19 | 1.0% | 0.469 | 2.0% | 0.428 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v2 | 1.0% | 0.521 | 1.0% | 0.474 | 2026 | True | False | None/None | None | None | 0.1 / True |  |
| webshop_qwen1.5b_duet_v21 | 1.0% | 0.095 | 1.5% | 0.549 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v29 | 1.0% | 0.511 | 1.5% | 0.509 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v30 | 1.0% | 0.520 | 2.5% | 0.523 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v32 | 1.0% | 0.465 | 2.5% | 0.274 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v33 | 1.0% | 0.520 | 1.5% | 0.452 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v39c_postfix | 1.0% | 0.511 | 1.0% | 0.456 | 2026 | True | True | 0.3/0.05 | 0.4 | 0.5 | 0.2 / False |  |
| webshop_qwen1.5b_duet_v7 | 1.0% | 0.473 | 2.0% | 0.552 | 2026 | False | False | None/None | None | None | 0.2 / True |  |
| webshop_qwen1.5b_duet_v9 | 1.0% | 0.533 | 2.5% | 0.473 | 2026 | False | False | None/None | None | None | 0.2 / False |  |
| ws_1_5b_swC02_da | 1.0% | 0.548 | 0.5% | 0.479 | 2026 | True | True | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False | gpu_mem=0.6 |
| webshop_qwen1.5b_duet_v4 | 0.5% | 0.343 | 1.5% | 0.473 | 2026 | True | False | None/None | None | None | OFF / True |  |
| webshop_qwen1.5b_duet_v6 | 0.5% | 0.305 | 1.5% | 0.488 | 2026 | True | False | None/None | None | None | 0.05 / True |  |
| webshop_qwen1.5b_onpolicy | 0.5% | 0.152 | 2.0% | 0.433 | 2026 | False | False | None/None | None | None | OFF / None | no baseline-sep, no teacher |
| webshop_qwen1.5b_duet_minus_baseline_sep | 0.0% | -0.100 | 0.5% | 0.177 | 2026 | True | True | 0.3/0.1 | 0.6 | 0.2 | 0.2 / False | no baseline-sep |
| webshop_qwen1.5b_duet_v25 | 0.0% | -0.041 | 1.5% | 0.523 | 2026 | True | False | None/None | None | None | 0.2 / False |  |
| webshop_qwen1.5b_duet_v5 | 0.0% | -0.100 | 0.0% | -0.100 | 2026 | True | False | None/None | None | None | 0.2 / True | no baseline-sep |

---

## 8. Bottom line

1. **A same-seed replicate reproduces the whole gap.** `ws_1_5b_swC02_da` shares
   `data.seed: 2026` with the paper run and differs only by `gpu_memory_utilization 0.75 -> 0.6`
   (verified inert: the only code deltas are dead branches for this config). It scored
   1.0% / 0.548 against the paper run's 35.5% / 0.706. No seed change was required to produce a
   34.5 pp swing.
2. **3.5% is unremarkable.** It is the 28th best of 65 historical step-100 evaluations
   (58th percentile), against a corpus median of 3.0%. 72% of runs finished at <= 5%.
3. **35.5% is the outlier.** Rank 1 of 65 on both metrics, +2.4 sd above the reward->strict-SR
   line and +5.0 sd above the train-success->val-strict-SR line. It is the argmax of a 64-run
   sweep and carries the corresponding winner's-curse selection bias.
4. **The steep-slope hypothesis is supported but insufficient.** Empirical slope in the
   [0.50, 0.75] reward window is **0.96 pp of strict SR per +0.01 of mean reward**
   (log-linear: x1.095 per +0.01). That predicts **3.6%** for seed 2025 (observed 3.5% — a perfect
   hit) and **21.3%** for seed 2026 (observed 35.5%). Reward alone explains 17.7 of the 32.0 pp;
   the rest is the paper run over-performing its own reward.
5. **Conditional on matched policy quality the metric has ~9 pp sd.** Among the 12 runs with
   late-training on-policy success in 5-11%, val@100 strict SR spans 2.0% - 35.5%.

**Verdict for this angle: task-sampling confound is ruled out; a genuine, reproducible
seed-specific effect is ruled out; what remains is ordinary run-to-run nondeterminism amplified by
a metric (strict SR) that is a steep, high-variance function of mean reward near a
phase transition that occurs between steps 50 and 100.** I found no evidence of an environment
fault in the seed-2025 run from this angle (its unparsed-row count, its validation task set, its
mu/disc_acc trajectories, and its training curves are all within the historical norm) — but this
angle cannot positively exclude one either.

## 9. Open items

* `webshop_qwen1.5b_duet_a100_seed2027` was launched 2026-07-26 00:57 (config snapshot and a 65 KB
  log exist) but has produced **no** `experiments/webshop/` directory. Its outcome is the single
  most valuable missing datapoint — a third seed under the current code. Not investigated further
  here (read-only constraint; training is live).
* The paper run's **+2.4 sd** positive residual on the reward->strict-SR line, and its **+5.0 sd**
  residual on the train->val line, are unexplained by anything in this analysis. Worth a separate
  look at whether its step-100 checkpoint caught an unusually favourable moment.
* Whether the same instability holds at 3B/7B was not tested here. Weak supporting datapoint:
  `ws_swC_v_pk03_v00` was run three times at 3B under evolving code
  (`.BUGGY_1717`, `.LATCH_V1_2234`, final) and produced **39.5% / 36.5% / 28.5%** strict SR
  (rewards 0.713 / 0.722 / 0.674) — an 11 pp spread, i.e. large but proportionally smaller than
  at 1.5B.
