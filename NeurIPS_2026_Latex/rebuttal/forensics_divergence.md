# Forensic divergence analysis: WebShop 1.5B DUET, seed 2026 vs seed 2025

Read-only analysis. No processes touched, no GPUs used, nothing under `experiments/` or
`checkpoints/` modified.

Artifacts analysed:

| run | log | experiment dir |
|---|---|---|
| seed 2026 ("paper run") | `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log` | `experiments/webshop/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` |
| seed 2025 ("new run") | `/data/home/qisheng/EvolAnalsis/logs/webshop_qwen1.5b_duet_a100_seed2025.log` | `experiments/webshop/webshop_qwen1.5b_duet_a100_seed2025` |
| **seed 2026 replicate** (found during analysis) | `/data/home/qisheng/EvolAnalsis/logs/ws_1_5b_swC02_da.log` | `experiments/webshop/ws_1_5b_swC02_da` |

---

## 0. Headline

**The 35.5% vs 3.5% gap is not a seed effect, not an environment fault, and not a
validation-task confound. It is run-to-run nondeterminism, amplified by measuring a
threshold-like metric (strict success = reward exactly 1.0) at the exact step where the
required behaviour is emerging.**

The decisive evidence is a third run, `ws_1_5b_swC02_da`, whose YAML differs from the paper
run in only three lines — `experiment_name`, `workspace_id`, and
`rollout.gpu_memory_utilization: 0.75 -> 0.6`. **Same seed 2026, verified identical 800
training task IDs (800/800 overlap), identical validation set.** It scored
**val@100 strict = 1.0%, mean reward 0.548** — i.e. *worse than the seed-2025 run*, and
34.5 points below the paper run it replicates.

```
run           seed  train tasks       val@50 R / strict   val@100 R / strict
paper2026     2026  set A             0.522 / 1.0%        0.706 / 35.5%
da2026        2026  set A (identical) 0.479 / 0.5%        0.548 /  1.0%
seed2025      2025  set B (89/800 ∩A) 0.439 / 1.5%        0.521 /  3.5%
```

The paper run's 35.5% is the **maximum over 63 WebShop-1.5B DUET runs** on disk
(median 3.0%, p90 13.1%, only 3/63 runs ≥ 20%).

---

## 1. Step-aligned comparison of all shared metrics

Both logs contain exactly 100 `step:N - ...` metric lines; 394 metric keys are common
(seed 2025 additionally logs `diag/group_teacher_minus_on_max_reward_{mean,std}`).
Parsed with `re.compile(r'step:(\d+) - (.*)')`, block-averaged over 10 steps.

```
metric                                  run     1-10  11-20  21-30  31-40  41-50  51-60  61-70  71-80  81-90 91-100
critic/success_onpolicy/mean            2026   0.004  0.019  0.004  0.027  0.007  0.028  0.007  0.046  0.084  0.125
critic/success_onpolicy/mean            2025   0.009  0.019  0.018  0.014  0.031  0.035  0.021  0.050  0.061  0.084
critic/rewards_onpolicy/mean            2026   0.074  0.241  0.476  0.535  0.522  0.546  0.565  0.607  0.640  0.654
critic/rewards_onpolicy/mean            2025   0.066  0.248  0.453  0.458  0.503  0.490  0.504  0.567  0.602  0.657
critic/score/mean                       2026   0.171  0.322  0.531  0.583  0.574  0.596  0.613  0.649  0.678  0.692
critic/score/mean                       2025   0.154  0.329  0.513  0.510  0.555  0.544  0.554  0.615  0.644  0.693
critic/rewards_teacher/mean             both   1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
actor/kl_loss                           2026   0.044  0.103  0.235  0.334  0.413  0.527  0.451  0.563  0.811  0.671
actor/kl_loss                           2025   0.023  0.089  0.273  0.324  0.467  0.824  1.018  0.774  0.858  0.943
actor/entropy_loss                      2026   0.456  0.354  0.546  0.534  0.585  0.615  0.608  0.613  0.544  0.526
actor/entropy_loss                      2025   0.426  0.364  0.459  0.516  0.556  0.537  0.568  0.569  0.467  0.504
actor/grad_norm                         2026   5.034  4.370  2.591  3.101  3.019  2.652  2.614  2.972  3.516  3.143
actor/grad_norm                         2025   4.331  3.966  3.932  5.097  5.298  8.830  7.171  5.573  4.942  5.731
response_length/mean                    2026    1979   2829   1813   1840   1763   1692   1726   1753   1874   2049
response_length/mean                    2025    2100   2793   2170   1843   1767   1746   1644   1703   2225   2103
response_length/clip_ratio              both   0.016 (constant in all 300 rows across 3 runs -> config artifact, not a truncation signal)
dr3/disc_acc                            2026   0.434  0.759  0.885  0.973  0.997  0.997  0.996  0.998  0.995  0.991
dr3/disc_acc                            2025   0.426  0.752  0.910  0.976  0.979  0.987  0.995  0.994  0.984  0.983
dr3/w_off_mean                          2026   0.996  0.949  0.871  0.738  0.670  0.604  0.647  0.639  0.663  0.660
dr3/w_off_mean                          2025   0.993  0.952  0.880  0.763  0.712  0.717  0.609  0.617  0.708  0.661
duet/teacher_gradient_share             2026   0.239  0.183  0.109  0.126  0.118  0.143  0.126  0.127  0.083  0.089
duet/teacher_gradient_share             2025   0.229  0.158  0.107  0.095  0.124  0.166  0.137  0.123  0.115  0.095
chord/mu                                2026   0.295  0.228  0.176  0.124  0.103  0.102  0.101  0.101  0.102  0.105
chord/mu                                2025   0.298  0.243  0.167  0.123  0.113  0.110  0.106  0.103  0.106  0.108
state_channel/progress_mean             2026   0.229  0.354  0.354  0.365  0.373  0.393  0.389  0.405  0.415  0.425
state_channel/progress_mean             2025   0.211  0.354  0.385  0.315  0.359  0.347  0.356  0.398  0.429  0.436
state_channel/coverage_mean             2026   0.838  0.850  0.850  0.838  0.875  0.887  0.875  0.863  0.838  0.863
state_channel/coverage_mean             2025   0.750  0.863  0.887  0.775  0.838  0.850  0.825  0.887  0.850  0.825
state_channel/bonus_vs_reward_ratio     2026   0.122  0.174  0.118  0.114  0.120  0.124  0.120  0.119  0.114  0.116
state_channel/bonus_vs_reward_ratio     2025   0.113  0.172  0.136  0.109  0.117  0.113  0.116  0.119  0.125  0.120
state_channel/shaped_ratio              2026   0.641  0.739  0.739  0.727  0.750  0.774  0.748  0.744  0.689  0.714
state_channel/shaped_ratio              2025   0.563  0.752  0.770  0.648  0.728  0.716  0.680  0.745  0.725  0.710
state_channel/progress_onpolicy_mean    2026   0.191  0.330  0.331  0.341  0.352  0.373  0.369  0.387  0.395  0.407
state_channel/progress_onpolicy_mean    2025   0.173  0.332  0.366  0.289  0.335  0.321  0.333  0.375  0.415  0.422
diag/teacher_sample_ratio               2026   0.105  0.106  0.106  0.105  0.109  0.111  0.109  0.108  0.105  0.108
diag/teacher_sample_ratio               2025   0.094  0.108  0.111  0.097  0.105  0.106  0.103  0.111  0.106  0.103
diag/group_teacher_minus_on_reward_mean 2026   0.918  0.725  0.513  0.451  0.451  0.450  0.436  0.380  0.335  0.336
diag/group_teacher_minus_on_reward_mean 2025   0.918  0.726  0.548  0.526  0.494  0.492  0.479  0.434  0.370  0.305
```

### Which metric moves first

Ranking all 394 shared metrics by `(mean_2025 - mean_2026)` over steps 51-100, normalised by
the pooled SD, and locating the first step at which the 10-step rolling difference exceeds
0.7 pooled SD:

| rank | metric | 26 (1-50) | 25 (1-50) | 26 (51-100) | 25 (51-100) | Δ/SD | first step |
|---|---|---|---|---|---|---|---|
| 1 | `actor/grad_norm` | 3.62 | 4.53 | **2.98** | **6.45** | **+1.85** | **~21** |
| 2 | `diag/llm_token_ratio_in_response` | 0.141 | 0.140 | 0.146 | 0.098 | −1.56 | 46 |
| 3 | `diag/teacher_llm_token_ratio` | 0.138 | 0.127 | 0.146 | 0.210 | +1.34 | 48 |
| 4 | `exp_replay_diag/adv/on/count` | 521 | 566 | 517 | 338 | −1.32 | 8 |
| 5 | `duet/adv_onpolicy_effective_abs_mean` | 0.160 | 0.160 | 0.183 | 0.150 | −0.91 | 50 |
| 6 | `actor/kl_loss` | 0.226 | 0.235 | **0.605** | **0.884** | +0.78 | 49 |

**`actor/grad_norm` is the first metric to diverge.** The seed-2026 run settles into a low
gradient-norm regime (2.0-3.0) from step ~20 onward; seed 2025 never does, and from step 46 it
jumps into a 5-13 regime:

```
grad_norm 2026 steps 46-70: 2.69 2.59 2.62 2.55 2.67 2.29 2.62 1.86 2.18 3.59 2.25 3.03 2.60 3.17 2.94 1.80 2.25 2.56 2.68 2.82 3.43 2.34 1.85 3.05 3.37
grad_norm 2025 steps 46-70: 11.0 3.67 3.38 4.99 5.30 8.54 7.82 8.75 8.78 13.0 9.97 7.50 9.49 7.95 6.50 7.84 9.00 5.76 6.35 6.99 7.37 7.99 7.26 8.18 4.98
```

KL follows about 3-6 steps later: `actor/kl_loss` for 2025 crosses 1.0 first at step 52 and
stays in a 0.8-1.4 band for the rest of the run; 2026 does not reach that band until step 88.

Note `grad_clip: 1.0` (`external/config_fallback/ppo_trainer.yaml:169`), so the *applied*
update magnitude is identical; the reported pre-clip norm only tells us the gradient estimate
is noisier / larger in seed 2025.

**No divergence at all** in: teacher reward (identically 1.000 every step in both), DR3
discriminator accuracy, `dr3/w_off_mean`, `duet/teacher_gradient_share`, `chord/mu`,
all `state_channel/*`, teacher sample ratio, response length. The DUET machinery behaved
identically in both runs.

---

## 2. Test of the "same trajectory, ~10 steps behind" claim — **NOT SUPPORTED**

Cross-correlation of the two curves at lags −20…+20 (positive lag = seed 2025 shifted earlier
to match 2026, i.e. "2025 is behind"):

| series | smoothing | best lag | r at best lag | r at lag 0 |
|---|---|---|---|---|
| `critic/rewards_onpolicy/mean` | none | **0** | 0.878 | 0.878 |
| `critic/rewards_onpolicy/mean` | 5-step MA | **0** | **0.968** | 0.968 |
| `critic/score/mean` | 5-step MA | **0** | 0.967 | 0.967 |
| `critic/success_onpolicy/mean` | none | −3 | 0.469 | 0.327 |
| `critic/success_onpolicy/mean` | 5-step MA | −3 | 0.779 | 0.769 |

MSE-optimal shift on the smoothed reward curve is **L = 1** (mse 0.00279 at L=1 vs 0.00288
at L=0, rising monotonically to 0.0233 at L=20). At L=10 the mean difference has already
*over*shot to −0.028, i.e. shifting by 10 makes the match worse in both MSE and level.

The same test on the behavioural metric that actually matters (fraction of the task's required
product options clicked, computed from training rollouts, see §5) gives best lag **−1**
(r = 0.928) — if anything seed 2025 was *ahead*.

**Conclusion: the two training curves are time-aligned, not shifted. The "10 steps behind"
description does not hold.** Seed 2025 simply sits ~0.03 lower in on-policy reward through the
middle of the run and closes that gap completely by step 100 (0.657 vs 0.654 in the final
block).

---

## 3. Pathologies

Present in seed 2025:
- **Gradient-norm regime shift** from step ~46: sustained 5-13 vs 2-3 (max 13.0 at step 55 vs
  6.98 at step 9 for 2026). Not a spike — a sustained regime.
- **Higher KL to reference**: 22/100 steps above 2× the run median, vs 9/100 for 2026;
  sustained 0.8-1.4 from step 52.
- **Greedy-decoding degeneracy at validation**: 8/200 episodes at step 100 are word-repetition
  loops that fill the 512-token response budget and emit no parseable action (vs 1/200 for
  2026). Example: `search[yellow heather men's dress shirts youth small small small small …]`
  → reward −0.1. These are exactly the 7 "UNPARSED" rows noted in the brief plus one more.
- **More invalid-action terminations**: 22/200 episodes at reward −0.1 (the configured
  `invalid_action_final_reward`) vs 10/200 for 2026.
- **More repeated identical actions**: 52/200 episodes vs 42/200.

Absent / not pathological:
- **No entropy collapse.** `actor/entropy_loss` 0.504 (2025) vs 0.526 (2026) over steps 91-100;
  2025 is marginally lower but both are healthy and rising through the run.
- **No truncation jump.** `response_length/clip_ratio` is a constant 0.016 in all 300 rows
  across all three runs — it is a config-derived artifact and carries no signal.
  `response_length/max` is comparable (mean 8444 vs 7594).
- **`chord/mu` behaves identically** (0.108 vs 0.105 final block; the whole decay schedule
  overlaps within 0.02).
- **No environment errors.** Grep over both logs: 0 `Traceback`, 0 `ConnectionError`,
  0 `ReadTimeout`, 0 retries, 0 "failed to". Seed 2025 has 2 `ERROR` lines — both Ray worker
  `SYSTEM_ERROR` messages at log lines 42473 and 42519 of 42523, i.e. **after** the step-99 and
  step-100 metric lines were already written. They are teardown noise, not a mid-run fault.

Important caveat on the "pathology" reading: the same-seed replicate `da2026` also ended with
high KL (1.10 / 1.38 in the last two blocks) and elevated grad norm (5.18 / 4.12) and scored
1.0%. Across the three runs the ordering is monotone —

```
              late KL   late grad_norm   val@100 strict
paper2026     0.67-0.81   3.1-3.5           35.5%
seed2025      0.86-0.94   4.9-5.7            3.5%
da2026        1.10-1.38   4.1-5.2            1.0%
```

— so the low-KL/low-grad-norm regime looks *associated* with the good outcome, but with n=3
this is suggestive only, and it is a property of the individual run, **not of the seed**.

---

## 4. Did the two runs train on different difficulty?

### 4a. Teacher-side (verified, matches the brief)
- Teacher rollouts per prompt: **0.8575** (2026) vs **0.8350** (2025)
  (`sum(luffy/total_teacher_rollouts)/800`).
- Prompts with ≥1 teacher trajectory: 686/800 vs 668/800.
- `critic/rewards_teacher/mean` = **1.000 at every one of the 100 steps in both runs** — the
  teacher never fails, so teacher-side difficulty is constant by construction.
- `diag/teacher_sample_ratio`: 0.1071 vs 0.1042 mean.

### 4b. On-policy side
`diag/group_teacher_minus_on_reward_mean` (teacher-minus-on-policy gap) is slightly *larger*
for seed 2025 through the middle of training (0.526 vs 0.451 at steps 31-40; 0.479 vs 0.436 at
61-70) but **crosses over by the end** (0.305 vs 0.336 at 91-100), because seed 2025's
on-policy reward catches up. Nothing here indicates a harder task set — it tracks the small
mid-run reward lag exactly.

### 4c. The training task sets are almost disjoint — but equally difficult

This is a real structural confound that the 3-line YAML diff hides. `data.seed` is consumed at
`agentevolver/module/trainer/ae_ray_trainer.py:895` and passed to
`TaskManager.load_tasks_from_environment`, which does
`random.seed(seed); random.shuffle(response); response[:max_tasks]`
(`agentevolver/module/task_manager/task_manager.py:155-162`).

- WebShop train pool = **6710** tasks (`Limiting tasks from 6710 to 800` in both logs).
- `max_train_tasks: 800`, `train_batch_size: 8`, 100 steps ⇒ **exactly one epoch over 800 tasks**.
- Verified from `rollout_log/task_*.jsonl`: each run saw 800 unique task IDs;
  **overlap = 89 (Jaccard 0.059)**. This matches the analytic expectation 800²/6710 = 95.4 and
  reproduces exactly under simulation of the same shuffle.

But the two 800-task draws are **not measurably different in difficulty**. Policy-independent
proxy = number of required product attributes parsed from each instruction:

| | n | mean required attrs | SD | attr-count histogram (0/1/2/3) | mean instruction words | mean price cap |
|---|---|---|---|---|---|---|
| seed 2026 | 800 | 2.004 | 0.73 | 30 / 120 / 467 / 183 | 27.50 | \$64.92 |
| seed 2025 | 800 | 1.995 | 0.74 | 36 / 111 / 474 / 179 | 27.65 | \$62.51 |

Difference in mean required attributes = **+0.009 (SE 0.037, t = 0.24)**. Statistically
indistinguishable.

### 4d. The validation set is identical — no eval-side confound
`load_tasks_from_environment` for val is called with `shuffle=False` and **no seed**
(`ae_ray_trainer.py:938-947`), so the val set cannot depend on `data.seed`. Confirmed
empirically: 200 rows per file in every run; seed 2026 and `da2026` each yield 200 unique
recoverable instructions with 100% overlap; seed 2025 yields 193 unique + 7 rows whose output
is a degenerate repetition loop with no environment observation at all (193 + 7 = 200). The 193
are a strict subset of the 200. **The eval sets are the same 200 tasks.**

Environment determinism check: among the 193 shared tasks, 12 had a byte-identical first
`search[...]` query in both runs; **all 12 returned byte-identical result pages** (same 10
product IDs, same order). The WebShop server behaved identically in the two runs.

---

## 5. Where the runs actually diverge: option selection under greedy decoding

WebShop gives reward 1.0 only when every required attribute *and* every required product
option is matched. Strict success is therefore a hard threshold on one specific behaviour:
click the required options on the item page **before** `click[buy now]`.

Validation runs at **temperature 0** (`val_kwargs` in the experiment YAML overrides only `n`
and `stop_sequences`; `external/config_fallback/ppo_trainer.yaml:436-446` gives
`temperature: 0`), while training rollouts run at **temperature 0.6**. This distinction is the
crux.

### Greedy (validation) behaviour

| run | step | mean reward | strict | unique option clicks / episode | frac of required options clicked | frac of episodes with ALL required options clicked |
|---|---|---|---|---|---|---|
| paper2026 | 50 | 0.522 | 1.0% | 0.02 | 0.005 | 0.000 |
| paper2026 | **100** | 0.706 | **35.5%** | **1.81** | **0.688** | **0.497** |
| da2026 | 50 | 0.479 | 0.5% | 0.01 | 0.003 | 0.000 |
| da2026 | **100** | 0.548 | **1.0%** | **0.00** | **0.000** | **0.000** |
| seed2025 | 50 | 0.439 | 1.5% | 0.17 | 0.057 | 0.027 |
| seed2025 | **100** | 0.521 | **3.5%** | **0.85** | **0.324** | **0.087** |

Strict success tracks "all required options clicked" almost one-for-one
(0.497→35.5%, 0.087→3.5%, 0.000→1.0%). At step 50 *no* run has the behaviour; between step 50
and 100 it emerges, and the three runs are caught at three different points on the way up.

Paired over the 193 shared validation tasks (identical tasks, identical requirements,
mean 2.05 required options each):

```
paired mean reward      2026 = 0.7003   2025 = 0.5437   diff = +0.1566  (SE 0.0271, t = 5.77)
paired exact-1.0         2026 = 70/193   2025 = 7/193
2026 better on 118 tasks, 2025 better on 25, tie on 50
required options hit    2026 = 1.42/2.05  2025 = 0.65/2.05
all required hit         2026 = 51.1%    2025 = 8.7%
among 67 tasks where BOTH reached >= 0.75:  exact-1.0  2026 = 49,  2025 = 7
```

Canonical seed-2025 failure (one of 25 episodes at exactly reward 0.8 = 4/5):
instruction asks for *color: khaki* and *size: x-large*; the environment correctly shows both
in `Clickable elements`; the policy clicks `x-large`, writes a rationale claiming khaki is
already satisfied, and buys. The environment did its job; the policy skipped a click.

### Sampled (training) behaviour — the two runs are the *same*

Recomputing the identical statistic over the 100 training-rollout files
(`rollout_log/{1..100}.jsonl`, temperature 0.6):

```
fraction of required options clicked, 5-step blocks, steps 71-100
  2026   0.208  0.239  0.300  0.340  0.387  0.549
  2025   0.193  0.231  0.334  0.398  0.451  0.396
unique option clicks per rollout
  2026   0.500  0.594  0.797  1.019  1.200  1.572
  2025   0.487  0.594  1.041  1.094  1.272  1.062
```

**Under sampling, seed 2025 is level with (steps 71-80) or ahead of (steps 81-95) seed 2026,
and only falls behind in the last 5 steps.** Best cross-correlation lag = −1, r = 0.928.

So both policies had learned to *sample* the option-clicking behaviour at essentially the same
rate. The difference is entirely in whether that behaviour had become the **argmax**. Note the
inversion: seed 2026's greedy option rate (0.688) is *above* its sampled rate (0.549), whereas
seed 2025's greedy rate (0.324) is *below* its sampled rate (0.396) — its mode is still
"buy now immediately". `da2026`'s greedy rate is 0.000 despite a comparable training curve.

This is why the gap is 10× in strict success but only 0.13 in mean validation reward and
~0 in final training reward: strict success is a max-margin readout of a behaviour that all
three runs are in the middle of acquiring at step 100.

---

## 6. Population context

Every `experiments/webshop/webshop_qwen1.5b_duet*` directory with a `validation_log/100.jsonl`,
plus the two `ws_1_5b_*` 1.5B runs — n = 63:

```
strict success @100:  mean 0.049   median 0.030   p90 0.131   max 0.355
runs with strict >= 0.20:  3/63 (4.8%)
runs with strict >= 0.30:  1/63 (1.6%)   <- the paper run
mean reward @100:     mean 0.478   median 0.520
corr(strict, mean reward) = 0.406
```

The paper run is the single best of 63. Seed 2025's 3.5% is at the population median.
`test_freq: 50` means only two validation points exist per run, both from a single greedy
rollout — there is no within-run replication to average over.

Binomial sampling noise at n=200 is small (SE = 3.4pp at p=0.355, 1.3pp at p=0.035), so the
32-point gap between *those two checkpoints* is real. The variance that matters is
**between runs**, not within an evaluation.

---

## 7. Verdict and what to do about it

**Not an environment fault.** No errors, no timeouts, no retries in either log; the two Ray
worker deaths in the seed-2025 log occur after the final metric line. The WebShop server
returned byte-identical pages for identical queries across the two runs. The validation set is
provably identical. Teacher data coverage matches to within 2%.

**Not a validation-task-sampling confound.** Val loading passes `shuffle=False` and no seed;
all three runs evaluated the same 200 instructions.

**There IS a training-task-sampling confound** — `data.seed` selects 800 of 6710 tasks, and the
two runs share only 89 — but it is not the cause: the draws are equal in measured difficulty
(t = 0.24), and the same-seed / identical-task-set replicate `ws_1_5b_swC02_da` reproduces the
*bad* number, not the good one.

**Not a genuine seed effect.** Seed 2026 produces 35.5% once and 1.0% once, on the same 800
tasks. The seed does not carry the signal.

**What it is:** run-to-run nondeterminism (vLLM async scheduling, FSDP reduction order,
environment worker interleaving — `gpu_memory_utilization` alone was enough to change the
outcome by 34.5 points) acting on a policy that is mid-way through acquiring the
option-selection behaviour at step 100, read out through a threshold metric that turns a ~10%
behavioural difference into a 10× headline difference.

Practical implications for the rebuttal:
1. Do not report WebShop 1.5B strict success @100 as a point estimate. It has a run-level SD
   comparable to its own mean.
2. Mean validation reward is far more stable (0.706 / 0.548 / 0.521 — a 0.18 spread vs a 35×
   spread in strict success) and should be the primary metric at this scale.
3. If a strict number is needed, train past the transition (step 150+) or validate more often
   than `test_freq: 50` and report the mean over several checkpoints and several runs.
4. `ws_1_5b_swC02_da` should be treated as a same-config replicate of the paper run and
   disclosed as such; it is the strongest single piece of evidence in this analysis.

---

## Appendix: how each number was obtained

| claim | command / file |
|---|---|
| 3-line YAML diff | `diff config/duet_paper_experiments_configs/webshop/sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml config/duet_paper_experiments_configs/rebuttal_neurips/webshop/webshop_qwen1.5b_duet_a100_seed2025.yaml` |
| `da2026` is a same-seed replicate | `diff .../sweep_1.5b/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml config/duet_paper_experiments_configs/sota_hunt_2026_05_03/ws_1_5b_swC02_da.yaml` → only `experiment_name`, `workspace_id`, `gpu_memory_utilization` |
| train task overlap | union of `experiments/webshop/*/rollout_log/task_{1..100}.jsonl` `task_id` fields; 800 unique each; paper∩da = 800, paper∩2025 = 89 |
| train pool = 6710 | `grep -oa "Limiting tasks from [0-9]* to [0-9]*" logs/*.log` |
| seed → task selection | `agentevolver/module/trainer/ae_ray_trainer.py:893-904`; `agentevolver/module/task_manager/task_manager.py:155-162` |
| val set seed-independent | `ae_ray_trainer.py:938-947` (`shuffle=False`, no `seed=`) |
| val temperature 0 | `external/config_fallback/ppo_trainer.yaml:436-446` |
| grad_clip 1.0 | `external/config_fallback/ppo_trainer.yaml:169` |
| all step metrics | `re.search(r'step:(\d+) - (.*)')` over each log; 100 rows, 394 shared keys |
| option-click statistics | `<action>...</action>` extraction from `validation_log/{50,100}.jsonl` and `rollout_log/{1..100}.jsonl`; option click = `click[...]` not matching `b[0-9a-z]{9}` / `buy now` / `back to search` / `< prev` / `next >` / `description` / `features` / `reviews` |
| required attributes | `(?:with\|and)\s+([a-z ]+):\s*([^,\.]+?)(?:,\|\s+and\s+\|\s*$\|\s+and price)` over the recovered instruction |
| env determinism | 12 tasks with identical first `search[...]`; compared the `[SEP] B######### [SEP]` product-ID sequence of the resulting page |
