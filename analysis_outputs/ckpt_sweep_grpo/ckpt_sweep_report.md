# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_grpo_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 0.0pp vs sampled drop 0.0pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 0.00, sampled 0.00 (|z|>1.96 = significant at 5%).

Frozen rule:

```
Let D_greedy  = peak-to-final drop of the greedy success rate (percentage points)
    D_sampled = peak-to-final drop of the sampled mean pass@1 (percentage points)
  * decoding artifact  : D_greedy >= 10pp AND D_sampled <= max(0.5*D_greedy, 5pp)
  * real degradation   : D_sampled >= 0.6*D_greedy AND D_sampled >= 7pp
  * intermediate       : anything else — both magnitudes reported side by side
```

## Curves (primary run)

| step | greedy SR | 95% CI | sampled pass@1 | 95% CI | sampled pass@4 |
|---:|---:|---:|---:|---:|---:|
| 50 | 33.6% | 26.0%-42.1% | 50.6% | 46.3%-54.9% | 75.0% |
| 100 | 39.1% | 31.0%-47.7% | 53.5% | 49.2%-57.8% | 78.1% |

- **greedy SR**: first 33.6% @50, peak 39.1% @100, final 39.1% @100, peak-to-final drop **0.0pp**, net first-to-final +5.5pp
- **sampled pass@1**: first 50.6% @50, peak 53.5% @100, final 53.5% @100, peak-to-final drop **0.0pp**, net first-to-final +2.9pp
- **sampled pass@4**: first 75.0% @50, peak 78.1% @100, final 78.1% @100, peak-to-final drop **0.0pp**, net first-to-final +3.1pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | greedy | 128 | 5167 | 247 | 20.95 | 30.5% | 1.5% | 1.5% |
| 50 | sampled | 512 | 6585 | 327 | 20.13 | 17.4% | 0.9% | 2.5% |
| 100 | greedy | 128 | 4893 | 290 | 16.90 | 35.2% | 2.1% | 2.0% |
| 100 | sampled | 512 | 4277 | 218 | 19.58 | 10.0% | 0.5% | 5.6% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 0 | 30.5% | 128 | 24.70 | 2.3% | 2026-08-08 22:06 |
| 10 | 25.8% | 128 | 25.76 | 0.0% | 2026-08-08 23:24 |
| 20 | 28.9% | 128 | 25.05 | 0.0% | 2026-08-09 00:37 |
| 30 | 47.7% | 128 | 21.12 | 1.6% | 2026-08-09 01:50 |
| 40 | 49.2% | 128 | 21.02 | 3.9% | 2026-08-09 03:07 |
| 50 | 35.2% | 128 | 21.07 | 26.6% | 2026-08-09 04:56 |
| 60 | 55.5% | 128 | 15.61 | 28.1% | 2026-08-09 06:27 |
| 70 | 41.4% | 128 | 20.52 | 7.0% | 2026-08-09 07:41 |
| 80 | 34.4% | 128 | 17.47 | 36.7% | 2026-08-09 08:58 |
| 90 | 22.7% | 128 | 21.13 | 25.8% | 2026-08-09 10:24 |
| 100 | 39.1% | 128 | 16.55 | 34.4% | 2026-08-09 11:55 |

## Episode end reasons

- step 50 greedy: {'max_steps': 46, 'env_terminated': 43, 'length_truncation': 39}
- step 50 sampled: {'env_terminated': 259, 'max_steps': 164, 'length_truncation': 89}
- step 100 greedy: {'env_terminated': 50, 'length_truncation': 45, 'max_steps': 33}
- step 100 sampled: {'env_terminated': 274, 'max_steps': 187, 'length_truncation': 51}

