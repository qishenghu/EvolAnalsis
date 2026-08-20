# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_grpo10_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 0.0pp vs sampled drop 2.0pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 0.00, sampled 0.68 (|z|>1.96 = significant at 5%).

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
| 10 | 28.9% | 21.8%-37.3% | 32.0% | 28.1%-36.2% | 51.6% |
| 20 | 35.9% | 28.1%-44.5% | 38.5% | 34.4%-42.8% | 57.0% |
| 30 | 43.8% | 35.5%-52.4% | 43.9% | 39.7%-48.3% | 65.6% |
| 40 | 47.7% | 39.2%-56.3% | 46.3% | 42.0%-50.6% | 61.7% |
| 50 | 46.1% | 37.7%-54.7% | 46.9% | 42.6%-51.2% | 69.5% |
| 60 | 47.7% | 39.2%-56.3% | 48.2% | 43.9%-52.6% | 71.9% |
| 70 | 50.0% | 41.5%-58.5% | 56.4% | 52.1%-60.7% | 81.2% |
| 80 | 65.6% | 57.0%-73.3% | 62.7% | 58.4%-66.8% | 84.4% |
| 90 | 71.9% | 63.5%-78.9% | 71.1% | 67.0%-74.9% | 89.8% |
| 100 | 77.3% | 69.4%-83.7% | 69.1% | 65.0%-73.0% | 90.6% |

- **greedy SR**: first 28.9% @10, peak 77.3% @100, final 77.3% @100, peak-to-final drop **0.0pp**, net first-to-final +48.4pp
- **sampled pass@1**: first 32.0% @10, peak 71.1% @90, final 69.1% @100, peak-to-final drop **2.0pp**, net first-to-final +37.1pp
- **sampled pass@4**: first 51.6% @10, peak 90.6% @100, final 90.6% @100, peak-to-final drop **0.0pp**, net first-to-final +39.1pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 2375 | 98 | 24.21 | 4.7% | 0.2% | 0.2% |
| 10 | sampled | 512 | 2653 | 106 | 24.99 | 0.2% | 0.0% | 0.4% |
| 20 | greedy | 128 | 2714 | 111 | 24.38 | 0.0% | 0.0% | 0.0% |
| 20 | sampled | 512 | 3229 | 135 | 23.91 | 0.0% | 0.0% | 0.3% |
| 30 | greedy | 128 | 2449 | 109 | 22.38 | 0.8% | 0.0% | 0.0% |
| 30 | sampled | 512 | 3052 | 133 | 23.03 | 0.2% | 0.0% | 1.4% |
| 40 | greedy | 128 | 2430 | 112 | 21.79 | 2.3% | 0.1% | 0.6% |
| 40 | sampled | 512 | 3073 | 137 | 22.35 | 0.0% | 0.0% | 1.8% |
| 50 | greedy | 128 | 1939 | 89 | 21.66 | 4.7% | 0.2% | 2.5% |
| 50 | sampled | 512 | 2086 | 93 | 22.38 | 0.0% | 0.0% | 3.3% |
| 60 | greedy | 128 | 3480 | 185 | 18.84 | 18.8% | 1.0% | 1.2% |
| 60 | sampled | 512 | 2869 | 131 | 21.88 | 0.6% | 0.0% | 1.0% |
| 70 | greedy | 128 | 3022 | 163 | 18.57 | 18.0% | 1.0% | 1.4% |
| 70 | sampled | 512 | 1779 | 86 | 20.76 | 0.0% | 0.0% | 1.6% |
| 80 | greedy | 128 | 3118 | 214 | 14.59 | 21.1% | 1.4% | 1.6% |
| 80 | sampled | 512 | 2334 | 121 | 19.29 | 2.3% | 0.1% | 1.3% |
| 90 | greedy | 128 | 2251 | 150 | 15.00 | 10.9% | 0.7% | 0.7% |
| 90 | sampled | 512 | 1554 | 89 | 17.40 | 0.2% | 0.0% | 0.1% |
| 100 | greedy | 128 | 843 | 52 | 16.27 | 0.0% | 0.0% | 0.0% |
| 100 | sampled | 512 | 1026 | 55 | 18.51 | 0.2% | 0.0% | 0.2% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 33.6% | 128 | 24.27 | 4.7% | 2026-08-15 22:55 |
| 20 | 39.1% | 128 | 24.07 | 0.8% | 2026-08-16 00:12 |
| 30 | 40.6% | 128 | 22.75 | 0.8% | 2026-08-16 01:28 |
| 40 | 45.3% | 128 | 21.34 | 5.5% | 2026-08-16 02:47 |
| 50 | 50.8% | 128 | 20.95 | 2.3% | 2026-08-16 04:08 |
| 60 | 41.4% | 128 | 19.91 | 14.8% | 2026-08-16 05:33 |
| 70 | 51.6% | 128 | 18.59 | 14.8% | 2026-08-16 06:49 |
| 80 | 63.3% | 128 | 16.43 | 14.8% | 2026-08-16 08:43 |
| 90 | 77.3% | 128 | 13.88 | 7.8% | 2026-08-16 09:54 |
| 100 | 73.4% | 128 | 16.53 | 2.3% | 2026-08-16 11:05 |

## Episode end reasons

- step 10 greedy: {'max_steps': 85, 'env_terminated': 37, 'length_truncation': 6}
- step 10 sampled: {'max_steps': 345, 'env_terminated': 164, 'malformed_action': 2, 'length_truncation': 1}
- step 20 greedy: {'max_steps': 82, 'env_terminated': 46}
- step 20 sampled: {'max_steps': 315, 'env_terminated': 197}
- step 30 greedy: {'max_steps': 71, 'env_terminated': 56, 'length_truncation': 1}
- step 30 sampled: {'max_steps': 286, 'env_terminated': 225, 'length_truncation': 1}
- step 40 greedy: {'max_steps': 64, 'env_terminated': 61, 'length_truncation': 3}
- step 40 sampled: {'max_steps': 274, 'env_terminated': 237, 'malformed_action': 1}
- step 50 greedy: {'max_steps': 63, 'env_terminated': 59, 'length_truncation': 6}
- step 50 sampled: {'max_steps': 272, 'env_terminated': 240}
- step 60 greedy: {'env_terminated': 61, 'max_steps': 43, 'length_truncation': 24}
- step 60 sampled: {'max_steps': 261, 'env_terminated': 247, 'length_truncation': 3, 'malformed_action': 1}
- step 70 greedy: {'env_terminated': 64, 'max_steps': 41, 'length_truncation': 23}
- step 70 sampled: {'env_terminated': 289, 'max_steps': 220, 'malformed_action': 3}
- step 80 greedy: {'env_terminated': 84, 'length_truncation': 27, 'max_steps': 17}
- step 80 sampled: {'env_terminated': 321, 'max_steps': 177, 'length_truncation': 12, 'malformed_action': 2}
- step 90 greedy: {'env_terminated': 92, 'max_steps': 22, 'length_truncation': 14}
- step 90 sampled: {'env_terminated': 364, 'max_steps': 141, 'malformed_action': 6, 'length_truncation': 1}
- step 100 greedy: {'env_terminated': 99, 'max_steps': 29}
- step 100 sampled: {'env_terminated': 354, 'max_steps': 152, 'malformed_action': 5, 'length_truncation': 1}

