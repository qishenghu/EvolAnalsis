# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_catalystv4_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 4.7pp vs sampled drop 0.0pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 0.89, sampled 0.00 (|z|>1.96 = significant at 5%).

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
| 10 | 28.1% | 21.1%-36.5% | 34.2% | 30.2%-38.4% | 54.7% |
| 20 | 47.7% | 39.2%-56.3% | 47.3% | 43.0%-51.6% | 64.8% |
| 30 | 61.7% | 53.1%-69.7% | 60.5% | 56.2%-64.7% | 74.2% |
| 40 | 69.5% | 61.1%-76.8% | 71.3% | 67.2%-75.0% | 89.8% |
| 50 | 65.6% | 57.0%-73.3% | 66.6% | 62.4%-70.5% | 87.5% |
| 60 | 68.0% | 59.5%-75.4% | 66.4% | 62.2%-70.4% | 89.1% |
| 70 | 78.9% | 71.0%-85.1% | 72.9% | 68.8%-76.5% | 90.6% |
| 80 | 62.5% | 53.9%-70.4% | 58.8% | 54.5%-63.0% | 75.0% |
| 90 | 68.0% | 59.5%-75.4% | 65.0% | 60.8%-69.0% | 82.0% |
| 100 | 74.2% | 66.0%-81.0% | 74.2% | 70.3%-77.8% | 85.9% |

- **greedy SR**: first 28.1% @10, peak 78.9% @70, final 74.2% @100, peak-to-final drop **4.7pp**, net first-to-final +46.1pp
- **sampled pass@1**: first 34.2% @10, peak 74.2% @100, final 74.2% @100, peak-to-final drop **0.0pp**, net first-to-final +40.0pp
- **sampled pass@4**: first 54.7% @10, peak 90.6% @70, final 85.9% @100, peak-to-final drop **4.7pp**, net first-to-final +31.2pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 1614 | 65 | 24.94 | 3.1% | 0.1% | 0.1% |
| 10 | sampled | 512 | 1851 | 76 | 24.30 | 0.2% | 0.0% | 9.3% |
| 20 | greedy | 128 | 1675 | 77 | 21.68 | 2.3% | 0.1% | 0.1% |
| 20 | sampled | 512 | 1807 | 82 | 22.00 | 0.0% | 0.0% | 0.3% |
| 30 | greedy | 128 | 1900 | 100 | 19.01 | 0.8% | 0.0% | 0.0% |
| 30 | sampled | 512 | 1953 | 102 | 19.21 | 0.0% | 0.0% | 0.0% |
| 40 | greedy | 128 | 1355 | 75 | 18.09 | 0.0% | 0.0% | 0.0% |
| 40 | sampled | 512 | 1359 | 78 | 17.44 | 0.0% | 0.0% | 0.0% |
| 50 | greedy | 128 | 1484 | 77 | 19.23 | 0.0% | 0.0% | 0.0% |
| 50 | sampled | 512 | 1451 | 78 | 18.57 | 0.0% | 0.0% | 0.0% |
| 60 | greedy | 128 | 1825 | 97 | 18.79 | 0.0% | 0.0% | 0.0% |
| 60 | sampled | 512 | 2083 | 106 | 19.59 | 0.0% | 0.0% | 0.0% |
| 70 | greedy | 128 | 1284 | 82 | 15.69 | 0.0% | 0.0% | 0.0% |
| 70 | sampled | 512 | 1535 | 88 | 17.46 | 0.0% | 0.0% | 0.0% |
| 80 | greedy | 128 | 1292 | 70 | 18.36 | 0.0% | 0.0% | 0.0% |
| 80 | sampled | 512 | 1436 | 75 | 19.21 | 0.0% | 0.0% | 0.2% |
| 90 | greedy | 128 | 1336 | 75 | 17.72 | 0.0% | 0.0% | 0.0% |
| 90 | sampled | 512 | 1404 | 76 | 18.41 | 0.0% | 0.0% | 0.0% |
| 100 | greedy | 128 | 1794 | 115 | 15.54 | 1.6% | 0.1% | 0.1% |
| 100 | sampled | 512 | 1791 | 110 | 16.33 | 0.0% | 0.0% | 0.0% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 30.5% | 128 | 24.66 | 3.9% | 2026-08-18 02:41 |
| 20 | 47.7% | 128 | 21.84 | 0.0% | 2026-08-18 03:52 |
| 30 | 57.0% | 128 | 19.61 | 0.0% | 2026-08-18 05:03 |
| 40 | 69.5% | 128 | 18.32 | 0.0% | 2026-08-18 06:14 |
| 50 | 64.1% | 128 | 18.68 | 0.0% | 2026-08-18 07:29 |
| 60 | 66.4% | 128 | 19.27 | 0.0% | 2026-08-18 08:47 |
| 70 | 74.2% | 128 | 16.53 | 0.0% | 2026-08-18 09:50 |
| 80 | 62.5% | 128 | 18.55 | 0.0% | 2026-08-18 10:55 |
| 90 | 64.8% | 128 | 18.12 | 0.0% | 2026-08-18 12:06 |
| 100 | 75.0% | 128 | 15.42 | 0.0% | 2026-08-18 13:19 |

## Episode end reasons

- step 10 greedy: {'max_steps': 88, 'env_terminated': 36, 'length_truncation': 4}
- step 10 sampled: {'max_steps': 335, 'env_terminated': 175, 'malformed_action': 1, 'length_truncation': 1}
- step 20 greedy: {'max_steps': 64, 'env_terminated': 61, 'length_truncation': 3}
- step 20 sampled: {'max_steps': 270, 'env_terminated': 242}
- step 30 greedy: {'env_terminated': 79, 'max_steps': 48, 'length_truncation': 1}
- step 30 sampled: {'env_terminated': 310, 'max_steps': 202}
- step 40 greedy: {'env_terminated': 89, 'max_steps': 39}
- step 40 sampled: {'env_terminated': 365, 'max_steps': 147}
- step 50 greedy: {'env_terminated': 84, 'max_steps': 44}
- step 50 sampled: {'env_terminated': 341, 'max_steps': 171}
- step 60 greedy: {'env_terminated': 87, 'max_steps': 41}
- step 60 sampled: {'env_terminated': 340, 'max_steps': 172}
- step 70 greedy: {'env_terminated': 101, 'max_steps': 27}
- step 70 sampled: {'env_terminated': 373, 'max_steps': 139}
- step 80 greedy: {'env_terminated': 80, 'max_steps': 48}
- step 80 sampled: {'env_terminated': 301, 'max_steps': 211}
- step 90 greedy: {'env_terminated': 87, 'max_steps': 41}
- step 90 sampled: {'env_terminated': 333, 'max_steps': 179}
- step 100 greedy: {'env_terminated': 95, 'max_steps': 31, 'length_truncation': 2}
- step 100 sampled: {'env_terminated': 380, 'max_steps': 132}

