# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_catalystv3_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 7.0pp vs sampled drop 3.3pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 1.17, sampled 1.14 (|z|>1.96 = significant at 5%).

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
| 10 | 29.7% | 22.5%-38.1% | 38.7% | 34.6%-43.0% | 57.0% |
| 20 | 27.3% | 20.4%-35.6% | 39.1% | 34.9%-43.4% | 60.9% |
| 30 | 34.4% | 26.7%-43.0% | 50.6% | 46.3%-54.9% | 76.6% |
| 40 | 41.4% | 33.2%-50.1% | 55.9% | 51.5%-60.1% | 76.6% |
| 50 | 46.9% | 38.4%-55.5% | 61.5% | 57.2%-65.6% | 80.5% |
| 60 | 67.2% | 58.7%-74.7% | 69.7% | 65.6%-73.5% | 86.7% |
| 70 | 40.6% | 32.5%-49.3% | 48.2% | 43.9%-52.6% | 71.9% |
| 80 | 64.1% | 55.5%-71.9% | 62.3% | 58.0%-66.4% | 81.2% |
| 90 | 56.2% | 47.6%-64.5% | 67.6% | 63.4%-71.5% | 87.5% |
| 100 | 60.2% | 51.5%-68.2% | 66.4% | 62.2%-70.4% | 84.4% |

- **greedy SR**: first 29.7% @10, peak 67.2% @60, final 60.2% @100, peak-to-final drop **7.0pp**, net first-to-final +30.5pp
- **sampled pass@1**: first 38.7% @10, peak 69.7% @60, final 66.4% @100, peak-to-final drop **3.3pp**, net first-to-final +27.7pp
- **sampled pass@4**: first 57.0% @10, peak 87.5% @90, final 84.4% @100, peak-to-final drop **3.1pp**, net first-to-final +27.3pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 1805 | 75 | 23.98 | 3.9% | 0.2% | 0.2% |
| 10 | sampled | 512 | 2410 | 102 | 23.65 | 0.4% | 0.0% | 0.1% |
| 20 | greedy | 128 | 1120 | 45 | 24.75 | 0.8% | 0.0% | 0.0% |
| 20 | sampled | 512 | 1439 | 61 | 23.64 | 0.4% | 0.0% | 0.1% |
| 30 | greedy | 128 | 1719 | 74 | 23.13 | 3.9% | 0.2% | 0.2% |
| 30 | sampled | 512 | 1603 | 74 | 21.53 | 0.4% | 0.0% | 0.6% |
| 40 | greedy | 128 | 1304 | 62 | 21.12 | 1.6% | 0.1% | 0.1% |
| 40 | sampled | 512 | 1272 | 65 | 19.70 | 0.4% | 0.0% | 0.2% |
| 50 | greedy | 128 | 1363 | 70 | 19.58 | 2.3% | 0.1% | 0.1% |
| 50 | sampled | 512 | 1297 | 72 | 18.09 | 0.4% | 0.0% | 0.2% |
| 60 | greedy | 128 | 1018 | 62 | 16.45 | 0.8% | 0.0% | 0.0% |
| 60 | sampled | 512 | 1331 | 77 | 17.29 | 0.2% | 0.0% | 1.2% |
| 70 | greedy | 128 | 874 | 40 | 21.75 | 0.0% | 0.0% | 0.0% |
| 70 | sampled | 512 | 1007 | 48 | 21.18 | 0.0% | 0.0% | 2.1% |
| 80 | greedy | 128 | 917 | 52 | 17.57 | 0.0% | 0.0% | 0.0% |
| 80 | sampled | 512 | 949 | 52 | 18.14 | 0.0% | 0.0% | 14.7% |
| 90 | greedy | 128 | 1468 | 80 | 18.26 | 3.9% | 0.2% | 0.2% |
| 90 | sampled | 512 | 1316 | 74 | 17.86 | 0.6% | 0.0% | 1.5% |
| 100 | greedy | 128 | 1325 | 76 | 17.36 | 2.3% | 0.1% | 0.1% |
| 100 | sampled | 512 | 1630 | 92 | 17.69 | 3.3% | 0.2% | 1.6% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 32.8% | 128 | 23.91 | 1.6% | 2026-08-15 10:14 |
| 20 | 28.1% | 128 | 24.50 | 0.8% | 2026-08-15 11:26 |
| 30 | 34.4% | 128 | 23.28 | 3.1% | 2026-08-15 12:41 |
| 40 | 43.0% | 128 | 21.04 | 0.8% | 2026-08-15 13:53 |
| 50 | 49.2% | 128 | 19.07 | 1.6% | 2026-08-15 15:06 |
| 60 | 68.0% | 128 | 16.52 | 1.6% | 2026-08-15 16:21 |
| 70 | 40.6% | 128 | 21.92 | 0.0% | 2026-08-15 17:32 |
| 80 | 62.5% | 128 | 17.40 | 0.8% | 2026-08-15 19:16 |
| 90 | 57.8% | 128 | 17.59 | 3.9% | 2026-08-15 20:18 |
| 100 | 57.8% | 128 | 17.48 | 1.6% | 2026-08-15 21:31 |

## Episode end reasons

- step 10 greedy: {'max_steps': 85, 'env_terminated': 38, 'length_truncation': 5}
- step 10 sampled: {'max_steps': 310, 'env_terminated': 198, 'malformed_action': 2, 'length_truncation': 2}
- step 20 greedy: {'max_steps': 92, 'env_terminated': 35, 'length_truncation': 1}
- step 20 sampled: {'max_steps': 310, 'env_terminated': 200, 'length_truncation': 2}
- step 30 greedy: {'max_steps': 79, 'env_terminated': 44, 'length_truncation': 5}
- step 30 sampled: {'env_terminated': 259, 'max_steps': 251, 'length_truncation': 2}
- step 40 greedy: {'max_steps': 73, 'env_terminated': 53, 'length_truncation': 2}
- step 40 sampled: {'env_terminated': 286, 'max_steps': 224, 'length_truncation': 2}
- step 50 greedy: {'max_steps': 65, 'env_terminated': 60, 'length_truncation': 3}
- step 50 sampled: {'env_terminated': 315, 'max_steps': 195, 'length_truncation': 2}
- step 60 greedy: {'env_terminated': 86, 'max_steps': 41, 'length_truncation': 1}
- step 60 sampled: {'env_terminated': 357, 'max_steps': 154, 'length_truncation': 1}
- step 70 greedy: {'max_steps': 76, 'env_terminated': 52}
- step 70 sampled: {'max_steps': 265, 'env_terminated': 247}
- step 80 greedy: {'env_terminated': 82, 'max_steps': 46}
- step 80 sampled: {'env_terminated': 319, 'max_steps': 193}
- step 90 greedy: {'env_terminated': 72, 'max_steps': 51, 'length_truncation': 5}
- step 90 sampled: {'env_terminated': 346, 'max_steps': 163, 'length_truncation': 3}
- step 100 greedy: {'env_terminated': 77, 'max_steps': 48, 'length_truncation': 3}
- step 100 sampled: {'env_terminated': 340, 'max_steps': 155, 'length_truncation': 17}

