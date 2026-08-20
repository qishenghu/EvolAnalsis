# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_grpo10_af_s1` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 13.3pp vs sampled drop 7.0pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 2.13, sampled 2.26 (|z|>1.96 = significant at 5%).

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
| 10 | 39.1% | 31.0%-47.7% | 38.9% | 34.7%-43.2% | 61.7% |
| 20 | 32.8% | 25.3%-41.3% | 37.3% | 33.2%-41.6% | 63.3% |
| 30 | 29.7% | 22.5%-38.1% | 32.0% | 28.1%-36.2% | 62.5% |
| 40 | 36.7% | 28.9%-45.3% | 43.2% | 38.9%-47.5% | 74.2% |
| 50 | 40.6% | 32.5%-49.3% | 45.1% | 40.9%-49.4% | 74.2% |
| 60 | 47.7% | 39.2%-56.3% | 53.5% | 49.2%-57.8% | 76.6% |
| 70 | 32.0% | 24.6%-40.5% | 34.2% | 30.2%-38.4% | 57.0% |
| 80 | 53.9% | 45.3%-62.3% | 58.4% | 54.1%-62.6% | 78.9% |
| 90 | 58.6% | 49.9%-66.8% | 57.0% | 52.7%-61.3% | 74.2% |
| 100 | 45.3% | 37.0%-53.9% | 51.4% | 47.0%-55.7% | 65.6% |

- **greedy SR**: first 39.1% @10, peak 58.6% @90, final 45.3% @100, peak-to-final drop **13.3pp**, net first-to-final +6.2pp
- **sampled pass@1**: first 38.9% @10, peak 58.4% @80, final 51.4% @100, peak-to-final drop **7.0pp**, net first-to-final +12.5pp
- **sampled pass@4**: first 61.7% @10, peak 78.9% @80, final 65.6% @100, peak-to-final drop **13.3pp**, net first-to-final +3.9pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 4478 | 191 | 23.43 | 7.0% | 0.3% | 0.3% |
| 10 | sampled | 512 | 5742 | 240 | 23.96 | 0.0% | 0.0% | 0.7% |
| 20 | greedy | 128 | 3820 | 151 | 25.38 | 0.0% | 0.0% | 0.0% |
| 20 | sampled | 512 | 4624 | 186 | 24.84 | 0.0% | 0.0% | 0.4% |
| 30 | greedy | 128 | 3936 | 155 | 25.47 | 0.0% | 0.0% | 0.0% |
| 30 | sampled | 512 | 4304 | 169 | 25.50 | 0.0% | 0.0% | 3.4% |
| 40 | greedy | 128 | 2941 | 125 | 23.51 | 2.3% | 0.1% | 0.1% |
| 40 | sampled | 512 | 3497 | 148 | 23.57 | 0.6% | 0.0% | 1.0% |
| 50 | greedy | 128 | 940 | 42 | 22.63 | 0.0% | 0.0% | 0.0% |
| 50 | sampled | 512 | 1008 | 45 | 22.45 | 0.0% | 0.0% | 0.2% |
| 60 | greedy | 128 | 879 | 42 | 20.88 | 0.0% | 0.0% | 0.0% |
| 60 | sampled | 512 | 978 | 47 | 21.00 | 0.0% | 0.0% | 0.1% |
| 70 | greedy | 128 | 1134 | 49 | 23.21 | 0.8% | 0.0% | 4.7% |
| 70 | sampled | 512 | 1004 | 41 | 24.21 | 0.0% | 0.0% | 25.3% |
| 80 | greedy | 128 | 673 | 35 | 19.25 | 0.0% | 0.0% | 0.0% |
| 80 | sampled | 512 | 767 | 38 | 20.14 | 0.0% | 0.0% | 0.1% |
| 90 | greedy | 128 | 567 | 31 | 18.58 | 0.0% | 0.0% | 0.0% |
| 90 | sampled | 512 | 637 | 32 | 20.10 | 0.0% | 0.0% | 0.1% |
| 100 | greedy | 128 | 571 | 27 | 21.01 | 0.0% | 0.0% | 0.0% |
| 100 | sampled | 512 | 606 | 29 | 20.57 | 0.0% | 0.0% | 0.1% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 35.2% | 128 | 23.26 | 10.9% | 2026-08-17 14:18 |
| 20 | 35.2% | 128 | 25.06 | 0.8% | 2026-08-17 15:37 |
| 30 | 35.2% | 128 | 25.22 | 0.0% | 2026-08-17 16:58 |
| 40 | 41.4% | 128 | 23.33 | 0.8% | 2026-08-17 18:16 |
| 50 | 38.3% | 128 | 22.89 | 0.0% | 2026-08-17 19:12 |
| 60 | 45.3% | 128 | 21.19 | 0.0% | 2026-08-17 20:31 |
| 70 | 30.5% | 128 | 23.72 | 0.8% | 2026-08-17 21:44 |
| 80 | 57.8% | 128 | 18.73 | 0.0% | 2026-08-17 23:01 |
| 90 | 56.2% | 128 | 19.23 | 0.0% | 2026-08-18 00:07 |
| 100 | 43.8% | 128 | 21.22 | 0.0% | 2026-08-18 01:17 |

## Episode end reasons

- step 10 greedy: {'max_steps': 69, 'env_terminated': 50, 'length_truncation': 9}
- step 10 sampled: {'max_steps': 308, 'env_terminated': 199, 'malformed_action': 5}
- step 20 greedy: {'max_steps': 86, 'env_terminated': 42}
- step 20 sampled: {'max_steps': 321, 'env_terminated': 191}
- step 30 greedy: {'max_steps': 90, 'env_terminated': 38}
- step 30 sampled: {'max_steps': 346, 'env_terminated': 164, 'malformed_action': 2}
- step 40 greedy: {'max_steps': 78, 'env_terminated': 47, 'length_truncation': 3}
- step 40 sampled: {'max_steps': 287, 'env_terminated': 221, 'length_truncation': 3, 'malformed_action': 1}
- step 50 greedy: {'max_steps': 76, 'env_terminated': 52}
- step 50 sampled: {'max_steps': 281, 'env_terminated': 231}
- step 60 greedy: {'max_steps': 67, 'env_terminated': 61}
- step 60 sampled: {'env_terminated': 274, 'max_steps': 238}
- step 70 greedy: {'max_steps': 86, 'env_terminated': 41, 'length_truncation': 1}
- step 70 sampled: {'max_steps': 335, 'env_terminated': 175, 'malformed_action': 2}
- step 80 greedy: {'env_terminated': 69, 'max_steps': 59}
- step 80 sampled: {'env_terminated': 299, 'max_steps': 213}
- step 90 greedy: {'env_terminated': 75, 'max_steps': 53}
- step 90 sampled: {'env_terminated': 292, 'max_steps': 220}
- step 100 greedy: {'max_steps': 70, 'env_terminated': 58}
- step 100 sampled: {'env_terminated': 263, 'max_steps': 249}

