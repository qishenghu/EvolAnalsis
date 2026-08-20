# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_catalystv2_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**INTERMEDIATE**

- greedy drop 7.8pp vs sampled drop 3.9pp fits neither frozen branch; treat the split as partial.
- peak-vs-final two-proportion z: greedy 1.28, sampled 1.26 (|z|>1.96 = significant at 5%).

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
| 10 | 43.8% | 35.5%-52.4% | 44.7% | 40.5%-49.1% | 59.4% |
| 20 | 41.4% | 33.2%-50.1% | 42.4% | 38.2%-46.7% | 55.5% |
| 30 | 35.2% | 27.4%-43.8% | 36.1% | 32.1%-40.4% | 44.5% |
| 40 | 36.7% | 28.9%-45.3% | 35.5% | 31.5%-39.8% | 43.8% |
| 50 | 39.8% | 31.8%-48.5% | 44.3% | 40.1%-48.7% | 60.9% |
| 60 | 38.3% | 30.3%-46.9% | 40.6% | 36.5%-44.9% | 53.9% |
| 70 | 36.7% | 28.9%-45.3% | 34.0% | 30.0%-38.2% | 50.8% |
| 80 | 33.6% | 26.0%-42.1% | 37.9% | 33.8%-42.2% | 57.8% |
| 87 | 35.9% | 28.1%-44.5% | 40.8% | 36.6%-45.1% | 61.7% |

- **greedy SR**: first 43.8% @10, peak 43.8% @10, final 35.9% @87, peak-to-final drop **7.8pp**, net first-to-final -7.8pp
- **sampled pass@1**: first 44.7% @10, peak 44.7% @10, final 40.8% @87, peak-to-final drop **3.9pp**, net first-to-final -3.9pp
- **sampled pass@4**: first 59.4% @10, peak 61.7% @87, final 61.7% @87, peak-to-final drop **0.0pp**, net first-to-final +2.3pp

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 2976 | 128 | 23.31 | 0.0% | 0.0% | 0.0% |
| 10 | sampled | 512 | 3581 | 158 | 22.62 | 0.0% | 0.0% | 4.7% |
| 20 | greedy | 128 | 3313 | 144 | 22.98 | 0.8% | 0.0% | 1.1% |
| 20 | sampled | 512 | 4318 | 193 | 22.42 | 0.0% | 0.0% | 36.9% |
| 30 | greedy | 128 | 2965 | 126 | 23.59 | 0.0% | 0.0% | 0.0% |
| 30 | sampled | 512 | 3430 | 145 | 23.60 | 0.0% | 0.0% | 16.4% |
| 40 | greedy | 128 | 4804 | 204 | 23.52 | 0.0% | 0.0% | 0.0% |
| 40 | sampled | 512 | 5295 | 224 | 23.67 | 0.0% | 0.0% | 4.8% |
| 50 | greedy | 128 | 2910 | 124 | 23.43 | 0.0% | 0.0% | 0.0% |
| 50 | sampled | 512 | 2937 | 131 | 22.42 | 0.0% | 0.0% | 3.8% |
| 60 | greedy | 128 | 2835 | 124 | 22.87 | 1.6% | 0.1% | 0.1% |
| 60 | sampled | 512 | 2873 | 124 | 23.17 | 0.0% | 0.0% | 6.1% |
| 70 | greedy | 128 | 2935 | 124 | 23.67 | 0.0% | 0.0% | 0.0% |
| 70 | sampled | 512 | 3216 | 131 | 24.46 | 0.0% | 0.0% | 4.3% |
| 80 | greedy | 128 | 3764 | 169 | 22.22 | 10.9% | 0.5% | 0.5% |
| 80 | sampled | 512 | 3138 | 133 | 23.55 | 0.0% | 0.0% | 34.0% |
| 87 | greedy | 128 | 5765 | 334 | 17.27 | 35.2% | 2.0% | 52.1% |
| 87 | sampled | 512 | 3810 | 168 | 22.65 | 0.0% | 0.0% | 83.6% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 38.3% | 128 | 23.72 | 0.8% | 2026-08-13 17:59 |
| 20 | 37.5% | 128 | 23.09 | 1.6% | 2026-08-13 19:12 |
| 30 | 33.6% | 128 | 23.56 | 0.8% | 2026-08-13 20:27 |
| 40 | 35.9% | 128 | 23.73 | 0.8% | 2026-08-13 21:47 |
| 50 | 46.9% | 128 | 22.41 | 0.0% | 2026-08-13 23:09 |
| 60 | 39.8% | 128 | 22.79 | 0.8% | 2026-08-14 00:30 |

## Episode end reasons

- step 10 greedy: {'max_steps': 72, 'env_terminated': 56}
- step 10 sampled: {'max_steps': 283, 'env_terminated': 229}
- step 20 greedy: {'max_steps': 74, 'env_terminated': 53, 'length_truncation': 1}
- step 20 sampled: {'max_steps': 294, 'env_terminated': 217, 'malformed_action': 1}
- step 30 greedy: {'max_steps': 83, 'env_terminated': 45}
- step 30 sampled: {'max_steps': 326, 'env_terminated': 185, 'malformed_action': 1}
- step 40 greedy: {'max_steps': 81, 'env_terminated': 47}
- step 40 sampled: {'max_steps': 328, 'env_terminated': 182, 'malformed_action': 2}
- step 50 greedy: {'max_steps': 77, 'env_terminated': 51}
- step 50 sampled: {'max_steps': 285, 'env_terminated': 227}
- step 60 greedy: {'max_steps': 77, 'env_terminated': 49, 'length_truncation': 2}
- step 60 sampled: {'max_steps': 304, 'env_terminated': 208}
- step 70 greedy: {'max_steps': 81, 'env_terminated': 47}
- step 70 sampled: {'max_steps': 333, 'env_terminated': 174, 'malformed_action': 5}
- step 80 greedy: {'max_steps': 71, 'env_terminated': 43, 'length_truncation': 14}
- step 80 sampled: {'max_steps': 317, 'env_terminated': 194, 'malformed_action': 1}
- step 87 greedy: {'env_terminated': 46, 'length_truncation': 45, 'max_steps': 37}
- step 87 sampled: {'max_steps': 291, 'env_terminated': 209, 'malformed_action': 12}

