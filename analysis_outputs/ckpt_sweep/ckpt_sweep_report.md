# Checkpoint sweep — greedy vs sampled decoding on the held-out 128

Experiment `p0_catalyst_af_s0` · ALFWorld val prefix (game indices 2420-2547, the exact 128 tasks and order the trainer evaluates, `ordered_newline_sha256=d90efe607c...42915`).

| decoder | sampling | rollouts/task | episodes/checkpoint |
|---|---|---|---|
| greedy  | temperature 0, top_p 1.0   | 1 | 128 |
| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |

Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.

## Verdict

**DECODING ARTIFACT**

- greedy falls 14.1pp from its peak while sampled pass@1 falls only 7.0pp (<= max(0.5*14.1, 5)). The policy distribution holds up; the argmax path does not.
- peak-vs-final two-proportion z: greedy 2.38, sampled 2.27 (|z|>1.96 = significant at 5%).

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
| 30 | 32.8% | 25.3%-41.3% | 37.7% | 33.6%-42.0% | 57.8% |
| 40 | 35.2% | 27.4%-43.8% | 45.5% | 41.2%-49.8% | 69.5% |
| 50 | 34.4% | 26.7%-43.0% | 41.0% | 36.8%-45.3% | 64.1% |
| 60 | 40.6% | 32.5%-49.3% | 46.7% | 42.4%-51.0% | 64.1% |
| 70 | 26.6% | 19.7%-34.8% | 39.6% | 35.5%-43.9% | 57.0% |

- **greedy SR**: first 32.8% @30, peak 40.6% @60, final 26.6% @70, peak-to-final drop **14.1pp**, net first-to-final -6.2pp
- **sampled pass@1**: first 37.7% @30, peak 46.7% @60, final 39.6% @70, peak-to-final drop **7.0pp**, net first-to-final +2.0pp
- **sampled pass@4**: first 57.8% @30, peak 69.5% @40, final 57.0% @70, peak-to-final drop **12.5pp**, net first-to-final -0.8pp

## Supplementary checkpoints (different run instance — not part of the ruling)

| step | greedy SR | sampled pass@1 | sampled pass@4 |
|---:|---:|---:|---:|
| 10 | 23.4% | 33.6% | 47.7% |
| 20 | 29.7% | 41.2% | 63.3% |

## Length and truncation (cross-validation)

| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 10 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 20 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 20 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 30 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 30 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 40 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 40 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 50 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 50 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 60 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 60 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 70 | greedy | 128 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |
| 70 | sampled | 512 | 0 | 0 | 0.00 | 0.0% | 0.0% | 0.0% |

## Trainer's own in-run greedy val log (same 128 tasks)

Cross-check for the sweep's greedy column. Note these files are overwritten by every restart of the experiment, so their provenance follows the file mtime.

| step | SR | n | mean decisions | trunc rate | log mtime |
|---:|---:|---:|---:|---:|---|
| 10 | 24.2% | 128 | 25.68 | 1.6% | 2026-08-11 22:56 |
| 20 | 30.5% | 128 | 24.67 | 3.9% | 2026-08-12 00:10 |
| 30 | 29.7% | 128 | 24.83 | 0.0% | 2026-08-11 15:59 |
| 40 | 34.4% | 128 | 23.98 | 0.0% | 2026-08-11 17:11 |
| 50 | 31.2% | 128 | 24.48 | 0.0% | 2026-08-11 18:30 |
| 60 | 41.4% | 128 | 22.30 | 0.0% | 2026-08-11 19:46 |
| 70 | 28.1% | 128 | 26.78 | 0.0% | 2026-08-11 20:57 |

## Episode end reasons

- step 10 greedy: {'?': 128}
- step 10 sampled: {'?': 512}
- step 20 greedy: {'?': 128}
- step 20 sampled: {'?': 512}
- step 30 greedy: {'?': 128}
- step 30 sampled: {'?': 512}
- step 40 greedy: {'?': 128}
- step 40 sampled: {'?': 512}
- step 50 greedy: {'?': 128}
- step 50 sampled: {'?': 512}
- step 60 greedy: {'?': 128}
- step 60 sampled: {'?': 512}
- step 70 greedy: {'?': 128}
- step 70 sampled: {'?': 512}

