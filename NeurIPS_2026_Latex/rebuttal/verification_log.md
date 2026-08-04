# Independent verification of numbers quoted in the responses

Everything below was re-derived by the lead agent from raw artifacts, independently of the agents
that drafted the responses. This exists so that no number reaches a confidence-4 reviewer without
having been checked twice.

## Verified exactly

| claim | quoted | recomputed | source |
|---|---|---|---|
| DR3 applies one correction, not $\hat w\cdot\rho_t$ | — | confirmed | `het_actor.py:1501-1507,1544`; `het_core_algos.py:1968-1970` |
| $\hat w$ is the α-relative ratio, bounded by $1/(1-\alpha)$ | — | confirmed; `use_relative_ratio` defaults True, never overridden | `dr3_ratio.py:846-852`; `grep -rn use_relative_ratio config/` → nothing |
| −BC ablation | 34.0% / 16.5% | 34.0% / 16.5% | `experiments/{alfworld,webshop}/*_duet_minus_bc/validation_log/100.jsonl` |
| −SC ablation | 31.0% / 1.0% | 31.0% / 1.0% | `*_duet_minus_sc/validation_log/100.jsonl` |
| −DR3 ablation | 47.5% / 9.5% | 47.5% / 9.5% | `*_duet_minus_dr3/…` |
| −baseline-sep ablation | 0.0% / 0.0% | 0.0% / 0.0% | `*_duet_minus_baseline_sep/…` |
| −DR3 three seeds | 47.5 / 38.0 / 41.0, sd 4.9 | identical | seeds 2026/2025/2027 |
| 7B ALFWorld | DUET 86.5, GRPO 85.0, LUFFY 82.5 | identical | `experiments/alfworld/alfworld_7b_*/validation_log/100.jsonl` |
| teacher gradient share, 1.5B vs 7B | 0.241 vs 0.079 (3.05×) | 0.241 vs 0.079 | `logs/alfworld_{qwen1.5b_duet_v39c_postfix,7b_duet}.log` |
| raw teacher rollouts, ALFWorld | 24,200 | 24,200 | `alfworld_qwen72b.jsonl` line count |
| success-filter keep rate | 80.6% | 80.6% (19,497/24,200) | ditto |
| cache diversity, ALFWorld | 4.4 distinct of 8.5; 36.5% identical pairs; 7.6% single-path | 4.4 / 8.5; 36.5% (28,849/78,974); 7.6% | parsed `<action>` sequences per task |
| realized teacher rollouts per prompt | 0.978 (AF) / 0.858 (WS) | 0.977 / 0.857 from the measured `diag/teacher_sample_ratio` distribution | `logs/*.log` |
| Llama-3.2-3B cross-family | LUFFY 19.5% vs GRPO 5.5% @50 | identical | `experiments/alfworld/alfworld_llama3b_*/validation_log/50.jsonl` |
| SFT+GRPO is compute-matched | 50 SFT + 50 GRPO = 100 steps | confirmed (`max_train_tasks` 400, batch 8, 1 epoch, both stages) | `*_sft.yaml`, `*_sft_rl.yaml` |
| WebShop teacher cache is synthesized from gold | — | confirmed: `is_synthesized: True`, `source: webshop_gold` | cache metadata |
| Table 1: 5 of 8 cited cells | — | match exactly | see `../data/number_audit_2026_07_26.md` |

## Corrected before submission

| claim as drafted | problem | corrected to |
|---|---|---|
| "the weight on teacher samples settles near 0.47" (7B) | Not supported. `dr3/w_off_mean` over 90 steps: mean 0.611, median 0.577, min 0.228, **last five 0.908 / 0.813 / 0.820 / 1.082 / 0.701** — it does not settle at 0.47. | "starts at ≈1.0 and drops to a run median of 0.58 (min 0.23)". The fade-out argument rests on `teacher_gradient_share` (0.241 → 0.079), which is verified. |
| "diag/teacher_sample_ratio is exactly 0.125 at every step" (my own first draft of the cache-stats note) | A three-line `tail` happened to catch three 0.125 steps. The real distribution is 0.125 ×83, 0.109 ×16, 0.094 ×1. | "0.977 teacher trajectories per prompt on ALFWorld; the fallback to an all-on-policy group fires on ~8% of prompts". |
| Planned Eq. 9 defence ("$\hat w \approx \pi_{\theta_{old}}/\pi_\beta$, so $\hat w\rho_t$ telescopes") | **Wrong.** Three independent adversarial audits refuted it; the code never forms that product. | Concede the notation; state the substitution actually implemented. See `evidence_eq9_dr3.md`. |

## Not verifiable on this machine — do not cite as verified

- The **3B column of Table 1** and the runs behind it (see `paper_corrections.md` §C0). LUFFY-AF
  (61.5%) and GRPO-WS (2.0%) match local logs; DUET/CHORD/SFT+GRPO 3B do not, and
  `analysis_reports/3b_master_experiment_table.md` records several of them as coming from a 4×H100
  machine with "no raw" logs here.
- **WebShop 7B DR3 diagnostics** — no `logs/webshop_7b_*.log` on this machine. Every DR3 fade-out
  number quoted is ALFWorld-only and must be described as such.
- A **dense held-out curve for SFT+GRPO**: `test_freq: 50` means only two validation points exist
  per run. What is reconstructible is the per-step *training* success rate; the responses say so.

## Result-driven corrections (added as runs land)

**2026-07-26 03:41 — WebShop seed 2025 came in at 3.5% strict (paper seed: 35.5%).** Verified that
the two configs differ only in `experiment_name`, `workspace_id`, and `seed`. Root-caused to a late
phase transition intersecting a threshold metric, not to a broken run: both training curves are
still rising at step 100 with seed 2025 ~10 steps behind, and 32.5% of its evaluation episodes sit
in the 0.75–0.90 band. Full analysis in `../data/webshop_seed_sensitivity.md`; the y9x6 response
now reports this spread rather than the favourable seed, and a 150-step diagnostic is queued.

**2026-07-26 — three of the four training curves quoted to bDeY were wrong.** Recomputed
`critic/success_onpolicy/mean` in 10-step blocks from both the training logs and, independently,
from the saved per-step trajectory files (teacher rollouts excluded); the two agree to ±0.005.

| curve | as drafted | verified |
|---|---|---|
| SFT stage, ALFWorld | 0.016, 0.007, 0.087, 0.264, 0.285 | ✓ same |
| SFT+GRPO stage, ALFWorld | …, 0.474, 0.308 | ✓ same (0.473) |
| DUET, ALFWorld | 0.065, 0.049, 0.072, 0.188, 0.322, 0.418, 0.431, 0.478, 0.432, 0.427 | ✗ → 0.011, 0.006, 0.019, 0.128, 0.272, 0.370, 0.397, 0.450, 0.408, 0.392 |
| SFT+GRPO, WebShop | 0.036, 0.152, 0.385, 0.457, 0.488 \| 0.508, 0.490, 0.590, 0.638, 0.609 | ✗ → 0.014, 0.029, 0.012, 0.012, 0.049 \| 0.030, 0.064, 0.062, 0.125, 0.136 |
| DUET, WebShop | 0.074, 0.241, 0.476, 0.535, 0.522, 0.546, 0.565, 0.607, 0.640, 0.654 | ✗ → 0.004, 0.019, 0.004, 0.026, 0.007, 0.028, 0.007, 0.046, 0.084, 0.125 |

The WebShop figures as drafted are in the range of a *reward* metric, not a success rate, so the
two environments were being compared on different quantities. The response now quotes the verified
success-rate curves for all four, and `critic/success_onpolicy/mean` is the only success metric
present in these logs (checked by enumerating every `critic/success*` key).

**Paired significance tests — all five verified.** Exact two-sided McNemar on the same 200-task
split, recomputed independently:

| comparison | quoted | recomputed | discordant |
|---|---|---|---|
| AF DUET vs SFT+GRPO | p = 2.2e-5 | 2.17e-5 | 51 v 16 |
| WS DUET vs SFT+GRPO | p = 7.9e-8 | 7.88e-8 | 40 v 5 |
| AF −BC vs CHORD | p = 0.076 | 0.0759 | 34 v 20 |
| WS −BC vs CHORD | p = 0.031 | 0.0309 | 14 v 4 |
| AF −SC vs CHORD | p = 0.31 | 0.312 | 28 v 20 |

**Live corroboration of the soft-matching mechanism.** In the running `obsnoise_soft` job the
State Channel initialises as `soft mode: TF-IDF profiles built for 2348 tasks (threshold=0.6,
obs_noise_p=0.3)` and logs `state_channel/coverage_mean = 0.561` at the first step — the same range
as the paper's clean-hash run (0.614 / 0.562 / 0.328 over its first three steps). So the offline
result (exact matching collapses under this noise, soft matching does not) reproduces inside the
training loop, independently of what the final success rate turns out to be.

**2026-07-26 (later) — the "seed effect" framing was itself wrong and is now corrected.** Three
hypotheses were tested against the artifacts:

| hypothesis | verdict | decisive evidence |
|---|---|---|
| WebShop environment was faulty during the seed-2025 run | **ruled out** | the 7 unparseable validation episodes are policy-side repetition loops (`search[... small small ...]`) each scored −0.1 by the grader; failure rate 11.0% vs 5.0% (seed 2026) and 58.5% (GRPO); episode lengths normal |
| the seed changed which teacher demonstrations match | **ruled out** | teacher coverage of each run's *actual* 800 tasks: 85.8% vs 83.5%, 3.97 vs 3.82 demos/task, realised 0.857 vs 0.834 teacher rollouts per prompt |
| the seed changed the training curriculum | **confirmed** | `ae_ray_trainer.py` shuffles the train split by `data.seed` and keeps the first `max_train_tasks`; recovered from saved rollouts the two runs share **89 of 800** tasks (ALFWorld replicates share ~265 of 800) |

Consequences, all verified from artifacts rather than assumed:
- Validation is seed-independent (`shuffle=False`, no seed passed), so both runs are scored on the
  same fixed 200 tasks.
- Table 1 is unaffected: all 1.5B methods use `seed: 2026`; CHORD and GRPO trained on **exactly**
  the same 800 tasks as DUET (100% overlap from the saved rollouts), SFT+GRPO's 400 are a subset.
- `data.task_seed` was added to decouple the task draw from the run seed (defaults to `data.seed`,
  so existing configs are unchanged); re-simulating the selection reproduces the observed overlap
  (92/800 unpinned vs 89 measured) and gives 800/800 when pinned.

The earlier claim in this workspace that the two runs were "identical configuration, only the seed
differs" is withdrawn: it is true of the YAML and false of what the runs actually trained on.

**2026-07-26 (later still) — the seed framing was wrong a second time; the real finding is worse.**
A four-angle forensic audit (3/3 adversarial verifiers refuted the leading explanations) turned up a
run already on disk that settles it:

| run | seed | training tasks | val@100 strict |
|---|---|---|---|
| `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` (paper) | 2026 | the 800 | 35.5% |
| `ws_1_5b_swC02_da` | 2026 | **identical 800, identical per-step order (verified 100/100 steps)** | **1.0%** |

I verified this independently of the agents: the YAML diff is `experiment_name`, `workspace_id`, and
`rollout.gpu_memory_utilization` 0.75 → 0.6 — nothing that can change a gradient — and the per-step
task-id sets match at all 100 steps (8 tasks per step, sets equal). Across 66 1.5B-WebShop runs on
disk the paper run ranks 1; this replica ranks 60; the median is 3.0%.

So the earlier corrected framing ("a genuine seed effect amplified by a threshold metric") is also
withdrawn. The accurate statement is that the 1.5B-WebShop cell is dominated by run-to-run
nondeterminism at a phase transition that the 100-step budget cuts through, and the submitted number
is the upper tail of that distribution. `DECISION_webshop_1p5b_cell.md` sets out the options; the
deciding replicates are running on a pinned curriculum.

**2026-07-26 — `ws_1_5b_swC02_da` withdrawn from the evidence.** The audit used it as a same-seed,
same-curriculum replica scoring 1.0%; the authors report the run was faulty when it executed, so it
is excluded and no conclusion rests on it. The technical observations about it remain true (same
seed, per-step task sets identical at 100/100, only `gpu_memory_utilization` differing) but they no
longer support an inference about reproducibility. The open question reverts to the plain one, which
the pinned-curriculum replicates now running will answer.

**2026-07-26 — `task_seed` verified in a live run, and a metric trap found while doing it.**
The first pinned replicate (`webshop_qwen1.5b_duet_a100_seed2027`) was checked against the paper run
as soon as it had saved a few steps: the executed config snapshot carries `seed=2027 task_seed=2026`,
and every task it has trained on so far is drawn from the reference's 800 (40/40 shared). Per-step
*order* differs, which is intended — the dataloader generator is seeded by `data.seed`.

While writing that check, two ways of over-reading the saved trajectories surfaced; both would have
inflated an on-policy success curve several-fold:

- the per-row `success` flag is **not** strict success. On WebShop it is true whenever the episode
  completed a purchase, so a row with `reward.outcome` = 0.43 still has `success: true`. Strict
  success is `reward.outcome >= 1.0`, which is what val@100 measures.
- teacher rollouts are flagged in `diag.is_teacher`, not `metadata.is_teacher`. Filtering on the
  latter alone leaves ~1 row in 8 that scored 1.0 by construction.

With both fixed, the trajectory-derived curve for the paper run (0.002, 0.016, 0.003, 0.027, 0.005,
0.026, 0.004, 0.044, 0.079, 0.114) matches the logged `critic/success_onpolicy/mean` (0.004, 0.019,
0.004, 0.026, 0.007, 0.028, 0.007, 0.046, 0.084, 0.125). This independently confirms the WebShop
training curves quoted in the response to bDeY, which were taken from the log.

**2026-07-26 — mechanism-level health check applied retrospectively to the low WebShop run.**
Before treating any replicate as "broken and worth restarting", we needed a detector that separates
a faulty run from a healthy one that landed low. `scripts/monitor_run_health.py` compares a run
against the reference at matched steps on the signals that say whether the algorithm is working.
Applied to the two completed WebShop runs at step 100:

| signal | paper run (35.5%) | seed-2025 run (3.5%) |
|---|---|---|
| `dr3/disc_acc` | 0.991 | 0.983 |
| `duet/teacher_gradient_share` | 0.089 | 0.095 |
| `chord/mu` | 0.105 | 0.108 |
| `state_channel/coverage_mean` | 0.863 | 0.825 |
| `diag/teacher_sample_ratio` | 0.108 | 0.103 |
| degenerate-repetition rate | 0.4% | 0.3% |
| `actor/entropy_loss` | 0.526 | 0.504 |
| **`actor/kl_loss`** | **0.671** | **0.943** |
| on-policy strict success | 0.125 | 0.084 |

**All four DUET mechanisms were working in the low run** — there was no fault to catch, so a
"kill and restart" would have been discarding a valid measurement. The one real difference is KL:
by 20-step block the paper run goes 0.073, 0.285, 0.470, 0.507, 0.741 while the low run goes 0.056,
0.298, 0.645, 0.896, 0.901 — identical for the first 40 steps, then the low run keeps drifting away
from the reference model where the paper run plateaus near 0.5.

That is now a **watch** signal in the monitor, not a kill criterion: drift is informative, but a
replicate must not be altered mid-flight or it stops being a replicate. If the pinned-curriculum
replicates reproduce the association (higher KL drift → lower landing), it is a mechanistic result
worth reporting, and it points at `kl_loss_coef` as the stabiliser for future work.

Kill criteria in force for the remaining runs, all mechanism faults with an identifiable cause and
none of them score-based: NaN/inf in loss or gradient, entropy collapse below 20% of reference,
KL above 5.0, degenerate repetition above 15% of rollouts, discriminator pinned at chance past
step 25, teacher mixing off, State-Channel coverage below 5%, or any infrastructure error in the
run log.
