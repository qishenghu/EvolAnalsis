# NeurIPS 2026 rebuttal workspace — DUET (Submission 32282)

Reviews: `neurips_reviews/review_{UyKJ,bDeY,y9x6}.txt` —
**UyKJ 4** (borderline accept), **bDeY 3** (borderline reject, confidence 4), **y9x6 3** (borderline reject).

Everything here was produced on 2026-07-26. Numbers in these files were recomputed from raw
artifacts (validation JSONL, training logs, teacher caches, source code) — not copied from earlier
summaries. Where something could not be verified it is listed under "gaps" in the relevant file.

## Responses (the deliverable)

| file | reviewer |
|---|---|
| `response_UyKJ.md` | UyKJ — generalization breadth, Pick-Two anomaly, discriminator confound |
| `response_bDeY.md` | bDeY — "principled" claim, Eq. 9, SC generality, three factual questions |
| `response_y9x6.md` | y9x6 — noisy/open-ended transfer, seeds, cache size/quality, BC attribution |

## Evidence packs (inputs to the responses; each cites file:line or log paths)

| file | what it establishes |
|---|---|
| `evidence_eq9_dr3.md` | **Our original planned answer to bDeY was wrong.** Code audit of what DR3 actually applies |
| `evidence_sc_generality.md` | What we can and cannot claim about DUET-without-SC vs CHORD |
| `evidence_teacher_quality.md` | Teacher-quality evidence via the measured teacher–student gap (1.5B vs 7B) |
| `evidence_pick_two.md` | Pick-Two is n=45, McNemar p=0.238, and the anomaly is 3B-only |
| `evidence_factual_bdey.md` | Group composition, teacher cache stats, SFT protocol, Table 1 audit |
| `paper_corrections.md` | The revision changes we commit to (Eq. 8, Eq. 9, Table 1, protocol) |
| `../data/webshop_seed_sensitivity.md` | The three hypotheses tested for the WebShop spread, and which survived |
| `forensics_VERDICT.md` | Full forensic audit: the 35.5% does not reproduce; a same-seed same-task replica scores 1.0% |
| `DECISION_webshop_1p5b_cell.md` | **Authors' decision** on what to claim for the 1.5B-WebShop cell |
| `verification_log.md` | Independent re-derivation of every number quoted in the responses, plus the three claims that were corrected before submission |

## Supporting measurements (`../data/`)

| file | result |
|---|---|
| `dr3_confound_summary.md` | Discriminator is blind to success: P(student) 0.861 (succ) vs 0.869 (fail) vs 0.280 (teacher), late training |
| `sc_noise_robustness.md` | Offline: under 30% observation noise, exact matching retains 6.6% of the SC progress signal; soft matching retains 101% |
| `sc_matching_end_to_end.md` | **End-to-end: DUET under 30% observation noise with soft matching scores 54.5% vs the paper's clean exact-matching 47.5%** |
| `soft_match_calibration.md` | Threshold sweep behind the soft matcher (θ=0.6 chosen), with the cross-task false-positive caveat |
| `number_audit_2026_07_26.md` | Every cited Table 1 cell recomputed; 5 match exactly, 3 WebShop cells differ by ≤1.0pp (headline Δ unchanged) |
| `a100_rebuttal_results.md` | Live results from the A100 queue |
| `h200_rebuttal_results.md` | Live results from the H200 queue (separate machine) |
| `rebuttal_tables.md` | Auto-generated multi-seed tables (`scripts/aggregate_rebuttal_results.py`) |
| `super_additivity.md` | ⚠ pre-submission analysis — **not citable**: its synergy arithmetic rests on the 36.0% WebShop cell that does not reproduce |

## ⚠️ Two items need the authors before this response is filed

**1. The 1.5B-WebShop cell needs its replicates before we can state a number.** The submitted
35.5% and the seed-2025 replicate (3.5%) differ by more than this cell can absorb, but the two are
not like-for-like: `data.seed` also selects which 800 of ~6.7k training tasks are used, so seed 2025
trained on a largely different curriculum (89 tasks shared). Like-for-like replicates on the paper's
own curriculum (`task_seed: 2026`) are running for DUET (seeds 2025/2027/2028) and for SFT+GRPO
(seeds 2025/2027). `DECISION_webshop_1p5b_cell.md` sets out what to claim under each outcome.
Note: the earlier audit leaned on `ws_1_5b_swC02_da` (1.0%) as a same-curriculum replica; the
authors report that run was faulty, so it is excluded from the evidence.

**2. Table 1's 3B column** cannot be recomputed on this machine, and the Appendix-F task-type figure
is built from *different runs* than Table 1 reports (figure DUET = `alfworld_3b_duet_0329`, overall
69.5%; Table 1 DUET = `alfworld_qwen3b_duet_v39b`, 77.5%, no local validation log). Our Pick-Two
answer to UyKJ quotes that figure, so a reviewer cross-referencing it against the main table would
find the mismatch. Full detail and the required actions are in `paper_corrections.md` §C0.

## The three load-bearing findings

1. **bDeY's Eq. 9 objection is half right, and conceding it is our strongest move.** The code never
   forms $\hat w \cdot \rho_t$. For teacher samples it *replaces* the behaviour log-probability
   (`old_lp_new[teacher] = log_prob.detach() − log ŵ`, `het_actor.py:1507`), so the single ratio in
   the clipped surrogate evaluates to $\hat w$ — exactly one correction, no double counting. Eq. 9's
   notation misdescribes our own implementation, and Eq. 8's $D/(1-D)$ is an intermediate: the
   applied weight is the α-relative ratio, bounded by $1/(1-\alpha)\approx 1.13$, so DR3 can only
   *down-weight* teacher samples. See `evidence_eq9_dr3.md` and `paper_corrections.md`.

2. **The discriminator is not a success detector.** Conditioning on success does not weaken it
   (`D_succ` 90–98% across training slices), and successful vs failed student rollouts receive
   near-identical P(student) (0.861 vs 0.869 late in training) while teacher trajectories sit at
   0.280. Answers UyKJ Q3 and y9x6 W2.

3. **The State Channel's dependence on exact matching is a property of the lookup operator, not of
   the method — now shown end to end.** Under 30% observation noise, which offline leaves exact
   matching with 6.6% of the clean progress signal, swapping in a dependency-free TF-IDF matcher
   over the *same* progress map gives **54.5%** strict success against the paper's clean
   exact-matching 47.5% (state coverage 0.654 vs 0.590). Read the 7-point gap as "at least as
   good": our comparable three-seed spread has sd 4.9pp. The two controls — same noise with exact
   matching, and soft matching on clean observations — are queued.

## Experiment status

A100 (GPUs 0,1,2,4 — paper-identical hardware), `run_a100_rebuttal_queue.sh`:

| phase | run | answers |
|---|---|---|
| M1 ✅ | WebShop DUET seed 2025 (own curriculum) → **3.5%** | y9x6 multi-seed; triggered the investigation below |
| M2 ✅ | ALFWorld SC obs-noise + **soft** matching → **54.5%** (paper clean-hash: 47.5%) | y9x6 W1 / bDeY W2 |
| M3 ▶ | WebShop DUET seed 2027, **`task_seed: 2026`** | like-for-like replicate on the paper's curriculum |
| M4 | ALFWorld SC obs-noise + **exact** matching | premise side of the same experiment |
| E1–E6 | WebShop SFT rerun (+curve), soft-clean control, SFT+GRPO seeds 2025/2027 (**`task_seed: 2026`**), teacher-14B, teacher-32B | bDeY Q2; baseline replicates; UyKJ Q1 |
| F0a/F0b | WebShop DUET **fixedtask** seeds 2025, 2028 | the two remaining DUET replicates on the paper's curriculum |
| F1 | **Shuffled progress map** — matched-magnitude shaping control | y9x6 "compare against simpler reward-shaping baselines" |
| F2 | **DUET on a Llama-3.2-3B student** (unchanged 72B Qwen teacher) | UyKJ "non-Qwen student" |
| F3 | WebShop seed 2025 at a **150-step** budget (own curriculum, matching M1) | is the take-off later, or absent? |

After M3, E3, E4, F0a and F0b land we will have five DUET points and three SFT+GRPO points on the
*same* 800-task curriculum, which is what decides `DECISION_webshop_1p5b_cell.md`.

`data.seed` in this codebase both seeds the run and selects which `max_train_tasks` of the split are
used, so two "seeds" train on different curricula (WebShop: 89 of 800 shared). `data.task_seed` was
added to pin the task draw; it defaults to `data.seed`, so all pre-existing configs are unchanged.
The replicate configs set `task_seed: 2026`; M1 and F3 deliberately do not, so F3 diagnoses the run
that actually produced 3.5%.

The shuffled control (F1) permutes progress values among each task's own states: coverage (90.4%)
and mean bonus (P 0.507 vs 0.523) are held fixed while corr(position, Φ) collapses from +0.772 to
+0.045 — separating "teacher-derived progress" from "a dense bonus of that size". The Llama run (F2)
uses the paper's DUET recipe with only Llama-appropriate memory knobs changed; the LUFFY and GRPO
Llama points already exist on disk (19.5% vs 5.5% at step 50).

Spare GPU 6: Qwen2.5-14B download → auto-starts teacher sampling via
`run_teacher14b_when_ready.sh` (aux env stack on ports 18011/18091). Pipeline smoke-tested
end-to-end on 2026-07-26.

H200 (separate machine, `/data/home/qisheng/DUET_H200`): ALFWorld DUET seeds 2025/2026/2027,
teacher cache 10%/1%, SFT rerun + curve. See `DUET_H200/H200_AGENT_HANDOFF.md`.

## Operational notes for whoever picks this up

- Both queues enforce a single instance via a PID file. Two concurrent queues will silently
  corrupt each other's runs — we lost a run to exactly that.
- `/proc/sys/net/ipv4/ip_local_port_range` on this host is **32768–60999**, which contains the
  AgentGym ports 36001/36003. vLLM binds random ports in that range, so `kill_port` in the env
  scripts now refuses to kill anything matching `ray::|vllm|EngineCore|main_ppo|launcher.py`.
  We lost a second run before adding that guard.
- Never use broad `pkill` patterns; other people's jobs share this machine (GPUs 3/5/7).
