# Evidence pack: "Is DUET SC-dependent / less general than CHORD?"

Target reviewers: **y9x6**, **bDeY** (shared concern).
Status: every number below was re-derived from raw artifacts in this repo on 2026-07-26.
Numbers I could **not** verify are listed in §6. Do not put an unverified number in the rebuttal.

---

## 0. TL;DR — the claim we can defend, and the one we cannot

**CAN defend (verified):**

1. DUET's **core** (teacher-separated GRPO baseline + DR3 density-ratio repair + DR3-gated
   adaptive BC) consumes **exactly CHORD's inputs**: teacher token sequences and a
   teacher/on-policy token mask. No observations, no teacher logits, no tokenizer alignment,
   no environment structure, no state matching. Verified in code (§2).
2. The State Channel is an **optional module**, not a prerequisite. With SC removed entirely
   (`state_channel.enable: false`), DUET on 1.5B-ALFWorld scores **31.0%** vs **CHORD 27.0%**
   and **SFT+GRPO 30.0%** — i.e. the core alone is at least CHORD-level, and retains
   **64.5%** of DUET's full gain over on-policy GRPO. (§1)
3. SC is an **interface** Φ(s)∈[0,1], not a single matching algorithm. Three instantiations
   exist in the code with very different engineering cost, and the ALFWorld one is the
   zero-effort teacher-derived map that the reviewers describe. (§3)
4. Where exact matching breaks (30% observation-word dropout), a **dependency-free TF-IDF
   cosine matcher** restores 93.3% of matches with progress MAE 0.015 vs the exact map's 6.4%
   hit rate — offline, on the real 72B ALFWorld teacher cache. (§4)

**CANNOT defend — do not write these:**

- ❌ "DUET does not depend on SC." On **1.5B-WebShop**, removing SC drops strict SR from
  36.0% → **1.0%**, which is *below* CHORD's 11.5%. SC is load-bearing there. Say so.
- ❌ "SC requires zero engineering across environments." Our **WebShop SOTA config uses
  `match_mode: attribute_aware`**, a hand-written ~210-line WebShop potential function. That
  IS domain engineering and bDeY is right about it. Reframe, do not deny.
- ❌ "DUET-without-SC beats CHORD" as a *statistically established* claim. The +4.0pp AF margin
  is single-seed; our own 3-seed spread on a comparable 1.5B-AF ablation cell is sd ≈ 4.9pp.
  Write "matches or exceeds", not "beats".

---

## 1. The −SC ablation, re-verified from raw validation logs

I recomputed strict SR (score ≥ 1.0) and mean reward directly from the 200-row
`validation_log/100.jsonl` files rather than trusting the tables.

| Cell | Config (`state_channel.enable`) | n | strict SR | mean reward | Source log |
|---|---|---|---|---|---|
| **1.5B-AF DUET −SC** | `false` (L248) | 200 | **31.00%** | 0.3100 | `experiments/alfworld/alfworld_qwen1.5b_duet_minus_sc/validation_log/100.jsonl` |
| 1.5B-AF CHORD | n/a (no SC block) | 200 | **27.00%** | 0.2700 | `experiments/alfworld/alfworld_qwen1.5b_chord/validation_log/100.jsonl` |
| 1.5B-AF SFT+GRPO | n/a | 200 | **30.00%** | 0.3000 | `experiments/alfworld/alfworld_qwen1.5b_sft_rl/validation_log/50.jsonl` (val@50 = end of protocol) |
| 1.5B-AF OnPolicy GRPO | n/a | 200 | 1.00% | 0.0100 | `experiments/alfworld/alfworld_qwen1.5b_onpolicy/validation_log/100.jsonl` |
| 1.5B-AF DUET (full) | `true` | 200 | 47.50% | 0.4750 | `experiments/alfworld/alfworld_qwen1.5b_duet_v39c_postfix/validation_log/100.jsonl` |
| **1.5B-WS DUET −SC** | `false` (L247) | 200 | **1.00%** | **0.4504** | `experiments/webshop/webshop_qwen1.5b_duet_minus_sc/validation_log/100.jsonl` |
| 1.5B-WS CHORD | n/a | 200 | **11.50%** | **0.6032** | `experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl` |
| 1.5B-WS SFT+GRPO | n/a | 200 | 18.00% | 0.6409 | `experiments/webshop/webshop_qwen1.5b_sft_rl/validation_log/50.jsonl` |
| 1.5B-WS OnPolicy GRPO | n/a | 200 | 0.50% | 0.1523 | `experiments/webshop/webshop_qwen1.5b_onpolicy/validation_log/100.jsonl` |
| 1.5B-WS DUET (full) | `true` | 200 | 35.50% (log) / 36.0% (paper) | 0.7057 | `experiments/webshop/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06/validation_log/100.jsonl` |

**Answer to the key question posed in the task:** *Yes on ALFWorld, no on WebShop.*
DUET-without-SC = **31.0%** vs CHORD **27.0%** → **+4.0pp** on 1.5B-ALFWorld.
DUET-without-SC = **1.0%** vs CHORD **11.5%** → **−10.5pp** on 1.5B-WebShop.

Derived quantities (all from the table above):

- Fraction of DUET's total gain over on-policy GRPO retained with SC removed:
  - ALFWorld: (31.0 − 1.0) / (47.5 − 1.0) = **64.5%**
  - WebShop, strict SR: (1.0 − 0.5) / (36.0 − 0.5) = **1.4%**
  - WebShop, **mean reward**: (0.4504 − 0.1523) / (0.7057 − 0.1523) = **53.9%**
    (CHORD's is (0.6032 − 0.1523)/(0.7057 − 0.1523) = 81.5%)
- The WebShop nuance worth stating: on the *continuous* reward the core alone still recovers
  ~54% of DUET's gain (0.152 → 0.450), it just almost never crosses WebShop's all-attributes-
  exact threshold. It is still below CHORD (0.603) on that metric. **Do not spin this as a win.**

**Seed-variance caveat (must be stated).** We only have multi-seed data for one 1.5B-AF
ablation cell, `−DR3`: 47.5% / 38.0% / 41.0% (seeds 2026 / 2025 / 2027, verified from
`experiments/alfworld/alfworld_qwen1.5b_duet_minus_dr3{,_seed2025,_seed2027}/validation_log/100.jsonl`)
→ mean 42.2%, sample sd **4.86pp**, range 9.5pp. A single-seed +4.0pp margin is inside one sd.
Correct wording: "**matches or exceeds** CHORD"; or "**is not worse than** CHORD".

---

## 2. Input requirements: the DUET core needs exactly what CHORD needs

Verified by reading the code. Each core component's full input list:

**(a) Teacher-separated GRPO baseline** —
`agentevolver/module/trainer/ae_ray_trainer.py:450-548`,
`compute_grpo_outcome_advantage_teacher_baseline_separated(token_level_rewards, response_mask,
index, teacher_mask, ...)`. Signature at L450-459. The only teacher-specific input is
`teacher_mask`, reduced at L480 to a per-sample boolean:
`is_teacher = (teacher_mask * response_mask).sum(dim=-1) > 0`. Everything else is the ordinary
GRPO group index and scalar rewards. **No observations, no env structure.**

**(b) DR3 density ratio** — `agentevolver/module/exp_manager/dr3_ratio.py`.
Module docstring L7: *"Do NOT require teacher logits / teacher tokenizer alignment."*
Features are built by `compute_sequence_features(log_prob, advantages, response_mask,
ref_log_prob, feature_mode, ...)` (L73-81). Every DUET/ablation/rebuttal config sets
`feature_mode: v3_aug` (e.g. `config/duet_paper_experiments_configs/webshop/sweep_1.5b/
webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml:81`). The `v3_aug` branch is **L98-145** and
its comment at L99-100 reads *"advantage-free … keep density-ratio narrative (no reward/adv
leakage)"*. The stacked feature vector L129-145 is exactly:
`lp_mean, lp_std, lp_min, lp_max, lp_low_ratio_10, lp_low_ratio_20, lp_high_ratio_1, resp_len,
kl_ref_mean, kl_ref_std, kl_ref_abs_mean, kl_ref_pos_ratio` — **12 sequence-level statistics of
log π and log π_ref. `advantages` is a formal parameter that this branch never reads.**
(The `v3` branch at L146-174 is the same story with 7 dims. Only the legacy `v1` (L92-97) and
`v2` (L175-214) branches touch `advantages`; no DUET config uses them.)
Call site: `het_actor.py:1289-1298`, labels are `is_offpolicy=teacher_sample` (L1301) —
i.e. "did this sequence come from the teacher", nothing else.
Independent empirical support that the discriminator is distributional and not a success
detector: `NeurIPS_2026_Latex/data/dr3_confound_diagnostic.md` (D_all 84.7%, D_succ 90.0%,
D_fail 88.0% held-out; mean P(student) = 0.853 for successful vs 0.855 for failed student
rollouts vs 0.287 for teacher).

**(c) Adaptive BC (μ)** — `het_actor.py:1775-1815`. Rule at L1777-1781:
`μ = clamp(μ_max · max(0, (1−d)/(1−d_floor)), μ_min, μ_max)` with `d = EMA(dr3/disc_acc)`.
The *only* driving signal is the DR3 discriminator's own accuracy (L1787). The BC loss itself,
`compute_chord_sft_loss(log_prob, response_mask, exp_mask, ...)`
(`het_core_algos.py:1767-1773`), takes only log-probs and the expert-token mask — **this is
literally CHORD's Eq. 6.**

**(d) Input parity with the CHORD baseline is exact.** Both read the same file with the same
flags: `alfworld_qwen1.5b_chord.yaml:182-189` and `..._duet_*.yaml` both set
`teacher_experience.data_path: data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl`,
`use_log_prob: false`. Both CHORD baseline configs have **no `state_channel` block at all**
(`grep -c state_channel` returns 0 for `alfworld_qwen1.5b_chord.yaml` and
`webshop_qwen1.5b_chord.yaml`). DR3 further runs in `apply_to: teacher_no_logprob` mode
(`het_actor.py:1124`, comment L1386), i.e. it is *designed* for the case where teacher logits
are unavailable — so DUET's core is if anything *less* demanding than methods needing
teacher log-probs.

→ **Defensible sentence:** "Three of DUET's four components read only `(teacher token ids,
teacher/on-policy mask, scalar task reward)`. That is byte-for-byte the input CHORD consumes.
Any setting in which CHORD can be run, DUET's core can be run unchanged."

---

## 3. SC portability: what is actually shared across environments, and what is not

`agentevolver/module/exp_manager/state_progress.py` implements **five** `match_mode` values.
Which one each paper number used:

| Env | Paper config | `match_mode` | Teacher-derived? | Hand-written env logic |
|---|---|---|---|---|
| ALFWorld (all scales) | `alfworld_*_duet*.yaml` (e.g. `alfworld_qwen1.5b_duet_v39c_postfix.yaml:253`) | `hash` | **yes** | 3 lines (`normalize_observation` L540-543) |
| WebShop (SOTA + all ablations) | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.yaml:251` | **`attribute_aware`** | **no** | **~210 lines** (L29-51, L56-91, L96-266) |
| WebShop (older configs, 10 files) | `webshop_3b_duet.yaml:249` etc. | `stage` | no | ~60 lines (L29-91) |
| SciWorld | `sciworld_3b_duet.yaml:240` / `sciworld_3b_duet_0406.yaml:243` | `hash` / `sciworld_stage` | yes / no | 0 / ~253 lines (L274-520) |
| Rebuttal (new) | `alfworld_qwen1.5b_duet_a100_*_soft.yaml:253` | `soft` | **yes** | same 3 lines as `hash` |

Counts: `grep -rn "match_mode" config/duet_paper_experiments_configs/webshop/` → 169
`attribute_aware`, 10 `stage`, **0 `hash`**. We have **no** WebShop result using the generic
teacher-derived matcher. This is the sharpest form of bDeY's point and we must not paper over it.

**What genuinely IS shared and unmodified (`hash` mode, L697-731):** build a per-task dict
{normalized teacher observation → max progress}, with `progress = j / max(T−1, 1)` (L709) and
`progress_map[obs] = max(existing, progress)` (L710). No env branch anywhere in that loop.
The only env-conditional code on this path is `normalize_observation` (L526-561): one suffix
strip per env — ALFWorld `"\nAVAILABLE ACTIONS:"` (L541), WebShop 4 regexes (L545-550),
SciWorld 2 regexes (L554-558). That is ~20 lines total for three environments, and it exists
only to stop the env's action menu from making every observation hash-unique.

**Two honest reframings that are stronger than the false "zero engineering" claim:**

1. *SC is an interface, not an algorithm.* The trainer only ever calls
   `ExpertProgressMap.get_potential(task_id, observation) → Φ ∈ [0,1]`
   (`state_progress.py:831-862`; injected at `ae_ray_trainer.py:3272-3419`). Everything
   downstream — β·P(τ), `exclude_teacher`, `grpo_decouple`, step deltas — is env-agnostic.
   A new environment needs *a* Φ, from whichever of the five sources is cheapest.
2. *Our WebShop result does not use state matching at all — which directly contradicts the
   premise of the objection.* `webshop_attribute_aware_potential` (L252-265) calls
   `classify_webshop_page` + `compute_attribute_match_score` on the observation string;
   `get_potential` L837-840 only checks `task_id in self._task_ids` (does this task have
   teacher data at all) and never queries a teacher state map. So y9x6's "SC depends on
   matching student states to teacher-visited states" is literally true only for our ALFWorld
   (`hash`) and rebuttal (`soft`) runs. Half of our main table already runs SC without any
   state matching. **The cost of that is domain knowledge instead — say both halves.**

→ **Defensible sentence:** "SC's *matching* recipe is unmodified across environments — the
per-environment code on that path is ~20 lines of suffix stripping. What is *not* uniform is
the choice of potential source: our ALFWorld runs use the zero-effort teacher-derived map;
our WebShop runs use a hand-written domain potential (~210 lines), which we agree is genuine
domain engineering. We report the cost of each rather than claiming a single free recipe."

---

## 4. The soft-matching calibration and how to phrase it before results land

**Offline calibration (already done, verifiable).**
`NeurIPS_2026_Latex/data/soft_match_calibration.md`, produced by
`scripts/calibrate_soft_matching.py` on the real cache
`data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl`
(100 sampled tasks; 60 tasks × ≤20 held-out observations; word-level dropout p=0.30 applied to
the **matcher's** copy of the observation only — the policy input is untouched,
`state_progress.py:787-798`).

| θ | hash hit-rate | soft hit-rate | soft progress MAE | cross-task FP |
|---|---|---|---|---|
| 0.3 | 6.0% | 93.8% | 0.013 | 73.4% |
| 0.5 | 5.8% | 92.8% | 0.014 | 48.6% |
| **0.6** | **6.4%** | **93.3%** | **0.015** | **33.6%** |
| 0.7 | 6.0% | 88.5% | 0.012 | 21.8% |

Operating point used by the training runs: θ=0.6, p=0.30
(`alfworld_qwen1.5b_duet_a100_obsnoise_soft.yaml:253-255`).
Implementation: pure-Python TF-IDF cosine, per-task profiles built at L733-759, queried at
L800-829 with a per-task IDF so tokens unseen in that task get max IDF and can only *lower*
similarity (L809-813). **No new dependency, no learned model, no env-specific code.**

**Three training runs are queued** (configs verified, no results yet):
`alfworld_qwen1.5b_duet_a100_obsnoise_hash` (hash + p=0.30 → SC starved),
`alfworld_qwen1.5b_duet_a100_obsnoise_soft` (soft + p=0.30 → SC restored),
`alfworld_qwen1.5b_duet_a100_soft_clean` (soft, no noise → drop-in regression check).
Reference points: full DUET 47.5%, −SC 31.0%, strongest baseline 30.0%.

**Two caveats to internalise before drafting** (a confidence-4 reviewer will spot both):

- The "hash collapses to 6%" row is *arithmetic*, not a discovery: we corrupt the query string
  and then do an exact lookup. Present it as the definition of the stress test, not as a
  finding. The finding is that a 30-line lexical matcher recovers 93% with MAE 0.015.
- **Cross-task FP is 33.6% at θ=0.6.** In training this means roughly a third of genuinely
  novel on-policy observations receive a non-zero Φ. Disclose the number. Mitigations we can
  state truthfully: ALFWorld scenes are lexically near-identical across tasks so the assigned
  progress is usually plausible; Φ enters only as a bounded bonus (β=0.2,
  `beta_decay: true → 0.3` in the rebuttal configs) on top of task reward; teacher samples are
  excluded (`exclude_teacher: true`). If asked to trade it away, θ=0.7 halves FP to 21.8% at
  88.5% hit rate.

**Outcome-robust phrasing.** Draft the rebuttal so it survives every outcome:

- *Fully positive* (soft+noise ≈ 45-47%, hash+noise ≈ 31%, soft+clean ≈ 47%):
  "Swapping the matcher — one config field, no algorithmic change — recovers the full SC
  benefit under 30% observation corruption, and is loss-free on clean observations."
- *Partially positive* (soft+noise lands between −SC 31% and full 47.5%): report the exact
  recovery fraction: "recovers X of the Y pp that observation noise removes." Still answers the
  objection, because the objection is about whether SC *can* be adapted, not about how much.
- *Flat* (soft+noise ≈ hash+noise): fall back to (i) the offline calibration — the matching
  problem itself is solvable off-the-shelf — and (ii) the −SC floor from §1: under total SC
  failure DUET degrades to a CHORD-level method on ALFWorld rather than breaking. Then state
  the negative training result plainly; a reported negative control is worth more here than a
  hedge.
- *Regression on `soft_clean`*: say "soft matching is a fallback for noisy observations and
  costs Z pp on clean ones; we default to exact matching where it is available."

Never write the outcome before the run finishes. Numbers to fill in live in
`NeurIPS_2026_Latex/data/a100_rebuttal_results.md`, which is currently **empty of results**
(only a note about three aborted duplicate-queue rows removed on 2026-07-26).

---

## 5. Rebuttal paragraphs (usable verbatim; every number verified)

> **On generality of the method's core.** We want to separate two claims, because we think the
> reviewers are right about one of them. DUET has four components; three of them —
> teacher-separated GRPO baselines, the DR³ density-ratio correction, and DR³-gated adaptive
> behaviour cloning — read *only* the teacher's token sequence and a teacher/on-policy token
> mask. They never see an observation, an environment state, a teacher log-probability, or a
> shared tokenizer: the DR³ discriminator's features are twelve sequence-level statistics of
> log π and log π_ref (mean/std/min/max, three tail fractions, length, four KL-to-reference
> moments), and the module is explicitly built so that teacher logits and tokenizer alignment
> are not required. This is byte-for-byte the input CHORD consumes; in any setting where CHORD
> can be run, DUET's core runs unchanged. Only the State Channel needs anything more.

> **How much of DUET survives if the State Channel cannot be built?** We ran exactly this
> ablation. With `state_channel.enable=false`, DUET reaches **31.0%** on 1.5B-ALFWorld, against
> **27.0%** for CHORD and **30.0%** for SFT+GRPO under the identical protocol (val@100, 200
> tasks, strict success) — i.e. the core alone matches or exceeds the strongest teacher-using
> baselines and retains 64.5% of DUET's total gain over on-policy GRPO (1.0% → 47.5%). We
> should be careful here: this is a single seed, and our three-seed spread on a comparable
> ablation cell has sd ≈ 4.9pp, so we claim parity with CHORD rather than a win. On
> 1.5B-WebShop the same ablation is **not** favourable to us: strict success falls to 1.0%,
> below CHORD's 11.5%, although mean reward (0.450 vs on-policy 0.152, CHORD 0.603) shows the
> core still does substantial work and the collapse is specific to WebShop's
> all-attributes-exact success criterion. We would rather state this than hide it: on WebShop
> the State Channel is load-bearing. The claim we defend is not "SC is unnecessary" but "SC is
> an optional module on a CHORD-compatible core, and its absence degrades DUET gracefully to
> baseline-competitive on ALFWorld."

> **On the State Channel needing exact state matching.** The trainer's only interaction with
> the State Channel is a call `Φ(task_id, observation) → [0,1]`; everything downstream is
> environment-agnostic. Exact hash matching over normalised teacher observations is one
> instantiation, and the one our ALFWorld results use — there the *entire* per-environment code
> is a three-line strip of the appended action menu, with the map itself
> (`progress = j/(T−1)`, keep the max over trajectories) shared verbatim across environments.
> It is not the only instantiation, and it is worth noting that our WebShop results do not use
> state matching at all: there Φ is computed from the observation itself by a page/attribute
> potential, so the specific dependency the reviewers describe does not apply to half of our
> main table. We should be equally clear about the cost of that alternative: the WebShop
> potential is roughly 210 lines of hand-written domain logic, and we agree this is real
> per-environment engineering rather than a free recipe. What we claim is that DUET offers a
> spectrum — zero-effort teacher-derived matching where observations are clean and repeatable,
> a hand-written potential where domain structure is cheap to encode, and no State Channel at
> all — and that the *core* algorithm is unchanged in all three cases.

> **On noisy / partially observable states (new experiment).** To test the failure mode
> directly rather than argue about it, we added a matcher-side corruption knob (word-level
> dropout on the observation the matcher sees; the policy's input is untouched) and a
> dependency-free TF-IDF cosine matcher, selected by a single config field. Calibrating
> offline on the real 72B ALFWorld teacher cache (100 tasks; 60 tasks × ≤20 held-out
> observations) at 30% word dropout: exact matching retains **6.4%** of hits — by construction,
> since the query string no longer hashes to the stored key — while soft matching retains
> **93.3%** with mean absolute error **0.015** on the recovered progress value (θ=0.6). The
> trade-off is a **33.6%** cross-task false-positive rate at that threshold, which we report
> rather than tune away: ALFWorld scenes are lexically similar across tasks, Φ enters only as a
> bounded bonus (β=0.2, decayed) on top of the task reward, and θ=0.7 reduces false positives
> to 21.8% at 88.5% recall. We are running the corresponding end-to-end comparison
> (hash+noise vs soft+noise vs soft on clean observations) and will report it in this thread.
> The point we are making is architectural: adapting the State Channel to a noisier
> observation space was a matcher swap, not a change to the objective, the advantage
> estimator, or the density-ratio correction.

---

## 6. Gaps and integrity flags — read before submitting

1. **Two paper cells do not reproduce exactly from the logs I can see (0.5pp each).**
   - `tables/main_results.tex` 1.5B-WebShop DUET = **36.0%**; recomputed from
     `experiments/webshop/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06/validation_log/100.jsonl`
     = **35.5%** (71/200 with score ≥ 1.0; 71 also at ≥0.999; next highest cluster is 0.909).
     Mean reward 0.7057 matches the paper's 0.706 exactly, so it is the same run.
   - `tables/main_results.tex` 1.5B-WebShop SFT+GRPO = **18.5%**; the only log present
     (`.../webshop_qwen1.5b_sft_rl/validation_log/50.jsonl`) gives **18.0%**; mean reward 0.6409
     matches the paper's 0.641.
   Both may be rounding/transcription from a different log. **Someone must resolve these before
   the rebuttal quotes 36.0%**, because a reviewer who asks for logs will hit them. Every
   ALFWorld cell I checked reproduces exactly.
2. **3B −SC does not exist.** `data/ablation_results.md` shows 8 attempted 3B −SC/−BC runs
   between 2026-05-06 and 05-07, all "val@100 MISSING — run failed or killed". So the
   "core without SC ≈ CHORD" evidence is **1.5B-ALFWorld only, single seed**. We cannot say it
   holds at 3B.
3. **No multi-seed −SC.** Multi-seed data exists only for the −DR3 AF cell (3 seeds). Any
   significance language about the 31.0 vs 27.0 comparison is unsupported.
4. **No WebShop run with the generic `hash` matcher** (0 of 179 WebShop configs). We therefore
   cannot claim the teacher-derived recipe transfers to WebShop — only that the *code path* is
   env-agnostic.
5. **No SciWorld results at all.** `experiments/sciworld/` does not exist. SciWorld configs and
   the `sciworld_stage` potential are code-only. Do not cite SciWorld as evidence of transfer;
   at most cite it as evidence the Φ interface accommodates a third environment.
6. **The three soft-matching training runs have not produced results.**
   `data/a100_rebuttal_results.md` contains no completed rows as of 2026-07-26 01:01.
7. **The cross-task FP number (33.6%) is a genuine weakness of the soft matcher** and will be
   the first thing a careful reviewer attacks in §4. Lead with it rather than being asked.
8. Not checked by me: whether the CHORD baseline's hyperparameters were tuned to the same
   budget as DUET's (relevant if a reviewer challenges the 27.0% number itself).
