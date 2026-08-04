# H200 → A100 final handoff for tonight's draft (2026-07-29 22:30)

Everything the draft needs from this machine, in one file. All numbers are from
`validation_log/*.jsonl` on disk here, strict success = score ≥ 1.0, n = 200, and every DUET
replicate ran the submission's **exact** configuration (4 GPU, micro=2, `task_seed: 2026`,
training task ids verified from each run's own saved rollouts). Where I recommend framing, it
is marked *recommendation* — the draft is yours and the user's call.

---

## 1. THE HEADLINE-CHANGING RESULT (landed 17:46–18:10 today, after my last package)

**The SFT→GRPO baseline replicates high. The paper's 30.0% is a low draw from the baseline's
own distribution.**

| SFT→GRPO (1.5B-ALFWorld) | endpoint strict SR |
|---|---|
| paper (seed 2026 lineage, verified by your audit) | 30.0% |
| replicate, seed 2025 lineage | 41.5% |
| replicate, seed 2027 (seed-matched end to end) | 44.0% |
| replicate, seed 2028 (seed-matched end to end) | 47.0% |
| **combined distribution (n=4)** | **40.6 ± 7.4** |

Config integrity: a full flattened diff of our replicates vs the paper's runs shows **zero**
substantive differences (only seed / GPU count, and these runs are gradient-equivalent across
GPU count: `use_dr3: false`, constant μ, world-size-independent 32×2 loss grouping). Each
replicate is seed-matched *end to end* — SFT(seed s) → its own checkpoint → GRPO(seed s).

**And the SFT stage alone matches DUET:**

| SFT-stage checkpoint (step 50 of stage 1) | strict SR |
|---|---|
| seed 2026 | 48.5% |
| seed 2027 | 48.5% |
| seed 2028 | 43.5% |
| **mean (n=3)** | **46.8** |

Seed-matched stage-1 → stage-2 deltas: 48.5→41.5 (−7.0), 48.5→44.0 (−4.5), 43.5→47.0 (+3.5).
In 2 of 3 pairs the GRPO stage *lowers* the SFT checkpoint.

**Against DUET's own distribution (n=5):** 34.5 / 39.5 / 41.0 / 47.5 / 50.5 → **42.6 ± 6.4**.

**Blunt reading: on 1.5B-ALFWorld, DUET (42.6±6.4), SFT→GRPO-replicated (40.6±7.4) and the
SFT-stage-alone (46.8) are statistically indistinguishable. The 17.5pp margin on this cell does
not survive replication of the baseline. Do not defend it.**

## 2. A config fact that reframes what "the baseline" is (verified in the executed runs' wandb configs)

The "SFT stage" is **not plain SFT**. Its executed config is GRPO + teacher mixing (1/group) +
**constant μ = 1.0 BC** + **`teacher_baseline_separation: enable: true`** on 400 tasks. I.e. it
is itself a correct-then-extract configuration — essentially *DUET minus DR3 minus SC, with μ
pinned at its maximum*. Likewise verified: **CHORD (27.0%) and LUFFY (5.5%) also ran with
baseline separation enabled.**

Three consequences worth using:

- **The strongest baseline on this cell already contains our Bias-1 correction.** What the
  46.8% "SFT stage" number shows is not "naive imitation matches DUET" but "our correction plus
  heavy constant BC matches DUET on ALFWorld-1.5B at this budget". The clean no-correction
  points remain LUFFY-style mixing *without* the rest (5.5% even WITH baseline separation) and
  on-policy GRPO (1.0%).
- **The attribution argument survives intact**: granting LUFFY our baseline-separation
  correction still leaves it at 5.5%; removing baseline separation from DUET collapses it to
  0.0. The corrections are load-bearing. What is genuinely at parity on this one cell is
  *adaptive-μ + DR3 + SC* versus *constant μ=1.0 on half the tasks* — i.e. the Stage-2
  refinements, at 1.5B, on ALFWorld, at 100 steps.
- At 7B the same comparison flips hard (DUET 86.5 vs LUFFY 82.5 below GRPO 85.0), and on
  WebShop your task-matched SFT→GRPO rerun went *down* (18.5 → 7.5). The parity is one cell,
  not the method.

*Recommendation*: restate the 1.5B-ALFWorld cell per `DECISION_alfworld_1p5b_cell.md` branch 3
(it landed on the branch we planned for): lead with distributions, concede the cell-level
parity with the replicated baseline plainly, keep the margins that survive (CHORD +15.6 on the
mean, LUFFY +37.1, GRPO +41.6), and let 3B/7B/WebShop carry scale. Being the ones to publish
the baseline's true distribution is our strongest possible answer to y9x6's "replicate the
baseline too" — nobody asks for that expecting the authors to report it against themselves.

## 3. DUET multi-seed set — final, n=5

| seed | val@50 | val@100 | 2nd-half change |
|---|---|---|---|
| 2026 (submitted) | 42.5 | **47.5** | +5.0 |
| 2025 | 34.0 | 39.5 | +5.5 |
| 2026 replicate | 38.0 | 34.5 | **−3.5** |
| 2027 | 36.0 | 41.0 | +5.0 |
| 2028 | 46.0 | 50.5 | +4.5 |
| **mean ± sd** | 39.3 ± 4.7 | **42.6 ± 6.4** | — |

- The dynamics claim must be stated as **4 of 5** (an earlier note of mine said 4/4 — the
  fifth broke it). Still sharp: DUET's worst second-half change (−3.5) ≈ CHORD's typical
  (−3.0), and GRPO/LUFFY lose 15.5/20.5.
- The one declining run shows the mild end of the length-drift signature (resp len 3.7k→6.3k
  after step 70, success 0.52→0.18) — same family as your WebShop finding, one coherent
  phenomenon across both environments. *Recommendation*: present WebShop 15.0–24.5 (vs 36.0)
  and ALFWorld 34.5–50.5 (vs 47.5) as ONE story — "submitted numbers are single draws that sit
  above their distribution means; we now report distributions" — not two incidents.

## 4. Everything else H200 has produced that the draft can already use

| item | number(s) | file |
|---|---|---|
| μ-floor attribution timing, 3 seeds | 71–100% (mean 84%) of the climb happens with BC at its floor | `data/bc_attribution_timing.md` |
| eval-point robustness | DUET +5.0 50→100 in the submitted run; CHORD −3.0, GRPO −15.5, LUFFY −20.5. **−DR3 is 14.0pp behind DUET at step 50** (42.5 vs 28.5) though equal at 100 — do not keep "scores identically" | `data/eval_point_robustness.md` |
| seed-variance decomposition | the old "sd 4.9pp" set shares only 33% of tasks pairwise; not seed variance | `data/alfworld_seed_variance_decomposition.md` |
| cache coverage of the real 800 | full 97.8% / sub10 55.9% / sub1 8.1%; realised teacher/prompt 0.98 / 0.56 / 0.08; +m=2 → a 24× supply curve | `data/teacher_cache_coverage_of_train800.md` |
| curriculum verification tool + result | every replicate's task ids ⊆ the paper's 800, 0 outside | `scripts/verify_curriculum.py` |
| micro-batch reproducibility rule | hold `world_size × micro` constant; verify `dr3/global_step`=4.5 at step 1 | `H200_REPORT.md` §2, reply §4 |

## 5. Still in flight here (mark ⏳ in tonight's draft; numbers by morning)

| run | answers | ETA (SGT) |
|---|---|---|
| `g4_shufsc_seed2027` — shuffled progress map, **paired within seed** vs the 41.0% true-map partner | bDeY W1 / y9x6 Q6, without the seed-variance objection to your single-run 6.5pp split | ~00:30 |
| `g4_shufsc_seed2028` — paired vs the 50.5% partner | same | ~06:30 |
| `g4_cache10` / `g4_cache1` / `g4_ntch2` | y9x6 cache-size + mixing-ratio | ~12:30 / ~18:30 / ~24:30 |

My two true-map anchors for the pairs: seed2027 = 41.0, seed2028 = 50.5. Your single shuffled
run was 41.0 vs true 47.5 (unpaired). With ALFWorld sd now known to be ~6pp, only the paired
version can support a decomposition claim; if the paired gaps come out small, the honest
statement is "the shuffled control is within seed noise on ALFWorld" and the SC ordering
argument should rest on the WebShop side and on the noise experiments.

## 6. Corrections to numbers currently in the draft (do these regardless of anything else)

1. `rebuttal_draft.md` ALFWorld seed table — replace with §3 above (the draft's "40.5/39.0/
   43.5 sd 2.3" block was already replaced once; make sure the n=5 version is what ships).
2. Any "≈2.8× our observed sd" style claims: ALFWorld sd is **6.4pp** (val@100, n=5).
3. `response_y9x6.md` still opens its multi-seed answer with the −DR3 47.5/38.0/41.0 "sd
   4.9pp" set — relabel per §4 row 3 (it is seed+curriculum variance).
4. `response_bDeY.md` Q2: the SFT-curve answer should now cite the *replicated* SFT stage
   (loss 0.84→0.15, train success 0.02→0.33, **val 43.5–48.5 at step 50, n=3**) — this also
   answers "was SFT properly executed" more strongly than the single curve.
5. EXPERIMENT_LOG.md:87 points the ALFWorld DUET cell at the `use_chord:false` config — widen
   your C7 to cover both environments.

## 7. Score assessment after today (my read; you own the final)

| reviewer | before | now | why |
|---|---|---|---|
| y9x6 (3) | 45–58% → 4 | **≈55%** → 4 | Multi-seed answered *more* honestly than asked (both method AND baseline distributions); attribution timing is 3-seed; cache/mixing land tomorrow. The parity concession costs margin but buys credibility; y9x6's own rating text already says "technically solid". |
| bDeY (3, conf 4) | ≈35% → 4 | **≈35%** → 4 | Eq. 9 concession + SFT curve now n=3 + the SC controls. The cell-parity finding actually *supports* their skepticism — conceding it precisely is the only move that can flip a conf-4 reviewer. |
| UyKJ (4) | ≈30% → 5 | **≈25%** → 5 | Unchanged except the Llama confound (my §4 of the reply) — if the micro=2 control run isn't done, the Llama story stays "variance, no conclusion", which caps this one at 4. |

**P(accept) ≈ 0.30–0.35.** The draft's biggest single lever tonight is coherence: one
distribution-first story across both 1.5B cells, told by us before any reviewer computes it.

## 8. File index in this package

- `H200_HANDOFF_FOR_DRAFT_2026_07_29.md` ← this file
- `DECISION_alfworld_1p5b_cell.md` — the branch logic §2 executes
- `H200_REPLY_2026_07_29.md` — yesterday's reply (Llama confound §4, C7 extension, retractions)
- `data/h200_seed_table.md`, `data/h200_rebuttal_results.md` — live tables, all runs
- `data/{bc_attribution_timing, eval_point_robustness, alfworld_seed_variance_decomposition,
  teacher_cache_coverage_of_train800}.md`
- `H200_REPORT.md` — full infra record incl. the micro-batch story (internal; camera-ready note)
- raw validation logs for every number quoted here: `validation_logs_h200/` (per-run 50/100 jsonl)
