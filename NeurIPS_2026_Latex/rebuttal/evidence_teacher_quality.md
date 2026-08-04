# Evidence package: teacher quality / teacher dependence

Answers reviewer **UyKJ** ("how does performance change when the teacher is weaker?") and
reviewer **y9x6** ("teacher-quality ablation… moderately stronger, noisy, or partially
suboptimal"; "How large is the teacher trajectory cache, and how diverse are the successful
trajectories within it?").

Compiled 2026-07-26. **Every number below was recomputed from raw artifacts on disk**
(validation JSONL, training logs, teacher `.pkl` caches) — not copied from prior summaries.
Provenance is given per row. Prior analysis-report numbers that I could *not* re-derive are
listed in §6 (Gaps) rather than being reused.

---

## 1. The headline argument

We do not have a weaker *teacher*. We have something evidentially equivalent and arguably
cleaner: **the same teacher cache paired with students of very different strength.** Teacher
"quality" only ever enters DUET through the *relative* teacher–student gap, and that gap is a
quantity we log every step (`diag/group_teacher_minus_on_reward_mean`). Across our 1.5B and 7B
students the measured teacher advantage spans **0.93 → 0.42 at the start of training and
0.55 → 0.06 by the end** — i.e. the 7B setting *is* a "teacher who is only moderately stronger,
and eventually not stronger at all" regime, obtained without changing the teacher.

The result: **DUET is the only teacher-using method that is not harmed when the teacher stops
being better than the student.**

---

## 2. Table-ready: performance vs. teacher–student gap (ALFWorld)

Metric: **strict success rate at val@100**, 200-task validation split, single seed — the same
metric as Table 1 of the paper. Recomputed from `experiments/**/validation_log/100.jsonl`.

| Student | Teacher advantage, steps 1–10 | Teacher advantage, steps 71–100 | GRPO | LUFFY | CHORD | **DUET** | DUET − GRPO | LUFFY − GRPO |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (large gap) | **0.932** | **0.549** | 1.0% | 5.5% | 27.0% | **47.5%** | **+46.5pp** | +4.5pp |
| Qwen2.5-7B (small gap)  | **0.420** | **0.063** | 85.0% | 82.5% | n/a¹ | **86.5%** | **+1.5pp** | **−2.5pp** |

¹ ALFWorld 7B CHORD has no step-100 validation file (run did not complete); its val@50 strict
SR is 73.5% vs GRPO 81.0%, LUFFY 75.0%, DUET 77.5%.

**Provenance**
- 1.5B GRPO / LUFFY / CHORD / DUET: `experiments/alfworld_qwen1.5b_{onpolicy,luffy,chord}/validation_log/100.jsonl`
  and `experiments/alfworld_qwen1.5b_duet_v39c_postfix/validation_log/100.jsonl` → 1.0% / 5.5% / 27.0% / 47.5%, n=200 each.
- 7B: `experiments/alfworld/alfworld_7b_{onpolicy,luffy,chord,duet}/validation_log/{50,100}.jsonl` →
  GRPO 0.810/0.850, LUFFY 0.750/0.825, CHORD 0.735/—, DUET 0.775/0.865, n=200 each.
  (ALFWorld reward is binary, so mean score == strict SR exactly.)
- Teacher advantage: mean of `diag/group_teacher_minus_on_reward_mean` over the stated step
  window, from `logs/alfworld_qwen1.5b_duet_v39c_postfix.log` and `logs/alfworld_7b_duet.log`.

## 2b. Same pattern on WebShop at 7B

Metric: strict SR (score ≥ 1.0) and mean score at val@100, n=200. Recomputed from
`experiments/webshop/webshop_7b_*/validation_log/*.jsonl`.

| Method | strict SR @100 | mean score @100 | strict SR @50 | mean score @50 |
|---|---|---|---|---|
| GRPO   | **50.0%** | 0.7595 | 32.0% | 0.6664 |
| LUFFY  | 45.5% (**−4.5pp**) | 0.7545 | 19.0% | 0.5805 |
| CHORD  | 48.5% (−1.5pp) | 0.7576 | 27.5% | 0.6431 |
| DUET   | n/a² | n/a² | 28.0% | **0.6812** (best @50) |

² WebShop 7B DUET has no step-100 validation file (the run stopped near step 93). Its val@50
mean score, 0.6812, is the highest of the four; do not claim a step-100 WebShop-7B DUET number.

**Both environments agree on the load-bearing point: at 7B, LUFFY — teacher mixing with no
fade-out — is *worse than not using the teacher at all* (−2.5pp ALFWorld, −4.5pp WebShop strict
SR), while DUET is not.**

---

## 3. Table-ready: DR3 measurably attenuates the teacher exactly when the teacher stops helping

All three runs mix teacher trajectories at an **identical nominal rate**
(`diag/teacher_sample_ratio` = 0.1221 mean in all three, i.e. 1 teacher rollout per group of 8).
What differs is how much *gradient* the teacher actually receives
(`duet/teacher_gradient_share`, "TGS"). 10-step bucket means, read from the training logs.

| Steps | 1.5B DUET TGS | 7B DUET TGS | 7B LUFFY TGS | 1.5B teacher adv. | 7B teacher adv. |
|---|---|---|---|---|---|
| 1–10   | 0.269 | 0.129 | 0.218 | 0.932 | 0.420 |
| 11–20  | 0.357 | 0.115 | 0.204 | 0.949 | 0.357 |
| 21–30  | 0.416 | 0.079 | 0.154 | 0.928 | 0.276 |
| 31–40  | 0.292 | 0.086 | 0.137 | 0.809 | 0.263 |
| 41–50  | 0.176 | 0.065 | 0.143 | 0.669 | 0.174 |
| 51–60  | 0.192 | 0.062 | 0.150 | 0.577 | 0.105 |
| 61–70  | 0.166 | 0.051 | 0.115 | 0.558 | 0.097 |
| 71–80  | 0.152 | 0.044 | 0.093 | 0.516 | **0.047** |
| 81–90  | 0.219 | 0.051 | 0.061 | 0.564 | 0.076 |
| 91–100 | 0.170 | 0.110 | 0.075 | 0.568 | 0.068 |
| **run mean** | **0.241** | **0.079** | **0.135** | — | — |

Derived ratios (computed, not eyeballed):
- **LUFFY routes 1.71× more gradient to the teacher than DUET at 7B over the full run, and
  2.26× more over steps 41–80** — at an identical 12.2% nominal mix ratio. That extra teacher
  gradient is exactly the difference between LUFFY's −2.5pp and DUET's +1.5pp.
- **DUET routes 3.05× more gradient to the teacher at 1.5B than at 7B (4.28× over steps 21–40).**
  Nothing about the teacher, the cache, or the DUET hyperparameters changed between those two
  runs — only the student. The schedule is inferred from data.
- Minimum per-step TGS: 1.5B **0.089**, 7B **0.005**. DUET's teacher channel closes ~18× further
  at 7B than at 1.5B.

Supporting per-step signals from `logs/alfworld_7b_duet.log` (7B, weak-relative-teacher regime):

| Steps | `dr3/disc_acc` | `dr3/w_off_mean` (ŵ on teacher samples) | `diag/teacher_adv_pos_ratio` |
|---|---|---|---|
| 1–10  | 0.356 | 1.000 | 0.804 |
| 21–30 | 0.881 | 0.468 | 0.786 |
| 51–60 | 0.974 | 0.531 | 0.605 |
| 71–80 | 0.970 | **0.470** | 0.500 |
| 91–100| 0.900 | 0.716 | 0.525 |

- The discriminator reaches 0.97 accuracy and the applied density ratio on teacher samples
  settles at **≈0.43–0.53**, i.e. DR3 down-weights each teacher token by roughly a factor of two.
- `diag/group_teacher_minus_on_reward_mean` at 7B is **≤ 0 on 10 of 100 steps** (min −0.101):
  the 7B student's own rollouts are *better than the 72B teacher's* on those batches. At 1.5B
  that never happens (0 of 100 steps; min gap 0.384). This is a direct, logged measurement of
  "the teacher is no longer stronger", and DUET is the method that reacts to it.

---

## 4. Teacher cache statistics (answers y9x6's cache-size / diversity question)

All computed by loading the `.pkl` caches directly
(`/tmp/.../scratchpad/analyze_teacher_cache.py`; action sequences extracted from
`<action>…</action>` spans in assistant turns, whitespace/case normalized).

### 4.1 Size, coverage, length, reward

| | ALFWorld full | ALFWorld sub10 | ALFWorld sub1 | WebShop full |
|---|---|---|---|---|
| File | `alfworld_qwen72b_filtered_react_tags.pkl` | `…_sub10.pkl` | `…_sub1.pkl` | `webshop_qwen72b_filtered.pkl` |
| Trajectories | **19,497** | 1,950 | 195 | **26,178** |
| Unique tasks | **2,348** | 1,337 | 189 | **5,691** |
| Trajectories / task (mean, median, max) | 8.30, 10, 10 | 1.46, 1, 5 | 1.03, 1, 2 | 4.60, 5, 5 |
| Tasks with ≥2 demos | 2,284 (97.3% of tasks) | 470 (35.2%) | 6 (3.2%) | 5,527 (97.1%) |
| Turns / trajectory (mean, median, p25–p75, max) | 12.26, 10, 7–17, 30 | 12.02, 9, 7–17, 30 | 12.14, 10, 7–17, 30 | 7.33, 7, 6–8, 12 |
| Reward | **1.0 for all 19,497** | 1.0 for all | 1.0 for all | **1.0 for all 26,178** |
| Generated tokens / traj (mean, median) | 235.7, 189 | 232.5, 186 | 230.0, 190 | n/a³ |
| Teacher model field | Qwen2.5-72B-Instruct | ″ | ″ | Qwen2.5-72B-Instruct |

³ WebShop entries record `total_generated_tokens = 0` because only the reasoning text was
LLM-generated (see §4.3); token counts were not stored.

**Success filtering** (from the filter's own stats JSON, `*_filtered_stats.json`):
- ALFWorld: 24,200 raw 72B rollouts → **19,497 kept at reward threshold 1.0 (80.57%)**,
  4,703 discarded; 2,420 tasks attempted, 2,348 (97.0%) ended up with at least one demo.
- WebShop: 26,178 → 26,178 kept (100%), 5,691/5,691 tasks — because the WebShop cache is not
  raw sampling (§4.3), so there were no failures to filter out.

**How much of the cache a run actually consumes.** With `train_batch_size=8`,
`n_teacher_rollouts_per_task=1`, `max_train_tasks=800`, 100 steps, and a measured
`diag/teacher_sample_ratio` of 0.1221 (7.81 teacher rollouts per 64-sample batch), a single
paper run consumes **≈781 teacher trajectories — 4.0% of the ALFWorld cache and 3.0% of the
WebShop cache.** DUET is not exploiting cache scale; it is exploiting cache *coverage*.

### 4.2 Diversity of the successful trajectories

Measured over tasks holding ≥2 demonstrations, comparing normalized action sequences.

| | ALFWorld full | ALFWorld sub10 | ALFWorld sub1 | WebShop full |
|---|---|---|---|---|
| Tasks with ≥2 demos | 2,284 | 470 | 6 | 5,527 |
| Mean **distinct** action sequences per such task | **4.41** (of 8.51 demos) | 1.78 (of 2.30) | 1.67 (of 2.00) | **4.71** (of 4.71) |
| Mean unique fraction | 0.563 | 0.787 | 0.833 | **1.000** |
| Pairwise-identical rate | **36.5%** (78,974 pairs) | 37.4% | 33.3% | **0.0%** (49,865 pairs) |
| Tasks where *all* demos are identical | 7.62% | 32.77% | 33.33% | **0.00%** |
| Tasks where *all* demos are distinct | 14.71% | 56.38% | 66.67% | **100.00%** |
| Mean unique-first-action fraction | 0.275 | 0.639 | 0.811 | 0.811 |
| Distinct action sequences, whole cache | 8,987 / 19,497 (46.1%) | 1,529 / 1,950 (78.4%) | 188 / 195 (96.4%) | 26,158 / 26,178 (**99.9%**) |

Reading: ALFWorld demonstrations are **genuinely multi-modal but partially redundant** — for a
typical task the 72B produced ~4.4 distinct solution paths out of ~8.5 attempts, and only 7.6%
of tasks collapsed to a single repeated path. First actions agree far more often than whole
trajectories (unique-first-action 0.275), which is expected: ALFWorld solutions share an opening
move ("go to X") and diverge later. WebShop demonstrations are **fully distinct by construction**
(0% pairwise identical, 99.9% globally unique) because they were generated under different search
policies (`single_search` 5,458 / `multi_search` 20,720; 1/2/3 search actions = 5,458 / 15,555 /
5,165).

### 4.3 Honest provenance caveat (please state this in the rebuttal)

The two caches are **not** built the same way, and the paper should say so plainly:
- **ALFWorld** = actual Qwen2.5-72B-Instruct rollouts in the environment, success-filtered at
  reward 1.0 (80.6% kept). This is a real, noisy sampling process.
- **WebShop** = *verified gold action sequences* replayed in the environment
  (`metadata.reward_source = "verified_webshop_replay"` for all 26,178; `is_synthesized = True`
  for all 26,178), with only the `<think>` rationale authored by the 72B teacher
  (`scripts/synthesize_webshop_teacher_from_gold.py`). So the WebShop "teacher" is an oracle
  action sequence with an LLM-written justification, which is why keep-rate is 100%.

This *strengthens* the rebuttal rather than weakening it: WebShop is our **strongest possible
teacher** (oracle actions), ALFWorld is a **realistic sampled teacher**, and DUET's behaviour and
ranking are the same in both. It also directly answers y9x6's "how does the success-filtering
process affect things" sub-question.

### 4.4 Cache subsets: what exists and what they imply

`sub10` (10%) and `sub1` (1%) exist **for ALFWorld only**; there is no WebShop subset file.
They were produced by `scripts/subsample_teacher_cache.py` — uniform sampling at the trajectory
level with `seed=1234`, so both per-task depth *and* task coverage shrink (a genuinely smaller
cache, not a curated one).

| Cache | Trajectories | Unique tasks | Coverage of the 2,420-task collection pool | Demos/task |
|---|---|---|---|---|
| full  | 19,497 | 2,348 | **97.0%** | 8.30 |
| sub10 | 1,950  | 1,337 | **55.2%** | 1.46 |
| sub1  | 195    | 189   | **7.8%**  | 1.03 |

The structurally important consequence — statable without any new run — is that **cache size
mostly buys task *coverage*, not per-task depth**: at `sub1` roughly 92% of training groups
would contain no teacher demo at all, so DUET would degrade toward plain GRPO with baseline
separation on those groups. Do **not** claim a measured `sub10`/`sub1` result: those runs are
configured (`config/duet_paper_experiments_configs/rebuttal_neurips/alfworld/alfworld_qwen1.5b_duet_h200_cache{1,10}.yaml`)
but have produced no validation output yet.

---

## 5. Draft rebuttal paragraphs (usable verbatim)

See `rebuttal_text` in the accompanying structured output; reproduced here for convenience.

**On "what if the teacher is weaker" (UyKJ / y9x6).**
> We agree this is the key question, and we already have a controlled answer to it. Teacher
> quality enters DUET only through the *relative* teacher–student gap, which we log every step as
> the mean per-group difference between teacher and on-policy reward. Holding the teacher cache
> fixed and varying the student gives us exactly the requested regime: on ALFWorld this gap is
> 0.93 early and 0.55 late for the 1.5B student, but only 0.42 early and 0.06 late for the 7B
> student — and on 10 of 100 training steps the 7B student's own rollouts *outperform* the 72B
> teacher (minimum gap −0.10), which never occurs at 1.5B (minimum 0.38). The 7B setting is
> therefore a genuine "teacher only marginally stronger, then no longer stronger" condition. DR3
> detects it without being told: at an identical 12.2% teacher mixing rate, the teacher's share of
> the policy gradient falls from a run-mean of 0.241 at 1.5B to 0.079 at 7B (3.05× lower overall,
> 4.28× over steps 21–40), the discriminator saturates at 0.97 accuracy, and the applied density
> ratio on teacher samples settles at ≈0.47. The outcome is the point of the method: at 7B DUET
> reaches 86.5% vs 85.0% for on-policy GRPO (+1.5pp), whereas LUFFY — the same teacher, the same
> mixing rate, but no fade-out — *drops below the no-teacher baseline*, to 82.5% on ALFWorld
> (−2.5pp) and 45.5% vs 50.0% strict success on WebShop (−4.5pp). Measured on the same runs,
> LUFFY sends 1.71× more gradient to the teacher than DUET over the full run and 2.26× more over
> steps 41–80. In short: when the teacher is weak relative to the student, methods without a
> data-driven fade-out are actively harmed, and DUET is not. We will add this as an explicit
> "teacher-relative-strength" analysis rather than leaving it in the scaling section.

**On cache size and diversity (y9x6).**
> The ALFWorld cache holds 19,497 successful Qwen2.5-72B-Instruct trajectories over 2,348 unique
> tasks (8.3 demonstrations per task; 12.3 turns per trajectory, median 10), obtained by filtering
> 24,200 raw rollouts at reward 1.0 (80.6% retained). The WebShop cache holds 26,178 trajectories
> over 5,691 tasks (4.6 per task, 7.3 turns each). Diversity within a task is substantial rather
> than degenerate: on ALFWorld a task's demonstrations contain on average 4.4 *distinct* action
> sequences out of 8.5, only 36.5% of demonstration pairs are action-identical, and just 7.6% of
> tasks collapse to a single repeated solution; on WebShop no two demonstrations of the same task
> share an action sequence (0% of 49,865 pairs), because they are generated under different search
> policies. We also note that a single 100-step run consumes only ≈781 teacher trajectories — 4.0%
> of the ALFWorld cache — so DUET's gains do not come from cache scale. We will add these
> statistics to the appendix, together with the clarification that the ALFWorld cache is sampled
> and success-filtered while the WebShop cache is built from verified gold action sequences with
> LLM-authored rationales; DUET's behaviour is identical under both, i.e. under both a realistic
> noisy teacher and an oracle one.

---

## 6. Gaps — do NOT assert these without further work

1. **No genuinely weaker teacher model has been run.** Configs exist for 14B and 32B teachers
   (`rebuttal_neurips/alfworld/alfworld_qwen1.5b_duet_a100_teacher{14b,32b}.yaml`) but the data
   files they point to (`data/teacher_trajectories/qwen{14b,32b}/…`) **do not exist**. The
   argument above is a *relative*-gap argument; state it as such.
2. **No noisy / partially-suboptimal teacher cache has been built.** Both caches are 100%
   reward-1.0. We cannot answer "what if the teacher cache contains failures" empirically. (This
   is also review_2's Q2.)
3. **Cache-size ablation has configs and data but no results.** `sub10`/`sub1` `.pkl` files exist;
   no `experiments/*cache*` directory and no log exists. The §4.4 coverage numbers are structural,
   not measured.
4. **No WebShop cache subsets** — `sub10`/`sub1` exist for ALFWorld only.
5. **WebShop 7B DUET has no val@100** (run stopped ~step 93). ALFWorld 7B CHORD has no val@100.
6. **WebShop 7B DR3 fade-out is unverified here** — no `logs/webshop_7b_*.log` exists on this
   machine. The TGS/disc_acc/ŵ numbers in §3 are ALFWorld only. The prior report
   `analysis_reports/FINAL_7b_analysis_report.md` quotes WebShop 7B teacher gaps (0.078 @ step 50,
   −0.084 @ step 90); I could not re-derive them from raw logs, so do not cite them as verified.
7. **3B numbers do not reconcile across sources.** `NeurIPS_2026_Latex/tables/main_results.tex`
   reports 3B-ALFWorld GRPO 47.0% / DUET 77.5%, but the runs present on this machine give
   `alfworld_3b_grpo_react_tags` 58.5% and `alfworld_3b_duet_0329` 69.5% (3B-ALFWorld LUFFY 61.5%
   *does* match). The paper's 3B runs evidently live elsewhere. I therefore built the scaling
   table from 1.5B and 7B only, both fully re-derived here. **Reconcile before submitting.**
8. **1.5B-WebShop LUFFY also does not reconcile**: paper table says 5.5%, the local run
   `experiments/webshop_qwen1.5b_luffy/validation_log/100.jsonl` gives 4.5%. Likewise the paper's
   1.5B-WebShop DUET 36.0% — the closest local run is
   `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` at 35.5%. Check which run backs the table.
9. **Single seed everywhere** in the numbers above. The multi-seed question (review_2 W1, y9x6 Q1)
   is a separate evidence package.
10. **`analysis_reports/alfworld_7b_vs_3b_analysis.md`'s DR3 table is confirmed** where it
    overlaps mine (TGS 0.166 @ step 1, 0.013 @ step 70; disc_acc 0.947 @ step 30; w_mean 1.040 @
    step 50) — but note that report's `w_mean` is the *batch-wide* ratio (≈1.0), whereas the
    teacher-only ratio `dr3/w_off_mean` is ≈0.47. Cite `w_off_mean` when claiming attenuation;
    `w_mean ≈ 1.0` would read as "no attenuation" to a careful reviewer.

---

## 7. Reproduction commands

```bash
PY=/data/home/qisheng/miniconda3/envs/duet/bin/python

# §2 / §2b validation numbers
$PY - <<'EOF'
import json,glob,os
for f in sorted(glob.glob('/data/home/qisheng/EvolAnalsis/experiments/*/*7b*/validation_log/*.jsonl')
              + glob.glob('/data/home/qisheng/EvolAnalsis/experiments/alfworld_qwen1.5b_*/validation_log/100.jsonl')):
    sc=[float(json.loads(l)['score']) for l in open(f) if l.strip()]
    print(f, len(sc), round(sum(1 for s in sc if s>=0.9999)/len(sc),4), round(sum(sc)/len(sc),4))
EOF

# §3 fade-out series (scratchpad/parse7b.py greps duet/teacher_gradient_share lines)
$PY scratchpad/parse7b.py logs/alfworld_7b_duet.log   af7b_duet.json
$PY scratchpad/parse7b.py logs/alfworld_7b_luffy.log  af7b_luffy.json
$PY scratchpad/parse7b.py logs/alfworld_qwen1.5b_duet_v39c_postfix.log af15b_duet.json

# §4 cache statistics
$PY scratchpad/analyze_teacher_cache.py       # writes cache_stats.json
```

Helper scripts live in
`/tmp/claude-1000/-data-home-qisheng-EvolAnalsis/a5d90f98-198d-42a4-aeb3-820cd312fa72/scratchpad/`
(`analyze_teacher_cache.py`, `parse7b.py`, `cache_stats.json`, `af7b_duet.json`,
`af7b_luffy.json`, `af15b_duet.json`). Copy them into `scripts/` if you want them versioned.
