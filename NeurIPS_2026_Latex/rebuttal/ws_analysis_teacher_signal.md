# WebShop 1.5B: does the teacher signal teach the long path, and does it reach the policy?

Audit date 2026-07-26. All paths relative to `/data/home/qisheng/EvolAnalsis`.
Read-only analysis; nothing was launched, killed, or modified.

Runs compared (the three from the seed forensics):

| label | checkpoint dir | seed | task draw | val@100 strict / mean |
|---|---|---|---|---|
| `paper` | `webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06` | 2026 | its own 800 | 35.5% / 0.706 |
| `seed2027` | `webshop_qwen1.5b_duet_a100_seed2027` | 2027 | `task_seed 2026` → same 800 | 2.5% / 0.544 |
| `seed2025` | `webshop_qwen1.5b_duet_a100_seed2025` | 2025 | different 800 | 3.5% / 0.521 |

Verified independently here: the union of training task ids over 100 steps is **800 vs 800 with
800 shared** for `paper` vs `seed2027`, and **89 shared** for `paper` vs `seed2025` — matching
`rebuttal/README.md`. (Per-step *ordering* differs in all pairs: 0/100 steps have identical
per-step task sets, so the two same-curriculum runs see the same tasks in a different order.)

Metric time series were parsed from the per-step lines in
`logs/webshop_qwen1.5b_duet_{swC_02_pk03_v10_floor06,a100_seed2025,a100_seed2027}.log`
(100/100 steps recovered per run, ~395 keys). Behavioural measurements come from
`checkpoints/agentevolver/<run>/Trajectory/trajectories_step_*.jsonl`, teacher rows excluded via
`diag.is_teacher` unless stated.

---

## TL;DR

1. **The skill is in the teacher data, and BC does put weight on it.** 38.7–38.9% of all teacher
   tokens that the BC/SFT loss sees are the tokens of option-click turns — the largest single
   category. This is identical in all three runs.
2. **But BC's weight on those tokens is cut 3× and reaches its floor ~30 steps before the skill
   even starts to recover.** `chord/mu` decays 0.295 → 0.11 by step 40 and sits at the 0.10 floor
   thereafter; the option-clicking rate bottoms out at step 51–60 and only rises after step 70.
   BC's share of the total actor loss falls from ~0.17 to ~0.05 over the same window.
3. **DR3 cannot be the differentiator.** `dr3/w_off_mean` is 0.660 (paper) vs 0.669 (seed2027) in
   the last 10 steps — a 1% difference. The higher `duet/teacher_gradient_share` in seed2027 is an
   *advantage-magnitude* statistic computed before DR3's weight is applied, and it is mechanically
   larger when the student is worse. It is not "more teaching".
4. **The State Channel is exactly blind to this skill, and provably so.** In the WebShop
   observation stream, **91.4% of option-click transitions leave the observation text
   byte-identical**. Any potential Φ(s) — the paper's Eq. 8 teacher-visit potential included, not
   only the hand-written `attribute_aware` one — therefore has ΔΦ = 0 across the option-clicking
   segment. Measured: mean ΔΦ = **−0.004** for option clicks (93.6% exactly zero) versus **+0.558**
   for item clicks and **+0.113** for searches.
5. **The apparent SC "reward" for option clicking is an averaging artifact, not a signal.**
   Episodes with ≥2 unique options have P(τ) higher by +0.142; collapse consecutive *duplicate*
   observations and the gap becomes **+0.0005**. 99.6% of it comes from re-averaging the same
   product page more times, not from progress.

This is a concrete, fixable design gap and it is the cleanest mechanistic result in the WebShop
seed investigation: the environment reward is the *only* one of DUET's three channels that credits
the behaviour which decides strict success, and it credits it only at the very end of the episode.

---

## 1. Action Channel (BC): the skill is in the data, the weight is not

### 1.1 What BC actually is here

`config: actor_rollout_ref.actor.use_chord: true`, `chord_use_token_weighting: false`
(`launcher_record/webshop_qwen1.5b_duet_a100_seed2027/yaml_backup.yaml:3-11,68`). With token
weighting off, `compute_chord_sft_loss` (`agentevolver/module/exp_manager/het_core_algos.py:1767`)
reduces to a **uniform token-mean NLL over the expert-masked tokens**, scaled by μ(t). Confirmed in
the logs: `chord/phi_mean = phi_min = phi_max = 1.0` at every step, and
`chord/weighted_sft_loss = chord/mu × chord/sft_loss` to 3 decimals at every block.

So "which teacher tokens teach the long path" is answered by counting tokens, with no reweighting.

### 1.2 Teacher-token composition (tokenized with the run's own tokenizer)

Every teacher row actually mixed into a batch over all 100 steps, assistant turns only (this is
exactly the BC loss support), tokenized with `/data/shared_models/Qwen2.5-1.5B-Instruct`.
Teacher assistant messages contain no `<think>` block — they are `<action>…</action>` only — so
these are the action tokens.

| run | teacher rows | tokens | search | item click | **option click** | nav | buy now | preamble |
|---|---|---|---|---|---|---|---|---|
| paper | 686 | 211,864 | 17.9% | 8.1% | **38.7%** | 4.6% | 25.6% | 5.2% |
| seed2027 | 686 | 211,871 | 17.7% | 8.1% | **38.9%** | 4.6% | 25.5% | 5.2% |
| seed2025 | 668 | 205,766 | 17.4% | 8.2% | **38.7%** | 4.7% | 25.8% | 5.2% |

Per-run teacher behaviour, measured on the rows actually sampled (not the cache):
**7.25 actions/trajectory, 2.10 unique options before `click[Buy Now]`, 89.5% click ≥2** — identical
to 2 decimal places in `paper` and `seed2027`, and 7.30 / 2.09 / 89.4% in `seed2025`.

**Verdict: the behaviour is in the data and BC's loss mass is concentrated on it.** Option-click
turns are the single largest token category — larger than search, item-click and nav combined
(30.6%). Nothing distinguishes the runs here: `paper` and `seed2027` differ by 7 tokens out of
211,864 in aggregate composition.

Note this is *not* because the two runs saw the same teacher demonstrations. With
`select_mode: random` and `max_trajectories_per_task: 6`, the per-task draw differs: only **1 of 686**
teacher rows is shared between `paper` and `seed2027` (0 of 686 with `seed2025`), yet the aggregate
option content is statistically indistinguishable. The teacher cache is homogeneous with respect to
this skill.

### 1.3 μ(t) versus the option-clicking curve

μ is `chord_mu_peak 0.3`, `chord_mu_valley 0.1`, `chord_mu_adaptive_mode: disc_acc`,
`chord_mu_d_floor 0.6`, i.e.
μ = 0.1 + 0.2·clamp((1 − EMA[`dr3/disc_acc`]) / 0.4, 0, 1) (`het_actor.py:1795-1806`). Since
`dr3/disc_acc` → 0.97 by steps 31–40 and 0.99 thereafter, the gate closes and μ pins to its floor.

Per 10-step block (block = steps 1-10, 11-20, …):

```
chord/mu                paper     0.295 0.228 0.176 0.124 0.103 0.102 0.101 0.101 0.102 0.105
                        seed2027  0.293 0.257 0.181 0.118 0.106 0.103 0.101 0.103 0.105 0.105
                        seed2025  0.298 0.243 0.167 0.123 0.113 0.110 0.106 0.103 0.106 0.108
BC share of actor loss  paper     0.163 0.116 0.148 0.094 0.081 0.084 0.050 0.072 0.087 0.058
 |wSFT|/(|wSFT|+|grpo|) seed2027  0.185 0.159 0.151 0.102 0.070 0.075 0.075 0.055 0.080 0.044
                        seed2025  0.179 0.171 0.123 0.154 0.104 0.074 0.097 0.076 0.095 0.095
unique options/episode  paper     0.44  0.84  0.16  0.20  0.10  0.08  0.14  0.35  0.73  1.18
 (on-policy, pre-buy)   seed2027  0.47  0.60  0.29  0.20  0.12  0.07  0.19  0.22  0.29  0.26
                        seed2025  0.44  0.79  0.37  0.22  0.16  0.24  0.19  0.31  0.93  1.03
% episodes with ≥2 opts paper     10.8  20.3   3.0   4.2   1.9   1.6   2.5   6.3  20.1  35.9
                        seed2027  10.5  13.8   5.9   4.7   2.3   1.2   3.8   3.5   5.2   2.4
                        seed2025  10.0  20.3   7.6   4.8   2.1   4.0   2.8   5.3  18.0  21.1
```

μ hits ~0.11 (within 10% of its floor) at step 40 in all three runs; the first step at which
μ ≤ 0.101 is 56 (paper) / 63 (seed2027) / never (seed2025, min 0.102). The option-clicking curve
bottoms at steps 51–60 and any recovery happens entirely in the regime where BC is at its floor and
contributes ~5% of the actor loss.

**Verdict on hypothesis 1: partially confirmed, with the direction the angle predicted, but this is
not what separates the runs.** BC's teaching signal for exactly this skill is attenuated 3× by step
40 and never restored — the fade-out is driven by `dr3/disc_acc`, which measures whether teacher and
student *sequences are distinguishable*, not whether the student has *acquired the option-clicking
skill*. That is a real design criticism (the fade-out criterion is orthogonal to skill acquisition).
But μ's trajectory is within 0.01 across all three runs at every block, so it cannot explain why one
run escaped and two did not.

---

## 2. DR3: same weight, and the gradient-share difference points the other way

`dr3/w_off_mean` per 10-step block:

```
paper     0.996 0.949 0.871 0.739 0.670 0.604 0.648 0.639 0.663 0.660
seed2027  0.993 0.954 0.915 0.733 0.679 0.635 0.626 0.654 0.668 0.669
seed2025  0.994 0.952 0.880 0.764 0.712 0.717 0.609 0.617 0.708 0.661
```

Identical to ~1%. `dr3/w_clipfrac_off = 0.000` at every step in every run, so no clipping
differentiates them either.

**Can DR3 concentrate weight on teacher trajectories that demonstrate the long path? No — it is
structurally incapable of it.** `compute_sequence_features`
(`agentevolver/module/exp_manager/dr3_ratio.py:73-165`) builds the discriminator input from
per-sequence *log-probability shape statistics only*: `lp_mean`, `lp_std`, `lp_min`, `lp_max`,
low/high-logp token fractions, `resp_len`, and KL-to-ref summaries. No feature encodes which actions
the trajectory took. DR3's ŵ can only answer "does this sequence look like the student's own
distribution", never "does this sequence demonstrate option clicking". Whatever DR3 does, it does it
uniformly with respect to the skill.

(A per-trajectory spread cannot be read off the logs either: `dr3/w_off_std` is 0.000 at almost
every step, but that is because `dr3/total_sample_count = 2` and `dr3/teacher_samples_micro ≤ 2` —
usually ≤1 teacher row per micro-batch, so the std is trivially 0. `dr3/logw_applied_std` is 0.000
for the same reason, with `dr3/applied_sample_count = 1`. This is a "cannot determine from logs",
not evidence of a scalar ŵ.)

### Why `duet/teacher_gradient_share` was higher in the worse run

`metrics["duet/teacher_gradient_share"] = Σ|adv_teacher| / (Σ|adv_teacher| + Σ|adv_onpolicy|)`
computed on per-sequence mean advantages (`ae_ray_trainer.py:3764-3766`). It is a **post-advantage,
pre-DR3** statistic: DR3's ŵ shifts `old_log_prob`, not the advantage, so ŵ does not enter this
number at all. It is a function of the GRPO group structure, in which teacher rows carry reward 1.0
and on-policy rows carry the student's reward.

Last-block numbers:

| | teacher \|adv\| | on-policy \|adv\| | grad share | on-policy reward |
|---|---|---|---|---|
| paper 91-100 | 0.152 | 0.190 | 0.089 | 0.654 |
| seed2027 91-100 | 0.213 | 0.179 | 0.134 | 0.590 |

seed2027's larger share comes from a **larger teacher advantage magnitude** (0.213 vs 0.152), which
is what a group produces when the on-policy members are worse relative to the teacher. Per-step
correlations, steps 21–100:
`corr(teacher_gradient_share, group_teacher_minus_on_reward_mean)` = +0.382 / +0.135 / +0.225
(paper / seed2027 / seed2025), and `corr(teacher_gradient_share, reward_onpolicy_mean)` =
−0.329 / −0.003 / −0.140 (pooled over the three runs, steps 21–100, n=240: **−0.165**).
The correlations are weak within a run, so this is a mechanism argument supported by a weak trend,
not a strong statistical fit; but the sign is consistent and the causal direction is unambiguous
from the definition.

**Verdict on hypothesis 2: refuted as stated.** seed2027 did not "receive more teacher gradient and
imitate less". `teacher_gradient_share` is partly an *inverse proxy for student competence*, so a
worse run mechanically scores higher on it. The runs are indistinguishable on DR3's actual output
(ŵ) and on the content of the teacher rows they sampled. **This is worth a sentence in the paper:
we currently present `teacher_gradient_share` decaying 50%→5% as evidence of a "data-driven teacher
curriculum". Part of that decay is the student getting better, which shrinks the teacher-minus-student
reward gap. The metric is not a clean read-out of DR3's fade-out.**

---

## 3. State Channel: Φ is flat over exactly the decisive skill

WebShop uses `algorithm.state_channel.match_mode: attribute_aware`
(`yaml_backup.yaml:188-197`, β = 0.2, `exclude_teacher: true`, `step_level.enable: false`), i.e.
`webshop_attribute_aware_potential` (`state_progress.py:252-265`):

```
Φ(obs) = WEBSHOP_ATTR_AWARE_STAGE[page_type(obs)]        # search_home 0.0 | search_results 0.15
                                                          # product_detail 0.35 | purchase_complete 1.0
       + 0.50 · (#instruction attributes AVAILABLE on the page / #checkable)   [product_detail only]
```

Read `compute_attribute_match_score` (`state_progress.py:205-249`) closely: it parses the target
attributes out of the instruction and checks whether each one **appears in the page's option list**
(`attrs['color'] in product['options']['color']`). It asks *is the right colour offered*, never
*has the right colour been clicked*. Φ is therefore identical before and after any option click on
the same product page, **by construction**.

### 3.1 The environment makes this unavoidable for any Φ(s)

Measured on option-click transitions in the saved rollouts (every 3rd step, all rows):

| run | option-click transitions | observation text byte-identical before→after | both sides `product_detail` |
|---|---|---|---|
| paper | 2,580 | **91.4%** | 91.4% |
| seed2027 | 1,847 | **91.1%** | 91.1% |

The WebShop observation does not encode which option buttons are currently selected. Clicking
`click[khaki]` returns the same page text. So **no state potential Φ(s) can assign different values
before and after a correct option click** — not the hand-written `attribute_aware` one, and not the
paper's Eq. 8 teacher-visit potential, which would hash both observations to the same key and hence
to the same Φ. This is a property of the observation space, not a bug in our potential.

### 3.2 Measured ΔΦ by action type (recomputed with the shipped code)

Mean Φ(s_{t+1}) − Φ(s_t), attributed to the action taken at t:

| run | search | item_click | **option_click** | nav | buy_now |
|---|---|---|---|---|---|
| paper | +0.113 (n=3842) | +0.558 (n=3090) | **−0.004 (n=3438), 93.6% exactly 0** | −0.105 | +0.000 |
| seed2027 | +0.117 (n=3761) | +0.563 (n=3011) | **−0.004 (n=3224), 93.2% exactly 0** | −0.098 | +0.000 |
| seed2025 | +0.114 (n=3901) | +0.549 (n=3126) | **+0.000 (n=4878), 92.0% exactly 0** | −0.113 | −0.016 |

The residual non-zero option-click transitions are dominated by
`product_detail → search_home` (invalid clicks that bounce), which carry a **negative** ΔΦ. So to the
extent Φ moves at all on option clicks, it moves *down*.

Splitting option clicks by whether the clicked value matches a required instruction attribute:

| run | clicks a required attribute | clicks something else |
|---|---|---|
| paper | n=2243, mean ΔΦ = **−0.010** | n=1061, mean ΔΦ = −0.054 |
| seed2027 | n=2193, mean ΔΦ = **−0.006** | n=934, mean ΔΦ = −0.073 |
| seed2025 | n=3089, mean ΔΦ = **−0.010** | n=1574, mean ΔΦ = −0.068 |

Clicking the *right* colour gets Φ ≈ 0. Clicking the wrong one gets Φ ≈ 0 too (slightly negative,
from bounces). **The State Channel is not merely silent on the skill — it cannot distinguish a
correct option selection from an incorrect one.**

**Verdict on hypothesis 3: confirmed. This is the concrete, fixable design gap.**

---

## 4. Is the SC bonus at least incidentally higher for episodes that click options?

`diag.sc_bonus` and `diag.sc_progress` are saved per rollout. Verified: `sc_bonus = 0.2 ×
sc_progress` exactly, and `sc_progress = mean_t Φ(s_t)` over
`extract_observations_from_steps(messages, "webshop", skip_initial=3)` — I reproduced
`diag.sc_progress` to <1e-6 on **84.5% of 10,285 on-policy rollouts** using the shipped code, with a
reproduction rate that is *flat in option count* (86.2% / 84.0% / 83.0% / 80.8% for 0 / 1 / 2 / 3+
options), so the 15.5% residual does not bias what follows.

### 4.1 Raw comparison: yes, but small

On-policy rows only, per 10-step block, split by unique options clicked before `Buy Now`:

```
paper     block:      1-10   11-20  21-30  31-40  41-50  51-60  61-70  71-80  81-90  91-100
  sc_bonus, ≤1 opt:  0.0330 0.0637 0.0658 0.0682 0.0699 0.0745 0.0730 0.0762 0.0726 0.0746
  sc_bonus, ≥2 opts: 0.0798 0.0745 0.0692 0.0629 0.0816 0.0703 0.1021 0.0934 0.1042 0.0932
  Δ:                 +.0468 +.0109 +.0034 -.0053 +.0117 -.0041 +.0291 +.0172 +.0317 +.0186
seed2027  Δ:         +.0483 +.0214 +.0093 +.0105 +.0333 +.0267 +.0367 +.0225 +.0304 +.0346
seed2025  Δ:         +.0444 +.0136 +.0217 -.0162 +.0235 +.0463 +.0354 +.0282 +.0304 +.0343
```

Pooled over all 100 steps and restricted to episodes that reached a product page:
paper 0.0904 vs 0.0737, seed2027 0.0906 vs 0.0730, seed2025 0.0958 vs 0.0712. So the ≥2-option group
does receive ~+0.017–0.025 more shaped reward. Note this is *the same in the failing run as in the
succeeding one* — SC treats the skill identically in both, so it cannot be the differentiator either.

### 4.2 The difference is a duplicate-state averaging artifact

P(τ) is the *mean* of Φ over observations. An option click adds one more observation at the
plateau value Φ ≈ 0.85 without changing it, which raises the mean purely by diluting the low-Φ
prefix. Test: recompute P(τ) after collapsing **consecutive identical observations** (steps 41–100,
on-policy, episodes that reached a product page):

| run | raw P(τ) ≤1 opt | raw P(τ) ≥2 opts | raw Δ | dedup ≤1 | dedup ≥2 | **dedup Δ** |
|---|---|---|---|---|---|---|
| paper | 0.4514 (n=2889) | 0.5933 (n=384) | +0.1419 | 0.4328 | 0.4332 | **+0.0005** |
| seed2027 | 0.4448 (n=3105) | 0.5799 (n=105) | +0.1351 | 0.4329 | 0.4197 | **−0.0132** |
| seed2025 | 0.4533 (n=2944) | 0.6109 (n=301) | +0.1576 | 0.4177 | 0.4312 | **+0.0134** |

Restricting to the 84.5% of episodes whose `sc_progress` I reproduce exactly gives the same answer
for the paper run: raw Δ +0.1421 → dedup Δ +0.0015.

**99.6% of the paper run's SC "reward" for option clicking survives only because the same product
page is averaged in twice.** An episode that clicks the same wrong button twice, or emits any no-op
that keeps it on the product page, collects exactly the same bonus. As a further control, matching
on the number of `product_detail` observations and then splitting by option count gives an
inconsistent sign (paper: −0.157 at n_pd=2, +0.020 at n_pd=3, +0.083 at n_pd=4).

**Verdict on hypothesis 4: SC is neutral to the skill.** The positive raw correlation is real but is
a dwell/averaging artifact of the mean aggregation, not a gradient toward option selection.

---

## 5. What this means

Of DUET's three channels, on WebShop:

| channel | does it teach "click every required option before buying"? |
|---|---|
| Action Channel (BC) | **Yes**, 38.7% of its token mass — but at μ = 0.10 (its floor, 3× down from peak) from step ~40, contributing ~5% of the actor loss during the entire window in which the skill has to be relearned. The fade-out is triggered by `dr3/disc_acc`, which is orthogonal to skill acquisition. |
| DR3 | **No.** Its features are log-prob shape statistics with no action content; ŵ is 0.66 vs 0.67 across runs; `teacher_gradient_share` differences track student competence, not teaching. |
| State Channel | **No, and it cannot.** ΔΦ = 0 on 91–94% of option clicks; correct and incorrect option clicks are indistinguishable; the +0.017 bonus for ≥2 options is a duplicate-observation averaging artifact (+0.0005 after dedup). |

So the environment's terminal reward is the only channel that credits the behaviour that decides
strict success. That is consistent with the local-optimum reading in the seed forensics: escaping
`search → click item → buy` requires a run of lucky terminal rewards, because nothing in DUET's
shaping makes the longer path locally attractive. It also explains why the escape is a *phase
transition* rather than a steady climb — the only teaching signal for the skill is (a) a 5%-of-loss
NLL term and (b) a sparse end-of-episode reward.

Caveat that must be stated: SC/BC/DR3 behave the same in all three runs, so **none of this explains
which run escaped**. It explains why escaping is hard and rare, not why it happened once. And
`seed2025` reaches 0.93–1.03 options/episode at steps 81–100 (close to `paper`'s 1.18) while
scoring 3.5% strict — option clicking is necessary but not sufficient for strict success.

---

## Reproduction

Scripts (scratchpad, read-only inputs):
`/tmp/claude-1000/-data-home-qisheng-EvolAnalsis/a5d90f98-198d-42a4-aeb3-820cd312fa72/scratchpad/{parse_logs.py,teacher_signal.py,teacher_signal2.py,ws_lib.py}`.
`ws_lib.py` (option/item/buy click classification) is shared with the sibling rollout analysis, so
option counts here are on the same definition as `data/webshop_seed_sensitivity.md`.

Code read for this audit:
- `agentevolver/module/exp_manager/state_progress.py:29-53` (stage tables), `:56-84`
  (`classify_webshop_page`), `:96-140` (`extract_instruction_attributes`), `:205-265`
  (`compute_attribute_match_score`, `webshop_attribute_aware_potential`), `:568-589`
  (`extract_observations_from_steps`, `skip_initial=3`), `:904-928`
  (`compute_trajectory_progress`, `agg_mode="mean"`), `:942-966` (coverage = 1.0 for
  `attribute_aware` when the task is in the map).
- `agentevolver/module/exp_manager/het_actor.py:1763-1815` (μ from `dr3/disc_acc`).
- `agentevolver/module/exp_manager/het_core_algos.py:1767-1827` (`compute_chord_sft_loss`).
- `agentevolver/module/exp_manager/dr3_ratio.py:73-165` (`compute_sequence_features`), `:920-932`
  (`w_off_*` metrics).
- `agentevolver/module/trainer/ae_ray_trainer.py:3747-3766` (`duet/teacher_gradient_share`).

## Gaps

- The per-teacher-row ŵ is not saved, and there is usually ≤1 teacher row per micro-batch, so
  "is DR3's weight concentrated on particular teacher trajectories" cannot be answered from disk.
  It is answered *structurally* (the features carry no action content), not empirically.
- 15.5% of on-policy rollouts do not reproduce `diag.sc_progress` bit-exactly from the saved
  `messages`; the mismatch rate is flat in option count and the dedup conclusion is unchanged when
  restricted to the exact subset, but the residual is unexplained.
- `duet/teacher_gradient_share` vs on-policy reward correlates only weakly within a run
  (−0.33 / −0.00 / −0.14, steps 21–100). The claim that the share partly tracks student competence
  rests on the metric's definition plus a weak trend, not on a strong fit.
- Whether repairing Φ would actually change the outcome is untested. Nothing here is an end-to-end
  result.
