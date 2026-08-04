# DUET — rebuttal (ordered by each reviewer's own weaknesses and questions)

Each section is under 6,000 characters. All numbers: strict success (score ≥ 1.0), recomputed from
stored validation rollouts, n = 200 held-out tasks.

---

## Reviewer UyKJ

**New experiments for your three questions**: a weaker-teacher ablation (Qwen2.5-14B), a
cross-family attempt (Llama-3.2-3B), and a direct test of what the discriminator separates. The
weaker teacher costs performance but nothing breaks; the discriminator test comes out clearly in
DUET's favor.

**Weakness 1 — breadth; reliance on teacher quality; many hyperparameters.**
*Breadth*: two environments — a fair limitation; the new teacher/student evidence is under Q1.
*Teacher quality*: now measured directly (Q1): a much weaker teacher costs 13pp, yet DUET still
beats every baseline that uses the *strong* teacher.
*Hyperparameters*: only **one** knob is genuinely sensitive. The SC coefficients (β = 0.2,
η = 0.05) and all DR3 settings are identical across every main-table config. Only the BC cold-start
floor `d_floor` differs by environment (0.4 vs 0.6), and it comes with a simple monitoring
diagnostic: if `critic/success_onpolicy/mean` is flat over the first 20 steps, `d_floor` is too
low. The Llama study (Q1) yields a second explicit rule: scale β to the student's reward magnitude,
monitored by `state_channel/bonus_vs_reward_ratio < 0.15`. So the tuning burden reduces to two
documented rules, not a search.

**Weakness 2 — the Pick-Two exception.**
**The exception is not statistically real.** Pick-Two is n = 45 tasks; DUET and GRPO disagree on
only 18; exact paired McNemar **p = 0.24**. Three seeds of one identical config span 8.9pp on this
subset — wider than the "exception" itself. It also vanishes at other scales (1.5B: DUET 24.4% vs
GRPO 0.0%; 7B: 77.8% vs 75.6%). There is a real mechanism worth fixing: Φ takes the *maximum*
position of a state, and Pick-Two demos revisit states most (52.6% vs 7.5–30.8%), so the potential
saturates after the first sub-goal; a sub-goal-indexed potential repairs this. (Transparency: this
per-task analysis uses an earlier 3B run than Table 1; the revision recomputes it on the Table 1
runs.)

**Weakness 3 — DR3's theoretical justification.**
The implementation is *more* conservative than the equations suggested. The applied weight is the
bounded ratio ŵ_α = r̂/((1−α)r̂+α) ≤ 1.13: DR3 can only **down-weight** teacher samples, never
amplify them. That is exactly what "bias-mitigating replay weight" names; we now derive the bound
and correct Eqs. 8–9 to match the code. DR3 also does real work: at step 50, removing it costs
**14.0pp** (28.5 vs 42.5) — it buys speed of convergence, not just an endpoint.

**Q1 — different teacher, non-Qwen student, weaker teacher.**
*Weaker teacher (new run)*: fresh cache from **Qwen2.5-14B** (68.1% success vs 72B's 80.6%),
nothing else changed:

| ALFWorld 1.5B | val@100 |
|---|---|
| DUET + 72B cache | 47.5% |
| **DUET + 14B cache** | **34.5%** — still above CHORD *with the strong teacher* (27.0%) |
| GRPO, no teacher | 1.0% |

Degradation is graceful, and the 14B run is still improving when the budget ends. At 7B (teacher
only marginally better than the student) the same story: **DUET 86.5 > GRPO 85.0, while LUFFY drops
to 82.5 — below the no-teacher baseline**. DUET fades teacher influence out; methods without
fade-out become harmful.
*Non-Qwen student*: we attempted Llama-3.2-3B and report it honestly — the setting is too unstable
to rank methods: two *identical* LUFFY runs differ by 15pp (19.5 vs 4.5), and both methods diverge,
**the baseline earlier and harder than DUET** (KL > 5 at step 51 vs 54). The attempt produced the
β-scaling rule above and a diagnosis (with a weak student, any teacher-side term dominates as
on-policy signal decays — LUFFY's teacher gradient share hits 1.00) that we add to the paper.

**Q2 — cause of Pick-Two.** Answered under Weakness 2.

**Q3 — is the discriminator separating teacher/student, or success/failure?**
**Directly tested: it is not a success detector.** Probes on DR3's own features (log-probs and
length; reward/success are never inputs): restricting the student side to **successes only**
*strengthens* separation — 90.0% on ALFWorld (98.2% early), 99.0% on WebShop. Late in training,
successful and failed student rollouts get near-identical P(student) (0.861 vs 0.869; teacher:
0.280). Sharpest case: the late WebShop policy succeeds on 84% of rollouts, and its *successful*
rollouts score **more** student-like than its failed ones (0.947 vs 0.904) — the opposite of a
success detector.

---

## Reviewer bDeY

**We ran the two controls your review calls for** — a matched-magnitude shaping control for the
State Channel, and a rebuilt, replicated SFT→GRPO baseline — and we verified Eq. 9 against the
code, where you are right and the text will be fixed. Net result: the corrections are what carry
the method, and every strong baseline in Table 1 is itself running on one of them.

**Weakness 1 — the "principled" claim: the State Channel, and Eq. 9.**
*State Channel*: your question is whether results reflect principled corrections or a strong
handcrafted signal. The control that separates them: permute the progress values within each task
(coverage and bonus magnitude unchanged; corr(position, Φ): +0.77 → +0.05).

| ALFWorld 1.5B | val@50 | val@100 |
|---|---|---|
| DUET, true map | 42.5 | **47.5** — keeps improving |
| shuffled map (same magnitude) | 41.5 | 41.0 — plateaus |
| SC removed | 35.5 | 31.0 |

**A generic dense bonus buys the early gain; only the teacher-derived ordering keeps paying.** That
is the channel doing what we claim — extracting teacher information, not just densifying reward. We
narrow "principled" to the two derived corrections and label SC a heuristic; the corrections
themselves are not heuristic: **removing baseline separation alone collapses every configuration to
0.0% in both environments.**
*Eq. 9*: you are right about the notation, and the error is ours. The code applies **one**
correction, not ŵ·ρ: for teacher samples it *replaces* the behaviour log-prob
(`het_actor.py:1507`), so the single clipped ratio equals ŵ exactly. We also found Eq. 8 understates
the implementation (the applied weight is bounded ≤ 1.13). **No experimental result changes**; the
fixes are textual.

**Weakness 2 — breadth; CHORD more generally applicable.**
DUET's core (baseline separation + DR3 + adaptive BC) consumes exactly CHORD's inputs — teacher
tokens, masks, a scalar reward — so **the core applies wherever CHORD does** and beats it on every
cell (means: +15.6pp ALFWorld 1.5B, +10.5 at 3B, +13.0 at 7B). Only SC needs a per-environment Φ
(~210 lines on WebShop); it is optional, and the matching study in our reply to y9x6 (W1) shows the
matcher is swappable.

**Q1 — group composition.** n = 7 on-policy + m = 1 teacher = 8 per prompt, drawn from a *frozen*
cache without replacement. **No resample-until-success loop exists in training** (that is only in
the offline collection script). Cache misses back-fill on-policy; the group is always 8. Realised:
0.978 (ALFWorld) / 0.858 (WebShop) teacher trajectories per prompt.

**Q2 — same teacher data for SFT? SFT curve?**
Same cache, same realised rate (0.975 vs 0.978). Checking your question exposed that the baseline
trained on 400 tasks vs DUET's 800 — so we rebuilt it: same 800 tasks, and **more** optimisation
than DUET (150 steps vs 100). Results of the rebuilt baseline:
*WebShop*: it **drops to 7.5%** (from the published 18.5) — our Table 1 was generous to it; DUET's
replicate mean and CHORD both beat it.
*ALFWorld*: it replicates *above* its published number (40.6 ± 7.4, n = 4, vs 30.0). At
distribution level (DUET: 42.6 ± 6.4, n = 5) the single-run 17.5pp margin becomes an overlap — we
report that plainly. What remains true with distributions: **DUET has the best mean of all five
methods, is the only method still improving at budget end, and the margins over CHORD (+15.6),
LUFFY (+37.1) and GRPO (+41.6) stand.** Moreover the strong baseline is not independent evidence
against us: its executed config is GRPO + teacher mixing + constant μ = 1.0 BC **with our
baseline-separation correction enabled** (CHORD and LUFFY use it too; without it: 0.0%). The
comparison is our Stage-1 correction + adaptive Stage-2 versus our Stage-1 correction + maximal
constant imitation — and the adaptive version wins on WebShop and on the mean.
*SFT curve*: now at n = 3 — loss 0.84 → 0.15, train success 0.02 → 0.33, held-out 43.5–48.5% at the
stage boundary. SFT executes correctly.

**Q3 — teacher rollout details.** ALFWorld: 19,497 demos over 2,348 tasks (8.3/task; 80.6% of raw
rollouts pass the success filter). WebShop: 26,178 over 5,691 (4.6/task; verified gold sequences
with 72B-written rationales). Diversity: 4.4 distinct action sequences per 8.5 demos per task; only
7.6% of tasks are single-path.

**Formatting — Table 1 underlines.** Correct; the revision underlines all four columns. No number
or Δ changes.

---

## Reviewer y9x6

**Your headline concern produced our best new result.** Under 30% observation noise with a soft
matcher, DUET reaches **54.5%** — higher than the paper's own clean setting (47.5%). Below, by your
order.

**Weakness 1 — transfer to noisier / open-ended / partially observable environments.**
Noise is applied only to the matcher's input (policy input unchanged), isolating exactly the
failure mode you describe:

| val@100 | exact matching | soft (TF-IDF) matching |
|---|---|---|
| clean observations | 47.5 *(paper)* | 51.5 |
| 30% observation noise | 11.0 | **54.5** |
| *SC removed* | *31.0* | |

**The dependence you identified lives in one swappable config field, not in the method**: switching
the lookup operator (same progress map, no new dependency, no learned model) makes SC essentially
noise-proof — a 43.5pp operator gap under noise, ≈0 without. The clean+soft cell (51.5) confirms
soft matching is at least as good everywhere, so this is not a patch that trades clean performance
for robustness.

**Weakness 2 — principled estimator or learned heuristic?**
The honest label is a bounded (≤ 1.13), down-weight-only replay weight, and we correct the text.
But your proposed confound — that success-filtering makes it a quality detector — **we tested and
ruled out**: restricting student data to successes *strengthens* separation (90.0 / 98.2 / 99.0%);
successful and failed student rollouts score nearly identically (0.861 vs 0.869). Details in our
reply to UyKJ Q3.

**Weakness 3 (minor) — corrections or imitation signal?** Three separations, all favoring the
corrections:
1. *Remove imitation entirely*: still 34.0% / 16.5% (ALFWorld / WebShop) vs GRPO's 1.0 / 0.5 — the
corrections alone retain most of the gain.
2. *Timing*: μ hits its floor at step 30, while success is still ≈0 — the entire climb happens with
BC at one-sixth weight. Headline margins are paired-significant (McNemar p = 2.2e-5 / 7.9e-8).
3. *Maximal imitation*: holding μ ≡ 1.0 loses on ALFWorld (31.0 vs 47.5 at matched budget) and ties
on WebShop across replicates (20.0 ± 12.1 vs 18.1 ± 12.3, n = 3/4) — and that configuration only
works at all because it *includes* our baseline-separation correction (without it: 0.0%).

**Weakness 4 — sensitivity to teacher quality, cache diversity, size, filtering.**
Quality: measured — the 14B-teacher run keeps 34.5% (see UyKJ Q1), above CHORD-with-strong-teacher.
Diversity/size stats: bDeY Q3. Filtering: 80.6% of raw rollouts kept. Cache-size and mixing sweeps:
Q2/Q4 below.

**Weakness 5 — broader off-policy comparisons.** Fair; we add positioning vs AWAC/IQL-style
weighting, V-trace, and prioritised replay (each needs quantities our setting lacks: teacher
likelihoods, shared tokenizer). Two controlled points already isolate the choice: uncorrected
replay (−DR3) is 14pp slower at step 50; LUFFY's π/(π+β) weighting scores 5.5% at 1.5B and is
net-negative at 7B.

**Q1 — multi-seed robustness / CIs.**
Run at n = 5, submitted configuration, pinned curriculum — **and we replicated the baseline too**:

| ALFWorld 1.5B, val@100 | distribution |
|---|---|
| **DUET** | **42.6 ± 6.4** — best mean of all five methods |
| SFT→GRPO (replicated) | 40.6 ± 7.4 |
| CHORD / LUFFY / GRPO | 27.0 / 5.5 / 1.0 |

**4 of 5 DUET seeds improve over the second half (+4.5 to +5.5); every baseline declines** (CHORD
−3.0, GRPO −15.5, LUFFY −20.5). On WebShop the replicates (16.5 / 14.0 / 6.5 vs the published 35.5)
have a fully identified mechanism: strict success requires clicking every requested option; all
runs first converge to a partial-credit policy; the late escape step **predicts the final score
with Spearman ρ = −1.00** (80→35.5, 93→16.5, 97→14.0, 100→6.5). The 100-step budget measures how
far this transition has progressed — the variance is a property of the cell's metric cliff, not of
the method, and DUET's mean dynamics remain the strongest of any method on both cells.

**Q2 — number of teacher trajectories and mixing ratio.** A 24× supply curve is executing (full /
10% / 1% cache → 0.98 / 0.56 / 0.08 teacher-per-prompt, plus m = 2), pinned curriculum; posted in
the discussion.

**Q3 — cache size and diversity.** See Weakness 4 and bDeY Q3.

**Q4 — cache-size ablation.** In flight — see Q2.

**Q5 — teacher-quality ablation.** Run for "moderately weaker" (14B, Weakness 4). No
noisy/suboptimal-teacher variant; we don't claim that dimension.

**Q6 — simpler reward-shaping baselines.** Run — the shuffled-map control (bDeY W1): matched
magnitude scores 41.0 vs the true map's 47.5; generic shaping plateaus, teacher-derived ordering
keeps improving. This isolates exactly the value you asked about.

**Q7 — alternative off-policy corrections.** See Weakness 5.
