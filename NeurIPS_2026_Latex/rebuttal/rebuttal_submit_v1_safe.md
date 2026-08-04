# DUET — rebuttal, trimmed to the per-reviewer limit

Each section below is under 6,000 characters including spaces. Source with full detail and
provenance: `rebuttal_draft.md`. **Protocol for every number: strict success (score ≥ 1.0),
recomputed from stored validation rollouts, n = 200.**

---

## Reviewer UyKJ

**Q3 — is the discriminator separating teacher/student, or success/failure?** Directly tested; it is
not a success detector. Probes on DR3's own features (log-prob statistics and length; reward,
advantage and success are never inputs) show that restricting the student side to **successes only**
does not weaken separation: 90.0% on ALFWorld (98.2% early), 99.0% on WebShop late. Late in ALFWorld
training, successful and failed student rollouts get near-identical P(student), 0.861 vs 0.869,
against 0.280 for teacher trajectories. WebShop late is the sharpest test — that student already
succeeds on 84% of rollouts — and there *successful* rollouts score **more** student-like than failed
ones (0.947 vs 0.904). Caveat: an offline probe of DR3's features, not the online discriminator.

**W4/Q2 — the Pick-Two exception.** The cell is too small to carry the interpretation we gave it:
n = 45 of 200, DUET and GRPO disagree on 18, exact paired McNemar **p = 0.238**, and three seeds of
one identical configuration span 8.9pp on that subset. It does not replicate off the cell (1.5B:
24.4% vs 0.0%; 7B: 77.8% vs 75.6%). One real limitation sits behind it: Φ is the *maximum*
normalised index at which a state occurs, and Pick-Two demonstrations are the most self-revisiting
type (52.6% contain a repeated state vs 7.5–30.8%), so the potential saturates inside the first
sub-goal. The fix is a sub-goal-indexed potential.

**W5 — DR3's justification.** Eqs. 8–9 misdescribe the implementation and both are corrected. The
applied weight is the bounded α-relative ratio ŵ_α = r̂/((1−α)r̂+α) ≤ 1/(1−α) ≈ 1.13, so DR3 can only
ever *down-weight* a teacher sample, never amplify one. Bounded and down-weight-only is exactly what
"bias-mitigating replay weight" names; we keep the term and now derive the bound.

**W2/Q1 — a weaker teacher.** Run. A fresh cache from **Qwen2.5-14B-Instruct** (4,094 demos over 727
of the 800 training tasks; 68.1% success rate against 72B's 80.6%), nothing else changed:

| ALFWorld 1.5B | val@50 | val@100 |
|---|---|---|
| DUET + 72B cache (paper) | 42.5% | **47.5%** |
| **DUET + 14B cache** | 33.5% | **34.5%** |
| GRPO (no teacher) | 16.5% | 1.0% |

DUET degrades with teacher quality — 13pp, we do not claim insensitivity — but gracefully: it still
exceeds CHORD *using the strong teacher* (27.0%) and is still improving at the end of the budget
(33.5 → 34.5) where every uncorrected method declines. A weaker teacher costs performance, not
stability.

**W1/Q1 — a non-Qwen student.** Run, and we report a negative result. On Llama-3.2-3B with the
unchanged 72B cache, **the setting cannot support a ranking**: two *nominally identical* LUFFY runs —
same config, same `data.seed`, same 800 task IDs verified from saved rollouts, functionally identical
code — gave **19.5%** (stable) and **4.5%** (diverged). DUET's run gave 8.5%, between them. Both DUET
and LUFFY then diverged numerically (KL > 5 at steps 54 and **51**; peak 5.16 and **7.58**), the
baseline earlier and harder. Any ordering we quoted would be an artifact.

What the setting *does* show, because it happens to both methods, is a real limitation. As the weak
student's on-policy success decays, the teacher share of the policy gradient climbs — to peaks of 0.91 (DUET)
and **1.00** (LUFFY) — against a median of 0.21 and a maximum of 0.62 across the whole Qwen
ALFWorld run, which never exceeds 0.9.
Neither auxiliary channel is scaled to the strength of the student's own learning signal. Porting
also required retuning the State Channel: with the ALFWorld β carried over, `bonus_vs_reward_ratio`
hit 0.20–0.67 against a healthy band of < 0.15 and the policy converged on a no-op that freezes the
observation (`inventory`, 68% of turns; 0 of 1736 turns with a valid action). **We ran the control
that could have refuted that diagnosis** — restoring the discriminator clock to the paper's recipe
with β back at 0.2, changing nothing else — and it collapses identically, scoring **0.0% (0/200)**
at val@50 with 26 consecutive zero-success steps. The State Channel dominance is the cause, not the
clock. Scaling β down 4× lifts the same setting to 8.5%. So **DUET as published does not transfer to
a student this much weaker without retuning SC** — a limitation of that module, but a repairable
one, and the diagnostic is a metric we already log.

**W3 — many hyperparameters.** SC coefficients (β = 0.2, η = 0.05) and DR3 settings (temperature 1.5,
2 steps/call) are *identical* across both main-table configs. The BC μ schedule is not: ALFWorld
(0.3, 0.05, d_floor 0.4, EMA 0.5), WebShop (0.3, 0.10, **0.6**, 0.2) — we state this rather than
imply one setting. Because the two environments' best d_floor values straddle the range we give a
diagnostic instead of a default: a flat `critic/success_onpolicy/mean` over the first 20 steps means
d_floor is too low. The Llama failure adds a second, concrete rule: **scale β to the student's reward
magnitude**, monitored by `state_channel/bonus_vs_reward_ratio` < 0.15.

---

## Reviewer bDeY

**W2 — Eq. 9 double-counts the density ratio.** You are right, and the error is ours. The code never
forms ŵρ_t. For teacher samples it *replaces* the behaviour log-probability (`het_actor.py:1507`):
log π̂_β := sg[log π_θ] − log ŵ(τ), so the single clipped ratio evaluates to ŵ(τ) exactly — one
correction, not two. ρ_t (Eq. 7) applies to on-policy samples only. We had prepared a defence arguing
the product telescopes; three independent code audits refuted it and we discarded it. Two further
errors we found ourselves, both against us: Eq. 8's D/(1−D) is only an intermediate (the applied
weight is bounded ≤ 1.13), and the WebShop configs run a policy-shaping variant Eq. 9 does not print.
**No experimental result changes**; the edits are Eqs. 8–9, an appendix variant, and the contribution
statement, which now describes the *diagnosis* of two biases rather than a derivation.

**W1 — is a hand-designed shaping signal doing the work?** We ran the control that separates them:
the progress values are permuted among each task's own states, holding coverage and bonus magnitude
fixed while corr(position, Φ) collapses from +0.772 to +0.045.

| ALFWorld 1.5B | val@50 | val@100 |
|---|---|---|
| DUET, true map | 42.5% | **47.5%** |
| shuffled map (matched magnitude) | 41.5% | **41.0%** |
| SC removed | 35.5% | 31.0% |
| CHORD | 30.0% | 27.0% |

**Of SC's 16.5pp benefit, ~10.0pp is available from any dense bonus of that magnitude; only ~6.5pp
needs the teacher-derived ordering.** Your reading is partly right and we will say so in the paper.
The residual is the durable part: the shuffled control matches the true map at step 50 and then
flattens, while the true map keeps improving. We also report where SC ablation hurts us: −SC gives
31.0% against SFT+GRPO's 30.0% on ALFWorld (parity), and 1.0% against CHORD's 11.5% on WebShop.

**Q2 — is SFT+GRPO given the same teacher data, and the SFT curve?** Same cache, same n_teacher, same
realised per-prompt rate (0.975 vs 0.978). **But it was not task-matched: 400 tasks against DUET's
800.** We found this ourselves and reran it on the same 800 tasks with *more* optimisation than DUET
(50 SFT + 100 GRPO = 150 steps vs 100). Result: **7.5%, mean reward 0.403** — *below* the 18.5% we
published, so our Table 1 number was generous to the baseline, and below both DUET's replicate mean
(18.7%) and CHORD. Caveats we volunteer: this baseline entered a partial format collapse (malformed
actions 0% → 27%), which suppresses strict success — but its mean reward fell too (0.640 → 0.393), so
the degradation is real, not only formatting. And at step 50, where both are format-clean, the
ordering reverses (baseline 8.0% vs DUET 1.0–1.5%) — a point that flatters the baseline structurally,
since DUET's escape from the partial-credit optimum happens at steps 80–100 and the baseline has by
then had 100 optimisation steps to DUET's 50. SFT stage executes correctly: ALFWorld on-policy success
0.016 → 0.285 pre-RL, `chord/sft_loss` 0.844 → 0.153.

**W3 — breadth, and CHORD's generality.** DUET's core (baseline separation + DR3 + adaptive BC)
consumes only teacher tokens, masks and a scalar reward — the same inputs CHORD needs — so the core
applies wherever CHORD does; it is SC that carries the extra requirement, and WebShop's Φ is ~210
lines of hand-written per-environment code. We have no math/reasoning result and will not claim one.
See the reply to y9x6 for the matching experiment that bounds SC's dependence.

**Q1 — group composition.** n = 7 on-policy + m = 1 teacher = 8 per prompt, from a *frozen* cache,
sampled without replacement from a fixed per-task list, touching no environment or LLM. **No
resample-until-success loop exists in training** — that is in the offline collection script only. A
cache miss back-fills on-policy, so the group is always 8. Realised: 0.978 (ALFWorld), 0.858
(WebShop) teacher trajectories per prompt.

**Q3 — cache details.** ALFWorld 19,497 demos over 2,348 tasks (8.30/task), filtered from 24,200 raw
rollouts (80.6% kept); WebShop 26,178 over 5,691 (4.60). Teacher Qwen2.5-72B-Instruct, successes only.
Diversity: 4.4 distinct action sequences per 8.5 demos; 7.6% single-path. Disclosure: the WebShop
entries are verified gold sequences replayed with 72B rationales, not sampled rollouts.

**Table 1 underlines.** Correct — CHORD is strongest non-DUET in both 3B columns and neither was
underlined. The revision underlines all four; no number or Δ changes.

---

## Reviewer y9x6

**W1 — noisier, open-ended, partially observable environments.** Your concern is right in a sharper
form than posed: a *failed* matcher is actively harmful, not merely uninformative. Noise is applied
only to the matcher's copy of the observation, so task difficulty is fixed. Same 800 train / 200 val
tasks; the only change is the lookup operator:

| val@100 | exact matching | soft (TF-IDF) matching |
|---|---|---|
| clean observations | 47.5% *(paper)* | **51.5%** |
| 30% observation noise | **11.0%** | **54.5%** |
| *SC removed entirely* | *31.0%* | |

Under noise, exact matching scores *below deleting the module*, because a missed state is scored
Φ = 0 rather than abstaining. We ran the clean+soft cell precisely because without it we could not
separate "soft matching repairs noise" from "soft matching is simply better"; it is at least as good
on clean observations (51.5 vs 47.5, inside our ≈5pp seed spread, so we claim only parity there).
**The unambiguous result is the interaction: a 43.5pp gap between operators under noise against ≈0
without it.** The dependence is a property of a replaceable component — one config field, the same
progress map, no new dependency, no learned model — not of the method.

**W2 — principled estimator or learned heuristic?** Heuristic, and we correct the paper: a bounded
(≤ 1.13), down-weight-only replay weight, not an exact likelihood ratio. Your specific confound —
that success-filtering makes the discriminator a quality detector — is testable, so we tested it: see
the probe results in our reply to UyKJ. Restricting the student side to successes *strengthens*
separation, and late in training successful and failed student rollouts score almost identically.

**W3 — the corrections, or the extra imitation?** Two directions. *Removing* imitation entirely still
gives 34.0% (ALFWorld) and 16.5% (WebShop) against GRPO's 1.0%/0.5%. *Timing*: μ reaches its floor at
**step 30**, when on-policy success is still 0.000 — the entire climb happens while BC carries
one-sixth of its peak weight. Headline margins are paired-significant (McNemar p = 2.2e-5, 7.9e-8).

**We must report the case against us here, and be precise about what it does and does not touch.**
Holding μ at full weight for the whole run is worse at matched budget on ALFWorld (48.0% @50 →
**31.0%** @100 vs DUET's 47.5%), but on WebShop the same configuration reaches **30.0%**, above our
own replicate mean of 12.3%. We checked whether that could be a lucky draw from a wide distribution
and **it cannot**: it escapes the partial-credit optimum at step 57, 23 steps earlier than any DUET
run — a systematic difference, not a lucky draw.

What this does *not* touch is Stage 1. That BC-only configuration **includes our baseline-separation
correction**, and ablating that correction alone gives **0.0%** on ALFWorld and **0.0%** on WebShop,
against 0.5% for on-policy GRPO. So the ordering on WebShop is: baseline separation + full imitation
30.0%, baseline separation + our full machinery 35.5% (replicates 16.5/14.0/6.5), baseline separation
+ CHORD's decaying schedule 11.5%, and without baseline separation everything collapses. The
defensible reading is that **the Stage-1 correction is doing the load-bearing work, and on this cell
we cannot show that the Stage-2 adaptive schedule improves on simply keeping imitation on.**

**Q1 — multi-seed.** ALFWorld at the submitted configuration, curriculum pinned: **47.5 / 41.0 / 50.5**
at val@100 (seed spread ≈5pp), and **every seed improves over the second half** (+4.5 to +5.0), which
no baseline does. 1.5B-WebShop reproduces only partially and we report all of it: against 35.5%,
pinned replicates give **16.5 / 14.0 / 6.5**, and curriculum-redrawn ones 3.5 / 2.5. `data.seed` sets
both run randomness *and* which 800 tasks are drawn; we added `data.task_seed` to separate them.
Mechanism: strict success needs every requested attribute clicked, 88.4% of tasks request ≥ 2, and
every run first converges on a partial-credit 3-action policy (0 successes in 10,735 under-clicking
episodes). Escape is a late event, and **escape step predicts the final score almost perfectly across
the four runs sharing the paper's task draw — 80 → 35.5%, 93 → 16.5%, 97 → 14.0%, 100 → 6.5%,
Spearman ρ = −1.00.** A 100-step budget measures how far a late phase transition has progressed, not
an asymptote. We do not claim the submitted number is representative; it is the earliest-escaping run
of four.

**Q3–Q5 — cache size, diversity, quality.** Statistics in our reply to bDeY. Teacher quality is run
(14B cache: 34.5% vs 47.5%, still far above the no-teacher 1.0%). Cache-size and mixing-ratio sweeps
are in progress and we will post them during the discussion period rather than assert a result now. We have no *noisy* or *deliberately suboptimal* teacher and will not claim that
dimension.

**W5/Q7 — broader off-policy comparisons.** Not run, and we say so. Controlled points we can offer:
uncorrected replay (−DR3) 47.5% / 9.5%, and LUFFY's π/(π+β) weighting 5.5% / 5.5% at 1.5B and
net-negative at 7B. The revision positions DR3 against AWAC/IQL advantage weighting, V-trace
truncation and prioritised replay, stating which quantities each needs that we lack (teacher
likelihoods, a shared tokenizer).
