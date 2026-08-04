# DUET — OpenReview rebuttal (English, translated from the Chinese master draft)

Six official comments: UyKJ ×2, bDeY ×2, y9x6 ×2 (each reviewer's reply split into Weaknesses and
Questions so the writing does not need to be compressed). Markdown tables and `$$` MathJax
verified; no em-dashes. All numbers: strict success (score ≥ 1.0), n = 200 held-out tasks.

---

## → Reviewer UyKJ, Part 1 of 2 (Weaknesses)

We sincerely thank the reviewer for the positive assessment and the insightful questions. During
the rebuttal period we ran three new sets of experiments targeted at them: a teacher-quality
ablation (72B to 14B), a cross-family experiment (a Llama-3.2-3B student driven by the Qwen
teacher), and a direct test of what the discriminator actually separates. To keep each comment
readable we reply in two parts, the weaknesses here and the questions in Part 2.

**Weakness 1: reliance on teacher quality; many hyperparameters.**

On teacher quality, we would first like to clarify the positioning. DUET's research goal is
precisely how to make the best use of a stronger teacher or expert model, so the availability of a
teacher is the problem setting itself rather than a hidden assumption of the method. The question
that genuinely needs answering, as the reviewer implies, is whether DUET's utilization mechanism
stays robust when teacher quality changes. We added a teacher-quality ablation for this, with
every other setting unchanged:

| ALFWorld 1.5B, val@100 | strong teacher (72B, 80.6% sample success) | weak teacher (14B, 68.1%) |
|---|---|---|
| **DUET** | **47.5%** | **34.5%** |
| CHORD (strong teacher) | 27.0% | n/a |
| LUFFY (strong teacher) | 5.5% | n/a |
| GRPO (no teacher) | 1.0% | 1.0% |

Three takeaways from this table:

1. Even after switching to a much weaker teacher, DUET still leads every baseline that uses the
strong teacher (34.5% against CHORD's 27.0% and LUFFY's 5.5%);
2. The 14B run is still improving at the end of the budget, so the degradation is graceful and the
mechanism does not fail;
3. On the 7B student, where the teacher's relative advantage has nearly vanished, DUET reaches
86.5 against GRPO's 85.0, while LUFFY falls to 82.5, below the no-teacher baseline. DUET's
adaptive machinery retires the teacher automatically once it stops being useful.

Performance scales smoothly with teacher quality and neither end breaks down, which we take as
direct evidence of robust teacher utilization.

On hyperparameters, DUET's design in fact goes the opposite way: it replaces hand-tuned schedules
with closed-form adaptive quantities, reducing rather than increasing the tuning burden:

- the BC weight μ is driven by discriminator accuracy through a closed-form rule, so no
hand-designed decay schedule (as CHORD requires) is needed;
- the teacher weight decays automatically as the student approaches the teacher, so no manual
annealing is needed;
- the coefficients of both channels and all DR3 settings are identical across every main-table
configuration, with zero per-environment tuning.

The only per-environment quantity is a single scalar, the BC cold-start floor `d_floor`, set once
and never touched during training.

**Weakness 2: the Pick-Two exception could be explained more clearly.**

We agree, and the phenomenon has a clean mechanistic explanation that localizes to one reparable
detail of the potential. Our Φ takes the maximum normalized position at which a state appears in
the demonstration, while Pick-Two is the task type that revisits states most often (52.6% of its
demonstrations contain a repeated state, against 7.5–30.8% for the other types): "put down item A,
then go find item B" naturally revisits earlier states. As a result, Φ saturates early within the
first sub-goal, and the shaping loses its discrimination for the second sub-goal.

This explains why the phenomenon appears only in Pick-Two, and also why it is not general evidence
that teacher demonstrations harm combinatorial exploration: at the other two scales, DUET leads on
the very same task type (1.5B: 24.4% against 0.0% for GRPO; 7B: 77.8% against 75.6%). The fix is
direct, indexing Φ by sub-goal, a local improvement inside the module, and the revision will
include this analysis.

**Weakness 3: DR3's theoretical justification.**

Discriminator-based density-ratio estimation is a standard technique with a mature theoretical
basis: under a Bayes-optimal discriminator, D/(1−D) is exactly the likelihood ratio. On top of
this, the two treatments for the success-filtered cache are deliberate and, we believe,
theoretically consistent:

1. **The correction targets exactly the replayed distribution.** What an off-policy correction
needs is the ratio of the policy to the actual sampling source, namely the success-filtered cache,
and the discriminator is trained precisely between cache samples and policy samples, so it
estimates exactly this correct object rather than a ratio to the raw teacher policy.
2. **The implementation uses the smoothed, bounded form** $\hat w_\alpha = \hat r/((1-\alpha)\hat r+\alpha)$,
which belongs to the same family of variance control as PPO clipping and V-trace truncation: it
reduces to D/(1−D) as α → 0, while guaranteeing the weight never exceeds 1.13, so teacher samples
can only be down-weighted and never amplified. This is the precise sense of the phrase
"bias-mitigating replay weight", and the revision will write the smoothed form out explicitly to
avoid ambiguity.

DR3's contribution is also directly measurable: removing it costs **14.0pp at step 50** (28.5
against 42.5). What it buys is speed of convergence, not merely an endpoint number.

---

## → Reviewer UyKJ, Part 2 of 2 (Questions)

Continuing our reply, we address the three questions below.

**Q1: a different or weaker teacher; a non-Qwen student.**

For a weaker teacher, the direct evidence is the table under Weakness 1: with the teacher's sample
success rate dropping by 12.5 points (80.6% to 68.1%), DUET at 34.5% still leads all baselines
that use the strong teacher. Together with the 7B evidence, where the teacher's relative advantage
nearly vanishes yet DUET still benefits smoothly while LUFFY turns harmful, this answers "what
happens when the teacher is weaker": performance scales smoothly with teacher quality, and the
utilization mechanism does not fail.

For a non-Qwen student, we ran a cross-family experiment with a **Llama-3.2-3B student and the
unchanged Qwen-72B teacher cache**. The only adaptation was a one-time setting of the SC
coefficient to the new student's initial reward scale, in line with the set-once principle
described under Weakness 1. Training has been stable throughout, and at the step-50 checkpoint
DUET reaches **15.0%, close to three times the no-teacher baseline (GRPO, 5.5%)**. This suggests
that teacher-experience utilization holds across model families: Qwen demonstrations effectively
drive a Llama student. The full 100-step results will be posted during the discussion period.

**Q2: what causes the Pick-Two exception.**

Please see our response to Weakness 2 in Part 1.

**Q3: is the discriminator separating teacher from student, or success from failure?**

This is a fair concern, since the cache indeed contains only successful trajectories. We designed
a direct test around the following logic: if what the discriminator has learned is really "success
versus failure", then after restricting the student side to successful trajectories only, both
sides consist of successes, and the separability should collapse.

The outcome is the opposite, with three pieces of evidence pointing the same way:

1. Training probes on the features DR3 actually uses (log-probability statistics and length;
reward and success labels are never inputs), restricting the student side to successes makes
teacher/student separability rise rather than fall: from 84.7% to 90.0% on ALFWorld over the whole
run, and from 94.0% to 99.0% on late-stage WebShop;
2. Late in training, successful and failed student trajectories receive nearly identical
P(student), 0.861 against 0.869, while teacher trajectories sit at 0.280: the features the
discriminator uses simply do not track success; they track the policy's fingerprint;
3. The most stringent cell: the late-stage WebShop policy already succeeds on 84% of its rollouts.
If the discriminator were a success detector, successful trajectories should look more
teacher-like; in fact they look more student-like than the failed ones (0.947 against 0.904), the
opposite direction.

We conclude that **the discriminator separates policy identity, not task outcome**. We hope these
new results address the reviewer's concerns, and we would be glad to run further analyses during
the discussion period.

---

## → Reviewer bDeY, Part 1 of 2 (Weaknesses)

We sincerely thank the reviewer for the rigorous technical review; the comment on Eq. 9 in
particular prompted us to re-derive the objective against the implementation line by line. During
the rebuttal period we also added the ablations that quantify the standalone contribution of the
principled corrections. To keep each comment readable we reply in two parts, the weaknesses here
and the questions in Part 2.

**Weakness 1: the "principled" claim; the State Channel and Eq. 9.**

The core of the question, as we read it, is whether the principled corrections themselves make a
large difference, or whether the gains mainly come from the shaping signal. The corrections'
independent contribution can be quantified directly by ablation. Three results, all measured
without depending on or modifying SC:

1. **Baseline separation (the correction for Bias 1)**: removing it alone drops every
configuration in both environments to **0.0%**. We would also point out that all teacher-mixing
baselines in Table 1 (SFT+GRPO, CHORD, LUFFY) train with this correction enabled as well; it is
the lowest load-bearing piece of the whole comparison, and even the baselines' numbers are built
on it.
2. **Removing SC entirely (no shaping signal at all)**: the core of corrections plus adaptive BC
still reaches **31.0%** on ALFWorld, against LUFFY's 5.5% and GRPO's 1.0%. A gain of more than
25 points comes purely from the corrections, with shaping playing no part.
3. **DR3 (the correction for Bias 2)**: removing it alone, with SC present, costs 14.0pp at step
50 on ALFWorld (28.5 against 42.5) and drops WebShop from 35.5 to **9.5**. The corrections carry
weight independently in both environments.

The results therefore support both statements at once, rather than one against the other: **the
corrections independently deliver the bulk of the gain, and SC further exploits the state-progress
information that comes for free with teacher trajectories** (in WebShop, whose reward structure is
the sparsest, this progress information is naturally worth more). The two are complementary
stages, correct then extract, not substitutes for each other.

On Eq. 9, we thank the reviewer for pressing on this point. The double-counting derived in the
review does not arise in the algorithm, and the key lies in the convention for the denominator of
$\rho_t$. We reference all samples, teacher and on-policy alike, against the previous-iterate
student policy $\pi_{\mathrm{old}}$ (the teacher's likelihood is unavailable in the first place,
which is exactly why DR3 is introduced), not against each sample's own behavior policy.
Concretely, for teacher samples the behavior log-probability is imputed by definition as

$$\log \hat\pi_\beta := \log \pi_{\mathrm{old}} - \log \hat w ,$$

so that the only ratio in the clipped objective is

$$\rho_t = \frac{\pi_\theta}{\hat\pi_\beta} = \frac{\pi_\theta}{\pi_{\mathrm{old}}}\cdot \hat w \approx \frac{\pi_\theta}{\pi_{\mathrm{old}}}\cdot\frac{\pi_{\mathrm{old}}}{q} = \frac{\pi_\theta}{q},$$

where $q$ is the cache (teacher) distribution and $\hat w \approx \pi_{\mathrm{old}}/q$ comes from
the discriminator. The two factors each cover one part: the drift term
$\pi_\theta/\pi_{\mathrm{old}}$ can be computed exactly, while the mismatch term
$\pi_{\mathrm{old}}/q$ cannot be computed exactly and is estimated by the discriminator. Composed,
they give precisely the full importance weight $\pi_\theta/q$ that a teacher sample requires,
**each factor appearing exactly once, with no duplication** (the same computable-versus-estimated
decomposition as V-trace; clipping acts on the composite ratio, which places the trust region
directly on the true weight). If the denominator of $\rho_t$ is instead read as each sample's own
behavior policy, one indeed arrives at the double-counting described in the review; the revision
will write out the imputation definition and the denominator convention explicitly so that the
formula and the algorithm correspond one to one. We are grateful for the push to make this
precise.

**Weakness 2: evaluation breadth; adaptation for domains like math; CHORD more generally
applicable.**

We would first like to make the problem setting explicit. DUET targets **multi-turn agentic
tasks**: the agent interacts with the environment repeatedly, and the observations form an
explicit trajectory. In this setting, states exist natively inside the interaction trajectory and
the progress table is built from it automatically; the large majority of multi-turn agentic tasks
(web operation, embodied household, tool use) have this structure. Mathematical reasoning is a
single-turn problem without environment interaction, a different problem class that lies outside
the scope of this paper. Within the intended scope, we believe DUET's generality is not inferior
to CHORD's: the DUET core (baseline separation, DR3, adaptive BC) consumes exactly the same inputs
as CHORD, namely teacher tokens, masks and a scalar reward, so **wherever CHORD is applicable, the
core is directly applicable**, and it is stronger on all four cells of Table 1 (+20.5, +24.5,
+10.5, +6.5pp).

Second, on "relying on exact state matching": this is exactly what we tested during the rebuttal,
and it is not a requirement of the method. The State Channel's matching operator is replaceable.
We swapped it for text-similarity soft matching (same progress table, no new dependencies) and
stress-tested with 30% word-dropout noise on the matcher's input:

| val@100 | exact matching | soft matching |
|---|---|---|
| clean observations | 47.5 (paper setting) | 51.5 |
| 30% observation noise | 11.0 | **54.5** |

Soft matching is no worse than exact matching on clean observations, and is nearly unaffected
under strong noise (**54.5%, above our own noise-free setting**). For environments with noisier or
weaker structure, the required "adaptation" is switching one matching operator, not substantial
re-engineering.

---

## → Reviewer bDeY, Part 2 of 2 (Questions and formatting)

Continuing our reply, we address the three questions and the formatting point below.

**Q1: the actual group composition; is m fixed, or resample-until-success?**

Each prompt is fixed at n = 7 on-policy plus m = 1 teacher, 8 in total. Teacher trajectories are
drawn **without replacement from a frozen cache**; the training process touches no environment and
calls no LLM. **There is no resample-until-success loop in training**; that loop exists only in
the offline data-collection stage. For tasks the cache does not cover, the slot automatically
back-fills with an on-policy rollout, so the group size is always 8. The realized teacher
trajectories per prompt are 0.978 (ALFWorld) and 0.858 (WebShop).

**Q2: does SFT use the same amount of teacher data? Can you add the SFT curve?**

Yes: the same cache file and the same sampling mode, with the realized per-prompt teacher ratio
matched to DUET's (0.975 against 0.978). The optimization budget is controlled as well: the
baseline's **total step count equals DUET's exactly** (50 SFT + 50 GRPO = 100 steps, against
DUET's 100), following the standard SFT-then-RL pipeline of warm-starting and then RL-finetuning
on the same task distribution. The SFT stage executes correctly: loss falls 0.84 → 0.15 and
training success rises 0.02 → 0.33 (the curve will be included in the revision). We additionally
verified that this budget does not shortchange the baseline: doubling its task count and raising
the total steps to 150 (against DUET's 100) in fact lowers it on WebShop to **7.5%** (published:
18.5%), so **Table 1 already reports this baseline at its stronger configuration**.

**Q3: details of the teacher rollouts.**

ALFWorld: 19,497 successful demonstrations covering 2,348 tasks (8.3 per task on average; 80.6% of
raw samples pass the success filter). WebShop: 26,178 demonstrations covering 5,691 tasks (4.6 per
task); the entries are environment-verified successful action sequences paired with 72B-generated
rationales. On diversity: of the 8.5 demonstrations per task on average, 4.4 have distinct action
sequences, and only 7.6% of tasks are single-path.

**Formatting: missing underlines in Table 1.**

The reviewer is right, and we apologize for the oversight. The revision underlines all four
columns; all numbers and the Δ row are unchanged. We hope these clarifications and new controls
address the concerns, and we remain happy to provide further details during the discussion.

---

## → Reviewer y9x6, Part 1 of 2 (Weaknesses 1–3)

We sincerely thank the reviewer for the thorough and constructive review. We spent much of the
rebuttal period on the experiments it called for, and the generalisation concern listed first in
fact produced our best new result: **under 30% observation noise, DUET reaches 54.5%, above our
own noise-free setting (47.5%)**. Given the number of points, we reply in two comments:
Weaknesses 1–3 here, and Weaknesses 4–5 together with the questions in Part 2.

**Weakness 1: transfer to noisier, more open-ended, partially observable environments.**

We stress-tested exactly the link the reviewer identified, end to end. The two operations,
concretely:

- **Noise**: each word of the state matcher's copy of the observation is independently dropped
with probability 0.3 (word dropout); the policy network's input is left untouched. Task difficulty
is therefore unchanged, and the only thing perturbed is the state-matching step the reviewer
pointed to;
- **Soft matching**: the default exact-hash lookup is replaced by TF-IDF text-similarity retrieval
over the same progress table, taking the most similar stored state above a threshold. One
configuration field, no new dependencies, no learned components.

End to end, on the same 800 training and 200 validation tasks:

| val@100 | exact matching | soft matching |
|---|---|---|
| clean observations | 47.5 (paper setting) | 51.5 |
| 30% observation noise | 11.0 | **54.5** |
| SC removed | 31.0 | |

The conclusion: **the dependence the reviewer identified lives in a replaceable configuration
component, not in the method itself**. With the text-similarity matcher (same progress table, no
new dependencies, no learned model), SC becomes essentially immune to the noise; and the
clean-plus-soft cell (51.5) shows this is not a patch that trades clean performance for
robustness, since soft matching is never worse than exact matching. The stress test also shows the
framework's extensibility: facing noisier environments, one only needs to swap the matching
operator.

**Weakness 2: a principled density-ratio estimator, or a learned heuristic?**

At the theory level, discriminator-based density-ratio estimation is a standard technique with a
mature basis: under a Bayes-optimal discriminator, D/(1−D) is exactly the likelihood ratio; the
ratio an off-policy correction needs is relative to the *actually replayed* distribution (the
success-filtered cache), which is precisely the distribution the discriminator is trained against;
and the implementation uses a smoothed, bounded form (variance control of the same family as PPO
clipping and V-trace truncation), which guarantees teacher samples are only ever down-weighted.

The deeper concern raised here, that success filtering might make the discriminator capture
trajectory quality rather than policy mismatch, is experimentally testable, and we designed a
supplementary probe experiment for it. The setup: we train probes on the features the
discriminator actually consumes (log-probability statistics and sequence length; reward and
success labels are **never provided**) to separate teacher trajectories from student trajectories,
and we compare two conditions: the student side using all rollouts, and the student side using
successful rollouts only. The logic of the test: if what the discriminator has learned is in
effect a success detector, then in the second condition both sides consist of successes and the
separation accuracy should collapse. The result is exactly the opposite: on ALFWorld the probe
accuracy **rises** from 84.7% to 90.0%, and in late-stage WebShop (where the policy already
succeeds on 84% of its rollouts, making it the most stringent test scenario) it rises from 94.0%
to 99.0%. As a further check, successful and failed student trajectories receive nearly identical
student-probability (0.861 against 0.869), while teacher trajectories sit at 0.280. These features
track *which policy produced the trajectory*, not *whether it succeeded*: DR3's weight reflects
policy mismatch, not trajectory quality.

**Weakness 3 (minor): do the gains come from the correction mechanisms, or from the imitation
signal?**

Two independent separation experiments, both pointing to the corrections. The core comparison
(ALFWorld 1.5B, equal budget, same curriculum):

| Imitation setting | val@50 | val@100 |
|---|---|---|
| **DUET (adaptive μ: high start, automatic decay)** | 42.5 | **47.5** |
| Imitation removed (μ = 0) | 40.5 | 34.0 |
| GRPO (no teacher) | 16.5 | 1.0 |

1. **Removing the imitation term entirely still retains most of the gain** (34.0 against GRPO's
1.0 on ALFWorld, with a second seed reaching 42.0; likewise 16.5 against 0.5 on WebShop): the
correction mechanisms are effective on their own, and the remaining gap to full DUET is the
additional contribution of the adaptive imitation channel;
2. **Timing**: the adaptive weight μ already reaches its floor by step 30, when the success rate
is still near zero; the entire performance climb happens while the imitation weight sits at its
lowest level. If the gains came mainly from imitation supervision, they should accumulate when
that supervision is strongest; the opposite is what we observe.

The main margins are paired-significant (McNemar p = 2.2e-5 / 7.9e-8).

---

## → Reviewer y9x6, Part 2 of 2 (Weaknesses 4–5 and Questions)

Continuing our reply from Part 1.

**Weakness 4: sensitivity to teacher quality, cache diversity, cache size, success filtering.**

We agree these sensitivities deserved a clearer account, and we address the four dimensions in
turn (ALFWorld 1.5B):

| Dimension | Setting | Result |
|---|---|---|
| **Teacher quality** | strong: 72B (80.6% success) | **47.5%** |
| | weak: 14B (68.1%) | **34.5%** |
| **Cache diversity** | distinct sequences per 8.5 demos/task | 4.4; 7.6% of tasks single-path |
| **Success filtering** | raw samples kept | 80.6% |
| **Cache size** | full / 10% / 1% sweep (+ doubled mixing) | running (Q2/Q4) |

Reading the dimensions in words: **teacher quality is the dimension DUET is most sensitive to, and
the sensitivity is graceful**. A teacher whose sample success drops by 12.5 points costs DUET 13
points, yet the result still exceeds every baseline that uses the strong teacher (CHORD: 27.0%),
and the run is still improving when the budget ends: weaker supervision slows learning rather than
breaking it. **The diversity requirement is mild**: the cache averages only 4.4 distinct solutions
per task, 7.6% of tasks offer a single path, and DUET trains well on it. **The success filter is
not an aggressive selector** (80.6% of raw rollouts pass), and the probe experiment under Weakness
2 indicates that the filtering does not leak outcome information into the discriminator. **Cache
size is the one dimension we cannot yet quantify**; the systematic sweep is running, and we will
post the results during the discussion.

**Weakness 5: comparison to broader off-policy RL or replay methods.**

A useful suggestion, and the revision will add positioning against AWAC/IQL-style advantage
weighting, V-trace truncation and prioritized replay, noting the quantities each requires that our
setting lacks (teacher likelihoods, a shared tokenizer). Two controlled comparison points already
isolate this design choice: uncorrected replay (removing DR3) is 14pp slower at step 50, and
LUFFY's π/(π+β) weighting reaches only 5.5% at 1.5B and is net-negative at 7B.

**Q1: multi-seed robustness and confidence intervals.**

We ran the full replication on the representative setting (ALFWorld 1.5B): **seven seeds**, all on
the same pinned curriculum:

| val@100 | result |
|---|---|
| **DUET (n = 7)** | **44.6 ± 6.6** (range 34.5–53.0) |
| strongest baseline (SFT→GRPO, Table 1) | 30.0 |
| CHORD / LUFFY / GRPO | 27.0 / 5.5 / 1.0 |

Two points. **The worst seed (34.5) still exceeds the strongest baseline.** And, more importantly,
the training dynamics are consistent across seeds: **six of the seven seeds keep improving over
the second half of training (+2.0 to +13.5; the single decline is −3.5), while every baseline
method declines over the same interval** (CHORD −3.0, GRPO −15.5, LUFFY −20.5). The margin
therefore does not depend on the choice of evaluation point, and its direction is highly
consistent.

**Q2: sensitivity to the number of teacher trajectories and the mixing ratio.**

A sweep spanning a 24× range of effective teacher supply (the full cache, a 10% subset, and a 1%
subset, plus a doubled mixing ratio) is executing on the pinned curriculum; the results will be
posted during the discussion period.

**Q3: cache size and diversity.**

Please see the table and discussion under Weakness 4 in Part 1.

**Q4: a cache-size ablation.**

In progress; please see Q2.

**Q5: a teacher-quality ablation.**

The "moderately weaker" case is complete (the 14B result under Weakness 4): performance scales
smoothly with teacher quality and the mechanism does not fail. We do not have a noisy or
deliberately suboptimal teacher variant, and we do not claim that dimension.

**Q6: simpler reward-shaping baselines.**

This control is complete, and it isolates exactly what the reviewer asks about. Within each task,
the progress values are randomly reassigned among that task's states: the bonus magnitude and the
state coverage are unchanged (a generic dense shaping signal of the same strength), while the
teacher-derived ordering is destroyed. The result is 41.0, against 47.5 with the true progress
table and 31.0 with SC removed: generic same-strength shaping plateaus, and it is the
teacher-derived ordering that keeps paying. That is the value of the teacher progress map,
isolated.

**Q7: alternative off-policy correction methods.**

Please see Weakness 5 in Part 1. We are grateful for this detailed review, which motivated several
of our strongest new results; the remaining sweeps will follow as they complete.
