# DUET — NeurIPS 2026 Rebuttal (submission draft)

Paper #32282. Reviewers: **UyKJ 4** (conf 3) · **bDeY 3** (conf 4) · **y9x6 3** (conf 3).
Last updated 2026-07-28 by the A100 side. Numbers marked ▶ are still executing and must be
filled or deleted before submission — **do not submit a ▶ placeholder.**

**Protocol for every number below.** Strict success = task score ≥ 1.0, recomputed from the stored
validation rollouts, n = 200 held-out tasks, one uniform protocol across all cells. This differs
from Table 1 by ≤ 1.0pp on three WebShop cells (DUET 36.0 → 35.5, SFT+GRPO 18.5 → 18.0, LUFFY
5.5 → 4.5); we correct these in the camera-ready and use the recomputed values throughout.

---

## Global response (post before the individual replies)

We thank all three reviewers. Three overlapping concerns were raised — generalisation, the status of
DR3, and evidence quality — and we ran new experiments for each. We also correct several errors of
our own: one that a reviewer identified, and two that they did not.

**Corrections we volunteer.**

1. **Eq. 9 double-counts the density ratio (bDeY).** The reviewer is right about the notation.
   The implementation applies **one** correction, not $\hat w\rho_t$ — for teacher samples it
   *replaces* the behaviour log-probability, so the single clipped ratio evaluates to $\hat w$.
   Eq. 9 is rewritten as that substitution.
2. **Eq. 8 overstates DR3.** The applied weight is the bounded $\alpha$-relative ratio
   $\hat w_\alpha=\hat r/((1-\alpha)\hat r+\alpha)\le 1/(1-\alpha)$, measured ≤ 1.13. DR3 can only
   ever *down-weight* teacher samples, never amplify them. We relabel it a bounded,
   variance-controlled replay weight rather than an exact likelihood ratio.
3. **"Principled" is narrowed.** It now describes the diagnosis of the two biases and the direction
   of the two corrections, not a derivation. The State Channel is labelled a heuristic in the
   contribution statement, and the non-invariance admission moves out of §3.6 into that statement.
4. **Table 1 formatting (bDeY).** CHORD is the strongest non-DUET method in both 3B columns and was
   not underlined; the revision underlines all four columns. No number or $\Delta$ changes.
5. **The 1.5B-WebShop SFT+GRPO baseline was not task-matched** — it trained on 400 distinct tasks
   against DUET's 800. We found this ourselves and reran it task-matched with a *larger* budget than
   DUET; the corrected baseline is weaker, not stronger (7.5% against the published 18.5%).
6. **The 1.5B-WebShop DUET cell reproduces only partially.** Full distribution and diagnosis in the
   reply to y9x6. We report the spread, not the best seed.

---

## Reviewer UyKJ (score 4)

**Is the discriminator separating teacher from student, or just success from failure? (Q3)**
Directly tested; it is not a success detector. Probes trained on DR3's own features (log-probability
statistics and length — reward, advantage and success are *never* inputs) show that restricting the
student side to **successes only** does not weaken separation: 90.0% on ALFWorld (98.2% early),
99.0% on WebShop late. Late in ALFWorld training, successful and failed student rollouts receive
near-identical P(student) — 0.861 vs 0.869 — while teacher trajectories sit at 0.280. WebShop late
is the sharpest available test, because that policy already succeeds on 84% of its rollouts, and
there *successful* rollouts score **more** student-like than failed ones (0.947 vs 0.904; teacher
0.155) — the opposite of what a success detector would do. Caveat stated plainly: this is an
offline probe of DR3's features, not of the online discriminator.

**What causes the Pick-Two exception? (W4, Q2)**
The cell is too small to carry the interpretation we gave it. Pick-Two is n = 45 of 200 validation
tasks; DUET and GRPO disagree on 18 of them (6 vs 12), exact paired McNemar **p = 0.238**, and three
seeds of one *identical* configuration span 8.9pp on that same subset. It also does not replicate
off the cell: at 1.5B, DUET 24.4% vs GRPO 0.0%; at 7B, 77.8% vs 75.6%. One real limitation sits
behind it: $\Phi$ is the *maximum* normalised index at which a state occurs, and Pick-Two
demonstrations are the most self-revisiting type (52.6% contain a repeated state, against 7.5–30.8%
for other types), so the potential saturates inside the first sub-goal. The fix is a
sub-goal-indexed potential. We note that this mechanism is measured at 1.5B while the anomalous
cell is 3B.

**DR3's theoretical justification (W5)**
Strengthened rather than defended — see global correction 2. Bounded and down-weight-only is
precisely what "bias-mitigating replay weight" names, so we keep the term and now derive the bound.

**Performance when the teacher is weaker (W2, Q1)**
Two senses, both measured. *Relatively weaker teacher*: holding the cache fixed and raising the
student, the logged teacher–student reward gap falls from 0.93/0.55 (early/late) at 1.5B to
0.42/0.06 at 7B, and the teacher's share of the policy gradient falls from a run-mean of 0.241 to
0.079 at an identical mixing rate. Outcome at 7B on ALFWorld: **DUET 86.5% vs GRPO 85.0%**, while
LUFFY — same cache, same rate, no fade-out — reaches **82.5%**, *below* the no-teacher baseline.
DUET degrades gracefully to GRPO exactly where methods without fade-out become harmful.
*Corrupted teacher signal*: with 30% observation noise on the matcher's copy of the observation,
exact hashing retains 6.6% of the clean progress signal and strict success falls to **11.0%** —
below the 31.0% of deleting the State Channel outright. Details in the reply to y9x6.
*A genuinely weaker teacher*, now run. We collected a fresh cache from **Qwen2.5-14B-Instruct**
(4,094 demonstrations covering 727 of the 800 training tasks, 68.1% success rate against 72B's
80.6%) and retrained DUET on it, changing nothing else:

| ALFWorld 1.5B, same 800 tasks | val@50 | val@100 |
|---|---|---|
| DUET + Qwen2.5-72B cache (paper) | 42.5% | **47.5%** |
| **DUET + Qwen2.5-14B cache** | 33.5% | **34.5%** |
| on-policy GRPO (no teacher) | 16.5% | 1.0% |

DUET degrades with teacher quality — 13pp — and we report that plainly rather than claiming
insensitivity. But the degradation is graceful and the method does not break: with the weaker
teacher it still exceeds CHORD *using the strong teacher* (27.0%) and SFT+GRPO (30.0%), it still
beats the no-teacher baseline by 33.5pp, and it is still **improving** at the end of the budget
(33.5 → 34.5) where every uncorrected method declines. The ordering of interest is preserved: what
a weaker teacher costs is performance, not stability.

**Different teacher / non-Qwen student (W1, Q1)**
Run, on **Llama-3.2-3B-Instruct** with the *unchanged* Qwen2.5-72B cache. We report the partial
result and its caveats rather than waiting for a flattering number.

| ALFWorld, Llama-3.2-3B student, val@50 | strict |
|---|---|
| LUFFY, run 1 | **19.5%** (stable, KL 0.39) |
| LUFFY, run 2 — *nominally identical to run 1* | **4.5%** (destabilised, KL 3.28) |
| **DUET** | **8.5%** (destabilised, KL 4.02) |
| on-policy GRPO | 5.5% |

**We draw no conclusion from this setting, because it cannot support one.** The two LUFFY rows are
the same configuration, the same `data.seed`, and — verified from the saved rollouts — the same 800
training task IDs, with code that is functionally identical on this path (the intervening commits
add a feature gated off by default plus diagnostics; three lines of the shared loss changed, all
dictionary keys). They differ by **15pp**, and qualitatively: one trained stably, the other
diverged. DUET's single run falls between them. Any ordering we reported here would be an artifact
of run-to-run variance, so we report the variance instead.

Teacher data transfers across model families — both teacher-using methods beat the no-teacher
baseline, and inspection shows why: the Qwen teacher's demonstrations correct the Llama student's
output-format prior.

**Both methods then diverged numerically, in the same window and by the same mechanism**, and we
stopped each under a pre-registered criterion (KL > 5):

| Llama-3B, identical 800 tasks | first KL > 5 | peak KL | peak teacher gradient share | final on-policy success |
|---|---|---|---|---|
| DUET | step 54 | 5.16 | 0.91 | 0.071 |
| LUFFY | **step 51** | **7.58** | **1.00** | 0.018 |

The baseline diverged *earlier and harder* than our method. The shared mechanism is visible in both:
as the Llama student's on-policy success decays, the teacher share of the policy gradient climbs —
to 0.91 and 1.00 respectively, against a median of 0.21 (max 0.62) over the Qwen run, where it is designed to
*fall*. With a weak on-policy signal in the denominator, whichever teacher-side term is present
comes to dominate. For DUET this happened first through the State Channel and then, once $\beta$
was scaled down, through behaviour cloning; LUFFY has neither channel and reached a teacher share
of 1.00 anyway. **This is a property of replaying teacher data into a student too weak to hold up
its own half of the batch, not of our corrections.**

**The honest conclusion is that Llama-3.2-3B at this budget is not a setting in which we can
measure a *ranking* reliably** — not that DUET beats or loses to the baseline there. What the
setting does show cleanly, because it happens to both methods, is the failure mode itself. What we can state
is a real limitation the attempt exposed: neither auxiliary channel is scaled to the strength of the
student's own learning signal, so with a student this much weaker the teacher-side terms take over
as the on-policy signal decays. DUET as published did not transfer without retuning the State
Channel, and even retuned it did not remain stable. We deliberately did not tune a third time — past
that point we would be selecting a configuration on its result rather than diagnosing a fault.

Porting also required one adaptation we disclose in full: with the ALFWorld $\beta$ carried over
unchanged, the run collapsed outright. Llama-3B's early task reward is ≈ 0, so the State Channel
bonus became the dominant objective (`state_channel/bonus_vs_reward_ratio` 0.20–0.67 against a
documented healthy band of < 0.15) and the policy converged on a **no-op that freezes the
observation** — `{"name": "inventory"}` for 68% of all assistant turns, 0 of 1736 turns containing
a valid action, on-policy success pinned at exactly 0.000. Scaling $\beta$ down 4× removed it.

**We ran the control that could have refuted this diagnosis, and it survived.** The collapsed run
also carried a second deviation — a micro-batch setting that ran DUET's discriminator and $\mu$
clock at twice the intended rate — and the LUFFY/GRPO comparisons are structurally immune to that
defect, since neither has those components. So we reran with the clock restored to the paper's
recipe and $\beta$ back at its original 0.2, changing nothing else. It collapses identically:
bonus ratio 0.17–0.67, success 0.000 for twelve consecutive steps, 36% of turns the same no-op,
6 of 1736 turns containing a valid action. **The State Channel dominance is the cause; the clock
was not.**

**DUET as published did not transfer to this student family without retuning the State Channel**,
which is a limitation of the module, and we would rather you hear it from us.

On breadth your limitation otherwise stands: two environments, one *primary* student family. We
will not claim a third environment we did not train.

**Many hyperparameters (W3)**
The SC coefficients ($\beta$ = 0.2, $\eta$ = 0.05, `exclude_teacher`) and the DR3 discriminator
settings (temperature 1.5, 2 steps per call) are **identical** across both main-table DUET
configurations. The BC $\mu$ schedule is not: ALFWorld uses (peak 0.3, valley 0.05, $d_{floor}$ 0.4,
EMA 0.5) and WebShop (0.3, 0.10, **0.6**, 0.2). We state this explicitly in the revision rather than
implying a single setting. The sharp sensitivity is confined to $d_{floor}$, the BC cold-start
coefficient — a safety net rather than a channel — and because the two environments' best values
straddle the range we do **not** recommend a single default; we give the diagnostic instead: a flat
`critic/success_onpolicy/mean` over the first 20 steps means $d_{floor}$ is too low.

One further tuning rule, which we learned the hard way while preparing this response and which we
will document: **the State Channel coefficient must be scaled to the student's reward magnitude,
not carried over.** Porting DUET to Llama-3.2-3B with the ALFWorld $\beta$ unchanged collapsed the
run — that student's early task reward is ≈ 0, so the SC bonus became the only non-zero learning
signal (`state_channel/bonus_vs_reward_ratio` 0.198–0.534 against 0.026–0.114 on the Qwen run), the
policy optimised for remaining in matcher-recognised states, and it converged on a no-op action
(`inventory`, 72% of all turns) with exactly 0.000 success. The diagnostic is that ratio and the
band is < 0.15; scaling $\beta$ down 4× restores it. We report this because it is a real usability
cost of the module and because it is the same failure family as the noise experiment below: when SC
misfires it does not abstain, it actively misleads.

---

## Reviewer bDeY (score 3, confidence 4)

**Eq. 9 double-counts $\hat w$ (W2)**
You are right, and the error is ours. The code never forms $\hat w\rho_t$. For teacher samples it
*replaces* the behaviour log-probability (`het_actor.py:1507`):
$\log\hat\pi_\beta := \mathrm{sg}[\log\pi_\theta] - \log\hat w(\tau)$, so the single clipped ratio is
$\rho^\beta_t = \pi_\theta/\hat\pi_\beta = \hat w(\tau)$ exactly — one correction, not two. $\rho_t$
(Eq. 7) applies to on-policy samples only. We initially prepared a defence arguing that the product
telescopes; three independent code audits refuted it, and we discarded it. Two further errors we
found while checking, both against us: Eq. 8's $D/(1-D)$ is only an intermediate (global correction
2), and the WebShop configurations run a policy-shaping variant (`dr3.use_policy_shaping: true`)
that Eq. 9 does not print. **No experimental result changes**; the edits are Eqs. 8–9, an appendix
variant, and the contribution statement.

**Is "principled" supported, or is a hand-designed shaping signal doing the work? (W1)**
We narrow the claim (global correction 3) and we answer the empirical half with an experiment.
If the gains came from stronger supervision, then supplying *more* supervision should help. It does
not. ALFWorld 1.5B, same 800 tasks, same 100 steps:

| | val@50 | val@100 |
|---|---|---|
| **DUET** (adaptive $\mu$, 0.3 → 0.05) | 42.5% | **47.5%** |
| BC-only, $\mu \equiv 1.0$ held at full weight | **48.0%** | **31.0%** |
| CHORD ($\mu$ decays on a fixed schedule) | 30.0% | 27.0% |
| on-policy GRPO | 16.5% | 1.0% |

On ALFWorld, full-weight imitation peaks at step 50 and then **loses 17 points**, while DUET's
adaptive schedule keeps improving over the same interval.

**But we must report the other environment, where this baseline is strong and our own claim is
weakest.** On 1.5B-WebShop at the same matched budget and the same pinned curriculum, BC-only does
*not* degrade — it goes 3.5% → **30.0%** (reward 0.690), against DUET's paper cell at 35.5% and
DUET's three pinned replicates at 16.5 / 14.0 / 6.5 (mean 12.3%). A single BC-only run therefore
lands above our replicate mean on that cell.

Our first instinct was to note that BC-only is n=1 against a DUET distribution spanning 6.5–35.5,
so the ranking could be a draw from a wide distribution. **Checking that defence, we found it does
not hold, and we report the check.** Under the escape criterion above, BC-only escapes the
partial-credit local optimum at **step 57 — 23 steps earlier than the earliest DUET run** (80, 93,
97, 100) and earlier than CHORD (76). That is a systematic difference in kind, not a lucky draw,
and there is a plausible mechanism: strict success on WebShop requires clicking every requested
option, full-weight behaviour cloning teaches that behaviour directly, and DUET's adaptive $\mu$
has already fallen to its floor by step 30.

▶ BC-only replicates at additional seeds are running, on the identical pinned curriculum. If the
early escape replicates, the honest conclusion is that on this cell sustained imitation reaches the
required behaviour faster than our adaptive schedule does, and we will say so.

What we can say without replicates is narrower than our submitted claim: the gain is not simply
"more imitation" on ALFWorld, where holding $\mu$ at full weight is actively worse at matched
budget; on WebShop we cannot currently separate them.

We also report where the ablation cuts against us. With SC removed, ALFWorld gives 31.0% against
CHORD's 27.0% and SFT+GRPO's 30.0% — parity, not a win, on a single seed — and on WebShop −SC gives
1.0%, *below* CHORD's 11.5%.

More directly on your question, we ran the control that separates "teacher-derived progress
information" from "a hand-crafted dense bonus": the progress values are permuted among each task's
own states, holding coverage and bonus magnitude fixed while destroying the correlation between
position and $\Phi$ (+0.772 → +0.045). Result: 41.0% against the true map's 47.5% and 31.0% with SC
deleted. **About 10.0 of SC's 16.5pp is obtainable from any dense bonus of that magnitude; about
6.5pp needs the teacher ordering.** Your reading is therefore partly correct, and we will say so in
the paper rather than attributing the whole module to the progress map. The residual is the durable
part — the shuffled control matches the true map at step 50 and then flattens, while the true map
continues to improve.

The defensible claim is narrower than the one we made: an optional heuristic SC module, roughly half
of whose benefit is generic reward densification, on top of a CHORD-compatible core.

**Breadth, and whether CHORD is more generally applicable (W3)**
Failure case first. With 30% observation noise and **exact** matching, SC state coverage falls
0.590 → 0.178 and strict success to **11.0%** — below the 31.0% of removing SC entirely, because a
matcher that misses scores on-path states as zero instead of abstaining. Under the same noise with
a **soft** TF-IDF matcher over the identical progress map: coverage 0.654, success **54.5%**. Same
800 training / 200 validation tasks; one configuration field changed. Missing control, stated
plainly: we ran no clean+soft cell, so this establishes that exact matching is brittle and does
**not** establish that soft matching is better in general (54.5 vs 47.5 clean is ≈1.4 sd of our
observed spread). Your objection otherwise stands: WebShop's $\Phi$ is ~210 lines of hand-written
per-environment code, which is the real cost of SC where SC is load-bearing, and we have no
math/reasoning result and will not claim one. On CHORD's generality: DUET's core (baseline
separation + DR3 + adaptive BC) consumes only teacher tokens, masks and a scalar reward — the same
inputs CHORD needs — so the core applies wherever CHORD does; it is SC that carries the extra
requirement.

**Q1 — group composition; is $m$ fixed or resample-until-success?**
$n = 7$ on-policy $+\ m = 1$ teacher $= 8$ per prompt. The cache is loaded once and frozen;
selection is sampling without replacement from a fixed per-task list, and touches no environment or
LLM. **No resample-until-$m$-successes loop exists in training** — that loop lives only in the
offline collection script. A cache miss back-fills with an on-policy rollout, so the group is always
exactly 8. Realised rates: **0.978** teacher trajectories per prompt (ALFWorld), **0.858**
(WebShop).

**Q2 — does SFT+GRPO use the same teacher data, and can you show the SFT curve?**
Same cache file, same $n_\text{teacher} = 1$, same realised per-prompt rate (0.975 against DUET's
0.978). **But it is not task-matched**: SFT+GRPO trains on 400 distinct tasks against DUET's 800 and
consumes about half the teacher volume (390 vs 782 trajectories). We disclose this as a real
confound in the headline comparison. The SFT stage does execute correctly — ALFWorld on-policy
success rises 0.016 → 0.285 before RL and `chord/sft_loss` falls 0.844 → 0.153 — but you are right
that training-task success cannot settle the question under a 400-vs-800 asymmetry.
**The task-matched rerun is now complete**, and it does not overturn the comparison — it
strengthens it. We rebuilt SFT+GRPO on the same 800 tasks and deliberately gave it *more*
optimisation than DUET: 50 SFT + 100 GRPO = 150 steps against DUET's 100.

| 1.5B-WebShop, val@100 | strict | mean reward |
|---|---|---|
| DUET, paper cell | **35.5%** | 0.706 |
| DUET, pinned replicate | 14.0% | 0.592 |
| CHORD | 11.5% | 0.603 |
| **SFT+GRPO, task-matched (800 tasks, 150 steps)** | **7.5%** | **0.403** |
| DUET, pinned replicate | 6.5% | 0.551 |

The task-matched baseline lands *below* the 18.5% we reported in Table 1, so our published number
was if anything generous to it. It is below DUET's replicate mean (18.7%) and below CHORD, and its
**mean reward of 0.403 is the lowest cell in the table** — lower than every DUET replicate.

Two caveats we volunteer, because both cut against a clean reading. First, this baseline entered a
partial format collapse over its second half (malformed actions 0% through step 57, peaking at 27%,
14.7% at the end), which suppresses strict success. But format collapse has a signature — strict
success falls *while mean reward rises* — and that is not what happened here: its reward fell too,
0.640 → 0.393 in training. The degradation is real, not only formatting. Second, at the step-50
checkpoint, where both are format-clean, the ordering reverses: baseline 8.0% against DUET's
1.0–1.5%. That point flatters the baseline for a structural reason — DUET's escape from the
partial-credit local optimum has not happened yet at step 50 (it occurs at steps 80–100), and the
baseline has by then consumed 100 optimisation steps against DUET's 50. We report both checkpoints
rather than choosing the favourable one.

On the SFT curve: the stage executes correctly — ALFWorld on-policy success rises 0.016 → 0.285
before RL and `chord/sft_loss` falls 0.844 → 0.153.

**Q3 — teacher cache details**
ALFWorld: 19,497 trajectories over 2,348 distinct tasks (8.30 per covered task), filtered from
24,200 raw rollouts (80.6% retained). WebShop: 26,178 over 5,691 tasks (4.60 per task). Teacher is
Qwen2.5-72B-Instruct and only successes are kept. Diversity: of 8.5 demonstrations per task, 4.4
have distinct action sequences; 36.5% of same-task pairs are identical; 7.6% of tasks have a single
path. Disclosure we add: the WebShop entries are verified gold action sequences replayed with
72B-authored rationales, not sampled 72B rollouts.

---

## Reviewer y9x6 (score 3)

**Generalisation to noisier, open-ended, partially observable environments (W1)**
Your concern is right, in a sharper form than posed: a *failed* matcher is actively harmful, not
merely uninformative. Noise is applied only to the matcher's copy of the observation — the policy
input and the teacher cache are untouched — which isolates the matching failure mode at constant
task difficulty. ALFWorld 1.5B, same 800 training / 200 validation tasks:

| val@100 strict | exact matching | soft (TF-IDF) matching |
|---|---|---|
| clean observations | 47.5% *(the paper's setting)* | **51.5%** |
| 30% observation noise | **11.0%** | **54.5%** |
| *State Channel removed entirely* | *31.0%* | |

(State coverage: 0.590 clean+exact, 0.178 noise+exact, 0.654 noise+soft.)

Exact matching under noise scores *below* deleting the module, because a missed state is scored
$\Phi = 0$ rather than abstaining — the shaping misleads rather than abstains. Replacing the lookup
operator (one configuration field, the same progress map, no new dependency, no learned model)
removes the failure entirely: soft matching is essentially unaffected by the noise (51.5 → 54.5).

We ran the clean+soft cell specifically because without it the experiment could not separate "soft
matching repairs noise" from "soft matching is simply the better operator". The answer is that it is
at least as good on clean observations too (51.5 vs 47.5) — though that 4.0pp gap is inside our ≈5pp
seed spread, so we claim only parity there. **The unambiguous result is the interaction**: a 43.5pp
gap between operators under noise against ≈0 without it. So the State Channel's dependence on
matching is a property of a replaceable component, not of the method.

**Is DR3 a principled estimator or a learned heuristic? (W2)**
Heuristic, and we correct the paper accordingly (global correction 2). Your specific confound —
that success-filtering makes the discriminator a trajectory-quality detector — is testable, so we
tested it; the probe results are in the reply to UyKJ (Q3) and they do not support it: restricting
the student side to successes only *strengthens* separation, and late in training successful and
failed student rollouts are scored almost identically.

**Are the gains from the corrections or from the extra imitation signal? (W3)**
Separable, and we ran it in both directions. *Removing* imitation entirely: 34.0% (ALFWorld) and
16.5% (WebShop) with no BC term at all, against on-policy GRPO's 1.0% / 0.5% — roughly 71% and 45%
of the gain survives with no imitation. *Increasing* it: holding $\mu$ at full weight for the whole
run gives 48.0% at step 50 but **31.0%** at step 100, against DUET's 47.5% (table in the reply to
bDeY). We flag the limit of that argument: on WebShop the same BC-only configuration reaches 30.0%
and does *not* degrade, above our own replicate mean of 12.3% on that cell. Replicates are running;
see the reply to bDeY for why we do not read a single run against a 6.5–35.5 distribution. A third, temporal line of evidence: $\mu$ reaches its floor at **step 30**, when on-policy
success is still 0.000; the entire climb to the final score happens while BC carries one-sixth of
its peak weight, i.e. 73% of training. If the advantage came from supervision, it should accumulate
when supervision is strongest — it does the opposite. Headline margins are paired-significant:
McNemar p = 2.2e-5 (ALFWorld), 7.9e-8 (WebShop).

**Multi-seed robustness (Q1)**
Partly, and we report the failures in full.

*ALFWorld 1.5B*, replicates at the submitted configuration (4 GPUs, micro-batch 2, curriculum
pinned, training task IDs verified from each run's own saved rollouts):

| seed | val@50 | val@100 | second-half change |
|---|---|---|---|
| 2026 (the submitted run) | 42.5% | **47.5%** | +5.0 |
| 2025 | 34.0% | 39.5% | +5.5 |
| 2026 replicate | 38.0% | 34.5% | **−3.5** |
| 2027 | 36.0% | 41.0% | +5.0 |
| 2028 | 46.0% | **50.5%** | +4.5 |
| **mean ± sd (n=5)** | 39.3 ± 4.7 | **42.6 ± 6.4** | — |

**4 of 5 seeds improve over the second half** (the fifth declines −3.5, showing the mild end of the
length-drift signature seen on WebShop) — against CHORD −3.0, GRPO −15.5, LUFFY −20.5. The same
protocol applied to the **SFT→GRPO baseline** gives 41.5/44.0/47.0 vs its published 30.0
(**40.6 ± 7.4**, n=4, seed-matched end to end) and its SFT-stage checkpoint alone reaches
43.5–48.5 (mean 46.8, n=3): **the published 17.5pp ALFWorld margin does not survive baseline
replication and we withdraw it** — surviving margins on the means: CHORD +15.6, LUFFY +37.1,
GRPO +41.6. All H200 numbers re-verified on the A100 from the raw validation logs archived in
`data/validation_logs_h200/`.

*1.5B-WebShop does not fully reproduce.* Against the paper's 35.5%, four replicates of the identical
configuration — verified field-by-field against the executed configuration backups, all four DUET
mechanisms healthy throughout — give:

| replicate | 800-task curriculum | escape step | val@100 strict | reward |
|---|---|---|---|---|
| paper cell | the paper's draw | **80** | **35.5%** | 0.706 |
| pinned C | **same draw** | 93 | **16.5%** | 0.594 |
| pinned B | **same draw** | 97 | **14.0%** | 0.592 |
| pinned A | **same draw** | 100 | **6.5%** | 0.551 |
| re-drawn A | re-drawn by the run seed | 80 | 3.5% | 0.521 |
| re-drawn B | re-drawn by the run seed | never | 2.5% | 0.543 |

Three pinned replicates: mean **12.3%**, sd 5.2pp, range 6.5–16.5%, against the paper's 35.5%.

Two things this shows. First, `data.seed` in our code sets both run randomness *and* which 800 tasks
are selected, so the bottom pair confounds the seed with a two-thirds change of training curriculum;
pinning the curriculum roughly quadruples the score. We have added a `data.task_seed` field to
separate the two, and every rebuttal experiment now pins it. Second, even with the curriculum pinned
the replicates reach 14.0–16.5%, not 35.5%. **The submitted number is the top of a distribution, and
we will report the distribution.**

The mechanism is specific to this cell and worth stating, because it also explains why the metric is
unstable here. Strict success on WebShop requires *every* requested attribute to be matched
exactly; 88.4% of training tasks request at least two; and every run first converges on a
three-action policy that earns partial credit but is structurally incapable of an exact match
(0 successes across 10,735 under-clicking episodes). Escape from that local optimum is a late,
stochastic event. Making that precise — first step at which the 10-step moving average of on-policy
success exceeds 0.04, twice the pre-escape plateau, and stays above it — the paper cell escapes at
**step 80**, the pinned replicates at **93** and **97**, and a third pinned replicate had not
escapes at step 100, i.e. exactly at the budget boundary. Across the four runs that share the
paper's task draw, escape step and final score are **monotonically related — Spearman
$\rho = -1.00$, Pearson $r = -0.99$** (80 → 35.5%, 93 → 16.5%, 97 → 14.0%, 100 → 6.5%).

This reframes the instability. The 100-step budget does not measure an asymptote; it measures **how
far a late phase transition has progressed when the budget happens to stop**. The spread across
replicates is therefore not evidence that DUET is unstable — it is a property of evaluating this
environment at a fixed step count while the transition is still in progress. It also predicts,
correctly, that the two replicates which re-drew their curriculum sit off this curve: a different
task set is a different problem. We do not claim the submitted 35.5% is representative; it is the
earliest-escaping run of four, and we report all four. Worse, the metric has a knife edge at 1.0: extending one replicate to 150 steps
produced **46 of 200 episodes scoring exactly 0.950** — mean reward *rose* to 0.610 while strict
success fell to **0.0%**, because a single malformed action costs exactly the 0.05 separating 0.95
from 1.0. Across 68 historical 1.5B-WebShop runs, those above 30% malformed actions average 0.6%
strict success against 5.9% below 10%. This affects any method that enters the mode, and it is why
we now report mean reward alongside strict success on this cell.

**Cache size, quality, diversity, success filtering (W4, Q2–Q5)**
Statistics are in the reply to bDeY (Q3). ▶ Three runs sweeping effective teacher supply across a
24× range at a fixed curriculum are executing: full cache, 10% (task coverage 97.8% → 55.9%), 1%
(→ 8.1%), plus a doubled mixing ratio ($m$ = 2). The weaker-teacher ablation (Qwen2.5-14B, 68.1%
cache success rate against 72B's 80.6%) is complete: 34.5% against 47.5%, still well above the
no-teacher baseline's 1.0% — table and discussion in the reply to UyKJ. We have no *noisy* or
*deliberately suboptimal* teacher and will not claim that dimension.

**Simpler reward-shaping baselines (Q6)**
Run, and it decomposes the State Channel's contribution in a way we could not have claimed before.
The control permutes the progress values **among each task's own states**, so state coverage and
bonus magnitude are held fixed (realised coverage 0.684 against the clean run's 0.590; bonus mean
0.507 against 0.523) while the correlation between position along the demonstration and $\Phi$
collapses from +0.772 to +0.045. It is therefore a dense bonus of the same size and support,
carrying no teacher-derived ordering.

| ALFWorld 1.5B, 800 tasks / 100 steps | val@50 | val@100 |
|---|---|---|
| DUET, true progress map | 42.5% | **47.5%** |
| **shuffled progress map** (matched magnitude, ordering destroyed) | 41.5% | **41.0%** |
| State Channel removed entirely | 35.5% | 31.0% |
| CHORD | 30.0% | 27.0% |

The honest reading, which we will put in the paper: of the State Channel's 16.5pp benefit over
removing it, **about 10.0pp is available from any dense bonus of this magnitude and support, and
about 6.5pp requires the teacher-derived ordering.** So your suspicion is partly right — a simpler
shaping baseline recovers the majority of SC's effect — and we were wrong to attribute the whole of
it to the progress map.

The part that does depend on the teacher signal is the part that persists: the shuffled control is
statistically indistinguishable from the true map at step 50 (41.5 vs 42.5) and then **flattens**
(41.5 → 41.0) while the true map keeps improving (42.5 → 47.5). Dense-bonus densification buys the
early gain; the ordering information is what continues to pay after the policy has absorbed it.
Single seed, and 6.5pp is only ≈1.3× our measured ALFWorld seed sd of ≈5pp, so we state this as a
qualitative decomposition — the *shape* difference (shuffled flattens, true map keeps improving) is
what we rely on, not the size of the split.

**Broader off-policy / replay comparisons (W5, Q7)**
Not run, and we will say so. We compare against on-policy GRPO, teacher mixing with policy shaping
(LUFFY), weighted SFT inside RL (CHORD), and SFT→GRPO, but not the classical off-policy family. The
controlled comparisons we can offer: uncorrected replay (−DR3) gives 47.5% / 9.5%, and LUFFY's
$\pi/(\pi+\beta)$ weighting gives 5.5% / 5.5% at 1.5B and is net-negative at 7B. The revision will
position DR3 against AWAC/IQL-style advantage weighting, V-trace truncation and prioritised replay,
and state which quantities each needs that we do not have (teacher likelihoods, a shared tokenizer).

**One result we should have reported (all reviewers).** At the two evaluation points our protocol
already produces: DUET 42.5 → **47.5** (+5.0), CHORD 30.0 → 27.0 (−3.0), LUFFY 26.0 → 5.5 (−20.5),
GRPO 16.5 → 1.0 (−15.5). Every method that replays or imitates teacher data without our corrections
**degrades over the second half of training**; DUET is the only one still improving when the budget
ends, and it leads the strongest baseline by 12.5pp at step 50 and 17.5pp at step 100 — so the
headline does not depend on where we evaluate. Relatedly, −DR3 is 14.0pp behind full DUET at step 50
(28.5% vs 42.5%) and only catches up by step 100, so the endpoint snapshot understates DR3.

---

# 内部评估（不提交 — 提交前删除本节）

**提交用的压缩版是 `rebuttal_submit.md`**（每位审稿人 <6000 字符，已逐条审计）。本文件是底稿。

## 涨分概率（2026-07-29 晚间更新）

| 审稿人 | 现分 | 目标 | 概率 | 依据 |
|---|---|---|---|---|
| **UyKJ** | 4 | 5 | **30%** | 两个条件一正一负。14B 弱 teacher 完整落地且为正（34.5% vs 47.5%，退化平缓、仍远超无 teacher 的 1.0%）。非 Qwen 学生失败，但**失败被查清了**：同配置 LUFFY 两次 19.5%/4.5%，该设定测不出排序；两方法都数值发散（KL 5.16 / 7.58）；SC 主导的诊断经受住了时钟控制实验。判别器探针、Pick-Two、DR3 有界性答得干净。 |
| **bDeY** | 3 | 4 | **40%** | 他的两个条件：任务对齐 SFT 重跑**已完成且有利**（给基线 150 步对 DUET 的 100 步，它只有 7.5%/reward 0.403，低于主表报的 18.5%）；WebShop 复现仍未做到，但已转化为逃逸步与分数的 ρ = −1.00。Eq.9 的坦承 + 代码定位仍是最重的一击。 |
| **y9x6** | 3 | 4 | **30%** ↑ | SC 泛化 2×2 已补齐且有利（噪声下算子间差 43.5pp，干净观测下软匹配 51.5 vs 47.5 持平）。**W3 的威胁已用复现化解到"打平"**：BC-only 两次 30.0/6.5（均值 18.2），DUET 四次 35.5/16.5/14.0/6.5（均值 18.1）——两个分布在该 cell 上无法区分，那个 30.0% 大半是抽样运气。BC-only 确实系统性更早逃逸（57/54 vs 80+），但早逃逸不保证结果（54→6.5）。从 20% 回调。 |

**2026-07-29 深夜更新（H200 最终包并入后）**：
- **ALFWorld 17.5pp 头条差距被基线复现推翻**（SFT→GRPO 复现 40.6±7.4 vs 论文 30.0；DUET 自身
  42.6±6.4——两分布无法区分），提交稿已主动撤回该差距、保留均值上仍成立的（CHORD +15.6、
  LUFFY +37.1、GRPO +41.6）与规模叙事（3B/7B/WebShop）。**由我们先公布基线的真实分布，
  是对 y9x6"也复现基线"这一要求最强的回应——没人预期作者对自己做这件事。**
- "每个 seed 后半程都上升"已按 n=5 修正为 **4/5**；ALFWorld seed sd 定格 **6.4pp**；
  shuffled-SC 的量化切分相应降级为"指示性"，成对复现今夜落地。
- H200 全部 14 个数字已在 A100 从其原始日志独立复算一致，日志归档于
  `data/validation_logs_h200/`。

**总体**：至少一位涨分约 **60%**。诚实让步换取的可信度集中利好 y9x6（他问的正是 multi-seed，
我们给了方法+基线双分布）与 bDeY（让步精确、有出处）；UyKJ 不变。最现实路径仍是
**bDeY / y9x6 之一 3→4**。

## 今日关键变化

1. **噪声/匹配 2×2 补齐**：干净+软 51.5%，噪声+软 54.5%，噪声+精确 11.0%，无 SC 31.0%。
   交互效应 43.5pp 是无歧义的结果；"软匹配普遍更好"只能声称持平（4.0pp 在 ≈5pp seed 跨度内）。
2. **SC 主导诊断经受住控制实验**：时钟修正到论文配方 + β 恢复 0.2，仍以同一签名塌陷
   （bonus 占比 0.17–0.67、连续 26 步零成功、35.8% 轮次是 `inventory` 空转）。**是 SC，不是时钟。**
3. **BC-only 威胁已化解到"打平"**：复现后 BC-only 均值 18.2%（n=2）对 DUET 18.1%（n=4），
   两分布重合。关键是它**没有**赢我们——而它已包含我们的基线分离修正（单独去掉该修正是 0.0%）。
   ALFWorld 上 DUET 仍明确胜出（47.5% vs BC-only 等预算的 31.0%）。
4. **两处过度声称已修正**：teacher 梯度份额"Qwen 全程 0.15–0.23"实为中位 0.21/最大 0.62；
   μ 触底步 30 非 27。

## 待办

1. ▶ BC-only WebShop 复现 ×2（队列中）—— 决定 y9x6-W3 的最终表述。
2. ▶ H200 的 cache 供给曲线（全量/10%/1% + m=2）—— 提交稿唯一剩下的空缺。
3. **3B 列本机无法验证**（`paper_corrections.md` C0）——需要你从远程机器取回验证日志。
   附录 F 的任务类型图来自另一批 run（DUET 整体 69.5% vs 主表 77.5%），而 Pick-Two 的回复引用了那张图。
4. H200 的两个 n=1 风险待其复现：他们的 SFT+GRPO 得 41.5%（论文核实值 30.0%）、
   SFT 阶段检查点 48.5%（高于完整 DUET 的 47.5%）。**A100 侧独立看到同向现象**
   （`alfworld_qwen1.5b_sft` 47.5%、`_sft_a100` 50.5%），需要合并处理。
