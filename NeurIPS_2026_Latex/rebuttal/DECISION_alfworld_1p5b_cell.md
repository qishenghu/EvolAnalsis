# Decision memo for the 2026-07-29 23:00 consolidation — what to claim for 1.5B-ALFWorld

H200 side. Everything here is from replicates at the submission's **exact** configuration
(4 GPUs, `ppo_micro_batch_size_per_gpu: 2`, `task_seed: 2026`), with each run's training task
ids verified against the paper's 800-task draw from its own saved rollouts.

## The data

**DUET, val@100, n = 5** (the submitted run plus four seed replicates):

| | 34.5 | 39.5 | 41.0 | 47.5 | 50.5 |
|---|---|---|---|---|---|
| | seed 2026 repl. | 2025 | 2027 | **2026, submitted** | 2028 |

mean **42.6**, sd **6.4**, range 34.5–50.5. The submitted number is above the mean but not the
maximum.

**Baselines** (single runs from the paper, except SFT+GRPO which we have begun replicating):

| baseline | n | value(s) | vs DUET mean | **vs DUET's worst draw** |
|---|---|---|---|---|
| SFT→GRPO | 2 | 30.0 (paper) / **41.5** (ours) | +6.9 | **−1.2** ⚠ |
| CHORD | 1 | 27.0 | +15.6 | **+7.5** |
| LUFFY | 1 | 5.5 | +37.1 | **+29.0** |
| on-policy GRPO | 1 | 1.0 | +41.6 | **+33.5** |

## What this settles, and what it does not

**Settled — three of the four comparisons are safe at this noise level.** Even DUET's *worst*
of five draws beats CHORD by 7.5pp, LUFFY by 29.0pp and on-policy GRPO by 33.5pp. No plausible
seed variance in the baselines closes those gaps.

**Not settled — SFT→GRPO.** Our replicate scores 41.5% against the paper's verified 30.0%. That
11.5pp difference is ≈1.8 sd of what we now know this environment does, so it is unremarkable as
variance — but it means **the 17.5pp headline margin is a single-draw-vs-single-draw comparison
between two distributions that overlap.** Four further baseline runs (two seed-matched
SFT→GRPO pairs) land ≈19:30 today.

**Also not settled — whether the baseline's own second stage hurts it.** Our SFT-stage
checkpoint scores 48.5% at its step 50, above full DUET's 47.5% at step 100, and its own 50
GRPO steps take it down to 41.5%. If that reproduces, a reviewer can fairly say the baseline is
mis-specified in our favour. Same four runs will tell us.

## Recommended claim, by outcome

**If the SFT→GRPO replicates come in near 30% (mean ≲ 33):**
Report DUET 42.6 ± 6.4 (n=5) against SFT+GRPO's distribution, note the margin is ≈10pp rather
than 17.5pp on a mean-to-mean basis, and keep the headline. The reduction is honest and small
enough to present as "the multi-seed number the reviewer asked for".

**If they come in near 40% (mean ≳ 38) — plan for this:**
Do **not** defend 17.5pp. Restate the ALFWorld-1.5B claim as:

1. *Three of four baselines are beaten by DUET's worst draw* (+7.5 / +29.0 / +33.5pp). Lead
   with this — it is robust to any seed variance the baselines might have.
2. *Against the strongest baseline the two distributions overlap at a 100-step budget*, and we
   report both rather than a point estimate. This is exactly what y9x6 asked for, and
   volunteering it is worth more than the pp we lose.
3. *The dynamics separate the methods where the endpoints do not*: 4 of 5 DUET runs gain
   4.5–5.5pp over the second half of the budget while every baseline loses (CHORD −3.0, GRPO
   −15.5, LUFFY −20.5). DUET's *worst* second-half behaviour (−3.5pp) is about CHORD's
   *typical* one.
4. Keep the 3B and 7B cells as the scale evidence; they are unaffected by this.

**If the SFT stage alone (48.5%) also reproduces above DUET:**
State it ourselves, in the response, as a limitation of the baseline protocol we adopted — 50
SFT + 50 GRPO was chosen for compute matching, and on ALFWorld the GRPO half is not helping the
baseline. Offer the SFT-stage checkpoint as an additional, stronger baseline in the
camera-ready. Being the ones to find it is much better than being told.

## What I would not do

- **Do not quote only val@50**, where DUET's spread is tighter (39.3 ± 4.7). The protocol is
  val@100 for every method in Table 1; switching evaluation points for one cell is
  cherry-picking and a reviewer with the logs would see it.
- **Do not drop the low replicate.** 34.5% is a legitimate draw at the paper's own
  configuration, and its mechanism is understood (response length drifts 3.7k → 6.3k after
  step 70 while success falls 0.52 → 0.18).
- **Do not claim "the only method still improving".** It is 4 of 5, not 5 of 5. Say "usually
  still improving, where every baseline reliably degrades".

## One thing worth deciding jointly

The A100 side's WebShop-1.5B replicates run 15.0–24.5% against a submitted 36.0%. Ours here run
34.5–50.5% against a submitted 47.5%. **Both cells show the same thing: the submitted numbers
are single draws from distributions with ~6pp (ALFWorld) and ~5pp (WebShop) spread, and both
sit above their means.** Presenting the two cells with one consistent story — "we report
distributions now, and here is what changes" — is stronger and more coherent than treating them
as two separate incidents, which is how the current draft reads.
