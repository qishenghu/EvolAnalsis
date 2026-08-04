# Open question: the 1.5B-WebShop cell of Table 1

Reviewer y9x6 asked for multi-seed results. On this cell the answer is not yet in, and what we can
claim depends on it. Replicates on the paper's own curriculum are running.

## Status

**`ws_1_5b_swC02_da` is excluded.** The forensic audit leaned on it as a same-seed, same-curriculum
replica scoring 1.0%, but the authors report that this run was faulty at the time it executed. It is
therefore not evidence about reproducibility and is not counted anywhere below. (Noted for
provenance: the exclusion rests on the authors' record of the run, not on a fault we could
independently identify in its logs.)

What remains open is the straightforward question the reviewer asked, and the replicates now running
answer it directly: **does the submitted 35.5% reproduce when only the run-time randomness changes?**

## What we know so far — including one like-for-like replicate

| run | seed | training tasks | **val@50** | val@100 strict | val@100 mean reward |
|---|---|---|---|---|---|
| `..._duet_swC_02_pk03_v10_floor06` (**the paper's**) | 2026 | the 800 | 1.0% | **35.5%** | 0.706 |
| `..._duet_a100_seed2025` | 2025 | a different 800 (89 shared) | 1.5% | **3.5%** | 0.521 |
| `..._duet_a100_seed2027` | 2027 | **the identical 800** (`task_seed: 2026`) | 1.0% | **2.5%** | 0.544 |

The third row is the controlled comparison, and it is not favourable. Code identity was verified
against the paper run's own `launcher_record` backup — `het_core_algos.py` and `dr3_ratio.py` are
byte-identical and the 113 changed lines in `het_actor.py` are *pure additions* of μ-mode branches
the paper's `disc_acc` config never enters — and the executed configs differ only in `seed` and
`task_seed`. Its curriculum was verified live: every task it trained on came from the reference's
800. All four DUET mechanisms were healthy throughout (`dr3/disc_acc` 0.987, teacher gradient share
0.131, `chord/mu` 0.105, SC coverage 0.850, degenerate repetition 0.4%).

**The pattern that matters: all three runs are indistinguishable at val@50 (1.0 / 1.5 / 1.0%).**
They diverge only across the last 50 steps — the paper's run completed its take-off inside the
budget and the two replicates did not. On mean reward, which is not a threshold, the two replicates
sit together (0.521, 0.544) below the paper run (0.706).

For context on volatility in this setting, across the 66 1.5B-WebShop runs on disk (a
hyperparameter sweep, not replicates) the paper's run is the highest and the median is 3.0%.

Why this cell is volatile at all: strict success on WebShop requires an exact match on every
requested attribute, and at a 100-step budget these runs are still inside their take-off. Train-side
strict success over steps 81–100 is 9.6% (paper) versus 6.5% (seed 2025), while mean reward over the
same window is 0.567 versus 0.546 — a 1.5× spread in the training signal becomes a 10× spread in the
reported metric. Whatever the replicates show, this cell needs to be reported with that in mind.

Ruled out with evidence: environment fault (100,924 requests, all HTTP 200, one server lifecycle,
route counts exactly as expected), teacher-cache mismatch (coverage 85.8% vs 83.5%, both inside a
Monte-Carlo band), validation-set difference (val loads unshuffled and seed-free; the union of
recovered instructions across all six validation logs is exactly 200), and train/test leakage
(intersection 0 for both task subsets). Details in `forensics_VERDICT.md`.

## What is *not* implicated

- **The other three cells of Table 1.** ALFWorld's reward is binary, so it has no knife-edge
  threshold; its three-seed replicate spreads 47.5 / 38.0 / 41.0 (sd 4.9pp). The 3B cells are
  separate runs on another machine and are untouched by this evidence — though note they are
  separately unverifiable here (see `paper_corrections.md` §C0).
- **The internal fairness of Table 1.** Every 1.5B method used `seed: 2026`; recovered from the
  saved rollouts, CHORD and on-policy GRPO trained on *exactly* the same 800 tasks as DUET, and
  SFT+GRPO's 400 are a subset. Whatever we do to this cell, the comparison was matched.

## The experiments that decide it (running now)

All on the paper's own 800-task curriculum (`task_seed: 2026`), so only run-time randomness varies:

| runs | purpose |
|---|---|
| **DUET seed 2027 at a 150-step budget, same 800 tasks — now first in the queue** | the decisive one: does the take-off simply happen later? |
| DUET seeds 2025, 2028 at 100 steps | completes the 4-point estimate at the submitted budget |
| SFT+GRPO seeds 2025, 2027 | is the *baseline* equally volatile at this budget? |

## The three options, and a recommendation

**Option A — report the distribution and keep the cell.** Replace the point estimate with
mean ± sd over the replicates and say plainly that the submitted number was a high draw. Defensible
only if the replicate mean still exceeds the baseline's replicate mean.

**Option B — report the cell on mean reward instead of strict success.** Mean reward is not
knife-edge and the spread is far smaller (0.706 vs 0.521 across the two runs so far). But this changes the metric after
seeing the result, and SFT+GRPO's single run already sits at 0.641 — so this may not rescue the
ordering either. Only acceptable if presented as an *additional* metric, never as a replacement.

**Option C — extend the budget for this cell** so every method is past the transition, and report
that. Scientifically the cleanest reading of the evidence, but it is a protocol change made after
seeing the result, and it costs a full re-run of every method in the column.

**Recommendation: A, with B as a supporting column, and C only if the 150-step diagnostic shows
every method converging to a stable ordering.** Concretely: report DUET and SFT+GRPO as mean ± sd
over the replicates on the fixed curriculum, keep strict success as the headline metric for
comparability with the submission, and add mean reward alongside. If the replicates cluster near
the submitted number, the cell stands and we simply report the spread. If they do not, say so and
either report the cell as a distribution or mark it inconclusive at this budget — but decide that
from the replicates, not from the two runs we have now.

Whatever the replicates show, the spread should be stated in the response rather than left for a
reviewer to find — they asked for exactly this measurement.

## Wording ready to use once the replicates land

> Reviewer y9x6 asked for multi-seed results, and on 1.5B-WebShop they change what we can claim.
> Replicating the submitted configuration on its own training tasks gives DUET {…} and SFT+GRPO
> {…} (mean ± sd over k runs). The cause is
> that WebShop's strict success requires an exact match on every attribute while a 100-step budget
> leaves these runs mid-take-off: a 1.5× spread in training reward becomes a 35× spread in strict
> success. We therefore report this cell as a distribution rather than a point and add mean reward
> (which is not knife-edge) alongside. The ALFWorld
> cells are unaffected — the reward there is binary and the three-seed spread is 47.5/38.0/41.0.
