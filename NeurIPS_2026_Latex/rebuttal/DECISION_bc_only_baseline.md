# Authors' decision needed: a behaviour-cloning-only run matches DUET on 1.5B-ALFWorld

Found 2026-07-27 while fact-checking the response to Reviewer y9x6, who asked precisely this:
*"does DUET outperform prior methods because of its separation and correction mechanisms, or
because it receives more effective supervised guidance from teacher trajectories?"*

This is discoverable from our own `experiments/` directory, so a reviewer can find it.

## The measurement

`alfworld_qwen1.5b_sft` is not supervised fine-tuning in the usual sense. Diffing it against the
CHORD config shows the **only** substantive differences are the BC schedule and the budget:

| | CHORD | `..._sft` |
|---|---|---|
| `chord_mu_peak` / `chord_mu_valley` | 0.9 / 0.05 | **1.0 / 1.0** |
| `chord_mu_decay_steps` | 25 | **0** (never decays) |
| `max_train_tasks` | 800 (100 steps) | 400 (50 steps) |
| `use_dr3` | false | false |
| State Channel | absent | absent |
| teacher mixing | on | on |

So it is GRPO with teacher mixing and **behaviour cloning held at full weight throughout** — i.e.
the "just imitate the teacher and never fade" configuration.

Results on the same fixed 200-task validation split:

| 1.5B-ALFWorld | val@50 | val@100 |
|---|---|---|
| **BC-only (μ ≡ 1.0, 400 tasks / 50 steps)** | **47.5%** | — (run ends) |
| DUET | 42.5% | **47.5%** |
| CHORD (μ decays) | 30.0% | 27.0% |
| SFT+GRPO (RL continued from the BC checkpoint) | 30.0% | — |
| on-policy GRPO | 16.5% | 1.0% |

| 1.5B-WebShop | val@50 | val@100 |
|---|---|---|
| **BC-only (μ ≡ 1.0)** | **7.0%** (reward 0.562) | — |
| DUET | 1.0% | 35.5% (does not reproduce — see `DECISION_webshop_1p5b_cell.md`) |
| SFT+GRPO | 18.0% (reward 0.641) | — |
| CHORD | 3.0% | 11.5% |
| on-policy GRPO | 2.0% | 0.5% |

## What this means

**On 1.5B-ALFWorld, full-weight behaviour cloning alone reaches DUET's final number using half the
optimisation steps and half the tasks.** At the matched step-50 checkpoint it is *ahead* of DUET
(47.5% vs 42.5%). Against that cell, DUET's teacher-utilisation machinery shows no gain.

**On 1.5B-WebShop the same configuration reaches only 7.0%**, where DUET reaches 35.5% and even
SFT+GRPO reaches 18.0%. So the result is environment-specific, not a general refutation.

Two further observations from the same table, both of which cut in our favour and are worth
reporting alongside:

- The BC-only solution is **not stable under continued RL**: continuing from that checkpoint with
  50 GRPO steps (the SFT+GRPO baseline) *drops* ALFWorld from 47.5% to 30.0%. DUET holds 47.5% while
  doing 100 steps of RL.
- The on-policy baseline **collapses with more training** on ALFWorld: 16.5% at step 50 to 1.0% at
  step 100. Whatever else is true, teacher data is preventing a real failure mode.

## Why it was not in the paper

`alfworld_qwen1.5b_sft` exists as the *first stage* of the SFT→GRPO baseline, not as a baseline in
its own right; its validation number was never read as a standalone result. That is the mistake: on
this cell it is the strongest non-DUET configuration we have, and Table 1 reports SFT+GRPO (30.0%)
as the strongest baseline instead.

## Options

**A — report it.** Add a BC-only row to the ablation/baseline table and state plainly that on
1.5B-ALFWorld it matches DUET, while on 1.5B-WebShop it reaches 7.0% against DUET's 35.5%. Pair it
with the stability observation (continued RL degrades the BC solution 47.5% → 30.0%) and the
GRPO-collapse observation. This concedes the ALFWorld margin and reframes the contribution as
"retains imitation-level performance while continuing to improve under RL, and works where
imitation does not".

**B — run the missing controls first.** The BC-only cell was run for 50 steps on 400 tasks. Before
conceding, run it at DUET's budget (800 tasks / 100 steps) on both environments. Two outcomes:
if it degrades with more training — which the SFT+GRPO trajectory suggests — the comparison at
matched budget is far less damaging, and we can report the matched-budget number honestly. If it
holds at 47.5%, option A is forced.

**C — say nothing.** Not viable. y9x6 asked this exact question, the run is in our own experiment
directory, and the ablation table we are already citing is generated from that directory.

**Recommendation: B, then A.** Running BC-only at the matched budget costs two runs (~7h) and is the
difference between conceding a margin and reporting a fair comparison. It should go to the front of
the queue, ahead of the remaining seed replicates. Whatever it shows, the cell gets reported —
having found this ourselves and reported it is much better than having a reviewer find it.
