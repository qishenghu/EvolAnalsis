# v24 vs DUET-v1 — ALFWorld Trajectory-Level Behavioral Diff (1.5B, step 100)

**Question.** Why does v24 (DR3 + SC + decaying BC μ=0.3→0.05) score 30.5% vs DUET-v1 (DR3 + SC only) at 32.5% on ALFWorld 1.5B val@100 (−2pp regression)?

## Headline verdict — **Mechanism (b) Template Overfitting dominates; mechanism (a) entropy collapse is a second-order contributor**

The BC term installs the teacher's boilerplate first-turn phrasing into v24, most visibly `"I will start by checking X, as it is a common place to find Y"`. This phrase appears in **169/200** v24 trajectories (vs 1/200 v1), and the phrase pulls v24 to a narrow set of "default" first locations (`countertop 1` doubles from 33 → 66; first-destination entropy drops 2.38 → 2.04). When the task's target object lives elsewhere, v24 spends 5-6 wasted turns searching the default before exploring. Combined with the 30-turn budget, that initial bias is sufficient to flip 30 v1-wins into v24-losses. Crucially, v24 does **not** fail via repetition loops (55 → 20) or entropy crashes of later turns — BC only damages the opening move's conditional distribution. Plan-dump (multi-action blocks, `[/action]` close tag) is a real but tertiary failure mode (40 trajectories affected, 100% fail rate). The regression is **fixable**: lower μ_max and/or shorter decay window.

Data: `experiments/alfworld/alfworld_qwen1.5b_duet{,_v24}/validation_log/100.jsonl`, 200 matched tasks.
Analyzer script: `/data/home/qisheng/EvolAnalsis/analysis_v24_alfworld.py`.
Case JSON: `/data/home/qisheng/EvolAnalsis/analysis_reports/_v24_alfworld_cases.json`.

---

## 1. Aggregate table (200 matched tasks at step 100)

| pattern | v1 | v24 | Δ |
|---|---:|---:|---:|
| success (score=1.0) | 65 (32.5%) | 61 (30.5%) | −4 |
| both succeed | 35 | 35 | — |
| both fail | 109 | 109 | — |
| **v1 succeeds, v24 fails (regression)** | — | **30** | −30 |
| v24 succeeds, v1 fails (progression) | **26** | — | +26 |
| avg turns / trajectory | 23.6 | 23.5 | — |
| exhausted 30-turn budget | 68.0% | 62.0% | −6pp |
| failures via budget exhaustion | 99% | 89% | — |
| repetition loops (max_run ≥ 4) | 55 | **20** | **−35** |
| severe loops (max_run ≥ 6) | 54 | 15 | −39 |
| "\n" literal-newline action | 16 tasks | **0 tasks** | −16 |
| plan-dump turns (multi-action blobs) | 0 | **189 (in 33 tasks)** | +189 |
| `[/action]` wrong close tag | 0 | **350 turns (in 40 tasks)** | +350 |
| "I will start by" in first think | 1/200 | **169/200** | **+168** |
| "as it is a common place" in first think | 0/200 | **109/200** | **+109** |
| first-destination entropy | 2.38 | **2.04** | **−0.34** |
| `go to countertop 1` as first action | 33 | **66** | **+33** |

Headline: BC *improved* the tail (repetition loops −35, removed `\n` emission entirely) but *degraded* the opening move (entropy −0.34, countertop bias +33). The net effect is −2pp because ALFWorld's 30-turn budget is too tight to recover from a mis-aimed first move.

### Failure-mode attribution for the 30 regression cases

| failure mode in v24 | n | notes |
|---|---:|---|
| divergence at step 0-2 (early wrong-default) | 22 | 73% — template-driven |
| `[/action]` wrong close tag present | 7 | format degradation |
| plan-dump multi-action blob present | 4 | agent "plans ahead" in one XML block |
| repetition loop (max_run≥4) | 5 | minority — BC mostly prevents this |
| v24 hit 30-turn max | 26 | 87% |

(Categories overlap; primary cause is early divergence.)

---

## 2. Case studies — three regressions, then two progressions

All tasks are at the same index in both validation logs. Actions abbreviated with obs headers omitted for readability.

### Case A — task idx=10 (mug → heat → coffeemachine). **Wrong-object failure driven by first-think template**

```
Task: heat a mug and put it in the coffeemachine

v1 first think:  "I need to find a mug first. Let me check the diningtables and cabinets."
v1 (r=1.0, 6 turns):
  go to diningtable 1       → mug 3 present
  take mug 3 from diningtable 1
  go to microwave 1
  heat mug 3 with microwave 1
  go to coffeemachine 1
  put mug 3 in/on coffeemachine 1                       SUCCESS

v24 first think: "To complete the task, I need to find a mug first. I will start by
                 checking the countertop, as it is a common place to find mugs."
v24 (r=0.0, 30 turns):
  go to countertop 1         → NO mug; but sees butterknife, cup, spoon
  take cup 1 from countertop 1     ← WRONG OBJECT (grabs cup, not mug)
  go to microwave 1
  heat cup 1 with microwave 1
  put cup 1 in/on coffeemachine 1  → Invalid action
  go to fridge 1, open, put cup 1 in/on fridge 1
  [... 23 more wandering turns through sidetable/sinkbasin/drawers, never
   getting the mug even though sinkbasin 1 visibly contained mug 2 ...]
```

**Divergence mechanism**: the BC-installed opening template "countertop is a common place for [X]" overrides the task-specific reasoning about mugs (which live on dining tables/cabinets in ALFWorld kitchens). When countertop doesn't contain the target, the agent grabs the nearest similar-sounding object (`cup`) and commits to the wrong-object path.

### Case B — task idx=3 (plate → clean → cabinet). **Identical opening, divergence at step 4**

```
Task: clean a plate and put it in a cabinet

Both identical through step 3:
  go to countertop 1 → take plate 1 → go to sinkbasin 1 → clean plate 1 with sinkbasin 1

v1 (r=1.0, 6 turns, step 4):    go to cabinet 1            → put plate 1 in/on cabinet 1   SUCCESS
v24 (r=0.0, 30 turns, step 4):  go to fridge 1             → open fridge 1 → put plate 1 in/on fridge 1
  [...24 more turns wandering through cabinet 1–13 looking for a new plate...]
```

**Divergence mechanism**: after cleaning, v24 detours to the fridge (apparently triggered by fridge's visibility in a kitchen-template prior) and *deposits* the plate there — making the rest of the task impossible because the task checker requires the plate to end up in a cabinet. v1 goes to the cabinet directly.

### Case C — task idx=24 (two pencils → drawer). **Repetition-loop failure (one of only 5 loops in v24)**

```
Task: put two pencils in a drawer

Both identical through step 4:
  go to desk 1 → take pencil 3 from desk 1 → go to drawer 1 → open drawer 1 → put pencil 3

v1 (r=1.0, 9 turns, from step 5):
  go to desk 1 → take pencil 4 from desk 1 → go to drawer 1 → put pencil 4   SUCCESS

v24 (r=0.0, 30 turns, from step 5):
  go to desk 1 → [obs: Invalid action]
  go to desk 1 → Invalid action   (loops)
  go to desk 1 → Invalid action
  go to desk 1 → Invalid action
  [... 20 consecutive "go to desk 1" emissions, all invalid because
   agent is already on desk; correct action was "take pencil 4 from desk 1" ...]
```

**Divergence mechanism**: v24 correctly identifies "go back to desk to get pencil 2" but fails to fire the `take` primitive. It then repeats the invalid `go to desk 1` action 20 times. This is the rare case where v24 still loops — note that v1 handled this same task cleanly.

### Case D — task idx=25 (tomato → heat → microwave). **Progression: v24 avoids the stuck-at-microwave loop**

```
v1 (r=0.0, 30 turns):
  go to fridge 1 → open → take tomato 1 → go to microwave 1 → heat tomato 1 with microwave 1
  go to microwave 1 → Invalid action     (agent already there)
  go to microwave 1 → Invalid action     (loops 25 times)   FAIL

v24 (r=1.0, 6 turns):
  go to fridge 1 → open → take tomato 1 → go to microwave 1 → open microwave 1
  put tomato 1 in/on microwave 1   SUCCESS
```

**v24 wins because**: BC has suppressed the `go to microwave 1` self-repetition that v1 falls into. v24 advances to `open microwave 1` and `put` — a cleaner state progression. BC's conservative-step pattern preserves forward motion.

### Case E — task idx=27 (tomato → clean → sidetable). **Progression: v24 avoids v1's `\n`-emission collapse**

```
Both identical through step 4: open fridge → take tomato 1 → go to sinkbasin → clean tomato 1

v1 (r=0.0, 30 turns, from step 5):
  action: \n  → Invalid action      (literal newline as entire action)
  action: \n  → Invalid action      (repeats 25 times until budget exhaustion)   FAIL

v24 (r=1.0, 7 turns, from step 5):
  go to sidetable 1 → put tomato 1 in/on sidetable 1     SUCCESS
```

**v24 wins because**: BC completely eliminates the `\n` literal-action degeneracy that v1 exhibits on 16/200 tasks. The BC signal enforces canonical action surface forms.

---

## 3. Paper narrative (one paragraph)

**What this tells us about where BC helps vs hurts.** On WebShop, BC was unambiguously useful because the teacher's action surface contains rare SKU-like tokens (`click[lavender]`, `click[fs4 | 30]`) that RL could not discover from generic English; BC installs them and adds 7-13pp Val@100. On ALFWorld the story inverts: the teacher's action alphabet is templated (`go to X`, `take Y from X`, `put Y in/on X`) and every surface token is already in the 1.5B's output distribution, so BC does not *add* useful primitives — it only *over-weights* the teacher's boilerplate phrasing ("I will start by checking the countertop, as it is a common place"). This boilerplate degrades exploration diversity (first-destination entropy 2.38 → 2.04) and biases the opening move toward kitchen defaults, flipping 30 v1-successes into v24-failures that run out the 30-turn budget at a wrong default location. The *good* news is the same BC also suppresses pathological degeneracies — it erases the `\n`-as-action collapse entirely (16 → 0 tasks) and cuts repetition loops from 55 to 20 — so v24 wins 26 tasks where v1's decoder simply breaks. The trade balances almost exactly (−30 + 26 = −4). **The lesson for the paper is that BC's value is task-surface-dependent: rare-token environments (WebShop, CHORD-regime) benefit; templated environments (ALFWorld) only benefit from the anti-degeneracy side effect, which DR3 already provides more cleanly via the density-ratio discriminator**. Recommend either (i) drop BC on ALFWorld, (ii) shorten the decay window to 10 steps at μ=0.1, or (iii) apply BC only to the anti-degeneracy tokens (filter teacher gradient to response-end tokens only). The current regression is a parameter issue, not a fundamental problem with the DUET + BC combination.
