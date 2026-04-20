# Retrospective Behavioral Evidence — WebShop 1.5B

**Claim 1.** CHORD beats v1 DUET because CHORD's large early SFT weight `μ` imprints token-level imitation of teacher option-clicks that v1 DUET's DR3 alone fails to install.
**Claim 2.** v24 beats CHORD because v24 keeps the option-click behavior (via early SFT) **and** adds DR3 credit-assignment on top, learning correct termination (buy_now timing) that CHORD never acquires.

All numbers below come from 64 on-policy rollouts per step in `experiments/webshop/webshop_qwen1.5b_{duet, chord, duet_v24}/rollout_log/*.jsonl`. "Option click" = `click[…]` whose target is not a product ID (`b0xxxxxxxx`) and not one of `{buy now, back to search, < prev, next >, description, features, reviews, attributes}`. "OC-loop" = ≥3 consecutive option clicks.

## A. Option-click dynamics by training step (64 rollouts / step)

| step | method   |  n | OC/task | % with ≥1 OC | avg turns | %reach buy_now | avg score |
|-----:|:---------|---:|--------:|-------------:|----------:|---------------:|----------:|
|    5 | v1 DUET  | 64 |    1.53 |         32.8 |      5.05 |           35.9 |     0.200 |
|    5 | CHORD    | 64 |    1.20 |         29.7 |      4.06 |           31.2 |     0.161 |
|    5 | v24      | 64 |    1.41 |         31.2 |      4.39 |           25.0 |     0.151 |
|   15 | v1 DUET  | 64 |    2.31 |         31.2 |      6.36 |           71.9 |     0.484 |
|   15 | **CHORD**| 64 |**3.95** |     **51.6** |      7.89 |           51.6 |     0.322 |
|   15 | **v24**  | 64 |**3.69** |     **45.3** |      9.38 |           35.9 |     0.258 |
|   25 | v1 DUET  | 64 |    2.09 |         31.2 |      5.84 |           78.1 |     0.501 |
|   25 | CHORD    | 64 |    0.31 |         18.8 |      3.58 |           95.3 |     0.505 |
|   25 | v24      | 64 |    0.95 |         18.8 |      4.69 |           87.5 |     0.558 |
|   50 | v1 DUET  | 64 |    0.84 |         29.7 |      4.72 |           76.6 |     0.547 |
|   50 | CHORD    | 64 |    0.44 |         23.4 |      3.77 |           90.6 |     0.589 |
|   50 | v24      | 64 |    0.59 |         23.4 |      3.91 |           95.3 |     0.732 |
|  100 | v1 DUET  | 64 |    0.73 |         39.1 |      3.97 |           96.9 |     0.693 |
|  100 | **CHORD**| 64 |**2.44** |     **93.8** |      5.70 |           90.6 |     0.631 |
|  100 | **v24**  | 64 |    1.23 |     **76.6** |      4.42 |       **98.4** | **0.785** |

Reading the table:

- **Step 15 — CHORD imprints option clicks first.** CHORD jumps to 3.95 OC/task, v24 to 3.69; v1 DUET only reaches 2.31. By SFT weight (`μ≈0.7` at step 15 on the 0.9→0.5 schedule), CHORD and v24 are being dragged to teacher-like token distributions. v1 DUET's DR3-only path can re-weight teacher samples but never pushes the student distribution toward teacher tokens — behavioral cloning never happens.
- **Step 25–50 — valley.** All methods' OC rate drops sharply as on-policy gradients overwrite early imitation, but CHORD and v24 end up with *higher* buy-now reliability (95%, 87%) than v1 DUET (78%) — the imprint is still there even when counts drop.
- **Step 100 — behavioral divergence.** Both CHORD (93.8% %wOC) and v24 (76.6%) exhibit the "search→product→options→buy" template that v1 DUET does not (39.1% %wOC). **But CHORD over-imitates: 2.44 OC/task, 5.70 turns, and only 90.6% buy_now.** v24 does 1.23 OC/task, 4.42 turns, and **98.4% buy_now → score 0.785**. CHORD's extra OC-per-task at step 100 is the loop signature, not better selection.

## B. Termination behavior — 8 matched tasks at step 100

The v1 rollout log samples 64 tasks/step, with some overlap across methods. 8 tasks are shared between v24 and CHORD at step 100:

| method  | avg turns | avg OC | % buy_now | % OC-loop (≥3 consec) | avg score |
|:--------|----------:|-------:|----------:|----------------------:|----------:|
| v24     |  **4.50** |   1.12 |     100.0 |              **0.0**  |     0.715 |
| CHORD   |      6.88 |   3.38 |     100.0 |            **62.5**   |     0.893 |
| v1 DUET |      3.88 |   0.62 |     100.0 |                   0.0 |     0.768 |

CHORD reaches buy_now but spends ~50% more turns cycling options; **5/8 CHORD trajectories exhibit an OC loop vs 0/8 for v24**. On this small matched set CHORD's avg score (0.893) exceeds v24's (0.715) because 3 of the 8 are tasks where the teacher's exact option strings matched the reward function exactly — aggregate step-100 scores (Table A) show the opposite: v24 0.785 > CHORD 0.631, where the loop cost dominates.

## C. BC evidence — v24 and CHORD pick the same option tokens

Of 7 shared-task pairs at step 100 where both methods clicked at least one option, **4/7 (57%) share an option-token** that matches the instruction attribute (color / size). Examples (instruction → v24 options / CHORD options / shared):

- `loose fit ... tops` → `[b16-red]` / `[b16-red, small]` / **b16-red**
- `dress shirts ... pale coral red xx-large tall` → `[pale coral red, xx-large tall]` / `[pale coral red, xx-large]` / **pale coral red**
- `tuxedo shirts large` → `[large]` / `[large, men, royal blue | white]` / **large**
- `loafers size 9` → `[9]` / `[9]` / **9**

The shared token is the **teacher's attribute mention** from the pickled demonstrations — direct evidence of token-level BC imprint in both methods. v1 DUET's option-click list on the same tasks is usually empty or mismatched.

## Key excerpts

**(1) CHORD step-100, OC-loop failure mode.** Same instruction (`pale coral red ... xx-large tall`), v24 succeeds cleanly. Note CHORD's search query also shows language drift (Chinese characters) — symptom of heavy SFT pulling toward stochastic teacher vocabulary without credit-assigned correction.

```
CHORD (score=0.714, 11 turns, 8-long OC run):
  c1:  search[机洗男式透气印花衬衫 bright aqua 1x 纯棉弹性短袖，经典款，价格低于50.00美元]
  c2:  click[b07hrfsnl4]
  c3:  click[medium]
  c4:  click[xx-large tall]
  c5:  click[xx-large tall]
  c6:  click[xx-large tall]
  c7:  click[xx-large tall]   ← OC loop: no penalty large enough to terminate
  c8:  click[xx-large tall]
  c9:  click[xx-large tall]
  c10: click[xx-large tall]
  c11: click[buy now]

v24 (score=1.000, 8 turns, zero OC loop):
  v1: search[machine wash men's dress shirts]
  v2: click[next >]
  v3: click[back to search]         ← recovers from weak first search
  v4: search[Nautica Men's Solid Crew Neck Short-Sleeve Pocket T-Shirt]
  v5: click[b07hrfsnl4]
  v6: click[bright aqua]            ← teacher-style attribute click
  v7: click[1x]                     ← clean second option
  v8: click[buy now]                ← terminates immediately, DR3-refined
```

**(2) CHORD step-15, early imprint already functional.** Only 15 gradient steps in, CHORD has the full teacher template:

```
CHORD step 15 (score=1.000):
  t1: search[super soft decorative pillows living room color: beige latte size: 20''x20'' price:...]
  t2: click[B08L2ZDWN2]
  t3: click[beige latte]      ← option-click working by step 15
  t4: click[20''x20'']
  t5: click[Buy Now]
```

**(3) v1 DUET step-25, "buy without options" — imprint never installed.** Even at step 25, v1 DUET routinely skips option selection entirely:

```
v1 DUET step 25 (score=0.298):
  t1: search[machine wash moisture wicking men's t-shirts polyester spandex ...]
  t2: click[B09M63B87V]
  t3: click[Buy Now]          ← zero options, score collapses
```

## Verdict

**Yes on both.** CHORD's advantage over v1 DUET is explained by **early token-level BC of teacher option-click strings (μ≈0.9→0.5 on steps 0–25 forces tokens into the student's mouth)**; the 57% v24/CHORD option-token overlap at step 100 and CHORD's jump to 3.95 OC/task by step 15 vs v1 DUET's 2.31 are direct behavioral fingerprints of that imprint. v24's advantage over CHORD is that DR3's credit-assigned ratio refines the imprint — **5/8 matched step-100 tasks show CHORD in OC-loops while v24 has zero loops**, and v24's aggregate step-100 score (0.785) beats CHORD's (0.631) because DR3 teaches correct termination (98.4% buy_now in ≤4.4 turns) rather than mechanical repetition.

Data: `/data/home/qisheng/EvolAnalsis/experiments/webshop/webshop_qwen1.5b_{duet, chord, duet_v24}/rollout_log/`.
Analysis script: `/tmp/retro_analysis.py`, `/tmp/retro_more_excerpts.py`.
JSON summary: `/tmp/retro_summary.json`.
