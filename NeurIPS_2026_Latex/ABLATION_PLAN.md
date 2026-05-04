# DUET Ablation Plan — 2-Day Sprint Toward NeurIPS 2026

**Created**: 2026-05-04 evening
**Last revised**: 2026-05-04 (full 4×4 ablation matrix; user-confirmed timing)
**Compute window**: 2026-05-05 00:00 → 2026-05-06 23:59 (~48h)
**Buffer**: 2026-05-07 reserved for paper writing & data integration
**Servers**: 4×A100-80GB (this server, "us") + 4×L20X-144GB ("L20X", collaborator)

---

## 1. Goal

Fill the **full ablation table** of the paper — every (mechanism × setting) cell — supporting the narrative that each of DUET's four mechanisms contributes to performance, across both model sizes and both environments.

| Mechanism | Paper role |
|---|---|
| Baseline separation (Fix 1) | Eliminates baseline contamination bias |
| DR3 density-ratio correction (Fix 2) | Eliminates off-policy mismatch + provides auto-fade |
| BC adaptive imitation (Extension 1) | Cold-start safety net while DR3 stabilizes |
| SC potential-based shaping (Extension 2) | Dense per-state guidance with policy invariance |

**Coverage target — full 4 × 4 matrix (16 cells)**:

| Mechanism removed | 1.5B-AF | 1.5B-WS | 3B-AF | 3B-WS |
|---|:-:|:-:|:-:|:-:|
| -baseline_sep | ✅ A1 | ✅ A5 | ✅ L1 | ✅ L5 |
| -DR3          | ✅ A2 | ✅ A6 | ✅ L2 | ✅ L6 |
| -BC           | ✅ A3 | ✅ A7 | ✅ L3 | ✅ L7 |
| -SC           | ✅ A4 | ✅ A8 | ✅ L4 | ✅ L8 |

**Server split**: A100 owns all 1.5B ablations (8 runs); L20X owns all 3B ablations (8 runs).

**Single-seed caveat**: All cells single-seed by necessity. WS cells have known higher variance (1.5B WS swC_02 reproduces in [1%, 36%] across local attempts) — handled by a caveat sentence in §Limitations and binomial CI on val@200 in the table footnote.

---

## 2. Time Budget

| Server | GPU config | Wall-clock per run | 8-run sequential total |
|---|---|---|---|
| 4×A100-80GB (us) | TP=4, 1.5B model | **~5h** (user-verified) | ~40h ✓ fits in 48h |
| 4×L20X-144GB (L20X) | TP=4, 3B model | **~8–10h** (estimate; L20X verify) | ~64–80h ⚠ tight; prioritize order below |

**Hard rule**: All experiments must finish by **2026-05-06 23:59 local**. If a run shows clear collapse before step 50, kill early and move to next.

**If L20X runs over budget**: drop the lowest-priority cells in the order **-SC on WS → -BC on WS → -baseline_sep on WS → -SC on AF**. The cells that **must** finish on L20X are the four **-DR3** cells (Fix 2 is the most novel mechanism — needs full cross-setting evidence).

---

## 3. A100 Schedule (Us — 8 runs, ~40h)

Sequential queue. Each run uses all 4 GPUs (TP=4). Priority order: -DR3 first across both envs (most critical), then -baseline_sep (Fix 1), then -BC, then -SC.

| ID | Experiment | Base config | Knob to flip | Wall-clock | Window |
|---|---|---|---|---|---|
| **A2** | 1.5B-AF -DR3 | `alfworld_qwen1.5b_duet_v39c_postfix.yaml` | `actor.use_dr3: false`; `actor.dr3.enable: false` | 5h | 05-05 00:00 → 05:00 |
| **A6** | 1.5B-WS -DR3 | 1.5B-WS SOTA (`swC_02` family — 36.0% cell) | same | 5h | 05-05 05:00 → 10:00 |
| **A1** | 1.5B-AF -baseline_sep | 1.5B-AF SOTA | `algorithm.grpo.teacher_baseline_separation.enable: false` | 5h | 05-05 10:00 → 15:00 |
| **A5** | 1.5B-WS -baseline_sep | 1.5B-WS SOTA | same | 5h | 05-05 15:00 → 20:00 |
| **A3** | 1.5B-AF -BC | 1.5B-AF SOTA | `actor.use_chord: false` | 5h | 05-05 20:00 → 05-06 01:00 |
| **A7** | 1.5B-WS -BC | 1.5B-WS SOTA | same | 5h | 05-06 01:00 → 06:00 |
| **A4** | 1.5B-AF -SC | 1.5B-AF SOTA | `exp_manager.state_channel.enable: false` | 5h | 05-06 06:00 → 11:00 |
| **A8** | 1.5B-WS -SC | 1.5B-WS SOTA | same | 5h | 05-06 11:00 → 16:00 |

**Output naming**: `<env>_qwen1.5b_duet_minus_<mechanism>` — e.g. `alfworld_qwen1.5b_duet_minus_dr3`. Save logs to `experiments/<env>/<run_name>/`. Log to wandb under `project_name: agentevolver`, `experiment_name: <run_name>`.

**Slack**: ~8h between A8 finish (05-06 16:00) and budget end (05-06 23:59) for re-runs of any failed cell.

---

## 4. L20X Schedule (Collaborator — 8 runs, ~64–80h)

**This is tight on 48h. Run in priority order — if running long, drop tail cells per §2 fallback rule.**

Sequential queue. Each run uses all 4 GPUs (TP=4). Priority order: same as A100.

| ID | Experiment | Base config | Knob to flip | Est. wall-clock |
|---|---|---|---|---|
| **L2** | 3B-AF -DR3 | 3B-AF DUET\* SOTA (the v39b config; 77.5%) | `actor.use_dr3: false`; `actor.dr3.enable: false` | 8–10h |
| **L6** | 3B-WS -DR3 | 3B-WS DUET\* SOTA (the v39b config; 45.5%) | same | 8–10h |
| **L1** | 3B-AF -baseline_sep | 3B-AF SOTA | `algorithm.grpo.teacher_baseline_separation.enable: false` | 8–10h |
| **L5** | 3B-WS -baseline_sep | 3B-WS SOTA | same | 8–10h |
| **L3** | 3B-AF -BC | 3B-AF SOTA | `actor.use_chord: false` | 8–10h |
| **L7** | 3B-WS -BC | 3B-WS SOTA | same | 8–10h |
| **L4** | 3B-AF -SC | 3B-AF SOTA | `exp_manager.state_channel.enable: false` | 8–10h |
| **L8** | 3B-WS -SC | 3B-WS SOTA | same | 8–10h |

**If forced to drop cells** (in this order): L8 (-SC WS) → L7 (-BC WS) → L5 (-baseline_sep WS) → L4 (-SC AF). Keep all four -DR3 cells at all costs.

---

## 5. Config Recipes (exact knobs, verified against `alfworld_qwen1.5b_duet_v39c_postfix.yaml`)

All ablations fork from the corresponding DUET\* SOTA config and flip **only** the listed knob(s). No other changes.

### -baseline_separation (Fix 1)
```yaml
algorithm:
  grpo:
    teacher_baseline_separation:
      enable: false   # was: true
```
**Operational meaning**: GRPO computes group baseline jointly over teacher + on-policy samples (LUFFY behavior). Should expose Bias 1.

### -DR3 (Fix 2)
```yaml
actor_rollout_ref:
  actor:
    use_dr3: false              # was: true
    dr3:
      enable: false             # was: true
```
**Operational meaning**: Teacher samples use the unmodified GRPO importance ratio (LUFFY behavior on the gradient side). Should show persistent teacher gradient share and damaged late-training performance.

### -BC (Extension 1)
```yaml
actor_rollout_ref:
  actor:
    use_chord: false            # was: true
```
**Fallback** (if `use_chord: false` causes loss-construction issues — verify in `het_actor.py` first):
```yaml
actor_rollout_ref:
  actor:
    use_chord: true
    chord_mu_peak: 0.0
    chord_mu_valley: 0.0
    chord_mu_adaptive: false
```
**Operational meaning**: No token-level imitation pressure. DUET runs with DR3-corrected GRPO + SC only. Should show slower cold-start, especially on weak base models.

### -SC (Extension 2)
```yaml
exp_manager:
  state_channel:
    enable: false               # was: true
```
**Operational meaning**: No per-step reward shaping. On-policy reward is sparse (env reward only). Should show reduced sample efficiency on long-horizon tasks.

---

## 6. Reporting Template

For each finished run, append a row to **`NeurIPS_2026_Latex/data/ablation_results.md`** (create on first run):

```markdown
| Run ID | Setting | Mechanism removed | val@100 SR (strict) | val@100 SR (lenient ≥0.9) | Reward mean | Notes |
|---|---|---|---|---|---|---|
| A2 | 1.5B-AF | -DR3 | XX.X% | XX.X% | 0.XXX | (e.g., "stable, no collapse") |
```

**Reference DUET\* (full)**:
- 1.5B-AF = 47.5%
- 1.5B-WS = 36.0%
- 3B-AF  = 77.5%
- 3B-WS  = 45.5%

**Δ interpretation**:
- Drop **<2pp**: weak/no contribution at this scale (discuss in §Limitations or §Discussion)
- Drop **2-5pp**: modest contribution
- Drop **>5pp**: clear positive contribution (the strong story)
- **>15pp drop or full collapse**: mechanism is *load-bearing* at this scale (best for narrative)

---

## 7. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Run crashes mid-training (OOM, deadlock) | Restart immediately; if 2nd attempt also fails, skip and note in `ablation_results.md` |
| -DR3 run on 3B explodes (loss diverges without correction) | This is itself a **finding**: report the divergence point as evidence Bias 2 is real and quantitatively significant |
| L20X queue overruns 48h | Drop tail cells per §2 fallback rule. **Never drop a -DR3 cell.** |
| WS single-seed noise gives unintuitive ablation result (e.g. -SC > full) | Mark with † in table; add caveat sentence; do not retract main claim |
| -BC config knob doesn't propagate (μ stays nonzero) | Verify by `chord/mu` wandb metric being exactly 0 throughout; if not, switch to fallback recipe in §5 |
| Q: should we also turn off `chord_mu_adaptive` when ablating -BC? | Yes if using the fallback recipe (set `chord_mu_adaptive: false` to lock μ=0) |

---

## 8. L20X Handoff Note (paste this into your message to L20X)

> Hi — for the NeurIPS 2026 ablation table, please run the following 8 experiments on 4×L20X-144GB by **2026-05-06 23:59** local. Each is a single config knob flip on top of your 3B DUET\* SOTA configs (the v39b configs that produced 77.5% on AF and 45.5% on WS).
>
> **Priority order (kill tail if overrunning — never drop any -DR3 cell)**:
>
> 1. **L2: 3B-AF -DR3** — `actor.use_dr3: false`, `actor.dr3.enable: false`. Save as `alfworld_qwen3b_duet_minus_dr3`.
> 2. **L6: 3B-WS -DR3** — same flip on 3B-WS SOTA. Save as `webshop_qwen3b_duet_minus_dr3`.
> 3. **L1: 3B-AF -baseline_sep** — `algorithm.grpo.teacher_baseline_separation.enable: false`. Save as `alfworld_qwen3b_duet_minus_baseline_sep`.
> 4. **L5: 3B-WS -baseline_sep** — same. Save as `webshop_qwen3b_duet_minus_baseline_sep`.
> 5. **L3: 3B-AF -BC** — `actor.use_chord: false`. Save as `alfworld_qwen3b_duet_minus_bc`.
> 6. **L7: 3B-WS -BC** — same. Save as `webshop_qwen3b_duet_minus_bc`.
> 7. **L4: 3B-AF -SC** — `exp_manager.state_channel.enable: false`. Save as `alfworld_qwen3b_duet_minus_sc`.
> 8. **L8: 3B-WS -SC** — same. Save as `webshop_qwen3b_duet_minus_sc`.
>
> When each run finishes, please send back: (a) val@100 strict success rate (score ≥ 1.0), (b) val@100 lenient (score ≥ 0.9) success rate, (c) reward_mean, (d) any anomalies (collapse, divergence, etc).
>
> Reference DUET\* numbers: 3B-AF = 77.5%, 3B-WS = 45.5%. Expected direction is **success rate drops** — bigger drop = stronger evidence for that mechanism.
>
> Full plan doc lives at `NeurIPS_2026_Latex/ABLATION_PLAN.md` — config recipes (§5), reporting template (§6), risk handling (§7) all there.

---

## 9. Open uncertainties (verify before launch)

- **DR3 disable knob**: `actor.use_dr3` vs `actor.dr3.enable` — both exist in SOTA config; setting both to `false` is safe, but only one is functionally load-bearing. Verify in `agentevolver/module/exp_manager/het_actor.py` and `dr3_ratio.py` if in doubt.
- **3B per-run wall-clock on L20X**: 8–10h is an extrapolation from the 1.5B 5h. Collaborator should confirm L1's actual time after step 100 — if >10h, kill tail cells per §2 rule.
- **1.5B-WS SOTA config exact filename** (used for A5–A8): `webshop_qwen1.5b_duet_swC_02_*.yaml` family. Pick the one matching `peak=0.3, valley=0.10, d_floor=0.6, ema_alpha=0.2, token_weighting=false` per `data/raw_data.md` (the 36.0% cell).
- **3B SOTA configs on L20X**: collaborator's local copy may differ slightly from `alfworld_qwen3b_duet_v39b.yaml` / 3B WS equivalent; trust whatever produced 77.5% / 45.5% on their server.
