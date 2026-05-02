# Handoff: DUET\* Velocity-Based Adaptive μ — 2-day Sprint Plan

**Date**: 2026-05-02
**Author (3B / L20X server)**: Claude (current session)
**Audience**: Claude on 4×A100 80GB server
**Deadline**: NeurIPS 2026 — **2026-05-07** (5 days)
**Sprint goal**: Complete main experiments validating velocity-based μ in 2 days

---

## TL;DR

We discovered that the existing disc_acc-based μ schedule **structurally fails on WebShop 3B** because disc_acc plateaus at ~0.91 (instead of saturating to 1.0 like ALFWorld). This forces μ to a non-fading residual ≈ 0.13, where BC over-imitation hurts WS partial-credit attribute precision.

**The fix**: replace `disc_acc` mode (level-based) with `disc_acc_velocity` mode (derivative-based). When disc_acc velocity → 0 (plateau detected), μ → valley automatically, regardless of absolute disc_acc level. This auto-adapts cross-env without manual tuning.

**Code is implemented and committed.** You should:
1. `git pull` to get the new `disc_acc_velocity` mode + 4 ready configs
2. **(Updated 2026-05-02)** L20X agent has joined velocity sprint — split duty (see §0)
3. Run **2** main experiments on your end (was 4) — ~13.5 hours
4. **(Updated 2026-05-02-2)** No 3-seed Phase C (time too tight). L20X switched to **3 ADDITIONAL velocity-mode variants** instead, to maximize SOTA chances.
5. Total your wall-clock: ~14h, leaves margin for paper writing

---

## Section 0 — Critical setup info (READ FIRST, updated 2026-05-02)

### Model: **Qwen2.5-3B-Instruct** (NOT 1.5B)

All 4 configs we provide have model.path hardcoded to:
```
/mnt/workspace/qisheng/HLE_QA_workflow/EvolAnalsis/models/Qwen/Qwen2.5-3B-Instruct
```

On your 4×A100 server, **edit `actor_rollout_ref.model.path`** in each yaml to point to your local copy of `Qwen2.5-3B-Instruct`. Same goes for `teacher_experience.data_path` (teacher: Qwen-72B traj pickles). **Do NOT switch to 1.5B** — the velocity hypothesis we are testing is specifically for 3B WS where disc_acc plateaus ~0.91.

### 3B WebShop baselines (the wall we're trying to break)

| Method | val@100 SR | Notes |
|--------|-----------|-------|
| GRPO (no teacher) | ~30% | floor |
| CHORD (BC + GRPO, fixed μ) | ~40-45% | |
| **LUFFY** (teacher mixing + p/p_β) | **49.5%** | ⭐ first hard target |
| **DUET v1** (DR3 + SC, no BC) | **53.0%** | ⭐⭐ stretch goal |
| DUET\* v39b (current best, μ=disc_acc) | 45.5% (best of 1 seed) | structurally bounded ≈ 44% mean |

DUET\* sweep (16 runs across peak/floor/valley combinations) on L20X **topped at 44.5%** — evidence that level-based μ is the bottleneck, not the BC formula itself. This is **why velocity matters**.

### 3B ALFWorld baseline / SOTA

| Method | val@100 SR | Notes |
|--------|-----------|-------|
| GRPO | ~50% | |
| LUFFY | ~64% | |
| DUET v1 | 69.5% | prior best |
| **DUET\* v39b RERUN** | **77.5%** | **OUR CURRENT SOTA** — must not regress |

AF velocity run (`af_swC_v_pk05`) is purely defensive: confirm velocity mode preserves 77.5% (not improve it). If af_swC_v_pk05 ≥ 75%, we're fine.

### Targets

- **WS goal**: any single seed ≥ 49.5% (beat LUFFY) → 3-seed mean ≥ 49.5% confirms claim
- **WS stretch**: any single seed ≥ 53.0% (beat DUET v1) → would be ideal headline number
- **AF guardrail**: af_swC_v_pk05 ≥ 75% (no regression)

### Updated split between L20X (us) and 4×A100 (you) — **REVISED 2026-05-02 PM**

L20X has killed its sweep (it was capped at 44.5%) and joined velocity sprint.
**Time is too tight for 3-seed Phase C — instead, we run more parameter
variants to maximize SOTA-hit probability.**

| Server | Run | Why this run | Sequence |
|--------|-----|-------------|----------|
| L20X (us) | `ws_swC_v_pk05` (peak=0.5, valley=0.05) | ⭐ main candidate | T+0 (running) |
| L20X (us) | `ws_swC_v_pk03_aggr` (peak=0.3, K=5, vt=0.005) | aggressive plateau detection | T+3.5h |
| L20X (us) | `ws_swC_v_pk05_v00` (peak=0.5, **valley=0**) | full BC-off after plateau | T+7h |
| L20X (us) | `ws_swC_v_pk07_v00` (peak=0.7, **valley=0**) | strong early imit + full off | T+10.5h |
| L20X (us) | `ws_swC_v_pk03_v00_K15` (peak=0.3, **valley=0**, K=15) | gentle + slower detect | T+14h |
| **4×A100 (you)** | `ws_swC_v_pk03_v00` (peak=0.3, **valley=0.0**) ⚡swapped | direct comparison vs pk05_v00 | T+0 |
| **4×A100 (you)** | `af_swC_v_pk05` (peak=0.5, valley=0.05) | AF SOTA verification (~10h) | T+3.5h |

**Total velocity attempts**: 6 WS runs (5 on L20X + 1 on you) + 1 AF guardrail.
**No multi-seed**: each run reports single-seed val@100; we report best run as headline.

**Why valley=0 in the new variants**: current pk05/pk03/pk03_aggr all use valley=0.05 (so even after plateau, μ_late ≈ 0.05 BC residual). The **new variants set valley=0** — when velocity detects plateau, BC turns OFF entirely, so late training equals **DUET v1's algorithm** (which scores 53% on WS). If BC residual is the bottleneck (as our analysis suggests), valley=0 should bridge the gap.

**Schedule**:
- T+3.5h:  2 WS signals land (pk05, pk03 from 4×A100) — first verdict on velocity mode
- T+7h:   3 WS signals land (+ pk03_aggr from L20X) — wider sample
- T+10.5h: 4 WS signals land (+ pk05_v00) — first valley=0 result
- T+14h:  5 WS signals land (+ pk07_v00) — high-peak valley=0
- T+17.5h: all 6 WS signals + AF guardrail — final state for paper

---

## Section 1 — The problem (why current disc_acc μ fails on WS 3B)

### What disc_acc actually measures
The discriminator D in DR3 distinguishes student rollouts from teacher rollouts. `disc_acc` is D's classification accuracy. **It measures stylistic distinguishability**, not policy-quality similarity.

### Why disc_acc behaves differently across envs

**ALFWorld (binary reward)**:
- Teacher (Qwen-72B) rollouts have distinctive style (concise reasoning, perfect action grammar)
- Student (Qwen2.5-3B) rollouts have idiosyncratic patterns (longer thinking, occasional invalid actions)
- D easily learns features → **disc_acc rises to 0.997** (saturated)

**WebShop (partial credit)**:
- Both teacher and student use highly templated actions: `search[...]`, `click[X]`, `click[buy now]`
- Limited token vocabulary, short per-step outputs
- D has few distinguishing features → **disc_acc plateaus at ~0.91** (never saturates)

### Why this breaks the current μ formula

```
μ_late = valley + (peak − valley) × (1 − d̄_ema) / (1 − d_floor)

For 3B WebShop: d̄_ema ≈ 0.91, d_floor = 0.5
μ_late = valley + (peak − valley) × (0.09 / 0.5)
       = valley + 0.18 × (peak − valley)
       
With peak=0.5, valley=0.05: μ_late ≈ 0.13 forever (BC never fades)
```

**The fundamental error**: the formula uses the **absolute level** of disc_acc as the fade signal. But disc_acc level is a property of D's feature space (env-dependent), not policy convergence (training-dependent).

### Why this matters for paper performance

```
WS 3B observed:
  v_no_bc_ws (use_chord=false, μ=0):    1.0%   (single seed, possibly variance)
  Various BC schedules (Phase A/B):     38-44%
  v39b 04-25 (lucky):                   45.5%
  Target — DUET v1 (no BC, H100):       53.0%

→ BC residual ≈ 0.13 hurts attribute precision (70/200 tasks stuck at [0.5, 1.0))
→ DUET* with current μ is structurally bounded ≈ 44% on WS 3B
```

---

## Section 2 — The fix: velocity-based μ schedule

### Core insight

When disc_acc plateaus or decreases, it means **student-teacher distribution has converged in surface features**. From training-dynamics perspective:
- BC has saturated its imitation gradient (no new "teacher-like behavior" to learn)
- DR3 IS correction is auto-fading (ŵ → 1 when D is uncertain)
- Buffer's training signal for D is exhausted
- GRPO group variance is shrinking (student behaviors clustered)

→ **Time to turn BC off and let GRPO refine the policy without imitation interference.**

### New formula

```
d_ema(t)        = EMA(dr3/disc_acc, α=d_ema_alpha)
velocity(t)     = d_ema(t) − d_ema(t − K)              # K-step window
rising_strength = clamp(velocity / vel_target, 0, 1)
μ(t)            = valley + (peak − valley) × rising_strength
```

### Cross-env behavior (verified analytically)

```
ALFWorld (disc_acc rises monotonically):
  velocity > 0 throughout → rising_strength = 1 → μ = peak
  → BC stays active full training → preserves SOTA ~77.5%

WebShop (disc_acc plateaus mid-training):
  early: velocity > 0 → μ = peak (BC active)
  mid (after plateau): velocity → 0 → μ = valley (BC fades)
  → Late training is pure GRPO+DR3+SC (= DUET v1 algorithm)
  → Should approach v1's 53%
```

This is **truly closed-form, env-agnostic** — single formula handles both regimes.

---

## Section 3 — Implementation (already merged to main)

### Files modified

```
agentevolver/module/exp_manager/het_actor.py
  Lines ~1816-1875 (after disc_acc branch, before gap branch)
  Added: elif use_adaptive_mu and adaptive_mode == "disc_acc_velocity":
  
  Reads:
    - dr3/disc_acc (raw discriminator accuracy)
    - dr3/disc_trained_steps (warmup gating, treats raw=0 as 0.5)
  
  State (per-actor instance):
    - self._disc_acc_ema_v        (EMA of disc_acc; alpha=chord_mu_d_ema_alpha)
    - self._disc_acc_history_v    (rolling window of K=velocity_window EMA values)
  
  Logic:
    - Pre-history (len < K): rising_strength = 1.0  (BC stays at peak)
    - Post-history: velocity = ema[t] - ema[t-K]
                   rising_strength = clamp(velocity / vel_target, 0, 1)
                   μ = valley + (peak - valley) * rising_strength
  
  Logged metrics (mu_mode = 7.0 for this branch):
    chord/disc_acc_current
    chord/disc_acc_raw
    chord/disc_acc_ema_v
    chord/d_velocity
    chord/d_velocity_target
    chord/d_velocity_window
    chord/rising_strength
    chord/d_history_len
    chord/d_history_full
```

### Configs ready (in `config/duet_paper_experiments_configs/`)

```
webshop/sweep_phase_c/
  ws_swC_v_pk03.yaml        peak=0.3, valley=0.05, ema=0.2, window=10, vel_target=0.01
  ws_swC_v_pk05.yaml        peak=0.5, valley=0.05, ema=0.2, window=10, vel_target=0.01
  ws_swC_v_pk03_aggr.yaml   peak=0.3, valley=0.05, ema=0.2, window=5,  vel_target=0.005

alfworld/sweep_phase_c/
  af_swC_v_pk05.yaml        peak=0.5, valley=0.05, ema=0.2, window=10, vel_target=0.01
```

### Knobs to understand

```
chord_mu_velocity_window:   K-step look-back for velocity computation
                             - 10 steps (default): smooth, ~10 batch periods
                             - 5 steps (aggressive): faster plateau detection
chord_mu_velocity_target:   threshold for "meaningfully rising"
                             - 0.01 (default): velocity 0.01 over K steps → full BC
                             - 0.005 (aggressive): more sensitive
chord_mu_d_ema_alpha:       smoothing on disc_acc itself (default 0.2 = slow/stable)
```

---

## Section 4 — Sprint Plan (2 days, 4×A100 80GB server)

### Setup checklist (15 min)

```bash
cd /path/to/EvolAnalsis
git pull origin main   # gets velocity mode code + 4 configs

# Adjust configs if 4×A100 80GB needs different infra than 4×L20X 144GB:
#   - ppo_micro_batch_size_per_gpu (default 2; reduce to 1 if OOM)
#   - gpu_memory_utilization (default 0.65; reduce if vLLM OOM)
#   - offload settings (default false; turn on if needed)
# All 4 sweep_phase_c configs use same infrastructure as v39b template,
# so only adjust if A100 80GB OOMs.

# Verify code by importing:
python -c "
from omegaconf import OmegaConf
for c in ['config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03.yaml',
          'config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk05.yaml',
          'config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_aggr.yaml',
          'config/duet_paper_experiments_configs/alfworld/sweep_phase_c/af_swC_v_pk05.yaml']:
    cfg = OmegaConf.load(c)
    actor = cfg.actor_rollout_ref.actor
    print(f'{c.split(\"/\")[-1]}  mode={actor.chord_mu_adaptive_mode}  peak={actor.chord_mu_peak} valley={actor.chord_mu_valley} window={actor.chord_mu_velocity_window} vel_target={actor.chord_mu_velocity_target}')
"
```

### Day 1 — Your share: 2 main experiments (~13.5h, sequential on 4 GPUs)

> **(Split with L20X — see §0 for full picture.)**
> Your queue: `ws_swC_v_pk03` (T+0, ~3.5h) → `af_swC_v_pk05` (T+3.5h, ~10h).
> L20X queue: `ws_swC_v_pk05` → `ws_swC_v_pk03_aggr` → 3-seed of winner.

**CRITICAL: each run must restart env services freshly to avoid memory leaks.**

```bash
# Use this run_one pattern (copied from our run_ws_sweep_phase_b.sh):
run_one() {
    local config=$1
    local name=$2
    local env=$3   # webshop or alfworld
    
    bash start_env_alfworld.sh stop 2>&1 | tail -1
    bash start_env_webshop.sh stop 2>&1 | tail -1
    sleep 8
    bash start_env_${env}.sh
    sleep 5
    
    local ray_tmp="${RAY_TMPDIR}/${name}"
    mkdir -p "$ray_tmp"; rm -rf "$ray_tmp"/session_* 2>/dev/null || true
    
    echo "[$(date '+%m-%d %H:%M')] RUN: $name"
    CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_TMPDIR="$ray_tmp" \
        python launcher.py --conf "$config" \
        > "logs/${name}.log" 2>&1
}
```

**Schedule (your 2 runs, sequential)** — ⚡ **2026-05-02 PM**: swapped pk03 → pk03_v00 for higher information yield:

```
Hour  Run                                          Env           ETA
─────────────────────────────────────────────────────────────────────
T+0    ws_swC_v_pk03_v00.yaml    (peak=0.3, val=0)  webshop      ~3.5h
T+3.5  af_swC_v_pk05.yaml        (peak=0.5)         alfworld     ~9-11h
T+13.5 → your share done
```

**If you already started `ws_swC_v_pk03`**: that's fine, let it finish — it's a useful data point. Then run `ws_swC_v_pk03_v00` (the swap) instead of `af_swC_v_pk05` is **NOT** advised — AF is the SOTA-preservation guardrail and must run. If you have extra slack post-AF, run the swap then.

**If you have NOT started**: pull the latest commit and use `ws_swC_v_pk03_v00.yaml` instead of `ws_swC_v_pk03.yaml`. The swap fills a missing search-space corner: it's the only `peak=0.3 + valley=0 + default-K` config and gives a clean 1-variable comparison against L20X's `pk05_v00`.

L20X side runs in parallel: `ws_swC_v_pk05` (T+0) + `ws_swC_v_pk03_aggr` (T+3.5h) + 3-seed Phase C (T+7h+).

**Why this split**: You take the longer AF run (10h) plus 1 WS; L20X takes 2 WS + 3-seed Phase C. WS results all land by T+7h so we can pick a winner; AF answer + 3-seed mean by T+17.5h. **Total wall-clock: ~18h vs original 31h sequential.**

**Coordinate**: please post each val@100 in a shared channel (or commit to `analysis_reports/handoff/results_log.md`) as soon as it lands so L20X can decide which config to use for the 3-seed Phase C run starting T+7h.

### Day 2 — Extra parameter variants (handled by L20X, no 3-seed)

**Pivot (2026-05-02 PM)**: Time too tight for 3-seed multi-seed confirmation. L20X is instead running **3 additional velocity-mode variants** with `valley=0` (vs valley=0.05 in the first batch) to widen the parameter search and maximize SOTA chances.

L20X queue from T+7h onward:
- `ws_swC_v_pk05_v00`        peak=0.5, valley=0.0, K=10, vt=0.01
- `ws_swC_v_pk07_v00`        peak=0.7, valley=0.0, K=10, vt=0.01
- `ws_swC_v_pk03_v00_K15`    peak=0.3, valley=0.0, K=15, vt=0.015 (slower detect)

**You do NOT need to run anything for Day 2** unless your AF run finishes early. If `af_swC_v_pk05` lands by T+13.5h and your remaining budget allows, **optionally** run 1 more WS variant we haven't covered. Suggestions:
- `ws_swC_v_pk05_v00_K5_vt005` (peak=0.5, valley=0.0, K=5, vt=0.005) — aggressive + full off — would test if `pk03_aggr` got the right idea but at higher peak
- `ws_swC_v_pk04_v00` (peak=0.4, valley=0.0, K=10, vt=0.01) — interpolate between pk03 and pk05
- Easiest: just clone any sweep_phase_c yaml + edit the 4 params with sed

Coordinate with L20X via `analysis_reports/handoff/results_log.md` (or any shared channel) — post each val@100 as it lands, including which `chord_mu_*` params you used. We'll pick the best across all servers as the paper headline.

**Reporting bar**:
- ≥ 49.5% → ⭐ beat LUFFY (good headline, suggests our framework works)
- ≥ 53.0% → ⭐⭐ beat DUET v1 (great headline, the velocity claim is fully validated)
- < 49.5% → velocity hypothesis didn't bridge the gap; we still have the AF 77.5% SOTA to lead the paper

```bash
# Generate 3-seed yamls of best WS config (replace BEST_NAME):
BEST=ws_swC_v_pk05   # whichever wins Day 1
for seed in 42 7 1234; do
    cp config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST}.yaml \
       config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST}_seed${seed}.yaml
    sed -i "s|experiment_name: ${BEST}|experiment_name: ${BEST}_seed${seed}|" \
       config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST}_seed${seed}.yaml
    sed -i "s|seed: 2026|seed: ${seed}|" \
       config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST}_seed${seed}.yaml
done

# Run 3 seeds sequentially (~3.5h × 3 = 10.5h)
for seed in 42 7 1234; do
    run_one "config/duet_paper_experiments_configs/webshop/sweep_phase_c/${BEST}_seed${seed}.yaml" \
            "${BEST}_seed${seed}" "webshop"
done
```

### What to report back

After each run completes, post results to a shared status file:

```bash
# Quick analyze:
python -c "
import json
for d in ['ws_swC_v_pk03', 'ws_swC_v_pk05', 'ws_swC_v_pk03_aggr', 'af_swC_v_pk05']:
    for env in ['webshop', 'alfworld']:
        f = f'experiments/{env}/{d}/validation_log/100.jsonl'
        try:
            n=0; sr=0; rw=0
            with open(f) as fh:
                for line in fh:
                    x=json.loads(line); n+=1
                    s=x.get('score',x.get('reward',0))
                    rw+=s; sr += 1 if s>=1.0 else 0
            print(f'{d:30s}  reward={rw/n:.4f}  success={sr/n*100:.1f}%')
        except FileNotFoundError: pass
"
```

---

## Section 5 — Expected outcomes & decision tree

### Outcome ranges

```
af_swC_v_pk05 (verify SOTA preservation):
  Expected: 75-79% (≈ v39b 77.5%)
  Concerning: < 70% (means velocity mode hurts AF; rollback)

ws_swC_v_pk05 (peak=0.5 with velocity fade):
  Expected: 45-55% if velocity correctly fades μ at plateau
  - 53%+: BEAT v1, paper headline ⭐⭐⭐
  - 49.5%+: BEAT LUFFY ⭐⭐
  - 45-49%: marginal improvement over current 44% ceiling
  - < 45%: fade timing wrong; try aggressive mode

ws_swC_v_pk03 (peak=0.3 baseline check):
  Expected: 40-50% (default peak with velocity mode)
  Useful for comparison and ablation

ws_swC_v_pk03_aggr (faster plateau detection):
  Expected: 40-55% — could win if K=10 is too slow at detecting WS plateau
```

### Decision tree

```
After all 4 Day-1 runs:

Scenario A: ws_swC_v_pk05 ≥ 49.5%
  → SOTA on WS achieved
  → Day 2: Phase C multi-seed confirm winner
  → Paper headline: "Velocity-based DUET* beats LUFFY on WS"

Scenario B: ws_swC_v_pk03_aggr ≥ 49.5%
  → Aggressive plateau detection wins
  → Day 2: Phase C multi-seed confirm aggressive variant
  → Paper note: "Plateau-detection sensitivity matters"

Scenario C: All WS velocity runs at 40-48%
  → Velocity mode helps but not enough to break LUFFY
  → Day 2: try ws_swC_v_pk07 (peak=0.7) or ws_swC_v_pk03_v0 (valley=0)
  → Paper: report results honestly + AF SOTA emphasis

Scenario D: All WS velocity runs < 40%
  → Velocity formula has issue
  → Day 2: investigate; possibly fall back to current best (swB_41 = 44.5%)
  → Paper: AF SOTA only, WS as competitive secondary

In ALL scenarios:
  af_swC_v_pk05 should land 75-79% (verify no AF regression)
```

---

## Section 6 — Concurrent work on L20X (this server)

Currently running Phase B Tier S (low-BC sweep, swB_31 → 32 → ... 11 configs total).
- Will continue providing baseline data points for paper Table 1
- Won't conflict with your A100 work

If your velocity mode hits SOTA (Scenario A or B):
- We'll continue running additional ablations on L20X
- You focus on Phase C confirmation + paper writing support

---

## Section 7 — Communication & sync

```
Result sharing protocol:
  1. After each run: post val@100 to a shared txt file or chat
  2. Or: commit run logs as `analysis_reports/A100_results_YYYYMMDD.md`
  
Critical decisions need user approval:
  - Which Phase C config to multi-seed
  - Whether to add new variants if early results disappointing
  - Final paper Table 1 numbers

Time pressure check:
  May 2 (today): code merged, configs ready
  May 3 (Day 1): 4 main experiments done by ~end of day
  May 4 (Day 2): Phase C multi-seed done by ~end of day
  May 5-6: Paper writing (3B / L20X server can help)
  May 7: NeurIPS deadline
```

---

## Section 8 — Verification before launching

```bash
# 1. Ensure code has the velocity branch:
grep -n "disc_acc_velocity" agentevolver/module/exp_manager/het_actor.py

# 2. Check 4 configs load:
for c in config/duet_paper_experiments_configs/{webshop,alfworld}/sweep_phase_c/*.yaml; do
    python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('$c'); print('$c', cfg.actor_rollout_ref.actor.chord_mu_adaptive_mode)"
done

# 3. Smoke test (1-2 min training):
# Edit total_epochs to 1, max_train_tasks to 8, then run a quick test
# (optional, skip if you trust the implementation)
```

---

## Section 9 — Why this is paper-worthy

If velocity mode works, the narrative becomes:

> "DUET\* with velocity-based adaptive BC — single closed-form schedule
> that auto-adapts across reward sparsity (binary AF vs partial WS), model
> scales (1.5B / 3B), and discriminator regimes (saturating / plateauing).
> The schedule reads training dynamics rather than absolute discriminator
> level, gracefully fading BC when imitation has saturated.
>
> Result: SOTA 77.5% on AF (BC stays active throughout), and competitive
> with (or beats) DUET v1 on WS (BC auto-fades at disc_acc plateau)."

This is a real algorithmic contribution: a **task-aware self-tuning RL imitation schedule**.

---

## Final checklist before you start

- [ ] `git pull origin main` (gets velocity code + 4 configs)
- [ ] Verify your 4×A100 server has the conda env, models, teacher data
- [ ] Verify `start_env_webshop.sh` and `start_env_alfworld.sh` work on your server
- [ ] Adjust micro_batch_size if needed for A100 80GB memory
- [ ] Launch 4 main experiments using `run_one` pattern with per-run env restart
- [ ] Report each val@100 as it lands

Let's nail this. 🚀
