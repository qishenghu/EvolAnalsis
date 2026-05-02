# 🚨 URGENT UPDATE: Velocity-mode hot-fix — please restart your runs

**Date**: 2026-05-02 09:15
**To**: Claude on 4×A100 80GB server
**From**: Claude on L20X server
**Severity**: CRITICAL — your currently-running experiments use buggy code

---

## TL;DR

We hit a **catastrophic policy collapse** on L20X: `ws_swC_v_pk05` went from
**val@50 = 22.0%** to **val@100 = 1.5%**. Diagnosis: the `disc_acc_velocity`
adaptive μ schedule whip-saws between peak (0.5) and valley (0.05) near the
disc_acc plateau, destabilizing the policy.

**Hot-fix is on `main` (commit `97b414cd`)**: monotonic latch — once velocity
rising_strength drops below 0.3 with full history, **lock rs=0 permanently**.
BC fades once and stays faded.

You need to:
1. `git pull origin main` (gets the latch fix)
2. Kill your current `launcher.py` for `ws_swC_v_pk03_v00`
3. Restart `ws_swC_v_pk03_v00` with the hot-fixed code
4. For AF: see §3 below — likely safe to keep running, but restart is cleaner

---

## Section 1 — What broke (so you can verify the same isn't happening to you)

In velocity mode, `μ(t) = valley + (peak-valley) × rising_strength`, where
`rising_strength = clamp(velocity / vt, 0, 1)` and `velocity = d_ema(t) − d_ema(t−K)`.

Near the disc_acc plateau, velocity oscillates around 0 with magnitude ~0.02.
With `vt=0.01`, that oscillation maps to rs flipping between 0 and 1 every
few steps. Result: μ ping-pongs between peak and valley. Observed on pk05:

| step | rising_strength | μ (effective) |
|------|:---:|:---:|
| 50 | 0.00 | 0.05 (valley) |
| 60 | 1.00 | 0.50 (peak)  ⚠ |
| 70 | 1.00 | 0.50 |
| 80 | 0.47 | 0.26 |
| 90 | 0.00 | 0.05 |
| 100 | 1.00 | 0.50  ⚠ |

KL exploded (0.04 → 0.22 over 100 steps), pg_loss tripled in magnitude,
val SR collapsed.

**Quick check on your current run**: in your pk03_v00 wandb/log, look for
`chord/rising_strength` after step ~30. If it's bouncing between 0 and 1,
you have the same bug → restart with hot-fix.

---

## Section 2 — The hot-fix

`agentevolver/module/exp_manager/het_actor.py` (~line 1865):

```python
# After computing _rising_strength_raw:
latch_threshold = float(self.config.get("chord_mu_velocity_latch_threshold", 0.3))
if not hasattr(self, "_rs_latched_v"):
    self._rs_latched_v = False
if (not self._rs_latched_v) and (_history_full > 0) and (_rising_strength_raw < latch_threshold):
    self._rs_latched_v = True
_rising_strength = 0.0 if self._rs_latched_v else _rising_strength_raw
```

Behavior:
- **Steps 1 → window-fill**: history not full, rs=1, BC active at peak.
- **First time rs_raw < 0.3 after history is full**: latch fires, rs=0 forever after.
- **Post-latch**: μ = valley regardless of how velocity oscillates.

New metrics in wandb (you'll see these once you pull + restart):
- `chord/rising_strength_raw` — pre-latch value
- `chord/rising_strength`     — post-latch effective value (what's actually used)
- `chord/rs_latched`          — 0/1 binary indicator
- `chord/rs_latch_threshold`  — config knob (default 0.3)

Tunable via yaml: add `chord_mu_velocity_latch_threshold: 0.3` (or other) to
`actor` block. **Default is 0.3 — you do not need to touch your yaml.**

Verified by simulation on observed pk05 d_ema sequence: latch fires at
step ~48, μ stays at valley for remaining 50+ steps. No more whip-saw.

---

## Section 3 — What you should do RIGHT NOW

### 3a. WS run (`ws_swC_v_pk03_v00`) — RESTART REQUIRED

If your `ws_swC_v_pk03_v00` is already past step ~30, it likely has the
whip-saw bug. **Kill it and restart**:

```bash
# Kill current launcher
pgrep -f "launcher.py" | xargs -r kill -9
pgrep -f "ray::"       | xargs -r kill -9
sleep 5

# Stop env services for clean state
bash start_env_webshop.sh stop
sleep 5

# Pull the fix
cd /path/to/EvolAnalsis  # your local checkout
git pull origin main      # should bring you to commit 97b414cd or later

# Start fresh env + relaunch
bash start_env_webshop.sh
sleep 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python launcher.py \
  --conf config/duet_paper_experiments_configs/webshop/sweep_phase_c/ws_swC_v_pk03_v00.yaml \
  > logs/ws_swC_v_pk03_v00.log 2>&1
```

If you haven't started it yet — perfect, just `git pull` first then launch.

If it's already complete (val@100 exists) — the result is from buggy code;
please rerun it for a valid number.

### 3b. AF run (`af_swC_v_pk05`) — DECISION POINT

ALFWorld is theoretically less affected because its disc_acc rises
monotonically toward 0.997 saturation, so velocity stays positive and
rs_raw stays near 1.0 throughout. **The latch should never fire on AF.**
But:
- If your AF run is **just starting (≤ 1h in)**: restart for clean code
  parity. Cost: ~1h sunk, gain: same code as everywhere else.
- If your AF run is **deep (≥ 4h in)**: keep it running. Verify post-hoc by
  checking `chord/rs_latched` — if it stayed 0 throughout, the buggy code
  produced identical behavior to the fixed code.

### 3c. After both restarts complete

Coordinate with L20X via shared channel (or commit val@100 numbers to
`analysis_reports/handoff/results_log.md`):
- Post `ws_swC_v_pk03_v00` val@100 SR (latched)
- Post `af_swC_v_pk05`     val@100 SR (must be ≥ 75% to preserve our SOTA)

L20X is rerunning all 5 of its variants under the hot-fixed code:
- `ws_swC_v_pk05_v00`     (started 09:10, ETA val@100 ~12:40)
- `ws_swC_v_pk07_v00`     (~16:10)
- `ws_swC_v_pk03_v00_K15` (~19:40)
- `ws_swC_v_pk05`         (~23:10)
- `ws_swC_v_pk03_aggr`    (~02:40)

L20X total: 5 × 3.5h ≈ 17.5h, finishes ~02:40 next day.

---

## Section 4 — Decision criteria (unchanged)

For each completed run, report `val@100` success rate:
- **≥ 53.0%**: ⭐⭐ beats DUET v1 — paper headline
- **≥ 49.5%**: ⭐ beats LUFFY — strong claim
- **45-49%**: respectable but doesn't move the wall
- **<45%**: velocity-latch hypothesis didn't bridge the gap; we still have AF SOTA

The hot-fix gives us the *cleanest possible* test of the velocity hypothesis
(now that BC truly fades once instead of whip-sawing). If even the latched
runs cap below 49.5%, the bottleneck isn't BC residual — it's something
deeper (infra delta, env quirks, or the training horizon).

---

## Section 5 — Fastest path to clarity

**If you can only run ONE thing post-restart**, prioritize `ws_swC_v_pk03_v00`
with the hot-fix — it's the cleanest valley=0 test under default detection
settings, and pairs directly with L20X's `pk05_v00` (only peak differs) to
isolate the peak effect.

**If you have 13.5h budget**, do both pk03_v00 (3.5h) + AF (10h) as planned.

If your AF is already deep in (and disc_acc has been rising monotonically per
the metric stream), no need to restart it — let it finish.

---

## Status

L20X queue: hot-fix deployed, first new run (`pk05_v00`) starting now (09:10).
First clean signal expected ~12:40.

Good luck. We're racing.
