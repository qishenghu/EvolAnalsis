"""Mechanism-level health monitor for a live rebuttal run.

Compares a running experiment against a reference run at MATCHED training steps, on the signals
that say whether the algorithm is working — not on the score. The distinction matters: killing a
run because its number looks low, and restarting until a number looks good, is selection bias and
would invalidate exactly the reproducibility claim we are trying to make. So the kill criteria
below are all mechanism faults with a known cause, never "behind the reference".

Kill-worthy (FAULT):
  - NaN/inf in loss or gradient norm
  - entropy collapse: policy entropy < 20% of the reference's at the same step for 10+ steps
  - KL blow-up: actor/kl_loss > 5x the reference's running max
  - format collapse: degenerate repetition in > 15% of rollouts (reference: ~1%)
  - dead teacher channel: dr3/disc_acc pinned at 0.5 (discriminator learning nothing) past step 20,
    or teacher_gradient_share stuck at 0 while the teacher mix ratio is non-zero
  - infrastructure: env errors, worker deaths, engine restarts in the run log

Watch-only (SLOW): behind the reference on reward/success but with every mechanism signal healthy.
This is NOT a kill condition — it is the measurement we are trying to make.

Usage:
  PYTHONPATH=. python scripts/monitor_run_health.py --run <name> --reference <name> --env webshop
  PYTHONPATH=. python scripts/monitor_run_health.py --run <name> --reference <name> --watch 900
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

METRICS = [
    "critic/success_onpolicy/mean",
    "actor/entropy_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "dr3/disc_acc",
    "dr3/w_off_mean",
    "duet/teacher_gradient_share",
    "chord/mu",
    "diag/teacher_sample_ratio",
    "state_channel/coverage_mean",
    "response_length/mean",
]


def series(log, key, limit=None):
    """Pull a metric's per-step series out of a (possibly multi-GB) training log."""
    if not os.path.exists(log):
        return []
    try:
        out = subprocess.run(["grep", "-oa", f"{key}:[0-9.eE+-]*", log],
                             capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return []
    vals = []
    for tok in out.split():
        try:
            vals.append(float(tok.split(":", 1)[1]))
        except Exception:
            pass
    return vals[:limit] if limit else vals


def malformed_action_rate(run, start_ts=None):
    """Fraction of recent rollouts containing an action that is not search[...] or click[...].

    A real, catchable failure mode: on 1.5B WebShop the policy can regress into emitting a bare
    `[asin]` before the correct `click[asin]`. Task competence looks intact (options still clicked,
    purchases still completed) but each malformed action costs the margin that separates a 0.95
    score from 1.0, so strict success collapses to zero while mean reward keeps rising. Measured
    across 68 historical 1.5B WebShop runs: those above 30% malformed average 0.6% strict success,
    those below 10% average 5.9%.
    """
    files = sorted(glob.glob(f"checkpoints/agentevolver/{run}/Trajectory/trajectories_step_*.jsonl"),
                   key=lambda f: int(f.rsplit("step_", 1)[1].split(".")[0]))
    if start_ts:
        files = [f for f in files if os.path.getmtime(f) >= start_ts - 30]
    if not files:
        return None, 0
    pat = re.compile(r"<action>\\n?(.*?)\\n?</action>")
    bad = n = 0
    for f in files[-5:]:
        for line in open(f):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if (x.get("diag") or {}).get("is_teacher"):
                continue
            acts = [a.strip().strip('"') for a in pat.findall(json.dumps(x.get("messages", "")))]
            if not acts:
                continue
            n += 1
            if any(not (a.startswith("search[") or a.startswith("click[")) for a in acts):
                bad += 1
    return (bad / n if n else None), n


def repetition_rate(run):
    """Fraction of the most recent rollouts whose output degenerates into token repetition."""
    files = sorted(glob.glob(f"checkpoints/agentevolver/{run}/Trajectory/trajectories_step_*.jsonl"),
                   key=lambda f: int(f.rsplit("step_", 1)[1].split(".")[0]))
    if not files:
        return None, 0
    pat = re.compile(r"\b(\w+(?: \w+){0,2})(?: \1){7,}")
    rep = n = 0
    for f in files[-5:]:
        for line in open(f):
            try:
                x = json.loads(line)
            except Exception:
                continue
            diag = x.get("diag") if isinstance(x.get("diag"), dict) else {}
            if diag.get("is_teacher"):
                continue
            n += 1
            if pat.search(json.dumps(x.get("messages", ""))[:20000]):
                rep += 1
    return (rep / n if n else None), n



def _expects_teacher_mixing(run: str) -> bool:
    """True if the run's EXECUTED config enables teacher experience replay.

    Falls back to True when the config cannot be read, so a genuinely broken teacher channel is
    still caught; the only thing this suppresses is the known-correct zero on baselines that do not
    mix by design (SFT->GRPO, on-policy GRPO).
    """
    path = f"launcher_record/{run}/yaml_backup.yaml"
    if not os.path.exists(path):
        return True
    try:
        import yaml
        cfg = yaml.safe_load(open(path)) or {}
        te = ((cfg.get("exp_manager") or {}).get("teacher_experience") or {})
        return bool(te.get("enable", False))
    except Exception:
        return True

def check(run, reference, env):
    faults, watches, lines = [], [], []
    log, ref_log = f"logs/{run}.log", f"logs/{reference}.log"

    # Count only rollouts written by the CURRENTLY running process. A killed-and-relaunched
    # experiment leaves its predecessor's files behind, and mixing them with a fresh log makes a
    # warming-up run look broken: 46 stale files once put the step count at 54 while the live
    # process was on step 6, so an in-warmup disc_acc read as a dead discriminator. The only
    # reliable boundary is the training process's own start time.
    traj = glob.glob(f"checkpoints/agentevolver/{run}/Trajectory/trajectories_step_*.jsonl")
    start_ts = None
    try:
        out = subprocess.run(["ps", "-eo", "pid,lstart,args"], capture_output=True, text=True,
                             timeout=30).stdout
        for ln in out.splitlines():
            if "agentevolver.main_ppo" in ln and run in ln:
                start_ts = time.mktime(time.strptime(" ".join(ln.split()[1:6]), "%a %b %d %H:%M:%S %Y"))
                break
    except Exception:
        pass
    if start_ts is None and os.path.exists(log):
        # not running (finished or not started): fall back to the log's own window
        start_ts = os.path.getmtime(log) - 86400
    fresh = [f for f in traj if start_ts is None or os.path.getmtime(f) >= start_ts - 30]
    steps, stale = len(fresh), len(traj) - len(fresh)
    lines.append(f"step {steps}" + (f" (+{stale} stale files from a previous instance, ignored)" if stale else ""))
    if steps == 0:
        return ["(no steps yet)"], [], ["still initialising"]

    cur = {k: series(log, k) for k in METRICS}
    ref = {k: series(ref_log, k, limit=steps) for k in METRICS}

    def last(k, n=10):
        v = cur.get(k) or []
        return sum(v[-n:]) / len(v[-n:]) if v else None

    def ref_last(k, n=10):
        v = ref.get(k) or []
        return sum(v[-n:]) / len(v[-n:]) if v else None

    # --- infrastructure ---
    # Fatal signatures kill the process, so any occurrence is a fault. A bare "Traceback" is NOT
    # fatal on its own: vLLM raises per-request errors (e.g. a single rollout whose prompt exceeds
    # the context window) that the rollout loop handles, and the batch stays intact. On 2026-07-28
    # this rule alone would have killed the Llama cross-family run at step 47 — every mechanism
    # signal healthy, on-policy success 0.287 — over ONE out-of-context rollout in ~3000. Require
    # either a fatal signature, or repeated tracebacks AND evidence that the batch is actually
    # losing trajectories.
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        for sig in ("EngineDeadError", "ActorDiedError", "CUDA out of memory"):
            if sig in txt:
                faults.append(f"infrastructure: '{sig}' in the run log")
        n_tb = txt.count("Traceback (most recent call last)")
        traj = cur.get("training/num_not_none_traj") or []
        short = [t for t in traj[-10:] if t < 64]
        if n_tb and short:
            faults.append(f"infrastructure: {n_tb} tracebacks AND the batch is losing trajectories "
                          f"({len(short)} of the last 10 steps below 64)")
        elif n_tb >= 20:
            watches.append(f"{n_tb} tracebacks in the log; batch size intact, so this is a "
                           f"per-request error rate, not a fault — check if it is climbing")

    # --- numerical ---
    for k in ("actor/grad_norm", "actor/kl_loss"):
        v = cur.get(k) or []
        if any(x != x or x in (float("inf"), float("-inf")) for x in v):
            faults.append(f"numerical: NaN/inf in {k}")

    # --- entropy collapse ---
    e, er = last("actor/entropy_loss"), ref_last("actor/entropy_loss")
    if e is not None and er and steps >= 15 and e < 0.2 * er:
        faults.append(f"entropy collapse: {e:.4f} vs reference {er:.4f} at step {steps}")
    if e is not None:
        lines.append(f"entropy {e:.4f}" + (f" (ref {er:.4f})" if er else ""))

    # --- KL drift ---
    # A run whose KL keeps climbing has drifted further from the reference model. Report it, but do
    # NOT claim it predicts the WebShop divergence: on 2026-07-27 the seed2025 replicate sat at KL
    # peaks of 1.72/1.50/1.42 — the band the diverged H200 replicate occupied — while healthy on
    # every mechanism signal (reward at run maximum, 1.4% malformed). KL elevation alone does not
    # separate the two. Likewise a pre-clip grad_norm of 9.59 is unremarkable: the paper run that
    # produced 35.5% exceeded it on 7 of 100 steps (p90 9.32, max 27.98). See
    # rebuttal/CORRECTION_for_H200_grad_spike.md. WATCH only — never a reason to alter a replicate.
    kl, klr = last("actor/kl_loss"), ref_last("actor/kl_loss")
    kl_all = cur.get("actor/kl_loss") or []
    if kl is not None:
        lines.append(f"kl_loss {kl:.4f}" + (f" (ref {klr:.4f})" if klr else ""))
        if klr and kl > 1.5 * klr and steps >= 40:
            watches.append(f"KL drift {kl:.3f} vs reference {klr:.3f} at step {steps} — informative "
                           "only; on its own this does NOT predict the WebShop divergence")
        if kl_all and max(kl_all) > 5.0:
            faults.append(f"KL blow-up: reached {max(kl_all):.2f}")

    # --- sustained-zero success ---
    # This IS the discriminative signature of the diverged replicate: 22 of its last 26 steps at
    # exactly 0.000, against 10/26 for the paper run and 9/26 for the healthy seed2025 replicate.
    # Reported as WATCH, not FAULT: a replicate's score is the measurement, never grounds to kill.
    # Step-aware: WebShop is near-uniformly zero early in training for EVERY method, so this check
    # is meaningless before the policy has had a chance to escape the partial-credit local optimum.
    # Measured on the paper cell itself (swC_02, which finishes at 35.5%): steps 11-36 are 17/26
    # zeros — indistinguishable from a run we would want to flag. Only apply late.
    # Compare against the reference over the SAME step window, not a fixed threshold. On WebShop the
    # escape from the partial-credit local optimum is a late event (onset ~step 74 in the paper
    # cell), so zeros are the norm before then and a fixed cut-off fires constantly. Measured on the
    # paper cell itself: steps 11-36 → 17/26 zeros, 43-68 → 15/26, 75-100 → 1/26. Only a run that is
    # much worse than the reference at the same point is worth noting.
    succ_all = cur.get("critic/success_onpolicy/mean") or []
    ref_all = ref.get("critic/success_onpolicy/mean") or [] if ref else []
    if len(succ_all) >= 26:
        window = succ_all[-26:]
        zeros = sum(1 for v in window if v == 0.0)
        ref_win = ref_all[max(0, len(succ_all) - 26):len(succ_all)]
        ref_zeros = sum(1 for v in ref_win if v == 0.0) if len(ref_win) >= 20 else None
        if ref_zeros is not None and zeros - ref_zeros >= 8:
            watches.append(f"sustained-zero success: {zeros}/26 recent steps exactly 0.000 against "
                           f"the reference's {ref_zeros}/26 over the same window — the signature "
                           f"that separated the diverged replicate; record it, do not act on it")
        elif ref_zeros is None and zeros >= 24 and steps >= 80:
            watches.append(f"sustained-zero success: {zeros}/26 recent steps exactly 0.000 "
                           f"(no reference window available for comparison)")

    # --- teacher channel alive ---
    acc = last("dr3/disc_acc")
    if acc is not None:
        lines.append(f"disc_acc {acc:.3f}")
        if steps >= 25 and abs(acc - 0.5) < 0.02:
            faults.append(f"dead discriminator: disc_acc pinned at {acc:.3f} past step {steps}")
    tgs, mix = last("duet/teacher_gradient_share"), last("diag/teacher_sample_ratio")
    if tgs is not None:
        lines.append(f"teacher_grad_share {tgs:.3f}")
        if steps >= 25 and tgs < 1e-6 and (mix or 0) > 0.01:
            faults.append("dead teacher channel: gradient share 0 while teacher samples are mixed in")
    if mix is not None:
        lines.append(f"teacher_mix {mix:.3f}")
        # Only a fault for runs that are SUPPOSED to mix. The SFT->GRPO baseline continues pure
        # on-policy RL from an SFT checkpoint, so teacher_sample_ratio 0.0 is its definition, not a
        # failure. Read the executed config rather than assuming.
        if steps >= 10 and mix < 0.01 and _expects_teacher_mixing(run):
            faults.append(f"teacher mixing off: teacher_sample_ratio {mix:.4f}")
    mu = last("chord/mu")
    if mu is not None:
        lines.append(f"mu {mu:.3f}")
    cov = last("state_channel/coverage_mean")
    if cov is not None:
        lines.append(f"sc_coverage {cov:.3f}")
        if steps >= 15 and cov < 0.05:
            faults.append(f"state channel starved: coverage {cov:.3f}")

    # --- malformed actions (WebShop-style format regression) ---
    if env == "webshop":
        mr, mn = malformed_action_rate(run, start_ts)
        if mr is not None:
            lines.append(f"malformed-action {mr:.1%} of {mn}")
            if mr > 0.25 and steps >= 30:
                faults.append(f"format regression: {mr:.1%} of recent rollouts emit a malformed "
                              f"action — strict success collapses to ~0 above this rate while mean "
                              f"reward keeps rising, so the score will look fine until validation")
            elif mr > 0.10 and steps >= 30:
                watches.append(f"malformed actions rising ({mr:.1%}); above ~25% this run's strict "
                               "success is unrecoverable")

    # --- format collapse (repetition) ---
    rr, nr = repetition_rate(run)
    if rr is not None:
        lines.append(f"repetition {rr:.1%} of {nr} recent rollouts")
        if rr > 0.15:
            faults.append(f"format collapse: {rr:.1%} of recent rollouts degenerate into repetition")

    # --- progress (watch only, never a kill reason) ---
    s, sr = last("critic/success_onpolicy/mean"), ref_last("critic/success_onpolicy/mean")
    if s is not None:
        lines.append(f"on-policy success {s:.3f}" + (f" (ref at same step {sr:.3f})" if sr else ""))
        if sr and s < 0.5 * sr and steps >= 30:
            watches.append(f"behind the reference at step {steps}: {s:.3f} vs {sr:.3f} — "
                           "mechanism signals are healthy, so this is the measurement, not a fault")
    return lines, faults, watches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--env", default="webshop")
    ap.add_argument("--watch", type=int, default=0, help="seconds between checks; 0 = once")
    args = ap.parse_args()
    while True:
        lines, faults, watches = check(args.run, args.reference, args.env)
        stamp = time.strftime("%m-%d %H:%M")
        print(f"[{stamp}] {args.run}: " + " | ".join(lines))
        for w in watches:
            print(f"    WATCH: {w}")
        for f in faults:
            print(f"    FAULT: {f}")
        print(f"    => {'FAULT — kill and diagnose' if faults else 'healthy'}", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
