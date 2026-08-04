"""Health-check a seed replicate: is it training on the curriculum we intended, and is it healthy?

A replicate is only interpretable if it differs from the reference run in run-time randomness
alone. `data.seed` in this codebase also selects which `max_train_tasks` of the split are used, so
replicate configs pin `data.task_seed`. This script verifies that the pin actually took effect in
the live run, and that nothing else looks wrong.

Usage:
  PYTHONPATH=. python scripts/check_replicate_health.py \
      --run webshop_qwen1.5b_duet_a100_seed2027 \
      --reference webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06
  # add --env alfworld for ALFWorld runs
"""
import argparse
import glob
import json
import os
import re
import statistics as stats

CKPT = "checkpoints/agentevolver"


def per_step_tasks(run):
    out = {}
    for f in glob.glob(f"{CKPT}/{run}/Trajectory/trajectories_step_*.jsonl"):
        step = int(f.rsplit("step_", 1)[1].split(".")[0])
        ids = set()
        for line in open(f):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("task_id") is not None:
                ids.add(str(x["task_id"]))
        if ids:
            out[step] = ids
    return out


def onpolicy_curve(run):
    """Per-step on-policy STRICT success from saved trajectories, teacher rollouts excluded.

    Two traps here, both of which produce numbers several times too high:
      - the row's `success` flag is not strict success. On WebShop it is true whenever the episode
        completed a purchase, so a row with `reward.outcome` 0.43 still has success=True. Strict
        success is `reward.outcome >= 1.0`, matching the val@100 metric.
      - teacher rollouts are flagged in `diag.is_teacher`, not `metadata.is_teacher`; missing them
        leaves ~1 in 8 rows that scored 1.0 by construction.
    """
    per = {}
    for f in glob.glob(f"{CKPT}/{run}/Trajectory/trajectories_step_*.jsonl"):
        step = int(f.rsplit("step_", 1)[1].split(".")[0])
        s = n = 0
        for line in open(f):
            try:
                x = json.loads(line)
            except Exception:
                continue
            diag = x.get("diag") if isinstance(x.get("diag"), dict) else {}
            md = x.get("metadata") if isinstance(x.get("metadata"), dict) else {}
            if diag.get("is_teacher") or md.get("is_teacher"):
                continue
            n += 1
            s += float((x.get("reward") or {}).get("outcome", 0.0)) >= 1.0
        if n:
            per[step] = s / n
    return [per[k] for k in sorted(per)]


def val(run, env, step):
    p = f"experiments/{env}/{run}/validation_log/{step}.jsonl"
    if not os.path.exists(p):
        return None
    sc = [float(json.loads(l).get("score", json.loads(l).get("reward", 0.0)))
          for l in open(p) if l.strip()]
    if not sc:
        return None
    return {"n": len(sc), "strict": sum(s >= 1.0 for s in sc) / len(sc) * 100,
            "reward": sum(sc) / len(sc), "neg": sum(s < 0 for s in sc) / len(sc) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--env", default="webshop")
    args = ap.parse_args()

    ok = True
    print(f"=== replicate health: {args.run}")
    print(f"    reference:        {args.reference}\n")

    # 1. did task_seed reach the run that actually executed?
    snap = f"launcher_record/{args.run}/yaml_backup.yaml"
    if os.path.exists(snap):
        txt = open(snap).read()
        m_seed = re.search(r"^\s*seed:\s*(\d+)\s*$", txt, re.M)
        m_task = re.search(r"^\s*task_seed:\s*(\d+)\s*$", txt, re.M)
        print(f"[config snapshot] seed={m_seed.group(1) if m_seed else '?'} "
              f"task_seed={m_task.group(1) if m_task else 'ABSENT'}")
        if not m_task:
            print("    WARNING: no task_seed in the executed config — this run drew its own curriculum")
            ok = False
    else:
        print(f"[config snapshot] {snap} not found (run may not have launched yet)")

    # 2. is it training on the reference curriculum?
    a, b = per_step_tasks(args.reference), per_step_tasks(args.run)
    if not b:
        print("[curriculum] no trajectories yet — rerun once the first steps are saved")
        return
    common = sorted(set(a) & set(b))
    same = sum(1 for s in common if a[s] == b[s])
    union_a = set().union(*a.values()) if a else set()
    union_b = set().union(*b.values()) if b else set()
    shared = len(union_a & union_b)
    print(f"[curriculum] steps compared {len(common)} | per-step task sets identical: {same}/{len(common)}")
    print(f"             task union: run {len(union_b)}, reference {len(union_a)}, shared {shared}")
    if union_b and shared / max(len(union_b), 1) < 0.99:
        print("    WARNING: the run is NOT on the reference curriculum")
        ok = False
    elif same != len(common):
        print("    NOTE: same task pool but a different per-step order (expected: the dataloader"
              " generator is seeded by data.seed, which differs by design)")

    # 3. is it healthy and on a comparable trajectory?
    ca, cb = onpolicy_curve(args.reference), onpolicy_curve(args.run)
    if cb:
        blocks = lambda c: [f"{stats.mean(c[i:i+10]):.3f}" for i in range(0, len(c), 10)]
        print(f"[on-policy success, 10-step blocks]")
        print(f"   reference: {', '.join(blocks(ca))}")
        print(f"   this run : {', '.join(blocks(cb))}   ({len(cb)} steps done)")
        k = len(cb)
        if k >= 10 and stats.mean(cb[:k]) < 0.25 * stats.mean(ca[:k]):
            print("    WARNING: far below the reference at the same point in training")
            ok = False

    # 4. validation, if it exists yet
    for step in (50, 100, 150):
        r_run, r_ref = val(args.run, args.env, step), val(args.reference, args.env, step)
        if r_run:
            ref = f" | reference {r_ref['strict']:.1f}% / {r_ref['reward']:.3f}" if r_ref else ""
            print(f"[val@{step}] strict {r_run['strict']:.1f}%  reward {r_run['reward']:.3f}  "
                  f"negative-score {r_run['neg']:.1f}%  n={r_run['n']}{ref}")

    # 5. crude env-health check on the run log
    log = f"logs/{args.run}.log"
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        bad = {k: txt.count(k) for k in ("Traceback", "EngineDeadError", "ActorDiedError",
                                         "CUDA out of memory", "failed to load")}
        hits = {k: v for k, v in bad.items() if v}
        print(f"[run log] {'clean' if not hits else 'ISSUES: ' + str(hits)}")
        if hits:
            ok = False

    print(f"\n=== {'HEALTHY' if ok else 'NEEDS ATTENTION'}")


if __name__ == "__main__":
    main()
