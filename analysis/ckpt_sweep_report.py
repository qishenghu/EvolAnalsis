#!/usr/bin/env python3
"""Checkpoint-sweep report: greedy vs sampled decoding on the 128-task val set.

Reads the sweep's rollout jsonl files and adjudicates the question the sweep was
built for — is the collapsing in-training val curve a *decoding artifact* or
*real generalisation loss*? The ruling thresholds are frozen in VERDICT_RULE
below and were fixed before any number existed.

Usage:
  python analysis/ckpt_sweep_report.py \
      --sweep-dir /projects_vol/gp_wangwy/qisheng/duet_h200/ckpt_sweep \
      --out-dir analysis_outputs/ckpt_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Frozen adjudication rule (see run_ckpt_sweep.pbs header).
VERDICT_RULE = """\
Let D_greedy  = peak-to-final drop of the greedy success rate (percentage points)
    D_sampled = peak-to-final drop of the sampled mean pass@1 (percentage points)
  * decoding artifact  : D_greedy >= 10pp AND D_sampled <= max(0.5*D_greedy, 5pp)
  * real degradation   : D_sampled >= 0.6*D_greedy AND D_sampled >= 7pp
  * intermediate       : anything else — both magnitudes reported side by side"""


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval — honest at the small n (128) this sweep runs."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> Optional[float]:
    if min(n1, n2) == 0:
        return None
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    return (p1 - p2) / se


def load_mode(sweep_dir: Path, step: int, mode: str) -> List[Dict[str, Any]]:
    """Prefer the consolidated file; fall back to per-shard files."""
    records: Dict[str, Dict[str, Any]] = {}
    # `shard*.jsonl` also matches the `shardN.jsonl.attempts.jsonl` telemetry
    # sidecars the collector writes.  Those carry `rollout_id` + `success` but
    # none of the episode payload, and they sort AFTER the real shard, so the
    # dedup below let them overwrite every real record — which is why the token
    # / decision / truncation table came out all zeros.  Match digits only.
    candidates = sorted(sweep_dir.glob(f"{step}_{mode}_shard[0-9].jsonl"))
    consolidated = sweep_dir / f"{step}_{mode}.jsonl"
    if consolidated.is_file() and not candidates:
        candidates = [consolidated]
    elif consolidated.is_file() and candidates:
        candidates = candidates  # shards are the source of truth (dedup below)
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            # a resumed collector can re-emit a slot; last write wins
            records[rec["rollout_id"]] = rec
    return list(records.values())


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_task[str(rec["task_id"])].append(rec)

    n = len(records)
    successes = sum(1 for r in records if bool(r.get("success")))
    tasks = len(by_task)
    pass_at_k_tasks = sum(
        1 for rs in by_task.values() if any(bool(r.get("success")) for r in rs)
    )
    tot_completion = 0.0
    tot_decisions = 0.0
    trunc_episodes = 0
    trunc_decisions = 0
    invalid_decisions = 0
    end_reasons: Counter = Counter()
    per_episode_tokens: List[float] = []
    for rec in records:
        meta = rec.get("metadata", {}) or {}
        totals = meta.get("api_and_context_totals", {}) or {}
        quality = meta.get("trace_quality", {}) or {}
        comp = float(totals.get("api_completion_tokens", 0.0))
        dec = float(totals.get("decision_count", quality.get("decision_count", 0)))
        tot_completion += comp
        tot_decisions += dec
        per_episode_tokens.append(comp)
        lt = int(quality.get("length_truncated_decisions",
                             totals.get("length_truncated_decisions", 0)))
        trunc_decisions += lt
        trunc_episodes += 1 if lt > 0 else 0
        invalid_decisions += int(quality.get("invalid_action_decisions", 0))
        end_reasons[str(meta.get("episode_end_reason", "?"))] += 1

    lo, hi = wilson(successes, n)
    return {
        "n": n,
        "tasks": tasks,
        "successes": successes,
        "pass1": successes / n,
        "pass1_ci": (lo, hi),
        "pass_at_k_tasks": pass_at_k_tasks,
        "pass_at_k": pass_at_k_tasks / tasks if tasks else 0.0,
        "mean_tokens_per_episode": sum(per_episode_tokens) / n,
        "mean_tokens_per_decision": (tot_completion / tot_decisions) if tot_decisions else 0.0,
        "mean_decisions": tot_decisions / n,
        "trunc_episode_rate": trunc_episodes / n,
        "trunc_decision_rate": (trunc_decisions / tot_decisions) if tot_decisions else 0.0,
        "invalid_action_decision_rate": (invalid_decisions / tot_decisions) if tot_decisions else 0.0,
        "end_reasons": dict(end_reasons.most_common()),
    }


def trainer_val_curve(repo: Path, experiment: str) -> Dict[int, Dict[str, Any]]:
    """The trainer's own in-run greedy val log, for cross-validation."""
    out: Dict[int, Dict[str, Any]] = {}
    d = repo / "experiments/alfworld" / experiment / "validation_log"
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.jsonl")):
        m = re.fullmatch(r"(\d+)", path.stem)
        if not m:
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        out[int(m.group(1))] = {
            "n": len(rows),
            "sr": sum(1 for r in rows if float(r.get("reward", 0)) > 0) / len(rows),
            "mean_decisions": sum(r.get("decision_count", 0) for r in rows) / len(rows),
            "trunc_rate": sum(1 for r in rows if r.get("truncated_by_length")) / len(rows),
            "mtime": path.stat().st_mtime,
        }
    return out


def curve_shape(points: List[tuple[int, float]]) -> Dict[str, Any]:
    if not points:
        return {}
    peak_step, peak = max(points, key=lambda kv: kv[1])
    final_step, final = points[-1]
    first_step, first = points[0]
    return {
        "first_step": first_step, "first": first,
        "peak_step": peak_step, "peak": peak,
        "final_step": final_step, "final": final,
        "drop_pp": (peak - final) * 100,
        "net_pp": (final - first) * 100,
    }


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/ckpt_sweep"))
    ap.add_argument("--primary-steps", default="30,40,50,60,70",
                    help="steps from one coherent training run (curve + verdict)")
    ap.add_argument("--supplementary-steps", default="10,20",
                    help="steps from a different run instance; reported, not ruled on")
    ap.add_argument("--experiment", default="p0_catalyst_af_s0")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    primary = [int(s) for s in args.primary_steps.split(",") if s.strip()]
    supplementary = [int(s) for s in args.supplementary_steps.split(",") if s.strip()]
    all_steps = supplementary + primary

    data: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for step in all_steps:
        entry = {}
        for mode in ("greedy", "sampled"):
            entry[mode] = summarise(load_mode(args.sweep_dir, step, mode))
        if entry["greedy"].get("n") or entry["sampled"].get("n"):
            data[step] = entry

    trainer = trainer_val_curve(repo, args.experiment)

    have_primary = [s for s in primary if s in data]
    greedy_pts = [(s, data[s]["greedy"]["pass1"]) for s in have_primary
                  if data[s]["greedy"].get("n")]
    sampled_pts = [(s, data[s]["sampled"]["pass1"]) for s in have_primary
                   if data[s]["sampled"].get("n")]
    passk_pts = [(s, data[s]["sampled"]["pass_at_k"]) for s in have_primary
                 if data[s]["sampled"].get("n")]
    g_shape, s_shape, k_shape = (curve_shape(greedy_pts), curve_shape(sampled_pts),
                                 curve_shape(passk_pts))

    verdict, reasoning = "insufficient data", []
    if g_shape and s_shape:
        dg, ds = g_shape["drop_pp"], s_shape["drop_pp"]
        if dg >= 10 and ds <= max(0.5 * dg, 5):
            verdict = "DECODING ARTIFACT"
            reasoning.append(
                f"greedy falls {dg:.1f}pp from its peak while sampled pass@1 falls only "
                f"{ds:.1f}pp (<= max(0.5*{dg:.1f}, 5)). The policy distribution holds up; "
                f"the argmax path does not.")
        elif ds >= 0.6 * dg and ds >= 7:
            verdict = "REAL DEGRADATION"
            reasoning.append(
                f"both decoders fall together (greedy {dg:.1f}pp, sampled {ds:.1f}pp >= "
                f"0.6x greedy and >= 7pp). The distribution itself got worse.")
        else:
            verdict = "INTERMEDIATE"
            reasoning.append(
                f"greedy drop {dg:.1f}pp vs sampled drop {ds:.1f}pp fits neither frozen "
                f"branch; treat the split as partial.")
        # statistical honesty at n=128 / n=512
        gs_peak = data[g_shape["peak_step"]]["greedy"]
        gs_fin = data[g_shape["final_step"]]["greedy"]
        z_g = two_proportion_z(gs_peak["successes"], gs_peak["n"],
                               gs_fin["successes"], gs_fin["n"])
        ss_peak = data[s_shape["peak_step"]]["sampled"]
        ss_fin = data[s_shape["final_step"]]["sampled"]
        z_s = two_proportion_z(ss_peak["successes"], ss_peak["n"],
                               ss_fin["successes"], ss_fin["n"])
        reasoning.append(
            f"peak-vs-final two-proportion z: greedy {z_g:.2f}, sampled {z_s:.2f} "
            f"(|z|>1.96 = significant at 5%).")

    lines: List[str] = []
    A = lines.append
    A("# Checkpoint sweep — greedy vs sampled decoding on the held-out 128")
    A("")
    A(f"Experiment `{args.experiment}` · ALFWorld val prefix (game indices 2420-2547, "
      "the exact 128 tasks and order the trainer evaluates, "
      "`ordered_newline_sha256=d90efe607c...42915`).")
    A("")
    A("| decoder | sampling | rollouts/task | episodes/checkpoint |")
    A("|---|---|---|---|")
    A("| greedy  | temperature 0, top_p 1.0   | 1 | 128 |")
    A("| sampled | temperature 0.9, top_p 1.0 | 4 | 512 |")
    A("")
    A("Greedy reproduces `rollout.val_kwargs` exactly; sampled reproduces the training "
      "rollout distribution (`rollout.temperature=0.9`), at n=4 instead of n=8.")
    A("")

    A("## Verdict")
    A("")
    A(f"**{verdict}**")
    A("")
    for r in reasoning:
        A(f"- {r}")
    A("")
    A("Frozen rule:")
    A("")
    A("```")
    A(VERDICT_RULE)
    A("```")
    A("")

    A("## Curves (primary run)")
    A("")
    A("| step | greedy SR | 95% CI | sampled pass@1 | 95% CI | sampled pass@4 |")
    A("|---:|---:|---:|---:|---:|---:|")
    for step in have_primary:
        g, s = data[step]["greedy"], data[step]["sampled"]
        gci = f"{pct(g['pass1_ci'][0])}-{pct(g['pass1_ci'][1])}" if g.get("n") else "-"
        sci = f"{pct(s['pass1_ci'][0])}-{pct(s['pass1_ci'][1])}" if s.get("n") else "-"
        A(f"| {step} | {pct(g['pass1']) if g.get('n') else '-'} | {gci} | "
          f"{pct(s['pass1']) if s.get('n') else '-'} | {sci} | "
          f"{pct(s['pass_at_k']) if s.get('n') else '-'} |")
    A("")
    for name, shape in (("greedy SR", g_shape), ("sampled pass@1", s_shape),
                        ("sampled pass@4", k_shape)):
        if shape:
            A(f"- **{name}**: first {pct(shape['first'])} @{shape['first_step']}, "
              f"peak {pct(shape['peak'])} @{shape['peak_step']}, "
              f"final {pct(shape['final'])} @{shape['final_step']}, "
              f"peak-to-final drop **{shape['drop_pp']:.1f}pp**, "
              f"net first-to-final {shape['net_pp']:+.1f}pp")
    A("")

    have_supp = [s for s in supplementary if s in data]
    if have_supp:
        A("## Supplementary checkpoints (different run instance — not part of the ruling)")
        A("")
        A("| step | greedy SR | sampled pass@1 | sampled pass@4 |")
        A("|---:|---:|---:|---:|")
        for step in have_supp:
            g, s = data[step]["greedy"], data[step]["sampled"]
            A(f"| {step} | {pct(g['pass1']) if g.get('n') else '-'} | "
              f"{pct(s['pass1']) if s.get('n') else '-'} | "
              f"{pct(s['pass_at_k']) if s.get('n') else '-'} |")
        A("")

    A("## Length and truncation (cross-validation)")
    A("")
    A("| step | mode | n | mean tok/episode | mean tok/decision | mean decisions | "
      "episodes w/ length-trunc | decisions length-trunc | invalid-action decisions |")
    A("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for step in sorted(data):
        for mode in ("greedy", "sampled"):
            m = data[step][mode]
            if not m.get("n"):
                continue
            A(f"| {step} | {mode} | {m['n']} | {m['mean_tokens_per_episode']:.0f} | "
              f"{m['mean_tokens_per_decision']:.0f} | {m['mean_decisions']:.2f} | "
              f"{pct(m['trunc_episode_rate'])} | {pct(m['trunc_decision_rate'])} | "
              f"{pct(m['invalid_action_decision_rate'])} |")
    A("")

    if trainer:
        A("## Trainer's own in-run greedy val log (same 128 tasks)")
        A("")
        A("Cross-check for the sweep's greedy column. Note these files are overwritten "
          "by every restart of the experiment, so their provenance follows the file mtime.")
        A("")
        A("| step | SR | n | mean decisions | trunc rate | log mtime |")
        A("|---:|---:|---:|---:|---:|---|")
        import datetime as _dt
        for step in sorted(trainer):
            t = trainer[step]
            when = _dt.datetime.fromtimestamp(t["mtime"]).strftime("%Y-%m-%d %H:%M")
            A(f"| {step} | {pct(t['sr'])} | {t['n']} | {t['mean_decisions']:.2f} | "
              f"{pct(t['trunc_rate'])} | {when} |")
        A("")

    A("## Episode end reasons")
    A("")
    for step in sorted(data):
        for mode in ("greedy", "sampled"):
            m = data[step][mode]
            if m.get("n"):
                A(f"- step {step} {mode}: {m['end_reasons']}")
    A("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / "ckpt_sweep_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "ckpt_sweep_metrics.json").write_text(
        json.dumps({"data": data, "trainer_val_log": trainer,
                    "shapes": {"greedy": g_shape, "sampled_pass1": s_shape,
                               "sampled_pass4": k_shape},
                    "verdict": verdict, "reasoning": reasoning},
                   indent=2, default=str),
        encoding="utf-8")
    print(f"wrote {report}")
    print("\n".join(lines[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
