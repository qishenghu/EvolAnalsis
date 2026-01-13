#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate per-group (per-task) gap heterogeneity using existing local logs:
- trajectories_step_*.jsonl: rollout-level info incl. reward and is_teacher

We compute, for each step:
  - per-task gap: gap_g = teacher_reward_g - mean(on_policy_rewards_g)
  - distribution stats across tasks: mean/std/p50/p90/max, frac(gap>thr)
  - optionally align with W&B reward and teacher_loss_scale (global scalar) to show
    why batch-mean gating can 'mis-kill' hard tasks if gap distribution is heavy-tailed.

Outputs:
  analysis/luffy_no_logprob_improvement_compare/out_v2/per_group_gap/...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_teacher_rollout(r: dict) -> bool:
    # common locations for the flag
    md = r.get("metadata", {}) or {}
    if "is_teacher" in md:
        return bool(md["is_teacher"])
    dg = r.get("diag", {}) or {}
    if "is_teacher" in dg:
        return bool(dg["is_teacher"])
    if "is_teacher" in r:
        return bool(r["is_teacher"])
    # fallback: some logs may mark by exp_mask/teacher_mask elsewhere; not available here
    return False


def get_task_id(r: dict) -> str:
    # prefer task_id; fallback to uid/data_id for grouping inside a step
    if "task_id" in r:
        return str(r["task_id"])
    if "uid" in r:
        return str(r["uid"])
    if "data_id" in r:
        return str(r["data_id"])
    raise KeyError("No task identifier found in rollout JSON.")


def get_reward(r: dict) -> float:
    # typical key is 'reward'
    if "reward" in r:
        rv = r["reward"]
        # common schema in this repo: {"outcome": 0/1, ...}
        if isinstance(rv, dict):
            if "outcome" in rv:
                return float(rv["outcome"])
            if "reward" in rv:
                return float(rv["reward"])
            if "score" in rv:
                return float(rv["score"])
        # sometimes already numeric
        return float(rv)
    # fallback: sometimes reward is duplicated in diag
    dg = r.get("diag", {}) or {}
    if "reward_sum" in dg:
        return float(dg["reward_sum"])
    # some logs use reward_scores or similar; keep minimal
    raise KeyError("No reward found in rollout JSON.")


def per_step_group_gaps(step_jsonl: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - df_groups: per-task rows: step, task_id, teacher_reward, on_mean, on_n, gap
      - df_step: one row per step: distribution stats over tasks for that step
    """
    step = int(re.search(r"trajectories_step_(\d+)\.jsonl$", step_jsonl.name).group(1))  # type: ignore
    rollouts = read_jsonl(step_jsonl)

    by_task: Dict[str, Dict[str, List[float] | float | None]] = {}
    for r in rollouts:
        tid = get_task_id(r)
        tr = by_task.setdefault(tid, {"teacher": None, "on": []})
        rew = get_reward(r)
        if is_teacher_rollout(r):
            tr["teacher"] = rew
        else:
            tr["on"].append(rew)  # type: ignore[arg-type]

    rows = []
    for tid, d in by_task.items():
        teacher = d["teacher"]
        on = d["on"]  # type: ignore[assignment]
        on_n = len(on)
        on_mean = float(np.mean(on)) if on_n else float("nan")
        teacher_reward = float(teacher) if teacher is not None else float("nan")
        gap = teacher_reward - on_mean if (not math.isnan(teacher_reward) and not math.isnan(on_mean)) else float("nan")
        rows.append(
            {
                "_step": step,
                "task_id": tid,
                "teacher_reward": teacher_reward,
                "on_mean_reward": on_mean,
                "on_n": on_n,
                "gap": gap,
            }
        )

    df_groups = pd.DataFrame(rows).sort_values(["_step", "task_id"]).reset_index(drop=True)
    g = pd.to_numeric(df_groups["gap"], errors="coerce").dropna()
    if len(g) == 0:
        df_step = pd.DataFrame([{"_step": step}])
        return df_groups, df_step

    def q(p: float) -> float:
        return float(np.quantile(g, p))

    df_step = pd.DataFrame(
        [
            {
                "_step": step,
                "n_tasks": float(len(g)),
                "gap_mean": float(g.mean()),
                "gap_std": float(g.std(ddof=0)),
                "gap_p50": q(0.50),
                "gap_p90": q(0.90),
                "gap_max": float(g.max()),
                "frac_gap_gt_0_5": float((g > 0.5).mean()),
                "frac_gap_gt_0_7": float((g > 0.7).mean()),
            }
        ]
    )
    return df_groups, df_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True, help="Path to Trajectory directory containing trajectories_step_*.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_steps", type=int, default=100)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_rows = []
    step_rows = []
    for step in range(1, args.max_steps + 1):
        fp = traj_dir / f"trajectories_step_{step}.jsonl"
        if not fp.exists():
            continue
        df_g, df_s = per_step_group_gaps(fp)
        group_rows.append(df_g)
        step_rows.append(df_s)

    df_groups = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    df_steps = pd.concat(step_rows, ignore_index=True) if step_rows else pd.DataFrame()

    df_groups.to_csv(out_dir / "per_task_gaps.csv", index=False)
    df_steps.to_csv(out_dir / "per_step_gap_stats.csv", index=False)

    # quick plots
    try:
        import matplotlib.pyplot as plt

        if not df_steps.empty and "gap_mean" in df_steps.columns:
            df_steps = df_steps.sort_values("_step")
            plt.figure(figsize=(12, 5))
            plt.plot(df_steps["_step"], df_steps["gap_mean"], label="gap_mean")
            if "gap_p90" in df_steps.columns:
                plt.plot(df_steps["_step"], df_steps["gap_p90"], label="gap_p90")
            if "gap_max" in df_steps.columns:
                plt.plot(df_steps["_step"], df_steps["gap_max"], label="gap_max")
            plt.title("Per-step per-task gap distribution (mean / p90 / max)")
            plt.xlabel("step")
            plt.ylabel("gap = teacher - mean(on)")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "gap_mean_p90_max.png", dpi=200)
            plt.close()

        # histogram of gaps pooled over steps
        if not df_groups.empty and "gap" in df_groups.columns:
            g = pd.to_numeric(df_groups["gap"], errors="coerce").dropna()
            if len(g):
                plt.figure(figsize=(7, 5))
                plt.hist(g, bins=30)
                plt.title("Pooled per-task gap histogram")
                plt.xlabel("gap")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(out_dir / "gap_hist.png", dpi=200)
                plt.close()
    except Exception:
        pass

    print("[OK] wrote:", out_dir / "per_task_gaps.csv")
    print("[OK] wrote:", out_dir / "per_step_gap_stats.csv")


if __name__ == "__main__":
    main()

