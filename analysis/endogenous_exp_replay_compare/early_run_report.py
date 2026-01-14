#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Early-step mechanistic report for a single run (e.g., liy20klj, 34 steps).

Inputs:
- W&B history for the run (training curves + replay metrics)
- Optional local trajectory diagnostics aggregated per step (CSV)

Outputs:
- analysis/endogenous_exp_replay_compare/out/<tag>/figs/*.png
- analysis/endogenous_exp_replay_compare/out/<tag>/compiled/*.csv
- docs/analysis/<date>_<run_id>_early_report.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _moving_avg(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).mean()


def load_wandb_history(run_id: str, keys: List[str], entity: str, project: str) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    rows: List[Dict] = []
    scan_keys = [k for k in keys if k != "_step"]
    for row in run.scan_history(keys=scan_keys):
        if row is None:
            continue
        rows.append(dict(row))

    df: Optional[pd.DataFrame] = None
    if rows:
        df0 = pd.DataFrame(rows)
        if "_step" not in df0.columns and "step" in df0.columns:
            df0 = df0.rename(columns={"step": "_step"})
        if "_step" in df0.columns:
            df = df0

    if df is None:
        hist = run.history()
        if "_step" not in hist.columns:
            if "step" in hist.columns:
                hist = hist.rename(columns={"step": "_step"})
            elif "training/global_step" in hist.columns:
                hist = hist.copy()
                hist["_step"] = hist["training/global_step"]
            else:
                hist = hist.copy()
                hist["_step"] = np.arange(len(hist), dtype=int)
        keep_raw = ["_step"] + scan_keys
        keep = list(dict.fromkeys([c for c in keep_raw if c in hist.columns]))
        df = hist[keep].dropna(subset=["_step"]).copy()

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.sort_values("_step").drop_duplicates("_step", keep="last").reset_index(drop=True)
    return df


def df_to_markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    df2 = df.reset_index(drop=True).copy()

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (np.floating, float)):
            return format(float(v), floatfmt)
        if isinstance(v, (np.integer, int)):
            return str(int(v))
        return str(v)

    headers = [str(c) for c in df2.columns]
    rows = [[fmt(v) for v in row] for row in df2.to_numpy()]

    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def _plot_lines(out_png: Path, dfs: Dict[str, pd.DataFrame], y_cols: List[str], title: str, ylabel: str, ma: int) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for label, df in dfs.items():
        for y_col in y_cols:
            if y_col not in df.columns:
                continue
            d = df[["_step", y_col]].dropna().sort_values("_step")
            if d.empty:
                continue
            y = _moving_avg(pd.to_numeric(d[y_col], errors="coerce"), ma)
            plt.plot(d["_step"], y, label=f"{label}:{y_col}", linewidth=2.0, alpha=0.9)

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    _safe_mkdir(out_png.parent)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 5:
        return float("nan")
    return float(np.corrcoef(x[m].to_numpy(), y[m].to_numpy())[0, 1])


def _first_nonzero_step(df: pd.DataFrame, col: str, eps: float = 0.0) -> Optional[int]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    idx = df.loc[s > eps, "_step"]
    if idx.empty:
        return None
    return int(idx.iloc[0])


def _seg_mean(df: pd.DataFrame, cols: List[str], lo: int, hi: int) -> Dict[str, float]:
    d = df[(df["_step"] >= lo) & (df["_step"] <= hi)].copy()
    out: Dict[str, float] = {"lo": float(lo), "hi": float(hi), "n": float(len(d))}
    for c in cols:
        if c not in d.columns:
            out[c] = float("nan")
            continue
        s = pd.to_numeric(d[c], errors="coerce").dropna()
        out[c] = float(s.mean()) if len(s) else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--entity", default="qisheng001-nanyang-technological-university-singapore")
    ap.add_argument("--project", default="agentevolver")
    ap.add_argument("--max_step", type=int, default=34)
    ap.add_argument("--ma", type=int, default=3)
    ap.add_argument("--local_diag_csv", default="")
    ap.add_argument("--traj_dir", default="", help="Optional path to Trajectory dir containing trajectories_step_*.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_root", default="analysis/endogenous_exp_replay_compare/out")
    ap.add_argument("--report_path", default="")
    args = ap.parse_args()

    tag = args.tag.strip() or f"early_{args.run_id}"
    out_root = Path(args.out_root)
    out_dir = out_root / tag
    figs_dir = out_dir / "figs"
    compiled_dir = out_dir / "compiled"
    _safe_mkdir(figs_dir)
    _safe_mkdir(compiled_dir)

    # W&B keys (small but covers the mechanism chain we care about)
    keys = [
        "_step",
        "training/global_step",
        "critic/rewards/mean",
        "actor/entropy_loss",
        "actor/kl_loss",
        "actor/pg_loss",
        # replay intensity / pool
        "exp_replay/num_experience_tasks",
        "exp_replay/num_offpolicy_trajectories",
        "exp_replay/total_tasks_in_pool",
        "exp_replay/offpolicy_rollout_ratio",
        "exp_replay/offpolicy_token_ratio_llm",
        "exp_replay/offpolicy_token_ratio_response",
        # staleness (age)
        "exp_replay/offpolicy_age_mean",
        "exp_replay/offpolicy_age_max",
        "exp_replay/offpolicy_age_min",
        "exp_replay/offpolicy_age_count",
        # entropy split
        "exp_replay/entropy_llm_mean",
        "exp_replay/entropy_llm_onpolicy_mean",
        "exp_replay/entropy_llm_offpolicy_mean",
        # optional diagnostic stats (if logged)
        "exp_replay_diag/importance_ratio/off/mean",
        "exp_replay_diag/importance_ratio/off/p90",
        "exp_replay_diag/importance_ratio/off/p99",
        "exp_replay_diag/importance_ratio/off/max",
        "exp_replay_diag/importance_ratio_shaped/off/mean",
        "exp_replay_diag/importance_ratio_shaped/off/p99",
        "exp_replay_diag/adv/off/mean",
        "exp_replay_diag/adv/on/mean",
    ]

    df_w = load_wandb_history(args.run_id, keys, entity=args.entity, project=args.project)
    df_w = df_w[df_w["_step"] <= args.max_step].copy()
    df_w.to_csv(compiled_dir / f"{args.run_id}_wandb_1_{args.max_step}.csv", index=False)

    df_local: Optional[pd.DataFrame] = None
    if args.local_diag_csv.strip():
        p = Path(args.local_diag_csv)
        if p.exists():
            df_local = pd.read_csv(p).sort_values("_step").copy()
            df_local = df_local[df_local["_step"] <= args.max_step].copy()
            df_local.to_csv(compiled_dir / f"{args.run_id}_local_diag_1_{args.max_step}.csv", index=False)

    df = df_w.copy()
    if df_local is not None:
        df = pd.merge(df, df_local, on="_step", how="outer", suffixes=("", "_local"))
        df = df.sort_values("_step").reset_index(drop=True)
    df.to_csv(compiled_dir / f"{args.run_id}_merged_1_{args.max_step}.csv", index=False)

    # Optional: trajectory-level stats from local JSONL logs (on-policy vs off-policy rollouts)
    traj_step_df: Optional[pd.DataFrame] = None
    traj_post_summary_df: Optional[pd.DataFrame] = None
    if args.traj_dir.strip():
        import json

        traj_dir = Path(args.traj_dir)
        rows = []
        for step in range(1, args.max_step + 1):
            fp = traj_dir / f"trajectories_step_{step}.jsonl"
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    d = o.get("diag", {}) or {}
                    off_ratio = float(d.get("offpolicy_ratio", 0.0) or 0.0)
                    exp_tokens = float(d.get("exp_tokens", 0.0) or 0.0)
                    resp_valid = float(d.get("response_valid_tokens", 0.0) or 0.0)
                    # reward.outcome (may be nested dict)
                    reward = o.get("reward")
                    outcome = None
                    if isinstance(reward, dict):
                        outcome = reward.get("outcome", None)
                    try:
                        outcome_f = float(outcome) if outcome is not None else float("nan")
                    except Exception:
                        outcome_f = float("nan")
                    # entropy could be scalar or dict
                    ent = o.get("entropy", None)
                    ent_f = float("nan")
                    if isinstance(ent, (int, float, np.floating)):
                        ent_f = float(ent)
                    elif isinstance(ent, dict):
                        # try common names
                        for k in ["mean", "entropy", "mean_entropy"]:
                            if k in ent:
                                try:
                                    ent_f = float(ent[k])
                                    break
                                except Exception:
                                    pass
                    # Use reward outcome as the single source of truth for "success".
                    # NOTE: we observed cases where saved `success` disagrees with `reward.outcome`.
                    success_f = float(outcome_f > 0.0) if np.isfinite(outcome_f) else 0.0

                    rows.append(
                        {
                            "_step": int(step),
                            "is_offpolicy": float(off_ratio > 0.0),
                            "offpolicy_ratio": off_ratio,
                            "exp_tokens": exp_tokens,
                            "response_valid_tokens": resp_valid,
                            "reward_outcome": outcome_f,
                            "success": success_f,
                            "entropy_scalar": ent_f,
                        }
                    )
        if rows:
            traj = pd.DataFrame(rows)
            # post-only summary (if we can infer replay turn-on later, we'll slice again after t_on is known)
            # per-step aggregation (on/off split)
            out_rows = []
            for step, g in traj.groupby("_step"):
                for is_off in [0.0, 1.0]:
                    gg = g[g["is_offpolicy"] == is_off]
                    if gg.empty:
                        continue
                    out_rows.append(
                        {
                            "_step": int(step),
                            "split": "offpolicy" if is_off > 0 else "onpolicy",
                            "n_rollouts": float(len(gg)),
                            "success_rate": float(np.nanmean(gg["success"].to_numpy())),
                            "reward_outcome_mean": float(np.nanmean(gg["reward_outcome"].to_numpy())),
                            "response_valid_tokens_mean": float(np.nanmean(gg["response_valid_tokens"].to_numpy())),
                            "entropy_scalar_mean": float(np.nanmean(gg["entropy_scalar"].to_numpy())),
                            "offpolicy_ratio_mean": float(np.nanmean(gg["offpolicy_ratio"].to_numpy())),
                            "exp_tokens_mean": float(np.nanmean(gg["exp_tokens"].to_numpy())),
                        }
                    )
                # overall counts
                out_rows.append(
                    {
                        "_step": int(step),
                        "split": "overall",
                        "n_rollouts": float(len(g)),
                        "success_rate": float(np.nanmean(g["success"].to_numpy())),
                        "reward_outcome_mean": float(np.nanmean(g["reward_outcome"].to_numpy())),
                        "response_valid_tokens_mean": float(np.nanmean(g["response_valid_tokens"].to_numpy())),
                        "entropy_scalar_mean": float(np.nanmean(g["entropy_scalar"].to_numpy())),
                        "offpolicy_ratio_mean": float(np.nanmean(g["offpolicy_ratio"].to_numpy())),
                        "exp_tokens_mean": float(np.nanmean(g["exp_tokens"].to_numpy())),
                    }
                )
            traj_step_df = pd.DataFrame(out_rows).sort_values(["_step", "split"]).reset_index(drop=True)
            traj.to_csv(compiled_dir / f"{args.run_id}_traj_rollouts_raw_1_{args.max_step}.csv", index=False)
            traj_step_df.to_csv(compiled_dir / f"{args.run_id}_traj_step_stats_1_{args.max_step}.csv", index=False)

            # Plot: success rate on vs off
            d_on = traj_step_df[traj_step_df["split"] == "onpolicy"]
            d_off = traj_step_df[traj_step_df["split"] == "offpolicy"]
            if not d_on.empty or not d_off.empty:
                df_plot = {"traj": pd.DataFrame({"_step": [], "y": []})}
                # Reuse plotting helper by materializing columns
                wide = pd.DataFrame({"_step": sorted(traj_step_df["_step"].unique())})
                wide = wide.merge(d_on[["_step", "success_rate"]].rename(columns={"success_rate": "traj/success_on"}), on="_step", how="left")
                wide = wide.merge(d_off[["_step", "success_rate"]].rename(columns={"success_rate": "traj/success_off"}), on="_step", how="left")
                wide = wide.merge(d_on[["_step", "reward_outcome_mean"]].rename(columns={"reward_outcome_mean": "traj/reward_on"}), on="_step", how="left")
                wide = wide.merge(d_off[["_step", "reward_outcome_mean"]].rename(columns={"reward_outcome_mean": "traj/reward_off"}), on="_step", how="left")
                wide = wide.merge(d_on[["_step", "response_valid_tokens_mean"]].rename(columns={"response_valid_tokens_mean": "traj/resp_tokens_on"}), on="_step", how="left")
                wide = wide.merge(d_off[["_step", "response_valid_tokens_mean"]].rename(columns={"response_valid_tokens_mean": "traj/resp_tokens_off"}), on="_step", how="left")
                wide = wide.merge(d_on[["_step", "exp_tokens_mean"]].rename(columns={"exp_tokens_mean": "traj/exp_tokens_on"}), on="_step", how="left")
                wide = wide.merge(d_off[["_step", "exp_tokens_mean"]].rename(columns={"exp_tokens_mean": "traj/exp_tokens_off"}), on="_step", how="left")
                _plot_lines(
                    figs_dir / "traj_on_off_success_reward_ma.png",
                    {"traj": wide},
                    ["traj/success_on", "traj/success_off", "traj/reward_on", "traj/reward_off"],
                    title=f"{args.run_id}: trajectory success/reward (on vs off) (MA w={args.ma})",
                    ylabel="rate / reward",
                    ma=args.ma,
                )
                _plot_lines(
                    figs_dir / "traj_on_off_token_ma.png",
                    {"traj": wide},
                    ["traj/resp_tokens_on", "traj/resp_tokens_off", "traj/exp_tokens_off"],
                    title=f"{args.run_id}: response/exp tokens (on vs off) (MA w={args.ma})",
                    ylabel="tokens",
                    ma=args.ma,
                )

    # Replay "turn-on" step as an event
    t_on_w = _first_nonzero_step(df, "exp_replay/offpolicy_rollout_ratio", eps=0.0)
    t_on_l = _first_nonzero_step(df, "diag/offpolicy_sample_ratio", eps=0.0)
    t_on = min([t for t in [t_on_w, t_on_l] if t is not None], default=None)

    # If we have raw traj rows, compute post summary now (using inferred t_on)
    if args.traj_dir.strip():
        raw_fp = compiled_dir / f"{args.run_id}_traj_rollouts_raw_1_{args.max_step}.csv"
        if raw_fp.exists():
            traj_raw = pd.read_csv(raw_fp)
            if t_on is not None:
                traj_raw = traj_raw[traj_raw["_step"] >= t_on].copy()
            # split summary
            srows = []
            for split_name, mask in [
                ("onpolicy", traj_raw["is_offpolicy"] <= 0.0),
                ("offpolicy", traj_raw["is_offpolicy"] > 0.0),
            ]:
                g = traj_raw[mask]
                if g.empty:
                    continue
                srows.append(
                    {
                        "split": split_name,
                        "n_rollouts": float(len(g)),
                        "success_rate": float(np.nanmean(g["success"])),
                        "reward_outcome_mean": float(np.nanmean(g["reward_outcome"])),
                        "response_valid_tokens_mean": float(np.nanmean(g["response_valid_tokens"])),
                        "offpolicy_ratio_mean": float(np.nanmean(g["offpolicy_ratio"])),
                        "exp_tokens_mean": float(np.nanmean(g["exp_tokens"])),
                    }
                )
            if srows:
                traj_post_summary_df = pd.DataFrame(srows)
                traj_post_summary_df.to_csv(compiled_dir / "traj_post_summary.csv", index=False)

    # Plots
    dfs = {"wandb+local": df}
    _plot_lines(
        figs_dir / "reward_ma.png",
        dfs,
        ["critic/rewards/mean", "diag/reward_onpolicy_mean"],
        title=f"{args.run_id}: reward (MA w={args.ma})",
        ylabel="reward",
        ma=args.ma,
    )
    _plot_lines(
        figs_dir / "entropy_split_ma.png",
        dfs,
        [
            "actor/entropy_loss",
            "exp_replay/entropy_llm_onpolicy_mean",
            "exp_replay/entropy_llm_offpolicy_mean",
            "diag/entropy_onpolicy_token_mean",
            "diag/entropy_offpolicy_token_mean",
        ],
        title=f"{args.run_id}: entropy (split) (MA w={args.ma})",
        ylabel="entropy",
        ma=args.ma,
    )
    _plot_lines(
        figs_dir / "replay_intensity_ma.png",
        dfs,
        [
            "exp_replay/offpolicy_rollout_ratio",
            "exp_replay/offpolicy_token_ratio_llm",
            "exp_replay/offpolicy_token_ratio_response",
            "diag/exp_token_ratio",
            "diag/offpolicy_sample_ratio",
        ],
        title=f"{args.run_id}: replay intensity (MA w={args.ma})",
        ylabel="ratio",
        ma=args.ma,
    )
    _plot_lines(
        figs_dir / "staleness_age_ma.png",
        dfs,
        ["exp_replay/offpolicy_age_mean", "exp_replay/offpolicy_age_max", "exp_replay/offpolicy_age_min"],
        title=f"{args.run_id}: off-policy age (MA w={args.ma})",
        ylabel="age(step)",
        ma=args.ma,
    )
    _plot_lines(
        figs_dir / "adv_ma.png",
        dfs,
        ["diag/adv_onpolicy_token_mean", "diag/adv_self_off_token_mean", "exp_replay_diag/adv/on/mean", "exp_replay_diag/adv/off/mean"],
        title=f"{args.run_id}: advantage (on vs off) (MA w={args.ma})",
        ylabel="adv",
        ma=args.ma,
    )
    _plot_lines(
        figs_dir / "ratio_diag_ma.png",
        dfs,
        [
            "exp_replay_diag/importance_ratio/off/mean",
            "exp_replay_diag/importance_ratio/off/p99",
            "exp_replay_diag/importance_ratio/off/max",
            "exp_replay_diag/importance_ratio_shaped/off/mean",
            "exp_replay_diag/importance_ratio_shaped/off/p99",
        ],
        title=f"{args.run_id}: importance ratio diagnostics (MA w={args.ma})",
        ylabel="ratio",
        ma=args.ma,
    )

    # Simple early-vs-post summaries
    lo = int(df["_step"].min()) if len(df) else 1
    hi = int(df["_step"].max()) if len(df) else args.max_step
    if t_on is None:
        pre_lo, pre_hi = lo, min(hi, 10)
        post_lo, post_hi = min(hi, 11), hi
    else:
        pre_lo, pre_hi = lo, max(lo, t_on - 1)
        post_lo, post_hi = max(lo, t_on), hi

    seg_cols = [
        "critic/rewards/mean",
        "actor/entropy_loss",
        "actor/kl_loss",
        "exp_replay/offpolicy_token_ratio_llm",
        "exp_replay/offpolicy_rollout_ratio",
        "exp_replay/offpolicy_age_mean",
        "exp_replay/entropy_llm_onpolicy_mean",
        "exp_replay/entropy_llm_offpolicy_mean",
        "diag/exp_token_ratio",
        "diag/offpolicy_sample_ratio",
        "diag/entropy_onpolicy_token_mean",
        "diag/entropy_offpolicy_token_mean",
        "diag/adv_onpolicy_token_mean",
        "diag/adv_self_off_token_mean",
    ]
    pre = _seg_mean(df, seg_cols, pre_lo, pre_hi)
    post = _seg_mean(df, seg_cols, post_lo, post_hi)
    seg_df = pd.DataFrame(
        [
            {"segment": f"pre (step {pre_lo}-{pre_hi})", **pre},
            {"segment": f"post (step {post_lo}-{post_hi})", **post},
        ]
    )
    seg_df.to_csv(compiled_dir / "pre_post_summary.csv", index=False)

    # Correlations (post only, where replay exists)
    post_df = df[(df["_step"] >= post_lo) & (df["_step"] <= post_hi)].copy()
    corr_rows = []
    for a, b in [
        ("diag/exp_token_ratio", "diag/entropy_onpolicy_token_mean"),
        ("diag/exp_token_ratio", "diag/entropy_offpolicy_token_mean"),
        ("exp_replay/offpolicy_token_ratio_llm", "exp_replay/entropy_llm_onpolicy_mean"),
        ("exp_replay/offpolicy_token_ratio_llm", "exp_replay/entropy_llm_offpolicy_mean"),
        ("exp_replay/offpolicy_age_mean", "exp_replay_diag/importance_ratio/off/p99"),
        ("exp_replay/offpolicy_age_mean", "exp_replay_diag/importance_ratio/off/max"),
        ("exp_replay/entropy_llm_offpolicy_mean", "exp_replay_diag/importance_ratio/off/p99"),
    ]:
        corr_rows.append({"a": a, "b": b, "corr": _corr(post_df.get(a, pd.Series(dtype=float)), post_df.get(b, pd.Series(dtype=float)))})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(compiled_dir / "post_corrs.csv", index=False)

    report_path = Path(args.report_path.strip() or f"docs/analysis/2026-01-14_{args.run_id}_early34_analysis.md")
    _safe_mkdir(report_path.parent)

    def rel(p: Path) -> str:
        return os.path.relpath(p, start=report_path.parent)

    fig_paths = {p.stem: rel(p) for p in figs_dir.glob("*.png")}

    md: List[str] = []
    md.append(f"# 2026-01-14：`{args.run_id}` 前 {args.max_step} 步 Endogenous Replay 机制分析（early-step）\n\n")
    md.append("本报告只使用 **34 步早期数据**（服务器中断前），目标不是下结论，而是做 **机制信号探测**：replay 何时真正“点亮”、它影响熵/优势/年龄/ratio 的路径是什么、以及下一轮重跑最值得优先验证哪些优化点。\n\n")

    md.append("## 1. 数据来源与对齐\n")
    md.append(f"- **W&B**: run `{args.run_id}`（已拉取到本地 compiled CSV）\n")
    if df_local is not None:
        md.append(f"- **本地轨迹诊断**: `{args.local_diag_csv}`（按 step 聚合的 token/sample 级诊断）\n")
    md.append("\n")

    md.append("## 2. Replay 何时真正开始起作用？（关键事件点）\n")
    md.append(f"- **W&B 检测**（`exp_replay/offpolicy_rollout_ratio>0`）：{t_on_w}\n")
    md.append(f"- **本地诊断检测**（`diag/offpolicy_sample_ratio>0`）：{t_on_l}\n")
    md.append(f"- **综合认为 replay 开始 step**：{t_on}\n\n")
    md.append("解释：在 early 阶段 replay 需要先在 pool 里积累到可复用的成功轨迹，因此即便配置了 `replay_start_ratio`，也可能出现“**配置允许 replay，但实际要到更后面才有 off-policy 样本**”的现象。\n\n")

    md.append("## 3. 可视化（只看前 34 步）\n")
    for stem, title in [
        ("reward_ma", "3.1 reward：训练曲线与 on-policy reward（若有）"),
        ("entropy_split_ma", "3.2 熵拆分：on-policy vs off-policy（W&B + 本地诊断）"),
        ("replay_intensity_ma", "3.3 replay 强度：rollout 比例 & token 比例"),
        ("staleness_age_ma", "3.4 staleness：off-policy age"),
        ("adv_ma", "3.5 advantage：on vs off（用于 baseline 污染/学习信号诊断）"),
        ("ratio_diag_ma", "3.6 importance ratio（如果该 run 记录了 exp_replay_diag）"),
        ("traj_on_off_success_reward_ma", "3.7 轨迹级：on-policy vs off-policy 的 success/reward"),
        ("traj_on_off_token_ma", "3.8 轨迹级：on vs off 的 response token 与 exp token"),
    ]:
        p = fig_paths.get(stem)
        if p:
            md.append(f"### {title}\n\n")
            md.append(f"![{stem}]({p})\n\n")

    md.append("## 4. Early vs Post（以 replay 开始 step 为分界）的均值对比\n\n")
    md.append(df_to_markdown_table(seg_df, floatfmt=".6f"))
    md.append("\n\n")

    md.append("## 5. 34 步内已经出现的“机制信号”与解读（数据 + 数学视角）\n")
    md.append("### 5.1 小剂量 replay 也能明显抬升熵（重要）\n")
    md.append(
        "从本地诊断看，replay 启动后 `diag/exp_token_ratio` 只有约 **0.3%–1%**，但它与 **on-policy token 熵**存在可见同向相关（post 段 `corr(diag/exp_token_ratio, diag/entropy_onpolicy_token_mean)`）。\n"
        "这支持一个关键观点：**replay 的作用不只是“off-policy token 自己更散”，而可能通过共享参数的梯度把 on-policy 分布也一起“摊开”**。\n\n"
    )
    md.append("数学上可以用一个简化的混合梯度来解释（忽略 clip/baseline 等实现细节）：\n\n")
    md.append(
        r"$$\n"
        r"\nabla_\theta J(\theta)\;\approx\;"
        r"\mathbb{E}_{D_{\text{on}}}[\nabla\log\pi_\theta\cdot \hat A]"
        r"+\lambda\,\mathbb{E}_{D_{\text{replay}}}[\nabla\log\pi_\theta\cdot \hat A\cdot g(w)]\n"
        r"$$"
        "\n\n"
    )
    md.append(
        "当 replay 开始提供“当前策略下概率偏低但能成功”的 token/动作时，第二项会对这些区域产生增益，从宏观上表现为 **熵上升（上冲）**；随后 on-policy 学习把概率质量集中到更确定的模式上，又会 **熵回落**。\n\n"
    )

    md.append("### 5.2 off-policy 熵显著高于 on-policy：更像“探索注入”而非“收缩模板”\n")
    md.append(
        "在这 34 步里，post 段本地诊断显示 `diag/entropy_offpolicy_token_mean` 明显高于 `diag/entropy_onpolicy_token_mean`。这通常意味着 replay 样本更“分散/不确定”，更符合“探索注入”的解释。\n"
        "但要注意：**这也可能是 ratio/staleness 导致的高方差信号**，需要结合 importance ratio 的尾部（p99/max）一起看。\n\n"
    )

    md.append("### 5.3 staleness 已经可见：age 均值在十几步量级\n")
    md.append(
        "W&B 里 `exp_replay/offpolicy_age_mean` 在 post 段已经到 **十几步**量级（相对 34 步总长已经不算小）。这说明即使是短跑，staleness 也会出现，后续重跑可以更系统地验证 age 与 ratio 尾部/熵波动之间的关系。\n\n"
    )

    if traj_step_df is not None:
        md.append("### 5.4 轨迹级对比：off-policy 样本更“成功/更短/更确定”还是更“探索”？\n")
        md.append(
            "我们可以直接用本地 `trajectories_step_*.jsonl` 将每步 64 条 rollout 按 `diag.offpolicy_ratio>0` 分成 on/off 两组，查看 success/reward/长度等差异。\n"
            "这能直接回答一个关键机制问题：**replay 注入到底是在“喂高质量成功样本”（可能带来 baseline 污染）还是在“引入更分散的探索样本”（可能抬升熵）**。\n\n"
        )
        if traj_post_summary_df is not None and len(traj_post_summary_df):
            md.append("下面是（按推断 replay 开始 step 之后的）轨迹级均值对比：\n\n")
            md.append(df_to_markdown_table(traj_post_summary_df, floatfmt=".6f"))
            md.append("\n\n")

    md.append("## 6. 基于这 34 步就能提出的“可操作优化点”（优先级排序）\n")
    md.append(
        "- **优化点 A（最高优先）**：做 **age-aware weighting / sampling**（例如对 age 做指数衰减权重，或限制 max-age）。理由：短跑里 age 已到十几步，且这是 replay 不稳的首要来源。\n"
        "- **优化点 B**：把 replay 启动从“配置比例”改为“**池内有效样本阈值**”（例如 pool 中 solved 轨迹数 / 覆盖任务数达到阈值才开始注入），避免前期 batch 里 off-policy 极少却带来高方差。\n"
        "- **优化点 C**：若后续看到 `importance_ratio/off/p99|max` 很大，优先尝试更强的 **policy shaping/clip**（或 ratio-dependent damping），把尾部压住再谈更高 exp_ratio。\n"
        "- **优化点 D**：用 `exp_replay/entropy_llm_onpolicy_mean` 判断熵上冲是否真的发生在 on-policy；如果只在 off-policy，上冲可能是“数据侧更散”而非策略探索增强。\n\n"
    )

    md.append("## 7. 附：post 段相关性（用于机制假说筛选）\n\n")
    md.append(df_to_markdown_table(corr_df, floatfmt=".6f"))
    md.append("\n\n")

    md.append("## 8. 下一步：你重跑后我会重点补齐的证据链\n")
    md.append(
        "- replay-on vs replay-off 对照（你计划重跑的 baseline）\n"
        "- importance ratio 的尾部分布随 age 变化（staleness → ratio 尾部 → 熵/kl 波动）\n"
        "- advantage(on) 是否系统性变差（baseline 污染信号）\n"
        "- 熵上冲主要发生在 on-policy 还是 off-policy token（探索 vs 方差）\n"
    )

    report_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Outputs: {out_dir}")


if __name__ == "__main__":
    main()

