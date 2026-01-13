#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Endogenous (self-generated) experience replay analysis for AlfWorld GRPO (ExGRPO-style).

We currently have no strict "pure GRPO without replay" baseline in W&B for bz=8, rollout=8, no-teacher.
So we analyze via:
1) within-run quasi-control: pre vs post replay_start_ratio (e.g., step<10 vs >=10)
2) cross-run dose comparison: exp_ratio / replay_start_ratio variants

Outputs:
- analysis/endogenous_exp_replay_compare/out/figs/*.png
- analysis/endogenous_exp_replay_compare/out/compiled/*.csv
- docs/analysis/2026-01-13_exgrpo_endogenous_experience_replay_analysis.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    replay_start_ratio: float
    exp_ratio: float


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
        # Robust step column handling
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
    if df.index.name is None:
        df2 = df.copy()
        df2.insert(0, "label", df2.index.astype(str))
    else:
        df2 = df.reset_index()

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


def _plot_lines(out_png: Path, dfs: Dict[str, pd.DataFrame], y_col: str, title: str, ylabel: str, ma: int) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for label, df in dfs.items():
        if y_col not in df.columns:
            continue
        d = df[["_step", y_col]].dropna().sort_values("_step")
        if d.empty:
            continue
        y = _moving_avg(d[y_col].astype(float), ma)
        plt.plot(d["_step"], y, label=label, linewidth=2.0, alpha=0.9)

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _safe_mkdir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_scatter(out_png: Path, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    import matplotlib.pyplot as plt

    d = df[[x_col, y_col]].dropna()
    if d.empty:
        return
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(d[x_col].astype(float), d[y_col].astype(float), s=18, alpha=0.7)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _safe_mkdir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def seg_stats(df: pd.DataFrame, cols: List[str], lo: int, hi: int) -> Dict[str, float]:
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
    ap.add_argument("--entity", default="qisheng001-nanyang-technological-university-singapore")
    ap.add_argument("--project", default="agentevolver")
    ap.add_argument("--ma", type=int, default=10)
    ap.add_argument("--out_dir", default="analysis/endogenous_exp_replay_compare/out")
    ap.add_argument("--report_path", default="docs/analysis/2026-01-13_exgrpo_endogenous_experience_replay_analysis.md")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    compiled_dir = out_dir / "compiled"
    _safe_mkdir(figs_dir)
    _safe_mkdir(compiled_dir)

    # Known no-teacher, bz=8, rollout=8 runs (all exp_replay enabled).
    # Note: some historical runs have empty/partial W&B history; we only keep
    # runs with sufficient logged steps for robust analysis.
    runs = [
        RunSpec("w1uq88b8", "ER exp_ratio=0.5, start=0.10 (main)", replay_start_ratio=0.10, exp_ratio=0.50),
        RunSpec("bpqipx70", "ER exp_ratio=0.25, start=0.10", replay_start_ratio=0.10, exp_ratio=0.25),
    ]

    keys = [
        "_step",
        "training/global_step",
        "critic/rewards/mean",
        "critic/rewards_sum/mean",
        "critic/advantages/mean",
        "actor/entropy_loss",
        "actor/kl_loss",
        "actor/pg_loss",
        "actor/ppo_kl",
        # experience replay runtime stats
        "exp_replay/num_experience_tasks",
        "exp_replay/num_offpolicy_trajectories",
        "exp_replay/total_tasks_in_pool",
        "exp_replay/skip_uid_set_size",
        "exp_mask_ratio",
        # validation (sparse)
        "val-aux/alfworld/reward/mean@1",
        "val-aux/alfworld/reward/mean@2",
        "val-aux/alfworld/reward/mean@3",
        "val-aux/alfworld/reward/best@64/mean",
    ]

    dfs: Dict[str, pd.DataFrame] = {}
    for rs in runs:
        df = load_wandb_history(rs.run_id, keys, entity=args.entity, project=args.project)
        df["label"] = rs.label
        df.to_csv(compiled_dir / f"{rs.run_id}.csv", index=False)
        dfs[rs.label] = df

    plots = [
        ("critic/rewards/mean", "Training reward: critic/rewards/mean", "reward_mean"),
        ("actor/entropy_loss", "Actor entropy_loss (proxy for entropy)", "entropy_loss"),
        ("actor/kl_loss", "Actor KL loss", "kl_loss"),
        ("actor/pg_loss", "Actor PG loss", "pg_loss"),
        ("exp_mask_ratio", "exp_mask_ratio (NOTE: includes padding; may look tiny)", "exp_mask_ratio"),
        ("exp_replay/total_tasks_in_pool", "Replay pool size: total_tasks_in_pool", "pool_size"),
        ("exp_replay/skip_uid_set_size", "skip_uid_set size (fully-solved tasks)", "skip_uid_set_size"),
    ]
    for col, title, stem in plots:
        _plot_lines(figs_dir / f"{stem}_ma.png", dfs, col, f"{title} (MA w={args.ma})", col, args.ma)

    # Within-run pre/post analysis for main run
    main_rs = runs[0]
    d0 = dfs[main_rs.label].copy()
    total_steps = int(d0["_step"].max()) if "_step" in d0.columns else 100
    start_step = max(1, int(np.floor(main_rs.replay_start_ratio * total_steps)))
    pre_lo, pre_hi = 1, max(1, start_step)
    post_lo, post_hi = max(pre_hi + 1, start_step + 1), total_steps

    seg_cols = [
        "critic/rewards/mean",
        "actor/entropy_loss",
        "actor/kl_loss",
        "exp_replay/num_experience_tasks",
        "exp_replay/num_offpolicy_trajectories",
        "exp_replay/total_tasks_in_pool",
        "exp_mask_ratio",
    ]
    pre = seg_stats(d0, seg_cols, pre_lo, pre_hi)
    post = seg_stats(d0, seg_cols, post_lo, post_hi)
    seg_df = pd.DataFrame(
        [
            {"segment": f"pre (step {pre_lo}-{pre_hi})", **pre},
            {"segment": f"post (step {post_lo}-{post_hi})", **post},
        ]
    ).set_index("segment")
    seg_df.to_csv(compiled_dir / f"{main_rs.run_id}_pre_post.csv")

    if "exp_replay/total_tasks_in_pool" in d0.columns and "critic/rewards/mean" in d0.columns:
        _plot_scatter(
            figs_dir / "scatter_reward_vs_pool.png",
            d0,
            x_col="exp_replay/total_tasks_in_pool",
            y_col="critic/rewards/mean",
            title=f"{main_rs.run_id}: reward vs pool_size (raw points)",
        )

    report_path = Path(args.report_path)
    _safe_mkdir(report_path.parent)

    def rel(p: Path) -> str:
        return os.path.relpath(p, start=report_path.parent)

    fig_rel = {p.stem: rel(p) for p in figs_dir.glob("*.png")}

    summary_rows = []
    for rs in runs:
        df = dfs[rs.label]
        last = df.sort_values("_step").iloc[-1]
        summary_rows.append(
            {
                "label": rs.label,
                "run_id": rs.run_id,
                "replay_start_ratio": rs.replay_start_ratio,
                "exp_ratio": rs.exp_ratio,
                "reward_last": float(last.get("critic/rewards/mean", np.nan)),
                "entropy_loss_last": float(last.get("actor/entropy_loss", np.nan)),
                "kl_loss_last": float(last.get("actor/kl_loss", np.nan)),
                "pool_size_last": float(last.get("exp_replay/total_tasks_in_pool", np.nan)),
                "skip_uid_set_size_last": float(last.get("exp_replay/skip_uid_set_size", np.nan)),
                "exp_mask_ratio_last": float(last.get("exp_mask_ratio", np.nan)),
                "val_reward_mean@1_last": float(last.get("val-aux/alfworld/reward/mean@1", np.nan)),
                "val_reward_mean@2_last": float(last.get("val-aux/alfworld/reward/mean@2", np.nan)),
                "val_reward_mean@3_last": float(last.get("val-aux/alfworld/reward/mean@3", np.nan)),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("label")
    summary.to_csv(compiled_dir / "summary.csv")

    md: List[str] = []
    md.append("# 2026-01-13：Self-generated Experience Replay（Endogenous, ExGRPO-style）机制分析报告\n\n")
    md.append("## 0. 背景与目标\n")
    md.append("本报告分析 **self-generated experience replay（endogenous experience）** 在 AlfWorld + GRPO 中为什么有效，以及它可能的缺陷与可改进点。\n\n")
    md.append("重要现实约束：当前 W&B 中 **bz=8, rollout=8, no-teacher** 的 runs 只看到开启 replay 的版本，缺少严格的“关 replay 的纯 GRPO”对照。因此我们采用：\n")
    md.append("- **准对照 1（within-run）**：同一 run 内 `replay_start_ratio` 前后（pre vs post）对比\n")
    md.append("- **准对照 2（dose comparison）**：跨 run 对比不同 `exp_ratio / replay_start_ratio`\n\n")

    md.append("## 1. 实验与 run 列表\n")
    for rs in runs:
        md.append(f"- **{rs.label}**：run id `{rs.run_id}`（exp_ratio={rs.exp_ratio}, replay_start_ratio={rs.replay_start_ratio}）\n")
    md.append("\n")

    md.append("## 2. 核心量化 summary（以最后一步为代表）\n\n")
    md.append(df_to_markdown_table(summary, floatfmt=".6f"))
    md.append("\n\n")

    md.append("## 3. 核心可视化（MA）\n")

    def img(stem: str, title: str) -> None:
        path = fig_rel.get(stem, None)
        if path:
            md.append(f"### {title}\n\n")
            md.append(f"![{stem}]({path})\n\n")

    img("reward_mean_ma", "3.1 训练 reward（critic/rewards/mean）")
    img("entropy_loss_ma", "3.2 entropy_loss（探索/收缩 proxy）")
    img("kl_loss_ma", "3.3 KL loss（更新幅度 proxy）")
    img("exp_mask_ratio_ma", "3.4 exp_mask_ratio（注意：包含 padding，数值会偏小）")
    img("pool_size_ma", "3.5 replay pool size（total_tasks_in_pool）")
    img("skip_uid_set_size_ma", "3.6 fully-solved tasks（skip_uid_set_size）")
    img("scatter_reward_vs_pool", "3.7 within-run：reward 与 pool size 的关系（散点）")

    md.append("## 4. Within-run 准对照：w1uq88b8 在 replay_start_ratio 前后的均值对比\n\n")
    md.append(
        f"以 `{main_rs.run_id}` 为例：配置 `replay_start_ratio={main_rs.replay_start_ratio}`，对应约在 **step≈{start_step}** 之后开始出现 experience tasks。\n\n"
    )
    md.append(df_to_markdown_table(seg_df, floatfmt=".6f"))
    md.append("\n\n")

    md.append("## 5. 数学化机制解释：为什么 ExGRPO-style replay 可能有效？\n")
    md.append("### 5.1 视角 A：把 replay 看成“分布工程（distribution engineering）”\n")
    md.append(
        "对每个 task 的一组 rollouts（大小固定为 n_rollout），replay 机制把历史成功轨迹（off-policy）混入当前 batch，使得组内 reward 分布更偏向成功区域，从而提升学习信号的 **信噪比** 与 **样本效率**。\n\n"
    )
    md.append("### 5.2 视角 B：重要性采样与 policy shaping（ExGRPO 的关键稳定化）\n")
    md.append("off-policy token 的重要性比率可写为：\n\n")
    md.append(r"$$w(\theta)=\exp(\log\pi_\theta(a|s)-\log\pi_{\text{old}}(a|s))=\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$$")
    md.append(
        "\n\n为了避免 \(w\) 的高方差导致训练不稳，ExGRPO 使用 policy shaping（例如 \(f(w)=\\frac{w}{w+\\beta}\\in(0,1)\)）来让 off-policy 权重有界，同时放大低概率信号、抑制高概率信号，从而在“学到成功经验”和“保持熵/探索”之间做折中。\n\n"
    )

    md.append("## 6. 机制缺陷与潜在风险（为什么它也可能失败/不够稳）\n")
    md.append(
        "- **缺陷 1：分组 baseline 污染（结构性风险）**：在 GRPO 的组内相对优势结构下，强成功轨迹会抬高 baseline，使探索性 on-policy 行为更容易变成负优势，从而压制长尾修复。\n"
    )
    md.append(
        "- **缺陷 2：经验陈旧（staleness）**：replay 轨迹来自旧 policy（policy_version 落后当前 step），会带来分布偏移与 \(w(\\theta)\) 方差上升；若不监控 ratio 与轨迹年龄，训练风险不可见。\n"
    )
    md.append(
        "- **缺陷 3：选择偏置（entropy argmin）**：低熵成功轨迹更“模板化”，可能降低多样性，长期不利于鲁棒性与错误恢复。\n"
    )
    md.append(
        "- **缺陷 4：`exp_mask_ratio` 可解释性不足**：它对 full sequence（含 padding）求均值，数值很小不代表 replay 没起作用；更需要“只在有效 response/LLM token 上归一化”的指标。\n\n"
    )

    md.append("## 7. 建议新增/补齐的诊断指标（建议你重跑时记录）\n")
    md.append("为了把 endogenous replay 的因果链条钉死，建议新增以下指标：\n")
    md.append("- **有效 token 归一化的 off-policy 比例**：`sum(exp_mask*loss_mask*response_mask)/sum(loss_mask*response_mask)`\n")
    md.append("- **经验陈旧度（age）**：记录 replayed trajectories 的 `policy_version`，并记录 `age = global_step - policy_version` 的 mean/max。\n")
    md.append("- **重要性比率分布（off-policy）**：记录 \(w\) 与 shaping 后 \(f(w)\) 的 mean/std/max、`P(w>10)`。\n")
    md.append("- **组内 baseline 污染强度（experience tasks）**：同一 group 内 `mean_reward_all - mean_reward_onpolicy`。\n")
    md.append("- **pool difficulty 分布**：对 `difficulty2task_dict` 每个 bucket（0..n_rollout）记录 count。\n\n")

    md.append("## 8. 面向统一框架（Exogenous + Endogenous）的下一步方向\n")
    md.append("更 solid 的 experiential GRPO 框架，应统一两类经验到同一套“可控注入 + 可解释诊断”机制：\n")
    md.append("- **Exogenous（teacher）**：强、低方差，但易 baseline 污染/探索受限 → baseline 分离 + 自适应退火\n")
    md.append("- **Endogenous（self-generated）**：更贴近 \(D_\\pi\)，利于长尾修复，但受陈旧度与重要性方差影响 → age-aware weighting + ratio diagnostics + 多样性约束\n\n")

    report_path.write_text("".join(md), encoding="utf-8")

    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Figures: {figs_dir}")
    print(f"[OK] Compiled CSVs: {compiled_dir}")


if __name__ == "__main__":
    main()

