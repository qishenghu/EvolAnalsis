#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare LUFFY no-logprob baseline vs improvements (teacher-baseline-sep / adaptive gating / both).

Data sources:
- W&B history: reward / losses / diag/teacher_loss_scale etc.
- Local Trajectory: batch_diag_step_*.json (robust for many diag/* metrics).

Outputs (default):
- analysis/luffy_no_logprob_improvement_compare/out/compiled/*.csv
- analysis/luffy_no_logprob_improvement_compare/out/figs/*.png
- docs/analysis/2026-01-13_luffy_no_logprob_improvements_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_id: str
    traj_dir: Path


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _moving_avg(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).mean()


def df_to_markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """
    Minimal DataFrame -> Markdown table without optional dependency (tabulate).
    - Includes index as the first column (named 'label' if index has no name).
    - Formats floats with floatfmt.
    """
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


def load_local_batch_diag(traj_dir: Path) -> pd.DataFrame:
    """
    Load batch-level diagnostic JSON files: batch_diag_step_{k}.json
    Returns DF with columns: _step + diag/* keys
    """
    pat = re.compile(r"batch_diag_step_(\d+)\.json$")
    rows: List[Dict] = []
    for fp in sorted(traj_dir.glob("batch_diag_step_*.json")):
        m = pat.search(fp.name)
        if not m:
            continue
        step = int(m.group(1))
        with fp.open("r", encoding="utf-8") as f:
            d = json.load(f)
        d["_step"] = step
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No batch_diag_step_*.json found under: {traj_dir}")
    df = pd.DataFrame(rows).sort_values("_step").reset_index(drop=True)
    return df


def load_wandb_history(
    run_id: str,
    keys: List[str],
    entity: str,
    project: str,
) -> pd.DataFrame:
    """
    Robustly load selected keys from W&B using scan_history to avoid wide tables.
    """
    import wandb  # local import: avoid hard dependency for offline use

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    rows: List[Dict] = []
    # Some W&B backends may return _step even if not requested; keep keys tight but robust.
    scan_keys = [k for k in keys if k != "_step"]
    for row in run.scan_history(keys=scan_keys):
        if row is None:
            continue
        rows.append(dict(row))
    df: Optional[pd.DataFrame] = None
    if rows:
        df0 = pd.DataFrame(rows)
        # Some W&B backends may return 'step' instead of '_step'
        if "_step" not in df0.columns and "step" in df0.columns:
            df0 = df0.rename(columns={"step": "_step"})
        if "_step" in df0.columns:
            df = df0

    if df is None:
        # fall back to wide history if scan_history is empty OR lacks step (common for some runs)
        hist = run.history()
        keep_raw = ["_step"] + [k for k in keys if k != "_step"]
        # de-dup while preserving order
        keep = list(dict.fromkeys([c for c in keep_raw if c in hist.columns]))
        if "_step" not in keep:
            raise RuntimeError(f"W&B history for {run_id} has no _step")
        df = hist[keep].dropna(subset=["_step"]).copy()

    # guard: drop duplicate column names (can happen if _step is returned twice)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.sort_values("_step").drop_duplicates("_step", keep="last").reset_index(drop=True)
    return df


def pick_reward_column(df: pd.DataFrame) -> str:
    """
    Prefer on-policy reward for LUFFY runs; else fallback to generic reward or local diag reward.
    """
    candidates = [
        "critic/rewards_onpolicy/mean",
        "critic/rewards/mean",
        "diag/reward_onpolicy_mean",
        "diag/group_non_teacher_reward_mean",
    ]
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    raise KeyError(f"No reward column found. Available columns sample: {list(df.columns)[:30]}")


def merge_sources(wandb_df: pd.DataFrame, local_df: pd.DataFrame) -> pd.DataFrame:
    # Outer-join then prefer W&B where present for overlapping keys.
    merged = pd.merge(local_df, wandb_df, on="_step", how="outer", suffixes=("_local", "_wandb"))
    merged = merged.sort_values("_step").reset_index(drop=True)

    # If a key exists as both *_local and *_wandb, consolidate into canonical key.
    for col in list(merged.columns):
        if col.endswith("_local"):
            base = col[:-6]
            wandb_col = base + "_wandb"
            if wandb_col in merged.columns:
                merged[base] = merged[wandb_col].combine_first(merged[col])
    # keep canonical + unique non-suffixed
    drop_cols = [c for c in merged.columns if c.endswith("_local") or c.endswith("_wandb")]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def summarize_run(df: pd.DataFrame, reward_col: str) -> Dict[str, float]:
    df = df.dropna(subset=[reward_col]).copy()
    df = df.sort_values("_step")
    reward = df[reward_col].astype(float)
    steps = df["_step"].astype(int)

    def seg_mean(a: pd.Series, lo: int, hi: int) -> float:
        m = (steps >= lo) & (steps <= hi)
        if not m.any():
            return float("nan")
        return float(a[m].mean())

    best_idx = int(reward.idxmax())
    best_step = int(df.loc[best_idx, "_step"])
    return {
        "steps": float(len(df)),
        "reward_auc_mean": float(reward.mean()),
        "reward_best": float(reward.max()),
        "reward_best_step": float(best_step),
        "reward_last": float(reward.iloc[-1]),
        "reward_early_mean_1_20": seg_mean(reward, 1, 20),
        "reward_mid_mean_21_60": seg_mean(reward, 21, 60),
        "reward_late_mean_61_100": seg_mean(reward, 61, 100),
    }


def _plot_lines(
    out_png: Path,
    dfs: Dict[str, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    ma_window: Optional[int] = None,
    baseline_label: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for label, df in dfs.items():
        if y_col not in df.columns:
            continue
        d = df[["_step", y_col]].dropna().sort_values("_step")
        if d.empty:
            continue
        y = d[y_col].astype(float)
        if ma_window is not None:
            y = _moving_avg(y, ma_window)
        lw = 2.5 if (baseline_label is not None and label == baseline_label) else 2.0
        plt.plot(d["_step"], y, label=label, linewidth=lw, alpha=0.9)

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _safe_mkdir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_delta(
    out_png: Path,
    dfs: Dict[str, pd.DataFrame],
    y_col: str,
    baseline_label: str,
    title: str,
    ylabel: str,
    ma_window: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    base = dfs[baseline_label][["_step", y_col]].dropna().sort_values("_step")
    if base.empty:
        return
    if ma_window is not None:
        base_y = _moving_avg(base[y_col].astype(float), ma_window).to_numpy()
    else:
        base_y = base[y_col].astype(float).to_numpy()
    base_x = base["_step"].astype(int).to_numpy()

    plt.figure(figsize=(12, 6))
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    for label, df in dfs.items():
        if label == baseline_label:
            continue
        if y_col not in df.columns:
            continue
        d = df[["_step", y_col]].dropna().sort_values("_step")
        if d.empty:
            continue
        # align on steps intersection
        inter = np.intersect1d(base_x, d["_step"].astype(int).to_numpy())
        if inter.size == 0:
            continue
        bmask = np.isin(base_x, inter)
        dmask = np.isin(d["_step"].astype(int).to_numpy(), inter)
        by = base_y[bmask]
        dy_raw = d[y_col].astype(float).to_numpy()[dmask]
        dy = _moving_avg(pd.Series(dy_raw), ma_window).to_numpy() if ma_window is not None else dy_raw
        plt.plot(inter, dy - by, label=f"{label} - {baseline_label}", linewidth=2.0, alpha=0.9)

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    _safe_mkdir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="qisheng001-nanyang-technological-university-singapore")
    ap.add_argument("--project", default="agentevolver")
    ap.add_argument("--ma", type=int, default=10)
    ap.add_argument("--out_dir", default="analysis/luffy_no_logprob_improvement_compare/out")
    ap.add_argument("--report_path", default="docs/analysis/2026-01-13_luffy_no_logprob_improvements_report.md")
    ap.add_argument("--include_v2", action="store_true", help="Include v2 annealing runs (pciujkve/t7doz8ru).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    compiled_dir = out_dir / "compiled"
    _safe_mkdir(figs_dir)
    _safe_mkdir(compiled_dir)

    runs = [
        RunSpec(
            label="Exp-0 baseline (LUFFY no-logprob)",
            run_id="mp49ntmm",
            traj_dir=Path(
                "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random_analysis_v1/Trajectory"
            ),
        ),
        RunSpec(
            label="Exp-1 (7.1 baseline-sep)",
            run_id="bjgtsf79",
            traj_dir=Path(
                "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__grpo_teacher_baseline_sep_v1/Trajectory"
            ),
        ),
        RunSpec(
            label="Exp-2 (7.2 adaptive gate)",
            run_id="ksy1eyh3",
            traj_dir=Path(
                "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__teacher_adaptive_gate_v1/Trajectory"
            ),
        ),
        RunSpec(
            label="Exp-3 (7.1 + 7.2)",
            run_id="0v8ecp6h",
            traj_dir=Path(
                "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__baseline_sep_plus_adaptive_gate_v1/Trajectory"
            ),
        ),
    ]

    if args.include_v2:
        runs.extend(
            [
                RunSpec(
                    label="Exp-2 v2 (7.2 adaptive gate annealed)",
                    run_id="pciujkve",
                    traj_dir=Path(
                        "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                        "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__teacher_adaptive_gate_v2/Trajectory"
                    ),
                ),
                RunSpec(
                    label="Exp-3 v2 (7.1 + 7.2 annealed)",
                    run_id="t7doz8ru",
                    traj_dir=Path(
                        "/home/qisheng/agent/AgentEvolver/checkpoints/agentevolver/"
                        "alfworld_3b_grpo_teacher72b_only_bz8_mix1_no_logprob_random__baseline_sep_plus_adaptive_gate_v2/Trajectory"
                    ),
                ),
            ]
        )

    # W&B keys: keep tight, but include fallbacks.
    wandb_keys = [
        "_step",
        # rewards
        "critic/rewards_onpolicy/mean",
        "critic/rewards/mean",
        # actor losses
        "actor/entropy_loss",
        "actor/kl_loss",
        "actor/pg_loss",
        # diag that may only exist in wandb (adaptive gate)
        "diag/teacher_loss_scale",
        "diag/teacher_gap_used",
        "diag/teacher_gap_ema",
        # diag also in local files (for cross-check)
        "diag/group_teacher_minus_on_reward_mean",
        "diag/reward_onpolicy_mean",
        "diag/reward_teacher_mean",
        "diag/adv_onpolicy_token_mean",
        "diag/adv_teacher_token_mean",
        "diag/entropy_onpolicy_token_mean",
        "diag/entropy_teacher_token_mean",
        "diag/onpolicy_adv_pos_ratio",
        "diag/teacher_token_ratio",
        "diag/group_non_teacher_reward_mean",
        "diag/group_all_reward_mean",
    ]

    merged_by_label: Dict[str, pd.DataFrame] = {}
    reward_col_by_label: Dict[str, str] = {}
    summaries: List[Dict] = []

    for rs in runs:
        local_df = load_local_batch_diag(rs.traj_dir)
        wandb_df = load_wandb_history(rs.run_id, wandb_keys, entity=args.entity, project=args.project)
        df = merge_sources(wandb_df, local_df)
        df = df.sort_values("_step").reset_index(drop=True)
        df["label"] = rs.label
        merged_by_label[rs.label] = df

        reward_col = pick_reward_column(df)
        reward_col_by_label[rs.label] = reward_col

        s = summarize_run(df, reward_col)
        s["label"] = rs.label
        s["run_id"] = rs.run_id
        s["reward_col"] = reward_col
        summaries.append(s)

        df.to_csv(compiled_dir / f"{rs.run_id}_merged.csv", index=False)

    summary_df = pd.DataFrame(summaries).set_index("label")
    summary_df.to_csv(compiled_dir / "summary.csv")

    # Use baseline's reward column as canonical if possible; otherwise plot per-run (less comparable).
    baseline_label = runs[0].label
    canonical_reward_col = reward_col_by_label[baseline_label]

    # If some runs don't have canonical reward column, backfill per-row from their own selected reward into reward_canon.
    dfs_for_plot: Dict[str, pd.DataFrame] = {}
    for label, df in merged_by_label.items():
        df = df.copy()
        if canonical_reward_col not in df.columns or df[canonical_reward_col].isna().all():
            df["reward_canon"] = df[reward_col_by_label[label]]
        else:
            df["reward_canon"] = df[canonical_reward_col]
        dfs_for_plot[label] = df

    # Figures: reward + deltas
    _plot_lines(
        figs_dir / "reward_mean_ma.png",
        dfs_for_plot,
        y_col="reward_canon",
        title=f"Reward (moving avg w={args.ma})",
        ylabel="reward",
        ma_window=args.ma,
        baseline_label=baseline_label,
    )
    _plot_delta(
        figs_dir / "reward_delta_vs_baseline_ma.png",
        dfs_for_plot,
        y_col="reward_canon",
        baseline_label=baseline_label,
        title=f"Reward delta vs baseline (moving avg w={args.ma})",
        ylabel="reward_delta",
        ma_window=args.ma,
    )

    # Core causal chain diags (prefer local)
    core_diags = [
        ("diag/group_teacher_minus_on_reward_mean", "Teacher minus on-policy reward (gap)"),
        ("diag/adv_onpolicy_token_mean", "On-policy token advantage mean"),
        ("diag/entropy_onpolicy_token_mean", "On-policy token entropy mean"),
    ]
    for col, title in core_diags:
        _plot_lines(
            figs_dir / f"{col.replace('/', '__')}_ma.png",
            dfs_for_plot,
            y_col=col,
            title=f"{title} (moving avg w={args.ma})",
            ylabel=col,
            ma_window=args.ma,
            baseline_label=baseline_label,
        )
        _plot_delta(
            figs_dir / f"{col.replace('/', '__')}_delta_vs_baseline_ma.png",
            dfs_for_plot,
            y_col=col,
            baseline_label=baseline_label,
            title=f"{title}: delta vs baseline (moving avg w={args.ma})",
            ylabel=f"{col}_delta",
            ma_window=args.ma,
        )

    # Adaptive gate: only meaningful for Exp-2/3, but plot anyway (missing will be skipped).
    _plot_lines(
        figs_dir / "diag__teacher_loss_scale_ma.png",
        dfs_for_plot,
        y_col="diag/teacher_loss_scale",
        title=f"Adaptive gate: diag/teacher_loss_scale (moving avg w={args.ma})",
        ylabel="teacher_loss_scale",
        ma_window=args.ma,
        baseline_label=baseline_label,
    )
    _plot_lines(
        figs_dir / "diag__teacher_gap_used_ma.png",
        dfs_for_plot,
        y_col="diag/teacher_gap_used",
        title=f"Adaptive gate: diag/teacher_gap_used (moving avg w={args.ma})",
        ylabel="teacher_gap_used",
        ma_window=args.ma,
        baseline_label=baseline_label,
    )

    # Write markdown report to docs (Chinese).
    report_path = Path(args.report_path)
    _safe_mkdir(report_path.parent)

    def rel_from_docs(p: Path) -> str:
        # docs/analysis/... -> ../../analysis/...
        # assume report lives under docs/analysis
        return os.path.relpath(p, start=report_path.parent)

    figs_rel = {k: rel_from_docs(v) for k, v in {
        "reward": figs_dir / "reward_mean_ma.png",
        "reward_delta": figs_dir / "reward_delta_vs_baseline_ma.png",
        "gap": figs_dir / "diag__group_teacher_minus_on_reward_mean_ma.png",
        "gap_delta": figs_dir / "diag__group_teacher_minus_on_reward_mean_delta_vs_baseline_ma.png",
        "adv": figs_dir / "diag__adv_onpolicy_token_mean_ma.png",
        "adv_delta": figs_dir / "diag__adv_onpolicy_token_mean_delta_vs_baseline_ma.png",
        "ent": figs_dir / "diag__entropy_onpolicy_token_mean_ma.png",
        "ent_delta": figs_dir / "diag__entropy_onpolicy_token_mean_delta_vs_baseline_ma.png",
        "gate": figs_dir / "diag__teacher_loss_scale_ma.png",
        "gate_gap": figs_dir / "diag__teacher_gap_used_ma.png",
    }.items()}

    # Add deltas in summary
    baseline_auc = float(summary_df.loc[baseline_label, "reward_auc_mean"])
    baseline_last = float(summary_df.loc[baseline_label, "reward_last"])
    summary_df = summary_df.copy()
    summary_df["reward_auc_delta_vs_baseline"] = summary_df["reward_auc_mean"] - baseline_auc
    summary_df["reward_last_delta_vs_baseline"] = summary_df["reward_last"] - baseline_last

    md = []
    title_suffix = "（含 v2 退火复跑）" if args.include_v2 else ""
    md.append(f"# 2026-01-13：LUFFY no-logprob（baseline）vs 两项改进（7.1/7.2）综合分析报告{title_suffix}\n")
    md.append("## 0. 实验设置与对比对象\n")
    md.append("本报告对齐了 4 个 run（相同任务/训练步数/teacher 配置，差异仅来自 7.1/7.2 开关），并使用两类数据源交叉验证：\n")
    md.append("- **W&B history**：reward、actor loss、以及 `diag/teacher_loss_scale` 等（门控专用）\n")
    md.append("- **本地 Trajectory**：`batch_diag_step_*.json`（包含 gap/adv/entropy 等核心因果链指标）\n\n")

    md.append("对比对象：\n")
    for rs in runs:
        md.append(f"- **{rs.label}**：run id `{rs.run_id}`；trajectory `{rs.traj_dir}`\n")
    md.append("\n")

    md.append("## 1. 一句话结论（先结论，后证据）\n")
    md.append("- **7.1（baseline/adv teacher-separation）是否有效**：看它是否显著降低 `diag/adv_onpolicy_token_mean` 的“系统性为负”程度、并改善 late 段 reward 回落。\n")
    md.append("- **7.2（teacher loss 自适应门控）是否有效**：看 `diag/teacher_loss_scale` 是否随 `diag/teacher_gap_used` 下降而自动退火，并在 late 段释放探索（entropy 不再塌陷/adv 不再长期负）。\n")
    md.append("- **7.1+7.2 是否互补**：看二者叠加是否同时做到“中期不掉速 + 后期不塌陷”。\n\n")

    md.append("## 2. 关键量化指标（可复现表格）\n")
    md.append("说明：`reward_auc_mean` 为对齐步数上的简单平均（可视为 AUC/steps）；分段均值使用 step 区间 early=1-20, mid=21-60, late=61-100。\n\n")
    md.append(df_to_markdown_table(summary_df, floatfmt=".6f"))
    md.append("\n\n")

    md.append("## 3. 核心可视化（结论主要基于这些图）\n")
    md.append(f"### 3.1 reward 曲线（w={args.ma} 滑动平均）\n\n")
    md.append(f"![reward_ma]({figs_rel['reward']})\n\n")
    md.append(f"### 3.2 reward 相对 baseline 的差值（w={args.ma}）\n\n")
    md.append(f"![reward_delta]({figs_rel['reward_delta']})\n\n")

    md.append("### 3.3 baseline 抬高强度：`diag/group_teacher_minus_on_reward_mean`\n\n")
    md.append(f"![gap]({figs_rel['gap']})\n\n")
    md.append(f"![gap_delta]({figs_rel['gap_delta']})\n\n")

    md.append("### 3.4 探索被压制的直接证据：`diag/adv_onpolicy_token_mean`\n\n")
    md.append(f"![adv]({figs_rel['adv']})\n\n")
    md.append(f"![adv_delta]({figs_rel['adv_delta']})\n\n")

    md.append("### 3.5 熵塌陷：`diag/entropy_onpolicy_token_mean`\n\n")
    md.append(f"![ent]({figs_rel['ent']})\n\n")
    md.append(f"![ent_delta]({figs_rel['ent_delta']})\n\n")

    md.append("### 3.6 7.2 门控是否真的在工作（只在 wandb 有）：`diag/teacher_loss_scale` 与 `diag/teacher_gap_used`\n\n")
    md.append(f"![gate]({figs_rel['gate']})\n\n")
    md.append(f"![gate_gap]({figs_rel['gate_gap']})\n\n")

    md.append("## 4. 机制解释（用最小数学形式把‘为什么有效/为什么可能有副作用’说清楚）\n")
    md.append("### 4.1 7.1：teacher-separation 为什么理论上应该改善 LUFFY 的 late 回落？\n")
    md.append("在 rollout-level LUFFY 里，每个 task 组内混入 teacher（高回报）会抬高组均值 baseline，从而把 on-policy 的相对优势系统性压成负值：\n\n")
    md.append(r"令每组共 $n$ 条 rollout，其中 $k$ 条是 teacher，on-policy 平均回报 $\mu_O$，teacher 平均回报 $\mu_T$，则组均值为 $\bar R=\mu_O+\frac{k}{n}(\mu_T-\mu_O)$，on-policy 的期望优势为 $\mathbb{E}[A_O]=\mu_O-\bar R=-\frac{k}{n}(\mu_T-\mu_O)<0$。")
    md.append("\n\n7.1 的目标就是：**让 on-policy baseline 只用 on-policy 自己的均值/方差**，避免 teacher 的高回报“以小搏大”地污染所有 on-policy 的优势符号。\n\n")

    md.append("### 4.2 7.2：自适应门控为什么可能同时带来“中期加速 + 后期不塌陷”？\n")
    md.append("7.2 本质是在 teacher loss 上乘一个随 gap 变化的系数：\n\n")
    md.append(r"$$\alpha_t=\mathrm{clip}\left(\frac{\mathrm{gap}_t-\epsilon}{\tau},\,\alpha_{\min},\,\alpha_{\max}\right),\quad \mathrm{gap}_t\approx \mathbb{E}[R_T-R_\pi]$$")
    md.append("\n\n当 on-policy 很弱（gap 大）时，teacher 信号强（\u03b1≈1）帮助快速进入可行策略子空间；当 on-policy 变强（gap 小）时，teacher 自动退火（\u03b1→0）释放探索与长尾修复空间，从而缓解 late 的熵塌陷与 reward 回落。\n\n")

    md.append("## 5. 针对 ICML 算法化的下一步（从这两条改进抽象出‘新算法’）\n")
    md.append("- **建议 A（结构性）**：把 LUFFY 明确写成“双分布/双目标”的优化：on-policy 目标保持纯 GRPO 组内比较；teacher 只作为一个受控的 shaping 项，且其权重由可观测 gap 自适应决定。\n")
    md.append("- **建议 B（诊断驱动的理论叙事）**：以本报告的三条因果链指标作为算法设计动机与验证闭环：\n")
    md.append("  - baseline 抬高：`group_teacher_minus_on_reward_mean`\n")
    md.append("  - 探索惩罚：`adv_onpolicy_token_mean`\n")
    md.append("  - 熵塌陷：`entropy_onpolicy_token_mean`\n")
    md.append("  再加上门控指标：`teacher_loss_scale` / `teacher_gap_used`，形成完整因果证据链。\n\n")

    report_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] Wrote report: {report_path}")
    print(f"[OK] Figures: {figs_dir}")
    print(f"[OK] Compiled CSVs: {compiled_dir}")


if __name__ == "__main__":
    main()

