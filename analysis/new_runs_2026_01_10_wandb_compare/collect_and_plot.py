"""
Collect W&B histories for several runs and generate comparison plots + CSV exports.

This script is intended for the 2026-01-10 analysis round comparing:
- LUFFY w/o log_prob (confidence sampling): uk74oszd
- LUFFY w/  log_prob (confidence sampling): nj1g3tzx
- LUFFY w/  log_prob (random sampling):     6iuti28h
- Vanilla GRPO baseline:                    9ggix50f

Outputs:
- analysis/new_runs_2026_01_10_wandb_compare/out/histories/<run_id>.csv
- analysis/new_runs_2026_01_10_wandb_compare/out/merged.csv
- analysis/new_runs_2026_01_10_wandb_compare/out/figs/*.png
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    teacher_use_log_prob: Optional[bool]  # None for baseline
    teacher_sampling: Optional[str]  # "confidence"/"random"/None


ENTITY = os.environ.get("WANDB_ENTITY") or "qisheng001-nanyang-technological-university-singapore"
PROJECT = os.environ.get("WANDB_PROJECT") or "agentevolver"

RUNS: List[RunSpec] = [
    RunSpec("uk74oszd", "LUFFY / no_logprob / sampling=confidence", False, "confidence"),
    RunSpec("nj1g3tzx", "LUFFY / logprob / sampling=confidence", True, "confidence"),
    RunSpec("6iuti28h", "LUFFY / logprob / sampling=random", True, "random"),
    RunSpec("9ggix50f", "Vanilla GRPO", None, None),
]

OUT_DIR = Path(__file__).resolve().parent / "out"
HIST_DIR = OUT_DIR / "histories"
FIG_DIR = OUT_DIR / "figs"


def _ensure_dirs() -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if len(x) == 0:
        return x
    w = min(window, len(x))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def _pick_step_col(df: pd.DataFrame) -> str:
    for col in ["training/global_step", "trainer/global_step", "global_step", "_step"]:
        if col in df.columns:
            return col
    return "_step"


def _safe_float(v) -> float:
    try:
        if v is None:
            return float("nan")
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        return float(v)
    except Exception:
        return float("nan")


def fetch_run_history(run_id: str) -> pd.DataFrame:
    import wandb  # local dependency

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Pull full history (100 steps scale, acceptable). If it becomes huge later, restrict keys here.
    df = run.history(pandas=True)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Normalize step column + keep consistent ordering
    step_col = _pick_step_col(df)
    if step_col != "_step":
        df["_step"] = df[step_col]
    df = df.sort_values("_step").reset_index(drop=True)

    # Keep only numeric columns (+ _step); convert non-numeric to NaN to ease plotting/merge
    keep = ["_step"]
    numeric_cols = []
    for c in df.columns:
        if c == "_step":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # try coercion
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                df[c] = coerced
                numeric_cols.append(c)
    keep.extend(numeric_cols)
    return df[keep]


def export_histories() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for rs in RUNS:
        df = fetch_run_history(rs.run_id)
        dfs[rs.run_id] = df
        out = HIST_DIR / f"{rs.run_id}.csv"
        df.to_csv(out, index=False)
    return dfs


def _merge_histories(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rs in RUNS:
        df = dfs.get(rs.run_id, pd.DataFrame()).copy()
        if len(df) == 0:
            continue
        df["run_id"] = rs.run_id
        df["label"] = rs.label
        df["teacher_use_log_prob"] = rs.teacher_use_log_prob
        df["teacher_sampling"] = rs.teacher_sampling
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, axis=0, ignore_index=True)
    return merged


def _plot_lines(
    merged: pd.DataFrame,
    y: str,
    title: str,
    fname: str,
    y_label: Optional[str] = None,
    window: int = 7,
    ylim: Optional[tuple] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for rs in RUNS:
        sub = merged[merged["run_id"] == rs.run_id].sort_values("_step")
        if len(sub) == 0 or y not in sub.columns:
            continue
        x = sub["_step"].to_numpy(dtype=np.float64)
        yy = sub[y].to_numpy(dtype=np.float64)
        yy_s = _moving_average(yy, window)
        plt.plot(x, yy_s, linewidth=2, label=rs.label)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(y_label or y)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=180)
    plt.close()


def _plot_delta_vs_baseline(
    merged: pd.DataFrame,
    y: str,
    baseline_run_id: str,
    title: str,
    fname: str,
    y_label: Optional[str] = None,
    window: int = 7,
    ylim: Optional[tuple] = None,
) -> None:
    import matplotlib.pyplot as plt

    base = merged[merged["run_id"] == baseline_run_id].sort_values("_step")
    if len(base) == 0 or y not in base.columns:
        return
    base_xy = base[["_step", y]].dropna()
    if len(base_xy) == 0:
        return
    base_map = dict(zip(base_xy["_step"].astype(int).tolist(), base_xy[y].astype(float).tolist()))

    plt.figure(figsize=(10, 5))
    for rs in RUNS:
        if rs.run_id == baseline_run_id:
            continue
        sub = merged[merged["run_id"] == rs.run_id].sort_values("_step")
        if len(sub) == 0 or y not in sub.columns:
            continue
        sub_xy = sub[["_step", y]].dropna()
        if len(sub_xy) == 0:
            continue
        xs = []
        ds = []
        for step_i, val in zip(sub_xy["_step"].astype(int).tolist(), sub_xy[y].astype(float).tolist()):
            if step_i in base_map:
                xs.append(step_i)
                ds.append(val - base_map[step_i])
        if not xs:
            continue
        x = np.asarray(xs, dtype=np.float64)
        d = np.asarray(ds, dtype=np.float64)
        d_s = _moving_average(d, window)
        plt.plot(x, d_s, linewidth=2, label=rs.label)

    plt.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(y_label or f"delta({y})")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=180)
    plt.close()


def _make_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rs in RUNS:
        sub = merged[merged["run_id"] == rs.run_id].sort_values("_step")
        if len(sub) == 0:
            continue
        last = sub.dropna(subset=["critic/rewards/mean"]).tail(1)
        last_reward = float(last["critic/rewards/mean"].iloc[0]) if len(last) else float("nan")
        best_reward = float(sub["critic/rewards/mean"].max()) if "critic/rewards/mean" in sub.columns else float("nan")

        def _col_last(col: str) -> float:
            if col not in sub.columns:
                return float("nan")
            t = sub.dropna(subset=[col]).tail(1)
            return float(t[col].iloc[0]) if len(t) else float("nan")

        rows.append(
            {
                "run_id": rs.run_id,
                "label": rs.label,
                "teacher_use_log_prob": rs.teacher_use_log_prob,
                "teacher_sampling": rs.teacher_sampling,
                "steps_logged": int(sub["_step"].nunique()),
                "reward_last": last_reward,
                "reward_best": best_reward,
                "reward_onpolicy_last": _col_last("critic/rewards_onpolicy/mean"),
                "reward_teacher_last": _col_last("critic/rewards_teacher/mean"),
                "teacher_token_ratio_last": _col_last("diag/teacher_token_ratio"),
                "teacher_rollouts_last": _col_last("luffy/total_teacher_rollouts"),
                "entropy_loss_last": _col_last("actor/entropy_loss"),
                "kl_loss_last": _col_last("actor/kl_loss"),
            }
        )
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def main() -> None:
    _ensure_dirs()
    dfs = export_histories()
    merged = _merge_histories(dfs)
    merged.to_csv(OUT_DIR / "merged.csv", index=False)

    summary = _make_summary(merged)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Core reward curves
    _plot_lines(
        merged,
        y="critic/rewards/mean",
        title="Reward mean (critic/rewards/mean) vs step",
        fname="reward_mean.png",
        y_label="reward_mean",
        window=7,
        ylim=(0, 1.0),
    )
    _plot_delta_vs_baseline(
        merged,
        y="critic/rewards/mean",
        baseline_run_id="9ggix50f",
        title="Delta reward_mean vs Vanilla GRPO (positive = better)",
        fname="reward_mean_delta_vs_grpo.png",
        y_label="delta_reward_mean",
        window=7,
        ylim=(-0.4, 0.4),
    )
    if "critic/rewards_onpolicy/mean" in merged.columns:
        _plot_lines(
            merged,
            y="critic/rewards_onpolicy/mean",
            title="On-policy reward mean vs step",
            fname="reward_onpolicy_mean.png",
            y_label="reward_onpolicy_mean",
            window=7,
            ylim=(0, 1.0),
        )
        _plot_delta_vs_baseline(
            merged,
            y="critic/rewards_onpolicy/mean",
            baseline_run_id="9ggix50f",
            title="Delta on-policy reward_mean vs Vanilla GRPO",
            fname="reward_onpolicy_mean_delta_vs_grpo.png",
            y_label="delta_reward_onpolicy_mean",
            window=7,
            ylim=(-0.4, 0.4),
        )
    if "critic/rewards_teacher/mean" in merged.columns:
        _plot_lines(
            merged,
            y="critic/rewards_teacher/mean",
            title="Teacher reward mean vs step",
            fname="reward_teacher_mean.png",
            y_label="reward_teacher_mean",
            window=7,
            ylim=(0, 1.0),
        )

    # Teacher usage / ratios
    for key, title, fname, ylim in [
        ("luffy/total_teacher_rollouts", "LUFFY: total teacher rollouts per step", "luffy_teacher_rollouts.png", None),
        ("diag/teacher_token_ratio", "diag/teacher_token_ratio vs step", "teacher_token_ratio.png", (0, 0.05)),
        ("diag/exp_token_ratio", "diag/exp_token_ratio vs step", "exp_token_ratio.png", (0, 0.05)),
        ("exp_mask_ratio", "exp_mask_ratio vs step", "exp_mask_ratio.png", (0, 0.2)),
    ]:
        if key in merged.columns:
            _plot_lines(merged, y=key, title=title, fname=fname, window=7, ylim=ylim)

    # Optimization diagnostics
    for key, title, fname in [
        ("actor/entropy_loss", "actor/entropy_loss vs step", "actor_entropy_loss.png"),
        ("actor/kl_loss", "actor/kl_loss vs step", "actor_kl_loss.png"),
        ("actor/pg_loss", "actor/pg_loss vs step", "actor_pg_loss.png"),
        ("actor/off_pg_loss", "actor/off_pg_loss vs step", "actor_off_pg_loss.png"),
        ("actor/on_pg_loss", "actor/on_pg_loss vs step", "actor_on_pg_loss.png"),
        ("actor/grad_norm", "actor/grad_norm vs step", "actor_grad_norm.png"),
        ("actor/ppo_kl", "actor/ppo_kl vs step", "actor_ppo_kl.png"),
    ]:
        if key in merged.columns:
            _plot_lines(merged, y=key, title=title, fname=fname, window=7)
            if key in {"actor/entropy_loss", "actor/kl_loss", "actor/pg_loss"}:
                _plot_delta_vs_baseline(
                    merged,
                    y=key,
                    baseline_run_id="9ggix50f",
                    title=f"Delta {key} vs Vanilla GRPO",
                    fname=f"{key.replace('/', '_')}_delta_vs_grpo.png",
                    y_label=f"delta_{key}",
                    window=7,
                )

    # Advantage-related
    for key, title, fname in [
        ("critic/advantages/mean", "critic/advantages/mean vs step", "adv_mean.png"),
        ("diag/adv_onpolicy_sample_mean", "diag/adv_onpolicy_sample_mean vs step", "adv_onpolicy_sample_mean.png"),
        ("diag/adv_teacher_sample_mean", "diag/adv_teacher_sample_mean vs step", "adv_teacher_sample_mean.png"),
        ("diag/group_teacher_minus_on_reward_mean", "group teacher - onpolicy reward gap vs step", "reward_gap_teacher_minus_on.png"),
    ]:
        if key in merged.columns:
            _plot_lines(merged, y=key, title=title, fname=fname, window=7)

    print("Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()


