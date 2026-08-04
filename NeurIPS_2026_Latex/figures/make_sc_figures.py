#!/usr/bin/env python3
"""
Generate publication-quality State Channel diagnostic figures for DUET (NeurIPS 2026).

Figure 1 (fig_sc_dynamics.pdf):
    2x2 grid (rows=ALFWorld/WebShop, cols=1.5B/3B).
    Each panel: SC progress, SC coverage (left y), bonus/reward ratio (right y).

Figure 2 (fig_reward_decomposition.pdf):
    2x2 grid, same layout. Stacked area: env reward, SC bonus, step deltas.

Usage:
    python make_sc_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver"
OUT_DIR = "/data/home/qisheng/EvolAnalsis/NeurIPS_2026_Latex/figures"

RUNS = {
    ("ALFWorld", "1.5B"): "alfworld_qwen1.5b_duet_v39c_postfix",
    ("ALFWorld", "3B"):   "alfworld_qwen3b_duet_v39b",
    ("WebShop", "1.5B"):  "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06",
    ("WebShop", "3B"):    "webshop_qwen3b_duet_v39b",
}

ENVS = ["ALFWorld", "WebShop"]
SIZES = ["1.5B", "3B"]

# Read every 5th step: 1, 5, 10, 15, ..., 100
STEPS = [1] + list(range(5, 101, 5))

# ---------------------------------------------------------------------------
# Matplotlib style (clean academic, NeurIPS)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "legend.handlelength": 1.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.5,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Colour palette (colour-blind friendly, based on ColorBrewer)
C_BLUE   = "#2166ac"
C_GREEN  = "#1b7837"
C_ORANGE = "#d95f02"
C_AREA_ENV   = "#66c2a5"   # teal-green
C_AREA_BONUS = "#8da0cb"   # periwinkle
C_AREA_DELTA = "#fc8d62"   # salmon-orange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(y, window=3):
    """Simple moving-average smoother (preserves endpoints)."""
    if len(y) <= window:
        return y
    out = np.array(y, dtype=float)
    hw = window // 2
    for i in range(hw, len(y) - hw):
        out[i] = np.mean(y[i - hw : i + hw + 1])
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_step_data(run_name, step):
    """Load per-trajectory diagnostics for one step, return on-policy only."""
    path = os.path.join(BASE, run_name, "Trajectory",
                        f"trajectories_step_{step}.jsonl")
    if not os.path.exists(path):
        return None

    records = []
    with open(path) as f:
        for line in f:
            traj = json.loads(line)
            diag = traj["diag"]
            if diag["is_teacher"]:
                continue  # on-policy only
            records.append(diag)
    return records


def gather_all_data():
    """Return nested dict: data[(env, size)] = {step -> aggregated metrics}."""
    all_data = {}
    for (env, size), run_name in RUNS.items():
        step_metrics = {}
        for step in STEPS:
            records = load_step_data(run_name, step)
            if records is None or len(records) == 0:
                continue

            sc_progress = [r["sc_progress"] for r in records
                           if r["sc_progress"] is not None]
            sc_coverage = [r["sc_coverage"] for r in records
                           if r["sc_coverage"] is not None]

            # Bonus / reward ratio -- use aggregate ratio (sum bonus / sum |reward|)
            # to avoid per-trajectory outliers when reward_sum ~ 0
            total_bonus = sum(abs(r.get("sc_bonus", 0.0) or 0.0) for r in records)
            total_reward = sum(abs(r.get("reward_sum", 0.0) or 0.0) for r in records)
            bonus_ratio = total_bonus / max(total_reward, 1e-6)

            # Reward decomposition
            reward_orig = [r.get("reward_original", 0.0) or 0.0 for r in records]
            sc_bonus = [r.get("sc_bonus", 0.0) or 0.0 for r in records]
            step_deltas_sum = []
            for r in records:
                deltas = r.get("sc_step_deltas", []) or []
                step_deltas_sum.append(sum(deltas))

            step_metrics[step] = {
                "sc_progress_mean": np.mean(sc_progress) if sc_progress else 0.0,
                "sc_coverage_mean": np.mean(sc_coverage) if sc_coverage else 0.0,
                "bonus_ratio": bonus_ratio,
                "reward_orig_mean": np.mean(reward_orig),
                "sc_bonus_mean": np.mean(sc_bonus),
                "step_deltas_mean": np.mean(step_deltas_sum),
                "n_onpolicy": len(records),
            }

        all_data[(env, size)] = step_metrics
        print(f"  Loaded {env} {size}: {len(step_metrics)} steps, "
              f"run={run_name}")
    return all_data


# ---------------------------------------------------------------------------
# Figure 1: SC Dynamics
# ---------------------------------------------------------------------------

def _draw_fig1_panel(ax, steps, prog, cov, ratio, title):
    """Draw one SC Dynamics panel."""
    # Light grid on left axis only
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)

    # Left y-axis: progress and coverage
    ln1, = ax.plot(steps, smooth(prog), color=C_BLUE, marker="o",
                   markersize=2.5, markeredgewidth=0, label="SC progress",
                   zorder=3)
    ln2, = ax.plot(steps, smooth(cov), color=C_GREEN, marker="s",
                   markersize=2.5, markeredgewidth=0, label="SC coverage",
                   zorder=3)
    ax.set_ylabel("Progress / Coverage")
    ax.set_ylim(-0.02, 1.05)

    # Right y-axis: bonus ratio
    ax2 = ax.twinx()
    ln3, = ax2.plot(steps, smooth(ratio), color=C_ORANGE, linestyle="--",
                    linewidth=1.3, marker="^", markersize=2.5,
                    markeredgewidth=0, label="Bonus / reward", zorder=3)
    ax2.set_ylabel("Bonus / Reward Ratio", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ratio_max = max(max(ratio) * 1.25, 0.25)
    ax2.set_ylim(-0.01, ratio_max)
    ax2.spines["right"].set_color(C_ORANGE)
    ax2.spines["right"].set_linewidth(0.7)

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_xlim(0, 102)
    ax.xaxis.set_major_locator(MultipleLocator(20))

    # Combined legend (placed in best location per panel)
    lines = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best", framealpha=0.9,
              edgecolor="0.8", fancybox=False)

    return ax


def make_fig1(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for row_i, env in enumerate(ENVS):
        for col_j, size in enumerate(SIZES):
            ax = axes[row_i, col_j]
            metrics = all_data.get((env, size), {})
            if not metrics:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            steps_list = sorted(metrics.keys())
            prog = [metrics[s]["sc_progress_mean"] for s in steps_list]
            cov  = [metrics[s]["sc_coverage_mean"] for s in steps_list]
            ratio = [metrics[s]["bonus_ratio"] for s in steps_list]

            title = f"{env}  \u2014  Qwen2.5-{size}"
            _draw_fig1_panel(ax, steps_list, prog, cov, ratio, title)

    fig.tight_layout(w_pad=3.0, h_pad=2.5)

    # Save PDF and PNG
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"fig_sc_dynamics.{ext}")
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Reward Decomposition (stacked area)
# ---------------------------------------------------------------------------

def _draw_fig2_panel(ax, steps, env_r, sc_b, sd, title, show_legend=False):
    """Draw one Reward Decomposition panel."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.4)

    # Clip negatives for stacking
    env_r_pos = np.clip(env_r, 0, None)
    sc_b_pos  = np.clip(sc_b, 0, None)
    sd_pos    = np.clip(sd, 0, None)

    y_stack = np.vstack([env_r_pos, sc_b_pos, sd_pos])

    ax.stackplot(steps, y_stack,
                 labels=["Env. reward", "SC bonus", "Step deltas"],
                 colors=[C_AREA_ENV, C_AREA_BONUS, C_AREA_DELTA],
                 alpha=0.72,
                 edgecolor="white", linewidth=0.4)

    # Thin total line
    total = env_r_pos + sc_b_pos + sd_pos
    ax.plot(steps, total, color="0.3", linewidth=0.8, alpha=0.6, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_xlim(0, 102)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylim(bottom=0)

    if show_legend:
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.8",
                  fancybox=False)


def make_fig2(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for row_i, env in enumerate(ENVS):
        for col_j, size in enumerate(SIZES):
            ax = axes[row_i, col_j]
            metrics = all_data.get((env, size), {})
            if not metrics:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            steps_list = sorted(metrics.keys())
            env_r = np.array([metrics[s]["reward_orig_mean"] for s in steps_list])
            sc_b  = np.array([metrics[s]["sc_bonus_mean"] for s in steps_list])
            sd    = np.array([metrics[s]["step_deltas_mean"] for s in steps_list])

            title = f"{env}  \u2014  Qwen2.5-{size}"
            show_leg = (row_i == 0 and col_j == 0)
            _draw_fig2_panel(ax, steps_list, env_r, sc_b, sd, title,
                             show_legend=show_leg)

    fig.tight_layout(w_pad=3.0, h_pad=2.5)

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"fig_reward_decomposition.{ext}")
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading trajectory data...")
    all_data = gather_all_data()
    print()

    # Print summary table
    print("=" * 78)
    print(f"{'Config':<25} {'Steps':<8} {'Final Prog':<12} "
          f"{'Final Cov':<12} {'Final Ratio':<12}")
    print("-" * 78)
    for (env, size) in [(e, s) for e in ENVS for s in SIZES]:
        metrics = all_data.get((env, size), {})
        if not metrics:
            print(f"{env} {size:<20} {'N/A':<8}")
            continue
        max_step = max(metrics.keys())
        m = metrics[max_step]
        print(f"{env} {size:<20} {len(metrics):<8} "
              f"{m['sc_progress_mean']:<12.4f} "
              f"{m['sc_coverage_mean']:<12.4f} "
              f"{m['bonus_ratio']:<12.4f}")
    print("=" * 78)
    print()

    print("Generating Figure 1: SC Dynamics...")
    make_fig1(all_data)
    print()

    print("Generating Figure 2: Reward Decomposition...")
    make_fig2(all_data)
    print()

    print("All figures generated successfully.")
