#!/usr/bin/env python3
"""
Generate publication-quality figures for the DUET NeurIPS 2026 paper.

Figure 1: Learning curves (2x2 grid: rows=env, cols=model_size)
Figure 2: DR3 + Teacher dynamics for DUET runs (2x2 grid)

Usage:
    python make_analysis_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ---------- paths ----------
CKPT_ROOT = "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver"
OUT_DIR = "/data/home/qisheng/EvolAnalsis/NeurIPS_2026_Latex/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 0,
})

COLORS = {
    "DUET": "#2196F3",
    "LUFFY": "#FF9800",
    "CHORD": "#4CAF50",
    "GRPO": "#9E9E9E",
}

# ---------- run definitions ----------
RUNS_FIG1 = {
    # (row_label, col_label): {method: run_name}
    ("ALFWorld", "1.5B"): {
        "DUET":  "alfworld_qwen1.5b_duet_v39c_postfix",
        "LUFFY": "alfworld_qwen1.5b_luffy",
        "CHORD": "alfworld_qwen1.5b_chord",
        "GRPO":  "alfworld_qwen1.5b_onpolicy",
    },
    ("ALFWorld", "3B"): {
        "DUET":  "alfworld_qwen3b_duet_v39b",
        "LUFFY": "alfworld_3b_luffy",
        "CHORD": "alfworld_3b_chord",
        "GRPO":  "alfworld_3b_grpo_react_tags",
    },
    ("WebShop", "1.5B"): {
        "DUET":  "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06",
        "LUFFY": "webshop_qwen1.5b_luffy",
        "CHORD": "webshop_qwen1.5b_chord",
        "GRPO":  "webshop_qwen1.5b_onpolicy",
    },
    ("WebShop", "3B"): {
        "DUET":  "webshop_qwen3b_duet_v39b",
        "LUFFY": "webshop_3b_luffy",
        # User-validated CHORD config (val@100 reward 0.728 / SR 39.0%);
        # the default `webshop_3b_chord` collapsed mid-training.
        "CHORD": "webshop_3b_chord_mu_0410",
        "GRPO":  "webshop_3b_onpolicy",
    },
}

DUET_RUNS_FIG2 = {
    ("ALFWorld", "1.5B"): "alfworld_qwen1.5b_duet_v39c_postfix",
    ("ALFWorld", "3B"):   "alfworld_qwen3b_duet_v39b",
    ("WebShop", "1.5B"):  "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06",
    ("WebShop", "3B"):    "webshop_qwen3b_duet_v39b",
}


# ---------- helpers ----------

def load_metric_series(run_name: str, key: str, max_step: int = 100) -> tuple:
    """Load a metric across steps 1..max_step. Returns (steps, values)."""
    traj_dir = os.path.join(CKPT_ROOT, run_name, "Trajectory")
    steps, vals = [], []
    for s in range(1, max_step + 1):
        fpath = os.path.join(traj_dir, f"batch_diag_step_{s}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            d = json.load(f)
        if key in d:
            steps.append(s)
            vals.append(d[key])
    return np.array(steps), np.array(vals)


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Rolling average with same-length output (edge-padded)."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    # Pad edges to keep length
    padded = np.concatenate([
        np.full(window // 2, y[0]),
        y,
        np.full(window - 1 - window // 2, y[-1]),
    ])
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


# ================================================================
# FIGURE 1: Learning Curves
# ================================================================

def make_fig1():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    panel_grid = [
        (0, 0, "ALFWorld", "1.5B"),
        (0, 1, "ALFWorld", "3B"),
        (1, 0, "WebShop", "1.5B"),
        (1, 1, "WebShop", "3B"),
    ]

    for row, col, env, size in panel_grid:
        ax = axes[row, col]
        runs = RUNS_FIG1[(env, size)]

        for method in ["GRPO", "CHORD", "LUFFY", "DUET"]:
            run_name = runs[method]
            steps, vals = load_metric_series(run_name, "diag/reward_onpolicy_mean")
            if len(vals) == 0:
                print(f"  WARNING: no data for {method} ({run_name})")
                continue
            smoothed = smooth(vals, window=5)
            ax.plot(steps, smoothed, color=COLORS[method], label=method,
                    alpha=0.95, zorder=3 if method == "DUET" else 2)
            # Light raw data behind
            ax.plot(steps, vals, color=COLORS[method], alpha=0.12, linewidth=0.7,
                    zorder=1)

        ax.set_title(f"{env}  --  Qwen2.5-{size}", fontweight="bold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 101)

        if row == 1:
            ax.set_xlabel("Training Step")
        if col == 0:
            ax.set_ylabel("On-Policy Success Rate")

    # Shared legend at top
    legend_handles = [
        Line2D([0], [0], color=COLORS[m], linewidth=2.0, label=m)
        for m in ["DUET", "LUFFY", "CHORD", "GRPO"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=10,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "fig_learning_curves.pdf")
    fig.savefig(out_path)
    print(f"[Fig 1] Saved to {out_path}")
    # Also save PNG for quick preview
    fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)


# ================================================================
# FIGURE 2: DR3 + Teacher Dynamics
# ================================================================

def make_fig2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    panel_grid = [
        (0, 0, "ALFWorld", "1.5B"),
        (0, 1, "ALFWorld", "3B"),
        (1, 0, "WebShop", "1.5B"),
        (1, 1, "WebShop", "3B"),
    ]

    COLOR_TSR = "#E91E63"       # teacher sample ratio
    COLOR_TAF = "#7B1FA2"       # teacher advantage fraction
    COLOR_REWARD = "#2196F3"    # on-policy reward

    for row, col, env, size in panel_grid:
        ax_left = axes[row, col]
        ax_right = ax_left.twinx()

        run_name = DUET_RUNS_FIG2[(env, size)]

        # --- Left axis: teacher_sample_ratio ---
        steps_tsr, vals_tsr = load_metric_series(run_name, "diag/teacher_sample_ratio")
        if len(vals_tsr) > 0:
            ax_left.plot(steps_tsr, smooth(vals_tsr, 5),
                         color=COLOR_TSR, linewidth=1.4, alpha=0.85,
                         label="Teacher Sample Ratio", linestyle="--")

        # --- Left axis: teacher advantage fraction ---
        # Fraction = |adv_teacher_token_mean| / (|adv_teacher_token_mean| + |adv_onpolicy_token_mean|)
        steps_at, vals_at = load_metric_series(run_name, "diag/adv_teacher_token_mean")
        steps_ao, vals_ao = load_metric_series(run_name, "diag/adv_onpolicy_token_mean")
        if len(vals_at) > 0 and len(vals_ao) > 0:
            # Align steps
            common_steps = sorted(set(steps_at) & set(steps_ao))
            at_dict = dict(zip(steps_at, vals_at))
            ao_dict = dict(zip(steps_ao, vals_ao))
            cs = np.array(common_steps)
            at_vals = np.array([at_dict[s] for s in common_steps])
            ao_vals = np.array([ao_dict[s] for s in common_steps])
            # Use absolute values for the fraction
            abs_at = np.abs(at_vals)
            abs_ao = np.abs(ao_vals)
            denom = abs_at + abs_ao
            denom = np.where(denom < 1e-8, 1e-8, denom)
            frac = abs_at / denom
            ax_left.plot(cs, smooth(frac, 5),
                         color=COLOR_TAF, linewidth=1.8, alpha=0.9,
                         label="Teacher Adv. Fraction")

        ax_left.set_ylim(-0.02, 1.02)
        ax_left.set_ylabel("Ratio / Fraction", color="black")
        ax_left.tick_params(axis="y", labelcolor="black")

        # --- Right axis: on-policy success ---
        steps_r, vals_r = load_metric_series(run_name, "diag/reward_onpolicy_mean")
        if len(vals_r) > 0:
            ax_right.plot(steps_r, smooth(vals_r, 5),
                          color=COLOR_REWARD, linewidth=2.0, alpha=0.9,
                          label="On-Policy Success")
        ax_right.set_ylim(-0.02, 1.02)
        ax_right.set_ylabel("On-Policy Success Rate", color=COLOR_REWARD)
        ax_right.tick_params(axis="y", labelcolor=COLOR_REWARD)
        ax_right.spines["right"].set_visible(True)
        ax_right.spines["right"].set_color(COLOR_REWARD)
        ax_right.spines["right"].set_alpha(0.5)

        ax_left.set_title(f"DUET  --  {env}  --  Qwen2.5-{size}", fontweight="bold")
        ax_left.set_xlim(0, 101)

        if row == 1:
            ax_left.set_xlabel("Training Step")

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=COLOR_TSR, linewidth=1.4, linestyle="--",
               label="Teacher Sample Ratio"),
        Line2D([0], [0], color=COLOR_TAF, linewidth=1.8,
               label="Teacher Adv. Fraction"),
        Line2D([0], [0], color=COLOR_REWARD, linewidth=2.0,
               label="On-Policy Success Rate"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=10,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUT_DIR, "fig_dr3_dynamics.pdf")
    fig.savefig(out_path)
    print(f"[Fig 2] Saved to {out_path}")
    fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)


# ================================================================
# Summary statistics (printed to stdout for reference)
# ================================================================

def print_summary():
    print("\n" + "=" * 70)
    print("SUMMARY: Final on-policy success rates (last 5-step average)")
    print("=" * 70)
    header = f"{'Env':<10} {'Size':<6} {'DUET':>8} {'LUFFY':>8} {'CHORD':>8} {'GRPO':>8}"
    print(header)
    print("-" * len(header))

    for (env, size), runs in RUNS_FIG1.items():
        row = f"{env:<10} {size:<6}"
        for method in ["DUET", "LUFFY", "CHORD", "GRPO"]:
            _, vals = load_metric_series(runs[method], "diag/reward_onpolicy_mean")
            if len(vals) >= 5:
                final = np.mean(vals[-5:])
                row += f" {final:>7.3f}"
            else:
                row += f" {'N/A':>7}"
        print(row)
    print("=" * 70)


# ================================================================
if __name__ == "__main__":
    print("Generating Figure 1: Learning Curves ...")
    make_fig1()
    print("Generating Figure 2: DR3 + Teacher Dynamics ...")
    make_fig2()
    print_summary()
    print("\nDone.")
