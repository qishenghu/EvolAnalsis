#!/usr/bin/env python3
"""
Generate learning curve figure for DUET NeurIPS 2026 paper.
2x2 grid: rows = environment (ALFWorld, WebShop); cols = model size (1.5B, 3B).
Each panel shows on-policy success/reward over 100 training steps for 4 methods.

For WebShop 3B, the CHORD baseline uses the user-validated `chord_mu_0410`
configuration (val@100 reward mean 0.728 / strict SR 39.0%); the original
single-fixed-mu CHORD baseline collapsed mid-training and is unsuitable for
a learning-curve comparison.

Usage:
    python make_learning_curves.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Paper-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.8,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver"
OUT_DIR = "/data/home/qisheng/EvolAnalsis/NeurIPS_2026_Latex/figures"
METRIC_KEY = "diag/reward_onpolicy_mean"
N_STEPS = 100
SMOOTH_WINDOW = 10

# Run name mapping: {setting: {method: run_dir}}.  Order = row-major over a
# 2x2 grid (rows = environment; cols = model size).
SETTINGS = {
    "ALFWorld 1.5B": {
        "DUET":  "alfworld_qwen1.5b_duet_v39c_postfix",
        "LUFFY": "alfworld_qwen1.5b_luffy",
        "CHORD": "alfworld_qwen1.5b_chord",
        "GRPO":  "alfworld_qwen1.5b_onpolicy",
    },
    "ALFWorld 3B": {
        "DUET":  "alfworld_qwen3b_duet_v39b",
        "LUFFY": "alfworld_3b_luffy",
        "CHORD": "alfworld_3b_chord",
        "GRPO":  "alfworld_3b_grpo_react_tags",
    },
    "WebShop 1.5B": {
        "DUET":  "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06",
        "LUFFY": "webshop_qwen1.5b_luffy",
        "CHORD": "webshop_qwen1.5b_chord",
        "GRPO":  "webshop_qwen1.5b_onpolicy",
    },
    "WebShop 3B": {
        "DUET":  "webshop_qwen3b_duet_v39b",
        "LUFFY": "webshop_3b_luffy",
        # chord_mu_0410: user-validated 4xH100 CHORD config that reaches
        # val@100 reward mean 0.728 (strict SR 39.0%) on WebShop 3B; the
        # default `webshop_3b_chord` baseline collapsed and is unsuitable
        # for a learning-curve comparison.
        "CHORD": "webshop_3b_chord_mu_0410",
        "GRPO":  "webshop_3b_onpolicy",
    },
}

# Y-axis labels per environment prefix
Y_LABELS = {
    "ALFWorld": "Success Rate",
    "WebShop":  "Mean Reward",
}

# Method styling
METHOD_STYLE = {
    "DUET":  dict(color="#1f77b4", linestyle="-",  linewidth=2.4, zorder=4),
    "LUFFY": dict(color="#ff7f0e", linestyle="--", linewidth=1.8, zorder=3),
    "CHORD": dict(color="#2ca02c", linestyle="-",  linewidth=1.8, zorder=2),
    "GRPO":  dict(color="#7f7f7f", linestyle="--", linewidth=1.8, zorder=1),
}

# Method order for legend
METHOD_ORDER = ["DUET", "LUFFY", "CHORD", "GRPO"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_curve(run_dir: str) -> np.ndarray:
    """Load metric values for steps 1..N_STEPS from batch_diag JSON files."""
    traj_dir = os.path.join(BASE_DIR, run_dir, "Trajectory")
    values = []
    for step in range(1, N_STEPS + 1):
        fp = os.path.join(traj_dir, f"batch_diag_step_{step}.json")
        with open(fp) as f:
            data = json.load(f)
        values.append(data[METRIC_KEY])
    return np.array(values, dtype=np.float64)


def smooth(y: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Simple moving average smoothing with same-length output."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    # Pad edges to keep length
    padded = np.concatenate([
        np.full(window // 2, y[0]),
        y,
        np.full(window - 1 - window // 2, y[-1]),
    ])
    return np.convolve(padded, kernel, mode="valid")


def compute_aulc(y: np.ndarray) -> float:
    """Area Under Learning Curve = mean of values over all steps."""
    return float(np.mean(y))

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure():
    setting_names = list(SETTINGS.keys())
    n_panels = len(setting_names)

    # 2x2 grid: rows = ALFWorld / WebShop; cols = 1.5B / 3B.
    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    axes = axes_grid.flatten()

    steps = np.arange(1, N_STEPS + 1)
    aulc_table = {}  # {setting: {method: aulc}}

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (ax, setting_name) in enumerate(zip(axes, setting_names)):
        runs = SETTINGS[setting_name]
        aulc_table[setting_name] = {}

        for method in METHOD_ORDER:
            run_dir = runs[method]
            raw = load_curve(run_dir)
            smoothed = smooth(raw)
            style = METHOD_STYLE[method]

            # Plot MA-5 smoothed curve (no raw underlay; cleaner read)
            ax.plot(steps, smoothed, label=method, **style)

            # AULC computed on raw data (unsmoothed)
            aulc_table[setting_name][method] = compute_aulc(raw)

        # Determine y-label based on environment
        env_prefix = setting_name.split()[0]
        ylabel = Y_LABELS.get(env_prefix, "Metric")

        # x-label only on bottom row
        if idx >= 2:
            ax.set_xlabel("Training Step")
        # y-label only on left column
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_labels[idx]}  {setting_name}", fontweight="bold",
                     loc="left")
        ax.set_xlim(1, N_STEPS)
        ax.set_ylim(0, 0.85)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend at top-center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=True, fancybox=False, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, "fig_learning_curves.pdf")
    png_path = os.path.join(OUT_DIR, "fig_learning_curves.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close(fig)

    return aulc_table


def print_aulc_table(aulc_table):
    """Print AULC table in a clean format."""
    print("\n" + "=" * 70)
    print("AULC (Area Under Learning Curve) -- higher is better")
    print("=" * 70)

    settings = list(aulc_table.keys())
    header = f"{'Method':<10}" + "".join(f"{s:>18}" for s in settings)
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        row = f"{method:<10}"
        for setting in settings:
            val = aulc_table[setting][method]
            row += f"{val:>18.4f}"
        print(row)

    # Also print final-step values
    print("\n" + "=" * 70)
    print("Final step (step 100) values")
    print("=" * 70)
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        row = f"{method:<10}"
        for setting in settings:
            run_dir = SETTINGS[setting][method]
            raw = load_curve(run_dir)
            row += f"{raw[-1]:>18.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    aulc_table = make_figure()
    print_aulc_table(aulc_table)
