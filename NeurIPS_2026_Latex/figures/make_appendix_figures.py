"""
Appendix figures for the DUET NeurIPS 2026 paper.

Figure 1 (fig_advantage_dist.pdf) — Advantage distributions with/without
baseline separation.  2 rows x 3 columns:
  Row 1: DUET (with baseline separation)
  Row 2: DUET -BS (without baseline separation)
  Columns: early / mid / late training
  Each cell: overlaid histograms of per-trajectory adv_mean for teacher (red)
  vs on-policy (blue) samples, aggregated across a 5-step window for density.
  Vertical dashed lines at distribution means, with numeric annotations.

Figure 2 (fig_entropy_comparison.pdf) — On-policy token entropy across methods.
  2x2 grid: {ALFWorld, WebShop} x {1.5B, 3B}.
  Methods: DUET, LUFFY, CHORD, GRPO.

Data sources:
  - Trajectory JSONL files: checkpoints/.../Trajectory/trajectories_step_{N}.jsonl
  - Batch diagnostics:      checkpoints/.../Trajectory/batch_diag_step_{N}.json
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path("/data/home/qisheng/EvolAnalsis")
CKPT = ROOT / "checkpoints" / "agentevolver"
OUT = ROOT / "NeurIPS_2026_Latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- NeurIPS-style aesthetics (match existing figures) ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
})


# ===========================================================================
#  Utility helpers
# ===========================================================================

def smooth(y, k=5):
    """Simple moving-average smoother (same as main figure script)."""
    if len(y) <= 1:
        return y
    pad = max(1, k // 2)
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(yp, kernel, mode="valid")


def load_trajectory_advantages_window(run_name, center_step, half_width=2):
    """Load per-trajectory adv_mean from JSONL files across a window of steps.
    Aggregates steps [center-half_width, center+half_width] for denser data.
    Returns (teacher_advs, onpolicy_advs) as numpy arrays."""
    teacher, onpolicy = [], []
    for step in range(max(1, center_step - half_width),
                      center_step + half_width + 1):
        fpath = CKPT / run_name / "Trajectory" / f"trajectories_step_{step}.jsonl"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                diag = d.get("diag", {})
                adv = diag.get("adv_mean")
                if adv is None:
                    continue
                if diag.get("is_teacher", False):
                    teacher.append(adv)
                else:
                    onpolicy.append(adv)
    return np.array(teacher), np.array(onpolicy)


def load_batch_diag_series(run_name, key, max_step=100):
    """Load a single metric from batch_diag_step_{N}.json files.
    Returns (steps, values) sorted by step."""
    traj_dir = CKPT / run_name / "Trajectory"
    if not traj_dir.exists():
        print(f"  [WARN] missing directory: {traj_dir}")
        return np.array([]), np.array([])
    steps, vals = [], []
    for step in range(1, max_step + 1):
        fpath = traj_dir / f"batch_diag_step_{step}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            d = json.load(f)
        if key in d:
            steps.append(step)
            vals.append(d[key])
    return np.array(steps), np.array(vals)


def find_latest_step(run_name, prefix="trajectories_step_"):
    """Find the largest step N for which trajectories_step_{N}.jsonl exists."""
    traj_dir = CKPT / run_name / "Trajectory"
    if not traj_dir.exists():
        return 0
    max_s = 0
    for f in traj_dir.iterdir():
        if f.name.startswith(prefix) and f.name.endswith(".jsonl"):
            try:
                s = int(f.stem.split("_")[-1])
                max_s = max(max_s, s)
            except ValueError:
                pass
    return max_s


# ===========================================================================
#  Figure 1: Advantage Distribution With / Without Baseline Separation
# ===========================================================================

print("=" * 60)
print("Figure 1: Advantage distributions (baseline separation)")
print("=" * 60)

RUN_WITH_BS = "alfworld_qwen1.5b_duet_v39c_postfix"
RUN_WITHOUT_BS = "alfworld_qwen1.5b_duet_minus_baseline_sep"

# Step centers for the three columns
STEP_CENTERS = [5, 30, 60]
HALF_W = 2  # aggregate steps [center-2, center+2]

latest_no_bs = find_latest_step(RUN_WITHOUT_BS)
print(f"Latest step for -BS run: {latest_no_bs}")
# For the -BS run, use step 60 if available, else latest
late_center_no_bs = 60 if latest_no_bs >= 62 else max(latest_no_bs - HALF_W, 1)
STEP_CENTERS_NOBS = [5, 30, late_center_no_bs]

COL_LABELS = [
    f"Steps {max(1,c-HALF_W)}\u2013{c+HALF_W}" for c in STEP_CENTERS
]

fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.2))

# Colors
CLR_ON = "#2196F3"       # on-policy histogram
CLR_ON_LINE = "#0D47A1"  # on-policy mean line
CLR_TE = "#E53935"       # teacher histogram
CLR_TE_LINE = "#B71C1C"  # teacher mean line

# First pass: collect data to determine shared x-ranges per column
all_data = {}  # (row, col) -> (teacher, onpolicy)
for row_idx, (run_name, centers) in enumerate([
    (RUN_WITH_BS, STEP_CENTERS),
    (RUN_WITHOUT_BS, STEP_CENTERS_NOBS),
]):
    for col_idx, center in enumerate(centers):
        t, o = load_trajectory_advantages_window(run_name, center, HALF_W)
        all_data[(row_idx, col_idx)] = (t, o)

# Determine shared x-range per column (across both rows)
col_ranges = []
for col_idx in range(3):
    all_vals = []
    for row_idx in range(2):
        t, o = all_data[(row_idx, col_idx)]
        if len(t) > 0:
            all_vals.extend(t.tolist())
        if len(o) > 0:
            all_vals.extend(o.tolist())
    if all_vals:
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = max(0.08 * (hi - lo), 0.02)
        col_ranges.append((lo - margin, hi + margin))
    else:
        col_ranges.append((-1, 1))

# Second pass: draw
for row_idx, (run_name, centers, row_label) in enumerate([
    (RUN_WITH_BS, STEP_CENTERS, "DUET"),
    (RUN_WITHOUT_BS, STEP_CENTERS_NOBS, r"DUET $-$BS"),
]):
    for col_idx, center in enumerate(centers):
        ax = axes[row_idx, col_idx]
        teacher_adv, onpolicy_adv = all_data[(row_idx, col_idx)]
        print(f"  {row_label} center {center}: teacher n={len(teacher_adv)}, "
              f"on-policy n={len(onpolicy_adv)}")

        lo, hi = col_ranges[col_idx]
        bins = np.linspace(lo, hi, 25)

        # Draw on-policy
        if len(onpolicy_adv) > 0:
            ax.hist(onpolicy_adv, bins=bins, alpha=0.50, color=CLR_ON,
                    label="On-policy", density=True, edgecolor="none")
            on_mean = np.mean(onpolicy_adv)
            ax.axvline(on_mean, color=CLR_ON_LINE, ls="--", lw=1.3, alpha=0.85)

        # Draw teacher
        if len(teacher_adv) > 0:
            ax.hist(teacher_adv, bins=bins, alpha=0.50, color=CLR_TE,
                    label="Teacher", density=True, edgecolor="none")
            te_mean = np.mean(teacher_adv)
            ax.axvline(te_mean, color=CLR_TE_LINE, ls="--", lw=1.3, alpha=0.85)

        # Annotate means at top of each panel
        if len(onpolicy_adv) > 0 and len(teacher_adv) > 0:
            gap = te_mean - on_mean
            ax.text(0.97, 0.95,
                    f"$\\mu_{{\\mathrm{{on}}}}$={on_mean:.2f}\n"
                    f"$\\mu_{{\\mathrm{{te}}}}$={te_mean:.2f}\n"
                    f"$\\Delta$={gap:.2f}",
                    transform=ax.transAxes, fontsize=6.5,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="0.7", alpha=0.85))

        # Shared x-range per column
        ax.set_xlim(lo, hi)

        # Title: column label on top row only
        if row_idx == 0:
            ax.set_title(COL_LABELS[col_idx], fontsize=9, fontweight="medium")

        # Row label on leftmost column
        if col_idx == 0:
            ax.set_ylabel(f"{row_label}\nDensity", fontsize=8.5)
        else:
            ax.set_ylabel("")

        # X label on bottom row
        if row_idx == 1:
            ax.set_xlabel("Advantage (per-trajectory mean)", fontsize=8)

        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25)

        # Legend in first cell only
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=7)

fig.tight_layout(h_pad=0.8, w_pad=0.5)
fig.savefig(OUT / "fig_advantage_dist.pdf", bbox_inches="tight")
print(f"[saved] {OUT / 'fig_advantage_dist.pdf'}")


# ===========================================================================
#  Figure 2: Entropy Comparison Across Methods
# ===========================================================================

print()
print("=" * 60)
print("Figure 2: Entropy comparison across methods")
print("=" * 60)

METHOD_COLORS = {
    "DUET":  "#2196F3",
    "LUFFY": "#FF9800",
    "CHORD": "#4CAF50",
    "GRPO":  "#9E9E9E",
}

METHOD_LINESTYLES = {
    "DUET":  "-",
    "LUFFY": "--",
    "CHORD": "-",
    "GRPO":  "--",
}

METHOD_LINEWIDTHS = {
    "DUET":  2.0,
    "LUFFY": 1.5,
    "CHORD": 1.5,
    "GRPO":  1.4,
}

# Panel configs: (env_label, size_label, {method: run_name})
PANELS = [
    # Row 0: ALFWorld
    ("ALFWorld", "1.5B", {
        "DUET":  "alfworld_qwen1.5b_duet_v39c_postfix",
        "LUFFY": "alfworld_qwen1.5b_luffy",
        "CHORD": "alfworld_qwen1.5b_chord",
        "GRPO":  "alfworld_qwen1.5b_onpolicy",
    }),
    ("ALFWorld", "3B", {
        "DUET":  "alfworld_qwen3b_duet_v39b",
        "LUFFY": "alfworld_3b_luffy",
        "CHORD": "alfworld_3b_chord",
        "GRPO":  "alfworld_3b_grpo_react_tags",
    }),
    # Row 1: WebShop
    ("WebShop", "1.5B", {
        "DUET":  "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06",
        "LUFFY": "webshop_qwen1.5b_luffy",
        "CHORD": "webshop_qwen1.5b_chord",
        "GRPO":  "webshop_qwen1.5b_onpolicy",
    }),
    ("WebShop", "3B", {
        "DUET":  "webshop_qwen3b_duet_v39b",
        "LUFFY": "webshop_3b_luffy",
        # User-validated CHORD config (val@100 reward 0.728 / SR 39.0%);
        # the default `webshop_3b_chord` collapsed mid-training.
        "CHORD": "webshop_3b_chord_mu_0410",
        "GRPO":  "webshop_3b_onpolicy",
    }),
]

ENTROPY_KEY = "diag/entropy_onpolicy_token_mean"
SMOOTH_K = 5

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)

# Map panels to grid positions:
# [0] ALFWorld 1.5B -> (0,0),  [1] ALFWorld 3B -> (0,1)
# [2] WebShop 1.5B -> (1,0),   [3] WebShop 3B -> (1,1)
grid_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]

draw_order = ["GRPO", "CHORD", "LUFFY", "DUET"]  # DUET on top

for panel_idx, (env_label, size_label, methods) in enumerate(PANELS):
    r, c = grid_pos[panel_idx]
    ax = axes[r, c]

    for method_name in draw_order:
        run_name = methods[method_name]
        xs, ys = load_batch_diag_series(run_name, ENTROPY_KEY, max_step=100)
        if len(xs) == 0:
            print(f"  [WARN] no entropy data for {method_name} / {run_name}")
            continue
        ys_smooth = smooth(ys, k=SMOOTH_K)
        ax.plot(xs, ys_smooth,
                label=method_name,
                color=METHOD_COLORS[method_name],
                linestyle=METHOD_LINESTYLES[method_name],
                linewidth=METHOD_LINEWIDTHS[method_name],
                zorder=10 if method_name == "DUET" else 5)
        print(f"  {env_label} {size_label} {method_name}: "
              f"{len(xs)} points, final={ys[-1]:.4f}")

    ax.set_title(f"{env_label} ({size_label})", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y")

    if r == 1:
        ax.set_xlabel("Training step")
    if c == 0:
        ax.set_ylabel("On-policy token entropy")

# Shared legend at bottom of figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.045, 1, 1], h_pad=1.0, w_pad=0.8)
fig.savefig(OUT / "fig_entropy_comparison.pdf", bbox_inches="tight")
print(f"[saved] {OUT / 'fig_entropy_comparison.pdf'}")

print("\nDone.")
