"""
Build the DR3 Dynamics figure for the DUET NeurIPS 2026 paper.

Three-panel figure (1 row, 3 columns) showing the Action Channel's core evidence:
data-driven teacher fade-out via density-ratio reweighting.

Panel (a): DR3 discriminator accuracy over training steps
Panel (b): Teacher gradient share over training steps (overlaid with on-policy success)
Panel (c): Mean density ratio (w_off_mean) on teacher samples over training steps

Data sources: training log files parsed for wandb-style metric lines.

Runs:
  - ALFWorld 1.5B DUET (v39c_postfix)
  - ALFWorld 7B DUET
  - WebShop 1.5B DUET (swC_02_pk03_v10_floor06)

Output:
  fig_dr3_dynamics.pdf / .png
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/data/home/qisheng/EvolAnalsis")
LOG = ROOT / "logs"
OUT = ROOT / "NeurIPS_2026_Latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NeurIPS-style aesthetics
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
})

# ---------------------------------------------------------------------------
# Colour palette  (colourblind-friendly)
# ---------------------------------------------------------------------------
C_ALF_15 = "#2166AC"   # blue
C_ALF_7B = "#4393C3"   # lighter blue (dashed)
C_WS_15  = "#D6604D"   # warm red-orange

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
METRIC_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.+)")
KV_RE = re.compile(r"([A-Za-z0-9_/]+):([-+0-9.eE]+)")


def parse_log(path, metric_keys):
    """Return {metric_key: {step: value}}."""
    out = {k: {} for k in metric_keys}
    if not path.exists():
        print(f"[WARN] missing log: {path}")
        return out
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = METRIC_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kvs = dict(KV_RE.findall(m.group(2)))
            for k in metric_keys:
                if k in kvs:
                    try:
                        out[k][step] = float(kvs[k])
                    except ValueError:
                        pass
    for k in metric_keys:
        print(f"  [{path.name}] {k}: {len(out[k])} steps")
    return out


def to_xy(d, max_step=100):
    """Convert dict {step: val} to sorted (xs, ys) arrays."""
    if not d:
        return np.array([]), np.array([])
    items = sorted(d.items())
    xs = np.array([s for s, _ in items if s <= max_step])
    ys = np.array([v for s, v in items if s <= max_step])
    return xs, ys


def smooth(y, k=3):
    """Moving average with edge-padding."""
    if len(y) <= 1 or k <= 1:
        return y
    pad = max(1, k // 2)
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(yp, kernel, mode="valid")[: len(y)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
ALL_KEYS = [
    "dr3/disc_acc",
    "dr3/w_off_mean",
    "duet/teacher_gradient_share",
    "critic/success_onpolicy/mean",
]

RUNS = [
    {
        "label": "ALFWorld-1.5B",
        "short": "ALF-1.5B",
        "path": LOG / "alfworld_qwen1.5b_duet_v39c_postfix.log",
        "color": C_ALF_15,
        "ls": "-",
        "marker": None,
    },
    {
        "label": "ALFWorld-7B",
        "short": "ALF-7B",
        "path": LOG / "alfworld_7b_duet.log",
        "color": C_ALF_7B,
        "ls": "--",
        "marker": None,
    },
    {
        "label": "WebShop-1.5B",
        "short": "WS-1.5B",
        "path": LOG / "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log",
        "color": C_WS_15,
        "ls": "-",
        "marker": None,
    },
]

print("Parsing logs ...")
for run in RUNS:
    run["data"] = parse_log(run["path"], ALL_KEYS)

# ---------------------------------------------------------------------------
# Build figure: 3 panels, 1 row
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
SMOOTH_K = 5
MAX_STEP = 100

# ===== Panel (a): Discriminator accuracy =====
ax = axes[0]
for run in RUNS:
    xs, ys = to_xy(run["data"]["dr3/disc_acc"], MAX_STEP)
    if len(xs) == 0:
        continue
    ax.plot(xs, smooth(ys, SMOOTH_K), label=run["label"],
            color=run["color"], linestyle=run["ls"], linewidth=1.8)
ax.set_xlabel("Training Step")
ax.set_ylabel("Discriminator Accuracy")
ax.set_title("(a)  DR3 Discriminator Accuracy", fontweight="bold")
ax.set_xlim(0, MAX_STEP)
ax.set_ylim(-0.02, 1.05)
ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
ax.text(MAX_STEP * 0.72, 0.52, "chance", fontsize=7, color="gray", alpha=0.7)
# Shade warmup region (first ~5 steps where discriminator buffers fill)
ax.axvspan(0, 5, color="#f0f0f0", alpha=0.6, zorder=0)
ax.text(2.5, 0.15, "warmup", fontsize=6.5, color="#999999", ha="center",
        style="italic", rotation=90)
ax.grid(True, axis="y")
ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#CCCCCC",
          framealpha=0.9)

# ===== Panel (b): Teacher gradient share + on-policy success (secondary axis) =====
ax = axes[1]
ax2 = ax.twinx()  # secondary y-axis for success rate
ax2.spines["top"].set_visible(False)

SMOOTH_B = 7  # heavier smoothing for noisier gradient share signal

for run in RUNS:
    xs, ys = to_xy(run["data"]["duet/teacher_gradient_share"], MAX_STEP)
    if len(xs) == 0:
        continue
    ax.plot(xs, smooth(ys * 100, SMOOTH_B), label=run["label"],
            color=run["color"], linestyle=run["ls"], linewidth=1.8)

# On-policy success on secondary axis (thin, partially transparent)
for run in RUNS:
    xs, ys = to_xy(run["data"]["critic/success_onpolicy/mean"], MAX_STEP)
    if len(xs) == 0:
        continue
    ax2.plot(xs, smooth(ys * 100, SMOOTH_B),
             color=run["color"], linestyle=run["ls"], linewidth=1.0, alpha=0.35)

ax.set_xlabel("Training Step")
ax.set_ylabel("Teacher Gradient Share (%)")
ax2.set_ylabel("On-Policy Success (%)", color="#999999", fontsize=8)
ax2.tick_params(axis="y", labelcolor="#999999", labelsize=7)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("#DDDDDD")
ax.set_title("(b)  Teacher Gradient Share", fontweight="bold")
ax.set_xlim(0, MAX_STEP)
ax.set_ylim(0, 60)
ax2.set_ylim(0, 105)

# Add a text annotation explaining the thin lines
ax2.text(MAX_STEP * 0.38, 97, "thin lines = on-policy success",
         fontsize=6.5, color="#999999", style="italic")
ax.axvspan(0, 5, color="#f0f0f0", alpha=0.6, zorder=0)

ax.grid(True, axis="y")

# ===== Panel (c): Mean density ratio on teacher samples =====
ax = axes[2]
for run in RUNS:
    xs, ys = to_xy(run["data"]["dr3/w_off_mean"], MAX_STEP)
    if len(xs) == 0:
        continue
    ax.plot(xs, smooth(ys, SMOOTH_K), label=run["label"],
            color=run["color"], linestyle=run["ls"], linewidth=1.8)
ax.set_xlabel("Training Step")
ax.set_ylabel(r"Mean Density Ratio $\hat{w}$")
ax.set_title(r"(c)  Teacher Weight $\hat{w}_{\mathrm{off}}$", fontweight="bold")
ax.set_xlim(0, MAX_STEP)
ax.set_ylim(bottom=0)
ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
ax.text(MAX_STEP * 0.65, 1.03, "unweighted", fontsize=7, color="gray", alpha=0.7)
ax.axvspan(0, 5, color="#f0f0f0", alpha=0.6, zorder=0)
ax.grid(True, axis="y")

# ---------------------------------------------------------------------------
# Final layout
# ---------------------------------------------------------------------------
fig.tight_layout(w_pad=2.5)

pdf_path = OUT / "fig_dr3_dynamics.pdf"
png_path = OUT / "fig_dr3_dynamics.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight", dpi=300)
print(f"\n[saved] {pdf_path}")
print(f"[saved] {png_path}")
plt.close(fig)
