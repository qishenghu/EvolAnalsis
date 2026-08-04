"""
Build Figure 2 (training curves) and Figure 3 (self-attenuation panels) for
the DUET NeurIPS paper.

Figure 2 — three training curves on 1.5B-WebShop over 100 steps:
  - OnPolicy GRPO   (logs/webshop_qwen1.5b_onpolicy.log)
  - CHORD           (logs/webshop_qwen1.5b_chord.log)
  - DUET (Ours)     (logs/webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log)
  Metric: critic/success_onpolicy/mean

Figure 3 — three panels on the DUET 1.5B-WebShop run:
  Panel A: dr3/w_off_mean (decreases ~0.50 -> ~0.05 as discriminator strengthens)
  Panel B: chord/mu (peak 0.30, decays to ~0.10 once disc_acc crosses d_floor)
  Panel C: dr3/disc_acc (increases ~0.30 -> ~0.99)

Output:
  /NeurIPS_2026_Latex/figures/fig2_training_curves.pdf
  /NeurIPS_2026_Latex/figures/fig3_self_attenuation.pdf
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/home/qisheng/EvolAnalsis")
LOG = ROOT / "logs"
OUT = ROOT / "NeurIPS_2026_Latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# NeurIPS-style aesthetics
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
    }
)


METRIC_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.+)")
KV_RE = re.compile(r"([A-Za-z0-9_/]+):([-+0-9.eE]+)")


def parse_log(path: Path, metric_keys):
    """Parse a training log; for each requested metric_key, return dict
    {global_step (or step): value} keyed on its log step. Returns dict-of-dicts."""
    out = {k: {} for k in metric_keys}
    if not path.exists():
        print(f"[WARN] missing log: {path}")
        return out
    seen_steps = set()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = METRIC_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            seen_steps.add(step)
            tail = m.group(2)
            kvs = dict(KV_RE.findall(tail))
            for k in metric_keys:
                if k in kvs:
                    try:
                        out[k][step] = float(kvs[k])
                    except ValueError:
                        pass
    print(
        f"[parse] {path.name}: {len(seen_steps)} unique steps; "
        + ", ".join(f"{k}={len(v)}" for k, v in out.items())
    )
    return out


def to_xy(d, max_step=100):
    if not d:
        return np.array([]), np.array([])
    items = sorted(d.items())
    xs = np.array([s for s, _ in items if s <= max_step])
    ys = np.array([v for s, v in items if s <= max_step])
    return xs, ys


def smooth(y, k=5):
    if len(y) <= 1:
        return y
    pad = max(1, k // 2)
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(yp, kernel, mode="valid")


# -------------------- Figure 2 --------------------
KEYS_F2 = ["critic/success_onpolicy/mean"]
RUNS = [
    ("OnPolicy GRPO", LOG / "webshop_qwen1.5b_onpolicy.log", "#7B7B7B", "--"),
    ("CHORD", LOG / "webshop_qwen1.5b_chord.log", "#1F77B4", "-."),
    ("DUET (Ours)", LOG / "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log", "#D62728", "-"),
]

fig, ax = plt.subplots(figsize=(4.2, 2.6))
for label, path, color, ls in RUNS:
    parsed = parse_log(path, KEYS_F2)
    xs, ys = to_xy(parsed["critic/success_onpolicy/mean"], max_step=100)
    if len(xs) == 0:
        print(f"[WARN] no data for {label}")
        continue
    ax.plot(xs, smooth(ys, k=5) * 100.0, label=label, color=color, linestyle=ls)
ax.set_xlabel("Training step")
ax.set_ylabel("On-policy success rate (\\%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, None)
ax.grid(True, axis="y")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_training_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_training_curves.png", bbox_inches="tight", dpi=200)
print(f"[saved] {OUT / 'fig2_training_curves.pdf'}")


# -------------------- Figure 3 --------------------
KEYS_F3 = ["dr3/w_off_mean", "chord/mu", "dr3/disc_acc"]
DUET_LOG = LOG / "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06.log"
parsed = parse_log(DUET_LOG, KEYS_F3)

fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.4), sharex=True)

# Panel A — effective teacher replay weight (w_hat on teacher samples)
xs, ys = to_xy(parsed["dr3/w_off_mean"], max_step=100)
axes[0].plot(xs, smooth(ys, k=3), color="#D62728")
axes[0].set_title(r"(A) Effective teacher replay weight $\hat w_\tau$")
axes[0].set_ylabel("Effective replay weight")
axes[0].set_ylim(bottom=0.0)
axes[0].grid(True, axis="y")

# Panel B — adaptive mu(t)
xs, ys = to_xy(parsed["chord/mu"], max_step=100)
axes[1].plot(xs, ys, color="#1F77B4")
axes[1].set_title(r"(B) Adaptive BC weight $\mu(t)$")
axes[1].set_ylabel(r"$\mu(t)$")
axes[1].set_ylim(0.0, 0.35)
axes[1].grid(True, axis="y")

# Panel C — discriminator accuracy
xs, ys = to_xy(parsed["dr3/disc_acc"], max_step=100)
axes[2].plot(xs, smooth(ys, k=3), color="#2CA02C")
axes[2].set_title(r"(C) Discriminator accuracy $\alpha(t)$")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0.0, 1.05)
axes[2].grid(True, axis="y")

for ax in axes:
    ax.set_xlabel("Training step")
    ax.set_xlim(0, 100)

fig.tight_layout()
fig.savefig(OUT / "fig3_self_attenuation.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_self_attenuation.png", bbox_inches="tight", dpi=200)
print(f"[saved] {OUT / 'fig3_self_attenuation.pdf'}")
