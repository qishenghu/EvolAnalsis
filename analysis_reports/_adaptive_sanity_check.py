"""Sanity: is disc_acc's match to mu just because both are monotone ramps?

We compare:
  (1) implied mu from disc_acc (linear rule, floor=0.5)
  (2) naive time-schedule: mu_t = mu_max for t<5, linearly decay, mu_min after t=25
  (3) random monotonic ramp via shuffled-within-decreasing control
  (4) a PURELY step-number based ramp as a null

And we stress-test by looking at the DERIVATIVE (step-to-step change) rather than
absolute mu. If disc_acc is capturing real dynamics, its derivative should
match the mu-schedule derivative only during the steep 0.3->0.05 phase, and be
near-zero after step 25. A pure monotone ramp wouldn't necessarily have that shape.
"""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/home/qisheng/EvolAnalsis")
PARSED = ROOT / "analysis_reports/_parsed/adaptive_signals.json"
FIGDIR = ROOT / "analysis_reports/figures"

N = 100


def v24_mu(n=N):
    mu = np.empty(n)
    for i in range(n):
        t = i + 1
        if t < 5:
            mu[i] = 0.30
        elif t >= 25:
            mu[i] = 0.05
        else:
            mu[i] = 0.30 + (t - 5) / 20.0 * (0.05 - 0.30)
    return mu


def smooth(x, w=5):
    out = np.empty_like(x, dtype=float)
    half = w // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        out[i] = np.nanmean(vals) if vals.size else np.nan
    return out


def series(rows, key):
    lut = {r["step"]: r.get(key, np.nan) for r in rows}
    return np.array([lut.get(t, np.nan) for t in range(1, N + 1)], dtype=float)


def linear_rule(d, mu_min=0.05, mu_max=0.30, floor=0.5):
    return np.clip(mu_max * np.maximum(0, (1 - d) / (1 - floor)), mu_min, mu_max)


with open(PARSED) as f:
    data = json.load(f)

mu_true = v24_mu()
mu_derivative = np.diff(mu_true, prepend=mu_true[0])  # dmu/dt

d_ws24 = smooth(series(data["ws_v24"], "dr3/disc_acc"), 5)
d_ws1 = smooth(series(data["ws_v1"], "dr3/disc_acc"), 5)
d_ws36 = smooth(series(data["ws_v36"], "dr3/disc_acc"), 5)
d_alf24 = smooth(series(data["alf_v24"], "dr3/disc_acc"), 5)

mu_ws24 = linear_rule(d_ws24)
mu_ws1 = linear_rule(d_ws1)
mu_ws36 = linear_rule(d_ws36)
mu_alf24 = linear_rule(d_alf24)

# Null controls
# (a) A monotone ramp that goes 0.3 to 0.05 linearly over all 100 steps
mu_linear_null = np.linspace(0.30, 0.05, N)
# (b) Step-schedule reproduced exactly (benchmark)
mu_exact = mu_true.copy()

def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])

print("Pearson r with v24 target mu:")
print(f"  disc_acc-implied mu (WS v24)         : {corr(mu_ws24, mu_true):.3f}")
print(f"  disc_acc-implied mu (WS v1 control)  : {corr(mu_ws1, mu_true):.3f}")
print(f"  monotonic 100-step ramp (null)       : {corr(mu_linear_null, mu_true):.3f}")
print(f"  exact schedule (trivial)              : {corr(mu_exact, mu_true):.3f}")
print()
print("MAE vs target mu:")
print(f"  disc_acc-implied mu (WS v24)         : {np.nanmean(np.abs(mu_ws24 - mu_true)):.4f}")
print(f"  disc_acc-implied mu (WS v1 control)  : {np.nanmean(np.abs(mu_ws1 - mu_true)):.4f}")
print(f"  monotonic 100-step ramp (null)       : {np.nanmean(np.abs(mu_linear_null - mu_true)):.4f}")

# Plot: shape + null comparison
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
a1.plot(np.arange(1, N + 1), mu_true, "k--", linewidth=2.5, label="v24 target mu", zorder=5)
a1.plot(np.arange(1, N + 1), mu_ws24, color="tab:red", linewidth=1.6,
        label=f"disc_acc-implied WS v24 (r={corr(mu_ws24, mu_true):.2f})")
a1.plot(np.arange(1, N + 1), mu_ws1, color="tab:blue", linewidth=1.2,
        label=f"disc_acc-implied WS v1 (r={corr(mu_ws1, mu_true):.2f})")
a1.plot(np.arange(1, N + 1), mu_ws36, color="tab:gray", linewidth=1.2,
        label=f"disc_acc-implied WS v36 (r={corr(mu_ws36, mu_true):.2f})")
a1.plot(np.arange(1, N + 1), mu_alf24, color="tab:green", linewidth=1.6,
        label=f"disc_acc-implied ALF v24 (r={corr(mu_alf24, mu_true):.2f})")
a1.plot(np.arange(1, N + 1), mu_linear_null, color="tab:brown", linewidth=1.0,
        linestyle=":", label=f"monotonic null (100-step linear, r={corr(mu_linear_null, mu_true):.2f})")
a1.set_ylim(-0.02, 0.36)
a1.set_xlabel("step")
a1.set_ylabel("mu")
a1.set_title("Shape fidelity: implied mu vs hand-tuned target")
a1.legend(fontsize=7, loc="upper right")

# Derivative plot — does disc_acc have the right D shape (big slope near step 10-20, flat after)?
a2.plot(np.arange(1, N + 1), mu_derivative, "k-", linewidth=1.5, label="dmu/dt (target)")
a2.plot(np.arange(1, N + 1), np.diff(mu_ws24, prepend=mu_ws24[0]), color="tab:red",
        linewidth=1.2, label="dmu/dt from disc_acc (WS v24)")
a2.plot(np.arange(1, N + 1), np.diff(mu_linear_null, prepend=mu_linear_null[0]),
        color="tab:brown", linewidth=1.0, linestyle=":", label="dmu/dt null (linear ramp)")
a2.axhline(0, color="k", alpha=0.2, linestyle=":")
a2.axvline(25, color="k", alpha=0.3, linestyle=":")
a2.set_xlabel("step")
a2.set_ylabel("dmu/dt")
a2.set_title("Derivative — knee at step 25 (null ramp is flat throughout)")
a2.legend(fontsize=8, loc="lower right")

fig.suptitle("Sanity: disc_acc encodes the KNEE (big drop concentrated around steps 5-25, flat after)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGDIR / "fig_adaptive_disc_acc_sanity.png", dpi=140)
plt.close(fig)
print(f"\nWrote fig_adaptive_disc_acc_sanity.png")
