"""Final focused analysis on dr3/disc_acc as the adaptive-mu signal.

- Produces a 2x2 figure: (a) disc_acc trajectory on all variants,
  (b) implied mu from disc_acc vs true mu (WS v24), (c) ALF v24 check,
  (d) robustness — how does the schedule change with different bracket choices?

- Also computes alternative mapping: mu = mu_max * (1 - disc_acc) -> raw linear,
  no normalization. This is the cheapest, most principled mapping.
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
OUT = ROOT / "analysis_reports/_parsed/disc_acc_final.json"

N_STEPS = 100


def v24_mu_schedule(n=N_STEPS):
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


def series_of(rows, key, n=N_STEPS):
    lut = {r["step"]: r.get(key, np.nan) for r in rows}
    return np.array([lut.get(t, np.nan) for t in range(1, n + 1)], dtype=float)


def smooth(x, w=5):
    out = np.empty_like(x, dtype=float)
    half = w // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        out[i] = np.nanmean(vals) if vals.size else np.nan
    return out


def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xs, ys = x[mask], y[mask]
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def mapping_linear(disc, mu_min=0.05, mu_max=0.30, floor=0.5):
    """mu = mu_max * max(0, 2*(1-disc)) linearly above floor acc=0.5.
    When disc_acc=0.5 (can't distinguish): mu = mu_max (full strength BC).
    When disc_acc=1.0 (fully separable): mu = 0 (no BC).
    Clip to [mu_min, mu_max]."""
    disc = np.asarray(disc, dtype=float)
    mu = mu_max * np.maximum(0.0, (1 - disc) / (1 - floor))
    return np.clip(mu, mu_min, mu_max)


def mapping_minmax(disc, mu_min=0.05, mu_max=0.30):
    """Percentile normalization of disc_acc."""
    finite = disc[np.isfinite(disc)]
    if finite.size < 3:
        return np.full_like(disc, np.nan)
    lo, hi = np.nanpercentile(disc, 5), np.nanpercentile(disc, 95)
    if hi - lo < 1e-10:
        return np.full_like(disc, np.nan)
    norm = np.clip((disc - lo) / (hi - lo), 0.0, 1.0)
    return mu_min + (1 - norm) * (mu_max - mu_min)


def main():
    with open(PARSED) as f:
        data = json.load(f)
    mu_true = v24_mu_schedule()
    steps = np.arange(1, N_STEPS + 1)

    variants = ["ws_v24", "ws_v1", "ws_v12", "ws_v36", "ws_v38", "alf_v24", "alf_v1"]
    palette = {
        "ws_v24": "tab:red",
        "ws_v1":  "tab:blue",
        "ws_v12": "tab:orange",
        "ws_v36": "tab:gray",
        "ws_v38": "tab:brown",
        "alf_v24": "tab:green",
        "alf_v1":  "tab:purple",
    }
    labels = {
        "ws_v24": "v24 WS (target)",
        "ws_v1":  "v1 WS",
        "ws_v12": "v12 WS",
        "ws_v36": "v36 WS const 0.05",
        "ws_v38": "v38 WS SPW",
        "alf_v24": "v24 ALF",
        "alf_v1":  "v1 ALF",
    }

    discs = {v: series_of(data.get(v, []), "dr3/disc_acc") for v in variants}

    # Build the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    # (a) Raw disc_acc trajectories
    ax = axes[0, 0]
    ax.axhline(0.5, color="k", alpha=0.2, linestyle=":")
    ax.axhline(1.0, color="k", alpha=0.2, linestyle=":")
    for v in variants:
        d = discs[v]
        if np.isfinite(d).any():
            ax.plot(steps, smooth(d, 3), color=palette[v], label=labels[v], linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("dr3/disc_acc")
    ax.set_title("(a) Discriminator accuracy — raw trajectories")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.45, 1.03)

    # (b) Implied mu from disc_acc: principled mapping AND percentile mapping
    ax = axes[0, 1]
    ax.plot(steps, mu_true, "k--", label="v24 target mu", linewidth=2.0)
    d_ws24 = discs["ws_v24"]
    mu_linear = mapping_linear(smooth(d_ws24, 5))
    mu_mm = mapping_minmax(smooth(d_ws24, 5))
    ax.plot(steps, mu_linear, color="tab:red", linewidth=1.6,
            label=f"mu_linear = 0.30*(1-disc)/0.5  (r={safe_corr(mu_linear, mu_true):.2f})")
    ax.plot(steps, mu_mm, color="darkorange", linewidth=1.4, linestyle="-.",
            label=f"mu_pct = percentile-map  (r={safe_corr(mu_mm, mu_true):.2f})")
    # Also show what v38's SPW mu actually did
    # Grab chord/mu series from ws_v38 if exists
    mu_v38 = series_of(data.get("ws_v38", []), "chord/mu")
    if np.isfinite(mu_v38).any():
        ax.plot(steps, mu_v38, color="tab:brown", alpha=0.5, linewidth=1.0, linestyle=":",
                label="v38 actual mu (SPW failed)")
    ax.set_xlabel("step")
    ax.set_ylabel("mu")
    ax.set_title("(b) Implied mu from disc_acc (WS v24): two mappings")
    ax.set_ylim(-0.02, 0.36)
    ax.legend(fontsize=7, loc="upper right")

    # (c) ALFWorld generalization: disc_acc + implied mu on ALF v24
    ax = axes[1, 0]
    ax.plot(steps, mu_true, "k--", label="v24 target mu (WS hand-tuned)", linewidth=1.7)
    d_alf = discs["alf_v24"]
    mu_alf = mapping_linear(smooth(d_alf, 5))
    ax.plot(steps, mu_alf, color="tab:green", label="ALF v24 implied mu (linear rule)", linewidth=1.6)
    # Also show ALF v1 for sanity
    d_alf1 = discs["alf_v1"]
    mu_alf1 = mapping_linear(smooth(d_alf1, 5))
    if np.isfinite(mu_alf1).any():
        ax.plot(steps, mu_alf1, color="tab:purple", label="ALF v1 implied mu", linewidth=1.2, linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("implied mu")
    ax.set_title("(c) ALFWorld self-adjustment: implied mu pins near mu_min (good)")
    ax.set_ylim(-0.02, 0.36)
    ax.legend(fontsize=8, loc="upper right")

    # (d) Knee sensitivity — show smoothing/floor variants for the linear mapping
    ax = axes[1, 1]
    ax.plot(steps, mu_true, "k--", label="v24 target mu", linewidth=2)
    for fl, clr in [(0.3, "tab:green"), (0.5, "tab:red"), (0.7, "tab:blue")]:
        mu_v = mapping_linear(smooth(discs["ws_v24"], 5), floor=fl)
        r = safe_corr(mu_v, mu_true)
        ax.plot(steps, mu_v, color=clr, linewidth=1.3, label=f"floor={fl} (r={r:.2f})")
    ax.set_xlabel("step")
    ax.set_ylabel("implied mu")
    ax.set_title("(d) Rule robustness: sensitivity to floor acc")
    ax.set_ylim(-0.02, 0.36)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Adaptive-mu discovery: dr3/disc_acc is the leading candidate signal", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / "fig_disc_acc_adaptive_rule.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")

    # Final stats dump
    stats = {
        "mappings": {},
        "alf_self_adjust": {},
        "ws_sanity": {},
    }
    for floor in [0.3, 0.5, 0.7]:
        m = mapping_linear(smooth(discs["ws_v24"], 5), floor=floor)
        stats["mappings"][f"linear_floor_{floor}"] = {
            "pearson_r": safe_corr(m, mu_true),
            "mae_vs_target": float(np.nanmean(np.abs(m - mu_true))),
            "at_step1": float(m[0]),
            "at_step10": float(m[9]),
            "at_step25": float(m[24]),
            "at_step50": float(m[49]),
            "at_step100": float(m[99]),
        }
    mu_mm = mapping_minmax(smooth(discs["ws_v24"], 5))
    stats["mappings"]["percentile"] = {
        "pearson_r": safe_corr(mu_mm, mu_true),
        "mae_vs_target": float(np.nanmean(np.abs(mu_mm - mu_true))),
        "at_step1": float(mu_mm[0]),
        "at_step10": float(mu_mm[9]),
        "at_step25": float(mu_mm[24]),
        "at_step50": float(mu_mm[49]),
        "at_step100": float(mu_mm[99]),
    }
    for v in ["ws_v24", "ws_v1", "ws_v12", "ws_v36", "ws_v38", "alf_v24", "alf_v1"]:
        d = smooth(discs[v], 5)
        mu_v = mapping_linear(d)
        stats["alf_self_adjust"][v] = {
            "disc_acc_mean": float(np.nanmean(d)),
            "disc_acc_step25": float(d[24]) if np.isfinite(d[24]) else None,
            "disc_acc_step100": float(d[99]) if np.isfinite(d[99]) else None,
            "implied_mu_mean": float(np.nanmean(mu_v)),
            "implied_mu_step25": float(mu_v[24]) if np.isfinite(mu_v[24]) else None,
            "implied_mu_step100": float(mu_v[99]) if np.isfinite(mu_v[99]) else None,
        }
    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else o)
    print(f"Wrote {OUT}")

    # Print final table
    print("\n=== Final numbers ===")
    print("Mapping                     r       MAE     mu@1    mu@25   mu@50   mu@100")
    for mn, m in stats["mappings"].items():
        print(f"{mn:26s} {m['pearson_r']:6.3f}  {m['mae_vs_target']:6.4f}  "
              f"{m['at_step1']:6.3f}  {m['at_step25']:6.3f}  {m['at_step50']:6.3f}  {m['at_step100']:6.3f}")

    print("\nCross-variant disc_acc & implied mu (linear rule, floor=0.5):")
    print(f"{'variant':10s} {'disc_mean':>10s} {'disc@25':>8s} {'disc@100':>9s} {'mu_mean':>8s} {'mu@25':>7s} {'mu@100':>7s}")
    for v, s in stats["alf_self_adjust"].items():
        def fmt(x):
            return f"{x:.3f}" if x is not None else "   -  "
        print(f"{v:10s} {fmt(s['disc_acc_mean']):>10s} {fmt(s['disc_acc_step25']):>8s} {fmt(s['disc_acc_step100']):>9s} "
              f"{fmt(s['implied_mu_mean']):>8s} {fmt(s['implied_mu_step25']):>7s} {fmt(s['implied_mu_step100']):>7s}")


if __name__ == "__main__":
    main()
