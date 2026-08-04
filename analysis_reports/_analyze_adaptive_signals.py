"""Analyze candidate adaptive-mu signals.

Computes per-step normalized trajectories and correlates them with v24's
hand-tuned mu schedule (0.3 -> 0.05 linearly over steps 5..25, then 0.05).

Produces:
  analysis_reports/figures/fig_adaptive_signal_candidates.png
  analysis_reports/_parsed/adaptive_signal_stats.json
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
OUTSTATS = ROOT / "analysis_reports/_parsed/adaptive_signal_stats.json"
FIGDIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 100


def v24_mu_schedule(n=N_STEPS):
    """0.3 for steps<5, linear 0.3->0.05 over 5..25, 0.05 after."""
    mu = np.empty(n)
    for i in range(n):
        t = i + 1  # 1-indexed step
        if t < 5:
            mu[i] = 0.30
        elif t >= 25:
            mu[i] = 0.05
        else:
            # linear interp from 0.3 at t=5 to 0.05 at t=25
            frac = (t - 5) / (25 - 5)
            mu[i] = 0.30 + frac * (0.05 - 0.30)
    return mu


def series_of(rows, key, n=N_STEPS):
    """Return a length-n series (steps 1..n) with NaN where missing."""
    lut = {r["step"]: r.get(key, np.nan) for r in rows}
    return np.array([lut.get(t, np.nan) for t in range(1, n + 1)], dtype=float)


def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xs, ys = x[mask], y[mask]
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def minmax_norm(x, lo, hi, smooth_window=5):
    """Normalize signal to [lo, hi] using robust min/max, with optional smoothing.
    If higher x means 'closer to teacher' (NLL low late), we MAP the INVERSE
    to high mu is not automatic; we provide two candidate mappings.
    Returns two arrays: one where low x -> lo, high x -> hi (direct)
                       one where low x -> hi, high x -> lo (inverse).
    """
    x = np.asarray(x, dtype=float)
    # smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        finite = np.where(np.isfinite(x), x, np.nan)
        # pad-edge reflect
        smoothed = np.empty_like(x)
        half = smooth_window // 2
        for i in range(len(x)):
            lo_i = max(0, i - half)
            hi_i = min(len(x), i + half + 1)
            vals = finite[lo_i:hi_i]
            vals = vals[np.isfinite(vals)]
            smoothed[i] = np.nanmean(vals) if vals.size else np.nan
        x = smoothed

    # Robust normalization: use 5th/95th percentile from finite
    finite = x[np.isfinite(x)]
    if finite.size < 3:
        return np.full_like(x, np.nan), np.full_like(x, np.nan)
    lo_x, hi_x = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    if hi_x - lo_x < 1e-10:
        return np.full_like(x, np.nan), np.full_like(x, np.nan)
    norm = np.clip((x - lo_x) / (hi_x - lo_x), 0.0, 1.0)
    direct = lo + norm * (hi - lo)
    inverse = lo + (1.0 - norm) * (hi - lo)
    return direct, inverse


def main():
    with open(PARSED) as f:
        data = json.load(f)

    # Build rows lookup
    variants = list(data.keys())
    steps = np.arange(1, N_STEPS + 1)
    mu_true = v24_mu_schedule(N_STEPS)

    # Candidate signals keyed by (label, wandb_key, mapping_hint)
    # mapping_hint: 'inverse' if LOW signal -> HIGH mu (e.g., kl_loss: late policy drifted so reduce BC)
    #               'direct'  if HIGH signal -> HIGH mu (e.g., sft_loss: teacher NLL high early)
    # We'll also show the flipped version's correlation in the table.
    candidates = [
        ("A chord/sft_loss",                   "chord/sft_loss",              "direct"),
        ("A2 chord/sft_loss_unweighted_mean",  "chord/sft_loss_unweighted_mean", "direct"),
        ("B chord/log_prob_mean",              "chord/log_prob_mean",         "inverse"),  # more negative early
        ("D duet/teacher_gradient_share",      "duet/teacher_gradient_share", "direct"),
        ("E actor/kl_loss",                    "actor/kl_loss",               "inverse"),
        ("F dr3/disc_acc",                     "dr3/disc_acc",                "inverse"),
        ("G duet/group_reward_variance_mean",  "duet/group_reward_variance_mean", "direct"),
        ("H response_length/mean",             "response_length/mean",         "inverse"),
        ("I dr3/w_off_mean",                   "dr3/w_off_mean",              "direct"),
    ]

    # Collect series for each variant we care about
    tracked_variants = ["ws_v24", "ws_v1", "ws_v12", "ws_v36"]
    rows_by_variant = {v: data.get(v, []) for v in tracked_variants}
    rows_alf24 = data.get("alf_v24", [])
    rows_alf1 = data.get("alf_v1", [])
    rows_ws38 = data.get("ws_v38", [])

    # Produce stats table
    stats = {
        "mu_schedule": mu_true.tolist(),
        "candidates": [],
    }

    # For the figure: 3x3 panels one per candidate
    ncols = 3
    nrows = int(math.ceil(len(candidates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), sharex=True)
    axes = axes.flatten()

    palette = {
        "ws_v24": ("tab:red", "v24 webshop (target)"),
        "ws_v1":  ("tab:blue", "v1 no BC"),
        "ws_v12": ("tab:orange", "v12 no BC variant"),
        "ws_v36": ("tab:gray", "v36 const mu=0.05"),
    }

    for idx, (label, key, hint) in enumerate(candidates):
        ax = axes[idx]
        ax2 = ax.twinx()
        # per-variant raw signals plotted on ax2, mu on ax
        entry = {"label": label, "key": key, "mapping_hint": hint,
                 "variant_stats": {}}
        # mu on left axis
        ax.plot(steps, mu_true, color="black", linestyle="--", linewidth=1.7, label="v24 true mu", zorder=5)
        ax.set_ylim(-0.02, 0.36)

        for v in tracked_variants:
            rows = rows_by_variant[v]
            x = series_of(rows, key, N_STEPS)
            if not np.isfinite(x).any():
                continue
            color, vlabel = palette[v]
            ax2.plot(steps, x, color=color, alpha=0.8, linewidth=1.2, label=vlabel)

            # Compute correlations for v24 webshop case only (target)
            if v == "ws_v24":
                r_raw = safe_corr(x, mu_true)
                direct, inverse = minmax_norm(x, 0.05, 0.3, smooth_window=5)
                r_dir = safe_corr(direct, mu_true)
                r_inv = safe_corr(inverse, mu_true)
                entry["variant_stats"][v] = {
                    "pearson_raw": r_raw,
                    "pearson_norm_direct": r_dir,
                    "pearson_norm_inverse": r_inv,
                    "implied_mu_direct": direct.tolist() if np.isfinite(direct).any() else None,
                    "implied_mu_inverse": inverse.tolist() if np.isfinite(inverse).any() else None,
                    "value_step1": float(x[0]) if np.isfinite(x[0]) else None,
                    "value_step25": float(x[24]) if np.isfinite(x[24]) else None,
                    "value_step100": float(x[99]) if np.isfinite(x[99]) else None,
                }
            else:
                x_arr = x
                entry["variant_stats"][v] = {
                    "value_step1": float(x_arr[0]) if np.isfinite(x_arr[0]) else None,
                    "value_step25": float(x_arr[24]) if np.isfinite(x_arr[24]) else None,
                    "value_step100": float(x_arr[99]) if np.isfinite(x_arr[99]) else None,
                    "mean": float(np.nanmean(x_arr)),
                }

        # ALFWorld v24 overlay (dotted green)
        x_alf = series_of(rows_alf24, key, N_STEPS)
        if np.isfinite(x_alf).any():
            ax2.plot(steps, x_alf, color="tab:green", alpha=0.6, linewidth=1.2, linestyle=":", label="alf v24")
            entry["variant_stats"]["alf_v24"] = {
                "value_step1": float(x_alf[0]) if np.isfinite(x_alf[0]) else None,
                "value_step25": float(x_alf[24]) if np.isfinite(x_alf[24]) else None,
                "value_step100": float(x_alf[99]) if np.isfinite(x_alf[99]) else None,
                "mean": float(np.nanmean(x_alf)),
            }
        # ALFWorld v1 overlay (dotted magenta)
        x_alf1 = series_of(rows_alf1, key, N_STEPS)
        if np.isfinite(x_alf1).any():
            ax2.plot(steps, x_alf1, color="tab:purple", alpha=0.45, linewidth=1.0, linestyle=":", label="alf v1")
            entry["variant_stats"]["alf_v1"] = {
                "value_step1": float(x_alf1[0]) if np.isfinite(x_alf1[0]) else None,
                "value_step25": float(x_alf1[24]) if np.isfinite(x_alf1[24]) else None,
                "value_step100": float(x_alf1[99]) if np.isfinite(x_alf1[99]) else None,
                "mean": float(np.nanmean(x_alf1)),
            }

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("mu (target)", color="black", fontsize=8)
        ax2.set_ylabel("signal", color="dimgray", fontsize=8)
        # Legend only once per panel, small
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")
        stats["candidates"].append(entry)

    # hide unused axes
    for k in range(len(candidates), len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        "Adaptive-mu signal discovery: does any wandb metric track v24's hand-tuned mu (0.30→0.05 over steps 5-25)?",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig_adaptive_signal_candidates.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")

    with open(OUTSTATS, "w") as f:
        json.dump(stats, f, indent=2, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else o)
    print(f"Wrote {OUTSTATS}")

    # Print correlation table for the paper writeup
    print("\n=== Correlation with v24 target mu (Pearson r on ws_v24) ===")
    print(f"{'signal':40s} {'hint':8s} {'r_raw':>8s} {'r_direct':>10s} {'r_inverse':>10s}")
    for e in stats["candidates"]:
        s = e["variant_stats"].get("ws_v24", {})
        print(f"{e['label']:40s} {e['mapping_hint']:8s} {s.get('pearson_raw', float('nan')):>8.3f} "
              f"{s.get('pearson_norm_direct', float('nan')):>10.3f} {s.get('pearson_norm_inverse', float('nan')):>10.3f}")

    # Build the cheapest empirical-check figure: for the best candidate, overlay implied mu vs true mu
    # Identify best by abs correlation after inverse (if hint=inverse) else direct
    best = None
    for e in stats["candidates"]:
        s = e["variant_stats"].get("ws_v24", {})
        r_dir = s.get("pearson_norm_direct")
        r_inv = s.get("pearson_norm_inverse")
        if r_dir is None and r_inv is None:
            continue
        # choose mapping matching hint
        if e["mapping_hint"] == "inverse":
            r = r_inv
            mu_imp = s.get("implied_mu_inverse")
        else:
            r = r_dir
            mu_imp = s.get("implied_mu_direct")
        if mu_imp is None:
            continue
        if best is None or (r is not None and abs(r) > abs(best[0])):
            best = (r, e["label"], mu_imp)

    if best is not None:
        fig2, axb = plt.subplots(1, 1, figsize=(7, 3.5))
        axb.plot(steps, mu_true, "k--", label="v24 hand-tuned mu (target)", linewidth=1.8)
        axb.plot(steps, best[2], color="tab:red", label=f"implied mu from {best[1]} (r={best[0]:.2f})")
        axb.set_xlabel("step")
        axb.set_ylabel("mu")
        axb.set_title(f"Cheapest offline check — best candidate: {best[1]}")
        axb.legend()
        fig2.tight_layout()
        out2 = FIGDIR / "fig_adaptive_signal_implied_mu.png"
        fig2.savefig(out2, dpi=140)
        plt.close(fig2)
        print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
