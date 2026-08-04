"""Detailed shape-matching analysis + ALFWorld generalization check.

Focus on SHAPE not just correlation:
  (1) does signal have a knee at step 25?
  (2) does signal self-adjust on ALFWorld (stay small, pinning mu near mu_min)?
  (3) offline "implied mu" under robust mapping: does it reproduce 0.30->0.05?

Writes:
  analysis_reports/figures/fig_adaptive_signal_shape_detail.png
  analysis_reports/figures/fig_adaptive_signal_alfworld_check.png
  analysis_reports/_parsed/adaptive_signal_shape.json
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
OUTSTATS = ROOT / "analysis_reports/_parsed/adaptive_signal_shape.json"
FIGDIR.mkdir(parents=True, exist_ok=True)

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
            frac = (t - 5) / (25 - 5)
            mu[i] = 0.30 + frac * (0.05 - 0.30)
    return mu


def series_of(rows, key, n=N_STEPS):
    lut = {r["step"]: r.get(key, np.nan) for r in rows}
    return np.array([lut.get(t, np.nan) for t in range(1, n + 1)], dtype=float)


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    half = w // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        out[i] = np.nanmean(vals) if vals.size else np.nan
    return out


def knee_step(x):
    """Find the step after which signal is within 10% of its final plateau.
    Simple: return the first step where signal falls below
    min + 0.1*(max-min) (if signal is decreasing) OR
    above max - 0.1*(max-min) (if signal is increasing)."""
    xs = smooth(x, 5)
    xf = xs[np.isfinite(xs)]
    if xf.size < 20:
        return None, None
    lo, hi = xf.min(), xf.max()
    rng = hi - lo
    if rng < 1e-10:
        return None, None
    # direction
    first_half = np.nanmean(xs[:20])
    second_half = np.nanmean(xs[-20:])
    if first_half > second_half:
        # decreasing; knee = first step where signal <= lo + 0.1*rng
        thresh = lo + 0.10 * rng
        for i, v in enumerate(xs):
            if np.isfinite(v) and v <= thresh:
                return i + 1, "decreasing"
    else:
        thresh = hi - 0.10 * rng
        for i, v in enumerate(xs):
            if np.isfinite(v) and v >= thresh:
                return i + 1, "increasing"
    return None, None


def robust_norm_to_mu(x, mu_min=0.05, mu_max=0.30, invert=False, smooth_w=5):
    xs = smooth(x, smooth_w)
    finite = xs[np.isfinite(xs)]
    if finite.size < 3:
        return np.full_like(xs, np.nan)
    lo = np.nanpercentile(xs, 5)
    hi = np.nanpercentile(xs, 95)
    if hi - lo < 1e-10:
        return np.full_like(xs, np.nan)
    norm = np.clip((xs - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    return mu_min + norm * (mu_max - mu_min)


def main():
    with open(PARSED) as f:
        data = json.load(f)

    mu_true = v24_mu_schedule(N_STEPS)
    steps = np.arange(1, N_STEPS + 1)

    # True mu knee: first step where mu hits mu_min + 0.1*(mu_max-mu_min)
    #   since mu is linearly interpolated from 5 to 25, this threshold is hit at step ~22.5
    mu_knee = knee_step(mu_true)[0]

    # candidate signals + mappings (best from pass 1)
    candidates = [
        ("dr3/disc_acc",                    "inverse"),  # r=0.97
        ("chord/sft_loss",                  "direct"),   # r=0.87
        ("chord/log_prob_mean",             "inverse"),  # r=0.87
        ("duet/teacher_gradient_share",     "direct"),   # r=0.88
        ("actor/kl_loss",                   "inverse"),  # r=0.73
        ("duet/group_reward_variance_mean", "direct"),   # r=0.81
        ("dr3/w_off_mean",                  "direct"),   # r=0.81
    ]

    tracked = ["ws_v24", "ws_v1", "ws_v12", "ws_v36"]
    palette = {
        "ws_v24": ("tab:red", "v24 WS (target)"),
        "ws_v1":  ("tab:blue", "v1 WS (no BC)"),
        "ws_v12": ("tab:orange", "v12 WS (no BC alt)"),
        "ws_v36": ("tab:gray", "v36 WS (const 0.05)"),
    }

    shape_stats = {"mu_knee_true": int(mu_knee) if mu_knee else None, "candidates": []}

    # Shape-detail figure: implied mu vs true mu for each candidate (WebShop v24)
    n = len(candidates)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for idx, (key, hint) in enumerate(candidates):
        ax = axes[idx]
        ax.plot(steps, mu_true, "k--", linewidth=1.7, label="v24 target mu")
        # True mu curve highlighting knee
        ax.axvline(mu_knee, color="k", alpha=0.2, linestyle=":")

        row_info = {"key": key, "mapping": hint}

        # Plot implied mu for each WS variant, but mark knee for v24
        for v in tracked:
            rows = data.get(v, [])
            x = series_of(rows, key)
            if not np.isfinite(x).any():
                continue
            mu_imp = robust_norm_to_mu(x, invert=(hint == "inverse"))
            color, vlabel = palette[v]
            ax.plot(steps, mu_imp, color=color, alpha=0.8, linewidth=1.2, label=f"implied mu ({vlabel})")
            if v == "ws_v24":
                ks, direction = knee_step(x)
                row_info["ws_v24_signal_knee"] = int(ks) if ks else None
                row_info["ws_v24_direction"] = direction
                row_info["ws_v24_implied_mu_at_step1"] = float(mu_imp[0])
                row_info["ws_v24_implied_mu_at_step25"] = float(mu_imp[24])
                row_info["ws_v24_implied_mu_at_step100"] = float(mu_imp[99])
                # shape error: MAE vs target
                err = np.nanmean(np.abs(mu_imp - mu_true))
                row_info["ws_v24_mae_vs_target"] = float(err)

        # Overlay ALFWorld v24 implied mu — dotted green
        rows_alf = data.get("alf_v24", [])
        x_alf = series_of(rows_alf, key)
        if np.isfinite(x_alf).any():
            # Use SAME percentile scale as WebShop v24 (so implied mu is meaningful cross-env).
            x_ws = series_of(data.get("ws_v24", []), key)
            xs_ws = smooth(x_ws, 5)
            lo_ws = np.nanpercentile(xs_ws, 5)
            hi_ws = np.nanpercentile(xs_ws, 95)
            xs_alf = smooth(x_alf, 5)
            if hi_ws - lo_ws > 1e-10:
                norm = np.clip((xs_alf - lo_ws) / (hi_ws - lo_ws), 0.0, 1.0)
                if hint == "inverse":
                    norm = 1 - norm
                mu_alf_cross = 0.05 + norm * (0.30 - 0.05)
                ax.plot(steps, mu_alf_cross, color="tab:green", alpha=0.7, linewidth=1.0,
                        linestyle=":", label="ALF v24 (cross-env scale)")
                row_info["alf_v24_mu_at_step25_cross"] = float(mu_alf_cross[24])
                row_info["alf_v24_mu_mean_cross"] = float(np.nanmean(mu_alf_cross))
                # Raw value at step 25
                row_info["alf_v24_raw_step25"] = float(x_alf[24]) if np.isfinite(x_alf[24]) else None
                row_info["ws_v24_raw_step25"] = float(x_ws[24]) if np.isfinite(x_ws[24]) else None

        ax.set_title(f"{key}  (hint={hint})", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("implied mu")
        ax.set_ylim(-0.02, 0.38)
        ax.legend(fontsize=6, loc="upper right")
        shape_stats["candidates"].append(row_info)

    for k in range(len(candidates), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Shape-detail: implied mu from each candidate signal (robust [P5,P95] -> [0.05,0.30])", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / "fig_adaptive_signal_shape_detail.png", dpi=140)
    plt.close(fig)
    print(f"Wrote fig_adaptive_signal_shape_detail.png")

    # ALFWorld check — raw signal trajectories on alf_v24 vs ws_v24 side-by-side
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows))
    axes = axes.flatten()
    for idx, (key, hint) in enumerate(candidates):
        ax = axes[idx]
        ax2 = ax.twinx()
        rows_ws = data.get("ws_v24", [])
        rows_alf = data.get("alf_v24", [])
        x_ws = series_of(rows_ws, key)
        x_alf = series_of(rows_alf, key)

        ax.plot(steps, mu_true, "k--", linewidth=1.5, label="v24 target mu")

        if np.isfinite(x_ws).any():
            ax2.plot(steps, x_ws, color="tab:red", label="WS v24", linewidth=1.3)
        if np.isfinite(x_alf).any():
            ax2.plot(steps, x_alf, color="tab:green", label="ALF v24", linewidth=1.3)

        # Annotate mean values
        ws_mean = np.nanmean(x_ws)
        alf_mean = np.nanmean(x_alf)
        ax.set_title(f"{key}  (WS mean={ws_mean:.3f} / ALF mean={alf_mean:.3f})", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("mu")
        ax2.set_ylabel(key, color="dimgray", fontsize=7)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="center right")

    for k in range(len(candidates), len(axes)):
        axes[k].axis("off")
    fig.suptitle("ALFWorld generalization check — does signal self-adjust on ALF (shrink toward mu_min)?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / "fig_adaptive_signal_alfworld_check.png", dpi=140)
    plt.close(fig)
    print(f"Wrote fig_adaptive_signal_alfworld_check.png")

    with open(OUTSTATS, "w") as f:
        json.dump(shape_stats, f, indent=2, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else o)
    print(f"Wrote {OUTSTATS}")

    # Print summary table
    print("\n=== Shape summary (v24 WS knee at step", mu_knee, ") ===")
    print(f"{'signal':40s} {'hint':8s} {'knee':>6s} {'MAE_vs_target':>14s} {'ALF_mean':>10s} {'WS_mean_step25':>14s}")
    for e in shape_stats["candidates"]:
        key = e["key"]
        hint = e["mapping"]
        knee = e.get("ws_v24_signal_knee")
        mae = e.get("ws_v24_mae_vs_target")
        alf_mean_mu = e.get("alf_v24_mu_mean_cross")
        ws_step25_raw = e.get("ws_v24_raw_step25")
        print(f"{key:40s} {hint:8s} {str(knee):>6s} {str(round(mae,4) if mae is not None else '?'):>14s} "
              f"{str(round(alf_mean_mu,3) if alf_mean_mu is not None else '?'):>10s} "
              f"{str(round(ws_step25_raw,4) if ws_step25_raw is not None else '?'):>14s}")


if __name__ == "__main__":
    main()
