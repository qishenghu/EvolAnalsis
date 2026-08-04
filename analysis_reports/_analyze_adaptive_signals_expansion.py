"""Empirically validate candidate adaptive-mu signals.

Key outputs:
  analysis_reports/fig_adaptive_signal_expansion.png
  analysis_reports/_parsed/adaptive_signal_expansion.json
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import pearsonr, spearmanr

OUT_DIR = Path("/data/home/qisheng/EvolAnalsis/analysis_reports")
PARSED = OUT_DIR / "_parsed" / "adaptive_signals.json"

# v24 hand-tuned schedule (piecewise linear in log-space given the logged chord/mu trajectory)
V24_SCHEDULE = [(1, 0.30), (10, 0.20), (25, 0.07), (50, 0.05), (100, 0.05)]

def interp_schedule(step, schedule):
    """Piecewise linear μ schedule lookup."""
    steps = [s for s, _ in schedule]
    vals = [v for _, v in schedule]
    return float(np.interp(step, steps, vals))


def extract(variant_rows, key):
    """Return (steps, vals) arrays for the given key. Missing values are skipped."""
    steps, vals = [], []
    for r in variant_rows:
        if key in r and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key])):
            steps.append(r["step"])
            vals.append(r[key])
    return np.array(steps), np.array(vals)


def align(s1, v1, s2, v2):
    """Return intersection of s1 and s2 with matching vals."""
    set2 = dict(zip(s2.tolist(), v2.tolist()))
    out_s, out_a, out_b = [], [], []
    for s, a in zip(s1, v1):
        if s in set2:
            out_s.append(s)
            out_a.append(a)
            out_b.append(set2[s])
    return np.array(out_s), np.array(out_a), np.array(out_b)


def smooth(x, w=5):
    if len(x) < w:
        return x
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, np.ones(w) / w, mode="valid")[: len(x)]


def fit_linear(x, y):
    """Fit μ = a + b * x, return (a, b, rho_linear, MAE)."""
    if len(x) < 2:
        return 0, 0, 0, np.inf
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    mae = float(np.mean(np.abs(pred - y)))
    r = float(pearsonr(x, y)[0]) if np.std(x) > 0 else 0
    return a, b, r, mae


def fit_affine_clip(x, y):
    """Fit μ = clip(a + b*x, μ_floor, μ_ceil) with scipy."""
    if len(x) < 3:
        return fit_linear(x, y)
    def loss(p):
        a, b, lo, hi = p
        pred = np.clip(a + b * x, lo, hi)
        return np.mean((pred - y) ** 2)
    x0 = [float(y.mean()), 0.0, 0.05, 0.3]
    try:
        res = optimize.minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5})
        a, b, lo, hi = res.x
        if lo > hi:
            lo, hi = hi, lo
        pred = np.clip(a + b * x, lo, hi)
        mae = float(np.mean(np.abs(pred - y)))
        return a, b, mae, (lo, hi)
    except Exception:
        return fit_linear(x, y) + ((0.05, 0.3),)


def knee_step(mu_pred, threshold=0.1):
    """First step where predicted μ drops below 0.1."""
    for i, m in enumerate(mu_pred):
        if m < threshold:
            return i + 1  # 1-indexed step
    return len(mu_pred)  # never crossed


def main():
    d = json.load(open(PARSED))
    ws24 = d["ws_v24"]
    alf24 = d["alf_v24"]
    ws39 = d["ws_v39"]
    alf39 = d["alf_v39"]
    alf1 = d["alf_v1"]

    # Reference μ from v24 hand-tuned
    steps_ws = np.arange(1, 101)
    mu_ref = np.array([interp_schedule(s, V24_SCHEDULE) for s in steps_ws])

    # Signals to test: (label, key, env_smooth, expected_direction)
    signals = [
        ("A: TGS", "duet/teacher_gradient_share", "decreasing"),
        ("B: dr3/ess_off_window", "dr3/ess_off_window", "decreasing"),
        ("C: dr3/w_std", "dr3/w_std", "decreasing"),
        ("D: actor/kl_loss", "actor/kl_loss", "decreasing"),
        ("E: chord/sft_loss (tchr NLL)", "chord/sft_loss", "decreasing"),
        ("F: chord/log_prob_std", "chord/log_prob_std", "decreasing"),
        ("G: adv_teacher_abs_mean", "duet/adv_teacher_effective_abs_mean", "decreasing"),
        ("H: 1/group_reward_variance", "duet/group_reward_variance_mean", "inverse_then_decreasing"),
        ("J: w_off/w ratio", "ratio_w_off_over_w", "decreasing"),
        ("K: disc_acc x ess_ratio", "disc_acc_x_ess", "decreasing"),
    ]

    # Build derived signals
    def build_derived(variant_rows):
        """Return derived signals per-step. Skip NaN/None to keep arrays clean."""
        rows_by_step = {r["step"]: r for r in variant_rows}
        out = {}
        steps_all = sorted(rows_by_step)
        ratio_w, disc_x_ess = [], []
        for s in steps_all:
            r = rows_by_step[s]
            w_off = r.get("dr3/w_off_mean")
            w_all = r.get("dr3/w_mean")
            if (w_off is not None and w_all is not None and w_all > 1e-6
                    and not math.isnan(w_off) and not math.isnan(w_all)):
                ratio_w.append((s, w_off / w_all))
            disc = r.get("dr3/disc_acc")
            ess = r.get("dr3/ess_off_window")
            ess_len = r.get("dr3/ess_window_len", 8.0)  # default window = 8
            if (disc is not None and ess is not None and ess_len is not None
                    and ess_len > 0 and not math.isnan(disc) and not math.isnan(ess)):
                ess_ratio = ess / ess_len
                disc_x_ess.append((s, disc * ess_ratio))
        out["ratio_w_off_over_w"] = ratio_w
        out["disc_acc_x_ess"] = disc_x_ess
        return out

    derived_ws24 = build_derived(ws24)
    derived_alf24 = build_derived(alf24)

    # For each signal, collect WS v24 and ALF v24 trajectory + stats
    results = {}
    for label, key, direction in signals:
        if key.startswith("dr3/") or key.startswith("actor/") or key.startswith("chord/") or key.startswith("duet/"):
            ws_s, ws_v = extract(ws24, key)
            alf_s, alf_v = extract(alf24, key)
        elif key == "ratio_w_off_over_w":
            pairs_ws = derived_ws24.get("ratio_w_off_over_w", [])
            pairs_alf = derived_alf24.get("ratio_w_off_over_w", [])
            ws_s = np.array([s for s, _ in pairs_ws])
            ws_v = np.array([v for _, v in pairs_ws])
            alf_s = np.array([s for s, _ in pairs_alf])
            alf_v = np.array([v for _, v in pairs_alf])
        elif key == "disc_acc_x_ess":
            pairs_ws = derived_ws24.get("disc_acc_x_ess", [])
            pairs_alf = derived_alf24.get("disc_acc_x_ess", [])
            ws_s = np.array([s for s, _ in pairs_ws])
            ws_v = np.array([v for _, v in pairs_ws])
            alf_s = np.array([s for s, _ in pairs_alf])
            alf_v = np.array([v for _, v in pairs_alf])
        else:
            continue

        # Align WS signal with mu_ref
        ws_s_aln, ws_v_aln, mu_aln = align(ws_s, ws_v, steps_ws, mu_ref)
        if len(ws_s_aln) < 5:
            continue

        # Apply transform for group_reward_variance (inverse)
        if key == "duet/group_reward_variance_mean":
            ws_v_use = 1.0 / (ws_v_aln + 1e-6)
            alf_v_use = 1.0 / (alf_v + 1e-6) if len(alf_v) else alf_v
        else:
            ws_v_use = ws_v_aln
            alf_v_use = alf_v

        # Guard against zero-variance signals (e.g. phi_mean constant=1 on WebShop)
        if np.std(ws_v_use) < 1e-10 or not np.any(np.isfinite(ws_v_use)):
            print(f"[skip] {label}: zero variance or non-finite")
            continue

        # Fit linear map signal → μ
        r_lin = pearsonr(ws_v_use, mu_aln)[0] if np.std(ws_v_use) > 0 else 0
        rho_lin = spearmanr(ws_v_use, mu_aln).correlation if np.std(ws_v_use) > 0 else 0

        # Linear fit μ = a + b*signal
        try:
            # Manual least-squares (safer than polyfit for edge cases)
            x_clean = np.asarray(ws_v_use, dtype=np.float64)
            y_clean = np.asarray(mu_aln, dtype=np.float64)
            mask_fin = np.isfinite(x_clean) & np.isfinite(y_clean)
            x_clean = x_clean[mask_fin]
            y_clean = y_clean[mask_fin]
            if len(x_clean) < 2 or np.std(x_clean) < 1e-10:
                raise ValueError("insufficient variance")
            sx = x_clean.mean(); sy = y_clean.mean()
            num = ((x_clean - sx) * (y_clean - sy)).sum()
            den = ((x_clean - sx) ** 2).sum()
            b = num / den if den > 0 else 0.0
            a = sy - b * sx
        except Exception as e:
            print(f"[skip] {label}: fit failed ({e})")
            continue
        mu_pred_ws = np.clip(a + b * ws_v_use, 0.0, 0.5)
        mae = float(np.mean(np.abs(mu_pred_ws - mu_aln)))
        knee = knee_step(mu_pred_ws, threshold=0.1)
        # v24 hand-tuned μ crosses 0.1 between step 10-25; linear interp says step 17
        v24_knee = 17

        # ALFWorld self-adjust: does signal suggest LOWER μ on ALFWorld than WebShop?
        alf_mu_pred = None
        alf_mean = None
        ws_mean = None
        alf_active_ratio = None
        if len(alf_v_use) > 5:
            alf_mu_pred_raw = np.clip(a + b * alf_v_use, 0.0, 0.5)
            alf_mu_pred = alf_mu_pred_raw.mean()
            alf_mean = alf_v_use.mean()
            ws_mean = ws_v_use.mean()
            # "active" = signal magnitude relative to WS.  smaller=good for ALFWorld
            # For normalization, compare mean over steps 10-50
            mask_ws = (ws_s_aln >= 10) & (ws_s_aln <= 50)
            mask_alf = (alf_s >= 10) & (alf_s <= 50)
            if mask_ws.sum() > 0 and mask_alf.sum() > 0:
                alf_active_ratio = alf_v_use[mask_alf].mean() / max(ws_v_use[mask_ws].mean(), 1e-6)

        results[label] = {
            "key": key,
            "direction": direction,
            "pearson_ws": float(r_lin),
            "spearman_ws": float(rho_lin) if rho_lin is not None else 0.0,
            "mae_ws": mae,
            "fit": {"a": float(a), "b": float(b)},
            "knee_pred": int(knee),
            "knee_v24": v24_knee,
            "ws_steps": ws_s_aln.tolist(),
            "ws_signal": ws_v_use.tolist(),
            "ws_mu_pred": mu_pred_ws.tolist(),
            "alf_steps": alf_s.tolist(),
            "alf_signal": alf_v_use.tolist(),
            "alf_mu_pred": (np.clip(a + b * alf_v_use, 0.0, 0.5).tolist() if len(alf_v_use) else []),
            "ws_mean_mid": (float(ws_v_use[(ws_s_aln >= 10) & (ws_s_aln <= 50)].mean()) if (np.sum((ws_s_aln >= 10) & (ws_s_aln <= 50)) > 0) else None),
            "alf_mean_mid": (float(alf_v_use[(alf_s >= 10) & (alf_s <= 50)].mean()) if (len(alf_v_use) and np.sum((alf_s >= 10) & (alf_s <= 50)) > 0) else None),
            "alf_active_ratio": (float(alf_active_ratio) if alf_active_ratio is not None else None),
        }

    # Also compute composite μ_hat via v24's hand-tuned schedule vs inferred mu_pred knee
    # Save JSON
    out_json = OUT_DIR / "_parsed" / "adaptive_signal_expansion.json"
    with open(out_json, "w") as f:
        # strip numpy types
        def sanitize(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, dict):
                return {k: sanitize(v) for k, v in x.items()}
            if isinstance(x, list):
                return [sanitize(v) for v in x]
            return x
        json.dump(sanitize(results), f, indent=2)
    print(f"Wrote {out_json}")

    # ===== v39 diagnosis =====
    diag = {}
    # mu_adaptive_gated trajectory in v39
    ws39_steps, ws39_mu_adapt = extract(ws39, "chord/mu")
    ws39_adapt_steps, ws39_adapt_vals = extract(ws39, "chord/mu_adaptive_gated")
    ws39_disc_s, ws39_disc_v = extract(ws39, "dr3/disc_acc")
    ws39_kl_s, ws39_kl_v = extract(ws39, "actor/kl_loss")
    ws24_kl_s, ws24_kl_v = extract(ws24, "actor/kl_loss")

    # Step where v39 mu drops below 0.1
    v39_knee = knee_step(ws39_mu_adapt.tolist(), 0.1)
    diag["v39_ws_mu_knee"] = int(v39_knee)
    diag["v24_hand_knee"] = 17  # from schedule
    # mean v39 kl over step 10-50 vs v24
    mask39 = (ws39_kl_s >= 10) & (ws39_kl_s <= 50)
    mask24 = (ws24_kl_s >= 10) & (ws24_kl_s <= 50)
    diag["v39_ws_kl_mid"] = float(ws39_kl_v[mask39].mean()) if mask39.sum() else None
    diag["v24_ws_kl_mid"] = float(ws24_kl_v[mask24].mean()) if mask24.sum() else None
    # mean mu over step 10-50
    mask39_mu = (ws39_steps >= 10) & (ws39_steps <= 50)
    mask24_steps, mask24_mu_vals = extract(ws24, "chord/mu")
    mask24_mu_window = (mask24_steps >= 10) & (mask24_steps <= 50)
    diag["v39_ws_mu_mid"] = float(ws39_mu_adapt[mask39_mu].mean()) if mask39_mu.sum() else None
    diag["v24_ws_mu_mid"] = float(mask24_mu_vals[mask24_mu_window].mean()) if mask24_mu_window.sum() else None
    # area under μ curve = cumulative BC application
    diag["v39_ws_mu_auc"] = float(np.trapz(ws39_mu_adapt, ws39_steps))
    diag["v24_ws_mu_auc"] = float(np.trapz(mask24_mu_vals, mask24_steps))
    print("\n=== v39 DIAGNOSIS ===")
    print(json.dumps(diag, indent=2))

    # Save v39 diagnosis
    v39_file = OUT_DIR / "_parsed" / "v39_diagnosis.json"
    with open(v39_file, "w") as f:
        json.dump(diag, f, indent=2)

    # ===== FIGURE =====
    n_signals = len(results)
    ncols = 3
    nrows = (n_signals + ncols - 1) // ncols + 1  # +1 row for summary panels
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows))
    axes_flat = axes.flatten()

    for i, (label, res) in enumerate(results.items()):
        ax = axes_flat[i]
        ax2 = ax.twinx()
        ws_s = np.array(res["ws_steps"])
        ws_sig = np.array(res["ws_signal"])
        ws_mu_pred = np.array(res["ws_mu_pred"])
        alf_s = np.array(res["alf_steps"])
        alf_sig = np.array(res["alf_signal"])

        # Plot signals on left axis
        ax.plot(ws_s, ws_sig, color="tab:blue", lw=2.0, label="WS v24 signal", alpha=0.9)
        if len(alf_sig):
            ax.plot(alf_s, alf_sig, color="tab:orange", lw=2.0, label="ALF v24 signal", alpha=0.9)
        # Reference v24 μ on right axis
        ax2.plot(steps_ws, mu_ref, color="black", lw=1.3, ls="--", label="v24 μ (hand-tuned)", alpha=0.7)
        ax2.plot(ws_s, ws_mu_pred, color="tab:red", lw=1.3, ls=":", label="μ predicted", alpha=0.8)

        ax.set_title(f"{label}\nr={res['pearson_ws']:.3f}  MAE={res['mae_ws']:.3f}  knee={res['knee_pred']} (v24=17)",
                     fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel(label.split(':')[-1].strip(), fontsize=8, color="tab:blue")
        ax2.set_ylabel("μ", fontsize=8, color="tab:red")
        ax2.set_ylim(0, 0.35)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        # ALFWorld self-adjust annotation
        alf_ratio = res.get("alf_active_ratio")
        color_tag = "green" if (alf_ratio is not None and alf_ratio < 0.8) else "red" if alf_ratio and alf_ratio > 1.05 else "orange"
        if alf_ratio is not None:
            ax.text(0.02, 0.97, f"ALF/WS mid = {alf_ratio:.2f}", color=color_tag, fontsize=7,
                    transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Clear trailing axes up to summary row
    for j in range(n_signals, (nrows - 1) * ncols):
        axes_flat[j].axis("off")

    # Bottom row: summary panels (last ncols axes)
    # 1. Alignment bar chart
    base_idx = (nrows - 1) * ncols
    ax_sum1 = axes_flat[base_idx]
    ax_sum2 = axes_flat[base_idx + 1]
    ax_sum3 = axes_flat[base_idx + 2]

    labels = list(results.keys())
    abs_rs = [abs(results[l]["pearson_ws"]) for l in labels]
    maes = [results[l]["mae_ws"] for l in labels]
    alf_ratios = [results[l].get("alf_active_ratio") or 1.0 for l in labels]

    # Shorten labels
    short_labels = [l.split(":")[0] for l in labels]
    y_pos = np.arange(len(short_labels))

    ax_sum1.barh(y_pos, abs_rs, color=["tab:green" if r > 0.85 else "tab:orange" if r > 0.6 else "tab:red" for r in abs_rs])
    ax_sum1.axvline(0.85, color="green", lw=0.5, ls="--", alpha=0.5)
    ax_sum1.set_yticks(y_pos)
    ax_sum1.set_yticklabels(short_labels, fontsize=8)
    ax_sum1.invert_yaxis()
    ax_sum1.set_xlabel("|Pearson r| (signal vs v24 μ)")
    ax_sum1.set_title("Alignment with v24 μ (high=good)", fontsize=9)
    ax_sum1.grid(True, alpha=0.3, axis="x")

    ax_sum2.barh(y_pos, maes, color="tab:blue")
    ax_sum2.set_yticks(y_pos)
    ax_sum2.set_yticklabels(short_labels, fontsize=8)
    ax_sum2.invert_yaxis()
    ax_sum2.set_xlabel("MAE (predicted μ vs v24)")
    ax_sum2.set_title("Prediction error", fontsize=9)
    ax_sum2.grid(True, alpha=0.3, axis="x")

    ax_sum3.barh(y_pos, alf_ratios, color=["tab:green" if r < 0.8 else "tab:red" if r > 1.05 else "tab:orange" for r in alf_ratios])
    ax_sum3.axvline(1.0, color="k", lw=0.5)
    ax_sum3.set_yticks(y_pos)
    ax_sum3.set_yticklabels(short_labels, fontsize=8)
    ax_sum3.invert_yaxis()
    ax_sum3.set_xlabel("ALF/WS signal ratio (steps 10-50, <1 = good cross-env)")
    ax_sum3.set_title("Cross-env self-adjustment", fontsize=9)
    ax_sum3.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Adaptive-μ candidate signals: WebShop v24 + ALFWorld v24 empirical validation",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = OUT_DIR / "fig_adaptive_signal_expansion.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # ===== TOP-4 MU RECONSTRUCTION FIGURE =====
    # Show predicted μ vs v24 hand-tuned for the 4 most promising signals
    top_labels = ["B: dr3/ess_off_window", "E: chord/sft_loss (tchr NLL)",
                  "F: chord/log_prob_std", "J: w_off/w ratio"]
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 4))
    for ax, lbl in zip(axes3, top_labels):
        res = results[lbl]
        ws_s = np.array(res["ws_steps"])
        ws_pred = np.array(res["ws_mu_pred"])
        alf_pred = np.array(res["alf_mu_pred"])
        alf_s = np.array(res["alf_steps"])
        ax.plot(steps_ws, mu_ref, color="black", lw=2.0, label="v24 hand-tuned μ (ref)")
        ax.plot(ws_s, ws_pred, color="tab:blue", lw=1.8, label="WS: predicted μ", alpha=0.85)
        if len(alf_pred):
            ax.plot(alf_s, alf_pred, color="tab:orange", lw=1.8, label="ALF: predicted μ", alpha=0.85)
        ax.axhline(0.1, color="gray", ls="--", alpha=0.4)
        ax.axvline(17, color="black", ls=":", alpha=0.4, label="v24 knee=17")
        ax.set_xlabel("step"); ax.set_ylabel("μ"); ax.set_ylim(-0.02, 0.4)
        ax.set_title(f"{lbl}\nr={res['pearson_ws']:+.3f}  MAE={res['mae_ws']:.3f}  "
                     f"ALF/WS={res.get('alf_active_ratio'):.2f}" if res.get('alf_active_ratio') else lbl,
                     fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle("Top-4 candidate signals: μ reconstruction on WebShop + ALFWorld", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png4 = OUT_DIR / "fig_adaptive_top4_mu_reconstruction.png"
    plt.savefig(out_png4, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png4}")

    # ===== v39 diagnosis figure: μ trajectory + KL + disc_acc =====
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes2[0]
    ws24_mu_s, ws24_mu_v = extract(ws24, "chord/mu")
    ws39_mu_s, ws39_mu_v = extract(ws39, "chord/mu")
    ax.plot(ws24_mu_s, ws24_mu_v, label="v24 (hand-tuned)", color="black", lw=2.0)
    ax.plot(ws39_mu_s, ws39_mu_v, label="v39 (adaptive)", color="tab:red", lw=2.0)
    ax.axhline(0.1, color="gray", ls="--", alpha=0.5, label="knee threshold")
    ax.axvline(17, color="black", ls=":", alpha=0.3, label="v24 knee (step 17)")
    ax.axvline(v39_knee, color="tab:red", ls=":", alpha=0.5, label=f"v39 knee (step {v39_knee})")
    ax.set_xlabel("step"); ax.set_ylabel("μ")
    ax.set_title("μ trajectory on WebShop")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ws24_disc_s, ws24_disc_v = extract(ws24, "dr3/disc_acc")
    ws39_disc_s, ws39_disc_v = extract(ws39, "dr3/disc_acc")
    ax.plot(ws24_disc_s, ws24_disc_v, label="v24 disc_acc", color="black", lw=2.0)
    ax.plot(ws39_disc_s, ws39_disc_v, label="v39 disc_acc", color="tab:red", lw=2.0)
    ax.set_xlabel("step"); ax.set_ylabel("disc_acc")
    ax.set_title("Discriminator accuracy (driver of v39 adaptive μ)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax.plot(ws24_kl_s, ws24_kl_v, label="v24 KL", color="black", lw=2.0)
    ax.plot(ws39_kl_s, ws39_kl_v, label="v39 KL", color="tab:red", lw=2.0)
    ax.set_xlabel("step"); ax.set_ylabel("actor/kl_loss")
    ax.set_title("Policy–ref KL drift (live effect of μ mismatch)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_yscale("symlog", linthresh=0.01)

    plt.suptitle("v39 diagnosis: why 10.5pp WebShop live gap vs v24", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png2 = OUT_DIR / "fig_v39_diagnosis.png"
    plt.savefig(out_png2, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png2}")

    # ===== Print ranking summary =====
    print("\n=== RANKING (by alignment + cross-env self-adjust) ===")
    for lbl, res in sorted(results.items(), key=lambda kv: (kv[1]["mae_ws"], -abs(kv[1]["pearson_ws"]))):
        a_r = res.get("alf_active_ratio")
        a_str = f"ALF/WS={a_r:.2f}" if a_r is not None else "ALF/WS=?"
        direction = "dec" if res["fit"]["b"] < 0 else "inc"
        print(f"  {lbl:35s}  r={res['pearson_ws']:+.3f}  MAE={res['mae_ws']:.3f}  "
              f"knee={res['knee_pred']:3d}  {a_str}  fit.b sign={direction}")


if __name__ == "__main__":
    main()
