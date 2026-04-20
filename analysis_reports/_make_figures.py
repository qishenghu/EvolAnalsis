"""Generate 5 figures for the WebShop 1.5B DUET retrospective report.

Outputs: /data/home/qisheng/EvolAnalsis/analysis_reports/figures/fig{1..5}*.png
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("/data/home/qisheng/EvolAnalsis/analysis_reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA = json.load(open("/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/metrics.json"))

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})

# ------------ Utility -----------------------------------------------
def series(variant: str, key: str):
    """Return (steps, values) arrays for variant's metric, filtered to non-NaN."""
    rows = DATA.get(variant, [])
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        xs.append(r["step"])
        ys.append(v)
    return np.array(xs), np.array(ys)


def smooth(y, w=5):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def quarter_mean(rows, key, q):
    """Mean of `key` over quarter q in [1..4] (steps 1-25, 26-50, 51-75, 76-100)."""
    lo = (q - 1) * 25 + 1
    hi = q * 25
    vals = [r.get(key) for r in rows if lo <= r["step"] <= hi and r.get(key) is not None]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ============== Figure 1: Variant Landscape ========================
def fig1_landscape():
    # Canonical Val@100 results (from EXPERIMENT_LOG / parsed logs).
    # Categories:
    #   baseline: gray
    #   no-BC DUET variants: blue
    #   BC-enabled DUET variants: red
    #   crash: black (negative values)
    rows = [
        # name, val@100, category
        ("OnPolicy",          0.152, "baseline"),
        ("SFT",               0.387, "baseline"),
        ("SFT+GRPO",          0.404, "baseline"),
        ("LUFFY",             0.573, "baseline"),
        ("CHORD",             0.603, "baseline"),
        # DUET no-BC variants (DR3 + SC combinations)
        ("v1 (DUET native)",  0.549, "no-BC"),
        ("v2",                0.521, "no-BC"),
        ("v3",                0.617, "no-BC"),
        ("v4 (SC off)",       0.343, "no-BC"),
        ("v5 (baseline_sep off)", 0.000, "no-BC"),
        ("v6",                0.305, "no-BC"),
        ("v7 (DR3 off)",      0.473, "no-BC"),
        ("v8 (step off)",     0.574, "no-BC"),
        ("v10 (temp=1.5)",    0.571, "no-BC"),
        ("v11 (clip=2)",      0.388, "no-BC"),
        ("v12 (DR3 stab)",    0.431, "no-BC"),
        ("v13",               0.477, "no-BC"),
        ("v14",               0.528, "no-BC"),
        ("v15",               0.556, "no-BC"),
        ("v16",               0.542, "no-BC"),
        ("v17",               0.508, "no-BC"),
        ("v18",               0.501, "no-BC"),
        ("v19",               0.469, "no-BC"),
        ("v20",               0.477, "no-BC"),
        ("v21 (decouple off)", 0.095, "no-BC"),
        ("v28 (ema)",         0.495, "no-BC"),
        ("v29 (rescue)",      0.511, "no-BC"),
        ("v30 (KL=0.01)",     0.520, "no-BC"),
        ("v33 (temp=3.0)",    0.520, "no-BC"),
        # DUET BC-enabled (add CHORD-style SFT)
        ("v22 (const \u03bc=0.05)",  0.462, "BC"),
        ("v23 (const \u03bc=0.1 stab)", 0.440, "BC"),
        ("v24 (decay \u03bc=0.3\u21920.05)", 0.678, "BC"),
        # Widened-clip crashes
        ("v25 (wide clip)",  -0.041, "crash"),
    ]
    # Sort by val ascending
    rows_sorted = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows_sorted]
    vals = np.array([r[1] for r in rows_sorted])
    cats = [r[2] for r in rows_sorted]
    color_map = {"baseline": "#7f7f7f", "no-BC": "#1f77b4", "BC": "#d62728", "crash": "#000000"}
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(14, 6.2))
    xs = np.arange(len(labels))
    ax.scatter(xs, vals, c=colors, s=70, edgecolors="white", linewidth=0.8, zorder=5)

    # Horizontal reference lines
    ax.axhline(0.603, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.6,
               label="CHORD 0.603")
    ax.axhline(0.549, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8,
               label="DUET v1 native 0.549")
    ax.axhline(0.678, color="#d62728", linestyle="-.", linewidth=1.2, alpha=0.8,
               label="DUET v24 (BC) 0.678")
    # Shaded no-BC ceiling band
    ax.axhspan(0.52, 0.58, color="#1f77b4", alpha=0.08,
               label="\"no-BC ceiling\" band 0.52\u20130.58")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Val@100 (reward_mean_all, 200 tasks)")
    ax.set_title("WebShop Qwen2.5-1.5B: full variant landscape (sorted by Val@100)\n"
                 "Red = BC-enabled, Blue = no-BC DUET, Gray = baselines, Black = divergent")
    ax.set_ylim(-0.12, 0.78)
    ax.legend(loc="upper left", frameon=True, ncol=2)

    # Callout annotations
    for i, (name, v, c) in enumerate(rows_sorted):
        if name.startswith("v24"):
            ax.annotate(f"WINNER\n{v:.3f}", xy=(i, v), xytext=(i-3, v+0.07),
                        fontsize=9, color="#b40000", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#b40000"))
        if name.startswith("v25"):
            ax.annotate("crash", xy=(i, v), xytext=(i+1, v-0.05),
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="->", color="black"))
        if name.startswith("v5 "):
            ax.annotate("collapse", xy=(i, v), xytext=(i+1, v+0.07),
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    out = FIG_DIR / "fig1_variant_landscape.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ============== Figure 2: CHORD vs DUET v1 training dynamics =========
def fig2_chord_vs_v1():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    pairs = [
        ("CHORD", "chord", "#d62728"),
        ("DUET v1 native", "duet_v1", "#1f77b4"),
    ]
    panels = [
        ("critic/rewards_onpolicy/mean", "On-policy reward mean (training)", axes[0, 0]),
        ("actor/kl_loss", "Actor KL loss (policy vs ref)", axes[0, 1]),
        ("response_length/mean", "Response length mean (tokens)", axes[1, 0]),
        ("actor/entropy_loss", "Actor entropy loss", axes[1, 1]),
    ]
    for label, variant, color in pairs:
        for key, title, ax in panels:
            xs, ys = series(variant, key)
            if len(xs) == 0:
                ax.set_title(f"{title}\n(no data for {label})")
                continue
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9)
            ax.plot(xs, smooth(ys, 5), color=color, linewidth=1.8, label=label)
            ax.set_title(title)
            ax.set_xlabel("training step")

    # Mark CHORD's mu decay window (0-25)
    for _, _, ax in panels:
        ax.axvspan(0, 25, color="orange", alpha=0.07)
    axes[0, 0].text(12.5, axes[0, 0].get_ylim()[0] + 0.01, "CHORD \u03bc 0.9\u21920.05",
                    fontsize=8, color="darkorange", ha="center")

    axes[0, 0].legend(loc="lower right")
    axes[0, 1].legend(loc="upper left")
    fig.suptitle("Figure 2. CHORD (0.603) vs native DUET v1 (0.549): where do they diverge?",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIG_DIR / "fig2_chord_vs_duet_v1_dynamics.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ============== Figure 3: The no-BC ceiling ==========================
def fig3_no_bc_ceiling():
    # Narrative-rescue variants (stability-only changes on top of v12)
    rescue = [
        ("v12 base",   0.431, "#1f77b4"),
        ("v28 (ema=0.1)",  0.495, "#1f77b4"),
        ("v29 (rescue)",   0.511, "#1f77b4"),
        ("v30 (KL=0.01)",  0.520, "#1f77b4"),
        ("v33 (temp=3.0)", 0.520, "#1f77b4"),
    ]
    ceiling = np.mean([v for _, v, _ in rescue[1:]])
    # Breakthroughs
    breakthrough = [
        ("CHORD",              0.603, "#7f7f7f"),
        ("v24 (BC \u03bc=0.3\u21920.05)", 0.678, "#d62728"),
    ]
    bars = rescue + breakthrough
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(labels))
    bars_artist = ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(ceiling, color="#1f77b4", linestyle="--", linewidth=1.1,
               label=f"no-BC stability ceiling = {ceiling:.3f}")
    ax.axhline(0.603, color="#7f7f7f", linestyle=":", linewidth=1.1,
               label="CHORD (BC only) = 0.603")
    ax.axhline(0.678, color="#d62728", linestyle="-.", linewidth=1.1,
               label="v24 (DR3+SC+decaying BC) = 0.678")
    # Annotate "no-BC" group band
    ax.axvspan(-0.5, 4.5, color="#1f77b4", alpha=0.05)
    ax.text(2, 0.72, "DR3+SC + stability-only rescues\ncluster at 0.50 regardless of mechanism",
            ha="center", fontsize=9, color="#1f4e79")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Val@100")
    ax.set_ylim(0, 0.78)
    ax.set_title("Figure 3. The no-BC ceiling: four stability rescues cannot replace a BC term")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "fig3_no_bc_ceiling.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ============== Figure 4: v12 vs v24 mechanism ========================
def fig4_v12_vs_v24():
    panels = [
        ("actor/grad_norm",                     "actor/grad_norm"),
        ("actor/kl_loss",                       "actor/kl_loss"),
        ("actor/entropy_loss",                  "actor/entropy_loss"),
        ("dr3/disc_acc",                        "dr3/disc_acc"),
        ("duet/teacher_gradient_share",         "duet/teacher_gradient_share"),
        ("state_channel/progress_onpolicy_mean","state_channel/progress_onpolicy_mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    for idx, (title, key) in enumerate(panels):
        ax = axes.flat[idx]
        for label, variant, color in [
            ("v12 (no-BC)", "duet_v12", "#1f77b4"),
            ("v24 (+BC \u03bc decay)", "duet_v24", "#d62728"),
        ]:
            xs, ys = series(variant, key)
            if len(xs) == 0:
                ax.set_title(f"{title}\n(no data)")
                continue
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=0.9)
            ax.plot(xs, smooth(ys, 5), color=color, linewidth=1.7, label=label)
        ax.axvspan(0, 25, color="orange", alpha=0.07)
        ax.set_title(title)
        ax.set_xlabel("step")
    # Put legend in a good spot
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Figure 4. v12 (no-BC, 0.431) vs v24 (BC-decayed, 0.678): same DR3 stabilization, one mechanism difference",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIG_DIR / "fig4_v12_vs_v24_mechanism.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ============== Figure 5: Scaling prediction ==========================
def fig5_scaling_prediction():
    # Empirical gaps (DUET - strongest baseline) by scale and env.
    # ALFWorld: 1.5B (DUET 32.5 vs CHORD 27.0 = +5.5pp)
    #          3B (DUET 69.5 vs LUFFY 61.5 = +8pp)
    #          7B (DUET 86.5 vs OnPolicy 85.0 = +1.5pp)
    # WebShop: 1.5B v1 DUET vs CHORD = 0.549-0.603 = -5.4pp
    #          1.5B v24 vs CHORD    = 0.678-0.603 = +7.5pp
    #          3B DUET vs CHORD     = 0.763-0.728 = +3.5pp
    scales = np.array([1.5, 3.0, 7.0])

    alfworld_duet_v1 = np.array([5.5, 8.0, 1.8])  # DUET - best baseline (pp)
    webshop_duet_v1  = np.array([-5.4, 3.5, -np.nan])  # v1 vs CHORD (1.5B/3B)
    webshop_duet_v24_known = np.array([7.5, np.nan, np.nan])  # only 1.5B known

    # Theory: BC contribution in pp = (v24 - v1) * 100 on WebShop 1.5B
    bc_contribution_1p5b = 0.678 - 0.549  # 0.129 = 12.9pp
    # BC contribution decays with scale (broader action prior -> cold-start milder).
    # Model: proportional to 1/size, anchored at 1.5B.
    def bc_curve_pp(size):
        # Returns contribution in pp
        return bc_contribution_1p5b * 100.0 * (1.5 / size)

    bc_pp_3b = bc_curve_pp(3.0)   # ~6.45pp
    bc_pp_7b = bc_curve_pp(7.0)   # ~2.76pp

    # Predicted v24 WebShop 3B = v1 WebShop 3B gap + BC contribution-at-3B
    webshop_duet_v24_pred = np.array([7.5, 3.5 + bc_pp_3b, float("nan")])

    # Predicted v24 ALFWorld = v1 ALFWorld gap + BC contribution (in pp) at that scale.
    bc_pp_1p5b = bc_contribution_1p5b * 100.0  # 12.9pp
    alfworld_duet_v24_pred = np.array([
        5.5 + bc_pp_1p5b,
        8.0 + bc_pp_3b,
        1.8 + bc_pp_7b,
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: empirical v1 DUET-vs-baseline across scales + v24 prediction
    ax = axes[0]
    ax.plot(scales, alfworld_duet_v1, "o-", color="#1f77b4", linewidth=2,
            label="DUET v1 (empirical)")
    ax.plot(scales, alfworld_duet_v24_pred, "X--", color="#d62728", linewidth=2,
            label="v24 (projected)")
    for x, y in zip(scales, alfworld_duet_v1):
        ax.annotate(f"{y:+.1f}", xy=(x, y), xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color="#1f77b4")
    for x, y in zip(scales, alfworld_duet_v24_pred):
        ax.annotate(f"{y:+.1f}", xy=(x, y), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color="#d62728", fontweight="bold")
    # BC contribution arrows
    for x, v1, v24 in zip(scales, alfworld_duet_v1, alfworld_duet_v24_pred):
        ax.annotate("", xy=(x, v24), xytext=(x, v1),
                    arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.5))
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(scales)
    ax.set_xticklabels(["1.5B", "3B", "7B"])
    ax.set_xlabel("Model scale")
    ax.set_ylabel("Gap (DUET - best baseline), pp")
    ax.set_title("ALFWorld: v1 empirical vs v24 projected\nBC contribution (red arrows) shrinks with scale")
    ax.set_ylim(0, 25)
    ax.legend(fontsize=9, loc="upper right")

    # Right: WebShop scaling
    ax = axes[1]
    webshop_scales = np.array([1.5, 3.0, 7.0])
    wv1 = np.array([-5.4, 3.5, float("nan")])  # v1 gap vs CHORD
    wv24 = np.array([7.5, 3.5 + bc_pp_3b, float("nan")])  # v24 gap

    mask = ~np.isnan(wv1)
    ax.plot(webshop_scales[mask], wv1[mask], "o-", color="#1f77b4", label="DUET v1 (empirical)")
    ax.plot(webshop_scales[0:1], wv24[0:1], "X", color="#d62728", markersize=14,
            label="v24 WebShop 1.5B (empirical)")
    # draw connection line in v24 predicted (1.5B -> 3B)
    ax.plot(webshop_scales[:2], wv24[:2], "x--", color="#d62728", linewidth=1.2, alpha=0.8,
            label="v24 3B (predicted)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    # Cleaner annotations: spread them apart
    ax.annotate("v1 -5.4pp", xy=(1.5, -5.4), xytext=(10, -12), textcoords="offset points",
                fontsize=9, color="#1f77b4")
    ax.annotate("v1 +3.5pp", xy=(3.0, 3.5), xytext=(10, -14), textcoords="offset points",
                fontsize=9, color="#1f77b4")
    ax.annotate("v24 +7.5pp", xy=(1.5, 7.5), xytext=(10, 8), textcoords="offset points",
                fontsize=9, color="#d62728", fontweight="bold")
    ax.annotate(f"v24 +{wv24[1]:.1f}pp (predicted)", xy=(3.0, wv24[1]),
                xytext=(10, 10), textcoords="offset points", fontsize=9, color="#d62728")
    ax.set_xticks([1.5, 3.0, 7.0])
    ax.set_xticklabels(["1.5B", "3B", "7B"])
    ax.set_xlabel("Model scale")
    ax.set_ylabel("Gap vs CHORD, pp (Val@100)")
    ax.set_title("WebShop: v24 projection\n1.5B +7.5pp (known), 3B \u2248 +{:.1f}pp (predicted)".format(wv24[1]))
    ax.set_ylim(-8, 15)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Figure 5. Scaling prediction: BC sub-term contribution shrinks with model scale",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig5_scaling_prediction.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig1_landscape()
    fig2_chord_vs_v1()
    fig3_no_bc_ceiling()
    fig4_v12_vs_v24()
    fig5_scaling_prediction()
    print("\nAll 5 figures written.")
