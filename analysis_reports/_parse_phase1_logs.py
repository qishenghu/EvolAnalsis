"""Parse Phase 1 webshop 1.5B DUET logs (v24 / v39 / v39b / v40b / v41b / v43a).

Extracts per-step metrics from each log and saves them to JSON. Also produces
summary statistics and the 5 required figures.
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/home/qisheng/EvolAnalsis")
LOGDIR = ROOT / "logs"
PARSED = ROOT / "analysis_reports/_parsed"
FIGDIR = ROOT / "analysis_reports/figures"
PARSED.mkdir(exist_ok=True, parents=True)
FIGDIR.mkdir(exist_ok=True, parents=True)

N_STEPS = 100

VARIANTS = {
    "v24": "webshop_qwen1.5b_duet_v24.log",
    "v39": "webshop_qwen1.5b_duet_v39.log",
    "v39b": "webshop_qwen1.5b_duet_v39b.log",
    "v40b": "webshop_qwen1.5b_duet_v40b.log",
    "v41b": "webshop_qwen1.5b_duet_v41b.log",
    "v43a": "webshop_qwen1.5b_duet_v43a.log",
}

# Keys we want to track (must match log keys exactly)
CORE_KEYS = [
    "chord/mu",
    "chord/mu_adaptive_gated",
    "chord/mu_mode",
    "chord/sft_loss",
    "chord/weighted_sft_loss",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/pg_loss",
    "critic/success_onpolicy/mean",
    "critic/rewards_onpolicy/mean",
    "dr3/disc_acc",
    "dr3/ess_off_window",
    "dr3/alpha",
    "duet/teacher_gradient_share",
    "diag/teacher_sample_ratio",
    # Mode-specific signals — may not exist in every log
    "chord/disc_acc_ema",
    "chord/nll_ema",
    "chord/nll_current",
    "chord/kl_cost_ema",
    "chord/kl_budget",
    "chord/kl_budget_ema",
    "chord/kl_step_mult",
    "chord/mu_lagrange_state",
    "chord/ess_anchor",
    "chord/ess_ratio",
    "chord/ess_ema",
]

STEP_PATTERN = re.compile(r"step:(\d+)\s*-\s*(.+?)(?=\x1b\[36m|$)")
KV_PATTERN = re.compile(r"([A-Za-z0-9_/]+):(-?\d+\.?\d*(?:[eE][+-]?\d+)?)")


def parse_log(path: Path, variant: str):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = re.search(r"step:(\d+)\s*-\s*(.*)", line)
            if not m:
                continue
            step = int(m.group(1))
            if step > N_STEPS:
                continue
            rest = m.group(2)
            # Extract key:value pairs. Simple split on " - " handles log format.
            row = {"step": step}
            parts = rest.split(" - ")
            for p in parts:
                if ":" not in p:
                    continue
                kv = p.rsplit(":", 1)
                if len(kv) != 2:
                    continue
                k, v = kv[0].strip(), kv[1].strip()
                # Only keep numeric
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            rows.append(row)
    # Take unique step (last observation wins in case of duplicates)
    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    rows = [by_step[s] for s in sorted(by_step.keys())]
    return rows


def series_of(rows, key, n=N_STEPS):
    lut = {r["step"]: r.get(key, np.nan) for r in rows}
    return np.array([lut.get(t, np.nan) for t in range(1, n + 1)], dtype=float)


def smooth(x, w=5):
    out = np.full_like(x, np.nan, dtype=float)
    half = w // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.nanmean(vals)
    return out


def v24_hand_schedule(n=N_STEPS):
    """Reconstruct v24's hand-tuned schedule from config:
    peak=0.30, valley=0.05, decay_steps=25, warmup_steps=0.
    Linear decay from step 1->25."""
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


def summarize_variant(variant, rows):
    mu = series_of(rows, "chord/mu")
    success = series_of(rows, "critic/success_onpolicy/mean")
    rewards = series_of(rows, "critic/rewards_onpolicy/mean")
    grad_norm = series_of(rows, "actor/grad_norm")
    kl_loss = series_of(rows, "actor/kl_loss")
    entropy_loss = series_of(rows, "actor/entropy_loss")
    sft_loss = series_of(rows, "chord/sft_loss")
    disc_acc = series_of(rows, "dr3/disc_acc")
    teacher_share = series_of(rows, "duet/teacher_gradient_share")
    mu_mode = series_of(rows, "chord/mu_mode")
    mu_gated = series_of(rows, "chord/mu_adaptive_gated")

    # Mode-specific
    disc_acc_ema = series_of(rows, "chord/disc_acc_ema")
    nll_ema = series_of(rows, "chord/nll_ema")
    nll_current = series_of(rows, "chord/nll_current")
    kl_cost_ema = series_of(rows, "chord/kl_cost_ema")
    kl_budget = series_of(rows, "chord/kl_budget")
    kl_budget_ema = series_of(rows, "chord/kl_budget_ema")
    kl_step_mult = series_of(rows, "chord/kl_step_mult")
    mu_lagrange = series_of(rows, "chord/mu_lagrange_state")
    ess_anchor = series_of(rows, "chord/ess_anchor")
    ess_ratio = series_of(rows, "chord/ess_ratio")
    ess_ema = series_of(rows, "chord/ess_ema")
    ess_off = series_of(rows, "dr3/ess_off_window")

    def _safe_mean(x, a, b):
        seg = x[a:b]
        seg = seg[np.isfinite(seg)]
        return float(np.nan if seg.size == 0 else np.mean(seg))

    def _safe_final(x):
        finite_idx = np.where(np.isfinite(x))[0]
        if finite_idx.size == 0:
            return float("nan")
        return float(x[finite_idx[-1]])

    summary = {
        "variant": variant,
        "n_rows": len(rows),
        "mu_mean_all": float(np.nanmean(mu)),
        "mu_mean_s1_25": _safe_mean(mu, 0, 25),
        "mu_mean_s26_50": _safe_mean(mu, 25, 50),
        "mu_mean_s51_100": _safe_mean(mu, 50, 100),
        "mu_s1": float(mu[0]) if np.isfinite(mu[0]) else float("nan"),
        "mu_s5": float(mu[4]) if len(mu) > 4 and np.isfinite(mu[4]) else float("nan"),
        "mu_s10": float(mu[9]) if len(mu) > 9 and np.isfinite(mu[9]) else float("nan"),
        "mu_s25": float(mu[24]) if len(mu) > 24 and np.isfinite(mu[24]) else float("nan"),
        "mu_s50": float(mu[49]) if len(mu) > 49 and np.isfinite(mu[49]) else float("nan"),
        "mu_s100": _safe_final(mu),
        "mu_std_all": float(np.nanstd(mu)),
        "mu_min": float(np.nanmin(mu)),
        "mu_max": float(np.nanmax(mu)),
        "success_at_100": _safe_final(success),
        "success_mean_s51_100": _safe_mean(success, 50, 100),
        "reward_at_100": _safe_final(rewards),
        "reward_mean_s51_100": _safe_mean(rewards, 50, 100),
        "grad_norm_mean_s1_25": _safe_mean(grad_norm, 0, 25),
        "grad_norm_mean_s26_100": _safe_mean(grad_norm, 25, 100),
        "kl_loss_mean_s1_25": _safe_mean(kl_loss, 0, 25),
        "kl_loss_mean_s26_100": _safe_mean(kl_loss, 25, 100),
        "entropy_loss_mean_s1_25": _safe_mean(entropy_loss, 0, 25),
        "entropy_loss_mean_s26_100": _safe_mean(entropy_loss, 25, 100),
        "sft_loss_mean_all": float(np.nanmean(sft_loss)),
        "disc_acc_mean_s26_50": _safe_mean(disc_acc, 25, 50),
        "teacher_share_mean_s51_100": _safe_mean(teacher_share, 50, 100),
        "mu_mode_first": float(mu_mode[0]) if np.isfinite(mu_mode[0]) else float("nan"),
        "mu_gated_mean": float(np.nanmean(mu_gated)),
    }
    # Mode-specific final values
    summary.update({
        "disc_acc_ema_s25": float(disc_acc_ema[24]) if len(disc_acc_ema) > 24 and np.isfinite(disc_acc_ema[24]) else float("nan"),
        "nll_ema_final": _safe_final(nll_ema),
        "nll_current_final": _safe_final(nll_current),
        "kl_cost_ema_final": _safe_final(kl_cost_ema),
        "kl_budget_final": _safe_final(kl_budget_ema) if np.isfinite(_safe_final(kl_budget_ema)) else _safe_final(kl_budget),
        "kl_step_mult_mean": float(np.nanmean(kl_step_mult)),
        "kl_step_mult_std": float(np.nanstd(kl_step_mult)),
        "mu_lagrange_final": _safe_final(mu_lagrange),
        "ess_anchor_final": _safe_final(ess_anchor),
        "ess_ratio_final": _safe_final(ess_ratio),
        "ess_ema_final": _safe_final(ess_ema),
        "ess_off_final": _safe_final(ess_off),
    })
    return summary


def main():
    all_data = {}
    all_summary = {}
    for variant, logname in VARIANTS.items():
        path = LOGDIR / logname
        if not path.exists():
            print(f"WARNING: missing log {path}")
            continue
        rows = parse_log(path, variant)
        print(f"{variant}: parsed {len(rows)} rows, steps 1..{max(r['step'] for r in rows) if rows else 0}")
        all_data[variant] = rows
        all_summary[variant] = summarize_variant(variant, rows)

    with open(PARSED / "phase1_rows.json", "w") as f:
        json.dump(all_data, f, indent=2)
    with open(PARSED / "phase1_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    # ----- FIGURES -----
    steps = np.arange(1, N_STEPS + 1)
    palette = {
        "v24":  "tab:red",
        "v39":  "tab:orange",
        "v39b": "tab:purple",
        "v40b": "tab:blue",
        "v41b": "tab:green",
        "v43a": "tab:brown",
    }
    labels = {
        "v24":  "v24 (hand, peak=0.30 valley=0.05 d=25)",
        "v39":  "v39 (disc EMA α=0.2)",
        "v39b": "v39b (disc EMA α=0.5)",
        "v40b": "v40b (NLL linear)",
        "v41b": "v41b (ESS saturating)",
        "v43a": "v43a (KL-Lagrangian)",
    }
    mu_true = v24_hand_schedule()

    # FIG 1: mu trajectories overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(steps, mu_true, "k--", lw=2.0, label="v24 hand (implied schedule)", alpha=0.7)
    for v in ["v24", "v39", "v39b", "v40b", "v41b", "v43a"]:
        if v not in all_data:
            continue
        mu = series_of(all_data[v], "chord/mu")
        ax.plot(steps, mu, color=palette[v], lw=1.5, alpha=0.85, label=labels[v])
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\mu$ (CHORD mixing)")
    ax.set_title(r"Phase 1 — $\mu$ trajectories (v24 hand-tuned vs 5 adaptive variants)")
    ax.set_ylim(-0.005, 0.35)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_phase1_mu_trajectories.png", dpi=150)
    plt.close()

    # FIG 2: performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        ("actor/grad_norm", "grad_norm"),
        ("actor/kl_loss", "kl_loss"),
        ("actor/entropy_loss", "entropy_loss"),
        ("critic/rewards_onpolicy/mean", "rewards_onpolicy"),
    ]
    for (key, title), ax in zip(panels, axes.flat):
        for v in ["v24", "v39", "v39b", "v40b", "v41b", "v43a"]:
            if v not in all_data:
                continue
            y = series_of(all_data[v], key)
            y_smooth = smooth(y, w=5)
            ax.plot(steps, y_smooth, color=palette[v], lw=1.4, alpha=0.9, label=labels[v])
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_phase1_performance_metrics.png", dpi=150)
    plt.close()

    # FIG 3: each variant's own adaptive signal
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # (a) disc_acc_ema for v39/v39b
    ax = axes[0, 0]
    for v in ["v39", "v39b"]:
        if v not in all_data:
            continue
        d = series_of(all_data[v], "chord/disc_acc_ema")
        if np.isfinite(d).any():
            ax.plot(steps, d, color=palette[v], lw=1.5, label=f"{v} chord/disc_acc_ema")
    # dr3/disc_acc underlay for v24 (pre-EMA)
    d24 = series_of(all_data.get("v24", []), "dr3/disc_acc") if "v24" in all_data else None
    if d24 is not None and np.isfinite(d24).any():
        ax.plot(steps, d24, color=palette["v24"], lw=0.9, alpha=0.5, label="v24 dr3/disc_acc (raw)")
    ax.set_title("disc_acc (adaptive signal for v39/v39b)")
    ax.set_xlabel("step"); ax.set_ylabel("disc_acc"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for v in ["v40b"]:
        if v not in all_data:
            continue
        e = series_of(all_data[v], "chord/nll_ema")
        c = series_of(all_data[v], "chord/nll_current")
        if np.isfinite(c).any():
            ax.plot(steps, c, color="tab:cyan", lw=0.8, alpha=0.5, label=f"{v} nll_current (raw)")
        if np.isfinite(e).any():
            ax.plot(steps, e, color=palette[v], lw=1.5, label=f"{v} nll_ema")
    ax.axhline(0.65, color="k", ls=":", label=r"$\tau$=0.65 floor")
    ax.set_title("NLL (adaptive signal for v40b)")
    ax.set_xlabel("step"); ax.set_ylabel("NLL"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for v in ["v43a"]:
        if v not in all_data:
            continue
        m = series_of(all_data[v], "chord/kl_step_mult")
        if np.isfinite(m).any():
            ax.plot(steps, m, color=palette[v], lw=1.5, label=f"{v} kl_step_mult")
    ax.axhline(1.0, color="k", ls=":", label="mult=1 (no change)")
    ax.set_title("kl_step_mult (adaptive signal for v43a)")
    ax.set_xlabel("step"); ax.set_ylabel("step multiplier"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for v in ["v41b"]:
        if v not in all_data:
            continue
        r = series_of(all_data[v], "chord/ess_ratio")
        if np.isfinite(r).any():
            ax.plot(steps, r, color=palette[v], lw=1.5, label=f"{v} ess_ratio")
        e = series_of(all_data[v], "chord/ess_ema")
        if np.isfinite(e).any():
            ax.plot(steps, e, color="tab:olive", lw=1.2, label=f"{v} ess_ema")
    ax.set_title("ESS ratio (adaptive signal for v41b)")
    ax.set_xlabel("step"); ax.set_ylabel("ratio"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_phase1_signals.png", dpi=150)
    plt.close()

    # FIG 4: v43a Lagrangian diagnostic
    if "v43a" in all_data:
        rows = all_data["v43a"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cost = series_of(rows, "chord/kl_cost_ema")
        budget_ema = series_of(rows, "chord/kl_budget_ema")
        budget = series_of(rows, "chord/kl_budget")
        # Use whichever is non-NaN more often
        b = budget_ema if np.isfinite(budget_ema).sum() > np.isfinite(budget).sum() else budget
        axes[0].plot(steps, cost, color="tab:red", lw=1.5, label="kl_cost_ema")
        axes[0].plot(steps, b, color="tab:green", lw=1.5, label="kl_budget(_ema)")
        axes[0].set_title("v43a: kl_cost vs kl_budget"); axes[0].grid(alpha=0.3); axes[0].legend()
        # Add inset: cost - budget (the exponent in dual ascent)
        diff = cost - b
        ax2 = axes[0].twinx()
        ax2.plot(steps, smooth(diff, 5), color="tab:gray", lw=1.0, alpha=0.6, label="cost - budget")
        ax2.axhline(0, color="k", ls=":", lw=0.5)
        ax2.set_ylabel("cost - budget (gray)")

        lag = series_of(rows, "chord/mu_lagrange_state")
        mult = series_of(rows, "chord/kl_step_mult")
        axes[1].plot(steps, lag, color="tab:brown", lw=1.5, label="mu_lagrange_state")
        ax2 = axes[1].twinx()
        ax2.plot(steps, mult, color="tab:gray", lw=1.0, alpha=0.6, label="kl_step_mult")
        axes[1].set_title("v43a: Lagrange state & step mult"); axes[1].grid(alpha=0.3)
        axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")

        mu = series_of(rows, "chord/mu")
        gated = series_of(rows, "chord/mu_adaptive_gated")
        axes[2].plot(steps, mu, color="tab:red", lw=1.5, label="chord/mu (applied)")
        axes[2].plot(steps, gated, color="tab:orange", lw=1.0, alpha=0.6, label="chord/mu_adaptive_gated")
        axes[2].set_title("v43a: chord/mu"); axes[2].grid(alpha=0.3); axes[2].legend()
        plt.tight_layout()
        plt.savefig(FIGDIR / "fig_v43a_lagrangian_diagnostic.png", dpi=150)
        plt.close()

    # FIG 5: v40b NLL pollution
    if "v40b" in all_data:
        rows = all_data["v40b"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sft = series_of(rows, "chord/sft_loss")
        # Count zero-valued steps
        zero_frac = float(np.mean(sft[np.isfinite(sft)] == 0.0)) if np.isfinite(sft).any() else 0.0
        axes[0].plot(steps, sft, color="tab:blue", lw=0.8, alpha=0.7, label=f"sft_loss (raw), zero frac={zero_frac:.2f}")
        axes[0].plot(steps, smooth(sft, 5), color="tab:blue", lw=1.5, alpha=1.0, label="sft_loss (MA5)")
        axes[0].axhline(0.65, color="k", ls=":", label=r"$\tau$=0.65 linear floor")
        axes[0].set_title("v40b: chord/sft_loss"); axes[0].grid(alpha=0.3); axes[0].legend()

        nll_ema = series_of(rows, "chord/nll_ema")
        nll_cur = series_of(rows, "chord/nll_current")
        axes[1].plot(steps, nll_cur, color="tab:cyan", lw=0.8, alpha=0.5, label="nll_current")
        axes[1].plot(steps, nll_ema, color="tab:blue", lw=1.5, label="nll_ema")
        axes[1].axhline(0.65, color="k", ls=":", label=r"$\tau$=0.65")
        axes[1].set_title("v40b: nll_ema vs nll_current"); axes[1].grid(alpha=0.3); axes[1].legend()

        mu = series_of(rows, "chord/mu")
        gated = series_of(rows, "chord/mu_adaptive_gated")
        axes[2].plot(steps, mu, color="tab:red", lw=1.5, label="chord/mu (applied)")
        axes[2].plot(steps, gated, color="tab:orange", lw=1.0, alpha=0.6, label="chord/mu_adaptive_gated")
        axes[2].set_title("v40b: chord/mu"); axes[2].grid(alpha=0.3); axes[2].legend()
        plt.tight_layout()
        plt.savefig(FIGDIR / "fig_v40b_nll_pollution.png", dpi=150)
        plt.close()

    # ----- Compute MAE between each variant's mu and v24 hand-tuned schedule -----
    mae_rows = {}
    for v in ["v24", "v39", "v39b", "v40b", "v41b", "v43a"]:
        if v not in all_data:
            continue
        mu = series_of(all_data[v], "chord/mu")
        # 3 windows: early (s1-s25), mid (s26-s50), full (s1-s100)
        diff = mu - mu_true
        def _mae(a, b):
            seg = diff[a:b]
            seg = seg[np.isfinite(seg)]
            return float("nan") if seg.size == 0 else float(np.mean(np.abs(seg)))
        mae_rows[v] = {
            "mae_s1_25": _mae(0, 25),
            "mae_s26_50": _mae(25, 50),
            "mae_s51_100": _mae(50, 100),
            "mae_full": _mae(0, 100),
        }
    with open(PARSED / "phase1_mae_vs_v24.json", "w") as f:
        json.dump(mae_rows, f, indent=2)

    # Print final summary table
    print("\n===== SUMMARY =====")
    keys_tbl = [
        "mu_s1", "mu_s5", "mu_s10", "mu_s25", "mu_s50", "mu_s100",
        "mu_mean_s1_25", "mu_mean_s26_50", "mu_mean_s51_100",
        "success_at_100", "success_mean_s51_100",
        "reward_at_100", "reward_mean_s51_100",
        "grad_norm_mean_s1_25", "grad_norm_mean_s26_100",
        "kl_loss_mean_s1_25", "kl_loss_mean_s26_100",
        "entropy_loss_mean_s1_25", "entropy_loss_mean_s26_100",
        "sft_loss_mean_all",
        "mu_mode_first",
        "disc_acc_ema_s25", "nll_ema_final",
        "kl_cost_ema_final", "kl_budget_final", "kl_step_mult_mean", "mu_lagrange_final",
        "ess_ratio_final", "ess_ema_final", "ess_off_final",
    ]
    header = "metric" + "".join(f"{v:>10}" for v in VARIANTS.keys())
    print(header)
    for k in keys_tbl:
        vals = [all_summary.get(v, {}).get(k, float("nan")) for v in VARIANTS.keys()]
        line = f"{k:<32}" + "".join(f"{x:>10.3f}" if np.isfinite(x) else f"{'nan':>10}" for x in vals)
        print(line)
    print("\n===== MAE vs v24 hand schedule =====")
    for v, mae in mae_rows.items():
        print(f"{v}: s1-25={mae['mae_s1_25']:.3f}  s26-50={mae['mae_s26_50']:.3f}  s51-100={mae['mae_s51_100']:.3f}  full={mae['mae_full']:.3f}")


if __name__ == "__main__":
    main()
