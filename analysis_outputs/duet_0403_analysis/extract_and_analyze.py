#!/usr/bin/env python3
"""Extract wandb data for DUET 0403 (j2rle81i) and LUFFY (o405qtk1), then create analysis plots."""

import wandb
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("analysis_outputs/duet_0403_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Runs to extract
RUNS = {
    "DUET_0403": "j2rle81i",
    "LUFFY": "o405qtk1",
}

# Metrics to extract
METRICS = [
    "critic/success_onpolicy/mean",
    "actor/kl_loss",
    "actor/entropy",
    "duet/teacher_gradient_share",
    "diag/adv_teacher_sample_mean",
    "diag/adv_onpolicy_sample_mean",
    "dr3/w_off_mean",
    "dr3/disc_acc",
    "state_channel/bonus_vs_reward_ratio",
    "actor/pg_loss",
    "actor/off_pg_loss",
    "diag/teacher_sample_ratio",
    "dr3/ess_ratio",
    "actor/total_loss",
]

def extract_wandb_data():
    """Extract all metrics from wandb runs."""
    api = wandb.Api()
    entity = "qisheng001-nanyang-technological-university-singapore"
    project = "agentevolver"
    all_data = {}

    for name, run_id in RUNS.items():
        print(f"Extracting {name} ({run_id})...")
        run = api.run(f"{entity}/{project}/{run_id}")
        run_data = {}

        for metric in METRICS:
            history = list(run.scan_history(keys=[metric, "_step"], page_size=10000))
            if history:
                steps = [h["_step"] for h in history if metric in h]
                values = [h[metric] for h in history if metric in h]
                run_data[metric] = {"steps": steps, "values": values}
                print(f"  {metric}: {len(steps)} points, range [{min(values):.4f}, {max(values):.4f}]")
            else:
                print(f"  {metric}: NO DATA")

        all_data[name] = run_data

    # Save raw data
    with open(OUT_DIR / "wandb_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved raw data to {OUT_DIR / 'wandb_data.json'}")
    return all_data


def plot_training_curve_with_collapse(data):
    """Plot 1: Training success curve with collapse region highlighted."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # DUET 0403
    d = data["DUET_0403"]["critic/success_onpolicy/mean"]
    ax.plot(d["steps"], d["values"], 'b-', linewidth=2, label="DUET 0403", alpha=0.9)

    # LUFFY
    if "LUFFY" in data and "critic/success_onpolicy/mean" in data["LUFFY"]:
        d2 = data["LUFFY"]["critic/success_onpolicy/mean"]
        ax.plot(d2["steps"], d2["values"], 'r-', linewidth=2, label="LUFFY", alpha=0.9)

    # Highlight collapse region
    ax.axvspan(80, 100, alpha=0.15, color='red', label='Collapse region (steps 80-100)')

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Success Rate (on-policy)", fontsize=13)
    ax.set_title("DUET 0403 vs LUFFY: Training Success with Collapse Region", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot1_training_curve_collapse.png", dpi=150)
    plt.close()
    print("Saved plot1_training_curve_collapse.png")


def plot_collapse_diagnostics(data):
    """Plot 2: KL + entropy + teacher_gradient_share around collapse."""
    duet = data["DUET_0403"]

    metrics_to_plot = [
        ("actor/kl_loss", "KL Loss", "tab:red"),
        ("actor/entropy", "Entropy", "tab:blue"),
        ("duet/teacher_gradient_share", "Teacher Grad Share", "tab:green"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i, (metric, label, color) in enumerate(metrics_to_plot):
        ax = axes[i]
        if metric in duet:
            d = duet[metric]
            ax.plot(d["steps"], d["values"], color=color, linewidth=1.5, label=label)
            ax.axvspan(80, 100, alpha=0.15, color='red')
            ax.set_ylabel(label, fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{metric}: NO DATA", transform=ax.transAxes, ha='center')

    axes[-1].set_xlabel("Training Step", fontsize=12)
    fig.suptitle("DUET 0403: Collapse Diagnostics (steps 80-100 highlighted)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot2_collapse_diagnostics.png", dpi=150)
    plt.close()
    print("Saved plot2_collapse_diagnostics.png")


def plot_full_diagnostics(data):
    """Plot 3: Full multi-panel diagnostics for DUET 0403."""
    duet = data["DUET_0403"]

    metrics_panels = [
        ("critic/success_onpolicy/mean", "Success Rate"),
        ("actor/kl_loss", "KL Loss"),
        ("actor/entropy", "Entropy"),
        ("duet/teacher_gradient_share", "Teacher Grad Share"),
        ("diag/adv_teacher_sample_mean", "Adv Teacher Mean"),
        ("diag/adv_onpolicy_sample_mean", "Adv OnPolicy Mean"),
        ("dr3/w_off_mean", "DR3 w_off Mean"),
        ("dr3/disc_acc", "DR3 Disc Accuracy"),
        ("state_channel/bonus_vs_reward_ratio", "SC Bonus/Reward Ratio"),
        ("actor/pg_loss", "PG Loss"),
        ("actor/off_pg_loss", "Off-PG Loss"),
        ("dr3/ess_ratio", "ESS Ratio"),
    ]

    n = len(metrics_panels)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
    axes = axes.flatten()

    for i, (metric, label) in enumerate(metrics_panels):
        ax = axes[i]
        if metric in duet:
            d = duet[metric]
            ax.plot(d["steps"], d["values"], 'b-', linewidth=1.2)
            ax.axvspan(80, 100, alpha=0.15, color='red')
            ax.set_title(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            # Add value annotations at key points
            steps_arr = np.array(d["steps"])
            vals_arr = np.array(d["values"])
            # Find values near step 70, 80, 90, 100 if they exist
            for target in [70, 80, 90, 100]:
                mask = np.abs(steps_arr - target) <= 2
                if mask.any():
                    idx = np.argmin(np.abs(steps_arr - target))
                    ax.annotate(f'{vals_arr[idx]:.3f}',
                               xy=(steps_arr[idx], vals_arr[idx]),
                               fontsize=7, color='darkred',
                               textcoords="offset points", xytext=(0, 8))
        else:
            ax.text(0.5, 0.5, f"NO DATA:\n{metric}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='gray')
            ax.set_title(label, fontsize=11, color='gray')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DUET 0403: Full Training Diagnostics (collapse region in red)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot3_full_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot3_full_diagnostics.png")


def plot_luffy_comparison(data):
    """Plot 4: LUFFY stability check at steps 80-100."""
    luffy = data.get("LUFFY", {})
    duet = data["DUET_0403"]

    compare_metrics = [
        ("critic/success_onpolicy/mean", "Success Rate"),
        ("actor/kl_loss", "KL Loss"),
        ("actor/entropy", "Entropy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (metric, label) in enumerate(compare_metrics):
        ax = axes[i]
        if metric in duet:
            d = duet[metric]
            ax.plot(d["steps"], d["values"], 'b-', linewidth=1.5, label="DUET 0403", alpha=0.8)
        if metric in luffy:
            d2 = luffy[metric]
            ax.plot(d2["steps"], d2["values"], 'r-', linewidth=1.5, label="LUFFY", alpha=0.8)
        ax.axvspan(80, 100, alpha=0.15, color='red')
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Training Step")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DUET 0403 vs LUFFY: Stability Comparison at Steps 80-100", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot4_luffy_comparison.png", dpi=150)
    plt.close()
    print("Saved plot4_luffy_comparison.png")


def print_collapse_analysis(data):
    """Print detailed numerical analysis of the collapse."""
    duet = data["DUET_0403"]

    print("\n" + "="*80)
    print("COLLAPSE ANALYSIS: DUET 0403 (steps 70-100)")
    print("="*80)

    for metric in METRICS:
        if metric not in duet:
            continue
        steps = np.array(duet[metric]["steps"])
        vals = np.array(duet[metric]["values"])

        # Get pre-collapse (60-80), collapse (80-100), and peak
        pre_mask = (steps >= 60) & (steps < 80)
        collapse_mask = (steps >= 80) & (steps <= 100)

        pre_vals = vals[pre_mask] if pre_mask.any() else np.array([])
        collapse_vals = vals[collapse_mask] if collapse_mask.any() else np.array([])

        peak_idx = np.argmax(vals) if "success" in metric else None

        print(f"\n{metric}:")
        if len(pre_vals) > 0:
            print(f"  Pre-collapse (60-80):  mean={np.mean(pre_vals):.4f}, std={np.std(pre_vals):.4f}")
        if len(collapse_vals) > 0:
            print(f"  Collapse (80-100):     mean={np.mean(collapse_vals):.4f}, std={np.std(collapse_vals):.4f}")
        if len(pre_vals) > 0 and len(collapse_vals) > 0:
            delta = np.mean(collapse_vals) - np.mean(pre_vals)
            pct = (delta / (abs(np.mean(pre_vals)) + 1e-8)) * 100
            print(f"  Change:                {delta:+.4f} ({pct:+.1f}%)")
        if peak_idx is not None:
            print(f"  Peak:                  {vals[peak_idx]:.4f} at step {steps[peak_idx]}")
        print(f"  Overall range:         [{vals.min():.4f}, {vals.max():.4f}]")

    # LUFFY comparison
    luffy = data.get("LUFFY", {})
    if "critic/success_onpolicy/mean" in luffy:
        print("\n" + "="*80)
        print("LUFFY STABILITY CHECK (steps 80-100)")
        print("="*80)
        for metric in ["critic/success_onpolicy/mean", "actor/kl_loss", "actor/entropy"]:
            if metric not in luffy:
                continue
            steps = np.array(luffy[metric]["steps"])
            vals = np.array(luffy[metric]["values"])
            pre_mask = (steps >= 60) & (steps < 80)
            collapse_mask = (steps >= 80) & (steps <= 100)
            pre_vals = vals[pre_mask] if pre_mask.any() else np.array([])
            collapse_vals = vals[collapse_mask] if collapse_mask.any() else np.array([])
            print(f"\n{metric}:")
            if len(pre_vals) > 0:
                print(f"  Steps 60-80:  mean={np.mean(pre_vals):.4f}")
            if len(collapse_vals) > 0:
                print(f"  Steps 80-100: mean={np.mean(collapse_vals):.4f}")
            if len(pre_vals) > 0 and len(collapse_vals) > 0:
                delta = np.mean(collapse_vals) - np.mean(pre_vals)
                print(f"  Change:       {delta:+.4f}")


def save_summary(data):
    """Save a text summary of findings."""
    duet = data["DUET_0403"]
    lines = []
    lines.append("# DUET 0403 Collapse Analysis Summary")
    lines.append(f"# Generated: 2026-04-02")
    lines.append(f"# Wandb run: j2rle81i")
    lines.append("")

    # Validation score
    lines.append("## Validation Score (step 50)")
    lines.append("- Average reward: 0.6456")
    lines.append("- Average success: 0.305 (30.5%)")
    lines.append("")

    # Key metrics at collapse
    lines.append("## Key Metrics at Collapse")
    for metric in METRICS:
        if metric not in duet:
            continue
        steps = np.array(duet[metric]["steps"])
        vals = np.array(duet[metric]["values"])
        pre_mask = (steps >= 60) & (steps < 80)
        collapse_mask = (steps >= 80) & (steps <= 100)
        pre_vals = vals[pre_mask] if pre_mask.any() else np.array([])
        collapse_vals = vals[collapse_mask] if collapse_mask.any() else np.array([])
        if len(pre_vals) > 0 and len(collapse_vals) > 0:
            delta = np.mean(collapse_vals) - np.mean(pre_vals)
            lines.append(f"- {metric}: {np.mean(pre_vals):.4f} -> {np.mean(collapse_vals):.4f} ({delta:+.4f})")

    with open(OUT_DIR / "analysis_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved analysis_summary.txt")


if __name__ == "__main__":
    # Step 1: Extract data
    data = extract_wandb_data()

    # Step 2: Create all plots
    plot_training_curve_with_collapse(data)
    plot_collapse_diagnostics(data)
    plot_full_diagnostics(data)
    plot_luffy_comparison(data)

    # Step 3: Numerical analysis
    print_collapse_analysis(data)

    # Step 4: Save summary
    save_summary(data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE. All outputs saved to:", OUT_DIR)
    print("="*80)
