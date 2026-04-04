#!/usr/bin/env python3
"""Extract wandb data for ALL DUET versions + LUFFY + GRPO for comprehensive comparison."""

import wandb
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("analysis_outputs/duet_0404_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ALL runs to compare
RUNS = {
    "GRPO": "27ysbdvi",
    "LUFFY": "o405qtk1",
    "DUET_orig": "bgokw3m6",
    "DUET_0402": "4izhjhlb",
    "DUET_0403": "j2rle81i",  # Note: team lead flagged this might share ID - verify
    "DUET_0404": "j2rle81i",  # Same ID? We'll check
}

# Comprehensive metrics
METRICS = [
    "critic/success_onpolicy/mean",
    "critic/reward_onpolicy/mean",
    "actor/kl_loss",
    "actor/entropy",
    "duet/teacher_gradient_share",
    "diag/adv_teacher_sample_mean",
    "diag/adv_onpolicy_sample_mean",
    "dr3/w_off_mean",
    "dr3/w_off_std",
    "dr3/disc_acc",
    "dr3/disc_loss",
    "state_channel/bonus_vs_reward_ratio",
    "state_channel/mean_bonus",
    "actor/pg_loss",
    "actor/off_pg_loss",
    "actor/total_loss",
    "diag/teacher_sample_ratio",
    "dr3/ess_ratio",
    "dr3/w_on_mean",
]

def extract_wandb_data():
    """Extract all metrics from wandb runs."""
    api = wandb.Api()
    entity = "qisheng001-nanyang-technological-university-singapore"
    project = "agentevolver"
    all_data = {}

    for name, run_id in RUNS.items():
        print(f"\nExtracting {name} ({run_id})...")
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            print(f"  Run name: {run.name}, state: {run.state}")
            print(f"  Created: {run.created_at}")

            # Get config summary too
            config_summary = {
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
            }

            run_data = {"_meta": config_summary}

            for metric in METRICS:
                history = list(run.scan_history(keys=[metric, "_step"], page_size=10000))
                if history:
                    steps = [h["_step"] for h in history if metric in h]
                    values = [h[metric] for h in history if metric in h]
                    run_data[metric] = {"steps": steps, "values": values}
                    if values:
                        print(f"  {metric}: {len(steps)} pts, [{min(values):.4f}, {max(values):.4f}], final={values[-1]:.4f}")
                else:
                    pass  # skip silent for cleaner output

            all_data[name] = run_data
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save
    with open(OUT_DIR / "wandb_all_runs.json", "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'wandb_all_runs.json'}")
    return all_data


def print_summary_table(all_data):
    """Print a formatted summary comparison table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*120)

    key_metrics = [
        ("critic/success_onpolicy/mean", "Success Rate"),
        ("critic/reward_onpolicy/mean", "Reward"),
        ("duet/teacher_gradient_share", "Teacher Grad Share"),
        ("dr3/disc_acc", "Disc Accuracy"),
        ("dr3/w_off_mean", "W_off Mean"),
        ("state_channel/bonus_vs_reward_ratio", "SC Bonus Ratio"),
        ("actor/kl_loss", "KL Loss"),
        ("dr3/ess_ratio", "ESS Ratio"),
    ]

    runs_order = ["GRPO", "LUFFY", "DUET_orig", "DUET_0402", "DUET_0403", "DUET_0404"]

    for metric_key, metric_name in key_metrics:
        print(f"\n--- {metric_name} ({metric_key}) ---")
        print(f"{'Run':<15} {'Peak':>10} {'@Step':>8} {'Final':>10} {'Final5avg':>12} {'Trend':>10}")
        print("-" * 75)

        for run_name in runs_order:
            if run_name not in all_data:
                continue
            data = all_data[run_name]
            if metric_key not in data:
                print(f"{run_name:<15} {'N/A':>10}")
                continue

            steps = data[metric_key]["steps"]
            values = data[metric_key]["values"]
            if not values:
                print(f"{run_name:<15} {'N/A':>10}")
                continue

            peak_val = max(values)
            peak_idx = values.index(peak_val)
            peak_step = steps[peak_idx]
            final_val = values[-1]
            final5 = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)

            # Trend: compare last 20% to middle 20%
            n = len(values)
            if n >= 10:
                mid_start = int(n * 0.4)
                mid_end = int(n * 0.6)
                late_start = int(n * 0.8)
                mid_avg = np.mean(values[mid_start:mid_end])
                late_avg = np.mean(values[late_start:])
                if late_avg > mid_avg * 1.05:
                    trend = "UP"
                elif late_avg < mid_avg * 0.95:
                    trend = "DOWN"
                else:
                    trend = "STABLE"
            else:
                trend = "N/A"

            print(f"{run_name:<15} {peak_val:>10.4f} {peak_step:>8d} {final_val:>10.4f} {final5:>12.4f} {trend:>10}")


def create_training_curves_plot(all_data):
    """Create comprehensive training curves comparison."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle("DUET Evolution: All Versions vs LUFFY (WebShop 3B)", fontsize=16, fontweight='bold')

    colors = {
        "GRPO": "#888888",
        "LUFFY": "#e74c3c",
        "DUET_orig": "#3498db",
        "DUET_0402": "#2ecc71",
        "DUET_0403": "#9b59b6",
        "DUET_0404": "#f39c12",
    }

    plot_configs = [
        ("critic/success_onpolicy/mean", "Success Rate (Primary)", axes[0, 0]),
        ("critic/reward_onpolicy/mean", "Reward", axes[0, 1]),
        ("duet/teacher_gradient_share", "Teacher Gradient Share", axes[0, 2]),
        ("dr3/disc_acc", "Discriminator Accuracy", axes[1, 0]),
        ("dr3/w_off_mean", "W_off (Teacher Weight)", axes[1, 1]),
        ("state_channel/bonus_vs_reward_ratio", "SC Bonus Ratio", axes[1, 2]),
        ("actor/kl_loss", "KL Loss", axes[2, 0]),
        ("dr3/ess_ratio", "ESS Ratio", axes[2, 1]),
        ("actor/entropy", "Entropy", axes[2, 2]),
    ]

    runs_order = ["GRPO", "LUFFY", "DUET_orig", "DUET_0402", "DUET_0403", "DUET_0404"]

    for metric_key, title, ax in plot_configs:
        for run_name in runs_order:
            if run_name not in all_data:
                continue
            data = all_data[run_name]
            if metric_key not in data:
                continue
            steps = data[metric_key]["steps"]
            values = data[metric_key]["values"]
            if not values:
                continue

            linewidth = 2.5 if run_name in ["LUFFY", "DUET_0404"] else 1.5
            alpha = 1.0 if run_name in ["LUFFY", "DUET_0404"] else 0.6
            linestyle = '--' if run_name == "LUFFY" else '-'

            ax.plot(steps, values, color=colors.get(run_name, "#000"),
                   linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                   label=run_name)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "all_versions_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved all_versions_comparison.png")
    plt.close()


def create_0404_vs_0402_plot(all_data):
    """Head-to-head: 0404 vs 0402 (which was better)."""
    if "DUET_0402" not in all_data or "DUET_0404" not in all_data:
        print("Cannot create 0404 vs 0402 plot - missing data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DUET 0404 vs 0402: What Changed?", fontsize=14, fontweight='bold')

    metrics_to_compare = [
        ("critic/success_onpolicy/mean", "Success Rate", axes[0, 0]),
        ("dr3/disc_acc", "Disc Accuracy", axes[0, 1]),
        ("dr3/w_off_mean", "W_off Mean", axes[0, 2]),
        ("duet/teacher_gradient_share", "Teacher Grad Share", axes[1, 0]),
        ("state_channel/bonus_vs_reward_ratio", "SC Bonus Ratio", axes[1, 1]),
        ("actor/kl_loss", "KL Loss", axes[1, 2]),
    ]

    for metric_key, title, ax in metrics_to_compare:
        for run_name, color, ls in [("DUET_0402", "#2ecc71", "-"), ("DUET_0404", "#f39c12", "-"), ("LUFFY", "#e74c3c", "--")]:
            if run_name not in all_data or metric_key not in all_data[run_name]:
                continue
            data = all_data[run_name][metric_key]
            ax.plot(data["steps"], data["values"], color=color, linewidth=2, linestyle=ls, label=run_name)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "0404_vs_0402_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved 0404_vs_0402_comparison.png")
    plt.close()


def create_luffy_deep_analysis(all_data):
    """Deep analysis of why LUFFY is so robust."""
    if "LUFFY" not in all_data:
        print("No LUFFY data")
        return

    luffy = all_data["LUFFY"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("LUFFY Deep Analysis: Why Is It So Robust?", fontsize=14, fontweight='bold')

    luffy_metrics = [
        ("critic/success_onpolicy/mean", "Success Rate", axes[0, 0]),
        ("critic/reward_onpolicy/mean", "Reward", axes[0, 1]),
        ("duet/teacher_gradient_share", "Teacher Grad Share", axes[0, 2]),
        ("actor/kl_loss", "KL Loss", axes[1, 0]),
        ("actor/entropy", "Entropy", axes[1, 1]),
        ("diag/teacher_sample_ratio", "Teacher Sample Ratio", axes[1, 2]),
    ]

    for metric_key, title, ax in luffy_metrics:
        if metric_key in luffy:
            steps = luffy[metric_key]["steps"]
            values = luffy[metric_key]["values"]
            ax.plot(steps, values, color="#e74c3c", linewidth=2)

            # Add smoothed line
            if len(values) > 10:
                window = min(10, len(values) // 3)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                if len(smooth_steps) == len(smoothed):
                    ax.plot(smooth_steps, smoothed, color="#c0392b", linewidth=2.5, linestyle='--', alpha=0.8, label='smoothed')

            # Annotate key stats
            if values:
                ax.axhline(y=np.mean(values[-5:]), color='gray', linestyle=':', alpha=0.5)
                ax.text(0.02, 0.98, f"Final: {values[-1]:.4f}\nPeak: {max(values):.4f}\nAvg(last5): {np.mean(values[-5:]):.4f}",
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "luffy_deep_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved luffy_deep_analysis.png")
    plt.close()


def create_version_progression_table(all_data):
    """Create a clear version progression analysis."""
    versions = [
        ("DUET_orig", "Original DUET", "DR3 + SC + gap_gate"),
        ("DUET_0402", "0402 fix", "disc_temp=2.5, gap_gate ON, no SC decouple, no adv clip"),
        ("DUET_0403", "0403 fix", "SC-GRPO decouple, adv clip +-5, gap_gate OFF, disc_temp=2.5"),
        ("DUET_0404", "0404 fix", "disc_temp=1.5, gap_gate OFF, SC decouple, adv clip"),
    ]

    lines = []
    lines.append("# DUET Version Progression Analysis")
    lines.append("")
    lines.append("## Validation Scores")
    lines.append("")
    lines.append("| Version | Val@50 Reward | Val@50 Success | Val@100 Reward | Val@100 Success |")
    lines.append("|---------|--------------|----------------|----------------|-----------------|")

    val_data = {
        "GRPO": {"50": (0.276, 0.01), "100": (0.402, 0.02)},
        "LUFFY": {"50": (0.509, 0.085), "100": (0.753, 0.495)},
        "DUET_orig": {"50": (0.599, 0.225), "100": (0.725, 0.325)},
        "DUET_0401": {"50": (0.517, 0.12), "100": (0.565, 0.18)},
        "DUET_0402": {"50": (0.483, 0.065), "100": (0.735, 0.355)},
        "DUET_0402_v2": {"50": (-0.100, 0.0), "100": (-0.100, 0.0)},
        "DUET_0403": {"50": (0.646, 0.305), "100": (0.679, 0.33)},
        "DUET_0404": {"50": (0.497, 0.02), "100": (0.646, 0.235)},
    }

    for name in ["GRPO", "LUFFY", "DUET_orig", "DUET_0401", "DUET_0402", "DUET_0403", "DUET_0404"]:
        d = val_data.get(name, {})
        r50, s50 = d.get("50", ("N/A", "N/A"))
        r100, s100 = d.get("100", ("N/A", "N/A"))

        s50_str = f"{s50*100:.1f}%" if isinstance(s50, float) else s50
        s100_str = f"{s100*100:.1f}%" if isinstance(s100, float) else s100
        r50_str = f"{r50:.3f}" if isinstance(r50, float) else r50
        r100_str = f"{r100:.3f}" if isinstance(r100, float) else r100

        marker = " **BEST**" if name == "LUFFY" else ""
        lines.append(f"| {name}{marker} | {r50_str} | {s50_str} | {r100_str} | {s100_str} |")

    lines.append("")
    lines.append("## Version Changes and Effects")
    lines.append("")
    lines.append("| Version | Key Changes | Val@100 Success | Outcome |")
    lines.append("|---------|-------------|-----------------|---------|")
    lines.append("| DUET_orig | Baseline DUET | 32.5% | Below LUFFY 49.5% |")
    lines.append("| DUET_0401 | Unknown changes | 18.0% | REGRESSION |")
    lines.append("| DUET_0402 | disc_temp=2.5, gap_gate ON | 35.5% | Best DUET so far |")
    lines.append("| DUET_0403 | SC decouple, adv clip, gap_gate OFF, disc_temp=2.5 | 33.0% | Slight regression from 0402 |")
    lines.append("| DUET_0404 | disc_temp=1.5, gap_gate OFF, SC decouple, adv clip | 23.5% | WORST since 0401 |")

    lines.append("")
    lines.append("## Training Dynamics Comparison")
    lines.append("")

    # Extract peak and final success from wandb
    for run_name in ["LUFFY", "DUET_orig", "DUET_0402", "DUET_0403", "DUET_0404"]:
        if run_name in all_data and "critic/success_onpolicy/mean" in all_data[run_name]:
            d = all_data[run_name]["critic/success_onpolicy/mean"]
            vals = d["values"]
            steps = d["steps"]
            if vals:
                peak = max(vals)
                peak_step = steps[vals.index(peak)]
                final = vals[-1]
                final5 = np.mean(vals[-5:]) if len(vals) >= 5 else final
                lines.append(f"### {run_name}")
                lines.append(f"- Peak success: {peak:.4f} @ step {peak_step}")
                lines.append(f"- Final success: {final:.4f}")
                lines.append(f"- Final 5-step avg: {final5:.4f}")
                lines.append(f"- Total steps: {len(vals)}")
                lines.append("")

    with open(OUT_DIR / "version_progression.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved version_progression.md")
    return "\n".join(lines)


if __name__ == "__main__":
    all_data = extract_wandb_data()
    print_summary_table(all_data)
    create_training_curves_plot(all_data)
    create_0404_vs_0402_plot(all_data)
    create_luffy_deep_analysis(all_data)
    progression = create_version_progression_table(all_data)
    print("\n" + progression)
