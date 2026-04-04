#!/usr/bin/env python3
"""Enhanced diagnostic plots for DUET 0403 collapse analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("analysis_outputs/duet_0403_analysis")

with open(OUT_DIR / "wandb_data.json") as f:
    data = json.load(f)

duet = data["DUET_0403"]
luffy = data["LUFFY"]


def smooth(values, window=5):
    """Simple moving average."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def plot_collapse_mechanism():
    """Main figure: 6-panel showing the collapse mechanism chain."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    panels = [
        # (row, col, metric, label, duet_color, luffy_key, ylabel)
        (0, 0, "critic/success_onpolicy/mean", "Success Rate (on-policy)", True),
        (0, 1, "actor/kl_loss", "KL Loss", True),
        (1, 0, "dr3/disc_acc", "DR3 Discriminator Accuracy", False),
        (1, 1, "dr3/w_off_mean", "DR3 w_off Mean (→1.0 = no correction)", False),
        (2, 0, "diag/adv_onpolicy_sample_mean", "On-policy Advantage Mean", False),
        (2, 1, "duet/teacher_gradient_share", "Teacher Gradient Share", True),
    ]

    for row, col, metric, title, show_luffy in panels:
        ax = axes[row][col]

        # DUET
        if metric in duet:
            d = duet[metric]
            steps, vals = np.array(d["steps"]), np.array(d["values"])
            ax.plot(steps, vals, 'b-', alpha=0.3, linewidth=0.8)
            # Smoothed
            if len(vals) >= 5:
                sm = smooth(vals, 5)
                sm_steps = steps[2:2+len(sm)]
                ax.plot(sm_steps, sm, 'b-', linewidth=2.5, label="DUET 0403")

        # LUFFY (where applicable)
        if show_luffy and metric in luffy:
            d2 = luffy[metric]
            steps2, vals2 = np.array(d2["steps"]), np.array(d2["values"])
            ax.plot(steps2, vals2, 'r-', alpha=0.3, linewidth=0.8)
            if len(vals2) >= 5:
                sm2 = smooth(vals2, 5)
                sm_steps2 = steps2[2:2+len(sm2)]
                ax.plot(sm_steps2, sm2, 'r-', linewidth=2.5, label="LUFFY")

        # Collapse region
        ax.axvspan(80, 100, alpha=0.1, color='red')
        ax.axvline(80, color='red', linestyle='--', alpha=0.4, linewidth=1)

        # Reference lines
        if metric == "dr3/w_off_mean":
            ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='w=1.0 (no correction)')
        if metric == "dr3/disc_acc":
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random chance')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc='best')
        if row == 2:
            ax.set_xlabel("Training Step", fontsize=11)

    fig.suptitle("DUET 0403 Collapse Mechanism Analysis\n"
                 "(red shading = collapse region steps 80-100)",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot5_collapse_mechanism.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot5_collapse_mechanism.png")


def plot_duet_vs_luffy_head_to_head():
    """Head-to-head full training curves with smoothing and divergence annotation."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # DUET
    d = duet["critic/success_onpolicy/mean"]
    steps_d, vals_d = np.array(d["steps"]), np.array(d["values"])
    ax.plot(steps_d, vals_d, color='#4477AA', alpha=0.25, linewidth=0.8)
    sm_d = smooth(vals_d, 7)
    sm_steps_d = steps_d[3:3+len(sm_d)]
    ax.plot(sm_steps_d, sm_d, color='#4477AA', linewidth=3, label="DUET 0403 (7-step avg)")

    # LUFFY
    d2 = luffy["critic/success_onpolicy/mean"]
    steps_l, vals_l = np.array(d2["steps"]), np.array(d2["values"])
    ax.plot(steps_l, vals_l, color='#CC6677', alpha=0.25, linewidth=0.8)
    sm_l = smooth(vals_l, 7)
    sm_steps_l = steps_l[3:3+len(sm_l)]
    ax.plot(sm_steps_l, sm_l, color='#CC6677', linewidth=3, label="LUFFY (7-step avg)")

    # Annotations
    ax.axvspan(80, 100, alpha=0.08, color='red')
    ax.annotate('Collapse begins\n(step ~82)',
                xy=(82, 0.45), xytext=(60, 0.75),
                fontsize=11, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

    # Phase annotations
    ax.axvspan(0, 30, alpha=0.03, color='green')
    ax.axvspan(30, 80, alpha=0.03, color='blue')
    ax.text(15, 0.02, "Warmup", ha='center', fontsize=9, color='green', alpha=0.7)
    ax.text(55, 0.02, "DUET leads", ha='center', fontsize=9, color='blue', alpha=0.7)
    ax.text(90, 0.02, "Collapse", ha='center', fontsize=9, color='red', alpha=0.7)

    # Final values
    last_duet = np.mean(vals_d[-5:])
    last_luffy = np.mean(vals_l[-5:])
    ax.text(steps_d[-1]+1, last_duet, f'{last_duet:.3f}', fontsize=10, color='#4477AA', fontweight='bold')
    ax.text(steps_l[-1]+1, last_luffy, f'{last_luffy:.3f}', fontsize=10, color='#CC6677', fontweight='bold')

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Success Rate (on-policy)", fontsize=13)
    ax.set_title("DUET 0403 vs LUFFY: Full Training Trajectory", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max(steps_d[-1], steps_l[-1]) + 5)
    ax.set_ylim(-0.02, 0.9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot6_head_to_head.png", dpi=150)
    plt.close()
    print("Saved plot6_head_to_head.png")


def plot_dr3_degradation_detail():
    """Detailed DR3 degradation: disc_acc vs w_off_mean vs success, showing the causal chain."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: disc_acc
    d = duet["dr3/disc_acc"]
    steps, vals = np.array(d["steps"]), np.array(d["values"])
    ax1.plot(steps, vals, 'g-', linewidth=2, label="DR3 disc_acc")
    ax1.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='Warning threshold (0.9)')
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random chance (0.5)')
    ax1.axvspan(80, 100, alpha=0.1, color='red')
    ax1.set_ylabel("Discriminator\nAccuracy", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_title("DR3 Degradation Chain → Policy Collapse", fontsize=13, fontweight='bold')

    # Panel 2: w_off_mean
    d = duet["dr3/w_off_mean"]
    steps_w, vals_w = np.array(d["steps"]), np.array(d["values"])
    ax2.plot(steps_w, vals_w, 'm-', linewidth=2, label="DR3 w_off_mean")
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='w=1 (uniform weight)')
    ax2.axvspan(80, 100, alpha=0.1, color='red')
    ax2.set_ylabel("Importance\nWeight Mean", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel 3: success rate
    d = duet["critic/success_onpolicy/mean"]
    steps_s, vals_s = np.array(d["steps"]), np.array(d["values"])
    ax3.plot(steps_s, vals_s, 'b-', alpha=0.4, linewidth=1)
    sm = smooth(vals_s, 7)
    sm_steps = steps_s[3:3+len(sm)]
    ax3.plot(sm_steps, sm, 'b-', linewidth=2.5, label="Success (7-step avg)")
    ax3.axvspan(80, 100, alpha=0.1, color='red')
    ax3.set_ylabel("Success Rate", fontsize=11)
    ax3.set_xlabel("Training Step", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    # Add arrows showing causal chain
    fig.text(0.02, 0.72, "1. disc_acc\n   degrades", fontsize=9, color='green', fontweight='bold',
             rotation=90, va='center')
    fig.text(0.02, 0.45, "2. w→1.0\n   (no correction)", fontsize=9, color='purple', fontweight='bold',
             rotation=90, va='center')
    fig.text(0.02, 0.2, "3. Success\n   collapses", fontsize=9, color='blue', fontweight='bold',
             rotation=90, va='center')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot7_dr3_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot7_dr3_degradation.png")


def plot_correlation_scatter():
    """Scatter: disc_acc vs success rate for DUET, colored by training phase."""
    d_acc = duet["dr3/disc_acc"]
    d_suc = duet["critic/success_onpolicy/mean"]

    # Align by step
    acc_dict = dict(zip(d_acc["steps"], d_acc["values"]))
    suc_dict = dict(zip(d_suc["steps"], d_suc["values"]))
    common_steps = sorted(set(acc_dict.keys()) & set(suc_dict.keys()))

    steps = np.array(common_steps)
    acc = np.array([acc_dict[s] for s in common_steps])
    suc = np.array([suc_dict[s] for s in common_steps])

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(acc, suc, c=steps, cmap='coolwarm', s=40, alpha=0.7, edgecolors='k', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Training Step')

    # Highlight collapse points
    mask = steps >= 80
    ax.scatter(acc[mask], suc[mask], facecolors='none', edgecolors='red', s=100, linewidth=2, label='Steps 80-100')

    ax.set_xlabel("DR3 Discriminator Accuracy", fontsize=12)
    ax.set_ylabel("Success Rate (on-policy)", fontsize=12)
    ax.set_title("DR3 Accuracy vs Success: Collapse Correlation", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot8_correlation.png", dpi=150)
    plt.close()
    print("Saved plot8_correlation.png")


if __name__ == "__main__":
    plot_collapse_mechanism()
    plot_duet_vs_luffy_head_to_head()
    plot_dr3_degradation_detail()
    plot_correlation_scatter()
    print("\nAll enhanced plots saved.")
