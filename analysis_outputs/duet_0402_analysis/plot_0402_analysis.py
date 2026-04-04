"""
DUET 0402 Analysis: 3-way comparison (0402 vs orig vs LUFFY)
Focus: disc_temperature=2.5 impact, divergence point, DR3 dynamics
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
with open("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/wandb_data.json") as f:
    data = json.load(f)

COLORS = {
    "DUET_0402": "#e74c3c",    # Red
    "DUET_orig": "#95a5a6",    # Gray
    "LUFFY":     "#2ecc71",    # Green
}
LABELS = {
    "DUET_0402": "DUET 0402 (T=2.5)",
    "DUET_orig": "DUET orig",
    "LUFFY":     "LUFFY",
}

def get(run, metric):
    d = data[run].get(metric, {"steps": [], "values": []})
    return np.array(d["steps"]), np.array(d["values"])

def smooth(y, window=5):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

def smooth_xy(x, y, window=5):
    sy = smooth(y, window)
    sx = x[:len(sy)]
    return sx, sy

# ================================================================
# FIGURE 1: 3-way Training Curves (main result)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
for run in ["LUFFY", "DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "critic/success_onpolicy/mean")
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.2, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])

# Mark divergence point
luffy_s, luffy_v = get("LUFFY", "critic/success_onpolicy/mean")
duet_s, duet_v = get("DUET_0402", "critic/success_onpolicy/mean")
# Find where LUFFY first consistently exceeds DUET 0402
# Interpolate to common steps
common_steps = sorted(set(luffy_s) & set(duet_s))
if len(common_steps) > 5:
    luffy_map = dict(zip(luffy_s, luffy_v))
    duet_map = dict(zip(duet_s, duet_v))
    diverge_step = None
    for i, s in enumerate(common_steps[5:], 5):
        # Check if LUFFY leads for 5+ consecutive steps
        ahead = all(luffy_map.get(common_steps[j], 0) > duet_map.get(common_steps[j], 0)
                    for j in range(max(0,i-4), i+1))
        if ahead:
            diverge_step = s
            break
    if diverge_step:
        ax.axvline(x=diverge_step, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.annotate(f'Divergence ~step {diverge_step}', xy=(diverge_step, 0.05),
                   fontsize=10, color='orange', ha='left')

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("On-Policy Success Rate", fontsize=12)
ax.set_title("WebShop 3B: Training Curves (DUET 0402 vs Orig vs LUFFY)", fontsize=14)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.02, 0.45)
fig.tight_layout()
fig.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/training_curves_3way.png", dpi=150)
print("Saved training_curves_3way.png")

# ================================================================
# FIGURE 2: teacher_gradient_share comparison
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: gradient share
ax = axes[0]
for run in ["LUFFY", "DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "duet/teacher_gradient_share")
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.3, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])
ax.set_xlabel("Training Step", fontsize=11)
ax.set_ylabel("Teacher Gradient Share", fontsize=11)
ax.set_title("Teacher Gradient Share Over Training", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

# Right: teacher advantages
ax = axes[1]
for run in ["LUFFY", "DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "diag/adv_teacher_sample_mean")
    if len(steps) > 0:
        # Clip for visualization (orig explodes)
        vals_clipped = np.clip(vals, -50, 50)
        ax.plot(steps, vals_clipped, alpha=0.3, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals_clipped, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])
ax.set_xlabel("Training Step", fontsize=11)
ax.set_ylabel("Teacher Advantage Mean (clipped ±50)", fontsize=11)
ax.set_title("Teacher Advantages Over Training", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/teacher_gradient_share.png", dpi=150)
print("Saved teacher_gradient_share.png")

# ================================================================
# FIGURE 3: DR3 w_hat comparison (0402 vs orig)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# w_off_mean
ax = axes[0]
for run in ["DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "dr3/w_off_mean")
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.3, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])
ax.set_xlabel("Step"); ax.set_ylabel("w_off mean")
ax.set_title("DR3 Weight (w_off_mean)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='w=1 (equal)')

# disc_acc
ax = axes[1]
for run in ["DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "dr3/disc_acc")
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.3, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])
ax.set_xlabel("Step"); ax.set_ylabel("Discriminator Accuracy")
ax.set_title("DR3 Discriminator Accuracy", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# gap_gate_mean
ax = axes[2]
for run in ["DUET_0402", "DUET_orig"]:
    steps, vals = get(run, "dr3/gap_gate_mean")
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.3, color=COLORS[run], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS[run], linewidth=2.5, label=LABELS[run])
ax.set_xlabel("Step"); ax.set_ylabel("Gap Gate Mean")
ax.set_title("DR3 Gap Gate", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/dr3_comparison.png", dpi=150)
print("Saved dr3_comparison.png")

# ================================================================
# FIGURE 4: Head-to-head 0402 vs LUFFY diagnostic panel
# ================================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

panel_metrics = [
    ("critic/success_onpolicy/mean", "On-Policy Success Rate"),
    ("critic/rewards_onpolicy/mean", "On-Policy Rewards"),
    ("duet/teacher_gradient_share", "Teacher Gradient Share"),
    ("diag/adv_teacher_sample_mean", "Teacher Advantage Mean"),
    ("diag/adv_onpolicy_sample_mean", "On-Policy Advantage Mean"),
    ("actor/kl_loss", "KL Divergence"),
    ("actor/entropy_loss", "Entropy"),
    ("actor/pg_loss", "PG Loss"),
    ("state_channel/bonus_vs_reward_ratio", "SC Bonus/Reward Ratio"),
]

for idx, (metric, title) in enumerate(panel_metrics):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    for run in ["LUFFY", "DUET_0402"]:
        steps, vals = get(run, metric)
        if len(steps) > 0:
            # Clip extreme values for visualization
            if "adv_teacher" in metric:
                vals = np.clip(vals, -50, 50)
            if "pg_loss" in metric:
                vals = np.clip(vals, -10, 10)
            ax.plot(steps, vals, alpha=0.2, color=COLORS[run], linewidth=1)
            sx, sy = smooth_xy(steps, vals, 5)
            ax.plot(sx, sy, color=COLORS[run], linewidth=2, label=LABELS[run])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("DUET 0402 vs LUFFY: Diagnostic Panel (WebShop 3B)", fontsize=14, fontweight='bold', y=0.98)
fig.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/diagnostic_panel.png", dpi=150)
print("Saved diagnostic_panel.png")

# ================================================================
# FIGURE 5: State Channel metrics (0402 only — orig SC was broken)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sc_metrics = [
    ("state_channel/progress_onpolicy_mean", "On-Policy Progress Mean"),
    ("state_channel/bonus_vs_reward_ratio", "Bonus / Reward Ratio"),
    ("state_channel/coverage_mean", "Coverage"),
]
for i, (metric, title) in enumerate(sc_metrics):
    ax = axes[i]
    steps, vals = get("DUET_0402", metric)
    if len(steps) > 0:
        ax.plot(steps, vals, alpha=0.3, color=COLORS["DUET_0402"], linewidth=1)
        sx, sy = smooth_xy(steps, vals, 5)
        ax.plot(sx, sy, color=COLORS["DUET_0402"], linewidth=2.5, label="DUET 0402")
    # For comparison, show orig (which was broken)
    steps2, vals2 = get("DUET_orig", metric)
    if len(steps2) > 0 and any(v != 0 for v in vals2):
        ax.plot(steps2, vals2, alpha=0.3, color=COLORS["DUET_orig"], linewidth=1)
        sx2, sy2 = smooth_xy(steps2, vals2, 5)
        ax.plot(sx2, sy2, color=COLORS["DUET_orig"], linewidth=2.5, label="DUET orig")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Step")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle("State Channel Metrics: 0402 (working) vs Orig (broken)", fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0402_analysis/state_channel_metrics.png", dpi=150)
print("Saved state_channel_metrics.png")

# ================================================================
# Print summary statistics
# ================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for run in ["DUET_0402", "DUET_orig", "LUFFY"]:
    print(f"\n--- {LABELS[run]} ---")
    s, v = get(run, "critic/success_onpolicy/mean")
    if len(v) > 0:
        print(f"  Success: first={v[0]:.4f}, last={v[-1]:.4f}, max={v.max():.4f} (step {s[v.argmax()]})")
        # Best 5-step average
        if len(v) >= 5:
            best5 = max(np.convolve(v, np.ones(5)/5, 'valid'))
            print(f"  Best 5-step avg: {best5:.4f}")

    s, v = get(run, "duet/teacher_gradient_share")
    if len(v) > 0:
        print(f"  Grad share: first={v[0]:.4f}, mid={v[len(v)//2]:.4f}, last={v[-1]:.4f}")

    s, v = get(run, "diag/adv_teacher_sample_mean")
    if len(v) > 0:
        print(f"  Teacher adv: first={v[0]:.4f}, last={v[-1]:.4f}")

    s, v = get(run, "actor/kl_loss")
    if len(v) > 0:
        print(f"  KL loss: first={v[0]:.4f}, last={v[-1]:.4f}, max={v.max():.4f}")

    s, v = get(run, "dr3/w_off_mean")
    if len(v) > 0:
        print(f"  w_off mean: first={v[0]:.4f}, last={v[-1]:.4f}")

    s, v = get(run, "dr3/gap_gate_mean")
    if len(v) > 0:
        print(f"  Gap gate: first={v[0]:.4f}, last={v[-1]:.4f}")

# Divergence analysis
print("\n" + "="*70)
print("DIVERGENCE ANALYSIS: LUFFY vs DUET 0402")
print("="*70)
ls, lv = get("LUFFY", "critic/success_onpolicy/mean")
ds, dv = get("DUET_0402", "critic/success_onpolicy/mean")
lmap = dict(zip(ls, lv))
dmap = dict(zip(ds, dv))
common = sorted(set(ls) & set(ds))
print(f"\nStep-by-step comparison (smoothed with window=3):")
lvals = [lmap[s] for s in common]
dvals = [dmap[s] for s in common]
lsmooth = smooth(np.array(lvals), 3)
dsmooth = smooth(np.array(dvals), 3)
for i in range(0, len(lsmooth), 10):
    step = common[i+1]  # offset for smoothing
    diff = lsmooth[i] - dsmooth[i]
    marker = " <<< LUFFY ahead" if diff > 0.02 else (" >>> DUET ahead" if diff < -0.02 else "")
    print(f"  Step {step:3d}: LUFFY={lsmooth[i]:.4f}, DUET={dsmooth[i]:.4f}, diff={diff:+.4f}{marker}")

# Print last 10 steps
print(f"\nLast 10 steps:")
for i in range(max(0, len(lsmooth)-10), len(lsmooth)):
    step = common[i+1]
    diff = lsmooth[i] - dsmooth[i]
    marker = " <<< LUFFY ahead" if diff > 0.02 else (" >>> DUET ahead" if diff < -0.02 else "")
    print(f"  Step {step:3d}: LUFFY={lsmooth[i]:.4f}, DUET={dsmooth[i]:.4f}, diff={diff:+.4f}{marker}")

print("\nDone!")
