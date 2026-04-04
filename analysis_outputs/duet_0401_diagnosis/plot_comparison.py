import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/wandb_data.json") as f:
    data = json.load(f)

# Remove the keys list
if "_available_keys_0401" in data:
    del data["_available_keys_0401"]

colors = {"DUET_0401": "#d62728", "DUET_orig": "#1f77b4", "LUFFY": "#2ca02c"}
labels = {"DUET_0401": "DUET 0401 (std floor + stage SC)", "DUET_orig": "DUET orig", "LUFFY": "LUFFY"}

def smooth(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid').tolist()

# ============================================================
# Figure 1: Training Curves (3 subplots)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("DUET 0401 vs DUET orig vs LUFFY — Training Curves", fontsize=14, fontweight='bold')

# 1a: Success rate
ax = axes[0]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "critic/success_onpolicy/mean"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, alpha=0.3, color=colors[run])
        sv = smooth(v)
        ss = s[len(s)-len(sv):]
        ax.plot(ss, sv, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("On-Policy Success Rate")
ax.set_xlabel("Step")
ax.set_ylabel("Success Rate")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1b: On-policy rewards
ax = axes[1]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "critic/rewards_onpolicy/mean"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, alpha=0.3, color=colors[run])
        sv = smooth(v)
        ss = s[len(s)-len(sv):]
        ax.plot(ss, sv, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("On-Policy Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1c: KL divergence
ax = axes[2]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "actor/kl_loss"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, alpha=0.3, color=colors[run])
        sv = smooth(v)
        ss = s[len(s)-len(sv):]
        ax.plot(ss, sv, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("KL Loss")
ax.set_xlabel("Step")
ax.set_ylabel("KL")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/training_curves.png", dpi=150)
plt.close()
print("Saved training_curves.png")

# ============================================================
# Figure 2: Teacher Gradient Share + Teacher Advantages
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Teacher Influence Comparison", fontsize=14, fontweight='bold')

# 2a: Teacher gradient share
ax = axes[0]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "duet/teacher_gradient_share"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("Teacher Gradient Share")
ax.set_xlabel("Step")
ax.set_ylabel("Share")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.125, color='gray', linestyle='--', alpha=0.5, label='Data ratio (12.5%)')

# 2b: Teacher advantage mean (clip for readability)
ax = axes[1]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "diag/adv_teacher_sample_mean"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = [min(x, 10) for x in data[run][m]["values"]]  # clip for readability
        ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("Teacher Advantage Mean (clipped at 10)")
ax.set_xlabel("Step")
ax.set_ylabel("Advantage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2c: On-policy advantage mean
ax = axes[2]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "diag/adv_onpolicy_sample_mean"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("On-Policy Advantage Mean")
ax.set_xlabel("Step")
ax.set_ylabel("Advantage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/teacher_influence.png", dpi=150)
plt.close()
print("Saved teacher_influence.png")

# ============================================================
# Figure 3: State Channel Metrics (0401 only, with orig for comparison)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("State Channel Metrics — DUET 0401 vs DUET orig", fontsize=14, fontweight='bold')

sc_metrics = [
    ("state_channel/coverage_mean", "SC Coverage"),
    ("state_channel/bonus_vs_reward_ratio", "SC Bonus/Reward Ratio"),
    ("state_channel/progress_mean", "SC Progress (all)"),
    ("state_channel/progress_onpolicy_mean", "SC Progress (on-policy)"),
    ("state_channel/bonus_total_mean", "SC Bonus Total"),
    ("state_channel/step_delta_mean", "SC Step Delta"),
]

for idx, (m, title) in enumerate(sc_metrics):
    ax = axes[idx // 3][idx % 3]
    for run in ["DUET_0401", "DUET_orig"]:
        if data[run][m]["values"]:
            s = data[run][m]["steps"]
            v = data[run][m]["values"]
            ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/state_channel_metrics.png", dpi=150)
plt.close()
print("Saved state_channel_metrics.png")

# ============================================================
# Figure 4: DR3 Discriminator
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("DR3 Discriminator — DUET 0401 vs DUET orig", fontsize=14, fontweight='bold')

ax = axes[0]
for run in ["DUET_0401", "DUET_orig"]:
    m = "dr3/disc_acc"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = data[run][m]["values"]
        ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("Discriminator Accuracy")
ax.set_xlabel("Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
    m = "actor/pg_loss"
    if data[run][m]["values"]:
        s = data[run][m]["steps"]
        v = [max(x, -100) for x in data[run][m]["values"]]  # clip
        ax.plot(s, v, label=labels[run], color=colors[run], linewidth=2)
ax.set_title("PG Loss (clipped at -100)")
ax.set_xlabel("Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/dr3_metrics.png", dpi=150)
plt.close()
print("Saved dr3_metrics.png")

# ============================================================
# Print summary table
# ============================================================
print("\n" + "="*80)
print("SUMMARY TABLE — Key Metrics at Final Step")
print("="*80)
print(f"{'Metric':<45} {'DUET 0401':>12} {'DUET orig':>12} {'LUFFY':>12}")
print("-"*80)

summary_metrics = [
    ("critic/success_onpolicy/mean", "Success Rate (on-policy)"),
    ("critic/rewards_onpolicy/mean", "Reward (on-policy)"),
    ("duet/teacher_gradient_share", "Teacher Gradient Share"),
    ("diag/adv_teacher_sample_mean", "Teacher Advantage Mean"),
    ("diag/adv_onpolicy_sample_mean", "On-Policy Advantage Mean"),
    ("actor/kl_loss", "KL Loss"),
    ("actor/entropy_loss", "Entropy Loss"),
    ("actor/pg_loss", "PG Loss"),
    ("state_channel/coverage_mean", "SC Coverage"),
    ("state_channel/bonus_vs_reward_ratio", "SC Bonus/Reward Ratio"),
    ("state_channel/bonus_total_mean", "SC Bonus Total"),
    ("dr3/disc_acc", "DR3 Disc Accuracy"),
]

for m, label in summary_metrics:
    vals = []
    for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
        if data[run].get(m, {}).get("values"):
            vals.append(f"{data[run][m]['values'][-1]:.4f}")
        else:
            vals.append("N/A")
    print(f"{label:<45} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

print("="*80)
