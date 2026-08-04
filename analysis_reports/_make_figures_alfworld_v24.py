"""Make v24 vs v1 diagnostic figures on ALFWorld 1.5B."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/data/home/qisheng/EvolAnalsis/analysis_reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = json.load(open("/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/alfworld_metrics.json"))
V1 = data["alfworld_duet_v1"]
V24 = data["alfworld_duet_v24"]
CHORD = data.get("alfworld_chord", [])
LUFFY = data.get("alfworld_luffy", [])

V1_COLOR = "#1f77b4"
V24_COLOR = "#d62728"
OTHER = "#888888"

def xy(rows, key, smooth=0):
    xs = [r["step"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    if smooth > 0:
        sm = []
        for i in range(len(ys)):
            lo = max(0, i-smooth+1)
            sm.append(sum(ys[lo:i+1])/(i+1-lo))
        ys = sm
    return xs, ys


def fmt(ax, ylabel, title=None):
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

# ------- Figure 1: Validation + training success (peak-regression story) --------
val_points_v1 = {50: 0.2750, 100: 0.3250}
val_points_v24 = {50: 0.3350, 100: 0.3050}

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axs[0]
xs, ys = xy(V1, "critic/success_onpolicy/mean", smooth=10)
ax.plot(xs, ys, color=V1_COLOR, label="DUET-v1 (no BC)", lw=2)
xs, ys = xy(V24, "critic/success_onpolicy/mean", smooth=10)
ax.plot(xs, ys, color=V24_COLOR, label="DUET-v24 (+ BC schedule)", lw=2)
ax.axvline(25, color="gray", linestyle=":", alpha=0.7, label="BC decay end (mu hits 0.05)")
fmt(ax, "Training success (10-step rolling)", "Training success rate")
ax.legend()

ax = axs[1]
checkpoints = [50, 100]
width = 12.0
x = [c - width/2 for c in checkpoints]
v1_vals = [val_points_v1[c] for c in checkpoints]
v24_vals = [val_points_v24[c] for c in checkpoints]
ax.bar([c-7 for c in checkpoints], v1_vals, 14, color=V1_COLOR, label="DUET-v1", alpha=0.85)
ax.bar([c+7 for c in checkpoints], v24_vals, 14, color=V24_COLOR, label="DUET-v24", alpha=0.85)
for c in checkpoints:
    v1 = val_points_v1[c]; v24 = val_points_v24[c]
    ax.text(c-7, v1+0.005, f"{v1:.3f}", ha="center", fontsize=9)
    ax.text(c+7, v24+0.005, f"{v24:.3f}", ha="center", fontsize=9)
ax.set_xticks(checkpoints)
ax.set_xlim(30, 120)
ax.set_ylim(0, 0.42)
fmt(ax, "Validation success@N (200 tasks)", "Held-out validation")
ax.legend()
ax.text(50, 0.40, "v24 leads\n+6.0pp", ha="center", color=V24_COLOR, fontsize=9, fontweight="bold")
ax.text(100, 0.40, "v24 lags\n-2.0pp", ha="center", color=V1_COLOR, fontsize=9, fontweight="bold")

plt.suptitle("v24 peaks mid-training then regresses vs v1", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_v24_alfworld_val_curve.png", dpi=130)
plt.close()

# ------- Figure 2: 4-panel training metrics (reward, grad_norm, entropy, kl) ----
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

ax = axs[0,0]
xs, ys = xy(V1, "critic/rewards_onpolicy/mean", smooth=10)
ax.plot(xs, ys, color=V1_COLOR, label="v1", lw=1.8)
xs, ys = xy(V24, "critic/rewards_onpolicy/mean", smooth=10)
ax.plot(xs, ys, color=V24_COLOR, label="v24", lw=1.8)
fmt(ax, "critic/rewards_onpolicy/mean (smoothed)", "On-policy reward")
ax.legend()

ax = axs[0,1]
xs, ys = xy(V1, "actor/grad_norm", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, label="v1", lw=1.8)
xs, ys = xy(V24, "actor/grad_norm", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, label="v24", lw=1.8)
fmt(ax, "actor/grad_norm (5-step smooth)", "Gradient norm")
ax.legend()

ax = axs[1,0]
xs, ys = xy(V1, "actor/entropy_loss", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, label="v1", lw=1.8)
xs, ys = xy(V24, "actor/entropy_loss", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, label="v24", lw=1.8)
fmt(ax, "actor/entropy_loss", "Entropy")
ax.legend()

ax = axs[1,1]
xs, ys = xy(V1, "actor/kl_loss", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, label="v1", lw=1.8)
xs, ys = xy(V24, "actor/kl_loss", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, label="v24", lw=1.8)
fmt(ax, "actor/kl_loss", "KL to reference")
ax.legend()

plt.suptitle("Training dynamics: v24 has strong regularization (lower grad_norm, lower KL)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_v24_alfworld_train_metrics.png", dpi=130)
plt.close()

# ------- Figure 3: BC diagnostics ---------
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axs[0]
xs, ys = xy(V24, "chord/mu")
ax.plot(xs, ys, color=V24_COLOR, lw=2, label="chord/mu (v24)")
ax.axhline(0.05, color="gray", linestyle=":", label="valley")
ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="peak")
ax.axvline(25, color="gray", linestyle=":", alpha=0.5)
fmt(ax, "chord/mu", "BC mixing weight")
ax.legend()

ax = axs[1]
xs, ys = xy(V24, "chord/sft_loss", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="sft_loss (v24)")
xs, ys = xy(V24, "chord/weighted_sft_loss", smooth=5)
ax.plot(xs, ys, color="#ff9800", lw=1.5, linestyle="--", label="weighted_sft_loss")
fmt(ax, "chord/sft_loss", "Teacher token cross-entropy")
ax.legend()

ax = axs[2]
xs, ys = xy(V1, "duet/teacher_gradient_share", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, lw=1.8, label="v1")
xs, ys = xy(V24, "duet/teacher_gradient_share", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="v24")
fmt(ax, "duet/teacher_gradient_share", "Teacher gradient share")
ax.legend()

plt.suptitle("BC activity + curriculum: mu decays as designed but teacher share stays moderate", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_v24_alfworld_bc_diag.png", dpi=130)
plt.close()

# ------- Figure 4: SC interaction ------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

ax = axs[0,0]
xs, ys = xy(V1, "state_channel/progress_onpolicy_mean", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, lw=1.8, label="v1")
xs, ys = xy(V24, "state_channel/progress_onpolicy_mean", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="v24")
fmt(ax, "state_channel/progress_onpolicy_mean", "SC progress (on-policy)")
ax.legend()

ax = axs[0,1]
xs, ys = xy(V1, "state_channel/bonus_vs_reward_ratio", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, lw=1.8, label="v1")
xs, ys = xy(V24, "state_channel/bonus_vs_reward_ratio", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="v24")
fmt(ax, "state_channel/bonus_vs_reward_ratio", "SC bonus magnitude")
ax.legend()

ax = axs[1,0]
xs, ys = xy(V1, "duet/adv_onpolicy_effective_abs_mean", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, lw=1.8, label="v1")
xs, ys = xy(V24, "duet/adv_onpolicy_effective_abs_mean", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="v24")
fmt(ax, "duet/adv_onpolicy_effective_abs_mean", "Advantage signal (on-policy |adv|)")
ax.legend()

ax = axs[1,1]
xs, ys = xy(V1, "response_length/mean", smooth=5)
ax.plot(xs, ys, color=V1_COLOR, lw=1.8, label="v1")
xs, ys = xy(V24, "response_length/mean", smooth=5)
ax.plot(xs, ys, color=V24_COLOR, lw=1.8, label="v24")
fmt(ax, "response_length/mean (tokens)", "Response length")
ax.legend()

plt.suptitle("SC + advantage signal + response length (late-phase drift in v24)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_v24_alfworld_sc_interaction.png", dpi=130)
plt.close()

print("Wrote 4 figures to", OUT_DIR)
