import json
import numpy as np

with open("/data/code/exp/EvolAnalsis/analysis_outputs/duet_0401_diagnosis/wandb_data.json") as f:
    data = json.load(f)

if "_available_keys_0401" in data:
    del data["_available_keys_0401"]

def get_at_steps(run, metric, step_points):
    """Get metric values at approximately the given steps"""
    m = data[run].get(metric, {})
    steps = m.get("steps", [])
    values = m.get("values", [])
    if not steps:
        return {s: None for s in step_points}
    result = {}
    for sp in step_points:
        # Find closest step
        closest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - sp))
        if abs(steps[closest_idx] - sp) <= 3:
            result[sp] = values[closest_idx]
        else:
            result[sp] = None
    return result

print("="*90)
print("DETAILED TRAINING PROGRESSION")
print("="*90)

checkpoints = [1, 10, 25, 50, 75, 99]
metrics_to_track = [
    ("critic/success_onpolicy/mean", "Success Rate"),
    ("critic/rewards_onpolicy/mean", "On-Policy Reward"),
    ("duet/teacher_gradient_share", "Teacher Grad Share"),
    ("diag/adv_teacher_sample_mean", "Teacher Adv Mean"),
    ("diag/adv_onpolicy_sample_mean", "On-Policy Adv Mean"),
    ("state_channel/bonus_total_mean", "SC Bonus"),
    ("state_channel/progress_onpolicy_mean", "SC Progress (onpol)"),
    ("actor/kl_loss", "KL Loss"),
]

for m_key, m_label in metrics_to_track:
    print(f"\n--- {m_label} ({m_key}) ---")
    print(f"{'Step':>6}", end="")
    for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
        print(f"  {run:>14}", end="")
    print()
    
    for sp in checkpoints:
        print(f"{sp:>6}", end="")
        for run in ["DUET_0401", "DUET_orig", "LUFFY"]:
            val = get_at_steps(run, m_key, [sp])[sp]
            if val is not None:
                if abs(val) > 100:
                    print(f"  {val:>14.1f}", end="")
                else:
                    print(f"  {val:>14.4f}", end="")
            else:
                print(f"  {'N/A':>14}", end="")
        print()

# Analyze the reward trajectories
print("\n" + "="*90)
print("CRITICAL ANALYSIS")
print("="*90)

# 1. Check if 0401 starts slower
d0401_success = data["DUET_0401"]["critic/success_onpolicy/mean"]["values"]
dorig_success = data["DUET_orig"]["critic/success_onpolicy/mean"]["values"]
dluffy_success = data["LUFFY"]["critic/success_onpolicy/mean"]["values"]

# Average success rate in first 20 steps
avg_first20_0401 = np.mean(d0401_success[:20]) if len(d0401_success) >= 20 else 0
avg_first20_orig = np.mean(dorig_success[:20]) if len(dorig_success) >= 20 else 0
avg_first20_luffy = np.mean(dluffy_success[:20]) if len(dluffy_success) >= 20 else 0

print(f"\n1. Average success rate in first 20 steps:")
print(f"   DUET 0401: {avg_first20_0401:.4f}")
print(f"   DUET orig: {avg_first20_orig:.4f}")
print(f"   LUFFY:     {avg_first20_luffy:.4f}")

# Average in last 20 steps
avg_last20_0401 = np.mean(d0401_success[-20:]) if len(d0401_success) >= 20 else 0
avg_last20_orig = np.mean(dorig_success[-20:]) if len(dorig_success) >= 20 else 0
avg_last20_luffy = np.mean(dluffy_success[-20:]) if len(dluffy_success) >= 20 else 0

print(f"\n2. Average success rate in last 20 steps:")
print(f"   DUET 0401: {avg_last20_0401:.4f}")
print(f"   DUET orig: {avg_last20_orig:.4f}")
print(f"   LUFFY:     {avg_last20_luffy:.4f}")

# 3. Teacher gradient share trajectory
tgs_0401 = data["DUET_0401"]["duet/teacher_gradient_share"]["values"]
tgs_orig = data["DUET_orig"]["duet/teacher_gradient_share"]["values"]
tgs_luffy = data["LUFFY"]["duet/teacher_gradient_share"]["values"]

print(f"\n3. Teacher Gradient Share trajectory:")
print(f"   DUET 0401: {tgs_0401[0]:.3f} → {tgs_0401[24]:.3f} → {tgs_0401[49]:.3f} → {tgs_0401[-1]:.3f}")
print(f"   DUET orig: {tgs_orig[0]:.3f} → {tgs_orig[24]:.3f} → {tgs_orig[49]:.3f} → {tgs_orig[-1]:.3f}")
print(f"   LUFFY:     {tgs_luffy[0]:.3f} → {tgs_luffy[24]:.3f} → {tgs_luffy[49]:.3f} → {tgs_luffy[-1]:.3f}")

# 4. On-policy advantages
adv_0401 = data["DUET_0401"]["diag/adv_onpolicy_sample_mean"]["values"]
adv_luffy = data["LUFFY"]["diag/adv_onpolicy_sample_mean"]["values"]

print(f"\n4. On-Policy Advantage Mean (first 10 steps):")
print(f"   DUET 0401: {[f'{x:.4f}' for x in adv_0401[:10]]}")
print(f"   LUFFY:     {[f'{x:.4f}' for x in adv_luffy[:10]]}")

# 5. SC bonus analysis
sc_bonus = data["DUET_0401"]["state_channel/bonus_total_mean"]["values"]
rewards = data["DUET_0401"]["critic/rewards_onpolicy/mean"]["values"]
print(f"\n5. SC Bonus vs On-Policy Reward (DUET 0401):")
for sp in [0, 9, 24, 49, 74, 98]:
    if sp < len(sc_bonus) and sp < len(rewards):
        ratio = sc_bonus[sp] / max(rewards[sp], 0.001)
        print(f"   Step {sp+1}: bonus={sc_bonus[sp]:.4f}, reward={rewards[sp]:.4f}, ratio={ratio:.4f}")

# 6. Check if teacher advantages in orig vs 0401 diverge
ta_0401 = data["DUET_0401"]["diag/adv_teacher_sample_mean"]["values"]
ta_orig = data["DUET_orig"]["diag/adv_teacher_sample_mean"]["values"]

print(f"\n6. Teacher Advantage divergence (DUET orig):")
for sp in [0, 9, 24, 49, 74, -1]:
    if sp < len(ta_orig):
        print(f"   Step {sp+1 if sp >= 0 else 'last'}: orig={ta_orig[sp]:.2f}, 0401={ta_0401[sp]:.4f}")

