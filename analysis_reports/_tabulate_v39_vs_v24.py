#!/usr/bin/env python3
"""Build comparison tables across 5 WebShop runs."""
import os, json, csv

PARSED = "/data/home/qisheng/EvolAnalsis/analysis_reports/_parsed/v39_vs_v24"
RUNS = ["v24", "v39_postfix", "v39b_postfix", "swA_04_peak05", "swA_11_pk05_v10"]
SNAP_STEPS = [1, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100]

data = {}
for run in RUNS:
    with open(os.path.join(PARSED, f"{run}.json")) as f:
        data[run] = {int(k): v for k, v in json.load(f).items()}


def at(run, step, key, default=None):
    """Return metric value at exact step, or nearest <= step within 2 steps."""
    d = data[run]
    if step in d and key in d[step]:
        return d[step][key]
    for offset in (1, 2):
        for cand in (step - offset, step + offset):
            if cand in d and key in d[cand]:
                return d[cand][key]
    return default


def fmt(v, prec=3):
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.{prec}f}"


def make_table(metric, label, prec=3):
    print(f"\n### {label}  (`{metric}`)\n")
    head = "| step | " + " | ".join(RUNS) + " |"
    sep = "|" + "---|" * (len(RUNS) + 1)
    print(head)
    print(sep)
    for s in SNAP_STEPS:
        row = [str(s)]
        for run in RUNS:
            row.append(fmt(at(run, s, metric), prec))
        print("| " + " | ".join(row) + " |")


# Effective BC weight: for v24, "chord/mu" is the applied weight; for adaptive,
# it's "chord/mu_adaptive_gated" (or "chord/mu" since the v39 family logs the
# gated value as chord/mu too in some builds). Inspect to confirm.
print("# v39 vs v24 WebShop 1.5B — Detailed Step-Step Tables")
print()
print("First, confirm what `chord/mu` represents in each run by checking step 1, 5, 25 raw vs gated.\n")
print("| run | s=1 mu | s=1 gated | s=10 mu | s=10 gated | s=25 mu | s=25 gated | s=50 mu | s=50 gated |")
print("|---|---|---|---|---|---|---|---|---|")
for run in RUNS:
    cells = [run]
    for s in (1, 10, 25, 50):
        cells.append(fmt(at(run, s, "chord/mu"), 4))
        cells.append(fmt(at(run, s, "chord/mu_adaptive_gated"), 4))
    print("| " + " | ".join(cells) + " |")

# 1. Effective BC weight (mu)
make_table("chord/mu", "BC weight `chord/mu`", prec=4)
make_table("chord/mu_adaptive_gated", "Adaptive gated mu (v39 only)", prec=4)

# 2. disc_acc trajectories
make_table("dr3/disc_acc", "Discriminator accuracy `dr3/disc_acc`", prec=3)
make_table("chord/disc_acc_ema", "EMA(disc_acc) `chord/disc_acc_ema`", prec=3)

# 3. SFT loss / weighted SFT loss
make_table("chord/sft_loss", "BC loss `chord/sft_loss`", prec=3)
make_table("chord/weighted_sft_loss", "Weighted BC loss `chord/weighted_sft_loss`", prec=4)
make_table("chord/n_expert_tokens", "Expert tokens `chord/n_expert_tokens`", prec=0)

# 4. Rewards / success
make_table("critic/rewards/mean", "Train mean reward `critic/rewards/mean`", prec=3)
make_table("critic/rewards_onpolicy/mean", "On-policy reward `critic/rewards_onpolicy/mean`", prec=3)
make_table("critic/success_onpolicy/mean", "On-policy train success `critic/success_onpolicy/mean`", prec=3)
make_table("critic/rewards_teacher/mean", "Teacher reward `critic/rewards_teacher/mean`", prec=3)

# 5. Validation
make_table("val-summary/webshop/reward_mean_all", "Val reward (val@N)", prec=3)
make_table("val-summary/webshop/success_rate_mean_all", "Val success rate (val@N)", prec=3)

# 6. Failure mode
make_table("response_length/mean", "Response length mean", prec=0)
make_table("response_length/clip_ratio", "Response length clip ratio", prec=3)
make_table("actor/entropy_loss", "Actor entropy", prec=3)
make_table("exp_replay/entropy_llm_onpolicy_mean", "On-policy LLM entropy (exp_replay)", prec=3)

# 7. State channel
make_table("state_channel/progress_onpolicy_mean", "SC on-policy progress", prec=3)
make_table("state_channel/progress_mean", "SC progress mean", prec=3)
make_table("state_channel/bonus_total_mean", "SC bonus mean", prec=4)
make_table("state_channel/bonus_vs_reward_ratio", "SC bonus/reward ratio", prec=3)

# Teacher gradient share (DR3)
make_table("dr3/teacher_gradient_share", "DR3 teacher_gradient_share", prec=3)
make_table("duet/teacher_gradient_share", "DUET teacher_gradient_share", prec=3)
make_table("dr3/w_mean", "DR3 w_hat mean", prec=3)
make_table("dr3/ess_off_window", "DR3 ess_off_window", prec=2)

# Compute integrals (AUC) for chord/mu and chord/mu_adaptive_gated
def auc(run, key, smin=1, smax=100):
    d = data[run]
    pts = sorted((s, v[key]) for s, v in d.items() if key in v and smin <= s <= smax)
    if len(pts) < 2:
        return None
    total = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


print("\n### AUC of BC weight over 100 steps\n")
print("| run | AUC(chord/mu) | AUC(chord/mu_adaptive_gated) |")
print("|---|---|---|")
for run in RUNS:
    print(f"| {run} | {fmt(auc(run, 'chord/mu'))} | {fmt(auc(run, 'chord/mu_adaptive_gated'))} |")

# Step at which disc_acc and disc_acc_ema first cross 0.95
print("\n### First step disc_acc / EMA crosses 0.95\n")
print("| run | disc_acc>=0.95 first step | EMA>=0.95 first step | disc_acc step1 | EMA step1 |")
print("|---|---|---|---|---|")
for run in RUNS:
    d = data[run]
    first_da = next((s for s in sorted(d.keys()) if d[s].get("dr3/disc_acc", 0) >= 0.95), None)
    first_ema = next((s for s in sorted(d.keys()) if d[s].get("chord/disc_acc_ema", 0) >= 0.95), None)
    print(f"| {run} | {first_da} | {first_ema} | {fmt(at(run, 1, 'dr3/disc_acc'))} | {fmt(at(run, 1, 'chord/disc_acc_ema'))} |")
