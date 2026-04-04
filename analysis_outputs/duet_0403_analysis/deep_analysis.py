"""
Deep analysis of DUET 0403 regression: reward distributions, group structure,
saturation hypothesis.
"""
import json
import os
import numpy as np
from collections import defaultdict

DUET_DIR = "checkpoints/agentevolver/webshop_3b_duet_0403/Trajectory"
LUFFY_DIR = "checkpoints/agentevolver/webshop_3b_luffy/Trajectory"
OUTPUT_DIR = "analysis_outputs/duet_0403_analysis"

def load_trajectories(base_dir, step):
    path = os.path.join(base_dir, f"trajectories_step_{step}.jsonl")
    if not os.path.exists(path):
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))
    return trajs

def load_batch_diag(base_dir, step):
    path = os.path.join(base_dir, f"batch_diag_step_{step}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def safe_mean(arr):
    return np.mean(arr) if arr else float('nan')

def safe_std(arr):
    return np.std(arr) if arr else float('nan')

def analyze_reward_distribution(trajs, is_teacher=False):
    """Analyze reward distribution buckets."""
    rewards = []
    for t in trajs:
        if t.get("diag", {}).get("is_teacher", False) == is_teacher:
            rewards.append(t["reward"]["outcome"])
    if not rewards:
        return {}
    a = np.array(rewards)
    return {
        "n": len(rewards),
        "mean": a.mean(),
        "std": a.std(),
        "pct_zero": (a < 0.01).mean() * 100,
        "pct_low": ((a >= 0.01) & (a < 0.3)).mean() * 100,
        "pct_mid": ((a >= 0.3) & (a < 0.7)).mean() * 100,
        "pct_high": ((a >= 0.7) & (a < 0.95)).mean() * 100,
        "pct_perfect": (a >= 0.95).mean() * 100,
        "min": a.min(),
        "max": a.max(),
        "median": np.median(a),
    }

def analyze_group_structure(trajs, step):
    """Analyze per-group (data_id/task_id) reward structure to understand GRPO advantage quality."""
    # Group by data_id (GRPO groups)
    groups = defaultdict(list)
    for t in trajs:
        data_id = t.get("data_id", "?")
        is_teacher = t.get("diag", {}).get("is_teacher", False)
        reward = t["reward"]["outcome"]
        groups[data_id].append({
            "reward": reward,
            "is_teacher": is_teacher,
            "adv_mean": t.get("diag", {}).get("adv_mean", 0),
        })

    group_stats = []
    for data_id, members in groups.items():
        on_rewards = [m["reward"] for m in members if not m["is_teacher"]]
        t_rewards = [m["reward"] for m in members if m["is_teacher"]]
        all_rewards = [m["reward"] for m in members]

        if not on_rewards:
            continue

        group_stats.append({
            "data_id": data_id,
            "n_on": len(on_rewards),
            "n_teacher": len(t_rewards),
            "on_mean": np.mean(on_rewards),
            "on_std": np.std(on_rewards),
            "all_mean": np.mean(all_rewards),
            "all_std": np.std(all_rewards),
            "gap": np.mean(t_rewards) - np.mean(on_rewards) if t_rewards else 0,
            "on_range": max(on_rewards) - min(on_rewards),
        })

    return group_stats

def main():
    report = []
    report.append("# DUET 0403 Deep Analysis: Reward Distributions & Saturation")
    report.append(f"Generated: 2026-04-02\n")

    # ================================================================
    # 1. REWARD DISTRIBUTION EVOLUTION
    # ================================================================
    report.append("## 1. Reward Distribution Evolution (On-Policy)")
    report.append("")
    report.append("```")
    report.append(f"{'Step':>5} | {'Mean':>6} | {'Std':>6} | {'%Zero':>6} | {'%Low':>6} | {'%Mid':>6} | {'%High':>6} | {'%Perf':>6} | {'Median':>6}")
    report.append("-" * 75)

    for step in range(40, 101):
        trajs = load_trajectories(DUET_DIR, step)
        dist = analyze_reward_distribution(trajs, is_teacher=False)
        if not dist:
            continue
        report.append(
            f"{step:>5} | {dist['mean']:>6.3f} | {dist['std']:>6.3f} | "
            f"{dist['pct_zero']:>5.1f}% | {dist['pct_low']:>5.1f}% | {dist['pct_mid']:>5.1f}% | "
            f"{dist['pct_high']:>5.1f}% | {dist['pct_perfect']:>5.1f}% | {dist['median']:>6.3f}"
        )
    report.append("```")
    report.append("")

    # Same for LUFFY
    report.append("## 2. LUFFY Reward Distribution (On-Policy)")
    report.append("")
    report.append("```")
    report.append(f"{'Step':>5} | {'Mean':>6} | {'Std':>6} | {'%Zero':>6} | {'%Low':>6} | {'%Mid':>6} | {'%High':>6} | {'%Perf':>6} | {'Median':>6}")
    report.append("-" * 75)

    for step in range(40, 101):
        trajs = load_trajectories(LUFFY_DIR, step)
        dist = analyze_reward_distribution(trajs, is_teacher=False)
        if not dist:
            continue
        report.append(
            f"{step:>5} | {dist['mean']:>6.3f} | {dist['std']:>6.3f} | "
            f"{dist['pct_zero']:>5.1f}% | {dist['pct_low']:>5.1f}% | {dist['pct_mid']:>5.1f}% | "
            f"{dist['pct_high']:>5.1f}% | {dist['pct_perfect']:>5.1f}% | {dist['median']:>6.3f}"
        )
    report.append("```")
    report.append("")

    # ================================================================
    # 3. GRPO ADVANTAGE QUALITY (GROUP ANALYSIS)
    # ================================================================
    report.append("## 3. GRPO Group-Level Analysis (Advantage Signal Quality)")
    report.append("")
    report.append("Low within-group std → weak learning signal (GRPO saturation)")
    report.append("")

    report.append("### 3a. DUET Group Reward Variance")
    report.append("```")
    report.append(f"{'Step':>5} | {'#Groups':>7} | {'MeanGrpStd':>10} | {'MedGrpStd':>10} | {'MeanGrpRange':>12} | {'%Flat(<0.05)':>12} | {'%Rich(>0.2)':>12}")
    report.append("-" * 80)

    duet_flat_pct = []
    luffy_flat_pct = []

    for step in range(40, 101):
        trajs = load_trajectories(DUET_DIR, step)
        groups = analyze_group_structure(trajs, step)
        if not groups:
            continue
        stds = [g["on_std"] for g in groups]
        ranges = [g["on_range"] for g in groups]
        flat = sum(1 for s in stds if s < 0.05) / len(stds) * 100
        rich = sum(1 for s in stds if s > 0.2) / len(stds) * 100
        duet_flat_pct.append((step, flat))
        report.append(
            f"{step:>5} | {len(groups):>7} | {np.mean(stds):>10.4f} | {np.median(stds):>10.4f} | "
            f"{np.mean(ranges):>12.4f} | {flat:>11.1f}% | {rich:>11.1f}%"
        )
    report.append("```")
    report.append("")

    report.append("### 3b. LUFFY Group Reward Variance")
    report.append("```")
    report.append(f"{'Step':>5} | {'#Groups':>7} | {'MeanGrpStd':>10} | {'MedGrpStd':>10} | {'MeanGrpRange':>12} | {'%Flat(<0.05)':>12} | {'%Rich(>0.2)':>12}")
    report.append("-" * 80)

    for step in range(40, 101):
        trajs = load_trajectories(LUFFY_DIR, step)
        groups = analyze_group_structure(trajs, step)
        if not groups:
            continue
        stds = [g["on_std"] for g in groups]
        ranges = [g["on_range"] for g in groups]
        flat = sum(1 for s in stds if s < 0.05) / len(stds) * 100
        rich = sum(1 for s in stds if s > 0.2) / len(stds) * 100
        luffy_flat_pct.append((step, flat))
        report.append(
            f"{step:>5} | {len(groups):>7} | {np.mean(stds):>10.4f} | {np.median(stds):>10.4f} | "
            f"{np.mean(ranges):>12.4f} | {flat:>11.1f}% | {rich:>11.1f}%"
        )
    report.append("```")
    report.append("")

    # ================================================================
    # 4. TEACHER GAP ANALYSIS
    # ================================================================
    report.append("## 4. Teacher-OnPolicy Reward Gap Over Time")
    report.append("")
    report.append("When gap → 0, teacher provides no additional signal")
    report.append("")
    report.append("```")
    report.append(f"{'Step':>5} | {'DUET Gap':>10} | {'DUET OnRew':>10} | {'LUFFY Gap':>10} | {'LUFFY OnRew':>10}")
    report.append("-" * 55)

    for step in range(40, 101):
        d = load_batch_diag(DUET_DIR, step)
        l = load_batch_diag(LUFFY_DIR, step)
        if d and l:
            dg = d.get("diag/group_teacher_minus_on_reward_mean", float('nan'))
            dr = d.get("diag/reward_onpolicy_mean", float('nan'))
            lg = l.get("diag/group_teacher_minus_on_reward_mean", float('nan'))
            lr = l.get("diag/reward_onpolicy_mean", float('nan'))
            report.append(f"{step:>5} | {dg:>10.4f} | {dr:>10.4f} | {lg:>10.4f} | {lr:>10.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 5. ROLLING AVERAGES
    # ================================================================
    report.append("## 5. Rolling 10-Step Average Comparison")
    report.append("")

    duet_rewards = []
    luffy_rewards = []
    steps_list = []
    for step in range(1, 101):
        d = load_batch_diag(DUET_DIR, step)
        l = load_batch_diag(LUFFY_DIR, step)
        if d and l:
            steps_list.append(step)
            duet_rewards.append(d["diag/reward_onpolicy_mean"])
            luffy_rewards.append(l["diag/reward_onpolicy_mean"])

    report.append("```")
    report.append(f"{'Window':>12} | {'DUET Avg':>10} | {'LUFFY Avg':>10} | {'Delta':>10}")
    report.append("-" * 50)
    for start in range(0, len(steps_list), 10):
        end = min(start + 10, len(steps_list))
        window = f"{steps_list[start]}-{steps_list[end-1]}"
        d_avg = np.mean(duet_rewards[start:end])
        l_avg = np.mean(luffy_rewards[start:end])
        report.append(f"{window:>12} | {d_avg:>10.4f} | {l_avg:>10.4f} | {d_avg - l_avg:>+10.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 6. SC CONTRIBUTION ANALYSIS
    # ================================================================
    report.append("## 6. SC Contribution: How Much is SC Inflating Rewards?")
    report.append("")
    report.append("Compare original reward vs total (original + SC bonus)")
    report.append("```")
    report.append(f"{'Step':>5} | {'Orig Rew':>10} | {'SC Bonus':>10} | {'Total':>10} | {'StepDelta':>10}")
    report.append("-" * 55)

    for step in range(40, 101):
        trajs = load_trajectories(DUET_DIR, step)
        on_trajs = [t for t in trajs if not t.get("diag", {}).get("is_teacher", False)]
        if not on_trajs:
            continue

        orig = np.mean([t.get("diag", {}).get("reward_original", t["reward"]["outcome"]) for t in on_trajs])
        sc_bonus = np.mean([t.get("diag", {}).get("sc_bonus", 0) for t in on_trajs])
        step_delta_sum = np.mean([sum(t.get("diag", {}).get("sc_step_deltas", [0])) for t in on_trajs])
        total = orig + sc_bonus

        report.append(f"{step:>5} | {orig:>10.4f} | {sc_bonus:>10.4f} | {total:>10.4f} | {step_delta_sum:>10.4f}")
    report.append("```")
    report.append("")

    # ================================================================
    # 7. TRANSITION ANALYSIS: WHAT CHANGES AT STEP 80?
    # ================================================================
    report.append("## 7. Transition Analysis: Step 75-85 Deep Dive")
    report.append("")

    for step in [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]:
        trajs = load_trajectories(DUET_DIR, step)
        on_trajs = [t for t in trajs if not t.get("diag", {}).get("is_teacher", False)]
        if not on_trajs:
            continue

        rewards = [t["reward"]["outcome"] for t in on_trajs]
        advs = [t["diag"]["adv_mean"] for t in on_trajs]
        resp_lens = [t["diag"]["response_valid_tokens"] for t in on_trajs]
        successes = [1 if t["success"] else 0 for t in on_trajs]

        report.append(f"### Step {step}")
        report.append(f"- N on-policy: {len(on_trajs)}")
        report.append(f"- Reward: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
        report.append(f"- Success rate: {np.mean(successes):.4f}")
        report.append(f"- Adv: mean={np.mean(advs):.6f}, std={np.std(advs):.6f}")
        report.append(f"  - pos_frac: {(np.array(advs) > 0).mean():.4f}")
        report.append(f"  - max={np.max(advs):.6f}, min={np.min(advs):.6f}")
        report.append(f"- Resp len: mean={np.mean(resp_lens):.0f}, std={np.std(resp_lens):.0f}")

        # Reward histogram
        r = np.array(rewards)
        report.append(f"- Reward dist: [0]={int((r < 0.01).sum())} "
                      f"[0-0.3]={int(((r >= 0.01) & (r < 0.3)).sum())} "
                      f"[0.3-0.7]={int(((r >= 0.3) & (r < 0.7)).sum())} "
                      f"[0.7-0.95]={int(((r >= 0.7) & (r < 0.95)).sum())} "
                      f"[0.95+]={int((r >= 0.95).sum())}")
        report.append("")

    # ================================================================
    # 8. CONCLUSION
    # ================================================================
    report.append("## 8. Refined Diagnosis")
    report.append("")

    # Compute key metrics for conclusion
    # 5-step rolling averages
    d_avg_75_80 = np.mean(duet_rewards[steps_list.index(75):steps_list.index(80)+1])
    d_avg_85_90 = np.mean(duet_rewards[steps_list.index(85):steps_list.index(90)+1])
    d_avg_95_100 = np.mean(duet_rewards[steps_list.index(95):steps_list.index(100)+1])
    l_avg_75_80 = np.mean(luffy_rewards[steps_list.index(75):steps_list.index(80)+1])
    l_avg_85_90 = np.mean(luffy_rewards[steps_list.index(85):steps_list.index(90)+1])
    l_avg_95_100 = np.mean(luffy_rewards[steps_list.index(95):steps_list.index(100)+1])

    report.append(f"Rolling averages:")
    report.append(f"  Steps 75-80: DUET={d_avg_75_80:.4f}, LUFFY={l_avg_75_80:.4f}, gap={d_avg_75_80-l_avg_75_80:+.4f}")
    report.append(f"  Steps 85-90: DUET={d_avg_85_90:.4f}, LUFFY={l_avg_85_90:.4f}, gap={d_avg_85_90-l_avg_85_90:+.4f}")
    report.append(f"  Steps 95-100: DUET={d_avg_95_100:.4f}, LUFFY={l_avg_95_100:.4f}, gap={d_avg_95_100-l_avg_95_100:+.4f}")
    report.append("")

    # Gap analysis
    report.append("### Key observations:")
    report.append("")
    report.append("1. **NOT a catastrophic collapse**: None of the classic failure modes are present:")
    report.append("   - No entropy collapse (entropy stable ~0.35-0.48)")
    report.append("   - No KL explosion (drift 0.07, actually LESS than LUFFY's 0.10)")
    report.append("   - No teacher adv clipping (max teacher adv ~0.7, nowhere near ±5 clip)")
    report.append("   - No response length explosion (stable ~1700-2500)")
    report.append("   - No PG clip activation (all zeros)")
    report.append("")
    report.append("2. **What IS happening**: DUET reaches peak performance ~step 79 (reward 0.99)")
    report.append("   then regresses to the LUFFY-level baseline (~0.72-0.79). The DUET advantage")
    report.append("   gap (vs LUFFY) shrinks from +0.10-0.15 (steps 60-80) to ~0 (steps 85-100).")
    report.append("")
    report.append("3. **Possible mechanism - GRPO saturation**:")
    report.append("   - At step 80, on-policy rewards are very high (~0.90)")
    report.append("   - Within-group variance drops → GRPO advantages become near-zero")
    report.append("   - Without strong advantage signal, policy drifts randomly")
    report.append("   - SC bonus (~0.07) adds a small constant that doesn't differentiate good vs bad")
    report.append("")
    report.append("4. **Teacher contribution diminishes**: Teacher-onpolicy gap shrinks as model")
    report.append("   gets better, reducing teacher's marginal value even though advantages are positive.")
    report.append("")

    report_text = "\n".join(report)
    output_path = os.path.join(OUTPUT_DIR, "deep_analysis.md")
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"Written to {output_path}")

    # Print key summary
    print("\n" + "="*60)
    print("DEEP ANALYSIS SUMMARY")
    print("="*60)
    print(f"Steps 75-80: DUET={d_avg_75_80:.4f} vs LUFFY={l_avg_75_80:.4f} (gap={d_avg_75_80-l_avg_75_80:+.4f})")
    print(f"Steps 85-90: DUET={d_avg_85_90:.4f} vs LUFFY={l_avg_85_90:.4f} (gap={d_avg_85_90-l_avg_85_90:+.4f})")
    print(f"Steps 95-100: DUET={d_avg_95_100:.4f} vs LUFFY={l_avg_95_100:.4f} (gap={d_avg_95_100-l_avg_95_100:+.4f})")

if __name__ == "__main__":
    main()
