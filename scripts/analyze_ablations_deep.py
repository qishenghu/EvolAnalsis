#!/usr/bin/env python3
"""
Deep ablation analysis for WebShop 1.5B DUET variants.
Extracts batch_diag + trajectory-level metrics for all variants across training.
"""

import json
import os
import sys
from collections import defaultdict
import numpy as np

BASE_DIR = "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver"
EXP_DIR = "/data/home/qisheng/EvolAnalsis/experiments/webshop"

VARIANTS = {
    "v1": "webshop_qwen1.5b_duet",
    "v2": "webshop_qwen1.5b_duet_v2",
    "v3": "webshop_qwen1.5b_duet_v3",
    "v4": "webshop_qwen1.5b_duet_v4",
    "v5": "webshop_qwen1.5b_duet_v5",
    "v6": "webshop_qwen1.5b_duet_v6",
    "v7": "webshop_qwen1.5b_duet_v7",
    "v8": "webshop_qwen1.5b_duet_v8",
    "v9": "webshop_qwen1.5b_duet_v9",
    "chord": "webshop_qwen1.5b_chord",
    "luffy": "webshop_qwen1.5b_luffy",
    "onpolicy": "webshop_qwen1.5b_onpolicy",
}

# Batch-diag metrics to extract
BATCH_METRICS = [
    "diag/reward_onpolicy_mean",
    "diag/onpolicy_adv_pos_ratio",
    "diag/entropy_onpolicy_token_mean",
    "diag/group_teacher_minus_on_reward_mean",
    "diag/llm_token_ratio_in_response",
    "diag/adv_onpolicy_sample_mean",
    "diag/adv_teacher_sample_mean",
    "diag/adv_teacher_token_mean",
    "diag/entropy_onpolicy_token_std",
    "diag/group_non_teacher_reward_mean",
    "diag/group_all_reward_mean",
    "diag/teacher_token_ratio",
]

STEPS = list(range(1, 101))

def load_batch_diag(variant_dir, step):
    """Load batch_diag_step_N.json"""
    path = os.path.join(BASE_DIR, variant_dir, "Trajectory", f"batch_diag_step_{step}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_trajectory_stats(variant_dir, step):
    """Load trajectories_step_N.jsonl and compute aggregate SC/DR3 stats"""
    path = os.path.join(BASE_DIR, variant_dir, "Trajectory", f"trajectories_step_{step}.jsonl")
    if not os.path.exists(path):
        return None

    stats = {
        "n_total": 0,
        "n_teacher": 0,
        "n_onpolicy": 0,
        "sc_progress_onpolicy": [],
        "sc_bonus_onpolicy": [],
        "sc_coverage_onpolicy": [],
        "sc_matched_states_onpolicy": [],
        "reward_original_onpolicy": [],
        "reward_original_teacher": [],
        "reward_sum_onpolicy": [],
        "reward_sum_teacher": [],
        "step_delta_sum_onpolicy": [],
        "offpolicy_ratio_teacher": [],
        "offpolicy_ratio_onpolicy": [],
        "adv_mean_teacher": [],
        "adv_mean_onpolicy": [],
        "old_log_prob_mean_onpolicy": [],
        "old_log_prob_mean_teacher": [],
        "response_valid_tokens_onpolicy": [],
        "response_valid_tokens_teacher": [],
        "teacher_old_logp_mean": [],
        "success_onpolicy": [],
        "success_teacher": [],
    }

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            diag = d.get("diag", {})
            reward = d.get("reward", {})
            is_teacher = diag.get("is_teacher", False)

            stats["n_total"] += 1

            if is_teacher:
                stats["n_teacher"] += 1
                stats["reward_original_teacher"].append(diag.get("reward_original", 0))
                stats["reward_sum_teacher"].append(diag.get("reward_sum", 0))
                stats["offpolicy_ratio_teacher"].append(diag.get("offpolicy_ratio", 0))
                stats["adv_mean_teacher"].append(diag.get("adv_mean", 0))
                stats["old_log_prob_mean_teacher"].append(diag.get("old_log_prob_mean", 0))
                stats["response_valid_tokens_teacher"].append(diag.get("response_valid_tokens", 0))
                stats["teacher_old_logp_mean"].append(diag.get("teacher_old_logp_mean", 0))
                stats["success_teacher"].append(1 if reward.get("outcome", 0) > 0.5 else 0)
            else:
                stats["n_onpolicy"] += 1
                stats["sc_progress_onpolicy"].append(diag.get("sc_progress", 0))
                stats["sc_bonus_onpolicy"].append(diag.get("sc_bonus", 0))
                stats["sc_coverage_onpolicy"].append(diag.get("sc_coverage", 0))
                stats["sc_matched_states_onpolicy"].append(diag.get("sc_matched_states", 0))
                stats["reward_original_onpolicy"].append(diag.get("reward_original", 0))
                stats["reward_sum_onpolicy"].append(diag.get("reward_sum", 0))
                stats["offpolicy_ratio_onpolicy"].append(diag.get("offpolicy_ratio", 0))
                stats["adv_mean_onpolicy"].append(diag.get("adv_mean", 0))
                stats["old_log_prob_mean_onpolicy"].append(diag.get("old_log_prob_mean", 0))
                stats["response_valid_tokens_onpolicy"].append(diag.get("response_valid_tokens", 0))
                stats["success_onpolicy"].append(1 if reward.get("outcome", 0) > 0.5 else 0)

                # Step-level delta sum
                rc = diag.get("reward_components", {})
                if isinstance(rc, dict):
                    stats["step_delta_sum_onpolicy"].append(rc.get("step_delta_sum", 0))

    return stats

def safe_mean(lst):
    if not lst:
        return float('nan')
    return np.mean(lst)

def safe_std(lst):
    if not lst:
        return float('nan')
    return np.std(lst)

def compute_validation_scores(variant_dir):
    """Load validation scores at step 50 and 100"""
    scores = {}
    for step in [50, 100]:
        path = os.path.join(EXP_DIR, variant_dir, "validation_log", f"{step}.jsonl")
        if not os.path.exists(path):
            scores[step] = float('nan')
            continue
        rewards_list = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    reward = d.get("reward", 0)
                    if isinstance(reward, dict):
                        reward = reward.get("outcome", reward.get("success_rate", 0))
                    rewards_list.append(float(reward) if reward is not None else 0.0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        scores[step] = np.mean(rewards_list) if rewards_list else float('nan')
    return scores


def main():
    print("=" * 120)
    print("DEEP ABLATION ANALYSIS: WebShop 1.5B DUET Variants")
    print("=" * 120)

    # ====================================================================
    # PART 1: Validation scores
    # ====================================================================
    print("\n\n>>> PART 1: VALIDATION SCORES")
    print("-" * 80)
    print(f"{'Variant':<12} {'Val@50':>10} {'Val@100':>10} {'Val@50 (N)':>12} {'Val@100 (N)':>12}")
    print("-" * 80)

    val_scores = {}
    for name, dir_name in VARIANTS.items():
        scores = compute_validation_scores(dir_name)
        val_scores[name] = scores

        # Count episodes
        for step in [50, 100]:
            path = os.path.join(EXP_DIR, dir_name, "validation_log", f"{step}.jsonl")
            n = 0
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            n += 1

        n50 = n100 = 0
        for step, n_key in [(50, 'n50'), (100, 'n100')]:
            path = os.path.join(EXP_DIR, dir_name, "validation_log", f"{step}.jsonl")
            count = 0
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            count += 1
            if step == 50:
                n50 = count
            else:
                n100 = count

        print(f"{name:<12} {scores.get(50, float('nan')):>10.4f} {scores.get(100, float('nan')):>10.4f} {n50:>12d} {n100:>12d}")

    # ====================================================================
    # PART 2: Batch-diag metrics across training (sampled steps)
    # ====================================================================
    print("\n\n>>> PART 2: BATCH-DIAG METRICS EVOLUTION (every 10 steps)")

    focus_variants = ["v1", "v2", "v8", "v9", "chord", "luffy", "onpolicy"]
    sample_steps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for metric in BATCH_METRICS:
        print(f"\n--- {metric} ---")
        header = f"{'Step':>6}"
        for name in focus_variants:
            header += f" {name:>12}"
        print(header)

        for step in sample_steps:
            row = f"{step:>6}"
            for name in focus_variants:
                data = load_batch_diag(VARIANTS[name], step)
                if data and metric in data:
                    val = data[metric]
                    row += f" {val:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    # ====================================================================
    # PART 3: Trajectory-level SC + DR3 diagnostics (key steps)
    # ====================================================================
    print("\n\n>>> PART 3: TRAJECTORY-LEVEL SC/DR3 DIAGNOSTICS")

    traj_variants = ["v1", "v8", "v2", "v9", "chord", "luffy"]
    traj_steps = [1, 10, 25, 50, 75, 100]

    traj_metrics = [
        ("SC Progress (on-policy mean)", "sc_progress_onpolicy"),
        ("SC Bonus (on-policy mean)", "sc_bonus_onpolicy"),
        ("SC Coverage (on-policy mean)", "sc_coverage_onpolicy"),
        ("SC Matched States (on-policy mean)", "sc_matched_states_onpolicy"),
        ("Step Delta Sum (on-policy mean)", "step_delta_sum_onpolicy"),
        ("Reward Original (on-policy mean)", "reward_original_onpolicy"),
        ("Reward Sum (on-policy mean)", "reward_sum_onpolicy"),
        ("Success Rate (on-policy)", "success_onpolicy"),
        ("On-policy Adv Mean", "adv_mean_onpolicy"),
        ("Teacher Adv Mean", "adv_mean_teacher"),
        ("Offpolicy Ratio (teacher mean)", "offpolicy_ratio_teacher"),
        ("Offpolicy Ratio (on-policy mean)", "offpolicy_ratio_onpolicy"),
        ("Log Prob (on-policy mean)", "old_log_prob_mean_onpolicy"),
        ("Response Tokens (on-policy mean)", "response_valid_tokens_onpolicy"),
        ("Response Tokens (teacher mean)", "response_valid_tokens_teacher"),
    ]

    for metric_label, metric_key in traj_metrics:
        print(f"\n--- {metric_label} ---")
        header = f"{'Step':>6}"
        for name in traj_variants:
            header += f" {name:>12}"
        print(header)

        for step in traj_steps:
            row = f"{step:>6}"
            for name in traj_variants:
                stats = load_trajectory_stats(VARIANTS[name], step)
                if stats and metric_key in stats and stats[metric_key]:
                    val = safe_mean(stats[metric_key])
                    row += f" {val:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

    # ====================================================================
    # PART 4: Head-to-head v1 vs v8 deep comparison
    # ====================================================================
    print("\n\n>>> PART 4: HEAD-TO-HEAD v1 vs v8 (step-delta ON vs OFF)")
    print("=" * 100)

    comparison_steps = [10, 25, 50, 75, 100]

    for step in comparison_steps:
        print(f"\n--- Step {step} ---")
        v1_diag = load_batch_diag(VARIANTS["v1"], step)
        v8_diag = load_batch_diag(VARIANTS["v8"], step)
        v1_traj = load_trajectory_stats(VARIANTS["v1"], step)
        v8_traj = load_trajectory_stats(VARIANTS["v8"], step)

        print(f"{'Metric':<50} {'v1 (step ON)':>15} {'v8 (step OFF)':>15} {'Delta':>12}")
        print("-" * 92)

        for metric in BATCH_METRICS:
            v1_val = v1_diag.get(metric, float('nan')) if v1_diag else float('nan')
            v8_val = v8_diag.get(metric, float('nan')) if v8_diag else float('nan')
            delta = v8_val - v1_val if not (np.isnan(v1_val) or np.isnan(v8_val)) else float('nan')
            print(f"  {metric:<48} {v1_val:>15.4f} {v8_val:>15.4f} {delta:>+12.4f}")

        # Trajectory-level comparisons
        if v1_traj and v8_traj:
            for label, key in [
                ("SC Progress (on-policy)", "sc_progress_onpolicy"),
                ("SC Bonus (on-policy)", "sc_bonus_onpolicy"),
                ("Step Delta Sum (on-policy)", "step_delta_sum_onpolicy"),
                ("Reward Sum (on-policy)", "reward_sum_onpolicy"),
                ("Success Rate (on-policy)", "success_onpolicy"),
                ("Offpolicy Ratio (teacher)", "offpolicy_ratio_teacher"),
            ]:
                v1_val = safe_mean(v1_traj[key]) if v1_traj[key] else float('nan')
                v8_val = safe_mean(v8_traj[key]) if v8_traj[key] else float('nan')
                delta = v8_val - v1_val if not (np.isnan(v1_val) or np.isnan(v8_val)) else float('nan')
                print(f"  [traj] {label:<44} {v1_val:>15.4f} {v8_val:>15.4f} {delta:>+12.4f}")

    # ====================================================================
    # PART 5: v8 vs CHORD deep comparison
    # ====================================================================
    print("\n\n>>> PART 5: HEAD-TO-HEAD v8 vs CHORD")
    print("=" * 100)

    for step in comparison_steps:
        print(f"\n--- Step {step} ---")
        v8_diag = load_batch_diag(VARIANTS["v8"], step)
        ch_diag = load_batch_diag(VARIANTS["chord"], step)
        v8_traj = load_trajectory_stats(VARIANTS["v8"], step)
        ch_traj = load_trajectory_stats(VARIANTS["chord"], step)

        print(f"{'Metric':<50} {'v8 (DUET)':>15} {'CHORD':>15} {'Delta':>12}")
        print("-" * 92)

        for metric in BATCH_METRICS:
            v8_val = v8_diag.get(metric, float('nan')) if v8_diag else float('nan')
            ch_val = ch_diag.get(metric, float('nan')) if ch_diag else float('nan')
            delta = v8_val - ch_val if not (np.isnan(v8_val) or np.isnan(ch_val)) else float('nan')
            print(f"  {metric:<48} {v8_val:>15.4f} {ch_val:>15.4f} {delta:>+12.4f}")

        if v8_traj and ch_traj:
            for label, key in [
                ("SC Progress (on-policy)", "sc_progress_onpolicy"),
                ("SC Bonus (on-policy)", "sc_bonus_onpolicy"),
                ("Reward Sum (on-policy)", "reward_sum_onpolicy"),
                ("Success Rate (on-policy)", "success_onpolicy"),
                ("Offpolicy Ratio (teacher)", "offpolicy_ratio_teacher"),
                ("On-policy Adv Mean", "adv_mean_onpolicy"),
                ("Teacher Adv Mean", "adv_mean_teacher"),
                ("Response Tokens (on-policy)", "response_valid_tokens_onpolicy"),
            ]:
                v8_val = safe_mean(v8_traj[key]) if v8_traj[key] else float('nan')
                ch_val = safe_mean(ch_traj[key]) if ch_traj[key] else float('nan')
                delta = v8_val - ch_val if not (np.isnan(v8_val) or np.isnan(ch_val)) else float('nan')
                print(f"  [traj] {label:<44} {v8_val:>15.4f} {ch_val:>15.4f} {delta:>+12.4f}")

    # ====================================================================
    # PART 6: Late-training dynamics (steps 70-100 average)
    # ====================================================================
    print("\n\n>>> PART 6: LATE-TRAINING AVERAGES (Steps 70-100)")
    print("=" * 100)

    late_steps = list(range(70, 101))
    focus_all = ["v1", "v2", "v8", "v9", "chord", "luffy", "onpolicy"]

    print(f"\n{'Metric':<50}", end="")
    for name in focus_all:
        print(f" {name:>12}", end="")
    print()
    print("-" * (50 + 13 * len(focus_all)))

    for metric in BATCH_METRICS:
        print(f"  {metric:<48}", end="")
        for name in focus_all:
            vals = []
            for step in late_steps:
                data = load_batch_diag(VARIANTS[name], step)
                if data and metric in data:
                    vals.append(data[metric])
            if vals:
                print(f" {np.mean(vals):>12.4f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    # ====================================================================
    # PART 7: Reward trajectory smoothness (for overfitting detection)
    # ====================================================================
    print("\n\n>>> PART 7: REWARD TRAJECTORY ANALYSIS")
    print("=" * 100)

    for name in focus_all:
        rewards = []
        for step in STEPS:
            data = load_batch_diag(VARIANTS[name], step)
            if data and "diag/reward_onpolicy_mean" in data:
                rewards.append(data["diag/reward_onpolicy_mean"])

        if not rewards:
            continue

        rewards = np.array(rewards)

        # Split into early/mid/late phases
        early = rewards[:30]  # steps 1-30
        mid = rewards[30:70]  # steps 31-70
        late = rewards[70:]   # steps 71-100

        # Compute peak, final, monotonicity
        peak_val = np.max(rewards)
        peak_step = np.argmax(rewards) + 1
        final_val = rewards[-1]

        # Moving average 10-step
        if len(rewards) >= 10:
            ma10 = np.convolve(rewards, np.ones(10)/10, mode='valid')
            peak_ma = np.max(ma10)
            peak_ma_step = np.argmax(ma10) + 5
            final_ma = ma10[-1]
        else:
            peak_ma = peak_val
            peak_ma_step = peak_step
            final_ma = final_val

        # Decline from peak (overfitting signal)
        decline = peak_ma - final_ma

        print(f"\n{name}:")
        print(f"  Early (1-30) mean:  {np.mean(early):.4f}")
        print(f"  Mid (31-70) mean:   {np.mean(mid):.4f}")
        print(f"  Late (71-100) mean: {np.mean(late):.4f}")
        print(f"  Raw peak: {peak_val:.4f} @ step {peak_step}")
        print(f"  MA10 peak: {peak_ma:.4f} @ ~step {peak_ma_step}")
        print(f"  Final MA10: {final_ma:.4f}")
        print(f"  Decline from MA10 peak: {decline:+.4f} ({'OVERFITTING' if decline > 0.03 else 'STABLE' if decline < 0.01 else 'MILD'})")

    # ====================================================================
    # PART 8: SC Bonus vs Reward Ratio analysis
    # ====================================================================
    print("\n\n>>> PART 8: SC BONUS ANALYSIS")
    print("=" * 100)

    sc_variants = ["v1", "v2", "v8", "v9"]

    for name in sc_variants:
        print(f"\n{name}:")
        for step in [1, 10, 25, 50, 75, 100]:
            stats = load_trajectory_stats(VARIANTS[name], step)
            if not stats:
                continue

            bonus = safe_mean(stats["sc_bonus_onpolicy"])
            reward_orig = safe_mean(stats["reward_original_onpolicy"])
            reward_sum = safe_mean(stats["reward_sum_onpolicy"])
            progress = safe_mean(stats["sc_progress_onpolicy"])
            coverage = safe_mean(stats["sc_coverage_onpolicy"])
            matched = safe_mean(stats["sc_matched_states_onpolicy"])
            delta_sum = safe_mean(stats["step_delta_sum_onpolicy"])

            ratio = bonus / reward_orig if reward_orig > 0.01 else float('nan')

            print(f"  Step {step:>3}: bonus={bonus:.4f} reward={reward_orig:.4f} ratio={ratio:.4f} "
                  f"progress={progress:.4f} coverage={coverage:.4f} matched={matched:.1f} delta_sum={delta_sum:.4f}")

    # ====================================================================
    # PART 9: Advantage distribution analysis
    # ====================================================================
    print("\n\n>>> PART 9: ADVANTAGE DISTRIBUTION ANALYSIS")
    print("=" * 100)

    print(f"\n{'Step':>6} {'Variant':>12} {'adv_on_mean':>12} {'adv_tch_mean':>12} {'adv_tch_tok':>12} {'adv_pos_r':>10} {'tch_pos_r':>10}")

    for step in [10, 25, 50, 75, 100]:
        for name in ["v1", "v8", "chord", "luffy"]:
            data = load_batch_diag(VARIANTS[name], step)
            if not data:
                continue
            print(f"{step:>6} {name:>12} "
                  f"{data.get('diag/adv_onpolicy_sample_mean', float('nan')):>12.4f} "
                  f"{data.get('diag/adv_teacher_sample_mean', float('nan')):>12.4f} "
                  f"{data.get('diag/adv_teacher_token_mean', float('nan')):>12.4f} "
                  f"{data.get('diag/onpolicy_adv_pos_ratio', float('nan')):>10.4f} "
                  f"{data.get('diag/teacher_adv_pos_ratio', float('nan')):>10.4f}")

    # ====================================================================
    # PART 10: Complete step-by-step reward for v8 and chord
    # ====================================================================
    print("\n\n>>> PART 10: STEP-BY-STEP REWARD: v8 vs CHORD")
    print("=" * 80)
    print(f"{'Step':>6} {'v8_reward':>12} {'chord_reward':>12} {'delta':>12} {'v8_adv+':>10} {'ch_adv+':>10}")

    for step in STEPS:
        v8 = load_batch_diag(VARIANTS["v8"], step)
        ch = load_batch_diag(VARIANTS["chord"], step)

        v8r = v8.get("diag/reward_onpolicy_mean", float('nan')) if v8 else float('nan')
        chr_ = ch.get("diag/reward_onpolicy_mean", float('nan')) if ch else float('nan')
        delta = v8r - chr_ if not (np.isnan(v8r) or np.isnan(chr_)) else float('nan')
        v8a = v8.get("diag/onpolicy_adv_pos_ratio", float('nan')) if v8 else float('nan')
        cha = ch.get("diag/onpolicy_adv_pos_ratio", float('nan')) if ch else float('nan')

        print(f"{step:>6} {v8r:>12.4f} {chr_:>12.4f} {delta:>+12.4f} {v8a:>10.4f} {cha:>10.4f}")

    # ====================================================================
    # PART 11: All variants final summary
    # ====================================================================
    print("\n\n>>> PART 11: ALL VARIANTS COMPREHENSIVE SUMMARY")
    print("=" * 120)

    all_names = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "chord", "luffy", "onpolicy"]

    # Late training averages (steps 80-100) for key metrics
    print(f"\n{'Variant':<12} {'Val@50':>8} {'Val@100':>8} {'Rwd80-100':>10} {'Entropy':>10} {'AdvPos':>8} {'TchGap':>10} {'LLMTokR':>10}")
    print("-" * 80)

    for name in all_names:
        v50 = val_scores.get(name, {}).get(50, float('nan'))
        v100 = val_scores.get(name, {}).get(100, float('nan'))

        late_vals = {}
        for metric_short, metric_full in [
            ("rwd", "diag/reward_onpolicy_mean"),
            ("ent", "diag/entropy_onpolicy_token_mean"),
            ("adv", "diag/onpolicy_adv_pos_ratio"),
            ("gap", "diag/group_teacher_minus_on_reward_mean"),
            ("tok", "diag/llm_token_ratio_in_response"),
        ]:
            vals = []
            for step in range(80, 101):
                data = load_batch_diag(VARIANTS[name], step)
                if data and metric_full in data:
                    vals.append(data[metric_full])
            late_vals[metric_short] = np.mean(vals) if vals else float('nan')

        print(f"{name:<12} {v50:>8.3f} {v100:>8.3f} {late_vals['rwd']:>10.4f} "
              f"{late_vals['ent']:>10.4f} {late_vals['adv']:>8.4f} {late_vals['gap']:>10.4f} {late_vals['tok']:>10.4f}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
