#!/usr/bin/env python3
"""
ALFWorld 7B vs 3B Deep Analysis for DUET Paper
Matches the WebShop analysis format to test the hypothesis:
"Off-policy teacher data provides diminishing returns for 7B models"
"""

import json
import os
import sys
from collections import defaultdict
import statistics

# ============================================================
# Configuration
# ============================================================
TRAJ_BASE = "/data/home/qisheng/EvolAnalsis/checkpoints/agentevolver"
VAL_BASE = "/data/home/qisheng/EvolAnalsis/experiments/alfworld"

EXPERIMENTS = {
    # 7B experiments
    "7B OnPolicy":  {"dir": "alfworld_7b_onpolicy",       "size": "7B", "method": "OnPolicy", "has_teacher": False},
    "7B LUFFY":     {"dir": "alfworld_7b_luffy",           "size": "7B", "method": "LUFFY",    "has_teacher": True},
    "7B CHORD":     {"dir": "alfworld_7b_chord",           "size": "7B", "method": "CHORD",    "has_teacher": True},
    "7B DUET":      {"dir": "alfworld_7b_duet",            "size": "7B", "method": "DUET",     "has_teacher": True},
    # 3B experiments
    "3B OnPolicy":  {"dir": "alfworld_3b_grpo_react_tags", "size": "3B", "method": "OnPolicy", "has_teacher": False},
    "3B LUFFY":     {"dir": "alfworld_3b_luffy",           "size": "3B", "method": "LUFFY",    "has_teacher": True},
    "3B DUET":      {"dir": "alfworld_3b_duet_0329",       "size": "3B", "method": "DUET",     "has_teacher": True},
}

# Key metrics to extract
METRICS = [
    "diag/reward_onpolicy_mean",
    "diag/reward_teacher_mean",
    "diag/group_all_reward_mean",
    "diag/group_non_teacher_reward_mean",
    "diag/group_teacher_minus_on_reward_mean",
    "diag/group_teacher_minus_on_reward_std",
    "diag/onpolicy_adv_pos_ratio",
    "diag/teacher_adv_pos_ratio",
    "diag/entropy_onpolicy_token_mean",
    "diag/entropy_teacher_token_mean",
    "diag/adv_onpolicy_sample_mean",
    "diag/adv_teacher_sample_mean",
    "diag/teacher_sample_ratio",
    "diag/offpolicy_sample_ratio",
    "diag/teacher_token_ratio",
    "diag/exp_token_ratio",
    "actor/on_pg_clipfrac",
    "actor/on_pg_cliphit_rate",
]

# ============================================================
# Data Loading
# ============================================================
def load_batch_diags(exp_name, exp_info):
    """Load all batch_diag JSON files for an experiment."""
    traj_dir = os.path.join(TRAJ_BASE, exp_info["dir"], "Trajectory")
    data = {}
    if not os.path.exists(traj_dir):
        print(f"WARNING: {traj_dir} does not exist", file=sys.stderr)
        return data
    for fname in os.listdir(traj_dir):
        if fname.startswith("batch_diag_step_") and fname.endswith(".json"):
            step = int(fname.replace("batch_diag_step_", "").replace(".json", ""))
            fpath = os.path.join(traj_dir, fname)
            with open(fpath) as f:
                data[step] = json.load(f)
    return data


def load_validation(exp_name, exp_info):
    """Load validation JSONL files."""
    val_dir = os.path.join(VAL_BASE, exp_info["dir"], "validation_log")
    data = {}
    if not os.path.exists(val_dir):
        print(f"WARNING: {val_dir} does not exist", file=sys.stderr)
        return data
    for fname in os.listdir(val_dir):
        if fname.endswith(".jsonl"):
            step = int(fname.replace(".jsonl", ""))
            fpath = os.path.join(val_dir, fname)
            scores = []
            with open(fpath) as f:
                for line in f:
                    d = json.loads(line)
                    scores.append(d.get("score", d.get("reward", 0.0)))
            data[step] = {
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "n": len(scores),
                "sum": sum(scores),
            }
    return data


# ============================================================
# Load everything
# ============================================================
print("Loading all experiment data...\n")
all_diags = {}
all_vals = {}
for name, info in EXPERIMENTS.items():
    all_diags[name] = load_batch_diags(name, info)
    all_vals[name] = load_validation(name, info)
    steps = sorted(all_diags[name].keys())
    print(f"  {name}: {len(steps)} training steps, {len(all_vals[name])} validation checkpoints")
    if steps:
        print(f"    Steps range: {steps[0]} - {steps[-1]}")

# ============================================================
# Helper Functions
# ============================================================
def get_metric_at_step(exp_name, step, metric):
    """Get a metric value at a specific step."""
    if step in all_diags[exp_name]:
        return all_diags[exp_name][step].get(metric, None)
    return None


def get_metric_series(exp_name, metric):
    """Get a sorted list of (step, value) for a metric."""
    series = []
    for step in sorted(all_diags[exp_name].keys()):
        v = all_diags[exp_name][step].get(metric, None)
        if v is not None:
            series.append((step, v))
    return series


def bucket_average(series, bucket_size=10):
    """Compute bucket averages for smoothing."""
    if not series:
        return []
    buckets = defaultdict(list)
    for step, val in series:
        bucket = ((step - 1) // bucket_size) * bucket_size + 1
        buckets[bucket].append(val)
    result = []
    for bucket in sorted(buckets.keys()):
        result.append((bucket, statistics.mean(buckets[bucket])))
    return result


def fmt(v, digits=4):
    """Format a number for table display."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def pct(v, digits=1):
    """Format as percentage."""
    if v is None:
        return "N/A"
    return f"{v*100:.{digits}f}%"


# ============================================================
# ANALYSIS 1: Final/Peak Performance Table
# ============================================================
print("\n" + "="*80)
print("SECTION 1: PERFORMANCE SUMMARY (Training Reward)")
print("="*80)

print("\n### Training Reward at Key Checkpoints")
print(f"{'Experiment':<16} {'Step 1':>10} {'Step 10':>10} {'Step 25':>10} {'Step 50':>10} {'Step 75':>10} {'Step 100':>10} {'Peak':>10} {'Peak@':>7}")

for name in ["3B OnPolicy", "3B LUFFY", "3B DUET", "", "7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]:
    if name == "":
        print("-" * 100)
        continue
    steps_check = [1, 10, 25, 50, 75, 100]
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    peak_val = max(series, key=lambda x: x[1]) if series else (0, 0)
    vals = []
    for s in steps_check:
        v = get_metric_at_step(name, s, "diag/reward_onpolicy_mean")
        vals.append(fmt(v))
    max_step = max(all_diags[name].keys()) if all_diags[name] else 0
    # For experiments that don't have step 100, show the last step
    print(f"{name:<16} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {vals[4]:>10} {vals[5]:>10} {fmt(peak_val[1]):>10} {peak_val[0]:>7}")


# ============================================================
# ANALYSIS 2: Validation Performance Table
# ============================================================
print("\n" + "="*80)
print("SECTION 2: VALIDATION PERFORMANCE (200 episodes)")
print("="*80)

print(f"\n{'Experiment':<16} {'Val@50 Mean':>12} {'Val@50 Std':>12} {'Val@100 Mean':>13} {'Val@100 Std':>12} {'N':>5}")
for name in ["3B OnPolicy", "3B LUFFY", "3B DUET", "", "7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]:
    if name == "":
        print("-" * 72)
        continue
    v50 = all_vals[name].get(50, {})
    v100 = all_vals[name].get(100, {})
    print(f"{name:<16} {fmt(v50.get('mean'), 4):>12} {fmt(v50.get('std'), 4):>12} {fmt(v100.get('mean'), 4):>13} {fmt(v100.get('std'), 4):>12} {v50.get('n', 'N/A'):>5}")


# ============================================================
# ANALYSIS 3: DUET Advantage Quantification (3B vs 7B)
# ============================================================
print("\n" + "="*80)
print("SECTION 3: DUET ADVANTAGE QUANTIFICATION")
print("="*80)

# Compute DUET advantage = (DUET - OnPolicy) / OnPolicy
print("\n### Training Reward: DUET vs OnPolicy")
print(f"{'Metric':<40} {'3B':>12} {'7B':>12} {'Collapse':>12}")
print("-" * 80)

for step_label, step in [("Step 1 reward", 1), ("Step 10 reward", 10), ("Step 25 reward", 25),
                          ("Step 50 reward", 50), ("Step 75 reward", 75), ("Step 100 reward", 100)]:
    for size_label, duet_name, base_name in [("3B", "3B DUET", "3B OnPolicy"), ("7B", "7B DUET", "7B OnPolicy")]:
        duet_v = get_metric_at_step(duet_name, step, "diag/reward_onpolicy_mean")
        base_v = get_metric_at_step(base_name, step, "diag/reward_onpolicy_mean")
        if size_label == "3B":
            d3b, b3b = duet_v, base_v
        else:
            d7b, b7b = duet_v, base_v

    if d3b is not None and b3b is not None and b3b > 0:
        adv_3b = (d3b - b3b) / b3b
    else:
        adv_3b = None
    if d7b is not None and b7b is not None and b7b > 0:
        adv_7b = (d7b - b7b) / b7b
    else:
        adv_7b = None

    if adv_3b is not None and adv_7b is not None and adv_3b != 0:
        collapse = 1.0 - (adv_7b / adv_3b) if adv_3b != 0 else None
    else:
        collapse = None

    print(f"{step_label:<40} {pct(adv_3b):>12} {pct(adv_7b):>12} {pct(collapse) if collapse is not None else 'N/A':>12}")

# Validation-based DUET advantage
print("\n### Validation: DUET vs OnPolicy")
print(f"{'Checkpoint':<40} {'3B':>12} {'7B':>12} {'Collapse':>12}")
print("-" * 80)
for step in [50, 100]:
    for size_label, duet_name, base_name in [("3B", "3B DUET", "3B OnPolicy"), ("7B", "7B DUET", "7B OnPolicy")]:
        duet_v = all_vals.get(duet_name, {}).get(step, {}).get("mean")
        base_v = all_vals.get(base_name, {}).get(step, {}).get("mean")
        if size_label == "3B":
            d3b, b3b = duet_v, base_v
        else:
            d7b, b7b = duet_v, base_v

    adv_3b = (d3b - b3b) / b3b if d3b and b3b and b3b > 0 else None
    adv_7b = (d7b - b7b) / b7b if d7b and b7b and b7b > 0 else None
    collapse = (1.0 - adv_7b / adv_3b) if adv_3b and adv_7b and adv_3b != 0 else None

    print(f"Val@{step:<36} {pct(adv_3b):>12} {pct(adv_7b):>12} {pct(collapse) if collapse is not None else 'N/A':>12}")

# LUFFY advantage too
print("\n### Validation: LUFFY vs OnPolicy")
print(f"{'Checkpoint':<40} {'3B':>12} {'7B':>12} {'Collapse':>12}")
print("-" * 80)
for step in [50, 100]:
    for size_label, method_name, base_name in [("3B", "3B LUFFY", "3B OnPolicy"), ("7B", "7B LUFFY", "7B OnPolicy")]:
        method_v = all_vals.get(method_name, {}).get(step, {}).get("mean")
        base_v = all_vals.get(base_name, {}).get(step, {}).get("mean")
        if size_label == "3B":
            d3b, b3b = method_v, base_v
        else:
            d7b, b7b = method_v, base_v

    adv_3b = (d3b - b3b) / b3b if d3b and b3b and b3b > 0 else None
    adv_7b = (d7b - b7b) / b7b if d7b and b7b and b7b > 0 else None
    collapse = (1.0 - adv_7b / adv_3b) if adv_3b and adv_7b and adv_3b != 0 else None

    print(f"Val@{step:<36} {pct(adv_3b):>12} {pct(adv_7b):>12} {pct(collapse) if collapse is not None else 'N/A':>12}")


# ============================================================
# ANALYSIS 4: Initial Capability (Step 1) Comparison
# ============================================================
print("\n" + "="*80)
print("SECTION 4: INITIAL CAPABILITY (Step 1)")
print("="*80)

print(f"\n{'Experiment':<16} {'Reward':>10} {'AdvPosRat':>10} {'Entropy':>10}")
print("-" * 50)
for name in ["3B OnPolicy", "3B LUFFY", "3B DUET", "", "7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]:
    if name == "":
        print("-" * 50)
        continue
    r = get_metric_at_step(name, 1, "diag/reward_onpolicy_mean")
    adv = get_metric_at_step(name, 1, "diag/onpolicy_adv_pos_ratio")
    ent = get_metric_at_step(name, 1, "diag/entropy_onpolicy_token_mean")
    print(f"{name:<16} {fmt(r):>10} {fmt(adv):>10} {fmt(ent):>10}")

print("\n7B/3B initial reward ratio (OnPolicy):")
r3b = get_metric_at_step("3B OnPolicy", 1, "diag/reward_onpolicy_mean")
r7b = get_metric_at_step("7B OnPolicy", 1, "diag/reward_onpolicy_mean")
if r3b and r7b and r3b > 0:
    print(f"  7B step-1 reward / 3B step-1 reward = {r7b/r3b:.3f}x ({fmt(r7b)} / {fmt(r3b)})")


# ============================================================
# ANALYSIS 5: Teacher Gap Evolution
# ============================================================
print("\n" + "="*80)
print("SECTION 5: TEACHER GAP EVOLUTION (teacher_reward - onpolicy_reward)")
print("="*80)

teacher_methods = ["LUFFY", "DUET"]
print(f"\n{'Step':<8}", end="")
for size in ["3B", "7B"]:
    for method in teacher_methods:
        name = f"{size} {method}"
        if name in EXPERIMENTS:
            print(f" {name:>12}", end="")
print()
print("-" * 60)

for step in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print(f"{step:<8}", end="")
    for size in ["3B", "7B"]:
        for method in teacher_methods:
            name = f"{size} {method}"
            if name in EXPERIMENTS:
                gap = get_metric_at_step(name, step, "diag/group_teacher_minus_on_reward_mean")
                print(f" {fmt(gap):>12}", end="")
    print()

# Bucketed teacher gap
print("\n### 10-Step Bucketed Teacher Gap")
print(f"{'Bucket':<10}", end="")
for size in ["3B", "7B"]:
    for method in teacher_methods:
        name = f"{size} {method}"
        if name in EXPERIMENTS:
            print(f" {name:>12}", end="")
print()
print("-" * 60)

for size in ["3B", "7B"]:
    for method in teacher_methods:
        name = f"{size} {method}"
        if name not in EXPERIMENTS:
            continue
        series = get_metric_series(name, "diag/group_teacher_minus_on_reward_mean")
        bucketed = bucket_average(series, 10)
        if size == "3B" and method == teacher_methods[0]:
            all_buckets_data = {}
        for bstart, bval in bucketed:
            if bstart not in all_buckets_data:
                all_buckets_data[bstart] = {}
            all_buckets_data[bstart][name] = bval

# Print bucketed data
for bstart in sorted(all_buckets_data.keys()):
    print(f"{bstart}-{bstart+9:<6}", end="")
    for size in ["3B", "7B"]:
        for method in teacher_methods:
            name = f"{size} {method}"
            if name in EXPERIMENTS:
                v = all_buckets_data.get(bstart, {}).get(name)
                print(f" {fmt(v):>12}", end="")
    print()


# ============================================================
# ANALYSIS 6: On-Policy Advantage Positive Ratio
# ============================================================
print("\n" + "="*80)
print("SECTION 6: ON-POLICY ADVANTAGE POSITIVE RATIO")
print("="*80)

print(f"\n{'Step':<8}", end="")
all_exp_names = ["3B OnPolicy", "3B LUFFY", "3B DUET", "7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]
for name in all_exp_names:
    print(f" {name:>14}", end="")
print()
print("-" * 120)

for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in all_exp_names:
        v = get_metric_at_step(name, step, "diag/onpolicy_adv_pos_ratio")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 7: Entropy Evolution
# ============================================================
print("\n" + "="*80)
print("SECTION 7: ON-POLICY ENTROPY EVOLUTION")
print("="*80)

print(f"\n{'Step':<8}", end="")
for name in all_exp_names:
    print(f" {name:>14}", end="")
print()
print("-" * 120)

for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in all_exp_names:
        v = get_metric_at_step(name, step, "diag/entropy_onpolicy_token_mean")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 8: Training Reward Curves (Bucketed)
# ============================================================
print("\n" + "="*80)
print("SECTION 8: TRAINING REWARD CURVES (10-Step Buckets)")
print("="*80)

print(f"\n{'Bucket':<10}", end="")
for name in all_exp_names:
    print(f" {name:>14}", end="")
print()
print("-" * 120)

all_reward_buckets = {}
for name in all_exp_names:
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    bucketed = bucket_average(series, 10)
    for bstart, bval in bucketed:
        if bstart not in all_reward_buckets:
            all_reward_buckets[bstart] = {}
        all_reward_buckets[bstart][name] = bval

for bstart in sorted(all_reward_buckets.keys()):
    print(f"{bstart}-{bstart+9:<7}", end="")
    for name in all_exp_names:
        v = all_reward_buckets.get(bstart, {}).get(name)
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 9: Teacher Sample Ratio
# ============================================================
print("\n" + "="*80)
print("SECTION 9: TEACHER SAMPLE RATIO (methods with teacher mixing)")
print("="*80)

teacher_exps = ["3B LUFFY", "3B DUET", "7B LUFFY", "7B CHORD", "7B DUET"]
print(f"\n{'Step':<8}", end="")
for name in teacher_exps:
    print(f" {name:>14}", end="")
print()
print("-" * 90)

for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in teacher_exps:
        v = get_metric_at_step(name, step, "diag/teacher_sample_ratio")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 10: All-Method Comparison at Key Checkpoints
# ============================================================
print("\n" + "="*80)
print("SECTION 10: ALL-METHOD COMPARISON (Reward + Validation)")
print("="*80)

# Comprehensive table
print(f"\n{'Method':<16} {'Size':>4} {'Train@50':>10} {'Train@100':>11} {'Val@50':>10} {'Val@100':>10} {'Peak Train':>11} {'Final Gap':>10}")
print("-" * 90)

for name in ["3B OnPolicy", "3B LUFFY", "3B DUET", "", "7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]:
    if name == "":
        print("-" * 90)
        continue
    info = EXPERIMENTS[name]
    t50 = get_metric_at_step(name, 50, "diag/reward_onpolicy_mean")
    t100 = get_metric_at_step(name, 100, "diag/reward_onpolicy_mean")
    v50 = all_vals.get(name, {}).get(50, {}).get("mean")
    v100 = all_vals.get(name, {}).get(100, {}).get("mean")
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    peak = max(series, key=lambda x: x[1])[1] if series else None

    # Final gap vs teacher
    gap = get_metric_at_step(name, 100, "diag/group_teacher_minus_on_reward_mean") if info["has_teacher"] else None

    print(f"{name:<16} {info['size']:>4} {fmt(t50):>10} {fmt(t100):>11} {fmt(v50):>10} {fmt(v100):>10} {fmt(peak):>11} {fmt(gap) if gap is not None else 'N/A':>10}")


# ============================================================
# ANALYSIS 11: Head-to-Head at Every Step (DUET vs OnPolicy)
# ============================================================
print("\n" + "="*80)
print("SECTION 11: DUET vs ONPOLICY HEAD-TO-HEAD (Every 10 Steps)")
print("="*80)

print(f"\n{'Step':<8} {'3B OnPol':>10} {'3B DUET':>10} {'3B Delta':>10} {'3B Rel%':>8} | {'7B OnPol':>10} {'7B DUET':>10} {'7B Delta':>10} {'7B Rel%':>8}")
print("-" * 100)

for step in range(1, 101, 5):
    vals = {}
    for name in ["3B OnPolicy", "3B DUET", "7B OnPolicy", "7B DUET"]:
        vals[name] = get_metric_at_step(name, step, "diag/reward_onpolicy_mean")

    if all(v is not None for v in vals.values()):
        d3 = vals["3B DUET"] - vals["3B OnPolicy"]
        r3 = d3 / vals["3B OnPolicy"] * 100 if vals["3B OnPolicy"] > 0 else 0
        d7 = vals["7B DUET"] - vals["7B OnPolicy"]
        r7 = d7 / vals["7B OnPolicy"] * 100 if vals["7B OnPolicy"] > 0 else 0
        print(f"{step:<8} {fmt(vals['3B OnPolicy']):>10} {fmt(vals['3B DUET']):>10} {d3:>+10.4f} {r3:>+7.1f}% | {fmt(vals['7B OnPolicy']):>10} {fmt(vals['7B DUET']):>10} {d7:>+10.4f} {r7:>+7.1f}%")


# ============================================================
# ANALYSIS 12: When does 7B OnPolicy catch teacher methods?
# ============================================================
print("\n" + "="*80)
print("SECTION 12: CONVERGENCE ANALYSIS - When does OnPolicy match teacher methods?")
print("="*80)

# For 7B: find the step where OnPolicy reward >= DUET/LUFFY reward
print("\n### 7B: Step where OnPolicy reward >= method reward")
for method_name in ["7B LUFFY", "7B CHORD", "7B DUET"]:
    on_series = dict(get_metric_series("7B OnPolicy", "diag/reward_onpolicy_mean"))
    method_series = dict(get_metric_series(method_name, "diag/reward_onpolicy_mean"))

    crossover_step = None
    for step in sorted(set(on_series.keys()) & set(method_series.keys())):
        if on_series[step] >= method_series[step]:
            crossover_step = step
            break

    if crossover_step:
        print(f"  {method_name}: OnPolicy catches at step {crossover_step} "
              f"(OnPolicy={on_series[crossover_step]:.4f}, {method_name}={method_series[crossover_step]:.4f})")
    else:
        # Check last available step
        common_steps = sorted(set(on_series.keys()) & set(method_series.keys()))
        if common_steps:
            last = common_steps[-1]
            gap = method_series[last] - on_series[last]
            print(f"  {method_name}: OnPolicy NEVER catches (final gap at step {last}: {gap:+.4f})")

# Same for 3B
print("\n### 3B: Step where OnPolicy reward >= method reward")
for method_name in ["3B LUFFY", "3B DUET"]:
    on_series = dict(get_metric_series("3B OnPolicy", "diag/reward_onpolicy_mean"))
    method_series = dict(get_metric_series(method_name, "diag/reward_onpolicy_mean"))

    crossover_step = None
    for step in sorted(set(on_series.keys()) & set(method_series.keys())):
        if on_series[step] >= method_series[step]:
            crossover_step = step
            break

    if crossover_step:
        print(f"  {method_name}: OnPolicy catches at step {crossover_step} "
              f"(OnPolicy={on_series[crossover_step]:.4f}, {method_name}={method_series[crossover_step]:.4f})")
    else:
        common_steps = sorted(set(on_series.keys()) & set(method_series.keys()))
        if common_steps:
            last = common_steps[-1]
            gap = method_series[last] - on_series[last]
            print(f"  {method_name}: OnPolicy NEVER catches (final gap at step {last}: {gap:+.4f})")


# ============================================================
# ANALYSIS 13: Reward Statistics (distribution at key steps)
# ============================================================
print("\n" + "="*80)
print("SECTION 13: GROUP REWARD DISTRIBUTION")
print("="*80)

print(f"\n### group_all_reward_mean (all samples including teacher)")
print(f"{'Step':<8}", end="")
for name in teacher_exps:
    print(f" {name:>14}", end="")
print()
print("-" * 90)
for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in teacher_exps:
        v = get_metric_at_step(name, step, "diag/group_all_reward_mean")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 14: Clip Fraction Analysis
# ============================================================
print("\n" + "="*80)
print("SECTION 14: PPO CLIP FRACTION (training stability)")
print("="*80)

print(f"\n{'Step':<8}", end="")
for name in all_exp_names:
    print(f" {name:>14}", end="")
print()
print("-" * 120)

for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in all_exp_names:
        v = get_metric_at_step(name, step, "actor/on_pg_clipfrac")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 15: Key Comparative Summary Statistics
# ============================================================
print("\n" + "="*80)
print("SECTION 15: KEY SUMMARY STATISTICS")
print("="*80)

# Compute summary stats
print("\n### Training Reward Summary")
print(f"{'Experiment':<16} {'Mean(all)':>10} {'Mean(1-50)':>12} {'Mean(51-100)':>13} {'Std(all)':>10} {'Final(100)':>11}")
print("-" * 80)

for name in all_exp_names:
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    if not series:
        continue
    all_vals_list = [v for _, v in series]
    first_half = [v for s, v in series if s <= 50]
    second_half = [v for s, v in series if s > 50]
    final = series[-1][1] if series else None

    print(f"{name:<16} {statistics.mean(all_vals_list):>10.4f} {statistics.mean(first_half) if first_half else 0:>12.4f} {statistics.mean(second_half) if second_half else 0:>13.4f} {statistics.stdev(all_vals_list) if len(all_vals_list) > 1 else 0:>10.4f} {fmt(final):>11}")


# ============================================================
# ANALYSIS 16: Teacher Advantage Positive Ratio (teacher learning signal strength)
# ============================================================
print("\n" + "="*80)
print("SECTION 16: TEACHER ADVANTAGE POSITIVE RATIO (does teacher still provide positive signal?)")
print("="*80)

print(f"\n{'Step':<8}", end="")
for name in teacher_exps:
    print(f" {name:>14}", end="")
print()
print("-" * 90)

for step in [1, 10, 25, 50, 75, 100]:
    print(f"{step:<8}", end="")
    for name in teacher_exps:
        v = get_metric_at_step(name, step, "diag/teacher_adv_pos_ratio")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# ANALYSIS 17: WebShop vs ALFWorld Comparison Summary
# ============================================================
print("\n" + "="*80)
print("SECTION 17: ALFWORLD vs WEBSHOP HYPOTHESIS TEST")
print("="*80)

print("""
Key WebShop findings to test against ALFWorld:
1. WebShop 7B GRPO reaches 0.760 val (same as teacher-augmented methods)
2. DUET advantage collapsed from +86% (3B) to +6.5% (7B) on WebShop
3. Teacher gap closes by step 50 at 7B (gap=0.078)
4. All teacher-mixing methods fail to beat on-policy at 7B
5. SC bonus gets proportionally weaker at high reward levels
""")

# 1. Does 7B OnPolicy match teacher methods at convergence?
print("### Test 1: Does 7B OnPolicy match teacher methods at convergence?")
v100_on = all_vals.get("7B OnPolicy", {}).get(100, {}).get("mean")
v100_duet = all_vals.get("7B DUET", {}).get(100, {}).get("mean")
v100_luffy = all_vals.get("7B LUFFY", {}).get(100, {}).get("mean")
print(f"  7B OnPolicy Val@100: {fmt(v100_on)}")
print(f"  7B DUET Val@100:     {fmt(v100_duet)}")
print(f"  7B LUFFY Val@100:    {fmt(v100_luffy)}")
if v100_on and v100_duet:
    diff = v100_duet - v100_on
    print(f"  DUET - OnPolicy gap: {diff:+.4f}")
    print(f"  -> {'YES, converged' if abs(diff) < 0.03 else 'NO, gap persists' if diff > 0 else 'OnPolicy WINS'}")

# 2. DUET advantage collapse
print("\n### Test 2: DUET advantage collapse 3B -> 7B")
v50_3b_on = all_vals.get("3B OnPolicy", {}).get(50, {}).get("mean")
v50_3b_duet = all_vals.get("3B DUET", {}).get(50, {}).get("mean")
v50_7b_on = all_vals.get("7B OnPolicy", {}).get(50, {}).get("mean")
v50_7b_duet = all_vals.get("7B DUET", {}).get(50, {}).get("mean")
if all(x is not None for x in [v50_3b_on, v50_3b_duet, v50_7b_on, v50_7b_duet]):
    adv_3b = (v50_3b_duet - v50_3b_on) / v50_3b_on * 100 if v50_3b_on > 0 else 0
    adv_7b = (v50_7b_duet - v50_7b_on) / v50_7b_on * 100 if v50_7b_on > 0 else 0
    print(f"  Val@50 DUET advantage at 3B: {adv_3b:+.1f}%")
    print(f"  Val@50 DUET advantage at 7B: {adv_7b:+.1f}%")
    if adv_3b != 0:
        print(f"  Collapse factor: {(1 - adv_7b/adv_3b)*100:.1f}%")
    print(f"  WebShop comparison: 3B +86%, 7B +6.5%, collapse 92.4%")

# 3. Teacher gap at step 50
print("\n### Test 3: Teacher gap at step 50")
for name in ["7B LUFFY", "7B DUET", "3B LUFFY", "3B DUET"]:
    gap = get_metric_at_step(name, 50, "diag/group_teacher_minus_on_reward_mean")
    print(f"  {name} teacher gap at step 50: {fmt(gap)}")
print(f"  WebShop 7B gap at step 50: 0.078 (for reference)")

# 4. Does 7B reach near-ceiling?
print("\n### Test 4: Does 7B reach reward ceiling (near 1.0)?")
for name in ["7B OnPolicy", "7B LUFFY", "7B CHORD", "7B DUET"]:
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    peak = max(series, key=lambda x: x[1]) if series else (0, 0)
    final = get_metric_at_step(name, 100, "diag/reward_onpolicy_mean")
    print(f"  {name}: peak={fmt(peak[1])} @step {peak[0]}, final={fmt(final)}")

for name in ["3B OnPolicy", "3B LUFFY", "3B DUET"]:
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    peak = max(series, key=lambda x: x[1]) if series else (0, 0)
    final = get_metric_at_step(name, 100, "diag/reward_onpolicy_mean")
    print(f"  {name}: peak={fmt(peak[1])} @step {peak[0]}, final={fmt(final)}")


# ============================================================
# ANALYSIS 18: Fine-grained early training (steps 1-20)
# ============================================================
print("\n" + "="*80)
print("SECTION 18: EARLY TRAINING DYNAMICS (Steps 1-20)")
print("="*80)

print(f"\n### On-Policy Reward (Steps 1-20)")
print(f"{'Step':<6}", end="")
for name in all_exp_names:
    print(f" {name:>14}", end="")
print()
print("-" * 110)

for step in range(1, 21):
    print(f"{step:<6}", end="")
    for name in all_exp_names:
        v = get_metric_at_step(name, step, "diag/reward_onpolicy_mean")
        print(f" {fmt(v):>14}", end="")
    print()

print(f"\n### Teacher Gap (Steps 1-20)")
print(f"{'Step':<6}", end="")
for name in teacher_exps:
    print(f" {name:>14}", end="")
print()
print("-" * 90)

for step in range(1, 21):
    print(f"{step:<6}", end="")
    for name in teacher_exps:
        v = get_metric_at_step(name, step, "diag/group_teacher_minus_on_reward_mean")
        print(f" {fmt(v):>14}", end="")
    print()


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*80)
print("FINAL ANALYSIS SUMMARY")
print("="*80)

# Compute key comparison numbers
print("\n--- Computing final comparison metrics ---")

# Average reward over all steps
for name in all_exp_names:
    series = get_metric_series(name, "diag/reward_onpolicy_mean")
    if series:
        avg = statistics.mean([v for _, v in series])
        print(f"  {name} avg reward (all steps): {avg:.4f}")

# Average teacher gap
print()
for name in teacher_exps:
    series = get_metric_series(name, "diag/group_teacher_minus_on_reward_mean")
    if series:
        avg = statistics.mean([v for _, v in series])
        first_10 = statistics.mean([v for s, v in series if s <= 10])
        last_10_vals = [v for s, v in series if s > 90]
        if not last_10_vals:
            last_10_vals = [v for s, v in series if s > max(s2 for s2, _ in series) - 10]
        last_10 = statistics.mean(last_10_vals) if last_10_vals else float('nan')
        print(f"  {name} teacher gap: avg={avg:.4f}, first10={first_10:.4f}, last10={last_10:.4f}")

print("\nDone.")
