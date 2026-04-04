#!/usr/bin/env python3
"""Comprehensive DUET 0402 vs orig vs LUFFY analysis."""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = Path("/data/code/exp/EvolAnalsis")
OUT = BASE / "analysis_outputs/duet_0402_analysis"

RUNS = {
    "duet_0402": BASE / "checkpoints/agentevolver/webshop_3b_duet_0402/Trajectory",
    "duet_orig": BASE / "checkpoints/agentevolver/webshop_3b_duet/Trajectory",
    "luffy": BASE / "checkpoints/agentevolver/webshop_3b_luffy/Trajectory",
}

VAL_LOGS = {
    "duet_0402": BASE / "experiments/webshop/webshop_3b_duet_0402/validation_log",
    "luffy": BASE / "experiments/webshop/webshop_3b_luffy/validation_log",
}


def load_batch_diags(traj_dir):
    """Load all batch_diag_step_*.json files, return dict: step -> data."""
    diags = {}
    for f in traj_dir.glob("batch_diag_step_*.json"):
        step = int(f.stem.split("_")[-1])
        with open(f) as fh:
            diags[step] = json.load(fh)
    return diags


def load_trajectory_diags(traj_dir, steps=None):
    """Load per-sample diag info from trajectories_step_*.jsonl.
    Returns dict: step -> list of sample dicts with diag info."""
    result = {}
    for f in traj_dir.glob("trajectories_step_*.jsonl"):
        step = int(f.stem.split("_")[-1])
        if steps is not None and step not in steps:
            continue
        samples = []
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                s = {
                    "is_teacher": d.get("diag", {}).get("is_teacher", False),
                    "adv_mean": d.get("diag", {}).get("adv_mean"),
                    "adv_std": d.get("diag", {}).get("adv_std"),
                    "old_log_prob_mean": d.get("diag", {}).get("old_log_prob_mean"),
                    "teacher_old_logp_mean": d.get("diag", {}).get("teacher_old_logp_mean"),
                    "reward_sum": d.get("diag", {}).get("reward_sum"),
                    "sc_progress": d.get("diag", {}).get("sc_progress"),
                    "sc_bonus": d.get("diag", {}).get("sc_bonus"),
                    "sc_coverage": d.get("diag", {}).get("sc_coverage"),
                    "sc_matched_states": d.get("diag", {}).get("sc_matched_states"),
                    "reward_original": d.get("diag", {}).get("reward_original"),
                    "reward_components": d.get("diag", {}).get("reward_components"),
                    "success": d.get("success", False),
                    "task_id": d.get("task_id"),
                    "uid": d.get("diag", {}).get("uid"),
                }
                samples.append(s)
        result[step] = samples
    return result


def load_validation_log(val_dir, step=100):
    """Load validation log JSONL."""
    path = val_dir / f"{step}.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# ============================================================
# 1. BATCH_DIAG TIMESERIES
# ============================================================
print("=" * 80)
print("1. LOADING BATCH_DIAG TIMESERIES")
print("=" * 80)

all_diags = {}
for name, path in RUNS.items():
    all_diags[name] = load_batch_diags(path)
    steps = sorted(all_diags[name].keys())
    print(f"  {name}: {len(steps)} steps, range [{steps[0]}, {steps[-1]}]")

# Key metrics to compare
metrics = [
    "diag/adv_teacher_sample_mean",
    "diag/adv_teacher_token_mean",
    "diag/adv_onpolicy_sample_mean",
    "diag/adv_onpolicy_token_mean",
    "diag/reward_onpolicy_mean",
    "diag/reward_teacher_mean",
    "diag/teacher_adv_pos_ratio",
    "diag/onpolicy_adv_pos_ratio",
    "diag/group_teacher_minus_on_reward_mean",
    "diag/entropy_onpolicy_token_mean",
    "diag/entropy_teacher_token_mean",
    "diag/teacher_sample_ratio",
    "diag/teacher_token_ratio",
]

print("\n" + "=" * 80)
print("2. TEACHER ADVANTAGE DYNAMICS (teacher_adv_sample_mean)")
print("=" * 80)

# Collect timeseries for key metrics
def get_timeseries(diags, metric):
    steps = sorted(diags.keys())
    vals = [diags[s].get(metric) for s in steps]
    return steps, vals

# Print comparison at key checkpoints
checkpoints = [1, 5, 10, 20, 30, 50, 75, 100]
print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: adv_teacher_sample_mean")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/adv_teacher_sample_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: adv_teacher_token_mean")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/adv_teacher_token_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print("\n" + "=" * 80)
print("3. ON-POLICY REWARD AND ADVANTAGE DYNAMICS")
print("=" * 80)

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: reward_onpolicy_mean")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/reward_onpolicy_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: adv_onpolicy_sample_mean")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/adv_onpolicy_sample_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: onpolicy_adv_pos_ratio")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/onpolicy_adv_pos_ratio")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: teacher_adv_pos_ratio")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/teacher_adv_pos_ratio")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print("\n" + "=" * 80)
print("4. REWARD GAP AND ENTROPY")
print("=" * 80)

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: group_teacher_minus_on_reward_mean")
print("-" * 80)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/group_teacher_minus_on_reward_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")

print(f"\n{'Step':>5} | {'DUET 0402':>15} | {'DUET orig':>15} | {'LUFFY':>15} | metric: entropy_onpolicy_token_mean")
print("-" * 80)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/entropy_onpolicy_token_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>15} | {vals[1]:>15} | {vals[2]:>15}")


# ============================================================
# 5. PER-SAMPLE TRAJECTORY ANALYSIS (select key steps)
# ============================================================
print("\n" + "=" * 80)
print("5. PER-SAMPLE TRAJECTORY ANALYSIS")
print("=" * 80)

sample_steps = [1, 10, 20, 50, 75, 100]

for name in ["duet_0402", "duet_orig", "luffy"]:
    print(f"\n--- {name} ---")
    trajs = load_trajectory_diags(RUNS[name], steps=sample_steps)
    for step in sample_steps:
        samples = trajs.get(step, [])
        if not samples:
            continue
        teachers = [s for s in samples if s["is_teacher"]]
        onpolicy = [s for s in samples if not s["is_teacher"]]

        t_adv = [s["adv_mean"] for s in teachers if s["adv_mean"] is not None]
        o_adv = [s["adv_mean"] for s in onpolicy if s["adv_mean"] is not None]
        t_rew = [s["reward_sum"] for s in teachers if s["reward_sum"] is not None]
        o_rew = [s["reward_sum"] for s in onpolicy if s["reward_sum"] is not None]
        o_success = sum(1 for s in onpolicy if s["success"]) / max(len(onpolicy), 1)

        # SC analysis for DUET runs
        sc_prog = [s["sc_progress"] for s in onpolicy if s.get("sc_progress") is not None]
        sc_bonus = [s["sc_bonus"] for s in onpolicy if s.get("sc_bonus") is not None]

        print(f"  Step {step}: {len(samples)} samples ({len(teachers)} teacher, {len(onpolicy)} onpolicy)")
        print(f"    Teacher adv:  mean={np.mean(t_adv):.4f}, std={np.std(t_adv):.4f}, min={np.min(t_adv):.4f}, max={np.max(t_adv):.4f}" if t_adv else "    Teacher adv:  N/A")
        print(f"    Onpolicy adv: mean={np.mean(o_adv):.4f}, std={np.std(o_adv):.4f}, min={np.min(o_adv):.4f}, max={np.max(o_adv):.4f}" if o_adv else "    Onpolicy adv: N/A")
        print(f"    Teacher rew:  mean={np.mean(t_rew):.4f}" if t_rew else "    Teacher rew:  N/A")
        print(f"    Onpolicy rew: mean={np.mean(o_rew):.4f}, success={o_success:.2%}" if o_rew else "    Onpolicy rew: N/A")
        if sc_prog:
            print(f"    SC progress:  mean={np.mean(sc_prog):.4f}, sc_bonus: mean={np.mean(sc_bonus):.4f}")


# ============================================================
# 6. TEACHER ADVANTAGE MAGNITUDE COMPARISON (THE KEY INSIGHT)
# ============================================================
print("\n" + "=" * 80)
print("6. TEACHER ADVANTAGE MAGNITUDE — THE KEY COMPARISON")
print("=" * 80)

# LUFFY uses teacher old_log_prob as importance weight -> huge advantages
# DUET uses DR3 w_hat to correct -> small advantages
# Let's compute the ratio

print("\nTeacher advantage token_mean ratio (LUFFY/DUET):")
for step in checkpoints:
    luffy_v = all_diags["luffy"].get(step, {}).get("diag/adv_teacher_token_mean")
    d0402_v = all_diags["duet_0402"].get(step, {}).get("diag/adv_teacher_token_mean")
    dorig_v = all_diags["duet_orig"].get(step, {}).get("diag/adv_teacher_token_mean")
    if luffy_v is not None and d0402_v is not None and abs(d0402_v) > 1e-10:
        ratio_0402 = luffy_v / d0402_v if abs(d0402_v) > 1e-10 else float('inf')
        ratio_orig = luffy_v / dorig_v if dorig_v and abs(dorig_v) > 1e-10 else float('inf')
        print(f"  Step {step:3d}: LUFFY={luffy_v:.2f}, 0402={d0402_v:.4f}, orig={dorig_v:.4f} | LUFFY/0402={ratio_0402:.1f}x, LUFFY/orig={ratio_orig:.1f}x")


# ============================================================
# 7. TEACHER SAMPLE OLD_LOG_PROB ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("7. TEACHER old_log_prob vs teacher_old_logp (importance weight proxy)")
print("=" * 80)

for name in ["duet_0402", "duet_orig", "luffy"]:
    print(f"\n--- {name} ---")
    trajs = load_trajectory_diags(RUNS[name], steps=[1, 50, 100])
    for step in [1, 50, 100]:
        samples = trajs.get(step, [])
        if not samples:
            continue
        teachers = [s for s in samples if s["is_teacher"]]
        if not teachers:
            continue
        olps = [s["old_log_prob_mean"] for s in teachers if s["old_log_prob_mean"] is not None]
        tolps = [s["teacher_old_logp_mean"] for s in teachers if s["teacher_old_logp_mean"] is not None]
        print(f"  Step {step}: {len(teachers)} teachers")
        if olps:
            print(f"    old_log_prob_mean:     mean={np.mean(olps):.4f}, std={np.std(olps):.4f}")
        if tolps:
            print(f"    teacher_old_logp_mean: mean={np.mean(tolps):.4f}, std={np.std(tolps):.4f}")
            # log ratio = teacher_old_logp - old_log_prob (IS weight in log space)
            if olps:
                log_ratios = [t - o for t, o in zip(tolps, olps)]
                print(f"    log IS ratio (teacher-current): mean={np.mean(log_ratios):.4f}, std={np.std(log_ratios):.4f}")


# ============================================================
# 8. VALIDATION FAILURE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("8. VALIDATION FAILURE ANALYSIS (step 100)")
print("=" * 80)

for name, val_dir in VAL_LOGS.items():
    results = load_validation_log(val_dir, step=100)
    if not results:
        results = load_validation_log(val_dir, step=50)
        if results:
            print(f"\n--- {name} (step 50, no step 100) ---")
        else:
            print(f"\n--- {name}: no validation data ---")
            continue
    else:
        print(f"\n--- {name} (step 100) ---")

    rewards = []
    for r in results:
        rew = r.get("reward", {})
        if isinstance(rew, dict):
            rewards.append(rew.get("outcome", 0))
        elif isinstance(rew, (int, float)):
            rewards.append(rew)

    n = len(rewards)
    perfect = sum(1 for r in rewards if r >= 0.999)
    high_partial = sum(1 for r in rewards if 0.5 < r < 0.999)
    low_partial = sum(1 for r in rewards if 0.001 < r <= 0.5)
    zero = sum(1 for r in rewards if r <= 0.001)

    print(f"  Total: {n}")
    print(f"  Perfect (r=1.0):   {perfect:3d} ({perfect/n:.1%})")
    print(f"  High (0.5<r<1.0):  {high_partial:3d} ({high_partial/n:.1%})")
    print(f"  Low (0<r≤0.5):     {low_partial:3d} ({low_partial/n:.1%})")
    print(f"  Zero (r=0):        {zero:3d} ({zero/n:.1%})")
    print(f"  Mean reward:       {np.mean(rewards):.4f}")
    print(f"  Median reward:     {np.median(rewards):.4f}")


# Also check step 50 for comparison
print("\n--- Validation at step 50 ---")
for name, val_dir in VAL_LOGS.items():
    results = load_validation_log(val_dir, step=50)
    if not results:
        continue
    rewards = []
    for r in results:
        rew = r.get("reward", {})
        if isinstance(rew, dict):
            rewards.append(rew.get("outcome", 0))
        elif isinstance(rew, (int, float)):
            rewards.append(rew)
    if rewards:
        print(f"  {name}: mean={np.mean(rewards):.4f}, n={len(rewards)}")


# ============================================================
# 9. COMPUTE EFFECTIVE TEACHER GRADIENT SHARE FROM ADVANTAGES
# ============================================================
print("\n" + "=" * 80)
print("9. EFFECTIVE TEACHER GRADIENT SHARE (proxy from advantages)")
print("=" * 80)

# Teacher gradient share ~ |teacher_adv| * teacher_token_ratio / (|teacher_adv| * teacher_token_ratio + |onpolicy_adv| * (1-teacher_token_ratio))
# This is an approximation; actual gradient share also depends on loss clipping, etc.
print(f"\n{'Step':>5} | {'DUET 0402':>12} | {'DUET orig':>12} | {'LUFFY':>12} | Proxy teacher gradient share")
print("-" * 70)
for step in checkpoints:
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        d = all_diags[name].get(step, {})
        t_adv = abs(d.get("diag/adv_teacher_token_mean", 0))
        o_adv = abs(d.get("diag/adv_onpolicy_token_mean", 0))
        t_ratio = d.get("diag/teacher_token_ratio", 0.02)

        t_grad = t_adv * t_ratio
        o_grad = o_adv * (1 - t_ratio)
        total = t_grad + o_grad
        share = t_grad / total if total > 0 else 0
        vals.append(f"{share:.1%}")
    print(f"{step:>5} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12}")


# ============================================================
# 10. FULL TIMESERIES DUMP FOR PLOTTING
# ============================================================
print("\n" + "=" * 80)
print("10. FULL TIMESERIES (every 5 steps) — reward_onpolicy_mean")
print("=" * 80)

all_steps = set()
for name in RUNS:
    all_steps.update(all_diags[name].keys())
all_steps = sorted(all_steps)

# Print every 5 steps
print(f"{'Step':>5} | {'DUET 0402':>12} | {'DUET orig':>12} | {'LUFFY':>12}")
print("-" * 50)
for step in all_steps:
    if step % 5 != 0 and step != 1:
        continue
    vals = []
    for name in ["duet_0402", "duet_orig", "luffy"]:
        v = all_diags[name].get(step, {}).get("diag/reward_onpolicy_mean")
        vals.append(f"{v:.4f}" if v is not None else "N/A")
    print(f"{step:>5} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12}")


# ============================================================
# 11. SC (STATE CHANNEL) BONUS ANALYSIS FOR DUET RUNS
# ============================================================
print("\n" + "=" * 80)
print("11. STATE CHANNEL BONUS ANALYSIS")
print("=" * 80)

for name in ["duet_0402", "duet_orig"]:
    print(f"\n--- {name} ---")
    trajs = load_trajectory_diags(RUNS[name], steps=sample_steps)
    for step in sample_steps:
        samples = trajs.get(step, [])
        if not samples:
            continue
        onpolicy = [s for s in samples if not s["is_teacher"]]
        sc_bonus_vals = [s["sc_bonus"] for s in onpolicy if s.get("sc_bonus") is not None]
        sc_prog_vals = [s["sc_progress"] for s in onpolicy if s.get("sc_progress") is not None]
        rew_orig = [s["reward_original"] for s in onpolicy if s.get("reward_original") is not None]
        rew_sum = [s["reward_sum"] for s in onpolicy if s.get("reward_sum") is not None]

        if sc_bonus_vals:
            bonus_ratio = np.mean(sc_bonus_vals) / max(np.mean(rew_orig), 1e-6) if rew_orig else 0
            print(f"  Step {step}: sc_bonus={np.mean(sc_bonus_vals):.4f}±{np.std(sc_bonus_vals):.4f}, "
                  f"sc_progress={np.mean(sc_prog_vals):.4f}, "
                  f"rew_orig={np.mean(rew_orig):.4f}, rew_total={np.mean(rew_sum):.4f}, "
                  f"bonus/reward={bonus_ratio:.2%}")


# ============================================================
# 12. DETAILED FAILURE COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("12. DETAILED FAILURE COMPARISON (step 100 validation)")
print("=" * 80)

val_data = {}
for name, val_dir in VAL_LOGS.items():
    results = load_validation_log(val_dir, step=100)
    if results:
        val_data[name] = results

if len(val_data) >= 2:
    # Map by task_id
    for name in val_data:
        print(f"\n--- {name}: sample validation entries ---")
        zero_rew = [r for r in val_data[name] if r.get("reward", {}).get("outcome", 0) <= 0.001]
        print(f"  Zero-reward samples: {len(zero_rew)}")
        for r in zero_rew[:5]:
            task = r.get("task_id", r.get("data_id", "?"))
            rew = r.get("reward", {})
            desc = rew.get("description", "")[:100] if isinstance(rew, dict) else ""
            messages = r.get("messages", [])
            n_turns = len(messages) if messages else 0
            print(f"    task={task}, turns={n_turns}, desc={desc}")


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
