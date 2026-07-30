#!/usr/bin/env python3
"""
Track option selection ability over training for DUET vs CHORD.
Hypothesis: CHORD's SFT loss on teacher data teaches option selection faster.
"""

import json
import re
import os
from collections import defaultdict

BASE = "/data/home/qisheng/EvolAnalsis"

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def is_option_click(action):
    if not action.startswith("click["):
        return False
    value = action[6:-1] if action.endswith("]") else action[6:]
    if value.lower() in ["buy now", "back to search", "< prev", "next >"]:
        return False
    if re.match(r'^[bB][0-9a-zA-Z]{9}$', value):
        return False
    return True

def count_options_in_training(path):
    """Count unique option clicks per on-policy trajectory."""
    data = load_jsonl(path)
    on_counts = []
    for entry in data:
        is_teacher = entry.get("diag", {}).get("is_teacher", False)
        if is_teacher:
            continue
        msgs = entry.get("messages", [])
        opts = set()
        for m in msgs:
            if m["role"] == "assistant":
                act_match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
                if act_match:
                    action = act_match.group(1).strip()
                    if is_option_click(action):
                        opts.add(action)
        on_counts.append(len(opts))
    return on_counts

def count_options_in_validation(path):
    """Count unique option clicks per validation trajectory."""
    data = load_jsonl(path)
    counts = []
    for entry in data:
        output = entry.get("output", "")
        actions = []
        parts = output.split("assistant\n")
        for part in parts[1:]:
            if "<action>" in part and "</action>" in part:
                act_start = part.index("<action>") + len("<action>")
                act_end = part.index("</action>")
                actions.append(part[act_start:act_end].strip())

        opts = set(a for a in actions if is_option_click(a))
        counts.append(len(opts))
    return counts

print("=" * 80)
print("  OPTION SELECTION ABILITY OVER TRAINING TIME")
print("=" * 80)

# Training trajectories over steps
steps_to_check = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
methods = {
    "DUET_v1": f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_duet/Trajectory/trajectories_step_{{step}}.jsonl",
    "CHORD":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_chord/Trajectory/trajectories_step_{{step}}.jsonl",
    "LUFFY":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_luffy/Trajectory/trajectories_step_{{step}}.jsonl",
}

print("\n--- On-Policy: Average Unique Options Per Trajectory ---")
print(f"{'Step':>5}", end="")
for method in methods:
    print(f"  {method:>10}", end="")
print()
print("-" * 45)

for step in steps_to_check:
    print(f"{step:>5}", end="")
    for method, template in methods.items():
        path = template.format(step=step)
        if os.path.exists(path):
            counts = count_options_in_training(path)
            avg = sum(counts)/len(counts) if counts else 0
            print(f"  {avg:>10.2f}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

# Also track buy completion rate over time
print("\n\n--- On-Policy: Buy Completion Rate Over Steps ---")
print(f"{'Step':>5}", end="")
for method in methods:
    print(f"  {method:>10}", end="")
print()
print("-" * 45)

for step in steps_to_check:
    print(f"{step:>5}", end="")
    for method, template in methods.items():
        path = template.format(step=step)
        if os.path.exists(path):
            data = load_jsonl(path)
            buy_count = 0
            total = 0
            for entry in data:
                is_teacher = entry.get("diag", {}).get("is_teacher", False)
                if is_teacher:
                    continue
                total += 1
                msgs = entry.get("messages", [])
                for m in msgs:
                    if m["role"] == "assistant":
                        act_match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
                        if act_match and act_match.group(1).strip() == "click[buy now]":
                            buy_count += 1
                            break
            rate = buy_count / total * 100 if total > 0 else 0
            print(f"  {rate:>9.1f}%", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

# On-policy training reward over time
print("\n\n--- On-Policy: Average Training Reward Over Steps ---")
print(f"{'Step':>5}", end="")
for method in methods:
    print(f"  {method:>10}", end="")
print()
print("-" * 45)

for step in steps_to_check:
    print(f"{step:>5}", end="")
    for method, template in methods.items():
        path = template.format(step=step)
        if os.path.exists(path):
            data = load_jsonl(path)
            scores = []
            for entry in data:
                is_teacher = entry.get("diag", {}).get("is_teacher", False)
                if is_teacher:
                    continue
                reward = entry.get("reward", {})
                outcome = reward.get("outcome", 0) if isinstance(reward, dict) else reward
                scores.append(outcome)
            avg = sum(scores)/len(scores) if scores else 0
            print(f"  {avg:>10.4f}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
    print()

# Validation option selection at step 50 vs step 100
print("\n\n--- Validation: Option Selection at Step 50 vs Step 100 ---")
val_methods = {
    "DUET_v1": (f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/50.jsonl",
                f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/100.jsonl"),
    "CHORD":   (f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/50.jsonl",
                f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl"),
    "LUFFY":   (f"{BASE}/experiments/webshop/webshop_qwen1.5b_luffy/validation_log/50.jsonl",
                f"{BASE}/experiments/webshop/webshop_qwen1.5b_luffy/validation_log/100.jsonl"),
}

for method, (p50, p100) in val_methods.items():
    c50 = count_options_in_validation(p50) if os.path.exists(p50) else []
    c100 = count_options_in_validation(p100) if os.path.exists(p100) else []

    avg50 = sum(c50)/len(c50) if c50 else 0
    avg100 = sum(c100)/len(c100) if c100 else 0

    # Count trajectories with >= 1 option
    has50 = sum(1 for c in c50 if c > 0)
    has100 = sum(1 for c in c100 if c > 0)

    print(f"  {method}:")
    print(f"    Step 50:  avg opts={avg50:.2f}, {has50}/{len(c50)} trajs with >=1 option")
    print(f"    Step 100: avg opts={avg100:.2f}, {has100}/{len(c100)} trajs with >=1 option")

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
