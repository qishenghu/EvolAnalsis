#!/usr/bin/env python3
"""
Deep dive into the option repetition loop problem in DUET vs CHORD.
The main analysis revealed DUET gets stuck clicking the same option repeatedly.
This script quantifies the loop frequency and characterizes the stuck-on patterns.
"""

import json
import re
from collections import Counter, defaultdict

BASE = "/data/home/qisheng/EvolAnalsis"

VAL_PATHS = {
    "DUET_v1": f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/100.jsonl",
    "DUET_v2": f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v2/validation_log/100.jsonl",
    "CHORD":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl",
    "LUFFY":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_luffy/validation_log/100.jsonl",
}

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def parse_actions(entry):
    output = entry.get("output", "")
    actions = []
    parts = output.split("assistant\n")
    for part in parts[1:]:
        if "<action>" in part and "</action>" in part:
            act_start = part.index("<action>") + len("<action>")
            act_end = part.index("</action>")
            actions.append(part[act_start:act_end].strip())
        elif "<action>" in part:
            act_start = part.index("<action>") + len("<action>")
            actions.append(part[act_start:].strip()[:200])
    return actions

def detect_option_loop(actions, threshold=3):
    """Detect if agent is stuck clicking the same option >= threshold times consecutively."""
    if len(actions) < threshold:
        return None, 0

    max_run = 1
    max_action = None
    cur_run = 1

    for i in range(1, len(actions)):
        if actions[i] == actions[i-1] and actions[i].startswith("click[") and not actions[i].startswith("click[buy"):
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
                max_action = actions[i]
        else:
            cur_run = 1

    if max_run >= threshold:
        return max_action, max_run
    return None, 0

def extract_instruction_from_output(output):
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)(?:\[SEP\])", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

print("=" * 80)
print("  OPTION REPETITION LOOP ANALYSIS")
print("=" * 80)

# 1. Quantify loop frequency across methods
print("\n--- Loop Frequency (consecutive same-option clicks >= 3) ---")
print(f"{'Method':<12} {'LoopTraj':>10} {'Total':>7} {'LoopRate':>9} {'AvgLoopLen':>11} {'MaxLoop':>8}")
print("-" * 67)

for method, path in VAL_PATHS.items():
    data = load_jsonl(path)
    loop_count = 0
    loop_lengths = []
    loop_actions_counter = Counter()

    for entry in data:
        actions = parse_actions(entry)
        loop_action, loop_len = detect_option_loop(actions)
        if loop_action:
            loop_count += 1
            loop_lengths.append(loop_len)
            loop_actions_counter[loop_action] += 1

    avg_loop = sum(loop_lengths)/len(loop_lengths) if loop_lengths else 0
    max_loop = max(loop_lengths) if loop_lengths else 0
    print(f"{method:<12} {loop_count:>10} {len(data):>7} {loop_count/len(data)*100:>8.1f}% {avg_loop:>11.1f} {max_loop:>8}")

    if loop_actions_counter:
        print(f"  Top stuck-on options: {loop_actions_counter.most_common(5)}")

# 2. What option does DUET get stuck on?
print("\n\n--- DUET_v1: Detailed Loop Analysis ---")
data = load_jsonl(VAL_PATHS["DUET_v1"])

loop_tasks = []
for i, entry in enumerate(data):
    actions = parse_actions(entry)
    loop_action, loop_len = detect_option_loop(actions)
    if loop_action:
        instr = extract_instruction_from_output(entry.get("output", ""))
        loop_tasks.append({
            "idx": i,
            "score": entry["score"],
            "loop_action": loop_action,
            "loop_len": loop_len,
            "total_actions": len(actions),
            "has_buy": any(a == "click[buy now]" for a in actions),
            "instruction": instr[:120],
        })

for lt in sorted(loop_tasks, key=lambda x: -x["loop_len"])[:10]:
    print(f"  Task {lt['idx']}: score={lt['score']:.3f}, loop={lt['loop_action']} x{lt['loop_len']}, "
          f"total_acts={lt['total_actions']}, has_buy={lt['has_buy']}")
    print(f"    Instr: {lt['instruction']}")

# 3. Does the loop cause buy-failure, or does the agent eventually buy?
print("\n\n--- Loop vs Buy Completion ---")
for method in ["DUET_v1", "DUET_v2", "CHORD", "LUFFY"]:
    data = load_jsonl(VAL_PATHS[method])

    loop_with_buy = 0
    loop_no_buy = 0
    no_loop_with_buy = 0
    no_loop_no_buy = 0

    loop_buy_scores = []
    loop_nobuy_scores = []
    noloop_buy_scores = []

    for entry in data:
        actions = parse_actions(entry)
        loop_action, loop_len = detect_option_loop(actions)
        has_buy = any(a == "click[buy now]" for a in actions)

        if loop_action:
            if has_buy:
                loop_with_buy += 1
                loop_buy_scores.append(entry["score"])
            else:
                loop_no_buy += 1
                loop_nobuy_scores.append(entry["score"])
        else:
            if has_buy:
                no_loop_with_buy += 1
                noloop_buy_scores.append(entry["score"])
            else:
                no_loop_no_buy += 1

    print(f"\n  {method}:")
    print(f"    Loop + Buy:       {loop_with_buy:>4}  (avg score: {sum(loop_buy_scores)/max(len(loop_buy_scores),1):.3f})")
    print(f"    Loop + No Buy:    {loop_no_buy:>4}  (avg score: {sum(loop_nobuy_scores)/max(len(loop_nobuy_scores),1):.3f})")
    print(f"    No Loop + Buy:    {no_loop_with_buy:>4}  (avg score: {sum(noloop_buy_scores)/max(len(noloop_buy_scores),1):.3f})")
    print(f"    No Loop + No Buy: {no_loop_no_buy:>4}")

# 4. How many options does each method select? (unique option clicks before buy)
print("\n\n--- Option Selection Depth ---")
for method in ["DUET_v1", "DUET_v2", "CHORD", "LUFFY"]:
    data = load_jsonl(VAL_PATHS[method])
    unique_options_counts = []
    total_option_clicks = []

    for entry in data:
        actions = parse_actions(entry)
        option_clicks = [a for a in actions if a.startswith("click[") and
                        not a.startswith("click[buy") and not a.startswith("click[B") and
                        not a.startswith("click[b") and not a.startswith("click[Back") and
                        not a.startswith("click[< ") and not a.startswith("click[Next")]
        unique_opts = set(option_clicks)
        unique_options_counts.append(len(unique_opts))
        total_option_clicks.append(len(option_clicks))

    avg_unique = sum(unique_options_counts) / len(unique_options_counts)
    avg_total = sum(total_option_clicks) / len(total_option_clicks)
    print(f"  {method}: avg unique options={avg_unique:.1f}, avg total option clicks={avg_total:.1f}, ratio={avg_unique/max(avg_total,0.01):.2f}")

# 5. CHORD option selection quality: does CHORD select MORE options correctly?
print("\n\n--- CHORD vs DUET: Score by Option Count ---")
for method in ["DUET_v1", "CHORD"]:
    data = load_jsonl(VAL_PATHS[method])
    by_opts = defaultdict(list)

    for entry in data:
        actions = parse_actions(entry)
        option_clicks = [a for a in actions if a.startswith("click[") and
                        not a.startswith("click[buy") and not a.startswith("click[B") and
                        not a.startswith("click[b") and not a.startswith("click[Back") and
                        not a.startswith("click[< ") and not a.startswith("click[Next")]
        unique_opts = len(set(option_clicks))
        by_opts[unique_opts].append(entry["score"])

    print(f"\n  {method}:")
    for n_opts in sorted(by_opts.keys()):
        scores = by_opts[n_opts]
        avg = sum(scores)/len(scores)
        print(f"    {n_opts} unique options: n={len(scores)}, avg_score={avg:.3f}")

# 6. The ACTUAL behavioral gap: count of unique options selected on product detail page
print("\n\n--- Head-to-Head: Unique Options on Same Tasks ---")
data_d = load_jsonl(VAL_PATHS["DUET_v1"])
data_c = load_jsonl(VAL_PATHS["CHORD"])

duet_more_opts = 0
chord_more_opts = 0
same_opts = 0
chord_better_more_opts = 0
chord_worse_more_opts = 0

for i in range(min(len(data_d), len(data_c))):
    acts_d = parse_actions(data_d[i])
    acts_c = parse_actions(data_c[i])

    opts_d = set(a for a in acts_d if a.startswith("click[") and
                not a.startswith("click[buy") and not a.startswith("click[B") and
                not a.startswith("click[b") and not a.startswith("click[Back") and
                not a.startswith("click[< ") and not a.startswith("click[Next"))
    opts_c = set(a for a in acts_c if a.startswith("click[") and
                not a.startswith("click[buy") and not a.startswith("click[B") and
                not a.startswith("click[b") and not a.startswith("click[Back") and
                not a.startswith("click[< ") and not a.startswith("click[Next"))

    gap = data_c[i]["score"] - data_d[i]["score"]

    if len(opts_c) > len(opts_d):
        chord_more_opts += 1
        if gap > 0.05:
            chord_better_more_opts += 1
    elif len(opts_d) > len(opts_c):
        duet_more_opts += 1
        if gap > 0.05:
            chord_worse_more_opts += 1
    else:
        same_opts += 1

print(f"  CHORD selects more unique options: {chord_more_opts}/200")
print(f"    Of which CHORD also scores higher: {chord_better_more_opts}")
print(f"  DUET selects more unique options: {duet_more_opts}/200")
print(f"    Of which CHORD still scores higher: {chord_worse_more_opts}")
print(f"  Same number of options: {same_opts}/200")

# 7. The DUET_v2 option collapse: v2 selects almost NO options
print("\n\n--- DUET_v2 Option Collapse ---")
data_v2 = load_jsonl(VAL_PATHS["DUET_v2"])
zero_option_tasks = 0
for entry in data_v2:
    actions = parse_actions(entry)
    option_clicks = [a for a in actions if a.startswith("click[") and
                    not a.startswith("click[buy") and not a.startswith("click[B") and
                    not a.startswith("click[b") and not a.startswith("click[Back") and
                    not a.startswith("click[< ") and not a.startswith("click[Next")]
    if len(option_clicks) == 0:
        zero_option_tasks += 1

print(f"  DUET_v2 tasks with ZERO option clicks: {zero_option_tasks}/200 ({zero_option_tasks/2:.1f}%)")

# Compare typical DUET_v2 trajectory pattern
print(f"\n  Sample DUET_v2 trajectories (no option clicks):")
shown = 0
for i, entry in enumerate(data_v2):
    actions = parse_actions(entry)
    option_clicks = [a for a in actions if a.startswith("click[") and
                    not a.startswith("click[buy") and not a.startswith("click[B") and
                    not a.startswith("click[b") and not a.startswith("click[Back") and
                    not a.startswith("click[< ") and not a.startswith("click[Next")]
    if len(option_clicks) == 0 and len(actions) >= 2:
        print(f"    Task {i} (score={entry['score']:.3f}): {actions}")
        shown += 1
        if shown >= 5:
            break

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
