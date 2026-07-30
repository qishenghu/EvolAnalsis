#!/usr/bin/env python3
"""
Final piece: What EXACTLY is different about option selection?
CHORD selects 1.1 unique options avg vs DUET 0.5. But is option selection driving the score gap?
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

def is_option_click(action):
    if not action.startswith("click["):
        return False
    value = action[6:-1] if action.endswith("]") else action[6:]
    # Not a product ASIN, not buy, not navigation
    if value.lower() in ["buy now", "back to search", "< prev", "next >"]:
        return False
    if re.match(r'^[bB][0-9a-zA-Z]{9}$', value):
        return False
    return True

def extract_instruction_from_output(output):
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)(?:\[SEP\])", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_required_attributes(instruction):
    """Extract color, size, fit, price from instruction."""
    attrs = {}
    color_match = re.search(r'color:\s*(.*?)(?:,\s*and|\s*$)', instruction)
    if color_match:
        attrs['color'] = color_match.group(1).strip()
    size_match = re.search(r'size:\s*(.*?)(?:,\s*and|\s*$)', instruction)
    if size_match:
        attrs['size'] = size_match.group(1).strip()
    return attrs

# 1. Decompose score gap into components
print("=" * 80)
print("  SCORE GAP DECOMPOSITION: CHORD vs DUET")
print("=" * 80)

data_d = load_jsonl(VAL_PATHS["DUET_v1"])
data_c = load_jsonl(VAL_PATHS["CHORD"])

# Categorize by behavior pattern
categories = defaultdict(lambda: {"d_scores": [], "c_scores": [], "count": 0, "d_actions": [], "c_actions": []})

for i in range(200):
    acts_d = parse_actions(data_d[i])
    acts_c = parse_actions(data_c[i])
    sd = data_d[i]["score"]
    sc = data_c[i]["score"]

    # Classify the behavioral difference
    d_opts = [a for a in acts_d if is_option_click(a)]
    c_opts = [a for a in acts_c if is_option_click(a)]
    d_unique_opts = set(d_opts)
    c_unique_opts = set(c_opts)
    d_has_buy = any(a == "click[buy now]" for a in acts_d)
    c_has_buy = any(a == "click[buy now]" for a in acts_c)
    d_prods = [a for a in acts_d if re.match(r'click\[[bB][0-9a-zA-Z]{9}\]', a)]
    c_prods = [a for a in acts_c if re.match(r'click\[[bB][0-9a-zA-Z]{9}\]', a)]

    # Determine behavior category
    if not d_has_buy and c_has_buy:
        cat = "DUET_NO_BUY_CHORD_BUYS"
    elif d_has_buy and not c_has_buy:
        cat = "DUET_BUYS_CHORD_NO_BUY"
    elif len(c_unique_opts) > len(d_unique_opts):
        cat = "CHORD_MORE_OPTIONS"
    elif len(d_unique_opts) > len(c_unique_opts):
        cat = "DUET_MORE_OPTIONS"
    elif d_prods and c_prods and d_prods[0].lower() != c_prods[0].lower():
        cat = "DIFFERENT_PRODUCT"
    else:
        cat = "SIMILAR_BEHAVIOR"

    categories[cat]["d_scores"].append(sd)
    categories[cat]["c_scores"].append(sc)
    categories[cat]["count"] += 1

print(f"\n{'Category':<30} {'N':>4} {'DUET_avg':>9} {'CHORD_avg':>10} {'Gap':>8} {'% of Total Gap':>14}")
print("-" * 85)

total_gap = sum(data_c[i]["score"] - data_d[i]["score"] for i in range(200))
for cat in sorted(categories.keys()):
    info = categories[cat]
    d_avg = sum(info["d_scores"]) / info["count"]
    c_avg = sum(info["c_scores"]) / info["count"]
    cat_gap = sum(c - d for c, d in zip(info["c_scores"], info["d_scores"]))
    pct = cat_gap / total_gap * 100 if total_gap != 0 else 0
    print(f"{cat:<30} {info['count']:>4} {d_avg:>9.3f} {c_avg:>10.3f} {c_avg-d_avg:>+8.3f} {pct:>13.1f}%")

print(f"\nTotal gap across all 200 tasks: {total_gap:+.2f} ({total_gap/200:+.4f} avg)")

# 2. For CHORD_MORE_OPTIONS category, what options does CHORD select that DUET doesn't?
print("\n\n--- CHORD_MORE_OPTIONS: What options does CHORD select? ---")
cat_data = []
for i in range(200):
    acts_d = parse_actions(data_d[i])
    acts_c = parse_actions(data_c[i])

    d_opts = set(a for a in acts_d if is_option_click(a))
    c_opts = set(a for a in acts_c if is_option_click(a))

    if len(c_opts) > len(d_opts):
        extra = c_opts - d_opts
        instr = extract_instruction_from_output(data_c[i].get("output", ""))
        attrs = extract_required_attributes(instr)
        cat_data.append({
            "idx": i,
            "d_score": data_d[i]["score"],
            "c_score": data_c[i]["score"],
            "d_opts": d_opts,
            "c_opts": c_opts,
            "extra_opts": extra,
            "instruction": instr[:150],
            "attrs": attrs,
        })

# Sort by gap
cat_data.sort(key=lambda x: x["c_score"] - x["d_score"], reverse=True)

print(f"Found {len(cat_data)} tasks where CHORD selects more options than DUET\n")
for cd in cat_data[:10]:
    print(f"  Task {cd['idx']}: DUET={cd['d_score']:.3f}, CHORD={cd['c_score']:.3f}")
    print(f"    DUET opts:  {cd['d_opts']}")
    print(f"    CHORD opts: {cd['c_opts']}")
    print(f"    CHORD extra: {cd['extra_opts']}")
    if cd['attrs']:
        print(f"    Required: {cd['attrs']}")
    print()

# 3. Option-to-instruction attribute matching
print("\n--- Option-Instruction Attribute Matching ---")
for method_name, path in [("DUET_v1", VAL_PATHS["DUET_v1"]), ("CHORD", VAL_PATHS["CHORD"])]:
    data = load_jsonl(path)
    color_match = 0
    size_match = 0
    color_total = 0
    size_total = 0

    for entry in data:
        instr = extract_instruction_from_output(entry.get("output", ""))
        attrs = extract_required_attributes(instr)
        acts = parse_actions(entry)
        opts = [a[6:-1] if a.endswith("]") else a[6:] for a in acts if is_option_click(a)]

        if "color" in attrs:
            color_total += 1
            color_val = attrs["color"].lower()
            if any(color_val in o.lower() or o.lower() in color_val for o in opts):
                color_match += 1

        if "size" in attrs:
            size_total += 1
            size_val = attrs["size"].lower()
            if any(size_val in o.lower() or o.lower() in size_val for o in opts):
                size_match += 1

    print(f"\n  {method_name}:")
    if color_total > 0:
        print(f"    Color selected: {color_match}/{color_total} ({color_match/color_total*100:.1f}%)")
    if size_total > 0:
        print(f"    Size selected:  {size_match}/{size_total} ({size_match/size_total*100:.1f}%)")

# 4. Training: how does CHORD's SFT loss help option selection? Look at training trajectories.
print("\n\n--- Training Step 100: Option Selection by Method ---")
TRAIN_PATHS = {
    "DUET_v1": f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_duet/Trajectory/trajectories_step_100.jsonl",
    "CHORD":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_chord/Trajectory/trajectories_step_100.jsonl",
    "LUFFY":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_luffy/Trajectory/trajectories_step_100.jsonl",
}

for method, path in TRAIN_PATHS.items():
    data = load_jsonl(path)
    on_policy_option_counts = []
    teacher_option_counts = []

    for entry in data:
        is_teacher = entry.get("diag", {}).get("is_teacher", False)
        msgs = entry.get("messages", [])
        opts = []
        for m in msgs:
            if m["role"] == "assistant":
                act_match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
                if act_match:
                    action = act_match.group(1).strip()
                    if is_option_click(action):
                        opts.append(action)

        if is_teacher:
            teacher_option_counts.append(len(set(opts)))
        else:
            on_policy_option_counts.append(len(set(opts)))

    on_avg = sum(on_policy_option_counts)/max(len(on_policy_option_counts), 1)
    t_avg = sum(teacher_option_counts)/max(len(teacher_option_counts), 1)
    print(f"  {method}: on-policy avg unique opts={on_avg:.1f} (n={len(on_policy_option_counts)}), "
          f"teacher avg={t_avg:.1f} (n={len(teacher_option_counts)})")

# 5. Score at step 50: was the gap already present?
print("\n\n--- Score Gap at Step 50 vs Step 100 ---")
VAL_50 = {
    "DUET_v1": f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/50.jsonl",
    "CHORD":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/50.jsonl",
}

for step_label, paths in [("Step 50", VAL_50), ("Step 100", {"DUET_v1": VAL_PATHS["DUET_v1"], "CHORD": VAL_PATHS["CHORD"]})]:
    data_d = load_jsonl(paths["DUET_v1"])
    data_c = load_jsonl(paths["CHORD"])

    d_opt_counts = []
    c_opt_counts = []

    for entry in data_d:
        acts = parse_actions(entry)
        opts = set(a for a in acts if is_option_click(a))
        d_opt_counts.append(len(opts))

    for entry in data_c:
        acts = parse_actions(entry)
        opts = set(a for a in acts if is_option_click(a))
        c_opt_counts.append(len(opts))

    d_avg = sum(d_opt_counts)/len(d_opt_counts)
    c_avg = sum(c_opt_counts)/len(c_opt_counts)
    d_score = sum(e["score"] for e in data_d)/len(data_d)
    c_score = sum(e["score"] for e in data_c)/len(data_c)

    print(f"  {step_label}: DUET opts={d_avg:.1f}, CHORD opts={c_avg:.1f} | DUET score={d_score:.4f}, CHORD score={c_score:.4f}")

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
