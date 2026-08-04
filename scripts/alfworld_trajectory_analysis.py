#!/usr/bin/env python3
"""
ALFWorld 7B vs 3B Trajectory-Level Case Analysis
Comprehensive behavioral analysis across methods and scales.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = "/data/home/qisheng/EvolAnalsis"
CKPT = f"{BASE}/checkpoints/agentevolver"
VAL = f"{BASE}/experiments/alfworld"

EXPERIMENTS = {
    # 7B experiments
    "7B-OnPolicy": "alfworld_7b_onpolicy",
    "7B-LUFFY": "alfworld_7b_luffy",
    "7B-CHORD": "alfworld_7b_chord",
    "7B-DUET": "alfworld_7b_duet",
    # 3B experiments
    "3B-OnPolicy": "alfworld_3b_grpo_react_tags",
    "3B-LUFFY": "alfworld_3b_luffy",
    "3B-DUET": "alfworld_3b_duet_0329",
}

# ALFWorld task types extracted from task descriptions
TASK_PATTERNS = {
    "put": r"\bput\b.*\bin\b",
    "clean": r"\bclean\b",
    "heat": r"\bheat\b",
    "cool": r"\bcool\b",
    "examine": r"\bexamine\b.*\bwith\b",
    "puttwo": r"\bput two\b",
}


def classify_task_type(query: str) -> str:
    """Classify ALFWorld task type from query text."""
    q = query.lower()
    if "put two" in q:
        return "puttwo"
    if "clean" in q:
        return "clean"
    if "heat" in q:
        return "heat"
    if "cool" in q:
        return "cool"
    if "examine" in q and "lamp" in q:
        return "examine"
    if "put" in q:
        return "put"
    return "other"


def extract_actions_from_messages(messages: list) -> List[str]:
    """Extract action strings from assistant messages in trajectory format."""
    actions = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        # Extract all <action>...</action> tags
        action_matches = re.findall(r"<action>(.*?)</action>", content, re.DOTALL)
        for a in action_matches:
            actions.append(a.strip())
    return actions


def extract_actions_from_output(output: str) -> List[str]:
    """Extract action strings from validation output format (alternating lines)."""
    actions = []
    action_matches = re.findall(r"<action>(.*?)</action>", output, re.DOTALL)
    for a in action_matches:
        actions.append(a.strip())
    return actions


def count_think_tags(messages_or_output) -> int:
    """Count <think> tags in content."""
    if isinstance(messages_or_output, list):
        text = " ".join(m.get("content", "") for m in messages_or_output)
    else:
        text = messages_or_output
    return len(re.findall(r"<think>", text))


def detect_repetition_loop(actions: List[str], threshold: int = 3) -> bool:
    """Detect if the same action appears consecutively >= threshold times."""
    if len(actions) < threshold:
        return False
    for i in range(len(actions) - threshold + 1):
        if len(set(actions[i : i + threshold])) == 1:
            return True
    return False


def detect_multi_action_tag(messages_or_output) -> bool:
    """Detect if a single assistant turn contains multiple <action> tags."""
    if isinstance(messages_or_output, list):
        for msg in messages_or_output:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            tags = re.findall(r"<action>", content)
            if len(tags) > 1:
                return True
    else:
        # Split by "assistant\n" and check each assistant turn
        turns = re.split(r"\nassistant\n", messages_or_output)
        for turn in turns:
            tags = re.findall(r"<action>", turn)
            if len(tags) > 1:
                return True
    return False


def detect_format_errors(messages_or_output) -> bool:
    """Detect if assistant output lacks proper action tags."""
    if isinstance(messages_or_output, list):
        for msg in messages_or_output:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if content == "OK. I'll follow your instructions and try my best to solve the task.":
                continue
            if "<action>" not in content:
                return True
    return False


def detect_nothing_happened(messages_or_output) -> int:
    """Count 'Nothing happened' responses (invalid actions)."""
    if isinstance(messages_or_output, list):
        text = " ".join(m.get("content", "") for m in messages_or_output if m.get("role") == "user")
    else:
        text = messages_or_output
    return len(re.findall(r"Nothing happened", text, re.IGNORECASE))


def detect_cjk(messages_or_output) -> bool:
    """Detect CJK characters in agent output."""
    if isinstance(messages_or_output, list):
        text = " ".join(m.get("content", "") for m in messages_or_output if m.get("role") == "assistant")
    else:
        text = messages_or_output
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", text))


def load_trajectories(exp_key: str, step: int) -> List[dict]:
    """Load trajectory JSONL for an experiment at a given step."""
    exp_dir = EXPERIMENTS[exp_key]
    path = f"{CKPT}/{exp_dir}/Trajectory/trajectories_step_{step}.jsonl"
    if not os.path.exists(path):
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            try:
                trajs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return trajs


def load_validation(exp_key: str, step: int) -> List[dict]:
    """Load validation JSONL for an experiment at a given step."""
    exp_dir = EXPERIMENTS[exp_key]
    path = f"{VAL}/{exp_dir}/validation_log/{step}.jsonl"
    if not os.path.exists(path):
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            try:
                trajs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return trajs


def analyze_trajectory(traj: dict, source: str = "rollout") -> dict:
    """Analyze a single trajectory and return features."""
    if source == "rollout":
        messages = traj.get("messages", [])
        actions = extract_actions_from_messages(messages)
        query = traj.get("query", "")
        success = traj.get("success", False)
        reward_data = traj.get("reward", {})
        reward = reward_data.get("outcome", 0.0) if isinstance(reward_data, dict) else float(reward_data)
        diag = traj.get("diag", {})
        is_teacher = diag.get("is_teacher", False)
        resp_tokens = diag.get("response_valid_tokens", 0)
        nothing_count = detect_nothing_happened(messages)
        repetition = detect_repetition_loop(actions)
        multi_action = detect_multi_action_tag(messages)
        format_err = detect_format_errors(messages)
        cjk = detect_cjk(messages)
        think_count = count_think_tags(messages)
    else:
        output = traj.get("output", "")
        actions = extract_actions_from_output(output)
        query = traj.get("input", "")
        # Extract task description from input
        if "Your task is to:" in query:
            query = query.split("Your task is to:")[-1].split("AVAILABLE")[0].strip()
        reward = traj.get("reward", 0.0)
        if isinstance(reward, dict):
            reward = reward.get("outcome", 0.0)
        success = reward > 0
        is_teacher = False
        resp_tokens = len(output.split())
        nothing_count = detect_nothing_happened(output)
        repetition = detect_repetition_loop(actions)
        multi_action = detect_multi_action_tag(output)
        format_err = False  # Different detection for validation
        cjk = detect_cjk(output)
        think_count = count_think_tags(output)
        diag = {}

    task_type = classify_task_type(query)

    return {
        "task_id": traj.get("task_id", ""),
        "task_type": task_type,
        "query": query,
        "success": success,
        "reward": reward,
        "num_actions": len(actions),
        "actions": actions,
        "is_teacher": is_teacher,
        "resp_tokens": resp_tokens,
        "nothing_happened_count": nothing_count,
        "repetition_loop": repetition,
        "multi_action_tag": multi_action,
        "format_error": format_err,
        "cjk_detected": cjk,
        "think_count": think_count,
        "diag": diag,
    }


# =============================================================
# SECTION 1: Step-1 Capability Comparison
# =============================================================
def section1_step1_comparison():
    print("\n" + "=" * 80)
    print("SECTION 1: Step-1 Capability Comparison (Pre-Training Baseline)")
    print("=" * 80)

    for exp_key in ["7B-OnPolicy", "3B-OnPolicy", "7B-DUET", "3B-DUET", "7B-LUFFY", "3B-LUFFY"]:
        trajs = load_trajectories(exp_key, 1)
        if not trajs:
            print(f"\n  {exp_key}: No data at step 1")
            continue

        analyses = [analyze_trajectory(t) for t in trajs]
        # Filter out teacher samples for fair comparison
        on_policy = [a for a in analyses if not a["is_teacher"]]
        teacher = [a for a in analyses if a["is_teacher"]]

        print(f"\n--- {exp_key} (step 1) ---")
        print(f"  Total trajectories: {len(analyses)} (on-policy: {len(on_policy)}, teacher: {len(teacher)})")

        if on_policy:
            successes = sum(1 for a in on_policy if a["success"])
            avg_reward = sum(a["reward"] for a in on_policy) / len(on_policy)
            avg_actions = sum(a["num_actions"] for a in on_policy) / len(on_policy)
            avg_tokens = sum(a["resp_tokens"] for a in on_policy) / len(on_policy)
            avg_nothing = sum(a["nothing_happened_count"] for a in on_policy) / len(on_policy)

            print(f"  On-policy success rate: {successes}/{len(on_policy)} = {successes / len(on_policy):.1%}")
            print(f"  On-policy mean reward: {avg_reward:.4f}")
            print(f"  On-policy avg actions: {avg_actions:.1f}")
            print(f"  On-policy avg response tokens: {avg_tokens:.0f}")
            print(f"  On-policy avg 'Nothing happened' per traj: {avg_nothing:.1f}")

            # Task type distribution
            type_counter = Counter(a["task_type"] for a in on_policy)
            print(f"  Task type distribution: {dict(type_counter)}")

            # Success rate by task type
            type_success = defaultdict(list)
            for a in on_policy:
                type_success[a["task_type"]].append(a["success"])
            print("  Success rate by task type:")
            for tt in sorted(type_success.keys()):
                vals = type_success[tt]
                rate = sum(vals) / len(vals)
                print(f"    {tt}: {sum(vals)}/{len(vals)} = {rate:.1%}")

            # Failure mode counts
            rep_loops = sum(1 for a in on_policy if a["repetition_loop"])
            multi_act = sum(1 for a in on_policy if a["multi_action_tag"])
            fmt_err = sum(1 for a in on_policy if a["format_error"])
            cjk = sum(1 for a in on_policy if a["cjk_detected"])
            premature = sum(1 for a in on_policy if a["num_actions"] <= 2)
            print(f"  Failure modes: rep_loops={rep_loops}, multi_action={multi_act}, format_err={fmt_err}, cjk={cjk}, premature(<= 2 actions)={premature}")


# =============================================================
# SECTION 2: Failure Mode Tracking Across Training
# =============================================================
def section2_failure_modes():
    print("\n" + "=" * 80)
    print("SECTION 2: Failure Mode Evolution Across Training Steps")
    print("=" * 80)

    steps = [1, 10, 30, 50, 70, 90, 100]
    methods = ["7B-OnPolicy", "7B-LUFFY", "7B-DUET", "3B-OnPolicy", "3B-LUFFY", "3B-DUET"]

    for method in methods:
        print(f"\n--- {method} ---")
        print(f"  {'Step':>5} | {'N':>4} | {'Succ%':>6} | {'AvgAct':>6} | {'RepLoop':>7} | {'MultiAct':>8} | {'FmtErr':>6} | {'CJK':>3} | {'Premature':>9} | {'NothHapp':>8}")
        print(f"  {'-----':>5}-+-{'----':>4}-+-{'------':>6}-+-{'------':>6}-+-{'-------':>7}-+-{'--------':>8}-+-{'------':>6}-+-{'---':>3}-+-{'---------':>9}-+-{'--------':>8}")

        for step in steps:
            trajs = load_trajectories(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t) for t in trajs]
            on_policy = [a for a in analyses if not a["is_teacher"]]

            if not on_policy:
                continue

            n = len(on_policy)
            succ = sum(1 for a in on_policy if a["success"]) / n * 100
            avg_act = sum(a["num_actions"] for a in on_policy) / n
            rep = sum(1 for a in on_policy if a["repetition_loop"])
            multi = sum(1 for a in on_policy if a["multi_action_tag"])
            fmt = sum(1 for a in on_policy if a["format_error"])
            cjk = sum(1 for a in on_policy if a["cjk_detected"])
            prem = sum(1 for a in on_policy if a["num_actions"] <= 2)
            noth = sum(a["nothing_happened_count"] for a in on_policy) / n

            print(f"  {step:>5} | {n:>4} | {succ:>5.1f}% | {avg_act:>6.1f} | {rep:>7} | {multi:>8} | {fmt:>6} | {cjk:>3} | {prem:>9} | {noth:>8.1f}")


# =============================================================
# SECTION 3: Head-to-Head Task Comparison at Step 50
# =============================================================
def section3_head_to_head():
    print("\n" + "=" * 80)
    print("SECTION 3: Head-to-Head Task Comparison at Step 50")
    print("=" * 80)

    methods_7b = ["7B-OnPolicy", "7B-LUFFY", "7B-DUET"]
    methods_3b = ["3B-OnPolicy", "3B-LUFFY", "3B-DUET"]

    for label, methods in [("7B Methods", methods_7b), ("3B Methods", methods_3b)]:
        print(f"\n=== {label} ===")

        # Load all trajectories at step 50
        task_data = defaultdict(dict)  # task_id -> {method: analysis}
        for method in methods:
            trajs = load_trajectories(method, 50)
            for t in trajs:
                a = analyze_trajectory(t)
                if not a["is_teacher"]:
                    task_data[a["task_id"]][method] = a

        # Find tasks present in all methods
        common_tasks = [tid for tid, d in task_data.items() if len(d) == len(methods)]
        print(f"  Common tasks across all {len(methods)} methods: {len(common_tasks)}")

        if not common_tasks:
            # Try pairwise
            for i, m1 in enumerate(methods):
                for m2 in methods[i + 1 :]:
                    pairs = [tid for tid, d in task_data.items() if m1 in d and m2 in d]
                    if pairs:
                        wins_m1 = sum(1 for tid in pairs if task_data[tid][m1]["reward"] > task_data[tid][m2]["reward"])
                        wins_m2 = sum(1 for tid in pairs if task_data[tid][m2]["reward"] > task_data[tid][m1]["reward"])
                        ties = len(pairs) - wins_m1 - wins_m2
                        print(f"  {m1} vs {m2}: {len(pairs)} common tasks, {m1} wins {wins_m1}, {m2} wins {wins_m2}, ties {ties}")
            continue

        # Head-to-head wins
        win_counts = Counter()
        for tid in common_tasks:
            best_method = max(methods, key=lambda m: task_data[tid][m]["reward"])
            best_reward = task_data[tid][best_method]["reward"]
            # Check for ties
            tied = [m for m in methods if task_data[tid][m]["reward"] == best_reward]
            if len(tied) == len(methods):
                win_counts["tie"] += 1
            else:
                for m in tied:
                    win_counts[m] += 1

        print(f"  Head-to-head wins (among {len(common_tasks)} common tasks):")
        for m in methods + ["tie"]:
            print(f"    {m}: {win_counts.get(m, 0)}")

        # Show reward distribution per method on common tasks
        print(f"\n  Mean reward on common tasks:")
        for m in methods:
            rewards = [task_data[tid][m]["reward"] for tid in common_tasks]
            avg = sum(rewards) / len(rewards)
            succ = sum(1 for r in rewards if r > 0)
            print(f"    {m}: avg={avg:.3f}, success={succ}/{len(rewards)}")


# =============================================================
# SECTION 4: Action Pattern Analysis by Task Type
# =============================================================
def section4_action_patterns():
    print("\n" + "=" * 80)
    print("SECTION 4: Action Pattern Analysis by Task Type")
    print("=" * 80)

    steps_to_check = [1, 50, 100]
    methods = ["7B-OnPolicy", "7B-DUET", "3B-OnPolicy", "3B-DUET"]

    for step in steps_to_check:
        print(f"\n=== Step {step} ===")
        for method in methods:
            trajs = load_trajectories(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t) for t in trajs]
            on_policy = [a for a in analyses if not a["is_teacher"]]
            if not on_policy:
                continue

            print(f"\n  {method}:")

            # Group by task type
            by_type = defaultdict(list)
            for a in on_policy:
                by_type[a["task_type"]].append(a)

            print(f"    {'Type':>8} | {'N':>3} | {'Succ%':>6} | {'AvgAct':>6} | {'AvgTokens':>9} | {'RepLoop':>7} | {'NothHapp':>8}")
            print(f"    {'--------':>8}-+-{'---':>3}-+-{'------':>6}-+-{'------':>6}-+-{'---------':>9}-+-{'-------':>7}-+-{'--------':>8}")
            for tt in sorted(by_type.keys()):
                items = by_type[tt]
                n = len(items)
                succ = sum(1 for a in items if a["success"]) / n * 100
                avg_act = sum(a["num_actions"] for a in items) / n
                avg_tok = sum(a["resp_tokens"] for a in items) / n
                rep = sum(1 for a in items if a["repetition_loop"])
                noth = sum(a["nothing_happened_count"] for a in items) / n
                print(f"    {tt:>8} | {n:>3} | {succ:>5.1f}% | {avg_act:>6.1f} | {avg_tok:>9.0f} | {rep:>7} | {noth:>8.1f}")


# =============================================================
# SECTION 5: Qualitative Case Studies
# =============================================================
def format_trajectory_summary(traj: dict, source: str = "rollout", max_actions: int = 15) -> str:
    """Format a concise summary of a trajectory for display."""
    if source == "rollout":
        messages = traj.get("messages", [])
        actions = extract_actions_from_messages(messages)
        query = traj.get("query", "")
        diag = traj.get("diag", {})
        is_teacher = diag.get("is_teacher", False)
    else:
        output = traj.get("output", "")
        actions = extract_actions_from_output(output)
        query = traj.get("input", "")
        if "Your task is to:" in query:
            query = query.split("Your task is to:")[-1].split("AVAILABLE")[0].strip()
        is_teacher = False

    lines = []
    lines.append(f"    Task: {query[:120]}")
    lines.append(f"    Teacher: {is_teacher} | Actions: {len(actions)}")
    for i, a in enumerate(actions[:max_actions]):
        lines.append(f"      [{i+1}] {a}")
    if len(actions) > max_actions:
        lines.append(f"      ... ({len(actions) - max_actions} more actions)")
    return "\n".join(lines)


def section5_case_studies():
    print("\n" + "=" * 80)
    print("SECTION 5: Qualitative Case Studies")
    print("=" * 80)

    # ---- Case A: Teacher data helping (DUET/LUFFY vs OnPolicy) ----
    print("\n--- Case A: Where Teacher Data Helps ---")
    # Find tasks where DUET/LUFFY succeeds at step 50 but OnPolicy fails
    duet_50 = load_trajectories("7B-DUET", 50)
    onpol_50 = load_trajectories("7B-OnPolicy", 50)
    luffy_50 = load_trajectories("7B-LUFFY", 50)

    # Build task_id -> trajectory maps
    duet_by_task = {}
    for t in duet_50:
        a = analyze_trajectory(t)
        if not a["is_teacher"]:
            duet_by_task[a["task_id"]] = (t, a)
    onpol_by_task = {}
    for t in onpol_50:
        a = analyze_trajectory(t)
        onpol_by_task[a["task_id"]] = (t, a)
    luffy_by_task = {}
    for t in luffy_50:
        a = analyze_trajectory(t)
        if not a["is_teacher"]:
            luffy_by_task[a["task_id"]] = (t, a)

    # Find teacher-helped cases
    helped_cases = []
    for tid in duet_by_task:
        if tid in onpol_by_task:
            d_a = duet_by_task[tid][1]
            o_a = onpol_by_task[tid][1]
            if d_a["success"] and not o_a["success"]:
                helped_cases.append(tid)

    print(f"  Tasks where DUET succeeds but OnPolicy fails: {len(helped_cases)}")
    for tid in helped_cases[:2]:
        print(f"\n  === Task {tid}: DUET SUCCESS, OnPolicy FAIL ===")
        print(f"  DUET trajectory:")
        print(format_trajectory_summary(duet_by_task[tid][0]))
        print(f"  OnPolicy trajectory:")
        print(format_trajectory_summary(onpol_by_task[tid][0]))

    # ---- Case B: Teacher data hurting (OnPolicy wins over DUET) ----
    print("\n\n--- Case B: Where Teacher Data May Hurt ---")
    hurt_cases = []
    for tid in onpol_by_task:
        if tid in duet_by_task:
            o_a = onpol_by_task[tid][1]
            d_a = duet_by_task[tid][1]
            if o_a["success"] and not d_a["success"]:
                hurt_cases.append(tid)

    print(f"  Tasks where OnPolicy succeeds but DUET fails: {len(hurt_cases)}")
    for tid in hurt_cases[:2]:
        print(f"\n  === Task {tid}: OnPolicy SUCCESS, DUET FAIL ===")
        print(f"  OnPolicy trajectory:")
        print(format_trajectory_summary(onpol_by_task[tid][0]))
        print(f"  DUET trajectory:")
        print(format_trajectory_summary(duet_by_task[tid][0]))

    # ---- Case C: Tasks both 3B and 7B struggle with ----
    print("\n\n--- Case C: Tasks Both 3B and 7B Struggle With ---")
    # Look at step 50 for both 3B and 7B OnPolicy
    onpol3b_50 = load_trajectories("3B-OnPolicy", 50)
    onpol3b_by_task = {}
    for t in onpol3b_50:
        a = analyze_trajectory(t)
        onpol3b_by_task[a["task_id"]] = (t, a)

    struggle_cases = []
    for tid in onpol_by_task:
        if tid in onpol3b_by_task:
            o7_a = onpol_by_task[tid][1]
            o3_a = onpol3b_by_task[tid][1]
            if not o7_a["success"] and not o3_a["success"]:
                struggle_cases.append(tid)

    print(f"  Tasks where both 7B and 3B OnPolicy fail at step 50: {len(struggle_cases)}")
    for tid in struggle_cases[:2]:
        print(f"\n  === Task {tid}: Both 7B and 3B FAIL ===")
        print(f"  7B OnPolicy trajectory:")
        print(format_trajectory_summary(onpol_by_task[tid][0]))
        print(f"  3B OnPolicy trajectory:")
        print(format_trajectory_summary(onpol3b_by_task[tid][0]))

    # ---- Case D: Showcase success at step 100 ----
    print("\n\n--- Case D: Successful 7B Trajectories at Step 100 ---")
    onpol_100 = load_trajectories("7B-OnPolicy", 100)
    successes_100 = []
    for t in onpol_100:
        a = analyze_trajectory(t)
        if a["success"] and not a["is_teacher"]:
            successes_100.append((t, a))

    # Pick shortest success (most efficient)
    if successes_100:
        successes_100.sort(key=lambda x: x[1]["num_actions"])
        print(f"\n  Most efficient success (fewest actions):")
        print(format_trajectory_summary(successes_100[0][0]))
        print(f"    Reward: {successes_100[0][1]['reward']}, Actions: {successes_100[0][1]['num_actions']}")

        # Pick a complex success (most actions but still succeeds)
        print(f"\n  Most complex success (most actions):")
        print(format_trajectory_summary(successes_100[-1][0]))
        print(f"    Reward: {successes_100[-1][1]['reward']}, Actions: {successes_100[-1][1]['num_actions']}")


# =============================================================
# SECTION 6: DUET-Specific Behavior Analysis
# =============================================================
def section6_duet_specific():
    print("\n" + "=" * 80)
    print("SECTION 6: DUET-Specific Behavior Analysis")
    print("=" * 80)

    # ---- 6a: Multi-action tag defect ----
    print("\n--- 6a: Multi-Action Tag Defect Detection ---")
    steps_to_check = list(range(1, 101))
    for method in ["7B-DUET", "3B-DUET", "7B-LUFFY", "3B-LUFFY", "7B-OnPolicy", "3B-OnPolicy"]:
        defect_steps = []
        for step in steps_to_check:
            trajs = load_trajectories(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t) for t in trajs]
            on_policy = [a for a in analyses if not a["is_teacher"]]
            multi_count = sum(1 for a in on_policy if a["multi_action_tag"])
            if multi_count > 0:
                defect_steps.append((step, multi_count, len(on_policy)))

        if defect_steps:
            print(f"  {method}: Multi-action defect found at {len(defect_steps)} steps")
            for s, c, n in defect_steps[:5]:
                print(f"    Step {s}: {c}/{n} trajectories ({c / n:.1%})")
            if len(defect_steps) > 5:
                print(f"    ... and {len(defect_steps) - 5} more steps")
            # Show first step range
            steps_with = [s for s, c, n in defect_steps]
            if steps_with:
                print(f"    Step range: {min(steps_with)} - {max(steps_with)}")
        else:
            print(f"  {method}: No multi-action defect detected")

    # ---- 6b: Teacher vs On-Policy quality comparison ----
    print("\n--- 6b: Teacher vs On-Policy Quality in DUET ---")
    for step in [1, 50, 90]:
        trajs = load_trajectories("7B-DUET", step)
        if not trajs:
            continue
        analyses = [analyze_trajectory(t) for t in trajs]
        teachers = [a for a in analyses if a["is_teacher"]]
        on_policy = [a for a in analyses if not a["is_teacher"]]

        print(f"\n  Step {step}:")
        if teachers:
            t_succ = sum(1 for a in teachers if a["success"]) / len(teachers)
            t_act = sum(a["num_actions"] for a in teachers) / len(teachers)
            t_rep = sum(1 for a in teachers if a["repetition_loop"])
            print(f"    Teacher (n={len(teachers)}): success={t_succ:.1%}, avg_actions={t_act:.1f}, rep_loops={t_rep}")
        else:
            print(f"    Teacher: no teacher samples")

        if on_policy:
            o_succ = sum(1 for a in on_policy if a["success"]) / len(on_policy)
            o_act = sum(a["num_actions"] for a in on_policy) / len(on_policy)
            o_rep = sum(1 for a in on_policy if a["repetition_loop"])
            print(f"    On-policy (n={len(on_policy)}): success={o_succ:.1%}, avg_actions={o_act:.1f}, rep_loops={o_rep}")

    # ---- 6c: DUET-specific failure patterns ----
    print("\n--- 6c: DUET-Specific Failure Patterns ---")
    # Compare failure distributions at step 50
    for method in ["7B-DUET", "7B-OnPolicy", "7B-LUFFY"]:
        trajs = load_trajectories(method, 50)
        if not trajs:
            continue
        analyses = [analyze_trajectory(t) for t in trajs]
        on_policy = [a for a in analyses if not a["is_teacher"]]
        failures = [a for a in on_policy if not a["success"]]

        if not failures:
            print(f"  {method}: No failures at step 50")
            continue

        # Analyze failure characteristics
        avg_actions = sum(a["num_actions"] for a in failures) / len(failures)
        avg_nothing = sum(a["nothing_happened_count"] for a in failures) / len(failures)
        rep = sum(1 for a in failures if a["repetition_loop"])
        premature = sum(1 for a in failures if a["num_actions"] <= 3)

        # Check if failures are concentrated in certain task types
        fail_types = Counter(a["task_type"] for a in failures)
        total_types = Counter(a["task_type"] for a in on_policy)

        print(f"\n  {method} failures at step 50:")
        print(f"    Total: {len(failures)}/{len(on_policy)} ({len(failures) / len(on_policy):.1%})")
        print(f"    Avg actions in failures: {avg_actions:.1f}")
        print(f"    Avg 'Nothing happened' in failures: {avg_nothing:.1f}")
        print(f"    Rep loops: {rep}, Premature: {premature}")
        print(f"    Failure rate by task type:")
        for tt in sorted(total_types.keys()):
            f_count = fail_types.get(tt, 0)
            t_count = total_types[tt]
            print(f"      {tt}: {f_count}/{t_count} = {f_count / t_count:.1%}")


# =============================================================
# SECTION 7: Validation Trajectory Analysis
# =============================================================
def section7_validation():
    print("\n" + "=" * 80)
    print("SECTION 7: Validation Trajectory Analysis")
    print("=" * 80)

    methods = ["7B-OnPolicy", "7B-LUFFY", "7B-DUET", "3B-OnPolicy", "3B-LUFFY", "3B-DUET"]
    val_steps = [50, 100]

    for step in val_steps:
        print(f"\n=== Validation at Step {step} ===")
        print(f"  {'Method':>15} | {'N':>4} | {'Succ%':>6} | {'AvgAct':>6} | {'RepLoop':>7} | {'MultiAct':>8} | {'CJK':>3} | {'NothHapp':>8}")
        print(f"  {'---------------':>15}-+-{'----':>4}-+-{'------':>6}-+-{'------':>6}-+-{'-------':>7}-+-{'--------':>8}-+-{'---':>3}-+-{'--------':>8}")

        for method in methods:
            trajs = load_validation(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t, source="validation") for t in trajs]

            n = len(analyses)
            succ = sum(1 for a in analyses if a["success"])
            avg_act = sum(a["num_actions"] for a in analyses) / n
            rep = sum(1 for a in analyses if a["repetition_loop"])
            multi = sum(1 for a in analyses if a["multi_action_tag"])
            cjk = sum(1 for a in analyses if a["cjk_detected"])
            noth = sum(a["nothing_happened_count"] for a in analyses) / n

            print(f"  {method:>15} | {n:>4} | {succ / n * 100:>5.1f}% | {avg_act:>6.1f} | {rep:>7} | {multi:>8} | {cjk:>3} | {noth:>8.1f}")

        # Score distributions
        print(f"\n  Score distribution at step {step}:")
        for method in methods:
            trajs = load_validation(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t, source="validation") for t in trajs]
            rewards = [a["reward"] for a in analyses]
            zero = sum(1 for r in rewards if r == 0)
            one = sum(1 for r in rewards if r == 1.0)
            other = len(rewards) - zero - one
            print(f"    {method:>15}: score=0: {zero}, score=1: {one}, other: {other} (total {len(rewards)})")

        # Failure pattern analysis for validation
        print(f"\n  Validation failure details at step {step}:")
        for method in methods:
            trajs = load_validation(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t, source="validation") for t in trajs]
            failures = [a for a in analyses if not a["success"]]
            if not failures:
                print(f"    {method}: No failures")
                continue

            fail_types = Counter(a["task_type"] for a in failures)
            total_types = Counter(a["task_type"] for a in analyses)

            rep_in_fail = sum(1 for a in failures if a["repetition_loop"])
            avg_act_fail = sum(a["num_actions"] for a in failures) / len(failures)

            print(f"    {method}: {len(failures)} failures, avg_actions={avg_act_fail:.1f}, rep_loops={rep_in_fail}")
            print(f"      Failure rate by type: ", end="")
            parts = []
            for tt in sorted(total_types.keys()):
                fc = fail_types.get(tt, 0)
                tc = total_types[tt]
                parts.append(f"{tt}={fc}/{tc}")
            print(", ".join(parts))


# =============================================================
# SECTION 8: Detailed Validation Case Studies
# =============================================================
def section8_validation_cases():
    print("\n" + "=" * 80)
    print("SECTION 8: Validation Trajectory Case Studies (Step 50 & 100)")
    print("=" * 80)

    for step in [50, 100]:
        print(f"\n=== Step {step} Case Studies ===")

        # Find interesting cases in 7B-OnPolicy validation
        trajs = load_validation("7B-OnPolicy", step)
        if not trajs:
            continue
        analyses = [(t, analyze_trajectory(t, source="validation")) for t in trajs]

        # Find a failure with repetition loop
        rep_failures = [(t, a) for t, a in analyses if a["repetition_loop"] and not a["success"]]
        if rep_failures:
            print(f"\n  -- Repetition Loop Failure (7B-OnPolicy, step {step}) --")
            t, a = rep_failures[0]
            output = t.get("output", "")
            actions = a["actions"]
            # Find the repeated sequence
            for i in range(len(actions) - 2):
                if actions[i] == actions[i + 1] == actions[i + 2]:
                    print(f"    Task type: {a['task_type']}")
                    print(f"    Total actions: {len(actions)}")
                    print(f"    Repeated action (starting at step {i + 1}): '{actions[i]}'")
                    print(f"    Actions around loop:")
                    for j in range(max(0, i - 1), min(len(actions), i + 5)):
                        print(f"      [{j + 1}] {actions[j]}")
                    break

        # Show a successful complex case
        successes = [(t, a) for t, a in analyses if a["success"]]
        if successes:
            # Pick one with many actions (complex task)
            successes.sort(key=lambda x: x[1]["num_actions"], reverse=True)
            t, a = successes[0]
            print(f"\n  -- Complex Success (7B-OnPolicy, step {step}) --")
            print(f"    Task type: {a['task_type']}")
            print(f"    Total actions: {a['num_actions']}")
            print(f"    Actions:")
            for i, act in enumerate(a["actions"][:20]):
                print(f"      [{i + 1}] {act}")
            if len(a["actions"]) > 20:
                print(f"      ... ({len(a['actions']) - 20} more)")


# =============================================================
# SECTION 9: Cross-Scale Comparison (7B vs 3B) Deep Dive
# =============================================================
def section9_cross_scale():
    print("\n" + "=" * 80)
    print("SECTION 9: Cross-Scale Deep Dive (7B vs 3B)")
    print("=" * 80)

    # Compare step 1 behavior in detail
    print("\n--- Step 1: Initial Capability ---")
    for scale_label, method in [("7B", "7B-OnPolicy"), ("3B", "3B-OnPolicy")]:
        trajs = load_trajectories(method, 1)
        if not trajs:
            continue
        analyses = [analyze_trajectory(t) for t in trajs]
        on_policy = [a for a in analyses if not a["is_teacher"]]

        # Analyze first action patterns
        first_actions = Counter()
        for a in on_policy:
            if a["actions"]:
                # Normalize first action
                first = a["actions"][0]
                if first.startswith("go to"):
                    first = "go to ..."
                elif first.startswith("take"):
                    first = "take ..."
                elif first.startswith("put"):
                    first = "put ..."
                first_actions[first] += 1

        print(f"\n  {scale_label} first action distribution:")
        for action, count in first_actions.most_common(10):
            print(f"    {action}: {count}")

        # Thinking pattern analysis
        think_counts = [a["think_count"] for a in on_policy]
        avg_think = sum(think_counts) / len(think_counts) if think_counts else 0
        think_usage = sum(1 for t in think_counts if t > 0) / len(think_counts) if think_counts else 0
        print(f"  {scale_label} think usage: {think_usage:.1%} of trajectories use <think>")
        print(f"  {scale_label} avg think per trajectory: {avg_think:.1f}")

    # Compare step 100 (end of training)
    print("\n--- Step 100: End-of-Training Comparison ---")
    for method in ["7B-OnPolicy", "7B-DUET", "3B-OnPolicy", "3B-DUET"]:
        trajs = load_trajectories(method, 100)
        if not trajs:
            continue
        analyses = [analyze_trajectory(t) for t in trajs]
        on_policy = [a for a in analyses if not a["is_teacher"]]
        if not on_policy:
            continue

        succ = sum(1 for a in on_policy if a["success"]) / len(on_policy)
        avg_act = sum(a["num_actions"] for a in on_policy) / len(on_policy)
        avg_tok = sum(a["resp_tokens"] for a in on_policy) / len(on_policy)
        rep = sum(1 for a in on_policy if a["repetition_loop"]) / len(on_policy)

        print(f"  {method:>15}: success={succ:.1%}, avg_actions={avg_act:.1f}, avg_tokens={avg_tok:.0f}, rep_loop_rate={rep:.1%}")


# =============================================================
# SECTION 10: Training Reward Curves from Trajectory Data
# =============================================================
def section10_reward_curves():
    print("\n" + "=" * 80)
    print("SECTION 10: Training Reward Curves from Trajectory Data")
    print("=" * 80)

    methods = ["7B-OnPolicy", "7B-LUFFY", "7B-DUET", "3B-OnPolicy", "3B-LUFFY", "3B-DUET"]
    steps = list(range(1, 101, 5)) + [100]
    steps = sorted(set(steps))

    for method in methods:
        print(f"\n  {method}:")
        print(f"    {'Step':>5} | {'N_on':>4} | {'N_teach':>7} | {'OnSucc%':>7} | {'OnRew':>6} | {'TeachSucc%':>10} | {'TeachRew':>8}")
        for step in steps:
            trajs = load_trajectories(method, step)
            if not trajs:
                continue
            analyses = [analyze_trajectory(t) for t in trajs]
            on_policy = [a for a in analyses if not a["is_teacher"]]
            teachers = [a for a in analyses if a["is_teacher"]]

            n_on = len(on_policy)
            n_t = len(teachers)
            on_succ = sum(1 for a in on_policy if a["success"]) / n_on * 100 if n_on else 0
            on_rew = sum(a["reward"] for a in on_policy) / n_on if n_on else 0
            t_succ = sum(1 for a in teachers if a["success"]) / n_t * 100 if n_t else 0
            t_rew = sum(a["reward"] for a in teachers) / n_t if n_t else 0

            t_succ_str = f"{t_succ:>9.1f}%" if n_t > 0 else "       N/A"
            t_rew_str = f"{t_rew:>8.3f}" if n_t > 0 else "     N/A"
            print(f"    {step:>5} | {n_on:>4} | {n_t:>7} | {on_succ:>6.1f}% | {on_rew:>6.3f} | {t_succ_str} | {t_rew_str}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ALFWorld 7B vs 3B Trajectory-Level Case Analysis")
    print("=" * 80)

    section1_step1_comparison()
    section2_failure_modes()
    section3_head_to_head()
    section4_action_patterns()
    section5_case_studies()
    section6_duet_specific()
    section7_validation()
    section8_validation_cases()
    section9_cross_scale()
    section10_reward_curves()

    print("\n\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)
