#!/usr/bin/env python3
"""
Comprehensive case-level analysis of 1.5B experiments for DUET paper.
Analyzes trajectory data, validation logs, failure modes, and behavioral differences.
"""

import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

BASE = "/data/home/qisheng/EvolAnalsis"
TRAJ_BASE = f"{BASE}/checkpoints/agentevolver"
VAL_BASE_ALF = f"{BASE}/experiments/alfworld"
VAL_BASE_WEB = f"{BASE}/experiments/webshop"

METHODS = ["onpolicy", "luffy", "chord", "duet", "sft_rl"]
METHODS_FULL = ["onpolicy", "luffy", "chord", "duet", "sft", "sft_rl"]
ANALYSIS_STEPS = [1, 5, 10, 20, 30, 50, 70, 100]
ENVS = ["alfworld", "webshop"]

# ALFWorld task type detection
ALFWORLD_TASK_TYPES = {
    "put": ["put", "place"],
    "clean": ["clean"],
    "heat": ["heat"],
    "cool": ["cool"],
    "examine": ["examine", "look at", "use desklamp"],
    "puttwo": ["two", "2"],
}


def detect_alfworld_task_type(messages):
    """Detect ALFWorld task type from the initial user message."""
    for m in messages:
        if m["role"] == "user":
            content = m["content"].lower()
            # Check puttwo first (more specific)
            if "two" in content or "2 " in content or " 2" in content:
                for keyword in ["put", "place"]:
                    if keyword in content:
                        return "puttwo"
            for ttype, keywords in ALFWORLD_TASK_TYPES.items():
                if ttype == "puttwo":
                    continue
                for kw in keywords:
                    if kw in content:
                        return ttype
            return "unknown"
    return "unknown"


def detect_alfworld_task_type_from_text(text):
    """Detect ALFWorld task type from input/output text."""
    text_lower = text.lower()
    if ("two" in text_lower or " 2 " in text_lower) and ("put" in text_lower or "place" in text_lower):
        return "puttwo"
    for ttype, keywords in ALFWORLD_TASK_TYPES.items():
        if ttype == "puttwo":
            continue
        for kw in keywords:
            if kw in text_lower:
                return ttype
    return "unknown"


def extract_actions_from_messages(messages):
    """Extract action list from message-format trajectory."""
    actions = []
    for m in messages:
        if m["role"] == "assistant":
            content = m["content"]
            # Skip the initial acknowledgment
            if content.startswith("OK. I'll"):
                continue
            # Extract action from <action> tags if present
            action_match = re.findall(r'<action>\s*(.*?)\s*</action>', content, re.DOTALL)
            if action_match:
                for a in action_match:
                    actions.append(a.strip())
            else:
                # Raw text action (ALFWorld sometimes)
                stripped = content.strip()
                if stripped and len(stripped) < 500:  # Likely an action, not a long ramble
                    actions.append(stripped)
    return actions


def extract_actions_from_text(output_text):
    """Extract action list from validation-format text output."""
    actions = []
    # Split by assistant/user turns
    parts = re.split(r'\n(?:assistant|user)\n', output_text)
    for part in parts:
        # Look for <action> tags
        action_match = re.findall(r'<action>\s*(.*?)\s*</action>', part, re.DOTALL)
        if action_match:
            for a in action_match:
                actions.append(a.strip())
    return actions


def count_actions_from_messages(messages):
    """Count number of agent actions (assistant turns excluding initial ack)."""
    count = 0
    for m in messages:
        if m["role"] == "assistant":
            if not m["content"].startswith("OK. I'll"):
                count += 1
    return count


def count_invalid_actions(messages):
    """Count invalid action responses from environment."""
    count = 0
    for m in messages:
        if m["role"] == "user" and "Invalid action" in m.get("content", ""):
            count += 1
    return count


def detect_cjk(text):
    """Detect CJK characters in text."""
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
    matches = cjk_pattern.findall(text)
    return len(matches) > 0, len(matches)


def detect_repetition_loop(actions, threshold=3):
    """Detect if the same action appears 3+ times consecutively."""
    if len(actions) < threshold:
        return False, 0
    max_repeat = 1
    current_repeat = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i-1]:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1
    return max_repeat >= threshold, max_repeat


def detect_format_errors(messages):
    """Detect format errors - assistant messages that are not valid actions."""
    errors = 0
    multi_action = 0
    for m in messages:
        if m["role"] == "assistant" and not m["content"].startswith("OK. I'll"):
            content = m["content"]
            # Check for multiple action tags
            action_count = len(re.findall(r'<action>', content))
            if action_count > 1:
                multi_action += 1
            # Check for malformed tags
            if '<action>' in content and '</action>' not in content:
                errors += 1
            elif '</action>' in content and '<action>' not in content:
                errors += 1
    return errors, multi_action


def detect_think_repetition(messages):
    """Detect repeated </think> tags."""
    for m in messages:
        if m["role"] == "assistant":
            think_count = m["content"].count("</think>")
            if think_count > 2:
                return True, think_count
    return False, 0


def analyze_response_length(messages):
    """Total response length (all assistant messages)."""
    total = 0
    for m in messages:
        if m["role"] == "assistant":
            total += len(m["content"])
    return total


def load_trajectories(env, method, step):
    """Load trajectory JSONL for given env/method/step."""
    path = f"{TRAJ_BASE}/{env}_qwen1.5b_{method}/Trajectory/trajectories_step_{step}.jsonl"
    if not os.path.exists(path):
        return None
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_validation(env, method, step):
    """Load validation JSONL."""
    if env == "alfworld":
        path = f"{VAL_BASE_ALF}/alfworld_qwen1.5b_{method}/validation_log/{step}.jsonl"
    else:
        path = f"{VAL_BASE_WEB}/webshop_qwen1.5b_{method}/validation_log/{step}.jsonl"
    if not os.path.exists(path):
        return None
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


# ==================== ANALYSIS FUNCTIONS ====================

def section_header(title):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def analyze_step1_capability():
    """Section 1: Step-1 capability analysis."""
    section_header("1. STEP-1 CAPABILITY ANALYSIS (All Methods)")

    for env in ENVS:
        print(f"\n--- {env.upper()} ---\n")
        print(f"{'Method':<12} {'Success%':>8} {'MeanRwd':>8} {'AvgActs':>8} {'AvgRspLen':>10} {'InvalidActs':>11} {'Entries':>7}")
        print("-" * 72)

        for method in METHODS:
            entries = load_trajectories(env, method, 1)
            if entries is None:
                print(f"{method:<12} {'N/A':>8}")
                continue

            # Filter on-policy only
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]

            successes = sum(1 for e in on_policy if e["success"])
            rewards = [e["reward"]["outcome"] for e in on_policy]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0

            action_counts = [count_actions_from_messages(e["messages"]) for e in on_policy]
            avg_actions = sum(action_counts) / len(action_counts) if action_counts else 0

            resp_lengths = [analyze_response_length(e["messages"]) for e in on_policy]
            avg_resp_len = sum(resp_lengths) / len(resp_lengths) if resp_lengths else 0

            invalid_counts = [count_invalid_actions(e["messages"]) for e in on_policy]
            avg_invalid = sum(invalid_counts) / len(invalid_counts) if invalid_counts else 0

            success_pct = successes / len(on_policy) * 100 if on_policy else 0

            print(f"{method:<12} {success_pct:>7.1f}% {mean_reward:>8.3f} {avg_actions:>8.1f} {avg_resp_len:>10.0f} {avg_invalid:>11.1f} {len(on_policy):>7}")


def analyze_failure_modes():
    """Section 2: Failure mode analysis across training steps."""
    section_header("2. FAILURE MODE ANALYSIS ACROSS TRAINING STEPS")

    for env in ENVS:
        print(f"\n{'='*60}")
        print(f"  {env.upper()} FAILURE MODES")
        print(f"{'='*60}")

        for method in METHODS:
            print(f"\n--- {method.upper()} ---")
            print(f"{'Step':>5} {'Rep.Loop':>9} {'FmtErr':>7} {'MultiAct':>9} {'CJK':>5} {'ThinkRep':>9} {'AvgInval':>9} {'N':>5}")
            print("-" * 65)

            steps_to_check = [s for s in ANALYSIS_STEPS if s <= (50 if method == "sft_rl" else 100)]

            for step in steps_to_check:
                entries = load_trajectories(env, method, step)
                if entries is None:
                    continue

                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    continue

                n = len(on_policy)
                rep_loops = 0
                fmt_errors = 0
                multi_actions = 0
                cjk_count = 0
                think_reps = 0
                total_invalid = 0

                for e in on_policy:
                    actions = extract_actions_from_messages(e["messages"])

                    # Repetition loops
                    is_rep, _ = detect_repetition_loop(actions)
                    if is_rep:
                        rep_loops += 1

                    # Format errors
                    fe, ma = detect_format_errors(e["messages"])
                    if fe > 0:
                        fmt_errors += 1
                    if ma > 0:
                        multi_actions += 1

                    # CJK
                    full_text = " ".join(m["content"] for m in e["messages"] if m["role"] == "assistant")
                    has_cjk, _ = detect_cjk(full_text)
                    if has_cjk:
                        cjk_count += 1

                    # Think repetition
                    has_think_rep, _ = detect_think_repetition(e["messages"])
                    if has_think_rep:
                        think_reps += 1

                    # Invalid actions
                    total_invalid += count_invalid_actions(e["messages"])

                avg_invalid = total_invalid / n

                print(f"{step:>5} {rep_loops:>5}/{n:<3} {fmt_errors:>3}/{n:<3} {multi_actions:>5}/{n:<3} {cjk_count:>2}/{n:<2} {think_reps:>5}/{n:<3} {avg_invalid:>9.1f} {n:>5}")


def analyze_training_success_evolution():
    """Section 3: Training success rate evolution."""
    section_header("3. TRAINING SUCCESS EVOLUTION")

    for env in ENVS:
        print(f"\n--- {env.upper()}: On-Policy Success Rate (%) ---\n")

        # Header
        header = f"{'Step':>5}"
        for method in METHODS:
            header += f" {method:>10}"
        print(header)
        print("-" * (5 + 11 * len(METHODS)))

        for step in ANALYSIS_STEPS:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'N/A':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'N/A':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'N/A':>10}"
                    continue
                success_rate = sum(1 for e in on_policy if e["success"]) / len(on_policy) * 100
                row += f" {success_rate:>9.1f}%"
            print(row)

        # Teacher vs on-policy for methods with teacher
        print(f"\n--- {env.upper()}: Teacher vs On-Policy Comparison ---\n")
        for method in ["luffy", "chord", "duet"]:
            print(f"\n  {method.upper()}:")
            print(f"  {'Step':>5} {'OnPol Suc%':>10} {'OnPol Rwd':>10} {'Tchr Suc%':>10} {'Tchr Rwd':>10} {'Tchr N':>7}")
            print("  " + "-" * 55)

            for step in ANALYSIS_STEPS:
                entries = load_trajectories(env, method, step)
                if entries is None:
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                teachers = [e for e in entries if e["diag"].get("is_teacher", False)]

                if not on_policy:
                    continue

                op_suc = sum(1 for e in on_policy if e["success"]) / len(on_policy) * 100
                op_rwd = sum(e["reward"]["outcome"] for e in on_policy) / len(on_policy)

                if teachers:
                    t_suc = sum(1 for e in teachers if e["success"]) / len(teachers) * 100
                    t_rwd = sum(e["reward"]["outcome"] for e in teachers) / len(teachers)
                    t_n = len(teachers)
                    print(f"  {step:>5} {op_suc:>9.1f}% {op_rwd:>10.3f} {t_suc:>9.1f}% {t_rwd:>10.3f} {t_n:>7}")
                else:
                    print(f"  {step:>5} {op_suc:>9.1f}% {op_rwd:>10.3f} {'N/A':>10} {'N/A':>10} {'0':>7}")

        # Mean reward evolution
        print(f"\n--- {env.upper()}: On-Policy Mean Reward ---\n")
        header = f"{'Step':>5}"
        for method in METHODS:
            header += f" {method:>10}"
        print(header)
        print("-" * (5 + 11 * len(METHODS)))

        for step in ANALYSIS_STEPS:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'N/A':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'N/A':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'N/A':>10}"
                    continue
                mean_rwd = sum(e["reward"]["outcome"] for e in on_policy) / len(on_policy)
                row += f" {mean_rwd:>10.3f}"
            print(row)


def analyze_validation():
    """Section 4: Validation analysis."""
    section_header("4. VALIDATION ANALYSIS")

    # ALFWorld validation
    print("\n--- ALFWORLD VALIDATION ---\n")
    print(f"{'Method':<12} {'Step':>5} {'Success%':>9} {'AvgScore':>9} {'N':>5}")
    print("-" * 45)

    for method in METHODS_FULL:
        for step in [50, 100]:
            val = load_validation("alfworld", method, step)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            success = sum(1 for s in scores if s > 0)
            avg_score = sum(scores) / len(scores)
            print(f"{method:<12} {step:>5} {success/len(scores)*100:>8.1f}% {avg_score:>9.3f} {len(scores):>5}")

    # ALFWorld by task type
    print(f"\n--- ALFWORLD VALIDATION BY TASK TYPE (Step 50) ---\n")
    task_type_results = {}

    for method in METHODS_FULL:
        val = load_validation("alfworld", method, 50)
        if val is None:
            continue
        task_type_results[method] = defaultdict(lambda: {"success": 0, "total": 0})

        for e in val:
            # Detect task type from input text
            ttype = detect_alfworld_task_type_from_text(e.get("input", ""))
            task_type_results[method][ttype]["total"] += 1
            if e["score"] > 0:
                task_type_results[method][ttype]["success"] += 1

    # Get all task types
    all_types = set()
    for m in task_type_results:
        all_types.update(task_type_results[m].keys())
    all_types = sorted(all_types)

    header = f"{'TaskType':<12}"
    for method in METHODS_FULL:
        if method in task_type_results:
            header += f" {method:>10}"
    print(header)
    print("-" * (12 + 11 * len(task_type_results)))

    for ttype in all_types:
        row = f"{ttype:<12}"
        for method in METHODS_FULL:
            if method not in task_type_results:
                continue
            data = task_type_results[method][ttype]
            if data["total"] > 0:
                pct = data["success"] / data["total"] * 100
                row += f" {pct:>6.1f}%({data['total']:>2})"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # WebShop validation
    print(f"\n--- WEBSHOP VALIDATION ---\n")
    print(f"{'Method':<12} {'Step':>5} {'MeanScore':>10} {'Score>0%':>9} {'Score>0.5%':>11} {'N':>5}")
    print("-" * 55)

    for method in METHODS_FULL:
        for step in [50, 100]:
            val = load_validation("webshop", method, step)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            mean_score = sum(scores) / len(scores)
            pos = sum(1 for s in scores if s > 0) / len(scores) * 100
            high = sum(1 for s in scores if s > 0.5) / len(scores) * 100
            print(f"{method:<12} {step:>5} {mean_score:>10.4f} {pos:>8.1f}% {high:>10.1f}% {len(scores):>5}")

    # WebShop score distribution
    print(f"\n--- WEBSHOP SCORE DISTRIBUTION (Step 50) ---\n")
    bins = [(0, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.01)]
    bin_labels = ["=0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "=1.0"]

    header = f"{'Method':<12}"
    for bl in bin_labels:
        header += f" {bl:>8}"
    print(header)
    print("-" * (12 + 9 * len(bin_labels)))

    for method in METHODS_FULL:
        val = load_validation("webshop", method, 50)
        if val is None:
            continue
        scores = [e["score"] for e in val]
        row = f"{method:<12}"
        for i, (lo, hi) in enumerate(bins):
            if i == 0:  # exact 0
                count = sum(1 for s in scores if s == 0)
            elif i == len(bins) - 1:  # exact 1.0
                count = sum(1 for s in scores if s >= 1.0)
            else:
                count = sum(1 for s in scores if lo < s <= hi)
            row += f" {count:>8}"
        print(row)


def analyze_duet_behavioral_advantages():
    """Section 5: DUET-specific behavioral advantages."""
    section_header("5. DUET BEHAVIORAL ADVANTAGES")

    for env in ENVS:
        print(f"\n--- {env.upper()}: CJK Collapse Rate (on-policy only) ---\n")
        print(f"{'Step':>5}", end="")
        for method in METHODS:
            print(f" {method:>10}", end="")
        print()
        print("-" * (5 + 11 * len(METHODS)))

        for step in [1, 10, 30, 50, 70, 100]:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'N/A':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'N/A':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'N/A':>10}"
                    continue
                cjk_count = 0
                for e in on_policy:
                    full_text = " ".join(m["content"] for m in e["messages"] if m["role"] == "assistant")
                    has_cjk, _ = detect_cjk(full_text)
                    if has_cjk:
                        cjk_count += 1
                pct = cjk_count / len(on_policy) * 100
                row += f" {pct:>9.1f}%"
            print(row)

        print(f"\n--- {env.upper()}: Format Error Rate (on-policy) ---\n")
        print(f"{'Step':>5}", end="")
        for method in METHODS:
            print(f" {method:>10}", end="")
        print()
        print("-" * (5 + 11 * len(METHODS)))

        for step in [1, 10, 30, 50, 70, 100]:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'N/A':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'N/A':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'N/A':>10}"
                    continue
                fmt_err_count = 0
                for e in on_policy:
                    fe, _ = detect_format_errors(e["messages"])
                    if fe > 0:
                        fmt_err_count += 1
                pct = fmt_err_count / len(on_policy) * 100
                row += f" {pct:>9.1f}%"
            print(row)

        # Repetition loop rate
        print(f"\n--- {env.upper()}: Repetition Loop Rate (3+ consecutive same action, on-policy) ---\n")
        print(f"{'Step':>5}", end="")
        for method in METHODS:
            print(f" {method:>10}", end="")
        print()
        print("-" * (5 + 11 * len(METHODS)))

        for step in [1, 10, 30, 50, 70, 100]:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'N/A':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'N/A':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'N/A':>10}"
                    continue
                rep_count = 0
                for e in on_policy:
                    actions = extract_actions_from_messages(e["messages"])
                    is_rep, _ = detect_repetition_loop(actions)
                    if is_rep:
                        rep_count += 1
                pct = rep_count / len(on_policy) * 100
                row += f" {pct:>9.1f}%"
            print(row)


def analyze_sft_rl():
    """Section 6: SFT+RL behavioral analysis."""
    section_header("6. SFT+RL BEHAVIORAL ANALYSIS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: SFT vs SFT+RL vs DUET Comparison ---\n")

        # SFT (no training trajectory, only validation)
        sft_val = load_validation(env, "sft", 50)
        sft_rl_val = load_validation(env, "sft_rl", 50)
        duet_val = load_validation(env, "duet", 50)
        onpol_val = load_validation(env, "onpolicy", 50)

        print(f"{'Method':<12} ", end="")
        if env == "alfworld":
            print(f"{'Success%':>9} {'AvgScore':>9}")
        else:
            print(f"{'MeanScore':>10} {'Score>0%':>9}")
        print("-" * 35)

        for name, val in [("SFT", sft_val), ("SFT+RL", sft_rl_val), ("DUET", duet_val), ("OnPolicy", onpol_val)]:
            if val is None:
                print(f"{name:<12}  N/A")
                continue
            scores = [e["score"] for e in val]
            if env == "alfworld":
                success = sum(1 for s in scores if s > 0) / len(scores) * 100
                avg = sum(scores) / len(scores)
                print(f"{name:<12} {success:>8.1f}% {avg:>9.3f}")
            else:
                mean_s = sum(scores) / len(scores)
                pos = sum(1 for s in scores if s > 0) / len(scores) * 100
                print(f"{name:<12} {mean_s:>10.4f} {pos:>8.1f}%")

        # SFT+RL training trajectory evolution
        print(f"\n  SFT+RL Training Evolution:")
        print(f"  {'Step':>5} {'Suc%':>7} {'MeanRwd':>8} {'AvgActs':>8} {'RepLoop%':>9} {'CJK%':>6}")
        print("  " + "-" * 45)

        for step in [1, 5, 10, 20, 30, 50]:
            entries = load_trajectories(env, "sft_rl", step)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            suc = sum(1 for e in on_policy if e["success"]) / len(on_policy) * 100
            rwd = sum(e["reward"]["outcome"] for e in on_policy) / len(on_policy)
            acts = sum(count_actions_from_messages(e["messages"]) for e in on_policy) / len(on_policy)

            rep = 0
            cjk = 0
            for e in on_policy:
                actions = extract_actions_from_messages(e["messages"])
                is_rep, _ = detect_repetition_loop(actions)
                if is_rep:
                    rep += 1
                full_text = " ".join(m["content"] for m in e["messages"] if m["role"] == "assistant")
                has_cjk, _ = detect_cjk(full_text)
                if has_cjk:
                    cjk += 1

            rep_pct = rep / len(on_policy) * 100
            cjk_pct = cjk / len(on_policy) * 100
            print(f"  {step:>5} {suc:>6.1f}% {rwd:>8.3f} {acts:>8.1f} {rep_pct:>8.1f}% {cjk_pct:>5.1f}%")


def generate_case_studies():
    """Section 7: Qualitative case studies."""
    section_header("7. QUALITATIVE CASE STUDIES")

    # ---- Case Study 1: DUET succeeds, others fail (ALFWorld) ----
    print("\n--- Case Study 1: DUET succeeds where others fail (ALFWorld, Step 50) ---\n")

    duet_val = load_validation("alfworld", "duet", 50)
    onpol_val = load_validation("alfworld", "onpolicy", 50)
    luffy_val = load_validation("alfworld", "luffy", 50)

    if duet_val and onpol_val:
        # Find tasks where DUET succeeds but OnPolicy fails
        # Validation entries don't have task_id, use input text as key
        duet_by_input = {}
        for e in duet_val:
            key = e["input"][:200]  # Use first 200 chars as key
            duet_by_input[key] = e

        onpol_by_input = {}
        for e in onpol_val:
            key = e["input"][:200]
            onpol_by_input[key] = e

        luffy_by_input = {}
        if luffy_val:
            for e in luffy_val:
                key = e["input"][:200]
                luffy_by_input[key] = e

        # Find cases where DUET succeeds, others don't
        duet_wins = []
        for key in duet_by_input:
            d_score = duet_by_input[key]["score"]
            o_score = onpol_by_input.get(key, {}).get("score", -1)
            l_score = luffy_by_input.get(key, {}).get("score", -1)
            if d_score > 0 and o_score <= 0 and l_score <= 0:
                duet_wins.append((key, d_score, o_score, l_score))

        print(f"Found {len(duet_wins)} tasks where DUET succeeds but OnPolicy AND LUFFY fail.\n")

        if duet_wins:
            # Show first example
            key = duet_wins[0][0]
            d_entry = duet_by_input[key]
            o_entry = onpol_by_input.get(key)

            # Extract task description
            input_text = d_entry["input"]
            # Find task goal
            task_lines = input_text.split("\n")
            for line in task_lines:
                if "task" in line.lower() or "goal" in line.lower() or "your " in line.lower():
                    print(f"Task: {line.strip()[:200]}")
                    break

            ttype = detect_alfworld_task_type_from_text(input_text)
            print(f"Task Type: {ttype}")
            print(f"DUET score: {d_entry['score']:.2f}")
            if o_entry:
                print(f"OnPolicy score: {o_entry['score']:.2f}")

            # Show DUET actions
            d_actions = extract_actions_from_text(d_entry["output"])
            print(f"\nDUET actions ({len(d_actions)} total):")
            for i, a in enumerate(d_actions[:8]):
                print(f"  [{i+1}] {a[:100]}")
            if len(d_actions) > 8:
                print(f"  ... ({len(d_actions) - 8} more)")

            # Show OnPolicy actions
            if o_entry:
                o_actions = extract_actions_from_text(o_entry["output"])
                print(f"\nOnPolicy actions ({len(o_actions)} total):")
                for i, a in enumerate(o_actions[:8]):
                    print(f"  [{i+1}] {a[:100]}")
                if len(o_actions) > 8:
                    print(f"  ... ({len(o_actions) - 8} more)")

    # ---- Case Study 2: DUET failure analysis ----
    print("\n\n--- Case Study 2: DUET failure (ALFWorld, Step 50) ---\n")

    if duet_val:
        # Find a DUET failure with interesting characteristics
        duet_failures = [e for e in duet_val if e["score"] <= 0]

        if duet_failures:
            # Find one with many actions (tried hard but failed)
            best_failure = None
            max_actions = 0
            for e in duet_failures:
                actions = extract_actions_from_text(e["output"])
                if len(actions) > max_actions and len(actions) < 25:  # Not a repetition loop
                    max_actions = len(actions)
                    best_failure = e

            if best_failure:
                input_text = best_failure["input"]
                task_lines = input_text.split("\n")
                for line in task_lines:
                    if "task" in line.lower() or "goal" in line.lower() or "your " in line.lower():
                        print(f"Task: {line.strip()[:200]}")
                        break

                ttype = detect_alfworld_task_type_from_text(input_text)
                print(f"Task Type: {ttype}")
                print(f"Score: {best_failure['score']:.2f}")

                actions = extract_actions_from_text(best_failure["output"])
                print(f"\nDUET actions ({len(actions)} total):")
                for i, a in enumerate(actions[:12]):
                    print(f"  [{i+1}] {a[:100]}")
                if len(actions) > 12:
                    print(f"  ... ({len(actions) - 12} more)")

                # Analyze what went wrong
                rep, max_rep = detect_repetition_loop(actions)
                if rep:
                    print(f"\n  DIAGNOSIS: Repetition loop detected (max {max_rep} consecutive)")

                full_text = best_failure["output"]
                has_cjk, cjk_n = detect_cjk(full_text)
                if has_cjk:
                    print(f"  DIAGNOSIS: CJK characters detected ({cjk_n} chars)")

                if "Invalid action" in full_text:
                    inv_count = full_text.count("Invalid action")
                    print(f"  DIAGNOSIS: {inv_count} invalid action(s) in trajectory")

    # ---- Case Study 3: WebShop search strategy comparison ----
    print("\n\n--- Case Study 3: WebShop Search Strategy Comparison (Step 50) ---\n")

    duet_val_w = load_validation("webshop", "duet", 50)
    onpol_val_w = load_validation("webshop", "onpolicy", 50)

    if duet_val_w and onpol_val_w:
        # Find matched tasks with score difference
        duet_by_input_w = {}
        for e in duet_val_w:
            key = e["input"][:200]
            duet_by_input_w[key] = e

        onpol_by_input_w = {}
        for e in onpol_val_w:
            key = e["input"][:200]
            onpol_by_input_w[key] = e

        # Find biggest score gap in DUET's favor
        gaps = []
        for key in duet_by_input_w:
            if key in onpol_by_input_w:
                d_score = duet_by_input_w[key]["score"]
                o_score = onpol_by_input_w[key]["score"]
                gaps.append((d_score - o_score, key, d_score, o_score))

        gaps.sort(reverse=True)

        if gaps:
            print(f"Top 5 tasks where DUET outperforms OnPolicy:")
            for i, (gap, key, d_s, o_s) in enumerate(gaps[:5]):
                print(f"  [{i+1}] Gap={gap:.2f} (DUET={d_s:.2f}, OnPol={o_s:.2f})")

            # Detailed comparison of the best case
            print(f"\nDetailed comparison of top case:")
            _, key, d_s, o_s = gaps[0]
            d_entry = duet_by_input_w[key]
            o_entry = onpol_by_input_w[key]

            # Extract instruction
            input_text = d_entry["input"]
            instr_match = re.search(r'Instruction:.*?(?:Find me|I need|I want|I\'m looking)(.*?)(?:\[SEP\]|$)', input_text, re.DOTALL)
            if instr_match:
                print(f"\nInstruction: ...{instr_match.group(0)[:200]}")
            else:
                # Just show part of input
                for line in input_text.split("\n"):
                    if "instruction" in line.lower() or "find" in line.lower():
                        print(f"\nInstruction: {line.strip()[:200]}")
                        break

            print(f"\nDUET score: {d_s:.2f}")
            print(f"OnPolicy score: {o_s:.2f}")

            # Extract first search query from each
            d_actions = extract_actions_from_text(d_entry["output"])
            o_actions = extract_actions_from_text(o_entry["output"])

            print(f"\nDUET actions ({len(d_actions)} total):")
            for i, a in enumerate(d_actions[:6]):
                print(f"  [{i+1}] {a[:120]}")

            print(f"\nOnPolicy actions ({len(o_actions)} total):")
            for i, a in enumerate(o_actions[:6]):
                print(f"  [{i+1}] {a[:120]}")

    # ---- Case Study 4: Training trajectory showing learning ----
    print("\n\n--- Case Study 4: Learning Progress - Same Task at Step 1 vs Step 50 ---\n")

    for env in ENVS:
        print(f"\n  {env.upper()}:")
        step1 = load_trajectories(env, "duet", 1)
        step50 = load_trajectories(env, "duet", 50)

        if not step1 or not step50:
            print("  Data not available")
            continue

        # Find same task_id that fails at step 1 but succeeds at step 50
        step1_by_task = {}
        for e in step1:
            if not e["diag"].get("is_teacher", False):
                step1_by_task[e["task_id"]] = e

        step50_by_task = {}
        for e in step50:
            if not e["diag"].get("is_teacher", False):
                step50_by_task[e["task_id"]] = e

        # Find improvement cases
        improved = []
        for tid in step1_by_task:
            if tid in step50_by_task:
                s1 = step1_by_task[tid]
                s50 = step50_by_task[tid]
                if not s1["success"] and s50["success"]:
                    improved.append(tid)
                elif s50["reward"]["outcome"] > s1["reward"]["outcome"] + 0.3:
                    improved.append(tid)

        print(f"  Tasks that improved from step 1 to step 50: {len(improved)}")

        if improved:
            tid = improved[0]
            s1 = step1_by_task[tid]
            s50 = step50_by_task[tid]

            print(f"\n  Task ID: {tid}")
            print(f"  Step 1: success={s1['success']}, reward={s1['reward']['outcome']:.2f}")
            print(f"  Step 50: success={s50['success']}, reward={s50['reward']['outcome']:.2f}")

            # Show step 1 actions
            a1 = extract_actions_from_messages(s1["messages"])
            print(f"\n  Step 1 actions ({len(a1)} total):")
            for i, a in enumerate(a1[:5]):
                print(f"    [{i+1}] {a[:100]}")
            if len(a1) > 5:
                print(f"    ... ({len(a1) - 5} more)")

            # Show step 50 actions
            a50 = extract_actions_from_messages(s50["messages"])
            print(f"\n  Step 50 actions ({len(a50)} total):")
            for i, a in enumerate(a50[:5]):
                print(f"    [{i+1}] {a[:100]}")
            if len(a50) > 5:
                print(f"    ... ({len(a50) - 5} more)")


def analyze_head_to_head_validation():
    """Head-to-head comparison on matched validation tasks."""
    section_header("8. HEAD-TO-HEAD VALIDATION COMPARISON")

    for env in ENVS:
        print(f"\n--- {env.upper()}: Pairwise Win/Tie/Loss (Step 50) ---\n")

        # Load all methods
        val_data = {}
        for method in METHODS_FULL:
            val = load_validation(env, method, 50)
            if val:
                val_data[method] = {}
                for e in val:
                    key = e["input"][:200]
                    val_data[method][key] = e["score"]

        if len(val_data) < 2:
            print("  Insufficient data for comparison")
            continue

        # Pairwise comparison
        methods_present = [m for m in METHODS_FULL if m in val_data]

        # Header
        print(f"{'Pair':<20} {'Win':>5} {'Tie':>5} {'Loss':>5} {'AvgGap':>8}")
        print("-" * 48)

        # Compare DUET against all others
        if "duet" in val_data:
            for other in methods_present:
                if other == "duet":
                    continue
                win = 0
                tie = 0
                loss = 0
                total_gap = 0
                count = 0

                for key in val_data["duet"]:
                    if key in val_data[other]:
                        d_score = val_data["duet"][key]
                        o_score = val_data[other][key]
                        if env == "alfworld":
                            # Binary
                            if d_score > 0 and o_score <= 0:
                                win += 1
                            elif d_score <= 0 and o_score > 0:
                                loss += 1
                            else:
                                tie += 1
                        else:
                            # Continuous
                            gap = d_score - o_score
                            if gap > 0.05:
                                win += 1
                            elif gap < -0.05:
                                loss += 1
                            else:
                                tie += 1
                        total_gap += d_score - o_score
                        count += 1

                avg_gap = total_gap / count if count > 0 else 0
                print(f"DUET vs {other:<11} {win:>5} {tie:>5} {loss:>5} {avg_gap:>+8.3f}")


def analyze_advantage_distributions():
    """Analyze advantage distributions across methods."""
    section_header("9. ADVANTAGE AND DIAGNOSTIC DISTRIBUTIONS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: Advantage Statistics (Step 50) ---\n")
        print(f"{'Method':<12} {'AdvMean':>8} {'AdvStd':>8} {'LogPMean':>9} {'RwdSum':>8} {'ValidTok':>9}")
        print("-" * 58)

        for method in METHODS:
            entries = load_trajectories(env, method, 50)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            adv_means = [e["diag"]["adv_mean"] for e in on_policy if e["diag"]["adv_mean"] is not None]
            adv_stds = [e["diag"]["adv_std"] for e in on_policy if e["diag"]["adv_std"] is not None]
            logp_means = [e["diag"]["old_log_prob_mean"] for e in on_policy if e["diag"]["old_log_prob_mean"] is not None]
            rwd_sums = [e["diag"]["reward_sum"] for e in on_policy if e["diag"]["reward_sum"] is not None]
            valid_toks = [e["diag"]["response_valid_tokens"] for e in on_policy]

            avg_adv = sum(adv_means) / len(adv_means) if adv_means else 0
            avg_std = sum(adv_stds) / len(adv_stds) if adv_stds else 0
            avg_logp = sum(logp_means) / len(logp_means) if logp_means else 0
            avg_rwd = sum(rwd_sums) / len(rwd_sums) if rwd_sums else 0
            avg_tok = sum(valid_toks) / len(valid_toks) if valid_toks else 0

            print(f"{method:<12} {avg_adv:>8.4f} {avg_std:>8.4f} {avg_logp:>9.4f} {avg_rwd:>8.3f} {avg_tok:>9.0f}")


def analyze_invalid_action_patterns():
    """Detailed analysis of invalid action patterns."""
    section_header("10. INVALID ACTION PATTERN ANALYSIS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: Invalid Action Analysis (Step 50, On-Policy) ---\n")

        for method in METHODS:
            entries = load_trajectories(env, method, 50)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            total_actions = 0
            total_invalid = 0
            trajectories_with_invalid = 0
            invalid_reasons = Counter()

            for e in on_policy:
                n_actions = count_actions_from_messages(e["messages"])
                n_invalid = count_invalid_actions(e["messages"])
                total_actions += n_actions
                total_invalid += n_invalid
                if n_invalid > 0:
                    trajectories_with_invalid += 1

                # Categorize invalid action reasons
                for m in e["messages"]:
                    if m["role"] == "user" and "Invalid action" in m.get("content", ""):
                        content = m["content"]
                        if "format" in content.lower():
                            invalid_reasons["format"] += 1
                        elif "available actions" in content.lower():
                            invalid_reasons["not_available"] += 1
                        else:
                            invalid_reasons["other"] += 1

            pct_traj = trajectories_with_invalid / len(on_policy) * 100
            pct_acts = total_invalid / total_actions * 100 if total_actions > 0 else 0

            print(f"  {method}:")
            print(f"    Trajectories with invalid actions: {trajectories_with_invalid}/{len(on_policy)} ({pct_traj:.1f}%)")
            print(f"    Total invalid / total actions: {total_invalid}/{total_actions} ({pct_acts:.1f}%)")
            print(f"    Invalid reasons: {dict(invalid_reasons)}")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 80)
    print("  DUET 1.5B COMPREHENSIVE CASE-LEVEL ANALYSIS")
    print("  Environments: ALFWorld, WebShop")
    print("  Methods: OnPolicy, LUFFY, CHORD, DUET, SFT, SFT+RL")
    print("=" * 80)

    analyze_step1_capability()
    analyze_failure_modes()
    analyze_training_success_evolution()
    analyze_validation()
    analyze_duet_behavioral_advantages()
    analyze_sft_rl()
    generate_case_studies()
    analyze_head_to_head_validation()
    analyze_advantage_distributions()
    analyze_invalid_action_patterns()

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
