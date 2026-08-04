#!/usr/bin/env python3
"""
Comprehensive case-level analysis of 1.5B experiments for DUET paper (v2).
Fixed task type detection, case studies, head-to-head matching.
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


# ====================== HELPER FUNCTIONS ======================

def detect_alfworld_task_type_from_task_str(task_str):
    """Detect ALFWorld task type from 'Your task is to: ...' string."""
    task_lower = task_str.lower()
    # Order matters: check multi-object tasks first
    if "two" in task_lower or " 2 " in task_lower:
        return "puttwo"
    if "look at" in task_lower or "desklamp" in task_lower or "examine" in task_lower:
        return "examine"
    if "heat" in task_lower:
        return "heat"
    if "cool" in task_lower:
        return "cool"
    if "clean" in task_lower:
        return "clean"
    if "put" in task_lower or "place" in task_lower:
        return "put"
    return "unknown"


def extract_task_goal_from_messages(messages):
    """Extract 'Your task is to: ...' from trajectory messages."""
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            idx = content.find("Your task is to:")
            if idx >= 0:
                # Extract until newline
                end = content.find("\n", idx)
                if end < 0:
                    end = len(content)
                return content[idx:end].strip()
    return None


def extract_task_goal_from_text(text):
    """Extract task goal from validation input/output text."""
    # Check in output first (first user message often has it)
    idx = text.find("Your task is to:")
    if idx >= 0:
        end = text.find("\n", idx)
        if end < 0:
            end = min(len(text), idx + 200)
        return text[idx:end].strip()
    return None


def extract_actions_from_messages(messages):
    """Extract action list from message-format trajectory."""
    actions = []
    for m in messages:
        if m["role"] == "assistant":
            content = m["content"]
            if content.startswith("OK. I'll"):
                continue
            action_match = re.findall(r'<action>\s*(.*?)\s*</action>', content, re.DOTALL)
            if action_match:
                for a in action_match:
                    actions.append(a.strip())
            else:
                stripped = content.strip()
                if stripped and len(stripped) < 500:
                    actions.append(stripped)
    return actions


def extract_actions_from_text(output_text):
    """Extract action list from validation-format text output."""
    actions = []
    action_match = re.findall(r'<action>\s*(.*?)\s*</action>', output_text, re.DOTALL)
    for a in action_match:
        actions.append(a.strip())
    return actions


def count_actions_from_messages(messages):
    """Count number of agent actions."""
    count = 0
    for m in messages:
        if m["role"] == "assistant" and not m["content"].startswith("OK. I'll"):
            count += 1
    return count


def count_invalid_actions_messages(messages):
    """Count invalid action responses from environment in messages format."""
    count = 0
    for m in messages:
        if m["role"] == "user":
            content = m.get("content", "")
            if "Invalid action" in content or "Nothing happened" in content:
                count += 1
    return count


def count_invalid_actions_text(text):
    """Count invalid actions in text format."""
    return text.count("Invalid action") + text.count("Nothing happened")


def detect_cjk(text):
    """Detect CJK characters."""
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
    matches = cjk_pattern.findall(text)
    return len(matches) > 0, len(matches)


def detect_repetition_loop(actions, threshold=3):
    """Detect consecutive repeated actions."""
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
    """Detect format errors."""
    errors = 0
    multi_action = 0
    for m in messages:
        if m["role"] == "assistant" and not m["content"].startswith("OK. I'll"):
            content = m["content"]
            action_count = len(re.findall(r'<action>', content))
            if action_count > 1:
                multi_action += 1
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
    """Total assistant response length."""
    total = 0
    for m in messages:
        if m["role"] == "assistant":
            total += len(m["content"])
    return total


def load_trajectories(env, method, step):
    """Load trajectory JSONL."""
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


def section_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# ====================== ANALYSIS SECTIONS ======================

def analyze_step1_capability():
    """Section 1: Step-1 baseline capability."""
    section_header("1. STEP-1 CAPABILITY ANALYSIS")

    for env in ENVS:
        print(f"\n--- {env.upper()} ---\n")
        print(f"{'Method':<12} {'Success%':>8} {'MeanRwd':>8} {'AvgActs':>8} {'AvgRspLen':>10} {'InvalidActs':>11} {'N':>5}")
        print("-" * 70)

        for method in METHODS:
            entries = load_trajectories(env, method, 1)
            if entries is None:
                print(f"{method:<12} {'N/A':>8}")
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            successes = sum(1 for e in on_policy if e["success"])
            rewards = [e["reward"]["outcome"] for e in on_policy]
            mean_reward = sum(rewards) / len(rewards)
            action_counts = [count_actions_from_messages(e["messages"]) for e in on_policy]
            avg_actions = sum(action_counts) / len(action_counts)
            resp_lengths = [analyze_response_length(e["messages"]) for e in on_policy]
            avg_resp_len = sum(resp_lengths) / len(resp_lengths)
            invalid_counts = [count_invalid_actions_messages(e["messages"]) for e in on_policy]
            avg_invalid = sum(invalid_counts) / len(invalid_counts)
            success_pct = successes / len(on_policy) * 100

            print(f"{method:<12} {success_pct:>7.1f}% {mean_reward:>8.3f} {avg_actions:>8.1f} {avg_resp_len:>10.0f} {avg_invalid:>11.1f} {len(on_policy):>5}")


def analyze_failure_modes():
    """Section 2: Failure mode evolution across training."""
    section_header("2. FAILURE MODE EVOLUTION")

    for env in ENVS:
        print(f"\n{'='*60}")
        print(f"  {env.upper()} FAILURE MODES")
        print(f"{'='*60}")

        for method in METHODS:
            print(f"\n--- {method.upper()} ---")
            print(f"{'Step':>5} {'RepLoop%':>9} {'FmtErr%':>8} {'CJK%':>6} {'ThinkRp%':>9} {'AvgInval':>9} {'N':>5}")
            print("-" * 55)

            max_step = 50 if method == "sft_rl" else 100
            steps_to_check = [s for s in ANALYSIS_STEPS if s <= max_step]

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
                cjk_count = 0
                think_reps = 0
                total_invalid = 0

                for e in on_policy:
                    actions = extract_actions_from_messages(e["messages"])
                    is_rep, _ = detect_repetition_loop(actions)
                    if is_rep:
                        rep_loops += 1

                    fe, ma = detect_format_errors(e["messages"])
                    if fe > 0 or ma > 0:
                        fmt_errors += 1

                    full_text = " ".join(m["content"] for m in e["messages"] if m["role"] == "assistant")
                    has_cjk, _ = detect_cjk(full_text)
                    if has_cjk:
                        cjk_count += 1

                    has_think_rep, _ = detect_think_repetition(e["messages"])
                    if has_think_rep:
                        think_reps += 1

                    total_invalid += count_invalid_actions_messages(e["messages"])

                avg_invalid = total_invalid / n
                print(f"{step:>5} {rep_loops/n*100:>8.1f}% {fmt_errors/n*100:>7.1f}% {cjk_count/n*100:>5.1f}% {think_reps/n*100:>8.1f}% {avg_invalid:>9.1f} {n:>5}")


def analyze_training_evolution():
    """Section 3: Training success evolution + teacher comparison."""
    section_header("3. TRAINING SUCCESS EVOLUTION")

    for env in ENVS:
        # On-policy success rate table
        print(f"\n--- {env.upper()}: On-Policy Success Rate (%) ---\n")
        header = f"{'Step':>5}"
        for method in METHODS:
            header += f" {method:>10}"
        print(header)
        print("-" * (5 + 11 * len(METHODS)))

        for step in ANALYSIS_STEPS:
            row = f"{step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'--':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'--':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'--':>10}"
                    continue
                sr = sum(1 for e in on_policy if e["success"]) / len(on_policy) * 100
                row += f" {sr:>9.1f}%"
            print(row)

        # Mean reward table
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
                    row += f" {'--':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'--':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'--':>10}"
                    continue
                mr = sum(e["reward"]["outcome"] for e in on_policy) / len(on_policy)
                row += f" {mr:>10.3f}"
            print(row)

        # Teacher vs on-policy
        print(f"\n--- {env.upper()}: Teacher vs On-Policy in DUET ---\n")
        print(f"{'Step':>5} {'OnPol_SR%':>10} {'OnPol_Rwd':>10} {'Tchr_SR%':>10} {'Tchr_Rwd':>10} {'N_tchr':>7}")
        print("-" * 55)

        for step in ANALYSIS_STEPS:
            entries = load_trajectories(env, "duet", step)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            teachers = [e for e in entries if e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            op_sr = sum(1 for e in on_policy if e["success"]) / len(on_policy) * 100
            op_rwd = sum(e["reward"]["outcome"] for e in on_policy) / len(on_policy)

            if teachers:
                t_sr = sum(1 for e in teachers if e["success"]) / len(teachers) * 100
                t_rwd = sum(e["reward"]["outcome"] for e in teachers) / len(teachers)
                t_n = len(teachers)
                print(f"{step:>5} {op_sr:>9.1f}% {op_rwd:>10.3f} {t_sr:>9.1f}% {t_rwd:>10.3f} {t_n:>7}")
            else:
                print(f"{step:>5} {op_sr:>9.1f}% {op_rwd:>10.3f} {'--':>10} {'--':>10} {'0':>7}")


def analyze_validation():
    """Section 4: Validation analysis with task type breakdown."""
    section_header("4. VALIDATION ANALYSIS")

    # ALFWorld validation scores
    print("\n--- ALFWORLD VALIDATION SCORES ---\n")
    print(f"{'Method':<12} {'Step':>5} {'Success%':>9} {'N':>5}")
    print("-" * 35)

    for method in METHODS_FULL:
        for step in [50, 100]:
            val = load_validation("alfworld", method, step)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            success = sum(1 for s in scores if s > 0)
            print(f"{method:<12} {step:>5} {success/len(scores)*100:>8.1f}% {len(scores):>5}")

    # ALFWorld by task type (from trajectories, since validation doesn't have clear task types)
    print(f"\n--- ALFWORLD TASK TYPE BREAKDOWN (Training Step 50, On-Policy) ---\n")

    task_type_data = {}
    for method in METHODS:
        entries = load_trajectories("alfworld", method, 50)
        if entries is None:
            continue
        on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
        type_counts = defaultdict(lambda: {"success": 0, "total": 0})

        for e in on_policy:
            task_goal = extract_task_goal_from_messages(e["messages"])
            if task_goal:
                ttype = detect_alfworld_task_type_from_task_str(task_goal)
            else:
                ttype = "unknown"
            type_counts[ttype]["total"] += 1
            if e["success"]:
                type_counts[ttype]["success"] += 1

        task_type_data[method] = dict(type_counts)

    # Get all types
    all_types = set()
    for m in task_type_data:
        all_types.update(task_type_data[m].keys())
    all_types = sorted(all_types)

    header = f"{'Type':<10}"
    for method in METHODS:
        if method in task_type_data:
            header += f" {method:>12}"
    print(header)
    print("-" * (10 + 13 * len(task_type_data)))

    for ttype in all_types:
        row = f"{ttype:<10}"
        for method in METHODS:
            if method not in task_type_data:
                continue
            data = task_type_data[method].get(ttype, {"success": 0, "total": 0})
            if data["total"] > 0:
                pct = data["success"] / data["total"] * 100
                row += f" {pct:>5.0f}%({data['total']:>2})"
            else:
                row += f" {'--':>12}"
        print(row)

    # WebShop validation
    print(f"\n--- WEBSHOP VALIDATION SCORES ---\n")
    print(f"{'Method':<12} {'Step':>5} {'MeanScore':>10} {'Score>0%':>9} {'Score>=0.5':>11} {'N':>5}")
    print("-" * 58)

    for method in METHODS_FULL:
        for step in [50, 100]:
            val = load_validation("webshop", method, step)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            mean_score = sum(scores) / len(scores)
            pos = sum(1 for s in scores if s > 0) / len(scores) * 100
            high = sum(1 for s in scores if s >= 0.5) / len(scores) * 100
            print(f"{method:<12} {step:>5} {mean_score:>10.4f} {pos:>8.1f}% {high:>10.1f}% {len(scores):>5}")

    # WebShop score distribution
    print(f"\n--- WEBSHOP SCORE DISTRIBUTION (Step 50 and Step 100) ---\n")
    for step in [50, 100]:
        print(f"  Step {step}:")
        bins = [(-0.2, 0), (0, 0.001), (0.001, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001)]
        bin_labels = ["<0", "=0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

        header = f"  {'Method':<12}"
        for bl in bin_labels:
            header += f" {bl:>7}"
        print(header)
        print("  " + "-" * (12 + 8 * len(bin_labels)))

        for method in METHODS_FULL:
            val = load_validation("webshop", method, step)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            row = f"  {method:<12}"
            for lo, hi in bins:
                count = sum(1 for s in scores if lo <= s < hi)
                row += f" {count:>7}"
            print(row)
        print()


def analyze_duet_behavioral_advantages():
    """Section 5: DUET vs others on key failure modes."""
    section_header("5. DUET BEHAVIORAL ADVANTAGES - CROSS-METHOD COMPARISON")

    # Collect all data into comparison tables
    for env in ENVS:
        print(f"\n{'='*60}")
        print(f"  {env.upper()}: Failure Mode Rates Across Methods and Steps")
        print(f"{'='*60}")

        # --- CJK Collapse ---
        print(f"\n  CJK Collapse Rate (on-policy):")
        header = f"  {'Step':>5}"
        for m in METHODS:
            header += f" {m:>10}"
        print(header)
        print("  " + "-" * (5 + 11 * len(METHODS)))

        for step in [1, 10, 30, 50, 70, 100]:
            row = f"  {step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'--':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'--':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'--':>10}"
                    continue
                cjk_n = 0
                for e in on_policy:
                    text = " ".join(m_["content"] for m_ in e["messages"] if m_["role"] == "assistant")
                    has, _ = detect_cjk(text)
                    if has:
                        cjk_n += 1
                row += f" {cjk_n/len(on_policy)*100:>9.1f}%"
            print(row)

        # --- Repetition Loops ---
        print(f"\n  Repetition Loop Rate (3+ consecutive same action, on-policy):")
        header = f"  {'Step':>5}"
        for m in METHODS:
            header += f" {m:>10}"
        print(header)
        print("  " + "-" * (5 + 11 * len(METHODS)))

        for step in [1, 10, 30, 50, 70, 100]:
            row = f"  {step:>5}"
            for method in METHODS:
                if method == "sft_rl" and step > 50:
                    row += f" {'--':>10}"
                    continue
                entries = load_trajectories(env, method, step)
                if entries is None:
                    row += f" {'--':>10}"
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    row += f" {'--':>10}"
                    continue
                rep_n = 0
                for e in on_policy:
                    actions = extract_actions_from_messages(e["messages"])
                    is_rep, _ = detect_repetition_loop(actions)
                    if is_rep:
                        rep_n += 1
                row += f" {rep_n/len(on_policy)*100:>9.1f}%"
            print(row)

        # --- Late-training stability (step 100 summary) ---
        if env == "webshop":
            print(f"\n  Late-Training Stability (Step 100, On-Policy):")
            print(f"  {'Method':<12} {'RepLoop%':>9} {'FmtErr%':>8} {'CJK%':>6} {'ThinkRp%':>9} {'Success%':>9}")
            print("  " + "-" * 55)

            for method in ["onpolicy", "luffy", "chord", "duet"]:
                entries = load_trajectories(env, method, 100)
                if entries is None:
                    continue
                on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
                if not on_policy:
                    continue
                n = len(on_policy)
                rep = cjk = fmt = think = suc = 0
                for e in on_policy:
                    actions = extract_actions_from_messages(e["messages"])
                    is_rep, _ = detect_repetition_loop(actions)
                    if is_rep: rep += 1
                    text = " ".join(m_["content"] for m_ in e["messages"] if m_["role"] == "assistant")
                    has, _ = detect_cjk(text)
                    if has: cjk += 1
                    fe, ma = detect_format_errors(e["messages"])
                    if fe > 0 or ma > 0: fmt += 1
                    has_tr, _ = detect_think_repetition(e["messages"])
                    if has_tr: think += 1
                    if e["success"]: suc += 1
                print(f"  {method:<12} {rep/n*100:>8.1f}% {fmt/n*100:>7.1f}% {cjk/n*100:>5.1f}% {think/n*100:>8.1f}% {suc/n*100:>8.1f}%")


def analyze_sft_rl():
    """Section 6: SFT+RL analysis."""
    section_header("6. SFT+RL BEHAVIORAL ANALYSIS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: SFT+RL Training Evolution ---\n")
        print(f"  {'Step':>5} {'Suc%':>7} {'MeanRwd':>8} {'AvgActs':>8} {'RepLoop%':>9} {'CJK%':>6} {'Inval%':>7}")
        print("  " + "-" * 55)

        for step in [1, 5, 10, 20, 30, 50]:
            entries = load_trajectories(env, "sft_rl", step)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue
            n = len(on_policy)
            suc = sum(1 for e in on_policy if e["success"]) / n * 100
            rwd = sum(e["reward"]["outcome"] for e in on_policy) / n
            acts = sum(count_actions_from_messages(e["messages"]) for e in on_policy) / n
            rep = cjk = total_invalid = total_acts = 0
            for e in on_policy:
                actions = extract_actions_from_messages(e["messages"])
                is_rep, _ = detect_repetition_loop(actions)
                if is_rep: rep += 1
                text = " ".join(m_["content"] for m_ in e["messages"] if m_["role"] == "assistant")
                has, _ = detect_cjk(text)
                if has: cjk += 1
                total_invalid += count_invalid_actions_messages(e["messages"])
                total_acts += count_actions_from_messages(e["messages"])

            inval_pct = total_invalid / total_acts * 100 if total_acts > 0 else 0
            print(f"  {step:>5} {suc:>6.1f}% {rwd:>8.3f} {acts:>8.1f} {rep/n*100:>8.1f}% {cjk/n*100:>5.1f}% {inval_pct:>6.1f}%")

        # Validation comparison
        print(f"\n  Validation Comparison (Step 50):")
        print(f"  {'Method':<12}", end="")
        if env == "alfworld":
            print(f"{'Success%':>9}")
        else:
            print(f"{'MeanScore':>10} {'Score>=0.5%':>11}")
        print("  " + "-" * 35)

        for name in ["sft", "sft_rl", "duet", "onpolicy", "luffy", "chord"]:
            val = load_validation(env, name, 50)
            if val is None:
                continue
            scores = [e["score"] for e in val]
            if env == "alfworld":
                suc = sum(1 for s in scores if s > 0) / len(scores) * 100
                print(f"  {name:<12} {suc:>8.1f}%")
            else:
                mean_s = sum(scores) / len(scores)
                high = sum(1 for s in scores if s >= 0.5) / len(scores) * 100
                print(f"  {name:<12} {mean_s:>10.4f} {high:>10.1f}%")


def generate_case_studies():
    """Section 7: Qualitative case studies from trajectories."""
    section_header("7. QUALITATIVE CASE STUDIES")

    # ---- Case Study 1: DUET success examples ----
    print("--- Case 1: DUET Successful Trajectories (ALFWorld, Step 50) ---\n")

    duet_50 = load_trajectories("alfworld", "duet", 50)
    if duet_50:
        on_policy = [e for e in duet_50 if not e["diag"].get("is_teacher", False)]
        successes = [e for e in on_policy if e["success"]]

        print(f"DUET on-policy successes at step 50: {len(successes)}/{len(on_policy)}")

        for i, e in enumerate(successes[:2]):
            task_goal = extract_task_goal_from_messages(e["messages"])
            ttype = detect_alfworld_task_type_from_task_str(task_goal) if task_goal else "unknown"
            actions = extract_actions_from_messages(e["messages"])
            n_invalid = count_invalid_actions_messages(e["messages"])

            print(f"\n  Example {i+1}: task_id={e['task_id']}, type={ttype}")
            print(f"  Goal: {task_goal}")
            print(f"  Actions: {len(actions)}, Invalid: {n_invalid}")
            for j, a in enumerate(actions[:10]):
                print(f"    [{j+1}] {a[:100]}")
            if len(actions) > 10:
                print(f"    ... ({len(actions) - 10} more)")

    # ---- Case Study 2: DUET vs OnPolicy same task ----
    print("\n\n--- Case 2: DUET vs OnPolicy on Same Task (ALFWorld, Step 50) ---\n")

    onpol_50 = load_trajectories("alfworld", "onpolicy", 50)
    if duet_50 and onpol_50:
        # Build task_id maps
        duet_by_task = {}
        for e in duet_50:
            if not e["diag"].get("is_teacher", False):
                duet_by_task.setdefault(e["task_id"], []).append(e)

        onpol_by_task = {}
        for e in onpol_50:
            if not e["diag"].get("is_teacher", False):
                onpol_by_task.setdefault(e["task_id"], []).append(e)

        # Find task where DUET succeeds but OnPolicy fails
        comparison_found = False
        for tid in duet_by_task:
            if tid not in onpol_by_task:
                continue
            duet_entries = duet_by_task[tid]
            onpol_entries = onpol_by_task[tid]

            duet_any_success = any(e["success"] for e in duet_entries)
            onpol_any_success = any(e["success"] for e in onpol_entries)

            if duet_any_success and not onpol_any_success:
                # Found a comparison case
                d_e = next(e for e in duet_entries if e["success"])
                o_e = onpol_entries[0]

                task_goal = extract_task_goal_from_messages(d_e["messages"])
                ttype = detect_alfworld_task_type_from_task_str(task_goal) if task_goal else "unknown"

                print(f"  Task ID: {tid}, Type: {ttype}")
                print(f"  Goal: {task_goal}")

                d_actions = extract_actions_from_messages(d_e["messages"])
                o_actions = extract_actions_from_messages(o_e["messages"])

                print(f"\n  DUET (SUCCESS, reward={d_e['reward']['outcome']:.2f}):")
                print(f"    {len(d_actions)} actions, {count_invalid_actions_messages(d_e['messages'])} invalid")
                for j, a in enumerate(d_actions[:8]):
                    print(f"    [{j+1}] {a[:100]}")
                if len(d_actions) > 8:
                    print(f"    ... ({len(d_actions) - 8} more)")

                print(f"\n  OnPolicy (FAIL, reward={o_e['reward']['outcome']:.2f}):")
                print(f"    {len(o_actions)} actions, {count_invalid_actions_messages(o_e['messages'])} invalid")
                for j, a in enumerate(o_actions[:8]):
                    print(f"    [{j+1}] {a[:100]}")
                if len(o_actions) > 8:
                    print(f"    ... ({len(o_actions) - 8} more)")

                comparison_found = True
                break

        if not comparison_found:
            print("  No task found where DUET succeeds but OnPolicy fails at step 50.")

            # Try finding any shared task with different outcomes
            for tid in duet_by_task:
                if tid not in onpol_by_task:
                    continue
                duet_entries = duet_by_task[tid]
                onpol_entries = onpol_by_task[tid]
                d_suc = sum(1 for e in duet_entries if e["success"])
                o_suc = sum(1 for e in onpol_entries if e["success"])
                if d_suc > o_suc:
                    print(f"  Best comparison: task_id={tid}, DUET success {d_suc}/{len(duet_entries)}, OnPol success {o_suc}/{len(onpol_entries)}")
                    break

    # ---- Case Study 3: WebShop comparison ----
    print("\n\n--- Case 3: WebShop DUET vs OnPolicy Search Strategies (Step 50) ---\n")

    duet_web = load_trajectories("webshop", "duet", 50)
    onpol_web = load_trajectories("webshop", "onpolicy", 50)

    if duet_web and onpol_web:
        duet_by_task_w = {}
        for e in duet_web:
            if not e["diag"].get("is_teacher", False):
                duet_by_task_w.setdefault(e["task_id"], []).append(e)

        onpol_by_task_w = {}
        for e in onpol_web:
            if not e["diag"].get("is_teacher", False):
                onpol_by_task_w.setdefault(e["task_id"], []).append(e)

        # Find cases where DUET gets higher reward
        duet_wins_web = []
        for tid in duet_by_task_w:
            if tid not in onpol_by_task_w:
                continue
            d_best = max(e["reward"]["outcome"] for e in duet_by_task_w[tid])
            o_best = max(e["reward"]["outcome"] for e in onpol_by_task_w[tid])
            if d_best > o_best + 0.1:
                duet_wins_web.append((tid, d_best, o_best))

        duet_wins_web.sort(key=lambda x: x[1] - x[2], reverse=True)

        print(f"Tasks where DUET outperforms OnPolicy by >0.1 reward: {len(duet_wins_web)}")

        for case_idx, (tid, d_rwd, o_rwd) in enumerate(duet_wins_web[:2]):
            d_e = max(duet_by_task_w[tid], key=lambda e: e["reward"]["outcome"])
            o_e = max(onpol_by_task_w[tid], key=lambda e: e["reward"]["outcome"])

            # Extract instruction
            for m in d_e["messages"]:
                if m["role"] == "user" and "Instruction:" in m["content"]:
                    instr = m["content"]
                    idx = instr.find("Instruction:")
                    end = instr.find("[SEP]", idx + 12)
                    if end > 0:
                        instr_text = instr[idx:end].strip()
                    else:
                        instr_text = instr[idx:idx+200].strip()
                    break
            else:
                instr_text = "(instruction not found)"

            print(f"\n  Case {case_idx + 1}: task_id={tid}")
            print(f"  Instruction: {instr_text[:200]}")
            print(f"  DUET reward: {d_rwd:.3f}, OnPolicy reward: {o_rwd:.3f}")

            d_actions = extract_actions_from_messages(d_e["messages"])
            o_actions = extract_actions_from_messages(o_e["messages"])

            print(f"\n  DUET ({len(d_actions)} actions):")
            for j, a in enumerate(d_actions[:5]):
                print(f"    [{j+1}] {a[:120]}")

            print(f"\n  OnPolicy ({len(o_actions)} actions):")
            for j, a in enumerate(o_actions[:5]):
                print(f"    [{j+1}] {a[:120]}")

    # ---- Case Study 4: OnPolicy collapse at step 100 ----
    print("\n\n--- Case 4: OnPolicy Collapse at Step 100 (WebShop) ---\n")

    onpol_100 = load_trajectories("webshop", "onpolicy", 100)
    if onpol_100:
        on_policy = [e for e in onpol_100 if not e["diag"].get("is_teacher", False)]

        # Find CJK collapse examples
        cjk_examples = []
        for e in on_policy:
            text = " ".join(m["content"] for m in e["messages"] if m["role"] == "assistant")
            has_cjk, n_chars = detect_cjk(text)
            if has_cjk:
                cjk_examples.append((e, n_chars))

        # Find format error examples
        fmt_examples = []
        for e in on_policy:
            fe, ma = detect_format_errors(e["messages"])
            if fe > 0 or ma > 0:
                fmt_examples.append((e, fe + ma))

        # Find think repetition examples
        think_examples = []
        for e in on_policy:
            has_tr, n_think = detect_think_repetition(e["messages"])
            if has_tr:
                think_examples.append((e, n_think))

        print(f"OnPolicy step 100 (n={len(on_policy)}):")
        print(f"  CJK collapse: {len(cjk_examples)} trajectories")
        print(f"  Format errors: {len(fmt_examples)} trajectories")
        print(f"  Think repetition: {len(think_examples)} trajectories")
        print(f"  Success rate: {sum(1 for e in on_policy if e['success'])/len(on_policy)*100:.1f}%")

        if cjk_examples:
            e, n = cjk_examples[0]
            print(f"\n  CJK collapse example (task_id={e['task_id']}, {n} CJK chars):")
            for m in e["messages"]:
                if m["role"] == "assistant" and not m["content"].startswith("OK. I'll"):
                    text = m["content"]
                    has, _ = detect_cjk(text)
                    if has:
                        print(f"    Assistant: {text[:200]}")
                        break

        if think_examples:
            e, n = sorted(think_examples, key=lambda x: -x[1])[0]
            print(f"\n  Think repetition example (task_id={e['task_id']}, {n} </think> tags):")
            for m in e["messages"]:
                if m["role"] == "assistant":
                    tc = m["content"].count("</think>")
                    if tc > 2:
                        print(f"    Assistant (first 300 chars): {m['content'][:300]}")
                        break


def analyze_head_to_head():
    """Section 8: Head-to-head on matched tasks."""
    section_header("8. HEAD-TO-HEAD VALIDATION ANALYSIS")

    for env in ENVS:
        for step in [50, 100]:
            val_data = {}
            for method in METHODS_FULL:
                val = load_validation(env, method, step)
                if val:
                    val_data[method] = val

            if len(val_data) < 2:
                continue

            print(f"\n--- {env.upper()} Step {step}: Validation Scores Summary ---\n")

            # Since validation entries are in the same order (same task set),
            # match by index
            n_tasks = min(len(v) for v in val_data.values())

            if "duet" in val_data:
                # Compare DUET against others entry by entry
                print(f"  DUET vs Others (matched by entry index, n={n_tasks}):")
                print(f"  {'Opponent':<12} {'DUET_Win':>9} {'Tie':>5} {'Opp_Win':>9} {'AvgGap':>8}")
                print("  " + "-" * 48)

                for other in ["onpolicy", "luffy", "chord", "sft", "sft_rl"]:
                    if other not in val_data:
                        continue
                    win = tie = loss = 0
                    total_gap = 0

                    for i in range(n_tasks):
                        d_score = val_data["duet"][i]["score"]
                        o_score = val_data[other][i]["score"]

                        if env == "alfworld":
                            if d_score > 0 and o_score <= 0:
                                win += 1
                            elif d_score <= 0 and o_score > 0:
                                loss += 1
                            else:
                                tie += 1
                        else:
                            gap = d_score - o_score
                            if gap > 0.05:
                                win += 1
                            elif gap < -0.05:
                                loss += 1
                            else:
                                tie += 1
                        total_gap += d_score - o_score

                    avg_gap = total_gap / n_tasks
                    print(f"  DUET vs {other:<8} {win:>9} {tie:>5} {loss:>9} {avg_gap:>+8.3f}")


def analyze_advantage_diagnostics():
    """Section 9: Advantage and diagnostic analysis."""
    section_header("9. ADVANTAGE AND DIAGNOSTIC DISTRIBUTIONS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: Training Diagnostics (Step 50) ---\n")
        print(f"{'Method':<12} {'AdvMean':>8} {'AdvStd':>8} {'LogP':>9} {'RwdSum':>8} {'ValidTok':>9}")
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

            print(f"{method:<12} {sum(adv_means)/max(len(adv_means),1):>8.4f} {sum(adv_stds)/max(len(adv_stds),1):>8.4f} {sum(logp_means)/max(len(logp_means),1):>9.4f} {sum(rwd_sums)/max(len(rwd_sums),1):>8.3f} {sum(valid_toks)/len(valid_toks):>9.0f}")


def analyze_invalid_actions():
    """Section 10: Invalid action patterns."""
    section_header("10. INVALID ACTION ANALYSIS")

    for env in ENVS:
        print(f"\n--- {env.upper()}: Invalid Action Analysis (Step 50, On-Policy) ---\n")
        print(f"{'Method':<12} {'TrajW/Inv%':>11} {'InvalActs':>10} {'TotalActs':>10} {'InvalRate':>10}")
        print("-" * 58)

        for method in METHODS:
            entries = load_trajectories(env, method, 50)
            if entries is None:
                continue
            on_policy = [e for e in entries if not e["diag"].get("is_teacher", False)]
            if not on_policy:
                continue

            total_actions = 0
            total_invalid = 0
            traj_with_inv = 0

            for e in on_policy:
                n_act = count_actions_from_messages(e["messages"])
                n_inv = count_invalid_actions_messages(e["messages"])
                total_actions += n_act
                total_invalid += n_inv
                if n_inv > 0:
                    traj_with_inv += 1

            traj_pct = traj_with_inv / len(on_policy) * 100
            inv_rate = total_invalid / total_actions * 100 if total_actions > 0 else 0
            print(f"{method:<12} {traj_pct:>10.1f}% {total_invalid:>10} {total_actions:>10} {inv_rate:>9.1f}%")


def summary_findings():
    """Final summary of key findings."""
    section_header("SUMMARY OF KEY FINDINGS")

    print("""
1. STEP-1 BASELINE:
   - 1.5B models start at 0% success on ALFWorld (all methods except SFT+RL).
   - SFT+RL starts at 39.1% ALFWorld / 85.9% WebShop thanks to SFT warmstart.
   - On WebShop, DUET shows highest step-1 on-policy success (17.5%) among
     non-SFT methods, though LUFFY starts at 0% (possibly noise).
   - Average 18-19 invalid actions per ALFWorld trajectory at step 1 -- the model
     cannot even produce valid actions consistently.

2. FAILURE MODES:
   - ALFWorld: Repetition loops are the dominant failure mode (42% at step 1).
     DUET reduces this to 5.4% by step 50, better than OnPolicy (32.8%).
   - WebShop OnPolicy: Catastrophic collapse at step 100 -- 70.3% format errors,
     15.6% CJK, 64% think repetition. Success drops to 23.4%.
   - DUET WebShop step 100: 0% repetition loops, 1.7% format errors, 0% CJK,
     94.8% success. DUET avoids the late-training collapse entirely.

3. TRAINING EVOLUTION:
   - ALFWorld: DUET peaks at 39.3% on-policy success (steps 50-70),
     matching LUFFY's peak. OnPolicy peaks at 26.6% then crashes to 0%.
   - WebShop: DUET reaches 94.8% by step 100 with steady improvement.
     OnPolicy collapses (23.4%). LUFFY reaches 100% but starts slower.
   - SFT+RL ALFWorld: Peaks at step 10 (73.4%) then degrades to 26.6% by step 50.
     RL phase actively harms ALFWorld performance after initial gains.
   - SFT+RL WebShop: Stable improvement, reaches 96.9% at step 50.

4. VALIDATION:
   - ALFWorld step 100: DUET 32.5% > CHORD 27.0% > LUFFY 5.5% > OnPolicy 1.0%.
     DUET is the only RL-only method that improves from step 50 to 100.
   - WebShop step 100: CHORD 60.3% > LUFFY 57.3% > DUET 54.9% > OnPolicy 15.2%.
   - SFT alone achieves 47.5% ALFWorld, but SFT+RL degrades to 30.0%.

5. DUET BEHAVIORAL ADVANTAGES:
   - Late-training stability: DUET avoids the catastrophic collapse that hits
     OnPolicy at step 100 (WebShop: 0% CJK vs 15.6%, 0% rep loops vs 0%).
   - ALFWorld step 100: OnPolicy has 76.6% repetition loops; DUET has 12.5%.
   - Invalid action rate at step 50 ALFWorld: DUET 22.0% < OnPolicy 28.6%.
   - DR3 + SC appear to provide a regularizing effect that prevents
     policy degradation in late training.

6. SFT+RL:
   - SFT gives the strongest initial policy (47.5% ALFWorld, 56.2% WebShop).
   - But SFT+RL (RL after SFT) degrades ALFWorld from 73.4% to 26.6%.
   - WebShop SFT+RL is best overall at step 50 (64.1%).
   - SFT warmstart is powerful but the RL phase can be destructive,
     especially for ALFWorld. DUET's from-scratch approach is more robust
     in the long run.
""")


# ====================== MAIN ======================

if __name__ == "__main__":
    print("=" * 80)
    print("  DUET 1.5B COMPREHENSIVE CASE-LEVEL ANALYSIS (v2)")
    print("  Environments: ALFWorld, WebShop")
    print("  Methods: OnPolicy, LUFFY, CHORD, DUET, SFT, SFT+RL")
    print("=" * 80)

    analyze_step1_capability()
    analyze_failure_modes()
    analyze_training_evolution()
    analyze_validation()
    analyze_duet_behavioral_advantages()
    analyze_sft_rl()
    generate_case_studies()
    analyze_head_to_head()
    analyze_advantage_diagnostics()
    analyze_invalid_actions()
    summary_findings()

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
