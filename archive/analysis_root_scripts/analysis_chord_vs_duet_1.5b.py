#!/usr/bin/env python3
"""
CHORD vs DUET 1.5B WebShop: Behavioral Generalization Gap Analysis
DUET train 0.602, val 0.549 vs CHORD train 0.576, val 0.603
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = "/data/home/qisheng/EvolAnalsis"

# ---- Data paths ----
VAL_PATHS = {
    "DUET_v1": f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/100.jsonl",
    "DUET_v2": f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v2/validation_log/100.jsonl",
    "CHORD":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl",
    "LUFFY":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_luffy/validation_log/100.jsonl",
}

TRAIN_PATHS = {
    "DUET_v1": f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_duet/Trajectory/trajectories_step_100.jsonl",
    "DUET_v2": f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_duet_v2/Trajectory/trajectories_step_100.jsonl",
    "CHORD":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_chord/Trajectory/trajectories_step_100.jsonl",
    "LUFFY":   f"{BASE}/checkpoints/agentevolver/webshop_qwen1.5b_luffy/Trajectory/trajectories_step_100.jsonl",
}

# ---- Parsing utilities ----

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def parse_validation_trajectory(entry):
    """Parse validation JSONL entry into structured trajectory."""
    output = entry.get("output", "")
    score = entry.get("score", 0)

    actions = []
    observations = []

    parts = output.split("assistant\n")
    for part in parts[1:]:
        # Extract action
        action = ""
        if "<action>" in part and "</action>" in part:
            act_start = part.index("<action>") + len("<action>")
            act_end = part.index("</action>")
            action = part[act_start:act_end].strip()
        elif "<action>" in part:
            act_start = part.index("<action>") + len("<action>")
            action = part[act_start:].strip()

        # Extract observation (after user\n)
        obs = ""
        if "user\n" in part:
            obs_start = part.index("user\n") + len("user\n")
            obs = part[obs_start:].strip()

        actions.append(action)
        observations.append(obs)

    return {
        "score": score,
        "actions": actions,
        "observations": observations,
        "input": entry.get("input", ""),
        "raw_output": output,
    }

def parse_training_trajectory(entry):
    """Parse training JSONL entry into structured trajectory."""
    msgs = entry.get("messages", [])
    diag = entry.get("diag", {})
    reward = entry.get("reward", {})

    actions = []
    observations = []

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "assistant":
            # Extract action from content
            if "<action>" in content:
                act_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
                if act_match:
                    actions.append(act_match.group(1).strip())
                else:
                    # action tag not closed
                    act_start = content.index("<action>") + len("<action>")
                    actions.append(content[act_start:].strip()[:200])
        elif role == "user" and not content.startswith("You are"):
            observations.append(content)

    outcome = reward.get("outcome", 0) if isinstance(reward, dict) else reward

    return {
        "task_id": entry.get("task_id"),
        "score": outcome,
        "actions": actions,
        "observations": observations,
        "is_teacher": diag.get("is_teacher", False),
        "sc_progress": diag.get("sc_progress", 0),
        "sc_bonus": diag.get("sc_bonus", 0),
        "reward_components": diag.get("reward_components", {}),
        "adv_mean": diag.get("adv_mean", 0),
    }

def extract_search_query(actions):
    """Get the first search query from action list."""
    for a in actions:
        if a.startswith("search[") and a.endswith("]"):
            return a[7:-1]
        elif a.startswith("search["):
            return a[7:]
    return None

def extract_instruction(text):
    """Extract the instruction from input/observation or output field."""
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)(?:\[SEP\]|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try from output field in case input is the system prompt
    match2 = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)(?:\[SEP\]|$)", text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return text[:200]

def extract_instruction_from_entry(entry):
    """Extract instruction trying both input and output fields."""
    # Try output first (has the actual user observations)
    output = entry.get("output", "")
    match = re.search(r"Instruction:\s*\[SEP\]\s*(.*?)(?:\[SEP\])", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to input
    return extract_instruction(entry.get("input", ""))

def classify_action(action):
    """Classify action type."""
    if action.startswith("search["):
        return "search"
    elif action.startswith("click[buy now]") or action == "click[buy now]":
        return "buy"
    elif action.startswith("click[back to search]") or action.startswith("click[Back to Search]"):
        return "back_search"
    elif action.startswith("click[< Prev]") or action.startswith("click[prev]"):
        return "prev"
    elif action.startswith("click[Next >]") or action.startswith("click[next]"):
        return "next"
    elif action.startswith("click[b") or action.startswith("click[B"):
        return "click_product"
    elif action.startswith("click["):
        return "click_option"
    else:
        return "other"

def has_cjk(text):
    """Check for CJK characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
            return True
    return False

def count_think_tags(text):
    """Count </think> tags — repetition signal."""
    return text.count("</think>")

def search_query_specificity(query):
    """Count words in search query as a proxy for specificity."""
    if not query:
        return 0
    return len(query.split())

# ---- Analysis Functions ----

def section_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def analysis_1_score_distributions():
    """Compare validation score distributions."""
    section_header("1. VALIDATION SCORE DISTRIBUTION COMPARISON (Step 100)")

    all_scores = {}
    for method, path in VAL_PATHS.items():
        data = load_jsonl(path)
        scores = [d["score"] for d in data]
        all_scores[method] = scores

    # Summary stats
    print(f"{'Method':<12} {'Mean':>7} {'Median':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'N':>5}")
    print("-" * 62)
    for method, scores in all_scores.items():
        scores_s = sorted(scores)
        mean = sum(scores) / len(scores)
        median = scores_s[len(scores_s)//2]
        std = (sum((s-mean)**2 for s in scores) / len(scores)) ** 0.5
        print(f"{method:<12} {mean:>7.4f} {median:>7.4f} {std:>7.4f} {min(scores):>7.4f} {max(scores):>7.4f} {len(scores):>5}")

    # Score bucket distribution
    buckets = [(0, 0.001, "=0"), (0.001, 0.3, "0-0.3"), (0.3, 0.5, "0.3-0.5"),
               (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.001, "0.9-1.0")]

    print(f"\nScore Distribution (% of 200 tasks):")
    print(f"{'Bucket':<10}", end="")
    for method in all_scores:
        print(f"{method:>12}", end="")
    print()
    print("-" * (10 + 12 * len(all_scores)))

    for lo, hi, label in buckets:
        print(f"{label:<10}", end="")
        for method, scores in all_scores.items():
            count = sum(1 for s in scores if lo <= s < hi)
            pct = count / len(scores) * 100
            print(f"{pct:>10.1f}% ", end="")
        print()

    # Completion rate (score > 0)
    print(f"\nCompletion Rates:")
    for method, scores in all_scores.items():
        nonzero = sum(1 for s in scores if s > 0.001)
        high = sum(1 for s in scores if s >= 0.7)
        perfect = sum(1 for s in scores if s >= 0.99)
        print(f"  {method}: {nonzero}/200 ({nonzero/2:.1f}%) scored > 0, "
              f"{high}/200 ({high/2:.1f}%) scored >= 0.7, "
              f"{perfect}/200 ({perfect/2:.1f}%) scored >= 1.0")

    return all_scores

def analysis_2_task_level_headtohead(all_val):
    """Task-by-task comparison."""
    section_header("2. TASK-LEVEL HEAD-TO-HEAD (Step 100)")

    methods_to_compare = [("DUET_v1", "CHORD"), ("DUET_v2", "CHORD"), ("LUFFY", "CHORD"), ("DUET_v1", "DUET_v2")]

    for method_a, method_b in methods_to_compare:
        data_a = load_jsonl(VAL_PATHS[method_a])
        data_b = load_jsonl(VAL_PATHS[method_b])

        # Match by position (same task ordering)
        wins_a = 0
        wins_b = 0
        ties = 0
        gaps = []

        for i in range(min(len(data_a), len(data_b))):
            sa = data_a[i]["score"]
            sb = data_b[i]["score"]
            gap = sa - sb
            gaps.append(gap)
            if gap > 0.05:
                wins_a += 1
            elif gap < -0.05:
                wins_b += 1
            else:
                ties += 1

        print(f"\n--- {method_a} vs {method_b} ---")
        print(f"  {method_a} wins: {wins_a}/{len(gaps)} ({wins_a/len(gaps)*100:.1f}%)")
        print(f"  {method_b} wins: {wins_b}/{len(gaps)} ({wins_b/len(gaps)*100:.1f}%)")
        print(f"  Ties (within 0.05): {ties}/{len(gaps)} ({ties/len(gaps)*100:.1f}%)")

        avg_gap = sum(gaps) / len(gaps)
        print(f"  Average gap ({method_a} - {method_b}): {avg_gap:+.4f}")

        # Where does method_b win big?
        big_losses = [(i, gaps[i]) for i in range(len(gaps)) if gaps[i] < -0.2]
        big_losses.sort(key=lambda x: x[1])

        big_wins = [(i, gaps[i]) for i in range(len(gaps)) if gaps[i] > 0.2]
        big_wins.sort(key=lambda x: -x[1])

        print(f"  Tasks where {method_b} wins by >0.2: {len(big_losses)}")
        print(f"  Tasks where {method_a} wins by >0.2: {len(big_wins)}")

    # Detailed DUET_v1 vs CHORD comparison
    print(f"\n--- DETAILED: Tasks where CHORD >> DUET_v1 (gap > 0.3) ---")
    data_d = load_jsonl(VAL_PATHS["DUET_v1"])
    data_c = load_jsonl(VAL_PATHS["CHORD"])

    chord_wins_big = []
    for i in range(min(len(data_d), len(data_c))):
        sd = data_d[i]["score"]
        sc = data_c[i]["score"]
        if sc - sd > 0.3:
            chord_wins_big.append((i, sd, sc, data_d[i], data_c[i]))

    chord_wins_big.sort(key=lambda x: x[2] - x[1], reverse=True)
    print(f"Found {len(chord_wins_big)} tasks where CHORD beats DUET by >0.3")

    # Analyze characteristics of these tasks
    if chord_wins_big:
        duet_zeros = sum(1 for _, sd, _, _, _ in chord_wins_big if sd < 0.01)
        chord_high = sum(1 for _, _, sc, _, _ in chord_wins_big if sc > 0.7)
        print(f"  Of these: DUET scores 0: {duet_zeros}, CHORD scores >0.7: {chord_high}")

        # Show top 5
        print(f"\n  Top 5 largest gaps:")
        for idx, (i, sd, sc, dd, dc) in enumerate(chord_wins_big[:5]):
            instr_d = extract_instruction_from_entry(dd)
            print(f"    Task {i}: DUET={sd:.3f}, CHORD={sc:.3f} (gap={sc-sd:.3f})")
            print(f"      Instruction: {instr_d[:120]}...")

    return chord_wins_big

def analysis_3_search_strategies():
    """Compare search query strategies."""
    section_header("3. SEARCH STRATEGY COMPARISON")

    # Validation search queries
    print("--- Validation (Step 100) ---")
    for method, path in VAL_PATHS.items():
        data = load_jsonl(path)
        queries = []
        for entry in data:
            traj = parse_validation_trajectory(entry)
            q = extract_search_query(traj["actions"])
            if q:
                queries.append(q)

        if not queries:
            print(f"  {method}: No search queries found")
            continue

        lengths = [len(q.split()) for q in queries]
        char_lengths = [len(q) for q in queries]
        avg_words = sum(lengths) / len(lengths)
        avg_chars = sum(char_lengths) / len(char_lengths)

        print(f"  {method}: {len(queries)} queries, avg {avg_words:.1f} words, avg {avg_chars:.0f} chars")

    # Training search queries — separate on-policy vs teacher
    print("\n--- Training (Step 100) ---")
    for method, path in TRAIN_PATHS.items():
        data = load_jsonl(path)
        on_queries = []
        teacher_queries = []
        for entry in data:
            traj = parse_training_trajectory(entry)
            q = extract_search_query(traj["actions"])
            if q:
                if traj["is_teacher"]:
                    teacher_queries.append(q)
                else:
                    on_queries.append(q)

        on_lengths = [len(q.split()) for q in on_queries] if on_queries else [0]
        t_lengths = [len(q.split()) for q in teacher_queries] if teacher_queries else [0]

        print(f"  {method}:")
        print(f"    On-policy: {len(on_queries)} queries, avg {sum(on_lengths)/max(len(on_lengths),1):.1f} words")
        if teacher_queries:
            print(f"    Teacher:   {len(teacher_queries)} queries, avg {sum(t_lengths)/max(len(t_lengths),1):.1f} words")

    # Compare actual query content: DUET vs CHORD on same tasks (training)
    print("\n--- Query Similarity: DUET vs CHORD (Training, same task_ids) ---")
    duet_data = load_jsonl(TRAIN_PATHS["DUET_v1"])
    chord_data = load_jsonl(TRAIN_PATHS["CHORD"])

    duet_queries_by_task = {}
    chord_queries_by_task = {}

    for entry in duet_data:
        traj = parse_training_trajectory(entry)
        if not traj["is_teacher"]:
            q = extract_search_query(traj["actions"])
            if q:
                duet_queries_by_task[traj["task_id"]] = q

    for entry in chord_data:
        traj = parse_training_trajectory(entry)
        if not traj.get("is_teacher", False):
            q = extract_search_query(traj["actions"])
            if q:
                chord_queries_by_task[traj["task_id"]] = q

    common_tasks = set(duet_queries_by_task.keys()) & set(chord_queries_by_task.keys())
    if common_tasks:
        exact_match = 0
        for tid in common_tasks:
            if duet_queries_by_task[tid].lower() == chord_queries_by_task[tid].lower():
                exact_match += 1
        print(f"  Common tasks: {len(common_tasks)}")
        print(f"  Exact query match: {exact_match}/{len(common_tasks)} ({exact_match/len(common_tasks)*100:.1f}%)")

        # Show some divergent examples
        print(f"\n  Sample divergent queries:")
        shown = 0
        for tid in sorted(common_tasks):
            dq = duet_queries_by_task[tid]
            cq = chord_queries_by_task[tid]
            if dq.lower() != cq.lower():
                print(f"    Task {tid}:")
                print(f"      DUET:  {dq[:100]}")
                print(f"      CHORD: {cq[:100]}")
                shown += 1
                if shown >= 5:
                    break

def analysis_4_action_sequences():
    """Compare action sequence structure."""
    section_header("4. ACTION SEQUENCE COMPARISON")

    for label, paths in [("Validation", VAL_PATHS), ("Training", TRAIN_PATHS)]:
        print(f"\n--- {label} (Step 100) ---")
        print(f"{'Method':<12} {'AvgActs':>8} {'BuyRate':>8} {'AvgScore':>9} {'CJK%':>6} {'ThinkRep':>9}")
        print("-" * 65)

        for method, path in paths.items():
            data = load_jsonl(path)

            n_actions_list = []
            buy_count = 0
            total = 0
            cjk_count = 0
            think_rep_count = 0
            scores = []
            action_type_counts = Counter()

            for entry in data:
                if label == "Validation":
                    traj = parse_validation_trajectory(entry)
                else:
                    traj = parse_training_trajectory(entry)
                    if traj["is_teacher"]:
                        continue

                total += 1
                actions = traj["actions"]
                n_actions_list.append(len(actions))
                scores.append(traj["score"])

                for a in actions:
                    action_type_counts[classify_action(a)] += 1
                    if has_cjk(a):
                        cjk_count += 1
                        break

                has_buy = any(a.startswith("click[buy now]") or a == "click[buy now]" for a in actions)
                if has_buy:
                    buy_count += 1

                raw = entry.get("output", "") if label == "Validation" else str(entry.get("messages", ""))
                if count_think_tags(raw) > 3:
                    think_rep_count += 1

            if total == 0:
                continue

            avg_acts = sum(n_actions_list) / total
            buy_rate = buy_count / total * 100
            avg_score = sum(scores) / total
            cjk_pct = cjk_count / total * 100
            think_pct = think_rep_count / total * 100

            print(f"{method:<12} {avg_acts:>8.1f} {buy_rate:>7.1f}% {avg_score:>9.4f} {cjk_pct:>5.1f}% {think_pct:>8.1f}%")

        # Action type breakdown for validation
        if label == "Validation":
            print(f"\n  Action Type Distribution (Validation):")
            for method, path in paths.items():
                data = load_jsonl(path)
                type_counts = Counter()
                total_acts = 0
                for entry in data:
                    traj = parse_validation_trajectory(entry)
                    for a in traj["actions"]:
                        t = classify_action(a)
                        type_counts[t] += 1
                        total_acts += 1

                print(f"  {method}: ", end="")
                for t in ["search", "click_product", "click_option", "buy", "back_search", "next", "other"]:
                    if type_counts[t] > 0:
                        print(f"{t}={type_counts[t]} ({type_counts[t]/total_acts*100:.0f}%) ", end="")
                print()

def analysis_5_training_task_diversity():
    """Training task diversity and teacher influence."""
    section_header("5. TRAINING TASK DIVERSITY & TEACHER INFLUENCE")

    for method, path in TRAIN_PATHS.items():
        data = load_jsonl(path)
        task_ids = set()
        teacher_count = 0
        on_policy_count = 0
        teacher_scores = []
        on_policy_scores = []

        for entry in data:
            traj = parse_training_trajectory(entry)
            task_ids.add(traj["task_id"])
            if traj["is_teacher"]:
                teacher_count += 1
                teacher_scores.append(traj["score"])
            else:
                on_policy_count += 1
                on_policy_scores.append(traj["score"])

        print(f"\n  {method}:")
        print(f"    Total trajectories: {len(data)}")
        print(f"    Unique task_ids: {len(task_ids)}")
        print(f"    Teacher: {teacher_count} ({teacher_count/len(data)*100:.1f}%)")
        print(f"    On-policy: {on_policy_count} ({on_policy_count/len(data)*100:.1f}%)")
        if teacher_scores:
            print(f"    Teacher avg score: {sum(teacher_scores)/len(teacher_scores):.4f}")
        if on_policy_scores:
            print(f"    On-policy avg score: {sum(on_policy_scores)/len(on_policy_scores):.4f}")

    # Task overlap between methods
    print(f"\n--- Task Overlap Between Methods (Step 100) ---")
    method_tasks = {}
    for method, path in TRAIN_PATHS.items():
        data = load_jsonl(path)
        tasks = set()
        for entry in data:
            tasks.add(entry.get("task_id"))
        method_tasks[method] = tasks

    for m1 in method_tasks:
        for m2 in method_tasks:
            if m1 < m2:
                overlap = method_tasks[m1] & method_tasks[m2]
                print(f"  {m1} & {m2}: {len(overlap)} shared tasks out of "
                      f"{len(method_tasks[m1])} / {len(method_tasks[m2])}")

    # SC and DR3 influence in DUET training
    print(f"\n--- DUET-specific: SC bonus & Advantage Distribution ---")
    for method in ["DUET_v1", "DUET_v2"]:
        if method not in TRAIN_PATHS:
            continue
        data = load_jsonl(TRAIN_PATHS[method])
        sc_bonuses = []
        adv_means = []
        on_scores = []
        for entry in data:
            traj = parse_training_trajectory(entry)
            if not traj["is_teacher"]:
                sc_bonuses.append(traj["sc_bonus"])
                adv_means.append(traj["adv_mean"])
                on_scores.append(traj["score"])

        if sc_bonuses:
            nonzero_sc = sum(1 for b in sc_bonuses if b > 0.001)
            avg_sc = sum(sc_bonuses) / len(sc_bonuses)
            print(f"\n  {method}:")
            print(f"    SC bonus: {nonzero_sc}/{len(sc_bonuses)} nonzero, avg={avg_sc:.4f}")
            print(f"    Advantage mean: {sum(adv_means)/len(adv_means):.4f}")
            print(f"    On-policy score: {sum(on_scores)/len(on_scores):.4f}")

            # Score vs SC bonus correlation
            if nonzero_sc > 0:
                sc_high = [s for s, b in zip(on_scores, sc_bonuses) if b > avg_sc]
                sc_low = [s for s, b in zip(on_scores, sc_bonuses) if b <= avg_sc]
                if sc_high and sc_low:
                    print(f"    High-SC samples avg score: {sum(sc_high)/len(sc_high):.4f} (n={len(sc_high)})")
                    print(f"    Low-SC samples avg score: {sum(sc_low)/len(sc_low):.4f} (n={len(sc_low)})")

def analysis_6_failure_case_studies(chord_wins_big):
    """Detailed failure analysis for tasks where CHORD >> DUET."""
    section_header("6. VALIDATION FAILURE CASE STUDIES")

    data_d = load_jsonl(VAL_PATHS["DUET_v1"])
    data_c = load_jsonl(VAL_PATHS["CHORD"])
    data_l = load_jsonl(VAL_PATHS["LUFFY"])

    # Find tasks where CHORD succeeds (>0.7) but DUET fails (<0.3)
    case_studies = []
    for i in range(min(len(data_d), len(data_c))):
        sd = data_d[i]["score"]
        sc = data_c[i]["score"]
        sl = data_l[i]["score"] if i < len(data_l) else None
        if sc > 0.6 and sd < 0.3:
            case_studies.append((i, sd, sc, sl, data_d[i], data_c[i], data_l[i] if i < len(data_l) else None))

    case_studies.sort(key=lambda x: x[2] - x[1], reverse=True)

    print(f"Found {len(case_studies)} tasks where CHORD > 0.6 AND DUET < 0.3\n")

    # Show up to 8 case studies
    for idx, (i, sd, sc, sl, dd, dc, dl) in enumerate(case_studies[:8]):
        traj_d = parse_validation_trajectory(dd)
        traj_c = parse_validation_trajectory(dc)
        traj_l = parse_validation_trajectory(dl) if dl else None

        instr = extract_instruction_from_entry(dd)

        sl_val = sl if sl is not None else 0.0
        print(f"--- Case {idx+1}: Task {i} | DUET={sd:.3f}, CHORD={sc:.3f}, LUFFY={sl_val:.3f} ---")
        print(f"  Instruction: {instr[:200]}")

        # DUET trajectory
        print(f"\n  DUET trajectory ({len(traj_d['actions'])} actions):")
        for j, a in enumerate(traj_d["actions"]):
            atype = classify_action(a)
            print(f"    [{j}] ({atype}) {a[:120]}")

        # Check for specific failures
        duet_issues = []
        if not any(a.startswith("click[buy now]") for a in traj_d["actions"]):
            duet_issues.append("NO_BUY")
        if len(traj_d["actions"]) <= 1:
            duet_issues.append("PREMATURE_TERMINATION")
        if any(has_cjk(a) for a in traj_d["actions"]):
            duet_issues.append("CJK_OUTPUT")
        raw_d = dd.get("output", "")
        if count_think_tags(raw_d) > 3:
            duet_issues.append("THINK_REPETITION")
        if duet_issues:
            print(f"  ** DUET Issues: {', '.join(duet_issues)}")

        # CHORD trajectory
        print(f"\n  CHORD trajectory ({len(traj_c['actions'])} actions):")
        for j, a in enumerate(traj_c["actions"]):
            atype = classify_action(a)
            print(f"    [{j}] ({atype}) {a[:120]}")

        # LUFFY trajectory
        if traj_l:
            print(f"\n  LUFFY trajectory ({len(traj_l['actions'])} actions):")
            for j, a in enumerate(traj_l["actions"]):
                atype = classify_action(a)
                print(f"    [{j}] ({atype}) {a[:120]}")

        # Diagnose the gap
        print(f"\n  DIAGNOSIS:")
        dq = extract_search_query(traj_d["actions"])
        cq = extract_search_query(traj_c["actions"])
        if dq and cq:
            if dq.lower() != cq.lower():
                print(f"    Search divergence: DUET='{dq[:80]}' vs CHORD='{cq[:80]}'")
            else:
                print(f"    Same search query")

        # Product selection comparison
        d_products = [a for a in traj_d["actions"] if classify_action(a) == "click_product"]
        c_products = [a for a in traj_c["actions"] if classify_action(a) == "click_product"]
        if d_products and c_products:
            if d_products[0] != c_products[0]:
                print(f"    Different product: DUET={d_products[0][:60]} vs CHORD={c_products[0][:60]}")
            else:
                print(f"    Same product selected")

        # Option selection comparison
        d_options = [a for a in traj_d["actions"] if classify_action(a) == "click_option"]
        c_options = [a for a in traj_c["actions"] if classify_action(a) == "click_option"]
        if d_options != c_options:
            print(f"    Options: DUET={d_options} vs CHORD={c_options}")

        print()

def analysis_7_v1_v2_comparison():
    """Compare DUET v1 (beta=0.2) vs v2 (beta=0.1)."""
    section_header("7. DUET v1 (beta=0.2) vs v2 (beta=0.1)")

    data_v1 = load_jsonl(VAL_PATHS["DUET_v1"])
    data_v2 = load_jsonl(VAL_PATHS["DUET_v2"])

    scores_v1 = [d["score"] for d in data_v1]
    scores_v2 = [d["score"] for d in data_v2]

    print(f"  v1 avg: {sum(scores_v1)/len(scores_v1):.4f}")
    print(f"  v2 avg: {sum(scores_v2)/len(scores_v2):.4f}")

    # Per-task comparison
    same_better = 0
    same_worse = 0
    same_tie = 0
    for i in range(min(len(data_v1), len(data_v2))):
        gap = scores_v2[i] - scores_v1[i]
        if gap > 0.05:
            same_better += 1
        elif gap < -0.05:
            same_worse += 1
        else:
            same_tie += 1

    print(f"\n  v2 better: {same_better}")
    print(f"  v2 worse:  {same_worse}")
    print(f"  Tie:       {same_tie}")

    # Do they fail on the same tasks?
    v1_fail = set(i for i in range(len(data_v1)) if scores_v1[i] < 0.1)
    v2_fail = set(i for i in range(len(data_v2)) if scores_v2[i] < 0.1)
    overlap_fail = v1_fail & v2_fail

    print(f"\n  v1 failures (<0.1): {len(v1_fail)}")
    print(f"  v2 failures (<0.1): {len(v2_fail)}")
    print(f"  Both fail:         {len(overlap_fail)}")
    print(f"  Only v1 fails:     {len(v1_fail - v2_fail)}")
    print(f"  Only v2 fails:     {len(v2_fail - v1_fail)}")

    # Failure mode comparison
    print(f"\n  Failure Mode Comparison:")
    for method, data, label in [(data_v1, scores_v1, "v1"), (data_v2, scores_v2, "v2")]:
        cjk = 0
        think_rep = 0
        no_buy = 0
        short_traj = 0

        for entry in method:
            traj = parse_validation_trajectory(entry)
            if any(has_cjk(a) for a in traj["actions"]):
                cjk += 1
            raw = entry.get("output", "")
            if count_think_tags(raw) > 3:
                think_rep += 1
            if not any(a.startswith("click[buy now]") for a in traj["actions"]):
                no_buy += 1
            if len(traj["actions"]) <= 1:
                short_traj += 1

        print(f"    {label}: CJK={cjk}, ThinkRep={think_rep}, NoBuy={no_buy}, Short={short_traj}")

def analysis_training_reward_trajectory():
    """Track training reward over steps for each method."""
    section_header("BONUS: TRAINING REWARD TRAJECTORY (Steps 1, 25, 50, 75, 100)")

    steps_to_check = [1, 10, 25, 50, 75, 100]

    for method_key in ["DUET_v1", "CHORD", "LUFFY"]:
        method_dir = TRAIN_PATHS[method_key].replace("trajectories_step_100.jsonl", "")
        print(f"\n  {method_key}:")
        for step in steps_to_check:
            fpath = f"{method_dir}trajectories_step_{step}.jsonl"
            if not os.path.exists(fpath):
                continue
            data = load_jsonl(fpath)
            on_scores = []
            teacher_scores = []
            for entry in data:
                traj = parse_training_trajectory(entry)
                if traj["is_teacher"]:
                    teacher_scores.append(traj["score"])
                else:
                    on_scores.append(traj["score"])

            on_avg = sum(on_scores)/len(on_scores) if on_scores else 0
            t_avg = sum(teacher_scores)/len(teacher_scores) if teacher_scores else 0
            print(f"    Step {step:>3}: on-policy={on_avg:.4f} (n={len(on_scores)}), teacher={t_avg:.4f} (n={len(teacher_scores)})")

def analysis_overfit_signal():
    """Look for overfitting signals: training vs validation score gap evolution."""
    section_header("BONUS: OVERFITTING ANALYSIS")

    # Compare step 50 vs 100 validation
    print("  Validation scores at Step 50 vs Step 100:")
    for method in ["DUET_v1", "DUET_v2", "CHORD", "LUFFY"]:
        path_50 = VAL_PATHS[method].replace("100.jsonl", "50.jsonl")
        path_100 = VAL_PATHS[method]

        if not os.path.exists(path_50):
            continue

        data_50 = load_jsonl(path_50)
        data_100 = load_jsonl(path_100)

        avg_50 = sum(d["score"] for d in data_50) / len(data_50)
        avg_100 = sum(d["score"] for d in data_100) / len(data_100)

        delta = avg_100 - avg_50
        print(f"    {method}: Step50={avg_50:.4f}, Step100={avg_100:.4f}, delta={delta:+.4f}")

    # Per-task: did any tasks get WORSE from step 50 to 100?
    print("\n  Per-task degradation (Step 50 -> 100):")
    for method in ["DUET_v1", "CHORD"]:
        path_50 = VAL_PATHS[method].replace("100.jsonl", "50.jsonl")
        path_100 = VAL_PATHS[method]

        if not os.path.exists(path_50):
            continue

        data_50 = load_jsonl(path_50)
        data_100 = load_jsonl(path_100)

        improved = 0
        degraded = 0
        stable = 0

        for i in range(min(len(data_50), len(data_100))):
            gap = data_100[i]["score"] - data_50[i]["score"]
            if gap > 0.05:
                improved += 1
            elif gap < -0.05:
                degraded += 1
            else:
                stable += 1

        print(f"    {method}: improved={improved}, degraded={degraded}, stable={stable}")

def analysis_detailed_failure_diagnosis():
    """Deep dive: exactly where in the pipeline do DUET failures happen?"""
    section_header("BONUS: FAILURE PIPELINE DIAGNOSIS (Validation Step 100)")

    for method in ["DUET_v1", "CHORD", "LUFFY"]:
        data = load_jsonl(VAL_PATHS[method])

        stage_fail = Counter()
        total = 0

        for entry in data:
            traj = parse_validation_trajectory(entry)
            total += 1
            actions = traj["actions"]
            action_types = [classify_action(a) for a in actions]
            score = traj["score"]

            if score >= 0.7:
                stage_fail["SUCCESS"] += 1
                continue

            # Where did it fail?
            if len(actions) == 0:
                stage_fail["NO_ACTIONS"] += 1
            elif "search" not in action_types:
                stage_fail["NO_SEARCH"] += 1
            elif "click_product" not in action_types:
                stage_fail["SEARCH_BUT_NO_PRODUCT_CLICK"] += 1
            elif "buy" not in action_types:
                if "click_option" in action_types:
                    stage_fail["SELECTED_OPTIONS_BUT_NO_BUY"] += 1
                else:
                    stage_fail["CLICKED_PRODUCT_BUT_NO_OPTIONS_OR_BUY"] += 1
            elif score < 0.3:
                stage_fail["BOUGHT_WRONG_PRODUCT"] += 1
            else:
                stage_fail["BOUGHT_PARTIAL_MATCH"] += 1

        print(f"\n  {method} (n={total}):")
        for stage, count in sorted(stage_fail.items(), key=lambda x: -x[1]):
            print(f"    {stage:<40} {count:>4} ({count/total*100:>5.1f}%)")

def analysis_repeat_think():
    """Examine the </think> repetition issue in detail."""
    section_header("BONUS: THINK TAG REPETITION ANALYSIS")

    for method in ["DUET_v1", "DUET_v2", "CHORD", "LUFFY"]:
        data = load_jsonl(VAL_PATHS[method])
        think_counts = []

        for entry in data:
            raw = entry.get("output", "")
            tc = count_think_tags(raw)
            think_counts.append(tc)

        avg_tc = sum(think_counts) / len(think_counts)
        max_tc = max(think_counts)
        over_3 = sum(1 for t in think_counts if t > 3)
        over_5 = sum(1 for t in think_counts if t > 5)

        print(f"  {method}: avg_think_tags={avg_tc:.1f}, max={max_tc}, >3_tags={over_3}, >5_tags={over_5}")

    # Show a DUET trajectory with many think tags
    print(f"\n  Example: DUET_v1 trajectory with most think tags:")
    data = load_jsonl(VAL_PATHS["DUET_v1"])
    worst_idx = max(range(len(data)), key=lambda i: count_think_tags(data[i].get("output", "")))
    worst = data[worst_idx]
    tc = count_think_tags(worst.get("output", ""))
    print(f"    Task {worst_idx}: score={worst['score']:.3f}, think_tags={tc}")
    traj = parse_validation_trajectory(worst)
    for j, a in enumerate(traj["actions"]):
        print(f"    [{j}] {a[:120]}")

# ---- Main ----

if __name__ == "__main__":
    print("=" * 80)
    print("  CHORD vs DUET 1.5B WebShop: Behavioral Generalization Gap Analysis")
    print("  DUET train=0.602, val=0.549 vs CHORD train=0.576, val=0.603")
    print("=" * 80)

    all_scores = analysis_1_score_distributions()
    chord_wins_big = analysis_2_task_level_headtohead(all_scores)
    analysis_3_search_strategies()
    analysis_4_action_sequences()
    analysis_5_training_task_diversity()
    analysis_6_failure_case_studies(chord_wins_big)
    analysis_7_v1_v2_comparison()
    analysis_training_reward_trajectory()
    analysis_overfit_signal()
    analysis_detailed_failure_diagnosis()
    analysis_repeat_think()

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
