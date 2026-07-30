#!/usr/bin/env python3
"""
v24 vs DUET-v1 ALFWorld trajectory diff analysis (step 100 validation).

Parses 200 matched trajectories, computes behavioral diff, emits case studies.
"""
import json
import re
import os
from collections import Counter, defaultdict

V24_PATH = "/data/home/qisheng/EvolAnalsis/experiments/alfworld/alfworld_qwen1.5b_duet_v24/validation_log/100.jsonl"
V1_PATH = "/data/home/qisheng/EvolAnalsis/experiments/alfworld/alfworld_qwen1.5b_duet/validation_log/100.jsonl"

# --- Parsing helpers --------------------------------------------------------

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def parse_trajectory(output):
    """
    Parse the output field into a list of turns.

    Each turn is (role, content). Splits on 'assistant\n' / 'user\n'.
    Returns list of dicts: {'role', 'text', 'action' (if role==assistant), 'think' (optional)}.
    """
    # Split on top-level role markers at line start
    # Use a regex that splits on \nassistant\n or \nuser\n
    parts = re.split(r"^(assistant|user)\n", output, flags=re.MULTILINE)
    # parts[0] is preamble (usually empty); then alternating role, text, role, text
    turns = []
    i = 1
    while i < len(parts):
        role = parts[i]
        text = parts[i+1] if i+1 < len(parts) else ""
        turn = {"role": role, "text": text}
        if role == "assistant":
            a = ACTION_RE.search(text)
            turn["action"] = a.group(1).strip() if a else None
            t = THINK_RE.search(text)
            turn["think"] = t.group(1).strip() if t else None
            turn["has_action_tag"] = bool(a)
        else:
            turn["observation"] = text
            # Pull available actions
            av = re.search(r"AVAILABLE ACTIONS:\s*(.*?)$", text, re.MULTILINE | re.DOTALL)
            turn["available"] = av.group(1).strip() if av else None
        turns.append(turn)
        i += 2
    return turns


def extract_task_signature(turns):
    """Task signature derived from first env observation + first think"""
    first_obs = turns[0]["observation"] if turns and turns[0]["role"] == "user" else ""
    # first env observation describes starting room layout
    first_think = turns[1].get("think", "") if len(turns) > 1 else ""
    return (first_obs[:400], first_think[:200])


def actions_of(turns):
    return [t["action"] for t in turns if t["role"] == "assistant"]

def n_turns(turns):
    return sum(1 for t in turns if t["role"] == "assistant")


def count_invalid_obs(turns):
    """Count env observations containing 'Nothing happened' (illegal action signal)."""
    n = 0
    for t in turns:
        if t["role"] == "user" and "Nothing happened" in t.get("observation", ""):
            n += 1
    return n


def count_repeats(actions):
    """Return max consecutive-repeat count and total unique-count ratio."""
    if not actions:
        return 0, 0.0
    max_run = 1
    cur = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i-1] and actions[i] is not None:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    # Fraction of duplicated actions (any)
    cnt = Counter(a for a in actions if a is not None)
    total = sum(cnt.values())
    dup = sum(v - 1 for v in cnt.values())
    dup_rate = dup / total if total > 0 else 0.0
    return max_run, dup_rate


def malformed_count(turns):
    return sum(1 for t in turns if t["role"] == "assistant" and not t.get("has_action_tag"))


def classify_action(a):
    """Rough ALFWorld action categorization."""
    if a is None:
        return "null"
    a = a.strip().lower()
    if a.startswith("go to"):
        return "go_to"
    if a.startswith("take "):
        return "take"
    if a.startswith("put "):
        return "put"
    if a.startswith("open "):
        return "open"
    if a.startswith("close "):
        return "close"
    if a.startswith("use "):
        return "use"
    if a.startswith("clean "):
        return "clean"
    if a.startswith("heat "):
        return "heat"
    if a.startswith("cool "):
        return "cool"
    if a.startswith("slice "):
        return "slice"
    if a.startswith("examine "):
        return "examine"
    if a.startswith("look"):
        return "look"
    if a.startswith("inventory"):
        return "inventory"
    return "other"


# --- Load both datasets ----------------------------------------------------

def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def main():
    v24_data = load(V24_PATH)
    v1_data = load(V1_PATH)
    print(f"v24 trajectories: {len(v24_data)}")
    print(f"v1 trajectories:  {len(v1_data)}")

    # Pair by index (ALFWorld val is deterministic)
    matched = list(zip(v1_data, v24_data))

    # Verify matching by first-observation sig
    mismatch = 0
    for i, (r1, r2) in enumerate(matched):
        t1 = parse_trajectory(r1["output"])
        t2 = parse_trajectory(r2["output"])
        s1 = extract_task_signature(t1)
        s2 = extract_task_signature(t2)
        if s1[0][:100] != s2[0][:100]:
            mismatch += 1
    print(f"Task-signature mismatches: {mismatch}/{len(matched)}")

    # Population stats
    stats = {"v1": defaultdict(list), "v24": defaultdict(list)}

    case_records = []

    for idx, (r1, r2) in enumerate(matched):
        t1 = parse_trajectory(r1["output"])
        t2 = parse_trajectory(r2["output"])

        a1 = actions_of(t1)
        a2 = actions_of(t2)

        stats["v1"]["reward"].append(r1["reward"])
        stats["v1"]["score"].append(r1["score"])
        stats["v1"]["n_turns"].append(n_turns(t1))
        stats["v1"]["invalid_obs"].append(count_invalid_obs(t1))
        mr1, dup1 = count_repeats(a1)
        stats["v1"]["max_run"].append(mr1)
        stats["v1"]["dup_rate"].append(dup1)
        stats["v1"]["malformed"].append(malformed_count(t1))
        stats["v1"]["null_acts"].append(sum(1 for a in a1 if a is None))

        stats["v24"]["reward"].append(r2["reward"])
        stats["v24"]["score"].append(r2["score"])
        stats["v24"]["n_turns"].append(n_turns(t2))
        stats["v24"]["invalid_obs"].append(count_invalid_obs(t2))
        mr2, dup2 = count_repeats(a2)
        stats["v24"]["max_run"].append(mr2)
        stats["v24"]["dup_rate"].append(dup2)
        stats["v24"]["malformed"].append(malformed_count(t2))
        stats["v24"]["null_acts"].append(sum(1 for a in a2 if a is None))

        case_records.append({
            "idx": idx,
            "v1_score": r1["score"],
            "v24_score": r2["score"],
            "v1_reward": r1["reward"],
            "v24_reward": r2["reward"],
            "v1_turns": n_turns(t1),
            "v24_turns": n_turns(t2),
            "v1_max_run": mr1,
            "v24_max_run": mr2,
            "v1_invalid": count_invalid_obs(t1),
            "v24_invalid": count_invalid_obs(t2),
            "v1_malformed": malformed_count(t1),
            "v24_malformed": malformed_count(t2),
            "v1_actions": a1,
            "v24_actions": a2,
            "t1_first_obs": t1[0].get("observation", "")[:400] if t1 and t1[0]["role"] == "user" else "",
            "t1_first_think": t1[1].get("think", "") if len(t1) > 1 else "",
            "t2_first_think": t2[1].get("think", "") if len(t2) > 1 else "",
        })

    n = len(matched)

    def pct(frac):
        return f"{100*frac:.1f}%"

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Success definition: ALFWorld score=1 means task completed successfully
    v1_succ = sum(1 for c in case_records if c["v1_score"] >= 1.0)
    v24_succ = sum(1 for c in case_records if c["v24_score"] >= 1.0)

    both_succ = sum(1 for c in case_records if c["v1_score"] >= 1.0 and c["v24_score"] >= 1.0)
    both_fail = sum(1 for c in case_records if c["v1_score"] < 1.0 and c["v24_score"] < 1.0)
    v1_win = sum(1 for c in case_records if c["v1_score"] >= 1.0 and c["v24_score"] < 1.0)
    v24_win = sum(1 for c in case_records if c["v24_score"] >= 1.0 and c["v1_score"] < 1.0)

    print(f"\n=== Aggregate Stats (n={n}) ===")
    print(f"Success Rate: v1={pct(v1_succ/n)} ({v1_succ}), v24={pct(v24_succ/n)} ({v24_succ})")
    print(f"Both succeed: {both_succ} ({pct(both_succ/n)})")
    print(f"Both fail: {both_fail} ({pct(both_fail/n)})")
    print(f"v1 wins (regression): {v1_win} ({pct(v1_win/n)})")
    print(f"v24 wins (progression): {v24_win} ({pct(v24_win/n)})")
    print(f"Net delta: {v24_win - v1_win}")

    # Trajectory behavior
    print(f"\nAvg turns: v1={mean(stats['v1']['n_turns']):.2f}, v24={mean(stats['v24']['n_turns']):.2f}")
    print(f"Avg invalid-action obs: v1={mean(stats['v1']['invalid_obs']):.2f}, v24={mean(stats['v24']['invalid_obs']):.2f}")
    print(f"Avg max-consecutive-repeat: v1={mean(stats['v1']['max_run']):.2f}, v24={mean(stats['v24']['max_run']):.2f}")
    print(f"Avg dup-action rate: v1={mean(stats['v1']['dup_rate']):.3f}, v24={mean(stats['v24']['dup_rate']):.3f}")
    print(f"Avg malformed: v1={mean(stats['v1']['malformed']):.2f}, v24={mean(stats['v24']['malformed']):.2f}")
    print(f"Avg null actions: v1={mean(stats['v1']['null_acts']):.2f}, v24={mean(stats['v24']['null_acts']):.2f}")

    # Repetition loop rate (max_run >= 4 = 4+ consecutive identical actions)
    v1_loops = sum(1 for m in stats["v1"]["max_run"] if m >= 4)
    v24_loops = sum(1 for m in stats["v24"]["max_run"] if m >= 4)
    print(f"\nRepetition loops (max_run>=4): v1={v1_loops}, v24={v24_loops}")

    v1_severe_loops = sum(1 for m in stats["v1"]["max_run"] if m >= 6)
    v24_severe_loops = sum(1 for m in stats["v24"]["max_run"] if m >= 6)
    print(f"Severe loops (max_run>=6): v1={v1_severe_loops}, v24={v24_severe_loops}")

    # Distribution of turn counts
    v1_short = sum(1 for t in stats["v1"]["n_turns"] if t <= 5)
    v24_short = sum(1 for t in stats["v24"]["n_turns"] if t <= 5)
    v1_long = sum(1 for t in stats["v1"]["n_turns"] if t >= 20)
    v24_long = sum(1 for t in stats["v24"]["n_turns"] if t >= 20)
    print(f"\nShort trajectories (<=5 turns): v1={v1_short}, v24={v24_short}")
    print(f"Long trajectories (>=20 turns): v1={v1_long}, v24={v24_long}")

    # What fraction of failures had severe repetition?
    v1_fail_loop = sum(1 for c in case_records if c["v1_score"] < 1.0 and c["v1_max_run"] >= 4)
    v24_fail_loop = sum(1 for c in case_records if c["v24_score"] < 1.0 and c["v24_max_run"] >= 4)
    v1_fails = sum(1 for c in case_records if c["v1_score"] < 1.0)
    v24_fails = sum(1 for c in case_records if c["v24_score"] < 1.0)
    print(f"\nFailures with loop: v1 {v1_fail_loop}/{v1_fails}, v24 {v24_fail_loop}/{v24_fails}")

    # Save case records and some pre-picked case studies
    # Regression cases (v1 wins)
    regression = [c for c in case_records if c["v1_score"] >= 1.0 and c["v24_score"] < 1.0]
    progression = [c for c in case_records if c["v24_score"] >= 1.0 and c["v1_score"] < 1.0]
    print(f"\nRegression cases (v1 succ, v24 fail): {len(regression)}")
    print(f"Progression cases (v24 succ, v1 fail): {len(progression)}")

    # For regression cases, compute divergence point
    for c in regression:
        # find first action where v1 and v24 diverge
        minlen = min(len(c["v1_actions"]), len(c["v24_actions"]))
        div = None
        for k in range(minlen):
            if c["v1_actions"][k] != c["v24_actions"][k]:
                div = k
                break
        if div is None and len(c["v1_actions"]) != len(c["v24_actions"]):
            div = minlen
        c["divergence_step"] = div

    # Save to JSON
    out_path = "/data/home/qisheng/EvolAnalsis/analysis_reports/_v24_alfworld_cases.json"
    with open(out_path, "w") as f:
        json.dump({
            "aggregate": {
                "n": n,
                "v1_success": v1_succ,
                "v24_success": v24_succ,
                "both_succ": both_succ,
                "both_fail": both_fail,
                "v1_win_regression": v1_win,
                "v24_win_progression": v24_win,
                "v1_avg_turns": mean(stats["v1"]["n_turns"]),
                "v24_avg_turns": mean(stats["v24"]["n_turns"]),
                "v1_avg_invalid": mean(stats["v1"]["invalid_obs"]),
                "v24_avg_invalid": mean(stats["v24"]["invalid_obs"]),
                "v1_avg_max_run": mean(stats["v1"]["max_run"]),
                "v24_avg_max_run": mean(stats["v24"]["max_run"]),
                "v1_avg_dup_rate": mean(stats["v1"]["dup_rate"]),
                "v24_avg_dup_rate": mean(stats["v24"]["dup_rate"]),
                "v1_avg_malformed": mean(stats["v1"]["malformed"]),
                "v24_avg_malformed": mean(stats["v24"]["malformed"]),
                "v1_loops_ge4": v1_loops,
                "v24_loops_ge4": v24_loops,
                "v1_severe_loops": v1_severe_loops,
                "v24_severe_loops": v24_severe_loops,
                "v1_short_le5": v1_short,
                "v24_short_le5": v24_short,
                "v1_long_ge20": v1_long,
                "v24_long_ge20": v24_long,
                "v1_fail_loop": v1_fail_loop,
                "v1_fails": v1_fails,
                "v24_fail_loop": v24_fail_loop,
                "v24_fails": v24_fails,
            },
            "cases": case_records,
            "regression": regression,
            "progression": progression,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print a few representative regression cases for inspection
    print("\n=== TOP 5 REGRESSION CASES (v1 succ, v24 fail) ===")
    for c in regression[:8]:
        print(f"\n-- Task idx={c['idx']} div_step={c.get('divergence_step')} --")
        print(f"   v1: {c['v1_turns']} turns, max_run={c['v1_max_run']}, invalid={c['v1_invalid']}")
        print(f"   v24: {c['v24_turns']} turns, max_run={c['v24_max_run']}, invalid={c['v24_invalid']}")
        print(f"   Goal hint: {c['t1_first_think'][:120]}")
        print(f"   v1 actions ({len(c['v1_actions'])}): {c['v1_actions'][:12]}")
        print(f"   v24 actions ({len(c['v24_actions'])}): {c['v24_actions'][:12]}")

    print("\n=== TOP 3 PROGRESSION CASES (v24 succ, v1 fail) ===")
    for c in progression[:5]:
        print(f"\n-- Task idx={c['idx']} --")
        print(f"   v1: {c['v1_turns']} turns, max_run={c['v1_max_run']}, invalid={c['v1_invalid']}")
        print(f"   v24: {c['v24_turns']} turns, max_run={c['v24_max_run']}, invalid={c['v24_invalid']}")
        print(f"   Goal hint: {c['t2_first_think'][:120]}")
        print(f"   v1 actions: {c['v1_actions'][:12]}")
        print(f"   v24 actions: {c['v24_actions'][:12]}")


if __name__ == "__main__":
    main()
