#!/usr/bin/env python3
"""Case analysis: why does v24 uniquely beat CHORD among 1.5B DUET variants?

Compares v1, v8, v12, v24, and CHORD on matched validation tasks at step 100.
Focus: option-selection phase (click[color], click[size]) where DUET variants fail.
"""

import json
import re
from collections import Counter, defaultdict

BASE = "/data/home/qisheng/EvolAnalsis"

VAL_PATHS = {
    "v1":    f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/100.jsonl",
    "v8":    f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v8/validation_log/100.jsonl",
    "v12":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v12/validation_log/100.jsonl",
    "v24":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v24/validation_log/100.jsonl",
    "CHORD": f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl",
}

# --------------------------- Parsing helpers ---------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

ASIN_RE = re.compile(r'^[bB][0-9a-zA-Z]{9}$')
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)

def parse_turns(entry):
    """Split output into (action, observation) turns aligned in time."""
    output = entry.get("output", "")
    turns = []  # list of dicts: {action, raw_assistant, observation}
    parts = output.split("assistant\n")
    for i, part in enumerate(parts[1:], start=1):
        # part contains assistant content then "\nuser\n<obs>"
        user_split = part.split("\nuser\n", 1)
        assistant_text = user_split[0]
        observation = user_split[1] if len(user_split) > 1 else ""
        act_match = ACTION_RE.search(assistant_text)
        action = act_match.group(1).strip() if act_match else None
        turns.append({
            "action": action,
            "assistant": assistant_text,
            "observation": observation,
        })
    return turns

def action_body(a):
    if a is None:
        return None
    m = re.match(r"(search|click)\[(.*?)\]\s*$", a, re.DOTALL)
    if not m:
        return None
    return (m.group(1), m.group(2).strip())

def is_option_click(action_str):
    body = action_body(action_str)
    if body is None or body[0] != "click":
        return False
    val = body[1].lower()
    if val in ("buy now", "back to search", "< prev", "next >", "description",
               "features", "reviews", "prev", "next"):
        return False
    if ASIN_RE.match(val):
        return False
    return True

def classify_action(action_str):
    body = action_body(action_str)
    if body is None:
        return "MALFORMED"
    kind, val = body
    v = val.lower()
    if kind == "search":
        return "SEARCH"
    if ASIN_RE.match(v):
        return "PRODUCT_CLICK"
    if v == "buy now":
        return "BUY_NOW"
    if v in ("back to search", "< prev", "prev", "next >", "next",
             "description", "features", "reviews"):
        return "NAV"
    return "OPTION_CLICK"

# --------------------------- Instruction parsing ---------------------------

INSTR_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*Back to Search", re.DOTALL)

def extract_instruction(entry):
    # from output's first observation, or from input
    out = entry.get("output", "")
    m = INSTR_RE.search(out)
    if m:
        return m.group(1).strip()
    return ""

# Attribute extraction aligned with WebShop phrasing.
ATTR_PATTERNS = {
    "color": re.compile(r'color:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "size":  re.compile(r'size:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "fit":   re.compile(r'fit type:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "style": re.compile(r'style:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
}

def extract_required_attrs(instr):
    out = {}
    for k, r in ATTR_PATTERNS.items():
        m = r.search(instr)
        if m:
            out[k] = m.group(1).strip().lower()
    # Price upper bound
    m = re.search(r"price lower than\s*([0-9.]+)", instr, re.IGNORECASE)
    if m:
        out["price_max"] = float(m.group(1))
    return out

# --------------------------- Trajectory metrics ---------------------------

def trajectory_stats(entry):
    turns = parse_turns(entry)
    actions = [t["action"] for t in turns]
    classes = [classify_action(a) for a in actions]
    has_search = "SEARCH" in classes
    has_prod_click = "PRODUCT_CLICK" in classes
    has_buy = "BUY_NOW" in classes
    malformed_n = classes.count("MALFORMED")
    opt_clicks = [a for a in actions if classify_action(a) == "OPTION_CLICK"]
    unique_opts = set(body[1].lower() for body in (action_body(a) for a in opt_clicks) if body)
    # Check whether option click landed on a product detail page (observation before contains "< Prev" and "Buy Now")
    return {
        "n_turns": len(turns),
        "has_search": has_search,
        "has_prod_click": has_prod_click,
        "has_buy": has_buy,
        "n_option_clicks": len(opt_clicks),
        "unique_opts": unique_opts,
        "malformed_n": malformed_n,
        "turns": turns,
        "classes": classes,
    }

# --------------------------- Attribute match ---------------------------

def attr_match_counts(required, unique_opts):
    """For each required attr, 1 if agent clicked a matching option token."""
    hits = {}
    for k, v in required.items():
        if k == "price_max":
            continue
        v_low = v.lower()
        matched = False
        # multi-word: check substring match
        for o in unique_opts:
            if v_low in o or o in v_low:
                matched = True
                break
        hits[k] = int(matched)
    return hits

# --------------------------- Main aggregation ---------------------------

def main():
    data = {k: load_jsonl(p) for k, p in VAL_PATHS.items()}
    N = min(len(v) for v in data.values())
    print(f"Loaded {N} trajectories for each of {list(data.keys())}")

    # Per-variant aggregate
    agg = {k: defaultdict(list) for k in VAL_PATHS}
    attr_matches = {k: defaultdict(list) for k in VAL_PATHS}

    for i in range(N):
        instr = extract_instruction(data["v24"][i])
        req = extract_required_attrs(instr)
        for variant in VAL_PATHS:
            s = trajectory_stats(data[variant][i])
            agg[variant]["score"].append(data[variant][i]["score"])
            agg[variant]["reward"].append(data[variant][i]["reward"])
            agg[variant]["n_turns"].append(s["n_turns"])
            agg[variant]["reached_product"].append(int(s["has_prod_click"]))
            agg[variant]["has_buy"].append(int(s["has_buy"]))
            agg[variant]["n_unique_opts"].append(len(s["unique_opts"]))
            agg[variant]["malformed"].append(s["malformed_n"])
            agg[variant]["reward_gt0"].append(int(data[variant][i]["reward"] > 0))
            hits = attr_match_counts(req, s["unique_opts"])
            for k, v in hits.items():
                attr_matches[variant][k].append(v)
            # store per-task unique_opts for later matching
            agg[variant].setdefault("unique_opts_list", []).append(s["unique_opts"])

    # Summary stats table
    print("\n=== Aggregate (200 tasks, step 100) ===")
    print(f"{'variant':<8}{'val_score':>10}{'reward>0':>10}{'reached':>10}{'buy_now':>10}{'avg_opts':>10}{'avg_steps':>11}{'malformed':>11}")
    for v in ("v1", "v8", "v12", "v24", "CHORD"):
        s = agg[v]
        avg = lambda x: sum(x)/len(x)
        print(f"{v:<8}{avg(s['score']):>10.4f}{avg(s['reward_gt0'])*100:>9.1f}%{avg(s['reached_product'])*100:>9.1f}%"
              f"{avg(s['has_buy'])*100:>9.1f}%{avg(s['n_unique_opts']):>10.2f}{avg(s['n_turns']):>11.2f}{avg(s['malformed']):>11.2f}")

    print("\n=== Attribute-match rate on tasks where attribute is required ===")
    for attr in ("color", "size", "fit", "style"):
        print(f"  {attr}:")
        for v in ("v1", "v8", "v12", "v24", "CHORD"):
            hits = attr_matches[v][attr]
            if not hits:
                print(f"    {v}: N/A (no tasks)")
                continue
            print(f"    {v}: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits)*100:.1f}%")

    # --------------------- Sample 20 matched tasks ---------------------
    # Pick 20 tasks where v24 succeeds (reward>0) but at least one of v1/v8/v12 fails,
    # to have cases illustrating the hypothesis
    print("\n=== Picking 20 informative matched tasks ===")
    interesting = []
    for i in range(N):
        r24 = data["v24"][i]["reward"]
        r12 = data["v12"][i]["reward"]
        r8 = data["v8"][i]["reward"]
        r1 = data["v1"][i]["reward"]
        rc = data["CHORD"][i]["reward"]
        # rank by how much v24 > v12
        gap_v24_v12 = r24 - r12
        interesting.append((i, gap_v24_v12, r1, r8, r12, r24, rc))
    # Sort by v24 - v12 gap (largest first)
    interesting.sort(key=lambda x: x[1], reverse=True)
    picked = interesting[:20]

    print(f"{'idx':>4}{'r_v1':>7}{'r_v8':>7}{'r_v12':>7}{'r_v24':>7}{'r_CHORD':>9}{'v24-v12':>9}")
    for row in picked:
        i, gap, r1, r8, r12, r24, rc = row
        print(f"{i:>4}{r1:>7.3f}{r8:>7.3f}{r12:>7.3f}{r24:>7.3f}{rc:>9.3f}{gap:>+9.3f}")

    # Aggregate on the 20-task sample
    print("\n=== 20-task matched sample breakdown ===")
    picked_idxs = [p[0] for p in picked]
    sample_agg = {v: defaultdict(list) for v in VAL_PATHS}
    for i in picked_idxs:
        instr = extract_instruction(data["v24"][i])
        req = extract_required_attrs(instr)
        for variant in VAL_PATHS:
            s = trajectory_stats(data[variant][i])
            sample_agg[variant]["reward"].append(data[variant][i]["reward"])
            sample_agg[variant]["reward_gt0"].append(int(data[variant][i]["reward"] > 0))
            sample_agg[variant]["reached_product"].append(int(s["has_prod_click"]))
            sample_agg[variant]["has_buy"].append(int(s["has_buy"]))
            sample_agg[variant]["n_unique_opts"].append(len(s["unique_opts"]))
            sample_agg[variant]["n_turns"].append(s["n_turns"])
            hits = attr_match_counts(req, s["unique_opts"])
            for k, v in hits.items():
                sample_agg[variant][f"hit_{k}"].append(v)

    print(f"{'variant':<8}{'avg_r':>7}{'r>0':>6}{'prod':>6}{'buy':>6}{'opts':>7}{'steps':>7}")
    for v in ("v1", "v8", "v12", "v24", "CHORD"):
        s = sample_agg[v]
        avg = lambda x: sum(x)/len(x) if x else 0
        print(f"{v:<8}{avg(s['reward']):>7.3f}{int(sum(s['reward_gt0'])):>5}{int(sum(s['reached_product'])):>5}{int(sum(s['has_buy'])):>5}"
              f"{avg(s['n_unique_opts']):>7.2f}{avg(s['n_turns']):>7.2f}")

    # Attribute match on the 20-task sample
    print("\n  Attribute match (of tasks where attribute required, within 20-task sample):")
    for attr in ("color", "size", "fit", "style"):
        print(f"  {attr}:")
        for v in ("v1", "v8", "v12", "v24", "CHORD"):
            hits = sample_agg[v].get(f"hit_{attr}", [])
            if not hits:
                continue
            print(f"    {v}: {sum(hits)}/{len(hits)}")

    # --------------------- Case studies ---------------------
    print("\n\n=== CASE STUDIES ===")
    # Pick top 5 where v24 succeeds and v12 fails
    case_ids = []
    for row in picked:
        i = row[0]
        if data["v24"][i]["reward"] > 0.5 and data["v12"][i]["reward"] < 0.3:
            case_ids.append(i)
        if len(case_ids) >= 5:
            break

    print(f"Selected {len(case_ids)} cases for detailed inspection: {case_ids}")

    cases_to_write = []
    for i in case_ids:
        instr = extract_instruction(data["v24"][i])
        req = extract_required_attrs(instr)
        case = {"task_idx": i, "instruction": instr, "required": req}
        for v in ("v1", "v8", "v12", "v24", "CHORD"):
            s = trajectory_stats(data[v][i])
            action_list = [t["action"] for t in s["turns"]]
            case[v] = {
                "reward": data[v][i]["reward"],
                "score": data[v][i]["score"],
                "n_turns": s["n_turns"],
                "unique_opts": sorted(s["unique_opts"]),
                "actions": action_list,
            }
        cases_to_write.append(case)

    # Dump cases JSON
    out_path = f"{BASE}/analysis_reports/_webshop_1.5b_v24_cases.json"
    with open(out_path, "w") as f:
        json.dump(cases_to_write, f, indent=2)
    print(f"\nDumped case details to {out_path}")

    # --------------- v24 vs CHORD on 20-task sample: where do they differ? ---------------
    print("\n=== v24 vs CHORD behavioral divergence on the 20-sample ===")
    v24_wins = 0; chord_wins = 0; tie = 0
    for i in picked_idxs:
        r24 = data["v24"][i]["reward"]; rc = data["CHORD"][i]["reward"]
        if r24 > rc: v24_wins += 1
        elif rc > r24: chord_wins += 1
        else: tie += 1
    print(f"  v24 > CHORD: {v24_wins}, CHORD > v24: {chord_wins}, tie: {tie}")

    # Population-level comparison v24 vs CHORD on full 200
    print("\n=== v24 vs CHORD on full 200 population ===")
    v24_wins = chord_wins = tie = 0
    v24wins_idx, chordwins_idx = [], []
    for i in range(N):
        r24 = data["v24"][i]["reward"]; rc = data["CHORD"][i]["reward"]
        if r24 > rc + 0.05:
            v24_wins += 1; v24wins_idx.append((i, r24-rc))
        elif rc > r24 + 0.05:
            chord_wins += 1; chordwins_idx.append((i, rc-r24))
        else:
            tie += 1
    print(f"  v24 > CHORD: {v24_wins}, CHORD > v24: {chord_wins}, ~tie: {tie}")
    v24wins_idx.sort(key=lambda x: -x[1])
    chordwins_idx.sort(key=lambda x: -x[1])
    print(f"  Top 5 v24 advantages (idx, gap): {v24wins_idx[:5]}")
    print(f"  Top 5 CHORD advantages (idx, gap): {chordwins_idx[:5]}")

    # --------------- Token-level BC hypothesis test ---------------
    # For the tasks where v24 beats v12: at the option-click step,
    # did v24 emit "click[color]" or "click[size]" etc., matching instruction attributes?
    # And did v12 emit buy-now or no option?
    print("\n=== Token-level option-clicking behavior (v24 vs v12, v24-beats-v12 cases) ===")
    option_hit = defaultdict(lambda: [0, 0])  # variant -> [hits, total]
    for row in picked:
        i = row[0]
        instr = extract_instruction(data["v24"][i])
        req = extract_required_attrs(instr)
        req_vals = set()
        for k, v in req.items():
            if k == "price_max": continue
            req_vals.add(v.lower())
        for variant in ("v1", "v8", "v12", "v24", "CHORD"):
            s = trajectory_stats(data[variant][i])
            opts = s["unique_opts"]
            matched = 0
            for rv in req_vals:
                if any(rv in o or o in rv for o in opts):
                    matched += 1
            option_hit[variant][0] += matched
            option_hit[variant][1] += max(len(req_vals), 1)
    for v in ("v1", "v8", "v12", "v24", "CHORD"):
        h, t = option_hit[v]
        print(f"  {v}: attr-opt hits {h}/{t} = {h/t*100:.1f}%")

    return cases_to_write, agg, sample_agg, option_hit


if __name__ == "__main__":
    cases, agg, sample_agg, option_hit = main()
