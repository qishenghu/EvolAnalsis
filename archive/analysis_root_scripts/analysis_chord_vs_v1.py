#!/usr/bin/env python3
"""Focused CHORD-vs-DUET-v1 trajectory diff at step 100 on WebShop 1.5B.

Question: why does CHORD (Val@100=0.603) beat DUET v1 (0.549) when both
use only teacher (s,a) pairs (no logits)?

Mechanism under test: CHORD's high-mu early BC (mu=0.9 -> 0.05 over 25 steps)
lexically imprints click[<teacher-option>] tokens while DUET v1's trajectory-level
DR3 surrogate cannot.
"""

import json
import re
from collections import defaultdict, Counter

BASE = "/data/home/qisheng/EvolAnalsis"
PATHS = {
    "CHORD": f"{BASE}/experiments/webshop/webshop_qwen1.5b_chord/validation_log/100.jsonl",
    "v1":    f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet/validation_log/100.jsonl",
    "v24":   f"{BASE}/experiments/webshop/webshop_qwen1.5b_duet_v24/validation_log/100.jsonl",
}

ASIN_RE = re.compile(r'^[bB][0-9a-zA-Z]{9}$')
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
INSTR_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*Back to Search", re.DOTALL)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def parse_turns(entry):
    out = entry.get("output", "")
    turns = []
    for part in out.split("assistant\n")[1:]:
        user_split = part.split("\nuser\n", 1)
        assistant = user_split[0]
        obs = user_split[1] if len(user_split) > 1 else ""
        m = ACTION_RE.search(assistant)
        action = m.group(1).strip() if m else None
        turns.append({"action": action, "observation": obs, "assistant": assistant})
    return turns

def action_body(a):
    if a is None:
        return None
    m = re.match(r"(search|click)\[(.*?)\]\s*$", a, re.DOTALL)
    if not m:
        return None
    return (m.group(1), m.group(2).strip())

def classify(a):
    b = action_body(a)
    if b is None:
        return "MALFORMED"
    k, v = b
    vl = v.lower()
    if k == "search":
        return "SEARCH"
    if ASIN_RE.match(vl):
        return "PRODUCT_CLICK"
    if vl == "buy now":
        return "BUY_NOW"
    if vl in ("back to search", "< prev", "prev", "next >", "next",
              "description", "features", "reviews"):
        return "NAV"
    return "OPTION_CLICK"

def extract_instruction(entry):
    m = INSTR_RE.search(entry.get("output", ""))
    return m.group(1).strip() if m else ""

ATTR_PATTERNS = {
    "color": re.compile(r'color:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "size":  re.compile(r'size:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "fit":   re.compile(r'fit type:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
    "style": re.compile(r'style:\s*([^,]+?)(?:,\s*and\b|$)', re.IGNORECASE),
}

def extract_required(instr):
    req = {}
    for k, r in ATTR_PATTERNS.items():
        m = r.search(instr)
        if m:
            req[k] = m.group(1).strip().lower()
    return req

def trajectory_stats(entry):
    turns = parse_turns(entry)
    actions = [t["action"] for t in turns]
    classes = [classify(a) for a in actions]
    opt_clicks = [a for a in actions if classify(a) == "OPTION_CLICK"]
    unique_opts = set()
    for a in opt_clicks:
        b = action_body(a)
        if b:
            unique_opts.add(b[1].lower())
    return {
        "n_turns": len(turns),
        "classes": classes,
        "actions": actions,
        "turns": turns,
        "has_prod": "PRODUCT_CLICK" in classes,
        "has_buy": "BUY_NOW" in classes,
        "n_option_clicks": len(opt_clicks),
        "unique_opts": unique_opts,
        "malformed": classes.count("MALFORMED"),
    }

def attr_hits(required, unique_opts):
    hits = {}
    for k, v in required.items():
        vl = v.lower()
        matched = any(vl in o or o in vl for o in unique_opts)
        hits[k] = int(matched)
    return hits

def first_option_click_step(actions, product_step):
    """Return step index of first OPTION_CLICK after product click, else None."""
    if product_step is None:
        return None
    for j in range(product_step + 1, len(actions)):
        if classify(actions[j]) == "OPTION_CLICK":
            return j
    return None

def product_click_step(actions):
    for j, a in enumerate(actions):
        if classify(a) == "PRODUCT_CLICK":
            return j
    return None

def total_response_length(entry):
    """Total chars in all assistant turns (proxy for response-length sum)."""
    t = parse_turns(entry)
    return sum(len(x["assistant"]) for x in t)

# ---------------------------------------------------------------
def main():
    data = {k: load_jsonl(p) for k, p in PATHS.items()}
    N = min(len(v) for v in data.values())
    print(f"Loaded {N} validation trajectories per variant.")

    # Aggregate
    agg = {k: defaultdict(list) for k in PATHS}
    attr_match = {k: defaultdict(list) for k in PATHS}
    for i in range(N):
        instr = extract_instruction(data["v1"][i]) or extract_instruction(data["CHORD"][i])
        req = extract_required(instr)
        for v in PATHS:
            e = data[v][i]
            s = trajectory_stats(e)
            agg[v]["reward"].append(e["reward"])
            agg[v]["score"].append(e["score"])
            agg[v]["reward_gt0"].append(int(e["reward"] > 0))
            agg[v]["n_turns"].append(s["n_turns"])
            agg[v]["has_prod"].append(int(s["has_prod"]))
            agg[v]["has_buy"].append(int(s["has_buy"]))
            agg[v]["n_opt_clicks"].append(s["n_option_clicks"])
            agg[v]["n_unique_opts"].append(len(s["unique_opts"]))
            agg[v]["any_opt"].append(int(s["n_option_clicks"] > 0))
            agg[v]["malformed"].append(s["malformed"])
            agg[v]["resp_len"].append(total_response_length(e))
            for k, h in attr_hits(req, s["unique_opts"]).items():
                attr_match[v][k].append(h)
            # "teacher-exact option match" = did they click an option token
            # exactly matching a required attribute string?
            exact_match = 0
            for k, rv in req.items():
                if rv.lower() in s["unique_opts"]:
                    exact_match = 1
                    break
            agg[v]["exact_teacher_opt"].append(exact_match)

    # Main comparison table
    print("\n=== FULL 200 POPULATION ===")
    print(f"{'variant':<8}{'val_score':>10}{'r>0':>8}{'reached_prod':>14}{'any_opt':>10}{'exact_opt':>12}{'avg_steps':>11}{'malformed/tr':>14}{'resp_len':>11}")
    for v in ("CHORD", "v1", "v24"):
        s = agg[v]
        avg = lambda x: sum(x)/len(x) if x else 0
        print(f"{v:<8}{avg(s['score']):>10.4f}"
              f"{avg(s['reward_gt0'])*100:>7.1f}%"
              f"{avg(s['has_prod'])*100:>13.1f}%"
              f"{avg(s['any_opt'])*100:>9.1f}%"
              f"{avg(s['exact_teacher_opt'])*100:>11.1f}%"
              f"{avg(s['n_turns']):>11.2f}"
              f"{avg(s['malformed']):>14.3f}"
              f"{avg(s['resp_len']):>11.0f}")

    print("\n=== Attribute-match rate (tasks where attribute required) ===")
    for attr in ("color", "size", "fit", "style"):
        line = f"  {attr:<6}"
        for v in ("CHORD", "v1", "v24"):
            h = attr_match[v][attr]
            if h:
                line += f"  {v}:{sum(h)}/{len(h)} ({sum(h)/len(h)*100:4.1f}%)"
        print(line)

    # === 20 matched tasks where CHORD > v1 (largest gaps) ===
    gaps = []
    for i in range(N):
        rc = data["CHORD"][i]["reward"]
        r1 = data["v1"][i]["reward"]
        gaps.append((i, rc - r1, rc, r1, data["v24"][i]["reward"]))
    # Take 20 with largest CHORD-minus-v1 gap (positive) plus 5 with reverse gap to be balanced
    gaps_sorted = sorted(gaps, key=lambda x: -x[1])
    picked20 = gaps_sorted[:20]

    print("\n=== 20 matched tasks (CHORD > v1 by largest gap) ===")
    print(f"{'idx':>4}{'CHORD':>7}{'v1':>7}{'v24':>7}{'gap':>8}")
    for row in picked20:
        i, gap, rc, r1, r24 = row
        print(f"{i:>4}{rc:>7.3f}{r1:>7.3f}{r24:>7.3f}{gap:>+8.3f}")

    # Per-variant behavior on these 20
    print("\n=== 20-task behavior ===")
    picked_idxs = [r[0] for r in picked20]
    tab = {v: defaultdict(int) for v in ("CHORD", "v1", "v24")}
    for i in picked_idxs:
        instr = extract_instruction(data["v1"][i]) or extract_instruction(data["CHORD"][i])
        req = extract_required(instr)
        for v in ("CHORD", "v1", "v24"):
            s = trajectory_stats(data[v][i])
            tab[v]["n"] += 1
            if s["has_prod"]:
                tab[v]["reached_prod"] += 1
            if s["n_option_clicks"] > 0:
                tab[v]["any_opt"] += 1
            for k, rv in req.items():
                if rv.lower() in s["unique_opts"]:
                    tab[v]["exact_opt"] += 1
                    break
            if data[v][i]["reward"] > 0:
                tab[v]["reward_gt0"] += 1
            # Count click attempts BEFORE buy_now
            actions = s["actions"]
            clicks_before_buy = 0
            for a in actions:
                c = classify(a)
                if c == "OPTION_CLICK":
                    clicks_before_buy += 1
                elif c == "BUY_NOW":
                    break
            tab[v]["clicks_before_buy_sum"] += clicks_before_buy

    print(f"{'var':<8}{'reached':>10}{'any_opt':>10}{'exact_opt':>12}{'reward>0':>11}{'avg clicks/tr':>15}")
    for v in ("CHORD", "v1", "v24"):
        t = tab[v]
        n = t["n"]
        print(f"{v:<8}{t['reached_prod']:>6}/{n}{t['any_opt']:>8}/{n}{t['exact_opt']:>9}/{n}{t['reward_gt0']:>8}/{n}{t['clicks_before_buy_sum']/n:>15.2f}")

    # === Case studies: 5 where CHORD succeeds and v1 fails on option click ===
    print("\n\n=== CASE STUDIES: 5 tasks where CHORD clicks teacher option, v1 does not ===")
    cases = []
    for row in picked20:
        i = row[0]
        instr = extract_instruction(data["v1"][i]) or extract_instruction(data["CHORD"][i])
        req = extract_required(instr)
        if not req:
            continue
        s1 = trajectory_stats(data["v1"][i])
        sc = trajectory_stats(data["CHORD"][i])
        # keep if CHORD hit an attribute exactly and v1 did not
        chord_exact = any(rv.lower() in sc["unique_opts"] for rv in req.values())
        v1_exact   = any(rv.lower() in s1["unique_opts"] for rv in req.values())
        if chord_exact and not v1_exact and data["CHORD"][i]["reward"] > data["v1"][i]["reward"]:
            cases.append(i)
        if len(cases) >= 6:
            break
    print(f"Found {len(cases)} such cases: {cases}")

    out_cases = []
    for i in cases:
        instr = extract_instruction(data["v1"][i]) or extract_instruction(data["CHORD"][i])
        req = extract_required(instr)
        case = {"task_idx": i, "instruction": instr, "required": req, "variants": {}}
        for v in ("CHORD", "v1", "v24"):
            s = trajectory_stats(data[v][i])
            actions = s["actions"]
            # Pinpoint product click + subsequent option-click window
            pc = product_click_step(actions)
            case["variants"][v] = {
                "reward": data[v][i]["reward"],
                "score": data[v][i]["score"],
                "n_turns": s["n_turns"],
                "n_option_clicks": s["n_option_clicks"],
                "unique_opts": sorted(s["unique_opts"]),
                "actions": actions,
                "product_click_step": pc,
                "first_option_step": first_option_click_step(actions, pc),
            }
        out_cases.append(case)

    outp = f"{BASE}/analysis_reports/_chord_vs_v1_cases.json"
    with open(outp, "w") as f:
        json.dump(out_cases, f, indent=2)
    print(f"Wrote detailed case data to {outp}")

    # === Confound checks ===
    print("\n\n=== CONFOUND CHECKS ===")
    print(f"{'variant':<8}{'avg_steps':>10}{'mal_trajs':>12}{'avg_malformed/tr':>18}{'avg_resp_len_chars':>20}")
    for v in ("CHORD", "v1", "v24"):
        s = agg[v]
        mal_trajs = sum(1 for x in s["malformed"] if x > 0)
        print(f"{v:<8}{sum(s['n_turns'])/len(s['n_turns']):>10.2f}"
              f"{mal_trajs:>9}/{len(s['malformed'])}"
              f"{sum(s['malformed'])/len(s['malformed']):>18.3f}"
              f"{sum(s['resp_len'])/len(s['resp_len']):>20.0f}")

    # Long-trajectory tail (option loop detection)
    print("\n=== Long-trajectory tail (n_turns >= 13) ===")
    for v in ("CHORD", "v1", "v24"):
        long_n = sum(1 for x in agg[v]["n_turns"] if x >= 13)
        print(f"  {v}: {long_n}/{len(agg[v]['n_turns'])} tasks with >=13 turns")

    # Action-distribution at first step after product click
    print("\n=== First action on product-detail page (CHORD vs v1 vs v24) ===")
    first_action_dist = {v: Counter() for v in ("CHORD", "v1", "v24")}
    for i in range(N):
        for v in ("CHORD", "v1", "v24"):
            s = trajectory_stats(data[v][i])
            pc = product_click_step(s["actions"])
            if pc is None or pc + 1 >= len(s["actions"]):
                first_action_dist[v]["NO_STEP_AFTER_PROD"] += 1
                continue
            first_action_dist[v][classify(s["actions"][pc + 1])] += 1
    print(f"{'category':<22}{'CHORD':>8}{'v1':>8}{'v24':>8}")
    cats = ["OPTION_CLICK", "BUY_NOW", "PRODUCT_CLICK", "SEARCH", "NAV", "MALFORMED", "NO_STEP_AFTER_PROD"]
    for c in cats:
        print(f"{c:<22}{first_action_dist['CHORD'][c]:>8}{first_action_dist['v1'][c]:>8}{first_action_dist['v24'][c]:>8}")

    # Response length per turn (proxy for entropy / verbosity)
    print("\n=== Response length per turn (chars, proxy for response variance) ===")
    for v in ("CHORD", "v1", "v24"):
        per_turn = []
        for i in range(N):
            e = data[v][i]
            turns = parse_turns(e)
            for t in turns:
                per_turn.append(len(t["assistant"]))
        if per_turn:
            mean = sum(per_turn)/len(per_turn)
            import statistics
            sd = statistics.pstdev(per_turn)
            print(f"  {v}: mean={mean:.0f}  sd={sd:.0f}  n_turns={len(per_turn)}")

    return out_cases

if __name__ == "__main__":
    main()
