"""Extract (s_product_page, a_teacher_click_option) pairs from teacher trajectories.

For each teacher trajectory, find assistant turns where the observation shows a product-detail page
(contains '< Prev' typically and option clickables like color/size) and the assistant emits
`<action>click[<option_name>]</action>` where <option_name> is NOT 'buy now', 'back to search',
'< prev', 'next >', or a B0X ASIN.

Outputs a JSONL list of (state_messages, teacher_action_string, page_type, task_id, rollout_id).
"""
import pickle
import json
import re
import os
from typing import List, Dict, Any, Optional

TEACHER_PATH = "/data/home/qisheng/EvolAnalsis/data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl"
OUT_PATH_OPTION = "/data/home/qisheng/EvolAnalsis/tmp_scripts/teacher_optionclick_states.jsonl"
OUT_PATH_SEARCH = "/data/home/qisheng/EvolAnalsis/tmp_scripts/teacher_search_states.jsonl"
OUT_PATH_BUY = "/data/home/qisheng/EvolAnalsis/tmp_scripts/teacher_buy_states.jsonl"

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
ASIN_RE = re.compile(r"^b0[0-9a-z]{8}$", re.IGNORECASE)

# tokens that are NOT option selections
NON_OPTION_CLICKS = {
    "buy now", "back to search", "< prev", "next >", "prev", "next",
    "description", "features", "reviews", "attributes", "search"
}


def classify_page(obs: str) -> str:
    low = obs.lower()
    # search home: "search bar availability: true" + very short
    if "search bar availability: true" in low and len(obs) < 400:
        return "search_home"
    # search results: "back to search" + "total results" + many ASINs
    if "total results" in low and "back to search" in low:
        return "search_results"
    # product detail: has 'buy now' clickable
    if "buy now" in low and "back to search" in low:
        return "product_detail"
    # subpage (after click on description etc)
    if "back to search" in low:
        return "other_product"
    return "unknown"


def extract_action(assistant_content: str) -> Optional[str]:
    m = ACTION_RE.search(assistant_content)
    if not m:
        return None
    return m.group(1).strip()


def action_type(action: str) -> str:
    a = action.strip().lower()
    if a.startswith("search["):
        return "search"
    if not a.startswith("click["):
        return "other"
    inner = a[len("click["):]
    if inner.endswith("]"):
        inner = inner[:-1]
    inner = inner.strip()
    if inner == "buy now":
        return "buy_now"
    if ASIN_RE.match(inner):
        return "click_asin"
    if inner in NON_OPTION_CLICKS:
        return "nav_click"
    return "click_option"


def extract_pairs(traj: Dict[str, Any]):
    messages = traj["messages"]
    # Iterate to find assistant turns; for each assistant turn, the *previous* user turn is the state
    results = {"option": [], "search": [], "buy": []}
    # Build prefix of messages up to each assistant turn
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        action = extract_action(m["content"])
        if not action:
            continue
        atype = action_type(action)
        # State = messages[:i] (the context the teacher saw)
        if atype not in ("search", "buy_now", "click_option"):
            continue
        # We particularly want product_detail for option clicks
        prev_user_content = ""
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                prev_user_content = messages[j]["content"]
                break
        page = classify_page(prev_user_content)
        if atype == "click_option" and page == "product_detail":
            results["option"].append({
                "task_id": traj.get("task_id"),
                "rollout_id": traj.get("rollout_id"),
                "page": page,
                "action": action,
                "state_messages": messages[:i],  # up to and not including the teacher turn
            })
        elif atype == "search":
            # grab search-home states (where search is the only option)
            results["search"].append({
                "task_id": traj.get("task_id"),
                "rollout_id": traj.get("rollout_id"),
                "page": page,
                "action": action,
                "state_messages": messages[:i],
            })
        elif atype == "buy_now" and page == "product_detail":
            results["buy"].append({
                "task_id": traj.get("task_id"),
                "rollout_id": traj.get("rollout_id"),
                "page": page,
                "action": action,
                "state_messages": messages[:i],
            })
    return results


def main():
    with open(TEACHER_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trajectories")
    n_kept_option = n_kept_search = n_kept_buy = 0
    # Only keep successful trajectories
    fo = open(OUT_PATH_OPTION, "w")
    fs = open(OUT_PATH_SEARCH, "w")
    fb = open(OUT_PATH_BUY, "w")
    n_seen = 0
    for traj in data:
        if not traj.get("success"):
            continue
        n_seen += 1
        pairs = extract_pairs(traj)
        for p in pairs["option"]:
            fo.write(json.dumps(p) + "\n")
            n_kept_option += 1
        for p in pairs["search"]:
            fs.write(json.dumps(p) + "\n")
            n_kept_search += 1
        for p in pairs["buy"]:
            fb.write(json.dumps(p) + "\n")
            n_kept_buy += 1
        # Cap to keep file reasonable
        if n_kept_option >= 2000 and n_kept_search >= 500 and n_kept_buy >= 500:
            break
    fo.close(); fs.close(); fb.close()
    print(f"Successful trajectories scanned: {n_seen}")
    print(f"Option-click state-action pairs: {n_kept_option}")
    print(f"Search state-action pairs: {n_kept_search}")
    print(f"Buy-now state-action pairs: {n_kept_buy}")


if __name__ == "__main__":
    main()
