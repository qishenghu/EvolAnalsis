#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align SciWorld teacher trajectories to the current training prompt format.

What it does:
- Replace the system prompt in each trajectory with the current prompt from
  `env_service/environments/sciworld/sciworld_env.py` (`_get_system_prompt`).
- Replace the per-turn user hint suffix to match the current `_get_action_hints()`
  format:
    - `Available actions: [...]`
    - `OBJ must be replaced with exactly one of the following candidates: [...]`
    - optional task-level focus constraint derived from the task description
  while dropping legacy hint blocks.

This is intended for updating existing teacher trajectory dumps (PKL/JSONL)
after prompt changes, so that teacher replay matches the on-policy rollout prompt.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple


LEGACY_ACTIONS_MARKER = "Valid actions:"
LEGACY_SUGGESTED_MARKER = "Suggested actions:"
LEGACY_NEARBY_OBJECTS_MARKER = "Nearby objects:"
LEGACY_OBJECTS_MARKER = "OBJ needs to be replaced with one of the following objects:"
CURRENT_ACTIONS_MARKER = "Available actions:"
CURRENT_OBJECTS_MARKER = "OBJ must be replaced with exactly one of the following candidates"
CURRENT_FOCUS_MARKER = "Important! You can only use FOCUS actions on these task-required targets:"


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_system_prompt_from_env_py(env_py: str) -> str:
    """
    Extract the triple-quoted string returned by _get_system_prompt() in sciworld_env.py.
    """
    text = _read_text(env_py)
    # Match: def _get_system_prompt(...): ... return ''' ... '''
    m = re.search(r"def\s+_get_system_prompt\s*\([^)]*\)\s*->\s*str\s*:\s*.*?\n\s*return\s+'''([\s\S]*?)'''\s*\n",
                  text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Failed to extract system prompt from {env_py}.")
    return m.group(1).strip()


def _try_parse_py_list(list_text: str) -> Tuple[Any, bool]:
    t = (list_text or "").strip()
    if not t:
        return [], True
    try:
        v = ast.literal_eval(t)
        return v, True
    except Exception:
        return t, False


def _extract_objects_from_legacy_hint(hint_block: str) -> Tuple[Any, bool]:
    """
    Given a legacy hint block containing Valid actions + OBJ needs..., extract the objects list.
    """
    if LEGACY_OBJECTS_MARKER in hint_block:
        after = hint_block.split(LEGACY_OBJECTS_MARKER, 1)[1].strip()
        return _try_parse_py_list(after)
    # Fallback: try to parse after the last ':' in the block
    if ":" in hint_block:
        after = hint_block.rsplit(":", 1)[1].strip()
        return _try_parse_py_list(after)
    return hint_block.strip(), False


def _extract_task_description_from_messages(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        mm = re.search(r"Task:\s*(.*?)\n\s*\nCurrent observation:\n", content, flags=re.S)
        if mm:
            return mm.group(1).strip()
    return ""


def _clean_focus_item(item: str) -> str:
    item = (item or "").strip()
    item = re.sub(r"^(the|a|an)\s+", "", item, flags=re.IGNORECASE)
    item = re.sub(r"\s+", " ", item)
    return item.strip()


def _extract_focus_items(task_description: str) -> List[str]:
    raw_items = re.findall(
        r"focus on\s+(\b\w+\b(?:\s+\b\w+\b)*)",
        task_description or "",
        flags=re.IGNORECASE,
    )
    cleaned: List[str] = []
    skip_generic = {"thing", "object", "item", "it"}
    for item in raw_items:
        focus_item = _clean_focus_item(item)
        if not focus_item:
            continue
        if focus_item.lower() in skip_generic:
            continue
        if focus_item not in cleaned:
            cleaned.append(focus_item)
    return cleaned


def build_focus_hint(task_description: str) -> str:
    focus_items = _extract_focus_items(task_description)
    if not focus_items:
        return ""
    targets = ", ".join(focus_items)
    return (
        "Important! You can only use FOCUS actions on these task-required targets: "
        f"{targets}.\n"
        "You cannot FOCUS on arbitrary objects. Please only use FOCUS as required by the task description, "
        "and focus on the target itself rather than its container."
    )


def _iter_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    out.append(json.loads(s))
                except Exception:
                    continue
    return out


def load_gold_records(inputs: List[str]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_jsonl(inputs):
        data_idx = rec.get("data_idx")
        if data_idx is None:
            continue
        by_id[str(data_idx)] = rec
    return by_id


def _extract_actions_from_hint(hint_block: str) -> Tuple[Any, bool]:
    if CURRENT_ACTIONS_MARKER in hint_block:
        after = hint_block.split(CURRENT_ACTIONS_MARKER, 1)[1].strip()
        if "\n" in after:
            after = after.split("\n", 1)[0].strip()
        return _try_parse_py_list(after)
    if LEGACY_ACTIONS_MARKER in hint_block:
        after = hint_block.split(LEGACY_ACTIONS_MARKER, 1)[1].strip()
        if "\n" in after:
            after = after.split("\n", 1)[0].strip()
        return _try_parse_py_list(after)
    return [], False


def render_hint(available_actions: Any, objs: Any, task_description: str) -> str:
    parts: List[str] = []
    if available_actions not in (None, "", []):
        parts.append(f"Available actions: {repr(available_actions) if not isinstance(available_actions, str) else available_actions}")
    if objs not in (None, "", []):
        obj_repr = repr(objs) if not isinstance(objs, str) else objs
        parts.append(
            "OBJ must be replaced with exactly one of the following candidates, "
            f"using the exact string as provided: {obj_repr}."
        )
    focus_hint = build_focus_hint(task_description)
    if focus_hint:
        parts.append(focus_hint)
    return "\n".join(parts).strip()


def build_init_user_content_from_gold(gold_rec: Dict[str, Any]) -> str:
    task_desc = gold_rec.get("task_description", "")
    init_obs = gold_rec.get("initial_observation", "")
    init_hints = gold_rec.get("init_hints", {}) or {}
    hint = render_hint(
        init_hints.get("possible_actions", []),
        init_hints.get("possible_objects", []),
        task_desc,
    )
    return f"Task: {task_desc}\n\nCurrent observation:\n{init_obs}\n\n{hint}".rstrip()


def build_step_user_content_from_gold(gold_rec: Dict[str, Any], step_idx: int) -> Optional[str]:
    steps = gold_rec.get("steps", []) or []
    if step_idx < 0 or step_idx >= len(steps):
        return None
    step = steps[step_idx]
    obs = step.get("observation", "")
    hint = render_hint(
        step.get("possible_actions", []),
        step.get("possible_objects", []),
        gold_rec.get("task_description", ""),
    )
    return f"{obs}\n\n{hint}".rstrip() if hint else str(obs)


def align_messages_with_gold(messages: List[Dict[str, Any]], system_prompt: str, gold_rec: Dict[str, Any]) -> Dict[str, int]:
    stats = {"system_updated": 0, "user_hint_updated": 0}
    user_turn_idx = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content") != system_prompt:
                m["content"] = system_prompt
                stats["system_updated"] += 1
        elif role == "user":
            if user_turn_idx == 0:
                new_c = build_init_user_content_from_gold(gold_rec)
            else:
                new_c = build_step_user_content_from_gold(gold_rec, user_turn_idx - 1)
                if new_c is None:
                    new_c = m.get("content", "")
            if m.get("content") != new_c:
                m["content"] = new_c
                stats["user_hint_updated"] += 1
            user_turn_idx += 1
    return stats


def rewrite_user_content_with_new_hint(
    content: str,
    task_description: str,
) -> Tuple[str, bool]:
    """
    Replace any existing hint suffix with the new hint format.
    Returns (new_content, changed).
    """
    if not isinstance(content, str):
        return content, False

    # Unified detection: locate the last hint block start among known markers,
    # then extract object candidates from it.
    markers = [
        CURRENT_ACTIONS_MARKER,
        LEGACY_ACTIONS_MARKER,
        LEGACY_SUGGESTED_MARKER,
        LEGACY_NEARBY_OBJECTS_MARKER,
        LEGACY_OBJECTS_MARKER,
        CURRENT_OBJECTS_MARKER,
        CURRENT_FOCUS_MARKER,
        "OBJ candidates",
    ]
    starts = [content.rfind(m) for m in markers]
    starts = [s for s in starts if s != -1]
    if starts:
        start = min(starts)
        before = content[:start].rstrip()
        hint_block = content[start:].strip()

        available_actions = None
        actions_ok = False
        if CURRENT_ACTIONS_MARKER in hint_block or LEGACY_ACTIONS_MARKER in hint_block:
            available_actions, actions_ok = _extract_actions_from_hint(hint_block)

        objs = None
        ok = False
        if CURRENT_OBJECTS_MARKER in hint_block:
            after = hint_block.split(CURRENT_OBJECTS_MARKER, 1)[1].strip()
            if "\n" in after:
                after = after.split("\n", 1)[0].strip()
            if after.startswith(":"):
                after = after[1:].strip()
            objs, ok = _try_parse_py_list(after.rstrip("."))
        elif LEGACY_OBJECTS_MARKER in hint_block:
            objs, ok = _extract_objects_from_legacy_hint(hint_block)
        elif LEGACY_NEARBY_OBJECTS_MARKER in hint_block:
            after = hint_block.split(LEGACY_NEARBY_OBJECTS_MARKER, 1)[1].strip()
            objs, ok = _try_parse_py_list(after)
        elif "OBJ candidates" in hint_block and ":" in hint_block:
            after = hint_block.split(":", 1)[1].strip()
            objs, ok = _try_parse_py_list(after)
        else:
            objs, ok = _extract_objects_from_legacy_hint(hint_block)

        new_hint = render_hint(
            available_actions if actions_ok else (available_actions or ""),
            objs if ok else (objs or ""),
            task_description,
        )
        return f"{before}\n\n{new_hint}".rstrip(), True

    return content, False


def align_messages(messages: List[Dict[str, Any]], system_prompt: str, gold_rec: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    if gold_rec is not None:
        return align_messages_with_gold(messages, system_prompt, gold_rec)
    stats = {"system_updated": 0, "user_hint_updated": 0}
    task_description = _extract_task_description_from_messages(messages)
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content") != system_prompt:
                m["content"] = system_prompt
                stats["system_updated"] += 1
        elif role == "user":
            new_c, changed = rewrite_user_content_with_new_hint(m.get("content", ""), task_description)
            if changed:
                m["content"] = new_c
                stats["user_hint_updated"] += 1
    return stats


def load_pickle(path: str) -> Any:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: str) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def align_pkl(input_pkl: str, output_pkl: str, system_prompt: str, gold_by_id: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, int]:
    items = load_pickle(input_pkl)
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a list in PKL, got {type(items)}")
    total_stats = {"n_items": len(items), "system_updated": 0, "user_hint_updated": 0}
    for it in items:
        msgs = it.get("messages")
        if isinstance(msgs, list):
            key = str(it.get("task_id", it.get("data_id", "")))
            s = align_messages(msgs, system_prompt, (gold_by_id or {}).get(key))
            total_stats["system_updated"] += s["system_updated"]
            total_stats["user_hint_updated"] += s["user_hint_updated"]
    dump_pickle(items, output_pkl)
    return total_stats


def align_jsonl(input_jsonl: str, output_jsonl: str, system_prompt: str, gold_by_id: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, int]:
    total = 0
    total_stats = {"n_items": 0, "system_updated": 0, "user_hint_updated": 0}
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages")
            if isinstance(msgs, list):
                key = str(obj.get("task_id", obj.get("data_id", "")))
                s = align_messages(msgs, system_prompt, (gold_by_id or {}).get(key))
                total_stats["system_updated"] += s["system_updated"]
                total_stats["user_hint_updated"] += s["user_hint_updated"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1
    total_stats["n_items"] = total
    return total_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_py", default="env_service/environments/sciworld/sciworld_env.py")
    ap.add_argument("--input_pkl", default=None)
    ap.add_argument("--output_pkl", default=None)
    ap.add_argument("--input_jsonl", default=None)
    ap.add_argument("--output_jsonl", default=None)
    ap.add_argument("--gold_inputs", nargs="+", default=["data/sciworld/gold_trajectories.jsonl"])
    ap.add_argument("--inplace", action="store_true", help="Overwrite outputs in place (with .bak backups).")
    args = ap.parse_args()

    system_prompt = extract_system_prompt_from_env_py(args.env_py)
    gold_by_id = load_gold_records(args.gold_inputs) if args.gold_inputs else {}

    if args.input_pkl:
        if args.inplace:
            out_pkl = args.input_pkl
        else:
            out_pkl = args.output_pkl or (os.path.splitext(args.input_pkl)[0] + "_aligned.pkl")

        if args.inplace:
            shutil.copy2(args.input_pkl, args.input_pkl + ".bak")
        stats_pkl = align_pkl(args.input_pkl, out_pkl, system_prompt, gold_by_id)
        print("[PKL]", stats_pkl)

    if args.input_jsonl:
        in_jsonl = args.input_jsonl
        out_jsonl = args.output_jsonl or (os.path.splitext(in_jsonl)[0] + "_aligned.jsonl")
        if args.inplace:
            bak = in_jsonl + ".bak"
            shutil.copy2(in_jsonl, bak)
            # IMPORTANT: never read+write the same JSONL path (would truncate to 0 bytes).
            stats_jsonl = align_jsonl(bak, in_jsonl, system_prompt, gold_by_id)
        else:
            stats_jsonl = align_jsonl(in_jsonl, out_jsonl, system_prompt, gold_by_id)
        print("[JSONL]", stats_jsonl)


if __name__ == "__main__":
    main()

