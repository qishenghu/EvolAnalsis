#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align SciWorld teacher trajectories to the current training prompt format.

What it does:
- Replace the system prompt in each trajectory with the current prompt from
  `env_service/environments/sciworld/sciworld_env.py` (`_get_system_prompt`).
- Replace the per-turn user hint suffix to match the current `_get_action_hints()`
  format (by default: "OBJ candidates: [...]"), dropping legacy "Valid actions: ..."
  blocks while keeping the object list.

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


def extract_hint_template_from_env_py(env_py: str) -> str:
    """
    Extract the f-string template assigned to hint_str in _get_action_hints().

    Example:
      hint_str = f"OBJ candidates: {valid_objs}"
      -> returns 'OBJ candidates: {valid_objs}'
    """
    text = _read_text(env_py)
    # Prefer the first occurrence inside _get_action_hints; keep it simple/robust.
    block = None
    m_block = re.search(r"def\s+_get_action_hints\s*\([^)]*\)\s*->\s*str\s*:\s*([\s\S]*?)\n\s*def\s",
                        text, re.MULTILINE)
    if m_block:
        block = m_block.group(1)
    else:
        block = text

    m = re.search(r"hint_str\s*=\s*f([\"'])(.*?)\1", block, re.MULTILINE)
    if not m:
        # Fallback to a sane default matching current codebase usage.
        return "OBJ candidates: {valid_objs}"
    return m.group(2)


def render_hint(template: str, objs: Any) -> str:
    """
    Render a hint from the extracted template.
    Supports `{valid_objs}` and `{len(valid_objs)}` placeholders.
    """
    s = template
    if isinstance(objs, str):
        objs_repr = objs
        objs_len = ""
    else:
        objs_repr = repr(objs)
        try:
            objs_len = str(len(objs))
        except Exception:
            objs_len = ""
    s = s.replace("{valid_objs}", objs_repr)
    s = s.replace("{len(valid_objs)}", objs_len)
    return s.strip()


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


def rewrite_user_content_with_new_hint(
    content: str,
    hint_template: str,
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
        LEGACY_ACTIONS_MARKER,
        LEGACY_SUGGESTED_MARKER,
        LEGACY_NEARBY_OBJECTS_MARKER,
        LEGACY_OBJECTS_MARKER,
        "OBJ candidates",
    ]
    starts = [content.rfind(m) for m in markers]
    starts = [s for s in starts if s != -1]
    if starts:
        start = min(starts)
        before = content[:start].rstrip()
        hint_block = content[start:].strip()

        # Prefer extracting actual object list from known markers.
        objs = None
        ok = False
        if LEGACY_OBJECTS_MARKER in hint_block:
            objs, ok = _extract_objects_from_legacy_hint(hint_block)
        elif LEGACY_NEARBY_OBJECTS_MARKER in hint_block:
            after = hint_block.split(LEGACY_NEARBY_OBJECTS_MARKER, 1)[1].strip()
            objs, ok = _try_parse_py_list(after)
        elif "OBJ candidates" in hint_block and ":" in hint_block:
            after = hint_block.split(":", 1)[1].strip()
            objs, ok = _try_parse_py_list(after)
        else:
            objs, ok = _extract_objects_from_legacy_hint(hint_block)

        new_hint = render_hint(hint_template, objs if ok else (objs or ""))
        return f"{before}\n\n{new_hint}".rstrip(), True

    return content, False


def align_messages(messages: List[Dict[str, Any]], system_prompt: str, hint_template: str) -> Dict[str, int]:
    stats = {"system_updated": 0, "user_hint_updated": 0}
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content") != system_prompt:
                m["content"] = system_prompt
                stats["system_updated"] += 1
        elif role == "user":
            new_c, changed = rewrite_user_content_with_new_hint(m.get("content", ""), hint_template)
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


def align_pkl(input_pkl: str, output_pkl: str, system_prompt: str, hint_template: str) -> Dict[str, int]:
    items = load_pickle(input_pkl)
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a list in PKL, got {type(items)}")
    total_stats = {"n_items": len(items), "system_updated": 0, "user_hint_updated": 0}
    for it in items:
        msgs = it.get("messages")
        if isinstance(msgs, list):
            s = align_messages(msgs, system_prompt, hint_template)
            total_stats["system_updated"] += s["system_updated"]
            total_stats["user_hint_updated"] += s["user_hint_updated"]
    dump_pickle(items, output_pkl)
    return total_stats


def align_jsonl(input_jsonl: str, output_jsonl: str, system_prompt: str, hint_template: str) -> Dict[str, int]:
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
                s = align_messages(msgs, system_prompt, hint_template)
                total_stats["system_updated"] += s["system_updated"]
                total_stats["user_hint_updated"] += s["user_hint_updated"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1
    total_stats["n_items"] = total
    return total_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_py", default="env_service/environments/sciworld/sciworld_env.py")
    ap.add_argument("--input_pkl", default="data/teacher_trajectories/sciworld_gold_qwen72b_filtered.pkl")
    ap.add_argument("--output_pkl", default=None)
    ap.add_argument("--input_jsonl", default=None)
    ap.add_argument("--output_jsonl", default=None)
    ap.add_argument("--inplace", action="store_true", help="Overwrite outputs in place (with .bak backups).")
    args = ap.parse_args()

    system_prompt = extract_system_prompt_from_env_py(args.env_py)
    hint_template = extract_hint_template_from_env_py(args.env_py)

    if args.inplace:
        out_pkl = args.input_pkl
    else:
        out_pkl = args.output_pkl or (os.path.splitext(args.input_pkl)[0] + "_aligned.pkl")

    if args.inplace:
        shutil.copy2(args.input_pkl, args.input_pkl + ".bak")
    stats_pkl = align_pkl(args.input_pkl, out_pkl, system_prompt, hint_template)
    print("[PKL]", stats_pkl, "hint_template=", repr(hint_template))

    if args.input_jsonl:
        in_jsonl = args.input_jsonl
        out_jsonl = args.output_jsonl or (os.path.splitext(in_jsonl)[0] + "_aligned.jsonl")
        if args.inplace:
            bak = in_jsonl + ".bak"
            shutil.copy2(in_jsonl, bak)
            # IMPORTANT: never read+write the same JSONL path (would truncate to 0 bytes).
            stats_jsonl = align_jsonl(bak, in_jsonl, system_prompt, hint_template)
        else:
            stats_jsonl = align_jsonl(in_jsonl, out_jsonl, system_prompt, hint_template)
        print("[JSONL]", stats_jsonl)


if __name__ == "__main__":
    main()

