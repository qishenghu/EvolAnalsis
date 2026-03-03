#!/usr/bin/env python3
"""
Replace system prompts in ScienceWorld teacher trajectory files.

Reads a teacher trajectory file (pkl or jsonl), replaces every system-role
message's content with the canonical system prompt from sciworld_env.py,
and writes out new .pkl and .jsonl files.

Usage:
    python scripts/replace_sciworld_system_prompt.py \
        --input data/teacher_trajectories/sciworld_gold_qwen72b_800_filtered.pkl \
        --suffix _newprompt
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

# Canonical system prompt from env_service/environments/sciworld/sciworld_env.py
SCIWORLD_SYSTEM_PROMPT = '''You are a scientific experiment assistant in a text-based simulation environment. Your task is to perform scientific experiments by interacting with objects in the environment.

At each step, you will receive:
1. The task description (what experiment you need to perform)
2. Your current observation (what you can see/do)
3. OBJ candidates (the objects that can be interacted with in the current state).

Available actions:
[
{"action": "open OBJ", "description": "open a container"},
{"action": "close OBJ", "description": "close a container"},
{"action": "activate OBJ", "description": "activate a device"},
{"action": "deactivate OBJ", "description": "deactivate a device"},
{"action": "connect OBJ to OBJ", "description": "connect electrical components"},
{"action": "disconnect OBJ", "description": "disconnect electrical components"},
{"action": "use OBJ [on OBJ]", "description": "use a device/item"},
{"action": "look around", "description": "describe the current room"},
{"action": "look at OBJ", "description": "describe an object in detail"},
{"action": "look in OBJ", "description": "describe a container's contents"},
{"action": "read OBJ", "description": "read a note or book"},
{"action": "move OBJ to OBJ", "description": "move an object to a container"},
{"action": "pick up OBJ", "description": "move an object to the inventory"},
{"action": "put down OBJ", "description": "drop an inventory item"},
{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"},
{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"},
{"action": "mix OBJ", "description": "chemically mix a container"},
{"action": "go to LOC", "description": "move to a new location"},
{"action": "eat OBJ", "description": "eat a food"},
{"action": "flush OBJ", "description": "flush a toilet"},
{"action": "focus on OBJ", "description": "signal intent on a task object"},
{"action": "wait", "description": "take no action for 10 iterations"},
{"action": "wait1", "description": "take no action for 1 iteration"},
{"action": "task", "description": "describe current task"},
{"action": "inventory", "description": "list your inventory"}
]

Important:
1. Read the task description carefully.
2. Plan your experiment steps logically.
3. Pay attention to the objects and locations available.
4. OBJ in the selected action should be replaced with one of the OBJ candidates using the exact string as provided.
5. If the environment returns "No known action matches that input.", that means your previous action is invalid and you should try more options.

In each turn, you must output your thought/reasoning and then output your action in the following format:
```
Thought:
your thoughts.
Action:
your next action
```
'''


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load trajectory data from pkl or jsonl."""
    ext = Path(path).suffix.lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ext in (".jsonl", ".json"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .pkl or .jsonl")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of trajectories, got {type(data)}")
    return data


def replace_system_prompts(data: List[Dict[str, Any]]) -> int:
    """Replace system prompts in-place. Returns count of replacements."""
    replaced = 0
    for entry in data:
        messages = entry.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = SCIWORLD_SYSTEM_PROMPT
                replaced += 1
                break  # only replace the first system message per entry
    return replaced


def save_data(data: List[Dict[str, Any]], output_stem: str):
    """Save data as both .pkl and .jsonl."""
    pkl_path = output_stem + ".pkl"
    jsonl_path = output_stem + ".jsonl"

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved pkl: {pkl_path} ({len(data)} entries)")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved jsonl: {jsonl_path} ({len(data)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Replace system prompts in ScienceWorld teacher trajectory files."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input teacher trajectory file (.pkl or .jsonl)",
    )
    parser.add_argument(
        "--suffix", "-s", default="_newprompt",
        help="Suffix to append to the output filename (default: '_newprompt')",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path stem (without extension). Overrides --suffix if set.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        return

    data = load_data(input_path)
    print(f"Loaded {len(data)} entries from {input_path}")

    replaced = replace_system_prompts(data)
    print(f"Replaced system prompts in {replaced}/{len(data)} entries")

    if args.output:
        output_stem = args.output
    else:
        p = Path(input_path)
        output_stem = str(p.parent / (p.stem + args.suffix))

    save_data(data, output_stem)


if __name__ == "__main__":
    main()
