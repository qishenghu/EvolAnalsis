#!/usr/bin/env python3
"""
Generate SciWorld task_ids.txt from the *official* ScienceWorld python package.

Why this script?
- `scripts/generate_task_ids.py` fetches task IDs from AgentEvolver's env_service.
- For SciWorld, sometimes you want the *full* official index space (data_idx) and
  then split it (e.g., exclude eval indices from `sciworld_test.json`).

This script reproduces the same `data_idx -> (taskName, variationIdx)` enumeration
as AgentGym SciWorld server (`AgentGym/agentenv-sciworld/agentenv_sciworld/environment.py`).

Usage (recommended):
  conda run -n agentenv-sciworld --no-capture-output \
    python scripts/generate_sciworld_task_ids.py \
      --split train \
      --output data/sciworld/task_ids.txt \
      --shuffle --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from typing import List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SciWorld task IDs (data_idx) from official ScienceWorld package")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Split to generate. 'train' excludes eval indices from --eval_file; 'test' uses eval indices; 'all' uses all indices.",
    )
    p.add_argument(
        "--eval_file",
        type=str,
        default="env_service/environments/sciworld/sciworld_test.json",
        help="Eval indices file (list of {'item_id': 'sciworld_123'}). Used for train/test split.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/sciworld/task_ids.txt",
        help="Output path (default: data/sciworld/task_ids.txt)",
    )
    p.add_argument("--shuffle", action="store_true", default=True, help="Shuffle task IDs (default: True)")
    p.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="Do not shuffle task IDs")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    p.add_argument("--max_tasks", type=int, default=None, help="Limit to first N tasks after split+shuffle")
    return p.parse_args()


def load_eval_indices(eval_file: str) -> List[int]:
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    idxs: List[int] = []
    for it in data:
        item_id = str(it.get("item_id", ""))
        m = re.match(r"^sciworld_(\d+)$", item_id)
        if not m:
            raise ValueError(f"Unexpected SciWorld eval item_id: {item_id}")
        idxs.append(int(m.group(1)))
    return idxs


def main() -> None:
    args = parse_args()

    # Import here so the script can be parsed without scienceworld installed.
    try:
        from scienceworld import ScienceWorldEnv
    except Exception as e:
        print("Failed to import scienceworld. Please run under conda env 'agentenv-sciworld'.", file=sys.stderr)
        raise

    # Keep consistent with AgentGym sciworld server enumeration.
    exceptions: Set[str] = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}

    env = ScienceWorldEnv()
    try:
        games = []
        for key, value in env.tasks.items():
            if str(key) in exceptions:
                continue
            max_var = env.getMaxVariations(value)
            for i in range(int(max_var)):
                games.append({"taskName": value, "variationIdx": i})
        total_all = len(games)
    finally:
        env.close()

    eval_idxs = load_eval_indices(args.eval_file) if args.split in ("train", "test") else []
    eval_set = set(eval_idxs)

    all_ids = list(range(total_all))
    if args.split == "all":
        task_ids = all_ids
    elif args.split == "test":
        # Keep only valid ids within range, but warn if file contains out-of-range indices.
        out_of_range = [i for i in eval_idxs if i < 0 or i >= total_all]
        if out_of_range:
            print(f"WARNING: {len(out_of_range)} eval indices are out of range [0, {total_all}). Example: {out_of_range[:5]}", file=sys.stderr)
        task_ids = [i for i in eval_idxs if 0 <= i < total_all]
    else:  # train
        task_ids = [i for i in all_ids if i not in eval_set]

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(task_ids)

    if args.max_tasks is not None and args.max_tasks > 0:
        task_ids = task_ids[: args.max_tasks]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Task IDs generated from official ScienceWorld package\n")
        f.write("# Environment: sciworld\n")
        f.write(f"# Split: {args.split}\n")
        f.write(f"# Total: {len(task_ids)}\n")
        f.write(f"# Total(all): {total_all}\n")
        if args.split in ("train", "test"):
            f.write(f"# Eval file: {args.eval_file}\n")
            f.write(f"# Eval size: {len(eval_set)}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Shuffle: {args.shuffle}, Seed: {args.seed}\n")
        if args.max_tasks:
            f.write(f"# Max tasks: {args.max_tasks}\n")
        f.write("#\n")
        for tid in task_ids:
            f.write(f"{tid}\n")

    print(f"✅ Saved {len(task_ids)} sciworld task ids to {args.output}")


if __name__ == "__main__":
    main()

