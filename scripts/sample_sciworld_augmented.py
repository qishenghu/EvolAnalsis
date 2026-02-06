#!/usr/bin/env python3
"""
Sample/filter records from sciworld_gold_augmented.jsonl for synthesis tests.

Typical use:
  python scripts/sample_sciworld_augmented.py \
    --input data/teacher_trajectories/sciworld_gold_augmented.jsonl \
    --output /tmp/sciworld_aug_verified_20.jsonl \
    --require_verified --require_success \
    --max_records 20 --shuffle --seed 42
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_records", type=int, default=50)
    p.add_argument("--shuffle", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require_verified", action="store_true", default=False)
    p.add_argument("--require_success", action="store_true", default=False)
    p.add_argument("--require_no_mismatch", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    assert os.path.exists(args.input), f"Input not found: {args.input}"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))

    if args.require_verified:
        records = [r for r in records if r.get("verified") is True]
    if args.require_success:
        # prefer replay_success if available, else fall back to original success
        records = [r for r in records if (r.get("replay_success") is True or r.get("success") is True)]
    if args.require_no_mismatch:
        records = [r for r in records if not (r.get("mismatches") or [])]

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(records)

    records = records[: max(0, int(args.max_records))]

    with open(args.output, "w", encoding="utf-8") as wf:
        for r in records:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()

