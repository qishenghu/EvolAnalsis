#!/usr/bin/env python3
"""
Convert ALFWorld teacher trajectories from "react" format (Thought:\n...\nAction:\n...)
to "react_tags" format (<think>...</think>\n<action>...</action>).

This is needed for Qwen 3 compatibility, where the model natively generates <think> tags.
Unifying ALFWorld to use react_tags format ensures consistency with WebShop and Qwen 3's
native thinking mode.

Usage:
    python scripts/convert_alfworld_react_to_tags.py \
        --input data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered.jsonl \
        --output data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.jsonl

    # Also convert .pkl if needed:
    python scripts/convert_alfworld_react_to_tags.py \
        --input data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered.jsonl \
        --output data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.jsonl \
        --output-pkl data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path


def convert_assistant_message(content: str) -> str:
    """
    Convert a single assistant message from react format to react_tags format.

    Cases handled:
    1. "Thought:\n...\n\nAction:\n..."  -> "<think>\n...\n</think>\n<action>\n...\n</action>"
    2. "Action:\n..."                    -> "<action>\n...\n</action>"
    3. Other (e.g., initial ack)         -> unchanged
    """
    content = content.strip()

    # Case 1: Thought + Action
    # Use rsplit to handle the LAST "Action:" (in case "Action" appears in the thought text)
    thought_match = re.match(r'^Thought:\s*\n(.*)', content, flags=re.DOTALL)
    if thought_match:
        rest = thought_match.group(1)
        # Split on the last occurrence of "Action:" line
        action_split = re.split(r'\n\s*Action:\s*\n', rest)
        if len(action_split) >= 2:
            thought_text = "\n".join(action_split[:-1]).strip()
            action_text = action_split[-1].strip()
            return f"<think>\n{thought_text}\n</think>\n<action>\n{action_text}\n</action>"
        else:
            # Thought without Action (rare) — wrap thought only
            thought_text = rest.strip()
            return f"<think>\n{thought_text}\n</think>"

    # Case 2: Action only (no Thought)
    action_only_match = re.match(r'^Action:\s*\n(.*)', content, flags=re.DOTALL)
    if action_only_match:
        action_text = action_only_match.group(1).strip()
        return f"<action>\n{action_text}\n</action>"

    # Case 3: Neither Thought nor Action (e.g., initial acknowledgment message)
    return content


def convert_trajectory(traj: dict) -> dict:
    """Convert all assistant messages in a trajectory to react_tags format."""
    traj = dict(traj)  # shallow copy
    new_messages = []
    for msg in traj["messages"]:
        if msg["role"] == "assistant":
            new_msg = dict(msg)
            new_msg["content"] = convert_assistant_message(msg["content"])
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)
    traj["messages"] = new_messages

    # Update metadata to record the format conversion
    if "metadata" not in traj or traj["metadata"] is None:
        traj["metadata"] = {}
    traj["metadata"]["action_format"] = "react_tags"
    traj["metadata"]["converted_from"] = "react"

    return traj


def main():
    parser = argparse.ArgumentParser(description="Convert ALFWorld teacher data from react to react_tags format")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--output-pkl", default=None, help="Optional: also output as pickle file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    trajectories = []
    converted_count = 0
    total_messages_converted = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)

            # Count how many assistant messages will change
            original_contents = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]
            converted = convert_trajectory(traj)
            new_contents = [m["content"] for m in converted["messages"] if m["role"] == "assistant"]

            changed = sum(1 for o, n in zip(original_contents, new_contents) if o != n)
            total_messages_converted += changed
            if changed > 0:
                converted_count += 1

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            trajectories.append(converted)

            if line_num % 5000 == 0:
                print(f"  Processed {line_num} trajectories...")

    print(f"\nConversion complete:")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Trajectories with changes: {converted_count}")
    print(f"  Total assistant messages converted: {total_messages_converted}")
    print(f"  Output JSONL: {output_path}")

    # Optional pickle output
    if args.output_pkl:
        pkl_path = Path(args.output_pkl)
        with open(pkl_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"  Output PKL: {pkl_path}")

    # Show a sample conversion for verification
    if trajectories:
        sample = trajectories[0]
        print(f"\n--- Sample conversion (task_id={sample.get('task_id', 'N/A')}) ---")
        for i, msg in enumerate(sample["messages"]):
            if msg["role"] == "assistant":
                content_preview = msg["content"][:150].replace("\n", "\\n")
                print(f"  messages[{i}] (assistant): {content_preview}...")


if __name__ == "__main__":
    main()
