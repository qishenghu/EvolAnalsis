#!/usr/bin/env python3
"""
Filter teacher trajectories by reward threshold.

This script filters teacher trajectories from a JSONL file, keeping only those
with reward >= threshold, and saves the results to both JSONL and PKL formats.

Usage:
    python scripts/filter_teacher_trajectories.py \
        --input data/teacher_trajectories/alfworld_qwen7b.jsonl \
        --output data/teacher_trajectories/alfworld_qwen7b_filtered \
        --threshold 1.0

Output:
    - {output}.jsonl: Filtered trajectories in JSONL format
    - {output}.pkl: Filtered trajectories in PKL format (faster to load)
"""

import argparse
import json
import pickle
import os
from typing import Dict, List, Any
from collections import defaultdict


def filter_trajectories(
    input_path: str,
    output_base: str,
    reward_threshold: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Filter teacher trajectories by reward threshold.
    
    Args:
        input_path: Path to input JSONL file
        output_base: Base path for output files (without extension)
        reward_threshold: Minimum reward to keep trajectory (default: 1.0)
        verbose: Whether to print progress information
        
    Returns:
        Statistics dictionary
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read and filter trajectories
    all_trajectories = []
    filtered_trajectories = []
    task_stats = defaultdict(lambda: {"total": 0, "kept": 0})
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                traj = json.loads(line)
                all_trajectories.append(traj)
                
                task_id = traj.get("task_id", "unknown")
                reward = traj.get("reward", 0.0)
                
                task_stats[task_id]["total"] += 1
                
                if reward >= reward_threshold:
                    filtered_trajectories.append(traj)
                    task_stats[task_id]["kept"] += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    # Save filtered trajectories to JSONL
    jsonl_path = f"{output_base}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for traj in filtered_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
    
    # Save filtered trajectories to PKL
    pkl_path = f"{output_base}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(filtered_trajectories, f)
    
    # Compute statistics
    stats = {
        "input_path": input_path,
        "output_jsonl": jsonl_path,
        "output_pkl": pkl_path,
        "reward_threshold": reward_threshold,
        "total_trajectories": len(all_trajectories),
        "kept_trajectories": len(filtered_trajectories),
        "filtered_out": len(all_trajectories) - len(filtered_trajectories),
        "keep_ratio": len(filtered_trajectories) / len(all_trajectories) if all_trajectories else 0,
        "total_tasks": len(task_stats),
        "tasks_with_kept": sum(1 for s in task_stats.values() if s["kept"] > 0),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Teacher Trajectory Filtering Results")
        print("=" * 60)
        print(f"Input file:        {input_path}")
        print(f"Output JSONL:      {jsonl_path}")
        print(f"Output PKL:        {pkl_path}")
        print(f"Reward threshold:  {reward_threshold}")
        print("-" * 60)
        print(f"Total trajectories:     {stats['total_trajectories']}")
        print(f"Kept trajectories:      {stats['kept_trajectories']}")
        print(f"Filtered out:           {stats['filtered_out']}")
        print(f"Keep ratio:             {stats['keep_ratio']:.2%}")
        print("-" * 60)
        print(f"Total unique tasks:     {stats['total_tasks']}")
        print(f"Tasks with kept trajs:  {stats['tasks_with_kept']}")
        print("=" * 60)
        
        # Show per-task statistics (top 10)
        if task_stats:
            print("\nPer-task statistics (showing up to 10 tasks):")
            sorted_tasks = sorted(task_stats.items(), key=lambda x: x[1]["kept"], reverse=True)
            for task_id, s in sorted_tasks[:10]:
                print(f"  Task {task_id}: {s['kept']}/{s['total']} kept")
            if len(sorted_tasks) > 10:
                print(f"  ... and {len(sorted_tasks) - 10} more tasks")
    
    # Save statistics
    stats_path = f"{output_base}_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"\nStatistics saved to: {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter teacher trajectories by reward threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter trajectories with reward >= 1.0 (default)
    python scripts/filter_teacher_trajectories.py \\
        --input data/teacher_trajectories/alfworld_qwen7b.jsonl \\
        --output data/teacher_trajectories/alfworld_qwen7b_filtered

    # Filter with custom threshold
    python scripts/filter_teacher_trajectories.py \\
        --input data/teacher_trajectories/alfworld_gpt4.jsonl \\
        --output data/teacher_trajectories/alfworld_gpt4_success \\
        --threshold 0.5
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSONL file containing teacher trajectories"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Base path for output files (without extension). Will create .jsonl and .pkl files"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=1.0,
        help="Minimum reward to keep trajectory (default: 1.0)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    filter_trajectories(
        input_path=args.input,
        output_base=args.output,
        reward_threshold=args.threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

