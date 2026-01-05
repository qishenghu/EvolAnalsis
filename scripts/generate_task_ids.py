#!/usr/bin/env python3
"""
Generate task_ids file from environment service.

This script mirrors the behavior of TaskManager.load_tasks_from_environment(),
fetching task IDs directly from the environment service.

Usage:
    # Generate task_ids from AlfWorld environment (train split)
    python scripts/generate_task_ids.py \
        --env_url http://localhost:8000 \
        --env_type alfworld \
        --split train \
        --output data/alfworld/task_ids.txt

    # Generate task_ids with sampling
    python scripts/generate_task_ids.py \
        --env_url http://localhost:8000 \
        --env_type webshop \
        --split train \
        --output data/webshop/task_ids.txt \
        --max_tasks 100 \
        --shuffle

    # List available task_ids without saving
    python scripts/generate_task_ids.py \
        --env_url http://localhost:8000 \
        --env_type alfworld \
        --split val \
        --dry_run
"""

import argparse
import os
import random
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use standard logging (loguru is optional)
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate task IDs file from environment service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Environment service configuration
    parser.add_argument(
        "--env_url", type=str, default="http://localhost:8000",
        help="Environment service URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--env_type", type=str, required=True,
        choices=["alfworld", "webshop", "sciworld", "appworld"],
        help="Environment type"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "dev", "test", "test_normal"],
        help="Data split (default: train)"
    )
    
    # Sampling configuration
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Maximum number of tasks to include (default: all)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Shuffle task IDs (default: True)"
    )
    parser.add_argument(
        "--no_shuffle", action="store_false", dest="shuffle",
        help="Do not shuffle task IDs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="Ratio of tasks to sample (0.0-1.0, default: 1.0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (default: data/{env_type}/{split}_task_ids.txt)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print task IDs without saving to file"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import EnvClient
    try:
        from agentevolver.client.env_client import EnvClient
    except ImportError:
        logger.error("Failed to import EnvClient. Make sure you're in the AgentEvolver project directory.")
        sys.exit(1)
    
    # Connect to environment service
    logger.info(f"Connecting to environment service: {args.env_url}")
    env_client = EnvClient(base_url=args.env_url)
    
    # Get task IDs from environment (mirrors TaskManager.load_tasks_from_environment)
    try:
        logger.info(f"Fetching task IDs for env_type={args.env_type}, split={args.split}")
        task_ids = env_client.get_env_profile(
            env_type=args.env_type,
            split=args.split,
        )
        # Convert to string (as in TaskManager)
        task_ids = [str(tid) for tid in task_ids]
        logger.info(f"Retrieved {len(task_ids)} task IDs from environment service")
    except Exception as e:
        logger.error(f"Failed to get task IDs from environment service: {e}")
        logger.error("Make sure the environment service is running and accessible.")
        sys.exit(1)
    
    if not task_ids:
        logger.warning("No task IDs retrieved from environment service")
        sys.exit(0)
    
    # Shuffle if requested (mirrors TaskManager behavior)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(task_ids)
        logger.info(f"Shuffled task IDs with seed={args.seed}")
    
    # Apply max_tasks limit (mirrors TaskManager behavior)
    if args.max_tasks is not None and args.max_tasks > 0 and args.max_tasks < len(task_ids):
        original_count = len(task_ids)
        task_ids = task_ids[:args.max_tasks]
        logger.info(f"Limited from {original_count} to {len(task_ids)} tasks (max_tasks={args.max_tasks})")
    
    # Apply sample_ratio
    if args.sample_ratio < 1.0:
        n_sample = int(len(task_ids) * args.sample_ratio)
        task_ids = task_ids[:n_sample]
        logger.info(f"Sampled {len(task_ids)} task IDs (ratio={args.sample_ratio})")
    
    # Dry run: just print
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"Environment: {args.env_type}")
        print(f"Split: {args.split}")
        print(f"Total task IDs: {len(task_ids)}")
        print(f"{'='*60}")
        
        if len(task_ids) <= 20:
            print("\nTask IDs:")
            for tid in task_ids:
                print(f"  {tid}")
        else:
            print(f"\nFirst 10 task IDs:")
            for tid in task_ids[:10]:
                print(f"  {tid}")
            print(f"  ... ({len(task_ids) - 20} more)")
            print(f"Last 10 task IDs:")
            for tid in task_ids[-10:]:
                print(f"  {tid}")
        return
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = f"data/{args.env_type}/{args.split}_task_ids.txt"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Save to file
    with open(output_path, 'w') as f:
        # Write header comments
        f.write(f"# Task IDs generated from environment service\n")
        f.write(f"# Environment: {args.env_type}\n")
        f.write(f"# Split: {args.split}\n")
        f.write(f"# Total: {len(task_ids)}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Shuffle: {args.shuffle}, Seed: {args.seed}\n")
        if args.max_tasks:
            f.write(f"# Max tasks: {args.max_tasks}\n")
        f.write(f"#\n")
        
        # Write task IDs
        for tid in task_ids:
            f.write(f"{tid}\n")
    
    logger.info(f"Saved {len(task_ids)} task IDs to {output_path}")
    print(f"\n✅ Successfully saved {len(task_ids)} task IDs to {output_path}")


if __name__ == "__main__":
    main()
