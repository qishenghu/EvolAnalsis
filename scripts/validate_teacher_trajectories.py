#!/usr/bin/env python3
"""
Validate teacher trajectory files.

Usage:
    python scripts/validate_teacher_trajectories.py \
        --input data/teacher_trajectories/alfworld_qwen72b.jsonl
"""

import json
import argparse
from collections import Counter


def validate(input_file: str, verbose: bool = False):
    """验证 teacher trajectory 文件格式"""
    stats = Counter()
    errors = []
    
    print(f"Validating: {input_file}")
    print("-" * 50)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                traj = json.loads(line)
                
                # 必需字段检查
                required_fields = ["task_id", "messages", "reward", "success", "metadata"]
                for field in required_fields:
                    if field not in traj:
                        errors.append(f"Line {line_num}: missing required field '{field}'")
                        stats["missing_field"] += 1
                        continue
                
                # metadata 检查
                metadata = traj.get("metadata", {})
                if not metadata.get("is_teacher"):
                    errors.append(f"Line {line_num}: metadata.is_teacher should be True")
                    stats["invalid_metadata"] += 1
                
                # 统计
                stats["total"] += 1
                if traj.get("success"):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                
                if metadata.get("has_log_prob") or traj.get("log_probs"):
                    stats["has_log_prob"] += 1
                else:
                    stats["no_log_prob"] += 1
                
                # 统计 teacher model
                teacher_model = traj.get("teacher_model", metadata.get("teacher_model", "unknown"))
                stats[f"model:{teacher_model}"] += 1
                
                # messages 检查
                messages = traj.get("messages", [])
                if not messages:
                    errors.append(f"Line {line_num}: messages is empty")
                    stats["empty_messages"] += 1
                else:
                    stats["total_turns"] += len(messages)
                
                if verbose:
                    print(f"  Line {line_num}: task_id={traj.get('task_id')}, "
                          f"success={traj.get('success')}, turns={len(messages)}")
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                stats["invalid_json"] += 1
    
    # 打印结果
    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)
    
    print(f"\nTotal trajectories: {stats['total']}")
    print(f"  - Successful: {stats['success']} ({100*stats['success']/max(stats['total'],1):.1f}%)")
    print(f"  - Failed: {stats['failed']} ({100*stats['failed']/max(stats['total'],1):.1f}%)")
    
    print(f"\nLog probability:")
    print(f"  - Has log_prob: {stats['has_log_prob']}")
    print(f"  - No log_prob: {stats['no_log_prob']}")
    
    if stats['total'] > 0:
        print(f"\nAverage turns per trajectory: {stats['total_turns']/stats['total']:.1f}")
    
    print(f"\nTeacher models:")
    for key, value in sorted(stats.items()):
        if key.startswith("model:"):
            print(f"  - {key[6:]}: {value}")
    
    if stats["invalid_json"] > 0:
        print(f"\n⚠️  Invalid JSON lines: {stats['invalid_json']}")
    if stats["missing_field"] > 0:
        print(f"⚠️  Lines with missing fields: {stats['missing_field']}")
    
    if errors and verbose:
        print("\nErrors:")
        for err in errors[:20]:  # 只显示前 20 个错误
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    # 总结
    if stats["invalid_json"] == 0 and stats["missing_field"] == 0:
        print("\n✅ Validation passed!")
        return True
    else:
        print(f"\n❌ Validation failed with {len(errors)} errors")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate teacher trajectory files")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    args = parser.parse_args()
    
    success = validate(args.input, args.verbose)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

