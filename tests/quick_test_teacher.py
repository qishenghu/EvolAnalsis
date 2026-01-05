#!/usr/bin/env python
"""快速测试 Teacher Trajectory 加载"""
import sys, os
sys.path.insert(0, '.')

import json
print("Step 1: Loading JSONL file...")

with open('data/teacher_trajectories/alfworld_qwen7b.jsonl', 'r') as f:
    lines = f.readlines()
    
print(f"Found {len(lines)} lines")

for i, line in enumerate(lines):
    if line.strip():
        d = json.loads(line)
        print(f"\nTraj {i}: task_id={d.get('task_id')}, reward={d.get('reward')}")
        print(f"  keys: {list(d.keys())}")
        if 'log_probs' in d:
            lp = d['log_probs']
            import numpy as np
            print(f"  log_probs: len={len(lp)}, mean={np.mean(lp):.4f}")
        if 'log_probs_per_turn' in d:
            print(f"  log_probs_per_turn: {len(d['log_probs_per_turn'])} turns")

print("\n" + "="*50)
print("Step 2: Testing ExperienceManager...")

try:
    from agentevolver.module.exp_manager.exp_manager import ExperienceManager
    
    config = {
        "teacher_experience": {
            "enable": True,
            "data_path": "data/teacher_trajectories/alfworld_qwen7b.jsonl",
            "exp_ratio": 0.25,
            "select_mode": "random",
            "use_log_prob": True,
        },
        "experience_replay": {"enable": False}
    }
    
    exp_manager = ExperienceManager(config)
    print(f"Teacher enabled: {exp_manager.teacher_enabled}")
    print(f"Tasks loaded: {len(exp_manager.teacher_task2trajectories)}")
    
    for tid, trajs in exp_manager.teacher_task2trajectories.items():
        print(f"  Task {tid}: {len(trajs)} trajectories")
        for j, t in enumerate(trajs):
            print(f"    [{j}] has_log_prob={t.metadata.get('has_log_prob')}, is_teacher={t.metadata.get('is_teacher')}")
    
    # 测试获取轨迹
    task_ids = list(exp_manager.teacher_task2trajectories.keys())
    retrieved = exp_manager.get_teacher_trajectories(task_ids, num_per_task=1)
    print(f"\nRetrieved {len(retrieved)} trajectories")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

