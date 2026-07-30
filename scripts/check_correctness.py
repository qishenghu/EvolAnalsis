import os
import json

file_path = "data/teacher_trajectories/qwen72b/alfworld_qwen72b.jsonl"

teacher_trajs = [json.loads(x) for x in open(file_path)]

task_perf = {}

for traj in teacher_trajs:
    task_id = str(traj['task_id'])
    if task_id not in task_perf:
        task_perf[task_id] = 0
    task_perf[task_id] = max(task_perf[task_id], traj['success'])

total_tasks = len(list(task_perf.keys()))
total_success = sum(list(task_perf.values()))

print(f"total_tasks count: {total_tasks}, total_success count: {total_success}")
