#!/usr/bin/env python
"""构建 CATALYST v2 难度自举画像(设计 §3:消灭 v1 治理冷启动聋子期)。

从既有训练 run 的 rollout_log 重建**每任务裸臂成功率**:
  * rollout_log/{step}.jsonl 行序是完成序(乱序),行内无 task_id;
  * 归因用同 step 的 task_{step}.jsonl:从行 input 提取 "Your task is to: X"
    目标行,与任务 query 文本包含匹配;同 step 内目标文本撞车的任务整步丢弃
    (宁缺毋错);
  * 带 "[Reference approach" 的行是 hint 臂,不计入裸 SR;
  * GRPO run 全裸臂,是主数据源(每任务约 2 epoch × 8 rollouts)。

⚠️ 不要喂 catalystv2 的 rollout_log:entry 臂行(教师前缀接管)在日志文本上
与裸臂不可区分,会把接管成功率灌进"裸 SR"污染画像。只用 GRPO / v1 目录。

输出:{"schema": "catalyst_task_stats_v1",
      "tasks": {task_id: {"sr_bare": float, "n_bare": int}}} + manifest 字段内联。

用法:
  python scripts/build_catalyst_task_stats.py \
      --rollout-dir experiments/alfworld/p0_grpo_af_s0/rollout_log \
      --rollout-dir experiments/alfworld/p0_catalyst_af_s0/rollout_log \
      --output data/catalyst_entry/alfworld_task_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

GOAL_RE = re.compile(r"Your task is to: ([^\n]+)")
HINT_MARKER = "[Reference approach"


def goal_of(text: str) -> str | None:
    match = GOAL_RE.search(text or "")
    return match.group(1).strip() if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-dir", action="append", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="只用 step <= N 的 rollout(v2 自举建议 30:难度画像应反映"
        "新手期能力,全程平均会被后期进步稀释,角点被漏判)",
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        default=None,
        help="可选:另一份 stats json,其 tasks 只用于补本次未覆盖的任务"
        "(典型用法:新手期画像为主,全程画像补 41%% 的未见任务)",
    )
    args = parser.parse_args()

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [succ, n]
    n_rows = n_attributed = n_ambiguous_steps = n_hint_rows = 0
    for rollout_dir in args.rollout_dir:
        step_files = sorted(
            glob.glob(str(rollout_dir / "[0-9]*.jsonl")),
            key=lambda f: int(os.path.basename(f)[:-6]),
        )
        for step_file in step_files:
            step = int(os.path.basename(step_file)[:-6])
            if args.max_step is not None and step > args.max_step:
                continue
            task_file = rollout_dir / f"task_{step}.jsonl"
            if not task_file.is_file():
                continue
            step_tasks = [
                json.loads(line) for line in task_file.open(encoding="utf-8")
            ]
            # 目标文本 → task_id;同 step 撞车的目标全部拉黑
            goal_to_task: dict[str, str] = {}
            blacklist: set[str] = set()
            for task in step_tasks:
                goal = goal_of(str(task.get("query") or ""))
                if goal is None:
                    continue
                if goal in goal_to_task:
                    blacklist.add(goal)
                else:
                    goal_to_task[goal] = str(task["task_id"])
            if blacklist:
                n_ambiguous_steps += 1
                for goal in blacklist:
                    goal_to_task.pop(goal, None)
            for line in open(step_file, encoding="utf-8"):
                row = json.loads(line)
                n_rows += 1
                if HINT_MARKER in row.get("input", ""):
                    n_hint_rows += 1
                    continue
                goal = goal_of(row.get("input", ""))
                task_id = goal_to_task.get(goal or "")
                if task_id is None:
                    continue
                n_attributed += 1
                bucket = stats[task_id]
                bucket[0] += int(float(row.get("score") or 0.0) >= 1.0)
                bucket[1] += 1

    tasks = {
        task_id: {
            "sr_bare": (succ / n if n else 0.0),
            "n_bare": n,
        }
        for task_id, (succ, n) in sorted(stats.items())
    }
    n_fallback = 0
    if args.fallback is not None:
        fallback_payload = json.loads(
            args.fallback.read_text(encoding="utf-8")
        )
        for task_id, row in (fallback_payload.get("tasks") or {}).items():
            if str(task_id) not in tasks:
                tasks[str(task_id)] = {
                    "sr_bare": float(row.get("sr_bare", 0.0)),
                    "n_bare": int(row.get("n_bare", 0)),
                }
                n_fallback += 1
    payload = {
        "schema": "catalyst_task_stats_v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": [str(d.resolve()) for d in args.rollout_dir],
        "max_step": args.max_step,
        "fallback": str(args.fallback) if args.fallback else None,
        "n_fallback_tasks": n_fallback,
        "n_rows": n_rows,
        "n_attributed": n_attributed,
        "n_hint_rows_skipped": n_hint_rows,
        "n_ambiguous_steps": n_ambiguous_steps,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    corner = sum(1 for t in tasks.values() if t["sr_bare"] < 0.125)
    print(
        f"[stats] {len(tasks)} tasks ({n_attributed}/{n_rows} rows "
        f"attributed, {n_hint_rows} hint rows skipped); "
        f"corner(<12.5%): {corner}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
