#!/usr/bin/env python
"""构建 CATALYST v2 entry-k 接管素材册(设计:CATALYST_v2_设计_2026-08-13.md §2)。

输入:教师成功轨迹 jsonl(openrouter_teacher_trajectory_v2,如
    data/teacher_trajectories/iclr2027_flash/alfworld_dsv4flash_success_dedup.jsonl)
输出:{"version","environment","tasks":{task_id:{"teacher_rollout_id",
    "init_messages","steps":[{"action","observation"}...]}}} + manifest。

选材:每任务取**最短**成功轨迹(重放成本最低、k 梯子最密)。
轨迹布局校验与试点 build_takeover_plan 逐字同构(init 前缀 + (assistant,user)
交替、末条 assistant);任何 decision 无合法 <action> 块 → 该轨迹弃用
(不是 fail-fast:换该任务的次短轨迹,全部不合格才丢任务并计数)。

**教师 think 从不进册**:steps 只存提取后的 action 文本与 env 观测。

用法:
  python scripts/build_catalyst_entry_book.py \
      --teacher data/teacher_trajectories/iclr2027_flash/alfworld_dsv4flash_success_dedup.jsonl \
      --env alfworld \
      --output data/catalyst_entry/alfworld_dsv4flash_entry.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentevolver.module.exp_manager.catalyst_entry import (  # noqa: E402
    ENTRY_BOOK_VERSION,
    extract_tagged_action,
)

TEACHER_SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def try_build_entry(record: dict) -> dict:
    """单条教师轨迹 → 册条目;布局/action 不合格抛 ValueError。"""
    messages = list(record["messages"])
    trace = list(record["decision_trace"])
    n_decisions = len(trace)
    if n_decisions < 2:
        raise ValueError(f"only {n_decisions} decision(s); need >= 2")
    first_user = next(
        (
            index
            for index, message in enumerate(messages)
            if str(message.get("role")) == "user"
        ),
        None,
    )
    if first_user is None:
        raise ValueError("no user init message")
    init_len = first_user + 1
    body = messages[init_len:]
    if len(body) != 2 * n_decisions - 1:
        raise ValueError(
            f"unexpected layout: {len(body)} post-init messages for "
            f"{n_decisions} decisions"
        )
    for offset, message in enumerate(body):
        expected_role = "assistant" if offset % 2 == 0 else "user"
        if str(message.get("role")) != expected_role:
            raise ValueError(
                f"alternation broken at post-init offset {offset}"
            )
    steps = []
    # 只需前 n−1 步的 (action, observation):k ≤ n−1(最后一步永远留给学生)
    for index in range(n_decisions - 1):
        action = extract_tagged_action(trace[index]["completion_content"])
        observation = str(body[2 * index + 1].get("content", ""))
        steps.append({"action": action, "observation": observation})
    return {
        "teacher_rollout_id": str(record.get("rollout_id", "")),
        "n_teacher_decisions": n_decisions,
        "init_messages": [
            {
                "role": str(m.get("role")),
                "content": str(m.get("content")),
            }
            for m in messages[:init_len]
        ],
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--env", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task-pool",
        type=Path,
        default=None,
        help="可选:任务清单(每行一个 task_id),用于覆盖率报告",
    )
    args = parser.parse_args()

    # 按任务收集成功轨迹,升序排轨迹长度(最短优先)
    by_task: dict[str, list[dict]] = defaultdict(list)
    n_lines = 0
    for line_number, line in enumerate(
        args.teacher.open("r", encoding="utf-8"), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"blank line at {args.teacher}:{line_number}")
        record = json.loads(line)
        if record.get("schema_version") != TEACHER_SCHEMA_VERSION:
            raise RuntimeError(
                f"schema mismatch at line {line_number}: "
                f"{record.get('schema_version')!r}"
            )
        if str(record.get("environment")) != args.env:
            raise RuntimeError(
                f"environment mismatch at line {line_number}: "
                f"{record.get('environment')!r} != {args.env!r}"
            )
        n_lines += 1
        if not bool(record.get("success")):
            continue
        by_task[str(record["task_id"])].append(record)

    tasks: dict[str, dict] = {}
    dropped: list[str] = []
    for task_id, records in sorted(by_task.items()):
        records.sort(key=lambda r: len(r.get("decision_trace") or []))
        entry = None
        for record in records:
            try:
                entry = try_build_entry(record)
                break
            except ValueError as error:
                print(
                    f"[build] task {task_id}: trajectory "
                    f"{record.get('rollout_id')} unusable: {error}",
                    file=sys.stderr,
                )
        if entry is None:
            dropped.append(task_id)
            continue
        tasks[task_id] = entry

    payload = {
        "version": ENTRY_BOOK_VERSION,
        "environment": args.env,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    args.output.write_text(body, encoding="utf-8")
    book_sha = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    coverage = None
    if args.task_pool is not None:
        pool = [
            line.strip()
            for line in args.task_pool.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        covered = sum(1 for t in pool if t in tasks)
        coverage = {"pool": len(pool), "covered": covered}

    manifest = {
        "builder": "scripts/build_catalyst_entry_book.py",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "version": ENTRY_BOOK_VERSION,
        "environment": args.env,
        "book_sha256": book_sha,
        "n_source_lines": n_lines,
        "n_tasks": len(tasks),
        "n_dropped_tasks": len(dropped),
        "dropped_task_ids": dropped[:50],
        "coverage": coverage,
        "sources": [
            {"path": str(args.teacher.resolve()), "sha256": sha256_file(args.teacher)}
        ],
    }
    manifest_path = args.output.with_name(
        args.output.name + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    print(
        f"[build] entry book: {len(tasks)} tasks ({len(dropped)} dropped) "
        f"-> {args.output}"
    )
    if coverage:
        print(
            f"[build] pool coverage: {coverage['covered']}/{coverage['pool']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
