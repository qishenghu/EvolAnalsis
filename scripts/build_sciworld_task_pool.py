#!/usr/bin/env python3
"""固化 SciWorld 教师采集用的规范任务池(canonical task pool)。

背景(2026-08-07,sciworld/deepsearch 接入教师采集器):
  采集器 ``canonical_task_pool("sciworld")`` 需要一个字节稳定的池文件——
  池文件的 sha256 会作为 ``source_sha256`` 进入每个采集战役的契约,
  因此不能在采集时临时调用 ``SciworldEnv.get_query_list``(其结果隐式依赖
  ``sciworld_test.json`` 的当前内容),而要在这里一次性生成并固化。

生成算法(与 SciworldEnv.get_query_list("train") 逐字一致):
  1. 读 ``env_service/environments/sciworld/sciworld_test.json``(AgentGym
     eval 200 题,item_id 形如 "sciworld_606"),解析出 eval 下标集合;
  2. 训练池 = [0, max(eval)+1) 中去掉 eval 下标,保持升序——即 eval 补集;
  3. 连同来源文件 sha256 等溯源信息写入
     ``data/sciworld/canonical_task_pool.json``。

字节稳定性军规:
  - 输出不含时间戳等可变字段,重跑必须逐字节一致(sha256 不变);
  - 若 ``sciworld_test.json`` 变了,本脚本会生成不同的池文件——此时旧契约
    的 source_sha256 校验会失败,属于预期的 fail-closed 行为。

用法:
  python scripts/build_sciworld_task_pool.py            # 生成(已存在且一致则跳过)
  python scripts/build_sciworld_task_pool.py --check    # 只校验不写盘
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = PROJECT_ROOT / "env_service/environments/sciworld/sciworld_test.json"
OUTPUT_FILE = PROJECT_ROOT / "data/sciworld/canonical_task_pool.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_indices() -> list[int]:
    """解析 eval 下标;与 SciworldEnv.get_query_list 的解析逐字一致。"""
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    indices: list[int] = []
    for item in data:
        item_id = str(item.get("item_id", ""))
        match = re.match(r"^sciworld_(\d+)$", item_id)
        if not match:
            raise ValueError(f"Unexpected SciWorld eval item_id: {item_id}")
        indices.append(int(match.group(1)))
    return indices


def build_payload() -> dict:
    eval_indices = load_eval_indices()
    eval_set = set(eval_indices)
    candidate_max = (max(eval_indices) if eval_indices else 0) + 1
    # 与 get_query_list("train") 一致:升序补集,id 为十进制字符串
    task_ids = [str(i) for i in range(candidate_max) if i not in eval_set]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("sciworld canonical pool contains duplicates")
    return {
        "environment": "sciworld",
        "generator": "scripts/build_sciworld_task_pool.py",
        "algorithm": (
            "complement-of-eval-v1: task_ids = [str(i) for i in "
            "range(max(eval_idx)+1) if i not in eval_idx], ascending; "
            "mirrors SciworldEnv.get_query_list('train') byte-for-byte"
        ),
        "eval_source_path": str(EVAL_FILE.relative_to(PROJECT_ROOT)),
        "eval_source_sha256": sha256_file(EVAL_FILE),
        "eval_count": len(eval_indices),
        "candidate_max": candidate_max,
        "count": len(task_ids),
        "task_ids": task_ids,
    }


def serialize(payload: dict) -> str:
    # 固定序列化形态(sort_keys + indent=2 + 末尾换行),保证重跑逐字节一致
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只校验,不写盘")
    args = parser.parse_args()

    text = serialize(build_payload())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if OUTPUT_FILE.exists():
        existing = OUTPUT_FILE.read_text(encoding="utf-8")
        if existing == text:
            print(f"OK: {OUTPUT_FILE} 已存在且逐字节一致 (sha256={digest})")
            return 0
        # 池文件参与既有契约,静默覆盖会使旧战役 source_sha256 失配——拒绝。
        print(
            f"FAIL: {OUTPUT_FILE} 已存在但内容不一致;"
            "如确要重建,请先人工移走旧文件(注意旧契约将失效)。"
        )
        return 1

    if args.check:
        print(f"CHECK-ONLY: 目标不存在,应生成 sha256={digest}")
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    payload = json.loads(text)
    print(
        f"written: {OUTPUT_FILE}\n"
        f"  pool_count={payload['count']} eval_count={payload['eval_count']} "
        f"candidate_max={payload['candidate_max']}\n"
        f"  file_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
