#!/usr/bin/env python3
"""CATALYST 提示素材构建(M1 工作项⑥)。

从 openrouter_teacher_trajectory_v2 jsonl(通常是 convert_teacher_v2_to_training
产出的 *_success_dedup.jsonl)提取全部覆盖任务的教师 think 摘要,做试点同款
卫生清洗,落盘 data/catalyst_hints/{env}_{teacher}.json + manifest。

清洗管线 = agentevolver/module/exp_manager/catalyst.py::build_hint_from_v2_record
(单一事实源;训练加载端与本脚本共用)。该管线是 2026-08 试点提取逻辑的正式
归档,已对试点产物逐字节回归验证:AF 120/120、WS 110/110 全等
(见 docs/design/CATALYST_IMPL_SPEC.md F9;回归 fixture 在
tests/test_catalyst_hints.py)。

输出格式 {task_id: {"raw": hint}} 与试点 hints 文件同构("raw" 键沿用,
CatalystHintBook / 试点采集器两端兼容)。

用法(纯 CPU,无重依赖;登录节点可跑):
  python scripts/build_catalyst_hints.py \
      --env alfworld --teacher dsv4flash \
      --input data/teacher_trajectories/iclr2027_flash/alfworld_dsv4flash_success_dedup.jsonl \
      --output-dir data/catalyst_hints

fail-fast:重复 task_id、缺 think 记录数超过 --max-missing-think-frac、
输出目录已有同名文件(除非 --force)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 只 import 清洗常量/函数(catalyst.py 顶层无 torch 等重依赖)。
from agentevolver.module.exp_manager.catalyst import (  # noqa: E402
    HINT_CLEAN_VERSION,
    HINT_MAX_CHARS,
    build_hint_from_v2_record,
)

SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            yield line_number, record


def build(
    inputs: Sequence[Path],
    *,
    env: str,
    max_missing_think_frac: float,
) -> tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    hints: Dict[str, Dict[str, str]] = {}
    n_records = 0
    missing_think: List[str] = []
    source_manifest = []
    for path in inputs:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input jsonl does not exist: {path}")
        source_manifest.append(
            {"path": str(path), "sha256": sha256_file(path)}
        )
        for line_number, record in iter_records(path):
            n_records += 1
            schema = record.get("schema_version")
            if schema != SCHEMA_VERSION:
                raise RuntimeError(
                    f"{path}:{line_number}: schema_version {schema!r} != "
                    f"{SCHEMA_VERSION!r}"
                )
            record_env = str(record.get("environment", "")).lower()
            if record_env and record_env != env:
                raise RuntimeError(
                    f"{path}:{line_number}: environment {record_env!r} != "
                    f"--env {env!r}"
                )
            task_id = str(record.get("task_id", "")).strip()
            if not task_id:
                raise RuntimeError(f"{path}:{line_number}: missing task_id")
            if task_id in hints:
                raise RuntimeError(
                    f"{path}:{line_number}: duplicate task_id {task_id!r} "
                    "(input must be the deduped success file)"
                )
            hint = build_hint_from_v2_record(record)
            if hint is None:
                # 缺 think 的任务不产 hint,训练侧自然落 R0(计数并列出)。
                missing_think.append(task_id)
                continue
            hints[task_id] = {"raw": hint}

    if n_records == 0:
        raise RuntimeError("no records found in inputs")
    missing_frac = len(missing_think) / n_records
    if missing_frac > max_missing_think_frac:
        raise RuntimeError(
            f"{len(missing_think)}/{n_records} records lack a <think> block "
            f"({missing_frac:.1%} > --max-missing-think-frac "
            f"{max_missing_think_frac:.1%}); first: {missing_think[:10]}"
        )
    if not hints:
        raise RuntimeError("cleaning produced zero hints")

    lengths = sorted(len(entry["raw"]) for entry in hints.values())
    manifest = {
        "builder": "scripts/build_catalyst_hints.py",
        "clean_version": HINT_CLEAN_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "schema_version": SCHEMA_VERSION,
        "sources": source_manifest,
        "n_source_records": n_records,
        "n_hints": len(hints),
        "n_missing_think": len(missing_think),
        "missing_think_task_ids": missing_think,
        "hint_max_chars": HINT_MAX_CHARS,
        "hint_len_min": lengths[0],
        "hint_len_max": lengths[-1],
        "hint_len_p50": lengths[len(lengths) // 2],
        "n_capped": sum(1 for n in lengths if n == HINT_MAX_CHARS),
    }
    return hints, manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CATALYST hint 素材构建(教师 think 摘要 + 试点同款清洗)"
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True,
        help="openrouter v2 教师 jsonl(建议 *_success_dedup.jsonl)",
    )
    parser.add_argument("--env", required=True,
                        choices=["alfworld", "webshop", "sciworld", "deepsearch"])
    parser.add_argument(
        "--teacher", required=True,
        help="教师短名(落盘文件名 {env}_{teacher}.json 的一部分,如 dsv4flash)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "catalyst_hints",
    )
    parser.add_argument(
        "--max-missing-think-frac", type=float, default=0.02,
        help="允许缺 <think> 的记录占比上限(超过即 fail-fast;默认 2%%)",
    )
    parser.add_argument("--force", action="store_true",
                        help="覆盖已存在的输出文件")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_path = args.output_dir / f"{args.env}_{args.teacher}.json"
    manifest_path = Path(str(output_path) + ".manifest.json")
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"output already exists: {output_path} (use --force to overwrite)"
        )

    hints, manifest = build(
        args.input,
        env=args.env,
        max_missing_think_frac=args.max_missing_think_frac,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(hints, ensure_ascii=False, sort_keys=True, indent=0) + "\n",
        encoding="utf-8",
    )
    manifest["output_sha256"] = sha256_file(output_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"built {manifest['n_hints']} hints "
        f"({manifest['n_missing_think']} tasks lack <think>) -> {output_path}\n"
        f"manifest -> {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
