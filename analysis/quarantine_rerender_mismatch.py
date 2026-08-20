#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重渲染失配 decision 隔离扫描器(纯 CPU,verify 语义)。

对学生/教师轨迹 jsonl 逐 decision 重放采集器的上下文渲染(复用
analysis/cll_teacher_profile.py 的同一套重放机构),把 prompt/completion
token sha 与记录不一致的 decision 落成隔离清单,供训练侧 mask:
凡在清单里的 (rollout_id, step_index),其记录的 prompt_token_ids_sha256
描述的 token 流与当前环境重建结果不同,不应作为可信的 token 级监督。

背景(2026-08 deepsearch p400 排查结论):失配根因是采集器进程(duet 环境,
tokenizers 0.21.2)与 vLLM 服务/校验(vllm2 环境, tokenizers 0.22.2)对稀有
文字(如泰卢固语 'తి')的分词版本差异,而非上下文截断非幂等。因此本扫描器
必须用与训练/校验一致的环境(vllm2)运行,清单口径 = "按当前环境重建会得到
不同 token 流的 decision"。

用法:
  PYTHONPATH=<repo根> /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/vllm2/bin/python \
      analysis/quarantine_rerender_mismatch.py \
      --input /path/deepsearch_qwen35_4b_student_rollouts_p400.jsonl \
      --config config/duet_paper_experiments_configs/iclr2027/collect_h200/deepsearch_qwen35_4b_collect_h200.yaml \
      --output analysis_outputs/quarantine/deepsearch_p400_quarantine.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "analysis"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

# 复用 cll_teacher_profile 的重放机构(其模块级代码会 chdir 到仓库根并
# 导入采集器,config/tokenizer/sha 语义全部与采集时同构)。
import cll_teacher_profile as replayer  # noqa: E402

collector = replayer.collector


def scan_file(
    path: Path,
    *,
    policy: Any,
    tokenizer: Any,
    max_records: Optional[int],
) -> Dict[str, Any]:
    """逐 decision 重放一个轨迹文件,返回失配清单与汇总。"""
    mismatches: List[Dict[str, Any]] = []
    bad_structure: List[Dict[str, Any]] = []
    n_records = 0
    n_decisions = 0
    affected: set[str] = set()

    for line_no, record in replayer.iter_jsonl(path, max_records):
        n_records += 1
        rollout_id = str(record.get("rollout_id"))
        try:
            replayed = list(
                replayer.replay_record_decisions(record, policy, tokenizer)
            )
        except Exception as error:
            # 结构不符 → 整条 rollout 无法校验,按整体隔离处理
            bad_structure.append(
                {"line": line_no, "rollout_id": rollout_id, "error": str(error)}
            )
            affected.add(rollout_id)
            continue

        for rd in replayed:
            n_decisions += 1
            if rd.prompt_sha_ok and rd.completion_sha_ok:
                continue
            decision = record["decision_trace"][rd.step_index]
            reasons = []
            if not rd.prompt_sha_ok:
                reasons.append("prompt_sha")
            if not rd.completion_sha_ok:
                reasons.append("completion_sha")
            mismatches.append(
                {
                    "rollout_id": rollout_id,
                    "step_index": rd.step_index,
                    "recorded_tokens": int(decision.get("prompt_token_count", -1)),
                    "rebuilt_tokens": len(rd.prompt_token_ids),
                    # 附加对账字段(不影响 mask 口径,便于人工复核)
                    "line": line_no,
                    "reason": "+".join(reasons),
                    "recorded_prompt_sha": rd.expected_prompt_sha,
                    "rebuilt_prompt_sha": rd.rebuilt_prompt_sha,
                }
            )
            affected.add(rollout_id)
        if n_records % 50 == 0:
            print(
                f"[{path.name}] {n_records} records, {n_decisions} decisions, "
                f"{len(mismatches)} mismatches",
                file=sys.stderr,
                flush=True,
            )

    return {
        "input": str(path),
        "summary": {
            "n_records": n_records,
            "n_records_bad_structure": len(bad_structure),
            "n_decisions": n_decisions,
            "n_mismatch_decisions": len(mismatches),
            "n_affected_rollouts": len(affected),
            "mismatch_rate": (len(mismatches) / n_decisions) if n_decisions else None,
        },
        "mismatch_decisions": mismatches,
        "bad_structure_rollouts": bad_structure,
        "affected_rollout_ids": sorted(affected),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重渲染失配 decision 隔离扫描器(供训练侧 mask)"
    )
    parser.add_argument("--input", type=Path, required=True, help="轨迹 jsonl")
    parser.add_argument(
        "--config", type=Path, required=True, help="采集时的学生 collect yaml"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="隔离清单 JSON 输出路径"
    )
    parser.add_argument(
        "--max-records", type=int, default=None, help="最多处理的记录数(默认全量)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # 与采集器/校验器同一套 config、契约、tokenizer 构造(见 cll_teacher_profile)
    config = collector.compose_student_config(args.config)
    student_contract = collector.validate_student_contract(config)
    resolved_config_sha = collector._source_config_hash(config)
    tokenizer, tokenizer_manifest = collector.load_student_tokenizer(
        student_contract["student_model_path"]
    )
    policy = replayer.StructuredContextPolicy(
        tokenizer, config.actor_rollout_ref.rollout
    )
    print(
        f"config ok: env={student_contract['environment']} "
        f"resolved_sha={resolved_config_sha[:12]} "
        f"tokenizer={tokenizer_manifest['path']}",
        file=sys.stderr,
    )

    import tokenizers
    import transformers

    result = scan_file(
        args.input.resolve(),
        policy=policy,
        tokenizer=tokenizer,
        max_records=args.max_records,
    )
    payload = {
        "config": str(args.config),
        "resolved_student_config_sha256": resolved_config_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # 清单口径依赖运行环境的分词行为,落盘版本便于追溯
        "runtime": {
            "python": sys.executable,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"quarantine list written to {args.output}", file=sys.stderr)
    # 有失配/坏结构时返回 3,便于上层脚本判断(与 cll_teacher_profile 一致)
    clean = (
        summary["n_mismatch_decisions"] == 0
        and summary["n_records_bad_structure"] == 0
    )
    return 0 if clean else 3


if __name__ == "__main__":
    raise SystemExit(main())
