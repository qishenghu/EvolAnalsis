#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CATALYST 纯度分析(预注册 H-A2):hindsight 去提示后的学生自写轨迹 CLL。

对 scripts/collect_student_rollouts_hinted.py 采集的 A1/A2 臂轨迹
(schema_version=student_rollout_trajectory_v1,记录字段结构与
openrouter_teacher_trajectory_v2 完全一致)做"去提示重放 + CLL 打分":

  1. 只处理 success==true 的记录;失败记录计数(n_skipped_failure)后跳过。
  2. hindsight 去提示:从 messages 中剥除采集时注入的教师提示块。剥除
     逻辑以 scripts/collect_student_rollouts_hinted.py 的 HINT_TEMPLATE
     常量为准(ast 静态提取,不 import 该重依赖模块;提取出的模板
     sha256 与输入文件旁的 *.manifest.json 里 hint_template_sha256 交叉
     校验)。模板 = prefix + "{hint}" + suffix:
         prefix = "\\n\\n[Reference approach from a colleague]\\n"
         suffix = "\\n[End of reference. The reference may be imperfect; "
                  "solve the task yourself.]"
     剥除 = content 中 [prefix 起, suffix 止](含两端)的整段。
     断言(违反则该记录计入 n_strip_assert_failures 并跳过,退出码非零):
       * 恰好剥了一条消息;
       * 剥后全部 messages 不再含 "[Reference approach"。
  3. 剥除后按 analysis/cll_teacher_profile.py 同一链路逐 decision 重放
     (reconstruct_ext_messages → StructuredContextPolicy.build)重建
     prompt,并对 completion 逐 token 打 CLL(compute_decision_cll,
     分块 lm_head,f16+base64 落盘)。

sha 语义(与母本 cll_teacher_profile 的关键差异):
  * prompt_token_ids_sha256 / prompt_messages_sha256 / raw sha /
    context_stats **必然失配**——我们故意剥掉了上下文里的 hint。
    如实记录 prompt_sha_match_rate(预期 0.0),但不据此判失败、
    不产 mismatch_examples。
  * completion_token_ids_sha256 必须 100% 匹配——completion 原文未动,
    重分词应逐字节复现。任何失配如实计数(mismatch_examples 仅收
    completion 失配)并使退出码非零。

H-A2 判据(离线比对,不在本脚本内判定):去提示后学生自写轨迹的
think 段 CLL p10 应比教师原文浅 ≥3 倍(教师 flash think p10:
AF=-1.73 / WS=-1.21;学生自然噪声地板:AF=-0.44 / WS≈0)。

用法:
  # 登录节点冒烟(纯 CPU:剥除 + 重建 + completion sha 断言,不跑 forward)
  python analysis/catalyst_purity_score.py --dry-run \
      --config config/duet_paper_experiments_configs/iclr2027/collect_h200/alfworld_qwen35_4b_collect_h200.yaml \
      --input /projects_vol/gp_wangwy/qisheng/duet_h200/catalyst_pilots/pa/alfworld_a1.jsonl \
      --max-records 3
  # GPU 完整打分
  python analysis/catalyst_purity_score.py \
      --config ... --input .../alfworld_a1.jsonl \
      --output analysis_outputs/catalyst_purity/alfworld_a1_dehinted.jsonl \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# 母本机构复用:import 时 cll_teacher_profile 会 chdir 到仓库根并把仓库根/
# scripts 加入 sys.path(其模块头 L94-101),config/tokenizer/policy/重放/
# 分段/CLL/f16 落盘全部走它同一套代码。
_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
import cll_teacher_profile as ctp  # noqa: E402

# ---------------------------------------------------------------------------
# HINT_TEMPLATE:以 fork 采集器源码为准(ast 静态提取,避免 import 采集器
# 拉起 env_worker/exp_manager 等重依赖)。
# ---------------------------------------------------------------------------
HINTED_COLLECTOR_PATH = ctp.PROJECT_ROOT / "scripts" / "collect_student_rollouts_hinted.py"
HINT_MARKER = "[Reference approach"


def load_hint_template(path: Path = HINTED_COLLECTOR_PATH) -> str:
    """从采集器源码 ast 提取模块级 HINT_TEMPLATE 常量(L951 附近)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "HINT_TEMPLATE" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str) or value.count("{hint}") != 1:
                raise RuntimeError(f"HINT_TEMPLATE 形状异常: {value!r}")
            return value
    raise RuntimeError(f"HINT_TEMPLATE not found in {path}")


HINT_TEMPLATE = load_hint_template()
HINT_PREFIX, HINT_SUFFIX = HINT_TEMPLATE.split("{hint}")
HINT_TEMPLATE_SHA256 = hashlib.sha256(HINT_TEMPLATE.encode("utf-8")).hexdigest()
assert HINT_MARKER in HINT_PREFIX, "HINT_MARKER 必须是 prefix 的子串(剥后全局断言依赖它)"


def manifest_template_sha_match(input_path: Path) -> Optional[bool]:
    """与输入文件旁 *.manifest.json 里的 hint_template_sha256 交叉校验。

    返回 True/False;manifest 不存在或没有该字段时返回 None(不视为错误,
    但写入 summary 供审计)。
    """
    manifest_path = Path(str(input_path) + ".manifest.json")
    if not manifest_path.is_file():
        return None

    def find_key(node: Any, key: str) -> Optional[Any]:
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find_key(value, key)
                if found is not None:
                    return found
        return None

    recorded = find_key(json.loads(manifest_path.read_text(encoding="utf-8")),
                        "hint_template_sha256")
    if recorded is None:
        return None
    return str(recorded) == HINT_TEMPLATE_SHA256


# ---------------------------------------------------------------------------
# hindsight 去提示
# ---------------------------------------------------------------------------
def strip_hint_messages(record: Dict[str, Any]) -> None:
    """就地剥除 record["messages"] 中的 hint 块;断言失败抛 ValueError。

    HintedAgentFlow(采集器 L1029-1050)把 HINT_TEMPLATE.format(hint=...)
    追加到首条 user init 消息**末尾**,因此每条成功记录应恰有一条消息
    含 hint 块,且剥除后消息即恢复无提示原文。
    """
    stripped = 0
    for index, msg in enumerate(record["messages"]):
        content = str(msg["content"])
        if HINT_MARKER not in content:
            continue
        start = content.find(HINT_PREFIX)
        if start < 0:
            raise ValueError(
                f"message {index}: 含 {HINT_MARKER!r} 但找不到完整 HINT_PREFIX"
            )
        end = content.find(HINT_SUFFIX, start + len(HINT_PREFIX))
        if end < 0:
            raise ValueError(f"message {index}: 有 HINT_PREFIX 但缺 HINT_SUFFIX")
        msg["content"] = content[:start] + content[end + len(HINT_SUFFIX):]
        stripped += 1
    if stripped != 1:
        raise ValueError(f"应恰好剥除 1 条消息,实际剥了 {stripped} 条")
    for index, msg in enumerate(record["messages"]):
        if HINT_MARKER in str(msg["content"]):
            raise ValueError(f"剥除后 message {index} 仍含 {HINT_MARKER!r}")


# ---------------------------------------------------------------------------
# 汇总:在母本 FileSummary 上加去提示/过滤相关计数
# ---------------------------------------------------------------------------
@dataclass
class PuritySummary(ctp.FileSummary):
    """fork 差异:新增成功过滤与剥除断言的计数字段。"""

    n_records_scanned: int = 0          # 文件里扫过的全部记录(含失败)
    n_skipped_failure: int = 0          # success!=true 被跳过的记录数
    n_strip_assert_failures: int = 0    # 剥除断言失败(应为 0)
    hint_template_sha_match: Optional[bool] = None  # 与 manifest 交叉校验

    def as_dict(self, verify_only: bool) -> Dict[str, Any]:
        out = super().as_dict(verify_only)
        out["n_records_scanned"] = self.n_records_scanned
        out["n_skipped_failure"] = self.n_skipped_failure
        out["n_strip_assert_failures"] = self.n_strip_assert_failures
        out["hint_template_sha_match_manifest"] = self.hint_template_sha_match
        # 语义提醒:prompt sha 失配是**设计使然**(上下文被剥了 hint)
        out["prompt_sha_mismatch_expected"] = True
        return out


# ---------------------------------------------------------------------------
# 主处理:母本 process_file 的薄 fork
# 差异:(1) 只取 success==true;(2) 重放前剥 hint;(3) prompt sha 失配不产
# mismatch_examples(预期行为),examples 只收 completion 失配;(4) --max-records
# 语义改为"最多处理 N 条成功记录"(冒烟取'前 3 条 success'需要);
# (5) --dry-run 相当于母本 verify-only(不加载模型),并逐记录打印断言结果。
# ---------------------------------------------------------------------------
def process_file(
    path: Path,
    *,
    policy: Any,
    tokenizer: Any,
    resolved_config_sha: str,
    max_records: Optional[int],
    dry_run: bool,
    model: Any = None,
    lm_head: Any = None,
    device: str = "cuda",
    logit_chunk: int = 512,
    out_handle: Any = None,
) -> PuritySummary:
    summary = PuritySummary(path=str(path))
    summary.hint_template_sha_match = manifest_template_sha_match(path)
    if summary.hint_template_sha_match is False:
        raise SystemExit(
            f"HINT_TEMPLATE sha256 与 {path}.manifest.json 记录不一致——"
            "采集时模板与当前源码不同,剥除逻辑不可信,拒绝继续"
        )

    for line_no, record in ctp.iter_jsonl(path, None):
        summary.n_records_scanned += 1
        if not bool(record.get("success")):
            summary.n_skipped_failure += 1
            continue
        if max_records is not None and summary.n_records >= max_records:
            break
        summary.n_records += 1

        rec_sha = str(
            (record.get("metadata") or {}).get("resolved_student_config_sha256", "")
        )
        if rec_sha and rec_sha != resolved_config_sha:
            summary.config_sha_mismatch += 1

        # ---- hindsight 去提示(就地修改 messages;record 为本行私有)----
        try:
            strip_hint_messages(record)
        except ValueError as error:
            summary.n_strip_assert_failures += 1
            if len(summary.structure_errors) < 5:
                summary.structure_errors.append(
                    f"line {line_no} ({record.get('rollout_id')}): strip: {error}"
                )
            if dry_run:
                print(
                    f"[dry-run] line {line_no} rollout={record.get('rollout_id')} "
                    f"STRIP-FAIL: {error}"
                )
            continue

        # ---- 去提示后重放:与母本同一链路 ----
        try:
            replayed = list(ctp.replay_record_decisions(record, policy, tokenizer))
        except Exception as error:
            summary.n_records_bad_structure += 1
            if len(summary.structure_errors) < 5:
                summary.structure_errors.append(
                    f"line {line_no} ({record.get('rollout_id')}): {error}"
                )
            continue

        rec_prompt_sha_ok = 0
        rec_completion_sha_ok = 0
        for rd in replayed:
            summary.n_decisions += 1
            summary.prompt_sha_match += int(rd.prompt_sha_ok)
            summary.prompt_len_match += int(rd.prompt_len_ok)
            summary.prompt_msgs_sha_match += int(rd.prompt_msgs_sha_ok)
            summary.raw_sha_match += int(rd.raw_sha_ok)
            summary.stats_match += int(rd.stats_ok)
            summary.completion_sha_match += int(rd.completion_sha_ok)
            summary.completion_len_match += int(rd.completion_len_ok)
            rec_prompt_sha_ok += int(rd.prompt_sha_ok)
            rec_completion_sha_ok += int(rd.completion_sha_ok)

            # completion sha 必须匹配;失配才是异常(prompt 失配是预期)
            if not rd.completion_sha_ok and len(summary.mismatch_examples) < 3:
                summary.mismatch_examples.append(
                    {
                        "line": line_no,
                        "rollout_id": record.get("rollout_id"),
                        "step_index": rd.step_index,
                        "completion_sha_ok": False,
                        "completion_tokens_rebuilt": len(rd.completion_token_ids),
                        "completion_tokens_recorded": record["decision_trace"][
                            rd.step_index
                        ].get("completion_token_count"),
                    }
                )

            labels, offsets_ok = ctp.label_completion_tokens(
                rd.completion_content, tokenizer, rd.completion_token_ids
            )
            summary.offsets_bad += int(not offsets_ok)
            for lab in labels:
                summary.seg_tokens[lab] += 1

            if dry_run:
                continue

            # ---------------- GPU 完整模式(母本同款) ----------------
            logp, ent, cll = ctp.compute_decision_cll(
                model, lm_head, device,
                rd.prompt_token_ids, rd.completion_token_ids, logit_chunk,
            )
            lab_arr = np.asarray(labels)
            seg_means: Dict[str, Any] = {}
            for seg in ctp.SEGMENTS:
                mask = lab_arr == seg
                if mask.any():
                    summary.seg_values[seg]["logp"].append(logp[mask].astype(np.float32))
                    summary.seg_values[seg]["H"].append(ent[mask].astype(np.float32))
                    summary.seg_values[seg]["cll"].append(cll[mask].astype(np.float32))
                    seg_means[seg] = {
                        "n": int(mask.sum()),
                        "logp": float(logp[mask].mean()),
                        "H": float(ent[mask].mean()),
                        "cll": float(cll[mask].mean()),
                    }
            out_record = {
                "file": str(path),
                "line": line_no,
                "rollout_id": record.get("rollout_id"),
                "task_id": record.get("task_id"),
                "environment": record.get("environment"),
                "step_index": rd.step_index,
                "dehinted": True,
                "prompt_tokens": len(rd.prompt_token_ids),
                "completion_tokens": len(rd.completion_token_ids),
                "prompt_sha_ok": rd.prompt_sha_ok,       # 预期 False
                "completion_sha_ok": rd.completion_sha_ok,  # 必须 True
                "segments_rle": ctp.rle_labels(labels),
                "logp_f16_b64": ctp.f16_b64(logp),
                "entropy_f16_b64": ctp.f16_b64(ent),
                "cll_f16_b64": ctp.f16_b64(cll),
                "seg_means": seg_means,
            }
            out_handle.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        if dry_run:
            n_dec = len(replayed)
            print(
                f"[dry-run] line {line_no} rollout={record.get('rollout_id')} "
                f"strip=OK decisions={n_dec} "
                f"completion_sha {rec_completion_sha_ok}/{n_dec} "
                f"prompt_sha {rec_prompt_sha_ok}/{n_dec} (预期 0)"
            )

        if summary.n_records % 25 == 0:
            print(
                f"[{path.name}] {summary.n_records} success records, "
                f"{summary.n_decisions} decisions, "
                f"completion_sha match {summary.completion_sha_match}/{summary.n_decisions}",
                file=sys.stderr,
                flush=True,
            )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CATALYST 纯度分析:去提示重放 + 学生 CLL 打分(H-A2)"
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True,
        help="A1/A2 臂 rollout jsonl(可多个;须同一环境,与 --config 匹配)",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="采集时使用的学生 collect yaml(决定上下文渲染契约)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="纯 CPU 冒烟:剥除 + 重建 + completion sha 断言,不加载模型",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="完整模式必填:per-decision CLL jsonl;dry-run 可选:写 JSON 汇总",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="每个输入文件最多处理的**成功**记录数(默认全量)",
    )
    parser.add_argument("--device", default="cuda", help="完整模式使用的设备")
    parser.add_argument(
        "--logit-chunk", type=int, default=512,
        help="logits 分块的位置数(块内物化 [chunk, vocab] float32)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.output is None:
        raise SystemExit("完整模式必须提供 --output")

    # --- config / 契约 / tokenizer:全部走采集器同一套函数(经 ctp)---
    config = ctp.collector.compose_student_config(args.config)
    student_contract = ctp.collector.validate_student_contract(config)
    resolved_config_sha = ctp.collector._source_config_hash(config)
    tokenizer, tokenizer_manifest = ctp.collector.load_student_tokenizer(
        student_contract["student_model_path"]
    )
    print(
        f"config ok: env={student_contract['environment']} "
        f"resolved_sha={resolved_config_sha[:12]} "
        f"tokenizer={tokenizer_manifest['path']}\n"
        f"hint_template sha256={HINT_TEMPLATE_SHA256[:12]} "
        f"prefix={HINT_PREFIX!r} suffix={HINT_SUFFIX!r}",
        file=sys.stderr,
    )
    policy = ctp.StructuredContextPolicy(tokenizer, config.actor_rollout_ref.rollout)

    model = None
    lm_head = None
    out_handle = None
    if not args.dry_run:
        import torch
        from transformers import AutoModelForCausalLM

        print(
            f"loading model (bf16) from {student_contract['student_model_path']} "
            f"on {args.device} ...",
            file=sys.stderr,
        )
        model = AutoModelForCausalLM.from_pretrained(
            student_contract["student_model_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(args.device)
        model.eval()
        lm_head = model.get_output_embeddings()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_handle = args.output.open("w", encoding="utf-8")

    summaries: List[PuritySummary] = []
    try:
        for path in args.input:
            path = path.resolve()
            print(f"processing {path} ...", file=sys.stderr)
            summaries.append(
                process_file(
                    path,
                    policy=policy,
                    tokenizer=tokenizer,
                    resolved_config_sha=resolved_config_sha,
                    max_records=args.max_records,
                    dry_run=args.dry_run,
                    model=model,
                    lm_head=lm_head,
                    device=args.device,
                    logit_chunk=args.logit_chunk,
                    out_handle=out_handle,
                )
            )
    finally:
        if out_handle is not None:
            out_handle.close()

    payload = {
        "analysis": "catalyst_purity_dehinted_cll",
        "hypothesis": "H-A2",
        "config": str(args.config),
        "resolved_student_config_sha256": resolved_config_sha,
        "hint_template_sha256": HINT_TEMPLATE_SHA256,
        "hinted_collector": str(HINTED_COLLECTOR_PATH),
        "dry_run": bool(args.dry_run),
        "max_records": args.max_records,
        "files": [s.as_dict(verify_only=args.dry_run) for s in summaries],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.dry_run:
        summary_path = Path(str(args.output) + ".summary.json")
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # 退出码:completion sha 必须全配、结构/剥除断言零失败;
    # prompt sha 失配是预期(去提示),不参与判定。
    all_ok = all(
        s.completion_sha_match == s.n_decisions
        and s.n_records_bad_structure == 0
        and s.n_strip_assert_failures == 0
        and s.n_records > 0
        for s in summaries
    )
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
