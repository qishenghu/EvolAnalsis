#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教师轨迹 CLL(centered log-likelihood)分段画像。

对 scripts/collect_openrouter_teacher_trajectories.py 采集的教师轨迹
(schema_version=openrouter_teacher_trajectory_v2)做两件事:

1. --verify-only(纯 CPU):
   逐 decision 重放采集器的上下文渲染,重建 prompt token ids,与记录里的
   `prompt_token_ids_sha256` 比对;同时把 completion 重新用学生 tokenizer 分词,
   按 <think>...</think> / <action>...</action> / 其余(other) 切成 token 段,
   输出每文件的汇总(记录数、decision 数、sha 匹配率、各段 token 占比)。

2. 完整模式(需 GPU):
   用 HF transformers 加载学生模型(Qwen3.5-4B, bf16, 单卡),对每个 decision 的
   (重建 prompt + completion) 做一次 forward,对 completion 的每个 token t 计算
       log p(x_t | 前缀)   —— 该 token 的对数似然
       H_t                 —— 该位置模型输出分布的熵
       CLL_t = log p + H_t —— "中心化对数似然"(以 -H = E[log p] 为中心)
   注意 Qwen3.5 词表 ~248K:绝不一次性物化 [seq, vocab] 的 float32 logits,
   而是取 base model 的 last_hidden_state 后按位置分块过 lm_head(见
   `compute_decision_cll`)。逐 decision 把 per-token 的 (logp, H, cll, segment)
   以 float16+base64 压缩数组写入 --output jsonl,并累计每文件的分段统计。

渲染语义的来源(本脚本不重新实现渲染,全部 import 采集器同一套代码):
  * scripts/collect_openrouter_teacher_trajectories.py
      - compose_student_config      L277-284(hydra 组合学生 config)
      - validate_student_contract   L287-380(v5 采集契约校验)
      - load_student_tokenizer      L396-426(tokenizer 工件 sha 门禁)
      - DecisionTraceRecorder.chat  L669-716:
          prompt_ids  = chat_template_ids(tokenizer, prompt_messages,
                                          add_generation_prompt=True)  (L673-675)
          completion_ids = tokenizer.encode(content, add_special_tokens=False)
                                                                       (L684-686)
      - DecisionTraceRecorder.observe L718-767:
          prompt_token_ids_sha256 = sha256(",".join(str(id)))          (L744-746)
          completion_token_ids_sha256 同构                              (L751-755)
          prompt_messages_sha256 = canonical_json_hash(prompt_messages) (L707)
  * agentevolver/module/context_manager/context_policy.py
      - StructuredContextPolicy.build L316-441:逐 decision 的受管上下文构造
        (recent_turns 保留、旧 turn 压成 action-only、history observation 裁剪、
         超预算 drop 旧 turn)。采集时 AgentFlow.execute(agent_flow.py L145)每步
        调 cmt.prepare_next_llm_context()(cmt_linear.py L241-255)→ 即本函数。
      - StructuredContextPolicy.ids_hash L146-149:与采集器 sha 完全同构。
  * agentevolver/module/context_manager/cmt_linear.py
      - Linear_CMT.__init__ L59-61:StructuredContextPolicy(tokenizer,
        config.actor_rollout_ref.rollout) —— 本脚本按同一调用方式构造 policy。
      - to_role_content_raw L882-890:记录里的 `messages` 就是 full_context 的
        原始 content 视图(remove_last_context L190-203 已弹掉末尾 env 消息,
        采集器 L965-967 写入)。context_policy.build 读的同样是 msg.content,
        因此用 `messages` + 重建 author 即可逐 decision 无损重放。
  * agentevolver/module/context_manager/cmt_base.py
      - chat_template_ids L7-18、ExtendedMessage L289-471。

消息结构 → author 重建(重放的唯一"适配层",语义依据见上):
  采集成功轨迹的 messages = init 前缀 + [assistant, user]*(T-1) + [assistant],
  其中 T = len(decision_trace)。init 前缀里可能含 assistant(固定的确认语),
  所以不能用"第一条 assistant"划界;用结构公式 n_init = len(messages)-(2T-1),
  并与 decision_trace[0].prompt_message_count 交叉校验(decision 0 的受管上下文
  恰为 init 前缀)。i < n_init → author="initialization";其后 assistant →
  "llm",user → "env"(与 save_init_input/save_llm_output/save_env_output 写入
  full_context 的 author 一致,cmt_linear.py L258-621)。

用法示例:
  # 纯 CPU 校验(每文件取 50 条)
  python analysis/cll_teacher_profile.py --verify-only \
      --config config/duet_paper_experiments_configs/iclr2027/collect_h200/alfworld_qwen35_4b_collect_h200.yaml \
      --input /path/alfworld_qwen35_122b_t1600_r1.jsonl /path/alfworld_dsv4flash_t1600_r1.jsonl \
      --max-records 50
  # 完整 GPU 模式
  python analysis/cll_teacher_profile.py \
      --config ... --input ... --output analysis_outputs/cll_teacher_profile/alfworld.jsonl \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径与采集器代码导入
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 采集器 compose_student_config 里 hydra searchpath 是 'file://config' 等相对
# 路径(相对 CWD),采集时从仓库根目录运行;这里保持同一语义。
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 采集器本体:config/契约/tokenizer/sha 语义全部从这里复用
import collect_openrouter_teacher_trajectories as collector  # noqa: E402

# 采集器渲染链路使用的同一套上下文构造代码(采集器经由
# AgentFlow -> Linear_CMT -> StructuredContextPolicy 间接调用)
from agentevolver.module.context_manager.cmt_base import ExtendedMessage  # noqa: E402
from agentevolver.module.context_manager.context_policy import (  # noqa: E402
    StructuredContextPolicy,
)

# 分段标签(固定顺序,输出统计用)
SEGMENTS = ("think", "action", "other")


# ---------------------------------------------------------------------------
# 逐 decision 重放
# ---------------------------------------------------------------------------
@dataclass
class ReplayedDecision:
    """一个 decision 的重放结果与校验标志。"""

    step_index: int
    prompt_token_ids: List[int]
    prompt_messages: List[Dict[str, str]]
    completion_token_ids: List[int]
    completion_content: str
    # 校验标志(与记录里的 sha/count 比对)
    prompt_sha_ok: bool = False
    prompt_len_ok: bool = False
    prompt_msgs_sha_ok: bool = False
    raw_sha_ok: bool = False
    stats_ok: bool = False
    completion_sha_ok: bool = False
    completion_len_ok: bool = False
    # 诊断信息
    rebuilt_prompt_sha: str = ""
    expected_prompt_sha: str = ""


def reconstruct_ext_messages(
    record: Dict[str, Any], tokenizer: Any
) -> Tuple[int, List[ExtendedMessage]]:
    """从记录的 `messages` 重建带 author 的 full_context 视图。

    结构与 author 语义见模块 docstring;失败(结构不符)抛 ValueError。
    """
    messages = record["messages"]
    trace = record["decision_trace"]
    n_dec = len(trace)
    if n_dec <= 0:
        raise ValueError("record has no decisions")
    # 成功轨迹:末尾 env 消息已被 remove_last_context 弹掉,因此
    # messages = init 前缀 + (assistant, user)*(T-1) + assistant
    n_init = len(messages) - (2 * n_dec - 1)
    if n_init < 1:
        raise ValueError(
            f"message/decision count mismatch: {len(messages)} msgs, {n_dec} decisions"
        )
    # 交叉校验 1:decision 0 的受管上下文恰是 init 前缀(无 turn 可压/可丢)
    pmc0 = int(trace[0].get("prompt_message_count", -1))
    if pmc0 != n_init:
        raise ValueError(
            f"init prefix length {n_init} != decision0 prompt_message_count {pmc0}"
        )
    # 交叉校验 2:策略 turn 必须严格 assistant/user 交替,且以 assistant 收尾
    for k in range(n_dec):
        if messages[n_init + 2 * k]["role"] != "assistant":
            raise ValueError(f"expected assistant at message {n_init + 2 * k}")
        if k < n_dec - 1 and messages[n_init + 2 * k + 1]["role"] != "user":
            raise ValueError(f"expected user at message {n_init + 2 * k + 1}")

    ext: List[ExtendedMessage] = []
    for i, msg in enumerate(messages):
        if i < n_init:
            author = "initialization"
        elif msg["role"] == "assistant":
            author = "llm"
        else:
            author = "env"
        # token_generator="manual" + token_arr=[]:build() 只读 author/role/
        # content,不消费 token_arr,避免无谓分词。content 即 to_role_content_raw
        # 的原文,与采集时 context_policy.build 读到的 msg.content 逐字节一致。
        ext.append(
            ExtendedMessage(
                author=author,
                role=str(msg["role"]),
                content=str(msg["content"]),
                token_arr=[],
                token_generator="manual",
                tokenizer=tokenizer,
            )
        )
    return n_init, ext


def replay_record_decisions(
    record: Dict[str, Any],
    policy: StructuredContextPolicy,
    tokenizer: Any,
) -> Iterator[ReplayedDecision]:
    """逐 decision 重建受管 prompt 并与 decision_trace 的审计字段比对。"""
    n_init, ext = reconstruct_ext_messages(record, tokenizer)
    for t, decision in enumerate(record["decision_trace"]):
        # decision t 采样时的 full_context = init 前缀 + 前 t 个 (llm, env) 对,
        # 即 messages[: n_init + 2t](恰好停在第 t 条策略 assistant 之前)。
        prefix = ext[: n_init + 2 * t]
        # 与采集时同一函数:StructuredContextPolicy.build(context_policy.py L316)
        result = policy.build(prefix)
        rebuilt_sha = policy.ids_hash(result.prompt_token_ids)
        expected_sha = str(decision.get("prompt_token_ids_sha256", ""))

        # completion 重新分词:与采集器 L684-686 完全一致的调用
        content = str(decision.get("completion_content", ""))
        completion_ids = list(tokenizer.encode(content, add_special_tokens=False))

        rd = ReplayedDecision(
            step_index=int(decision.get("step_index", t)),
            prompt_token_ids=list(result.prompt_token_ids),
            prompt_messages=list(result.messages),
            completion_token_ids=completion_ids,
            completion_content=content,
            rebuilt_prompt_sha=rebuilt_sha,
            expected_prompt_sha=expected_sha,
        )
        rd.prompt_sha_ok = rebuilt_sha == expected_sha
        rd.prompt_len_ok = len(result.prompt_token_ids) == int(
            decision.get("prompt_token_count", -1)
        )
        # prompt_messages_sha256:采集器 L707 canonical_json_hash(prompt_messages)
        rd.prompt_msgs_sha_ok = collector.canonical_json_hash(
            result.messages
        ) == str(decision.get("prompt_messages_sha256", ""))
        # raw(未压缩)prompt sha:build() 的 raw_prompt_hash,对应记录里的
        # raw_prompt_token_ids_sha256(采集器 L747 <- context_result.raw_prompt_hash)
        rd.raw_sha_ok = str(result.raw_prompt_hash) == str(
            decision.get("raw_prompt_token_ids_sha256", "")
        )
        rd.stats_ok = dict(result.stats) == dict(decision.get("context_stats") or {})
        rd.completion_sha_ok = policy.ids_hash(completion_ids) == str(
            decision.get("completion_token_ids_sha256", "")
        )
        rd.completion_len_ok = len(completion_ids) == int(
            decision.get("completion_token_count", -1)
        )
        yield rd


# ---------------------------------------------------------------------------
# completion 分段:<think>...</think> / <action>...</action> / other
# ---------------------------------------------------------------------------
def segment_char_spans(content: str) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    """求 think 段与 action 段的字符区间。

    语义对齐采集器的判定习惯(collect L771-776 _valid_tagged_action /
    context_policy L199-224 _action_only):
      * post-think 部分取最后一个 '</think>' 之后;
      * think 段 = [第一个 '<think>'(缺失则 0), 最后一个 '</think>' 结束);
      * action 段 = post-think 里所有 '<action>...</action>' 匹配(IGNORECASE|DOTALL)。
    标签本身计入所属段。
    """
    think_span: Optional[Tuple[int, int]] = None
    close_idx = content.rfind("</think>")
    if close_idx >= 0:
        end = close_idx + len("</think>")
        open_idx = content.find("<think>")
        start = open_idx if 0 <= open_idx < close_idx else 0
        think_span = (start, end)
        post_begin = end
    else:
        post_begin = 0
    action_spans: List[Tuple[int, int]] = []
    for m in re.finditer(
        r"<action>.*?</action>", content[post_begin:], flags=re.IGNORECASE | re.DOTALL
    ):
        action_spans.append((post_begin + m.start(), post_begin + m.end()))
    return think_span, action_spans


def label_completion_tokens(
    content: str, tokenizer: Any, expected_ids: Sequence[int]
) -> Tuple[List[str], bool]:
    """用 fast-tokenizer 的 offset mapping 把每个 completion token 标成段。

    返回 (labels, offsets_ok)。offsets_ok=False 表示带 offset 的分词结果与
    采集器式 encode(add_special_tokens=False) 的 ids 不一致(理论上不该发生),
    此时退化为整段 'other' 以免误标。
    """
    enc = tokenizer(
        content, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    if ids != list(expected_ids):
        return ["other"] * len(expected_ids), False

    think_span, action_spans = segment_char_spans(content)

    def overlap(s: int, e: int, span: Tuple[int, int]) -> bool:
        return s < span[1] and e > span[0]

    labels: List[str] = []
    for s, e in offsets:
        if e <= s:  # 零宽 token(不应出现),按后一字符归属
            e = s + 1
        if think_span is not None and overlap(s, e, think_span):
            labels.append("think")  # 跨界 token 优先归 think
        elif any(overlap(s, e, a) for a in action_spans):
            labels.append("action")
        else:
            labels.append("other")
    return labels, True


def rle_labels(labels: Sequence[str]) -> str:
    """把逐 token 段标签做游程压缩,如 'think:64,other:1,action:9'。"""
    return ",".join(f"{key}:{len(list(grp))}" for key, grp in groupby(labels))


# ---------------------------------------------------------------------------
# 完整模式:分块 logits 的 CLL 计算
# ---------------------------------------------------------------------------
def compute_decision_cll(
    model: Any,
    lm_head: Any,
    device: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    logit_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一个 decision 做 forward,返回 completion 每 token 的 (logp, H, cll)。

    显存要点:Qwen3.5 词表 ~248K,一次性物化 [seq, vocab] float32 logits 需要
    数十 GB。这里只跑 base decoder 拿 last_hidden_state([seq, hidden],bf16,
    已含末端 RMSNorm),再按位置分块过 lm_head:每块 [chunk, vocab] float32,
    chunk=512 时约 0.5GB 级瞬时占用。
    位置约定:token x_t(t 从 prompt_len 起)的分布由 hidden[t-1] 给出,故取
    hidden[prompt_len-1 : seq_len-1] 与 targets = ids[prompt_len:] 对齐。
    """
    import torch

    ids = list(prompt_ids) + list(completion_ids)
    prompt_len = len(prompt_ids)
    seq_len = len(ids)
    input_t = torch.tensor([ids], dtype=torch.long, device=device)

    logp_out = np.zeros(len(completion_ids), dtype=np.float32)
    ent_out = np.zeros(len(completion_ids), dtype=np.float32)

    with torch.inference_mode():
        # get_decoder():不带 lm_head 的 base model,避免 HF 自动算全量 logits
        hidden = model.get_decoder()(input_ids=input_t).last_hidden_state[0]
        targets = input_t[0, prompt_len:]
        n = len(completion_ids)
        for start in range(0, n, logit_chunk):
            end = min(start + logit_chunk, n)
            h = hidden[prompt_len - 1 + start : prompt_len - 1 + end]
            logits = lm_head(h).float()                      # [c, V] fp32
            log_z = torch.logsumexp(logits, dim=-1)          # [c]
            tgt = targets[start:end]
            tgt_logit = logits.gather(1, tgt[:, None]).squeeze(1)
            logp = tgt_logit - log_z                         # log p(x_t)
            probs = torch.softmax(logits, dim=-1)
            # H = logZ - E_p[logit] 数学上等于 -(p*logp).sum,避免再存 logp_all
            ent = log_z - (probs * logits).sum(dim=-1)
            logp_out[start:end] = logp.cpu().numpy()
            ent_out[start:end] = ent.cpu().numpy()
            del logits, probs
    cll = logp_out + ent_out
    return logp_out, ent_out, cll


def f16_b64(values: np.ndarray) -> str:
    """float16 + base64 压缩一维数组(小端字节序)。"""
    return base64.b64encode(
        np.asarray(values, dtype="<f2").tobytes()
    ).decode("ascii")


# ---------------------------------------------------------------------------
# 汇总统计
# ---------------------------------------------------------------------------
@dataclass
class FileSummary:
    """一个输入文件的累计汇总。"""

    path: str
    n_records: int = 0
    n_records_bad_structure: int = 0
    n_decisions: int = 0
    prompt_sha_match: int = 0
    prompt_len_match: int = 0
    prompt_msgs_sha_match: int = 0
    raw_sha_match: int = 0
    stats_match: int = 0
    completion_sha_match: int = 0
    completion_len_match: int = 0
    offsets_bad: int = 0
    seg_tokens: Dict[str, int] = field(
        default_factory=lambda: {seg: 0 for seg in SEGMENTS}
    )
    config_sha_mismatch: int = 0
    mismatch_examples: List[Dict[str, Any]] = field(default_factory=list)
    structure_errors: List[str] = field(default_factory=list)
    # 完整模式:每段的 per-token cll/logp/H 值池(np.float32)
    seg_values: Dict[str, Dict[str, List[np.ndarray]]] = field(
        default_factory=lambda: {
            seg: {"logp": [], "H": [], "cll": []} for seg in SEGMENTS
        }
    )

    def as_dict(self, verify_only: bool) -> Dict[str, Any]:
        total_seg = sum(self.seg_tokens.values())
        out: Dict[str, Any] = {
            "path": self.path,
            "n_records": self.n_records,
            "n_records_bad_structure": self.n_records_bad_structure,
            "n_decisions": self.n_decisions,
            "prompt_sha_match_rate": _rate(self.prompt_sha_match, self.n_decisions),
            "prompt_len_match_rate": _rate(self.prompt_len_match, self.n_decisions),
            "prompt_msgs_sha_match_rate": _rate(
                self.prompt_msgs_sha_match, self.n_decisions
            ),
            "raw_sha_match_rate": _rate(self.raw_sha_match, self.n_decisions),
            "context_stats_match_rate": _rate(self.stats_match, self.n_decisions),
            "completion_sha_match_rate": _rate(
                self.completion_sha_match, self.n_decisions
            ),
            "completion_len_match_rate": _rate(
                self.completion_len_match, self.n_decisions
            ),
            "offsets_inconsistent_decisions": self.offsets_bad,
            "config_sha_mismatch_records": self.config_sha_mismatch,
            "segment_token_counts": dict(self.seg_tokens),
            "segment_token_fractions": {
                seg: _rate(cnt, total_seg) for seg, cnt in self.seg_tokens.items()
            },
            "mismatch_examples": self.mismatch_examples,
            "structure_errors": self.structure_errors[:5],
        }
        if not verify_only:
            seg_stats: Dict[str, Any] = {}
            for seg in SEGMENTS:
                pools = self.seg_values[seg]
                if not pools["cll"]:
                    seg_stats[seg] = None
                    continue
                stat: Dict[str, Any] = {}
                for name in ("logp", "H", "cll"):
                    arr = np.concatenate(pools[name])
                    stat[name] = {
                        "n": int(arr.size),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "p10": float(np.percentile(arr, 10)),
                        "p25": float(np.percentile(arr, 25)),
                        "p50": float(np.percentile(arr, 50)),
                        "p75": float(np.percentile(arr, 75)),
                        "p90": float(np.percentile(arr, 90)),
                    }
                seg_stats[seg] = stat
            out["segment_cll_stats"] = seg_stats
        return out


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def iter_jsonl(path: Path, max_records: Optional[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """按行读 jsonl,产出 (1 起始的行号, record)。"""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_records is not None and line_no > max_records:
                break
            if not line.strip():
                continue
            yield line_no, json.loads(line)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_file(
    path: Path,
    *,
    policy: StructuredContextPolicy,
    tokenizer: Any,
    resolved_config_sha: str,
    max_records: Optional[int],
    verify_only: bool,
    model: Any = None,
    lm_head: Any = None,
    device: str = "cuda",
    logit_chunk: int = 512,
    out_handle: Any = None,
    dump_dir: Optional[Path] = None,
) -> FileSummary:
    """处理一个输入文件;verify_only=False 时同时计算 CLL 并写 out_handle。"""
    summary = FileSummary(path=str(path))
    for line_no, record in iter_jsonl(path, max_records):
        summary.n_records += 1
        # config 溯源:记录采集时的 resolved config sha 与本次 --config 是否一致。
        # 不一致仍可重放(渲染只依赖契约字段),但要显式计数并提醒。
        rec_sha = str(
            (record.get("metadata") or {}).get("resolved_student_config_sha256", "")
        )
        if rec_sha and rec_sha != resolved_config_sha:
            summary.config_sha_mismatch += 1

        try:
            replayed = list(replay_record_decisions(record, policy, tokenizer))
        except Exception as error:  # 结构不符:计数并留样本,不中断整文件
            summary.n_records_bad_structure += 1
            if len(summary.structure_errors) < 5:
                summary.structure_errors.append(
                    f"line {line_no} ({record.get('rollout_id')}): {error}"
                )
            continue

        for rd in replayed:
            summary.n_decisions += 1
            summary.prompt_sha_match += int(rd.prompt_sha_ok)
            summary.prompt_len_match += int(rd.prompt_len_ok)
            summary.prompt_msgs_sha_match += int(rd.prompt_msgs_sha_ok)
            summary.raw_sha_match += int(rd.raw_sha_ok)
            summary.stats_match += int(rd.stats_ok)
            summary.completion_sha_match += int(rd.completion_sha_ok)
            summary.completion_len_match += int(rd.completion_len_ok)

            # 失败样本:留前 3 个,便于诊断(必要时把重建上下文 dump 到磁盘)
            if not (rd.prompt_sha_ok and rd.completion_sha_ok):
                if len(summary.mismatch_examples) < 3:
                    summary.mismatch_examples.append(
                        {
                            "line": line_no,
                            "rollout_id": record.get("rollout_id"),
                            "step_index": rd.step_index,
                            "expected_prompt_sha": rd.expected_prompt_sha,
                            "rebuilt_prompt_sha": rd.rebuilt_prompt_sha,
                            "rebuilt_prompt_tokens": len(rd.prompt_token_ids),
                            "recorded_prompt_tokens": record["decision_trace"][
                                rd.step_index
                            ].get("prompt_token_count"),
                            "prompt_msgs_sha_ok": rd.prompt_msgs_sha_ok,
                            "completion_sha_ok": rd.completion_sha_ok,
                        }
                    )
                    if dump_dir is not None:
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        dump_path = dump_dir / (
                            f"{path.stem}_line{line_no}_step{rd.step_index}.json"
                        )
                        dump_path.write_text(
                            json.dumps(
                                {
                                    "rollout_id": record.get("rollout_id"),
                                    "rebuilt_prompt_messages": rd.prompt_messages,
                                    "recorded_decision": record["decision_trace"][
                                        rd.step_index
                                    ],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

            # 分段标注(两种模式都做,verify 模式用来出 token 占比)
            labels, offsets_ok = label_completion_tokens(
                rd.completion_content, tokenizer, rd.completion_token_ids
            )
            summary.offsets_bad += int(not offsets_ok)
            for lab in labels:
                summary.seg_tokens[lab] += 1

            if verify_only:
                continue

            # ---------------- GPU 完整模式 ----------------
            logp, ent, cll = compute_decision_cll(
                model,
                lm_head,
                device,
                rd.prompt_token_ids,
                rd.completion_token_ids,
                logit_chunk,
            )
            lab_arr = np.asarray(labels)
            seg_means: Dict[str, Any] = {}
            for seg in SEGMENTS:
                mask = lab_arr == seg
                if mask.any():
                    summary.seg_values[seg]["logp"].append(
                        logp[mask].astype(np.float32)
                    )
                    summary.seg_values[seg]["H"].append(ent[mask].astype(np.float32))
                    summary.seg_values[seg]["cll"].append(
                        cll[mask].astype(np.float32)
                    )
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
                "prompt_tokens": len(rd.prompt_token_ids),
                "completion_tokens": len(rd.completion_token_ids),
                "prompt_sha_ok": rd.prompt_sha_ok,
                "completion_sha_ok": rd.completion_sha_ok,
                "segments_rle": rle_labels(labels),
                # per-token 压缩数组:float16 little-endian + base64
                "logp_f16_b64": f16_b64(logp),
                "entropy_f16_b64": f16_b64(ent),
                "cll_f16_b64": f16_b64(cll),
                "seg_means": seg_means,
            }
            out_handle.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        if summary.n_records % 10 == 0:
            print(
                f"[{path.name}] {summary.n_records} records, "
                f"{summary.n_decisions} decisions, "
                f"prompt_sha match {summary.prompt_sha_match}/{summary.n_decisions}",
                file=sys.stderr,
                flush=True,
            )
    return summary


def print_verify_table(summaries: List[FileSummary]) -> None:
    """verify-only 模式的汇总表(stdout)。"""
    header = (
        f"{'file':<44} {'recs':>5} {'decs':>6} {'prompt_sha':>11} "
        f"{'compl_sha':>10} {'think%':>7} {'action%':>8} {'other%':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        total_seg = sum(s.seg_tokens.values()) or 1
        print(
            f"{Path(s.path).name:<44} {s.n_records:>5} {s.n_decisions:>6} "
            f"{s.prompt_sha_match}/{s.n_decisions:>{max(1, len(str(s.n_decisions)))}}"
            f"{'':>2}"
            f" {s.completion_sha_match}/{s.n_decisions:>1}"
            f" {100.0 * s.seg_tokens['think'] / total_seg:>6.2f}%"
            f" {100.0 * s.seg_tokens['action'] / total_seg:>7.2f}%"
            f" {100.0 * s.seg_tokens['other'] / total_seg:>6.2f}%"
        )
    print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="教师轨迹 CLL 分段画像(重放采集器渲染 + sha 校验 + CLL)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="教师轨迹 jsonl(可多个;须同一环境,与 --config 匹配)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="采集时使用的学生 collect yaml(决定上下文渲染契约)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只做 sha 校验与分段统计(纯 CPU,不加载模型)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="完整模式必填:per-decision CLL jsonl 输出路径;"
        "verify 模式可选:写 JSON 汇总",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="每个输入文件最多处理的记录数(默认全量)",
    )
    parser.add_argument("--device", default="cuda", help="完整模式使用的设备")
    parser.add_argument(
        "--logit-chunk",
        type=int,
        default=512,
        help="logits 分块的位置数(块内物化 [chunk, vocab] float32)",
    )
    parser.add_argument(
        "--dump-mismatch",
        type=Path,
        default=None,
        help="可选目录:把前几个 sha 失配 decision 的重建上下文 dump 成 JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.verify_only and args.output is None:
        raise SystemExit("完整模式必须提供 --output")

    # --- 学生 config / 契约 / tokenizer:全部走采集器同一套函数 ---
    config = collector.compose_student_config(args.config)
    student_contract = collector.validate_student_contract(config)
    resolved_config_sha = collector._source_config_hash(config)
    tokenizer, tokenizer_manifest = collector.load_student_tokenizer(
        student_contract["student_model_path"]
    )
    print(
        f"config ok: env={student_contract['environment']} "
        f"resolved_sha={resolved_config_sha[:12]} "
        f"tokenizer={tokenizer_manifest['path']}",
        file=sys.stderr,
    )
    # 与 Linear_CMT.__init__(cmt_linear.py L59-61)相同的构造方式
    policy = StructuredContextPolicy(tokenizer, config.actor_rollout_ref.rollout)

    model = None
    lm_head = None
    out_handle = None
    if not args.verify_only:
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

    summaries: List[FileSummary] = []
    try:
        for path in args.input:
            path = path.resolve()
            print(f"processing {path} ...", file=sys.stderr)
            summary = process_file(
                path,
                policy=policy,
                tokenizer=tokenizer,
                resolved_config_sha=resolved_config_sha,
                max_records=args.max_records,
                verify_only=args.verify_only,
                model=model,
                lm_head=lm_head,
                device=args.device,
                logit_chunk=args.logit_chunk,
                out_handle=out_handle,
                dump_dir=args.dump_mismatch,
            )
            summaries.append(summary)
    finally:
        if out_handle is not None:
            out_handle.close()

    # --- 输出汇总 ---
    payload = {
        "config": str(args.config),
        "resolved_student_config_sha256": resolved_config_sha,
        "verify_only": bool(args.verify_only),
        "max_records": args.max_records,
        "files": [s.as_dict(args.verify_only) for s in summaries],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print_verify_table(summaries)

    if args.verify_only and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.verify_only:
        summary_path = Path(str(args.output) + ".summary.json")
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # 任何 sha 失配都以非零退出码提示(便于上层脚本检查)
    all_ok = all(
        s.prompt_sha_match == s.n_decisions
        and s.completion_sha_match == s.n_decisions
        and s.n_records_bad_structure == 0
        for s in summaries
    )
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
