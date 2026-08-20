#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DeepSearch closed-book 污染预检(参数记忆旁路 gate)。

立项 gate 背景(docs/design/DEEPSEARCH_ENV.md):DeepSearch 域要求任务
必须依赖检索才能答对。若 Qwen3.5-4B 在**禁用检索**的 closed-book 设定下
直接回答 val200 的问题就能拿到较高 EM(阈值:>10%),说明该子集存在
参数记忆旁路(模型预训练时背过答案),该域应降级/换任务子集。

做法:
  1. 用 vLLM 离线推理(`from vllm import LLM, SamplingParams`,不起服务、
     不连检索),对 val200 的每个问题用 chat 模板构造 closed-book prompt:
     system 明确"不允许检索,直接给最终答案,格式 answer[你的答案]",
     user 给问题;greedy 解码(temperature=0)。
  2. 解析 answer[...] 标签(先去掉 <think> 段;无标签时兜底取最后一个
     非空行),用环境同一套归一化(normalize_answer/exact_match/f1_score,
     直接从 env_service.environments.deepsearch.deepsearch_env import,
     保证与在线 reward 判定逐字节一致)计算严格 EM 与 F1。
  3. 输出总表 + 按 hop(3/4)分组的 EM/F1,写
     analysis_outputs/deepsearch_closedbook/结果.json 并打印摘要。

运行环境:vllm2 conda 环境的 python(vLLM 0.21.0),单卡即可:
  /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/vllm2/bin/python \\
      analysis/deepsearch_closedbook_eval.py \\
      [--model /projects_vol/gp_wangwy/models/Qwen3.5-4B] \\
      [--tasks data/deepsearch/tasks_train_pool.jsonl] \\
      [--ids data/deepsearch/task_ids_val200_seed2026.txt] \\
      [--max-samples 200] [--tp 1]

注意:重依赖(vllm/transformers)延迟到 main() 内 import——模块本身可被
py_compile / import 检查而不触碰 GPU。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 路径:保证能 import 仓库内的 env_service(EM/F1 归一化的唯一权威实现)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# EM/F1/归一化直接复用环境实现:与 DeepsearchEnv 在线 reward 完全同一套判定
from env_service.environments.deepsearch.deepsearch_env import (  # noqa: E402
    exact_match,
    f1_score,
    normalize_answer,
)

DEFAULT_MODEL = "/projects_vol/gp_wangwy/models/Qwen3.5-4B"
DEFAULT_TASKS = PROJECT_ROOT / "data" / "deepsearch" / "tasks_train_pool.jsonl"
DEFAULT_IDS = (
    PROJECT_ROOT / "data" / "deepsearch" / "task_ids_val200_seed2026.txt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "analysis_outputs" / "deepsearch_closedbook" / "结果.json"
)
# 立项 gate:closed-book EM 超过该阈值 → 判定存在参数记忆旁路,该域降级
EM_GATE_THRESHOLD = 0.10

# closed-book 指令:与环境 system prompt 的答案格式(answer[...]、短语而非
# 句子)保持一致,但明确禁止检索、要求直接作答。
SYSTEM_PROMPT = (
    "You are answering a multi-hop question directly from your own knowledge. "
    "You are NOT allowed to search, browse, or look anything up. "
    "Give your final answer in the exact format answer[your answer]. "
    "The final answer must be a short exact phrase (an entity, name, date or "
    "number), not a sentence."
)


def build_user_prompt(question: str) -> str:
    """user 消息:给问题 + 重申格式与禁检索约束。"""
    return (
        f"Question: {question}\n\n"
        "Answer directly from memory (no searching). "
        "Respond with answer[your final answer]."
    )


# ---------------------------------------------------------------------------
# 答案解析
# ---------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"answer\[(.*?)\]", re.IGNORECASE | re.DOTALL)


def extract_prediction(text: str) -> Tuple[str, bool]:
    """从模型输出提取预测答案。

    优先级:
      1. 去掉思考段(取最后一个 '</think>' 之后的部分);
      2. 在其中找 answer[...] 标签(取最后一个匹配);
      3. 找不到再在全文找一次(防止答案被留在 think 段内);
      4. 兜底:仍无标签时取 post-think 部分的最后一个非空行。
    返回 (预测文本, 是否命中 answer 标签)。
    """
    text = str(text or "")
    post_think = text.split("</think>")[-1]
    matches = _ANSWER_RE.findall(post_think)
    if not matches:
        matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip(), True
    lines = [line.strip() for line in post_think.splitlines() if line.strip()]
    if lines:
        return lines[-1], False
    return "", False


# ---------------------------------------------------------------------------
# 任务加载
# ---------------------------------------------------------------------------
def load_val_tasks(
    tasks_path: Path, ids_path: Path, max_samples: Optional[int]
) -> List[Dict[str, Any]]:
    """按 id 列表的顺序取 val 任务(id 文件顺序即 seed2026 抽样顺序)。"""
    id_order = [tok for tok in ids_path.read_text().split() if tok.strip()]
    if not id_order:
        raise RuntimeError(f"id 列表为空: {ids_path}")
    by_id: Dict[str, Dict[str, Any]] = {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_id[str(record["task_id"])] = record
    missing = [task_id for task_id in id_order if task_id not in by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} 个 val id 在任务文件中缺失(如 {missing[:3]});"
            f"tasks={tasks_path} ids={ids_path}"
        )
    tasks = [by_id[task_id] for task_id in id_order]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"--max-samples 必须为正: {max_samples}")
        tasks = tasks[: max_samples]
    return tasks


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def summarize(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """总表 + 按 hop 分组的 EM/F1/标签命中率。"""

    def _agg(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group)
        return {
            "n": n,
            "em": (sum(s["em"] for s in group) / n) if n else None,
            "f1": (sum(s["f1"] for s in group) / n) if n else None,
            "answer_tag_rate": (
                sum(1 for s in group if s["has_answer_tag"]) / n
            )
            if n
            else None,
        }

    by_hop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_hop[str(sample.get("hop"))].append(sample)
    return {
        "overall": _agg(samples),
        "by_hop": {hop: _agg(group) for hop, group in sorted(by_hop.items())},
    }


# ---------------------------------------------------------------------------
# CLI 与主流程
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSearch closed-book 污染预检(禁检索直接答 val200)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="学生模型路径")
    parser.add_argument(
        "--tasks", type=Path, default=DEFAULT_TASKS, help="任务 jsonl 文件"
    )
    parser.add_argument(
        "--ids", type=Path, default=DEFAULT_IDS, help="val 任务 id 列表文件"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="只评前 N 个 val 任务"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="tensor parallel 大小(默认单卡)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="每题生成上限(Qwen3.5 默认思考模式,需给 <think> 段留空间)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="结果 JSON 输出路径"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tasks = load_val_tasks(args.tasks.resolve(), args.ids.resolve(), args.max_samples)
    print(
        f"[closed-book] 模型={args.model} 任务数={len(tasks)} "
        f"tp={args.tp} greedy(temperature=0)",
        file=sys.stderr,
        flush=True,
    )

    # 重依赖延迟 import:py_compile / 模块 import 检查不触碰 GPU
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # chat 模板用模型自带的(Qwen3.5 默认 thinking-on),与训练/采集一致
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    prompts: List[str] = []
    for task in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task["question"])},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=int(args.tp),
        trust_remote_code=True,
    )
    # greedy:temperature=0(污染预检要的是确定性的"模型最相信的答案")
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(args.max_new_tokens),
    )
    outputs = llm.generate(prompts, sampling)  # vLLM 保证输出与输入同序

    samples: List[Dict[str, Any]] = []
    for task, output in zip(tasks, outputs):
        text = output.outputs[0].text if output.outputs else ""
        prediction, has_tag = extract_prediction(text)
        golds = list(task["golden_answers"])
        samples.append(
            {
                "task_id": task["task_id"],
                "hop": task.get("hop"),
                "question": task["question"],
                "golden_answers": golds,
                "prediction": prediction,
                "prediction_normalized": normalize_answer(prediction),
                "has_answer_tag": has_tag,
                # 严格 EM / token F1:与环境 reward 同一套实现
                "em": exact_match(prediction, golds),
                "f1": f1_score(prediction, golds),
                "finish_reason": str(
                    getattr(output.outputs[0], "finish_reason", "")
                )
                if output.outputs
                else "",
                "raw_output_tail": text[-600:],  # 只留尾部,便于人工抽查
            }
        )

    stats = summarize(samples)
    overall_em = float(stats["overall"]["em"] or 0.0)
    gate = {
        "threshold_em": EM_GATE_THRESHOLD,
        "closedbook_em": overall_em,
        "bypass_detected": overall_em > EM_GATE_THRESHOLD,
        "verdict": (
            "FAIL:closed-book EM 超阈值,存在参数记忆旁路,建议该域降级/换任务子集"
            if overall_em > EM_GATE_THRESHOLD
            else "PASS:closed-book EM 低于阈值,未见明显参数记忆旁路"
        ),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "tasks_file": str(args.tasks.resolve()),
        "ids_file": str(args.ids.resolve()),
        "n_samples": len(samples),
        "generation": {
            "mode": "vllm_offline",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": int(args.max_new_tokens),
            "tensor_parallel_size": int(args.tp),
            "system_prompt": SYSTEM_PROMPT,
        },
        "gate": gate,
        "overall": stats["overall"],
        "by_hop": stats["by_hop"],
        "samples": samples,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 打印摘要(总表 + 按 hop 分组)
    print("\n===== DeepSearch closed-book 污染预检 =====")
    print(f"模型          : {args.model}")
    print(f"样本数        : {len(samples)}")
    ov = stats["overall"]
    print(
        f"总体          : EM={ov['em']:.4f}  F1={ov['f1']:.4f}  "
        f"answer标签率={ov['answer_tag_rate']:.4f}"
    )
    for hop, agg in stats["by_hop"].items():
        print(
            f"hop={hop:<4}(n={agg['n']:>3}): EM={agg['em']:.4f}  "
            f"F1={agg['f1']:.4f}"
        )
    print(f"gate({EM_GATE_THRESHOLD:.0%}) : {gate['verdict']}")
    print(f"结果文件      : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
