"""v6 近窗渲染域(keep-recent-m think + tool_response 包裹)契约测试。

设计:docs/research/CATALYST_v6_设计_思考保全与峰值锚定_2026-08-21.md §⓪
四条硬契约:
  1. strip 域(旧配置)行为回归:渲染结果不含任何历史 think、无包裹
     ——现网 v4/GRPO 配置在新代码下逐字节不变;
  2. 近窗域:恰好最近 m 个 llm 轮的 think 可见,更老轮渲染为空 think 块;
  3. 逐轮帽:超帽 think 被截且带截断标记,action 段不受影响;
  4. 模板窗界:观察包裹必须在裁剪之后(闭合标签永不被裁掉),任务陈述
     保持非 tool_response(否则模板 think 保留窗口整体重置)。
"""

import os
import re
import types

import pytest

from agentevolver.module.context_manager.context_policy import (
    StructuredContextPolicy,
)

_TOKENIZER_CANDIDATES = [
    os.environ.get("STABLE_TRUNC_TOKENIZER", ""),
    "/projects_vol/gp_wangwy/models/Qwen3.5-4B",
    "/data/shared_models/Qwen3.5-4B-think",
]


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    for path in _TOKENIZER_CANDIDATES:
        if path and os.path.isdir(path):
            return AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, local_files_only=True
            )
    pytest.skip("Qwen3.5 tokenizer not found on this host")


class _Msg:
    def __init__(self, author, role, content):
        self.author = author
        self.role = role
        self.content = content


def _log(n_turns=6):
    msgs = [
        _Msg("initialization", "system", "You are an agent."),
        _Msg(
            "initialization",
            "user",
            "TASK: put a pencil on the desk. You are in a room.",
        ),
    ]
    for i in range(n_turns):
        msgs.append(
            _Msg(
                "llm",
                "assistant",
                f"<think>\nreasoning turn {i}: check drawer {i}.\n</think>"
                f"\n\n<action>\ngo to drawer {i}\n</action>",
            )
        )
        msgs.append(_Msg("env", "user", f"On drawer {i} you see nothing."))
    return msgs


def _cfg(**over):
    base = dict(
        enabled=True,
        max_prompt_tokens=8000,
        recent_turns=3,
        min_recent_turns=1,
        history_observation_max_tokens=256,
        reasoning_history_tokens=0,
    )
    base.update(over)
    return types.SimpleNamespace(
        context_management=types.SimpleNamespace(**base),
        max_model_len=32768,
        response_length=10240,
    )


def test_strip_regime_regression(qwen_tokenizer):
    policy = StructuredContextPolicy(qwen_tokenizer, _cfg())
    text = qwen_tokenizer.decode(policy.build(_log()).prompt_token_ids)
    assert "reasoning turn" not in text
    assert "<tool_response>" not in text


def test_recent_window_keeps_last_m(qwen_tokenizer):
    policy = StructuredContextPolicy(
        qwen_tokenizer,
        _cfg(
            observation_tool_response=True,
            reasoning_recent_turns=3,
            reasoning_max_tokens_per_turn=512,
        ),
    )
    text = qwen_tokenizer.decode(policy.build(_log()).prompt_token_ids)
    kept = [i for i in range(6) if f"reasoning turn {i}" in text]
    assert kept == [3, 4, 5]
    assert text.count("<tool_response>") >= 6
    assert len(re.findall(r"<think>\s*</think>", text)) >= 3
    # 任务陈述必须保持非 tool_response,否则模板窗界重置
    assert "TASK:" in text
    task_zone = text[text.find("TASK:") - 60 : text.find("TASK:")]
    assert "<tool_response>" not in task_zone


def test_per_turn_reasoning_cap(qwen_tokenizer):
    msgs = _log(4)
    msgs[2] = _Msg(
        "llm",
        "assistant",
        "<think>\n" + "very long reasoning. " * 400
        + "\n</think>\n\n<action>\ngo to drawer 0\n</action>",
    )
    policy = StructuredContextPolicy(
        qwen_tokenizer,
        _cfg(
            observation_tool_response=True,
            reasoning_recent_turns=4,
            reasoning_max_tokens_per_turn=64,
        ),
    )
    text = qwen_tokenizer.decode(policy.build(msgs).prompt_token_ids)
    assert "[context clipped]" in text
    assert "go to drawer 0" in text  # action 段不受帽影响


def test_wrap_survives_observation_clipping(qwen_tokenizer):
    # 老轮观察超 history_observation_max_tokens 被裁时,闭合标签必须完好
    msgs = _log(6)
    msgs[3] = _Msg("env", "user", "On drawer 0 you see: " + "item, " * 400)
    policy = StructuredContextPolicy(
        qwen_tokenizer,
        _cfg(
            observation_tool_response=True,
            reasoning_recent_turns=3,
            history_observation_max_tokens=64,
        ),
    )
    result = policy.build(msgs)
    for message in result.messages:
        content = message["content"]
        if content.startswith("<tool_response>"):
            assert content.endswith("</tool_response>")
