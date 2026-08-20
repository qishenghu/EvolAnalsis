"""CATALYST 测试共用基建:字节级 FakeTokenizer、样本构造、环境管理器桩。

FakeTokenizer 是 thinking-chatml 风格(chat_template 含 <|im_start|> 与
</think>),token = UTF-8 字节值——确定性、免下载、纯 CPU;足以驱动
StructuredContextPolicy / chat_template_ids / parse_response_ids_to_steps。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

from agentevolver.schema.trajectory import Reward, Sample


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    # thinking-chatml 风格:step_parser/_is_thinking_chatml_template 走字面分支
    chat_template = "<|im_start|>fake</think>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(str(text).encode("utf-8"))

    def decode(self, ids, **kwargs) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        add_special_tokens: bool = False,
    ):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return self.encode(text) if tokenize else text


def make_rollout_cfg(
    *,
    enabled: bool = True,
    max_prompt_tokens: int = 8192,
    recent_turns: int = 2,
    snapshot_training: bool = True,
):
    return OmegaConf.create(
        {
            "max_model_len": max_prompt_tokens + 2048,
            "response_length": 2048,
            "context_management": {
                "enabled": enabled,
                "max_prompt_tokens": max_prompt_tokens,
                "recent_turns": recent_turns,
                "min_recent_turns": 1,
                "history_observation_max_tokens": 160,
                "recent_observation_max_tokens": -1,
                "allow_current_observation_truncation": False,
                "reasoning_history_tokens": 0,
                "snapshot_training": snapshot_training,
                "snapshot_selection": "token_weighted",
                "snapshot_selection_seed": 2025,
            },
        }
    )


def onpolicy_extras(response_len: int, *, expected_rollouts: int = 2) -> Dict[str, Any]:
    """快照式在线样本的 extras(与 get_extra 产出的关键键对齐)。"""
    return {
        "rollout_mode": "sample",
        "is_experience_replay": False,
        "is_teacher": False,
        "has_log_prob": False,
        "snapshot_training": True,
        "rollout_log_probs": [0.0] * response_len,
        "expected_group_rollouts": expected_rollouts,
        "old_log_probs": None,
    }


def replay_extras(task_id: str = "t0", inserted_step: int = 0) -> Dict[str, Any]:
    """CATALYST 重放样本的 extras 契约(规格 F4 四元组 + 归属字段)。"""
    return {
        "is_experience_replay": True,
        "is_catalyst_replay": True,
        "is_teacher": False,
        "has_log_prob": False,
        "snapshot_training": False,
        "rollout_log_probs": None,
        "rollout_mode": None,
        "old_log_probs": None,
        "catalyst_arm": "replay",
        "task_id": task_id,
        "catalyst_replay_inserted_step": inserted_step,
    }


def make_sample(
    *,
    data_id: str,
    task_id: str,
    rollout_id: str,
    prompt_text: str,
    response_text: str,
    extras: Dict[str, Any],
    tokenizer: Optional[FakeTokenizer] = None,
    max_prompt_len: int = 8192,
    max_response_len: int = 2048,
    reward: float = 1.0,
) -> Sample:
    tokenizer = tokenizer or FakeTokenizer()
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response_text)
    input_ids = prompt_ids + response_ids
    position_ids = list(range(len(input_ids)))
    sample = Sample(
        data_id=str(data_id),
        task_id=str(task_id),
        rollout_id=str(rollout_id),
        minor_index_id=0,
        messages=[{"role": "user", "content": prompt_text},
                  {"role": "assistant", "content": response_text}],
        messages_raw=[{"role": "user", "content": prompt_text},
                      {"role": "assistant", "content": response_text}],
        input_ids=input_ids,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        attention_mask=[1] * len(input_ids),
        prompt_attention_mask=[1] * len(prompt_ids),
        response_attention_mask=[1] * len(response_ids),
        loss_mask=[0] * len(prompt_ids) + [1] * len(response_ids),
        prompt_loss_mask=[0] * len(prompt_ids),
        response_loss_mask=[1] * len(response_ids),
        position_ids=position_ids,
        prompt_position_ids=position_ids[: len(prompt_ids)],
        response_position_ids=position_ids[len(prompt_ids):],
        reward_scores=Reward(outcome=reward, success_rate=reward).model_dump(),
        max_prompt_len=max_prompt_len,
        max_response_len=max_response_len,
        max_model_len=max_prompt_len + max_response_len,
    )
    extras = dict(extras)
    if extras.get("rollout_log_probs") is not None:
        extras["rollout_log_probs"] = [0.0] * len(response_ids)
    sample.extras = extras
    return sample


def make_env_manager(*, world_size: int = 1, rollout_n: int = 2):
    """object.__new__ 桩(沿用 tests/test_grpo_group_integrity.py 惯例)。"""
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager

    manager = object.__new__(ParallelEnvManager)
    manager.config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": world_size, "nnodes": 1},
            "algorithm": {"adv_estimator": "grpo"},
            "data": {"max_prompt_length": 8192, "max_response_length": 2048},
        }
    )
    manager.tokenizer = FakeTokenizer()
    manager.pad_token_id = 0
    manager.rollout_n = rollout_n
    return manager


def make_cmt_stub(samples: List[Sample], extras: Dict[str, Any]):
    """group_tokenize 返回预置样本;get_extra 读 .extras(测试内覆写)。"""
    return SimpleNamespace(
        group_tokenize=lambda: list(samples),
        extras=dict(extras),
    )


# ---------------------------------------------------------------------------
# T5/T9 共用:构造一条带 hint 的两 decision 成功 episode + 一致的快照
# ---------------------------------------------------------------------------
def build_hinted_episode(
    *,
    hint: str = "Check the sinkbasin first, then clean and place the soap.",
    task_id: str = "task9",
    rollout_id: str = "3",
    with_hint: bool = True,
):
    """返回 (cmt_stub, policy, tokenizer, rollout_cfg)。

    快照 prompt ids 用与训练同一条链(ext 重构 + policy.build)预先算出——
    与真实 rollout 的 _capture_decision_snapshot 语义一致(审计应通过)。
    """
    from agentevolver.module.context_manager.cmt_base import ExtendedMessage
    from agentevolver.module.context_manager.context_policy import (
        StructuredContextPolicy,
    )
    from agentevolver.module.exp_manager.catalyst import (
        inject_hint_into_init_messages,
    )

    tokenizer = FakeTokenizer()
    rollout_cfg = make_rollout_cfg()
    policy = StructuredContextPolicy(tokenizer, rollout_cfg)

    init_messages = [
        {"role": "system", "content": "You are an agent in a house."},
        {"role": "assistant", "content": "OK. I'll follow your instructions."},
        {"role": "user", "content": "Task: put a clean soapbar in cabinet."},
    ]
    if with_hint:
        init_messages = inject_hint_into_init_messages(init_messages, hint)

    a0 = "<think>\nFind the soapbar; sinkbasin is likely.\n</think>\n<action>\ngo to sinkbasin 1\n</action>"
    o0 = "You arrive at sinkbasin 1. On it you see a soapbar 1."
    a1 = "<think>\nTake it and clean it.\n</think>\n<action>\ntake soapbar 1 from sinkbasin 1\n</action>"

    full = (
        [
            SimpleNamespace(author="initialization", role=m["role"], content=m["content"])
            for m in init_messages
        ]
        + [
            SimpleNamespace(author="llm", role="assistant", content=a0),
            SimpleNamespace(author="env", role="user", content=o0),
            SimpleNamespace(author="llm", role="assistant", content=a1),
        ]
    )
    n_init = len(init_messages)

    def _ext_prefix(count: int):
        return [
            ExtendedMessage(
                author=m.author,
                role=m.role,
                content=m.content,
                token_arr=[],
                token_generator="manual",
                tokenizer=tokenizer,
            )
            for m in full[:count]
        ]

    snapshots = []
    for t, content in enumerate([a0, a1]):
        prompt_ids = list(policy.build(_ext_prefix(n_init + 2 * t)).prompt_token_ids)
        snapshots.append(
            SimpleNamespace(
                step_index=t,
                prompt_token_ids=prompt_ids,
                completion_token_ids=tokenizer.encode(content),
                assistant_content=content,
            )
        )

    cmt = SimpleNamespace(
        task_id=task_id,
        rollout_id=rollout_id,
        full_context=full,
        decision_snapshots=snapshots,
        reward=Reward(outcome=1.0, success_rate=1.0),
        metadata=(
            {"catalyst_arm": "hint", "catalyst_hint_sha256": "deadbeef"}
            if with_hint
            else {}
        ),
        discarded=False,
    )
    return cmt, policy, tokenizer, rollout_cfg
