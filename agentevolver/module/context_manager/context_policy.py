"""Deterministic context construction for long-horizon agent rollouts.

The important boundary in this module is immutability: a context policy may
decide what the *next* model call sees, but it must never rewrite the condition
under which an already-sampled action is trained.  ``DecisionSnapshot`` stores
that condition at sampling time; the actor/ref path consumes the snapshot
instead of rebuilding a post-hoc transcript at the end of the episode.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from agentevolver.module.context_manager.cmt_base import chat_template_ids


class ContextBudgetError(RuntimeError):
    """Raised when even the protected context cannot fit the configured budget."""


@dataclass(frozen=True)
class DecisionSnapshot:
    """The exact policy condition and sampled tokens for one environment step."""

    step_index: int
    prompt_messages: List[Dict[str, str]]
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_log_probs: List[float] = field(default_factory=list)
    prompt_hash: str = ""
    raw_prompt_hash: str = ""
    template_hash: str = ""
    assistant_content: str = ""
    finish_reason: str | None = None
    truncated_by_length: bool = False
    stop_reason: str | None = None
    context_stats: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBuildResult:
    messages: List[Dict[str, str]]
    prompt_token_ids: List[int]
    stats: Dict[str, int]
    raw_prompt_hash: str = ""


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


class StructuredContextPolicy:
    """Pure, token-budgeted context builder.

    The policy keeps initialization messages and a configurable number of
    recent interaction turns. Older turns are represented deterministically as
    action + compact observation pairs and are dropped oldest-first only when
    the hard prompt-token budget requires it. No LLM summarizer is involved, so
    the same raw event log always produces the same prompt.
    """

    _HINT_PREFIXES = (
        "available actions:",
        "obj candidates:",
        "obj must be replaced",
        "you can use:",
        "clickable elements:",
    )

    def __init__(self, tokenizer: Any, rollout_config: Any):
        self.tokenizer = tokenizer
        cm = _cfg_get(rollout_config, "context_management", None)
        self.enabled = bool(_cfg_get(cm, "enabled", False))
        default_budget = int(_cfg_get(rollout_config, "max_model_len", 0)) - int(
            _cfg_get(rollout_config, "response_length", 0)
        )
        self.max_prompt_tokens = int(
            _cfg_get(cm, "max_prompt_tokens", default_budget)
        )
        self.recent_turns = max(0, int(_cfg_get(cm, "recent_turns", 3)))
        self.min_recent_turns = max(
            0, int(_cfg_get(cm, "min_recent_turns", 1))
        )
        self.history_observation_max_tokens = int(
            _cfg_get(cm, "history_observation_max_tokens", 256)
        )
        self.recent_observation_max_tokens = int(
            _cfg_get(
                cm,
                "recent_observation_max_tokens",
                -1,
            )
        )
        # Current observations contain the legal action set/clickables in both
        # ALFWorld and WebShop.  Cropping one merely to satisfy a budget silently
        # changes the task, so it is opt-in and disabled for the correctness
        # baseline.
        self.allow_current_observation_truncation = bool(
            _cfg_get(cm, "allow_current_observation_truncation", False)
        )
        self.reasoning_history_tokens = int(
            _cfg_get(cm, "reasoning_history_tokens", 0)
        )
        # v6 近窗域(2026-08-21 定稿):
        # - observation_tool_response:环境观察在渲染层包
        #   <tool_response>…</tool_response>。Qwen3.5 模板以"最后一条非
        #   tool_response 的 user 消息"为界,其后 assistant 轮的 think
        #   原生保留;不包则每条观察都是"新的用户提问",历史 think 全剥
        #   (即 strip 域,旧行为)。
        # - reasoning_recent_turns:近窗 m——最近 m 个 llm 轮保留 think,
        #   更老轮 action-only(模板渲染为空 think 块)。0 = 关闭(退回
        #   reasoning_history_tokens 语义)。
        # - reasoning_max_tokens_per_turn:窗内逐轮 think 硬帽(outlier
        #   保险丝;教师 122B think 实测 p99=311,>512 仅 0.4%)。
        self.reasoning_recent_turns = max(
            0, int(_cfg_get(cm, "reasoning_recent_turns", 0))
        )
        self.reasoning_max_tokens_per_turn = int(
            _cfg_get(cm, "reasoning_max_tokens_per_turn", 512)
        )
        self.observation_tool_response = bool(
            _cfg_get(cm, "observation_tool_response", False)
        )
        self.snapshot_training = bool(
            _cfg_get(cm, "snapshot_training", self.enabled)
        )
        self.snapshot_selection = str(
            _cfg_get(cm, "snapshot_selection", "last")
        ).lower()
        self.snapshot_selection_seed = int(
            _cfg_get(cm, "snapshot_selection_seed", 0)
        )
        if self.enabled and self.max_prompt_tokens <= 0:
            raise ValueError(
                "context_management.max_prompt_tokens must be positive"
            )
        if self.min_recent_turns > self.recent_turns:
            raise ValueError(
                "context_management.min_recent_turns cannot exceed recent_turns"
            )
        if self.snapshot_selection not in {"first", "last", "token_weighted"}:
            raise ValueError(
                "context_management.snapshot_selection must be one of "
                "first, last, token_weighted"
            )

    def _prompt_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        return chat_template_ids(
            self.tokenizer, messages, add_generation_prompt=True
        )

    @staticmethod
    def ids_hash(token_ids: Sequence[int]) -> str:
        payload = ",".join(str(int(token_id)) for token_id in token_ids)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @property
    def template_hash(self) -> str:
        template = str(getattr(self.tokenizer, "chat_template", "") or "")
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def _encode_text(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _clip_text(self, text: str, limit: int) -> tuple[str, bool]:
        if limit is None or limit < 0:
            return text, False
        ids = self._encode_text(text)
        if len(ids) <= limit:
            return text, False
        if limit <= 0:
            return "", True

        marker = "\n...[context clipped]...\n"
        marker_ids = self._encode_text(marker)
        if limit <= len(marker_ids) + 8:
            return self.tokenizer.decode(ids[:limit]), True
        available = limit - len(marker_ids)
        head_n = max(1, int(available * 0.65))
        tail_n = max(1, available - head_n)
        clipped = (
            self.tokenizer.decode(ids[:head_n])
            + marker
            + self.tokenizer.decode(ids[-tail_n:])
        )
        return clipped, True

    @classmethod
    def _strip_action_hints(cls, content: str) -> str:
        kept: List[str] = []
        for line in str(content).split("\n"):
            if line.strip().lower().startswith(cls._HINT_PREFIXES):
                break
            kept.append(line)
        result = "\n".join(kept).strip()
        return result if result else str(content)

    @staticmethod
    def _post_think(content: str) -> str:
        if "</think>" not in content:
            return content
        return content.split("</think>")[-1].lstrip("\n").strip()

    @classmethod
    def _action_only(cls, content: str) -> str:
        post = cls._post_think(str(content))
        matches = list(
            re.finditer(
                r"<action>(.*?)</action>", post, flags=re.IGNORECASE | re.DOTALL
            )
        )
        if matches:
            action = matches[-1].group(1).strip()
            if action:
                return f"<action>\n{action}\n</action>"

        # Backward-compatible ReAct form.
        lines = post.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.lower().startswith("action:"):
                continue
            value = stripped[len("action:") :].strip()
            if not value:
                value = next(
                    (candidate.strip() for candidate in lines[i + 1 :] if candidate.strip()),
                    "(empty)",
                )
            return f"Action: {value}"
        return post.strip()

    @staticmethod
    def _as_message(ext_msg: Any, content: str | None = None) -> Dict[str, str]:
        if content is None:
            content = str(getattr(ext_msg, "content", ""))
        return {"role": str(ext_msg.role), "content": str(content)}

    @staticmethod
    def _split_prefix_and_turns(ext_messages: Sequence[Any]) -> tuple[List[Any], List[List[Any]]]:
        prefix: List[Any] = []
        tail: List[Any] = []
        seen_policy_turn = False
        for msg in ext_messages:
            if not seen_policy_turn and getattr(msg, "author", None) == "initialization":
                prefix.append(msg)
            else:
                seen_policy_turn = True
                tail.append(msg)

        turns: List[List[Any]] = []
        current: List[Any] = []
        for msg in tail:
            if getattr(msg, "author", None) == "llm":
                if current:
                    turns.append(current)
                current = [msg]
            elif current:
                current.append(msg)
            else:
                # Defensive support for a user/env message without a preceding
                # assistant turn. Keep it as an indivisible unit.
                turns.append([msg])
        if current:
            turns.append(current)
        return prefix, turns

    def _reasoning_keep_indices(self, turns: List[List[Any]]) -> set[int]:
        if self.reasoning_recent_turns > 0:
            llm_indices = [
                i
                for i, turn in enumerate(turns)
                if turn and getattr(turn[0], "author", None) == "llm"
            ]
            return set(llm_indices[-self.reasoning_recent_turns:])
        if self.reasoning_history_tokens < 0:
            return {
                i
                for i, turn in enumerate(turns)
                if turn and getattr(turn[0], "author", None) == "llm"
            }
        if self.reasoning_history_tokens == 0:
            return set()

        used = 0
        keep: set[int] = set()
        for i in reversed(range(len(turns))):
            turn = turns[i]
            if not turn or getattr(turn[0], "author", None) != "llm":
                continue
            content = str(getattr(turn[0], "content", ""))
            if "</think>" not in content:
                continue
            think = content.split("</think>")[0]
            n_tokens = len(self._encode_text(think))
            if used + n_tokens > self.reasoning_history_tokens:
                continue
            used += n_tokens
            keep.add(i)
        return keep

    def _clip_reasoning(self, content: str) -> str:
        """窗内 llm 轮的逐轮 think 硬帽(只截 think 段,action 段不动)。"""
        cap = self.reasoning_max_tokens_per_turn
        if cap < 0 or "</think>" not in content:
            return content
        head, _, _ = content.partition("</think>")
        think = head.split("<think>")[-1].strip("\n")
        clipped, was_clipped = self._clip_text(think, cap)
        if not was_clipped:
            return content
        post = self._post_think(content)
        return f"<think>\n{clipped}\n</think>\n\n{post}"

    def _wrap_tool_response(self, content: str) -> str:
        return f"<tool_response>\n{content}\n</tool_response>"

    def _render_turn(
        self, turn: List[Any], *, old: bool, keep_reasoning: bool
    ) -> tuple[List[Dict[str, str]], int]:
        rendered: List[Dict[str, str]] = []
        n_clipped = 0
        for msg in turn:
            author = getattr(msg, "author", None)
            raw = str(getattr(msg, "content", ""))
            if author == "llm":
                content = (
                    self._clip_reasoning(raw)
                    if keep_reasoning
                    else self._action_only(raw)
                )
            elif author == "env":
                content = self._strip_action_hints(raw) if old else raw
                limit = (
                    self.history_observation_max_tokens
                    if old
                    else (
                        self.recent_observation_max_tokens
                        if self.allow_current_observation_truncation
                        else -1
                    )
                )
                content, clipped = self._clip_text(content, limit)
                n_clipped += int(clipped)
                # 包裹必须在裁剪之后:闭合标签被裁掉会使该观察被模板
                # 误判为真实用户提问,think 保留窗口整体重置。
                if self.observation_tool_response:
                    content = self._wrap_tool_response(content)
            else:
                content = raw
            rendered.append(self._as_message(msg, content))
        return rendered, n_clipped

    def build(self, ext_messages: Sequence[Any]) -> ContextBuildResult:
        if not ext_messages:
            raise ContextBudgetError("cannot build context from an empty event log")

        if not self.enabled:
            messages = [self._as_message(msg) for msg in ext_messages]
            ids = self._prompt_ids(messages)
            return ContextBuildResult(
                messages=messages,
                prompt_token_ids=ids,
                stats={
                    "raw_prompt_tokens": len(ids),
                    "managed_prompt_tokens": len(ids),
                    "compressed_turns": 0,
                    "dropped_turns": 0,
                    "clipped_observations": 0,
                },
                raw_prompt_hash=self.ids_hash(ids),
            )

        prefix, turns = self._split_prefix_and_turns(ext_messages)
        prefix_messages = [self._as_message(msg) for msg in prefix]
        raw_messages = prefix_messages + [
            self._as_message(msg) for turn in turns for msg in turn
        ]
        raw_prompt_ids = self._prompt_ids(raw_messages)
        raw_token_count = len(raw_prompt_ids)

        split = max(0, len(turns) - self.recent_turns)
        old_turns = turns[:split]
        recent_turns = turns[split:]
        keep_indices = self._reasoning_keep_indices(turns)

        old_units: List[List[Dict[str, str]]] = []
        recent_units: List[List[Dict[str, str]]] = []
        clipped_observations = 0
        for i, turn in enumerate(old_turns):
            unit, clipped = self._render_turn(
                turn, old=True, keep_reasoning=i in keep_indices
            )
            old_units.append(unit)
            clipped_observations += clipped
        for j, turn in enumerate(recent_turns, start=split):
            unit, clipped = self._render_turn(
                turn, old=False, keep_reasoning=j in keep_indices
            )
            recent_units.append(unit)
            clipped_observations += clipped

        def flatten() -> List[Dict[str, str]]:
            return copy.deepcopy(
                prefix_messages
                + [message for unit in old_units for message in unit]
                + [message for unit in recent_units for message in unit]
            )

        messages = flatten()
        prompt_ids = self._prompt_ids(messages)
        dropped_turns = 0

        # Remove the least useful, already-compressed history first.
        while len(prompt_ids) > self.max_prompt_tokens and old_units:
            old_units.pop(0)
            dropped_turns += 1
            messages = flatten()
            prompt_ids = self._prompt_ids(messages)

        # Preserve at least min_recent_turns, but a hard budget takes priority
        # over keeping additional recent turns.
        while (
            len(prompt_ids) > self.max_prompt_tokens
            and len(recent_units) > self.min_recent_turns
        ):
            recent_units.pop(0)
            dropped_turns += 1
            messages = flatten()
            prompt_ids = self._prompt_ids(messages)

        if self.allow_current_observation_truncation:
            # This path is reserved for a benchmark-validated renderer.  It is
            # deliberately opt-in: the default baseline fails rather than lose
            # current legal actions without an audit trail.
            for _ in range(8):
                if len(prompt_ids) <= self.max_prompt_tokens:
                    break
                user_index = next(
                    (
                        i
                        for i in reversed(range(len(messages)))
                        if messages[i]["role"] == "user"
                    ),
                    None,
                )
                if user_index is None:
                    break
                content = messages[user_index]["content"]
                # tool_response 包裹的观察:先解包再裁,裁完重包——
                # 裁掉闭合标签会重置模板的 think 保留窗口。
                wrapped = (
                    self.observation_tool_response
                    and content.startswith("<tool_response>")
                    and content.endswith("</tool_response>")
                )
                if wrapped:
                    content = content[len("<tool_response>"): -len("</tool_response>")].strip("\n")
                content_tokens = self._encode_text(content)
                over = len(prompt_ids) - self.max_prompt_tokens
                new_limit = max(32, len(content_tokens) - over - 32)
                if new_limit >= len(content_tokens):
                    break
                clipped_content, _ = self._clip_text(content, new_limit)
                messages[user_index]["content"] = (
                    self._wrap_tool_response(clipped_content)
                    if wrapped
                    else clipped_content
                )
                clipped_observations += 1
                prompt_ids = self._prompt_ids(messages)

        if len(prompt_ids) > self.max_prompt_tokens:
            raise ContextBudgetError(
                "protected context does not fit: "
                f"{len(prompt_ids)} > {self.max_prompt_tokens} prompt tokens"
            )

        stats = {
            "raw_prompt_tokens": raw_token_count,
            "managed_prompt_tokens": len(prompt_ids),
            "compressed_turns": len(turns) - len(recent_turns),
            "dropped_turns": dropped_turns,
            "clipped_observations": clipped_observations,
        }
        return ContextBuildResult(
            messages=messages,
            prompt_token_ids=prompt_ids,
            stats=stats,
            raw_prompt_hash=self.ids_hash(raw_prompt_ids),
        )
