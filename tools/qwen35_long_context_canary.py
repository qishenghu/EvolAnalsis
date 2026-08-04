#!/usr/bin/env python3
"""Deterministic Qwen3.5 long-context management canary.

This is a CPU-only integration check.  It reads the production snapshot-gate
configuration, creates a deliberately over-budget multi-turn event log, and
passes that log through ``Linear_CMT`` with the real Qwen3.5 tokenizer.  The
canary fails closed unless all of the following contracts hold:

* 22,528 prompt tokens + 10,240 response tokens == 32,768 model tokens;
* the raw prompt is over budget and management really clips old observations
  and evicts complete oldest-first turns;
* initialization, every configured recent turn, and the current observation
  survive exactly (historical reasoning is intentionally action-only);
* the managed prompt is no longer than 22,528 tokens and is deterministic;
* the decision snapshot contains the exact messages and token IDs sent to the
  model for the sampled action.

No model weights, GPU, rollout server, or environment service are required.

Examples::

    /data/home/qisheng/miniconda3/envs/duet2/bin/python \
      tools/qwen35_long_context_canary.py --profile all --pretty

    python tools/qwen35_long_context_canary.py --profile alfworld \
      --model /data/shared_models/Qwen3.5-4B-think
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Direct ``python tools/...`` execution puts only ``tools`` on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agentevolver.module.context_manager.cmt_base import (
    ExtendedMessage,
    chat_template_ids,
)
from agentevolver.module.context_manager.cmt_linear import Linear_CMT


DEFAULT_MODEL_DIR = Path("/data/shared_models/Qwen3.5-4B-think")
EXPECTED_MAX_PROMPT_TOKENS = 22_528
EXPECTED_RESPONSE_LENGTH = 10_240
EXPECTED_MAX_MODEL_LEN = 32_768
CLIP_MARKER = "...[context clipped]..."


class CanaryContractError(RuntimeError):
    """Raised when the production long-context contract is not satisfied."""


@dataclass(frozen=True)
class ProfileSpec:
    """A production config and enough synthetic turns to force eviction."""

    name: str
    config_path: Path
    turn_count: int


PROFILE_SPECS = {
    "alfworld": ProfileSpec(
        name="alfworld",
        config_path=REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld"
        / "alfworld_qwen35_4b_grpo_snapshot_gate.yaml",
        # 188 old observations at the configured 160-token cap exceed 22.5K.
        turn_count=190,
    ),
    "webshop": ProfileSpec(
        name="webshop",
        config_path=REPO_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/webshop"
        / "webshop_qwen35_4b_grpo_snapshot_gate.yaml",
        # 68 old observations at the configured 512-token cap exceed 22.5K.
        turn_count=72,
    ),
}


@dataclass(frozen=True)
class ProfileContract:
    """Resolved context fields used by the production rollout."""

    name: str
    config_path: str
    max_prompt_tokens: int
    response_length: int
    max_model_len: int
    max_env_len: int
    recent_turns: int
    min_recent_turns: int
    history_observation_max_tokens: int
    recent_observation_max_tokens: int
    allow_current_observation_truncation: bool
    reasoning_history_tokens: int
    snapshot_training: bool
    snapshot_selection: str
    snapshot_selection_seed: int
    env_type: str
    turn_count: int


@dataclass(frozen=True)
class CanaryReport:
    """Auditable evidence emitted after every invariant has passed."""

    profile: str
    config_path: str
    model_path: str
    max_prompt_tokens: int
    response_length: int
    max_model_len: int
    recent_turns: int
    min_recent_turns: int
    history_observation_max_tokens: int
    turn_count: int
    raw_prompt_tokens: int
    managed_prompt_tokens: int
    compressed_turns: int
    dropped_turns: int
    clipped_observations: int
    retained_turns: int
    first_retained_turn: int
    retained_clipped_history_observations: int
    snapshot_prompt_tokens: int
    snapshot_prompt_hash: str
    deterministic: bool
    whole_turn_eviction_verified: bool
    initialization_protected: bool
    recent_turns_protected: bool
    current_observation_protected: bool
    snapshot_exact: bool


@dataclass(frozen=True)
class _CompletionToken:
    token_id: int
    logprob: float = -0.125


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryContractError(message)


def _read_int(node: Any, key: str) -> int:
    value = getattr(node, key, None)
    _require(value is not None, f"production config is missing {key}")
    return int(value)


def load_profile_contract(profile: str) -> ProfileContract:
    """Read, resolve, and validate one production snapshot-gate profile."""

    _require(profile in PROFILE_SPECS, f"unknown canary profile: {profile}")
    spec = PROFILE_SPECS[profile]
    _require(spec.config_path.is_file(), f"config does not exist: {spec.config_path}")
    config = OmegaConf.load(spec.config_path)
    rollout = config.actor_rollout_ref.rollout
    context = rollout.context_management

    contract = ProfileContract(
        name=profile,
        config_path=str(spec.config_path),
        max_prompt_tokens=_read_int(context, "max_prompt_tokens"),
        response_length=_read_int(rollout, "response_length"),
        max_model_len=_read_int(rollout, "max_model_len"),
        max_env_len=_read_int(rollout, "max_env_len"),
        recent_turns=_read_int(context, "recent_turns"),
        min_recent_turns=_read_int(context, "min_recent_turns"),
        history_observation_max_tokens=_read_int(
            context, "history_observation_max_tokens"
        ),
        recent_observation_max_tokens=_read_int(
            context, "recent_observation_max_tokens"
        ),
        allow_current_observation_truncation=bool(
            context.allow_current_observation_truncation
        ),
        reasoning_history_tokens=_read_int(context, "reasoning_history_tokens"),
        snapshot_training=bool(context.snapshot_training),
        snapshot_selection=str(context.snapshot_selection),
        snapshot_selection_seed=_read_int(context, "snapshot_selection_seed"),
        env_type=str(config.env_service.env_type),
        turn_count=spec.turn_count,
    )

    _require(
        contract.max_prompt_tokens == EXPECTED_MAX_PROMPT_TOKENS,
        "canary requires production max_prompt_tokens=22528; got "
        f"{contract.max_prompt_tokens}",
    )
    _require(
        contract.response_length == EXPECTED_RESPONSE_LENGTH,
        "canary requires production response_length=10240; got "
        f"{contract.response_length}",
    )
    _require(
        contract.max_model_len == EXPECTED_MAX_MODEL_LEN,
        "canary requires production max_model_len=32768; got "
        f"{contract.max_model_len}",
    )
    _require(
        contract.max_prompt_tokens + contract.response_length
        == contract.max_model_len,
        "prompt and response budgets do not exactly fill max_model_len",
    )
    _require(contract.recent_turns > 0, "recent_turns must be positive")
    _require(
        0 < contract.min_recent_turns <= contract.recent_turns,
        "min_recent_turns must be in [1, recent_turns]",
    )
    _require(
        contract.history_observation_max_tokens > 0,
        "history observation cap must be positive",
    )
    _require(
        contract.recent_observation_max_tokens == -1,
        "canary requires lossless recent observations (-1 cap)",
    )
    _require(
        not contract.allow_current_observation_truncation,
        "canary requires protected current observations",
    )
    _require(
        contract.reasoning_history_tokens == 0,
        "canary assumes production action-only historical reasoning",
    )
    _require(contract.snapshot_training, "snapshot_training must be enabled")
    _require(
        contract.turn_count > contract.recent_turns,
        "synthetic event log needs both old and recent turns",
    )
    return contract


def _isolated_cmt_config(contract: ProfileContract):
    """Build the minimum resolved config consumed by ``Linear_CMT``."""

    context_management = {
        "enabled": True,
        "max_prompt_tokens": contract.max_prompt_tokens,
        "recent_turns": contract.recent_turns,
        "min_recent_turns": contract.min_recent_turns,
        "history_observation_max_tokens": (
            contract.history_observation_max_tokens
        ),
        "recent_observation_max_tokens": contract.recent_observation_max_tokens,
        "allow_current_observation_truncation": (
            contract.allow_current_observation_truncation
        ),
        "reasoning_history_tokens": contract.reasoning_history_tokens,
        "snapshot_training": contract.snapshot_training,
        "snapshot_selection": contract.snapshot_selection,
        "snapshot_selection_seed": contract.snapshot_selection_seed,
    }
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": contract.response_length,
                    "max_model_len": contract.max_model_len,
                    "max_env_len": contract.max_env_len,
                    "sliding_window_size": -1,
                    "reasoning_context_budget": 0,
                    "context_management": context_management,
                }
            },
            "data": {
                "max_prompt_length": contract.max_prompt_tokens,
                "max_response_length": contract.response_length,
            },
            "env_service": {
                "env_type": contract.env_type,
                "env_params": {"action_format": "react_tags"},
            },
            "exp_manager": {
                "experience_template": "Here are related experiences: {}"
            },
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
        }
    )


def _assistant_content(turn: int) -> str:
    return (
        f"private reasoning for deterministic turn {turn}\n"
        "</think>\n\n"
        f"<action>\nCANARY_ACTION_{turn}\n</action>"
    )


def _old_observation(turn: int) -> str:
    # About 600 real Qwen3.5 tokens: above both production history caps, but
    # small enough that the complete raw canary stays below tokenizer.model_max.
    body = " ".join(["historical detail"] * 300)
    return (
        f"CANARY_OBS_{turn}\n{body}\n"
        "AVAILABLE ACTIONS: look, search, click, buy"
    )


def _recent_observation(turn: int) -> str:
    # Every byte is checked after context construction.  In particular, the
    # legal-action suffix proves that the current/recent environment state was
    # not passed through the old-observation hint stripper.
    body = " ".join(["live state"] * 180)
    return (
        f"CANARY_OBS_{turn}\n{body}\n"
        f"AVAILABLE ACTIONS: exact-live-action-{turn}, inspect, finish"
    )


def _event(tokenizer: Any, *, author: str, role: str, content: str) -> ExtendedMessage:
    # ContextPolicy reads immutable role/author/content fields.  Avoiding
    # per-message token_arr construction keeps this 150K-token canary quick;
    # the complete model-facing prompt is still rendered with the real chat
    # template below and again inside snapshot capture.
    return ExtendedMessage(
        author=author,
        role=role,
        content=content,
        token_arr=[],
        tokenizer=tokenizer,
        token_generator="manual",
    )


def _extract_pair_indices(messages: Iterable[dict[str, str]]) -> list[int]:
    """Validate assistant/user pairing and return the exact retained turn IDs."""

    tail = list(messages)
    _require(len(tail) % 2 == 0, "managed history contains a partial turn")
    indices: list[int] = []
    action_pattern = re.compile(
        r"^<action>\nCANARY_ACTION_(\d+)\n</action>$"
    )
    observation_pattern = re.compile(r"^CANARY_OBS_(\d+)(?:\n|$)")
    for offset in range(0, len(tail), 2):
        assistant, observation = tail[offset : offset + 2]
        _require(
            assistant.get("role") == "assistant"
            and observation.get("role") == "user",
            f"managed turn {offset // 2} is not an assistant/user pair",
        )
        action_match = action_pattern.fullmatch(str(assistant.get("content", "")))
        observation_match = observation_pattern.match(
            str(observation.get("content", ""))
        )
        _require(action_match is not None, "managed assistant action is malformed")
        _require(
            observation_match is not None, "managed observation sentinel is missing"
        )
        action_index = int(action_match.group(1))
        observation_index = int(observation_match.group(1))
        _require(
            action_index == observation_index,
            "assistant and observation from different turns were spliced together",
        )
        indices.append(action_index)
    return indices


def _sample_output(tokenizer: Any) -> dict[str, Any]:
    content = (
        "final canary reasoning\n</think>\n\n"
        "<action>\nCANARY_FINAL_ACTION\n</action>"
    )
    token_ids = list(tokenizer.encode(content, add_special_tokens=False))
    token_ids.append(int(tokenizer.eos_token_id))
    return {
        "role": "assistant",
        "content": content,
        "sampled_content": content,
        "tokens": [_CompletionToken(token_id) for token_id in token_ids],
        "finish_reason": "stop",
        "stop_reason": "stop",
    }


def run_canary(
    tokenizer: Any,
    *,
    profile: str,
    model_path: str | Path = DEFAULT_MODEL_DIR,
) -> CanaryReport:
    """Run one production-profile canary or raise ``CanaryContractError``."""

    contract = load_profile_contract(profile)
    cmt = Linear_CMT(_isolated_cmt_config(contract), tokenizer)
    _require(
        cmt.max_seq_length == contract.max_prompt_tokens,
        "Linear_CMT prompt capacity differs from the production prompt budget",
    )

    system_content = f"CANARY_SYSTEM_INIT::{profile}::never-drop"
    initial_observation = f"CANARY_INITIAL_OBSERVATION::{profile}::never-drop"
    cmt.save_init_input(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": initial_observation},
        ]
    )

    recent_start = contract.turn_count - contract.recent_turns
    raw_observations: dict[int, str] = {}
    for turn in range(contract.turn_count):
        observation = (
            _recent_observation(turn)
            if turn >= recent_start
            else _old_observation(turn)
        )
        raw_observations[turn] = observation
        cmt.full_context.extend(
            [
                _event(
                    tokenizer,
                    author="llm",
                    role="assistant",
                    content=_assistant_content(turn),
                ),
                _event(
                    tokenizer,
                    author="env",
                    role="user",
                    content=observation,
                ),
            ]
        )

    first_build = cmt.context_policy.build(cmt.full_context)
    second_build = cmt.context_policy.build(cmt.full_context)
    _require(first_build == second_build, "same event log produced different prompts")
    stats = first_build.stats
    _require(
        stats["raw_prompt_tokens"] > contract.max_prompt_tokens,
        "synthetic raw prompt did not exceed the 22,528-token budget",
    )
    _require(
        stats["managed_prompt_tokens"] == len(first_build.prompt_token_ids),
        "managed token statistic differs from the rendered prompt length",
    )
    _require(
        len(first_build.prompt_token_ids) <= contract.max_prompt_tokens,
        "managed prompt exceeds max_prompt_tokens",
    )
    _require(stats["dropped_turns"] > 0, "whole-turn eviction did not trigger")
    _require(
        stats["compressed_turns"] == recent_start,
        "unexpected number of old/compressed turns",
    )
    _require(
        stats["clipped_observations"] == recent_start,
        "not every deliberately oversized historical observation was clipped",
    )

    messages = first_build.messages
    _require(len(messages) >= 2, "managed prompt lost initialization messages")
    _require(
        messages[0] == {"role": "system", "content": system_content}
        and messages[1]
        == {"role": "user", "content": initial_observation},
        "system or initial observation changed during management",
    )

    retained_indices = _extract_pair_indices(messages[2:])
    expected_indices = list(range(stats["dropped_turns"], contract.turn_count))
    _require(
        retained_indices == expected_indices,
        "eviction was not complete-turn, oldest-first suffix retention",
    )
    recent_indices = list(range(recent_start, contract.turn_count))
    _require(
        retained_indices[-contract.recent_turns :] == recent_indices,
        "configured recent turns were evicted or reordered",
    )

    retained_observation_messages = messages[3::2]
    retained_observations = {
        index: str(message["content"])
        for index, message in zip(retained_indices, retained_observation_messages)
    }
    retained_clipped = 0
    for turn in retained_indices:
        rendered = retained_observations[turn]
        if turn < recent_start:
            _require(
                CLIP_MARKER in rendered,
                f"retained old observation {turn} was not visibly clipped",
            )
            _require(
                len(tokenizer.encode(rendered, add_special_tokens=False))
                <= contract.history_observation_max_tokens,
                f"retained old observation {turn} exceeds its token cap",
            )
            retained_clipped += 1
        else:
            _require(
                rendered == raw_observations[turn],
                f"recent observation {turn} was modified",
            )
    _require(retained_clipped > 0, "no clipped history remained for inspection")
    _require(
        retained_observations[contract.turn_count - 1]
        == raw_observations[contract.turn_count - 1],
        "current observation was not retained exactly",
    )

    independently_rendered_ids = chat_template_ids(
        tokenizer, messages, add_generation_prompt=True
    )
    _require(
        independently_rendered_ids == first_build.prompt_token_ids,
        "ContextBuildResult token IDs differ from its managed messages",
    )

    actual_model_prompt = cmt.prepare_next_llm_context()
    _require(
        actual_model_prompt == messages,
        "Linear_CMT did not return the verified managed prompt",
    )
    actual_model_prompt_ids = chat_template_ids(
        tokenizer, actual_model_prompt, add_generation_prompt=True
    )
    _require(
        actual_model_prompt_ids == first_build.prompt_token_ids,
        "actual model-facing prompt IDs differ from the verified build",
    )
    cmt.save_llm_output(_sample_output(tokenizer), actual_model_prompt)
    _require(len(cmt.decision_snapshots) == 1, "decision snapshot was not captured")
    snapshot = cmt.decision_snapshots[0]
    _require(
        snapshot.prompt_messages == actual_model_prompt,
        "snapshot messages differ from the actual model-facing prompt",
    )
    _require(
        snapshot.prompt_token_ids == actual_model_prompt_ids,
        "snapshot token IDs differ from the actual model-facing prompt",
    )
    _require(
        snapshot.context_stats == stats,
        "snapshot context statistics differ from its managed prompt",
    )
    _require(
        snapshot.prompt_hash
        == cmt.context_policy.ids_hash(actual_model_prompt_ids),
        "snapshot prompt hash does not identify its token IDs",
    )

    return CanaryReport(
        profile=profile,
        config_path=contract.config_path,
        model_path=str(model_path),
        max_prompt_tokens=contract.max_prompt_tokens,
        response_length=contract.response_length,
        max_model_len=contract.max_model_len,
        recent_turns=contract.recent_turns,
        min_recent_turns=contract.min_recent_turns,
        history_observation_max_tokens=(
            contract.history_observation_max_tokens
        ),
        turn_count=contract.turn_count,
        raw_prompt_tokens=stats["raw_prompt_tokens"],
        managed_prompt_tokens=stats["managed_prompt_tokens"],
        compressed_turns=stats["compressed_turns"],
        dropped_turns=stats["dropped_turns"],
        clipped_observations=stats["clipped_observations"],
        retained_turns=len(retained_indices),
        first_retained_turn=retained_indices[0],
        retained_clipped_history_observations=retained_clipped,
        snapshot_prompt_tokens=len(snapshot.prompt_token_ids),
        snapshot_prompt_hash=snapshot.prompt_hash,
        deterministic=True,
        whole_turn_eviction_verified=True,
        initialization_protected=True,
        recent_turns_protected=True,
        current_observation_protected=True,
        snapshot_exact=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[*PROFILE_SPECS, "all"],
        default="all",
        help="production context profile to verify (default: all)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="local Qwen3.5 tokenizer/model directory",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="pretty-print the JSON report"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require(args.model.is_dir(), f"model/tokenizer directory not found: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    selected = list(PROFILE_SPECS) if args.profile == "all" else [args.profile]
    reports = [
        asdict(run_canary(tokenizer, profile=profile, model_path=args.model))
        for profile in selected
    ]
    payload: dict[str, Any] = {"status": "pass", "reports": reports}
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
