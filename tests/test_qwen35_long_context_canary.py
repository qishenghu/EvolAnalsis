"""CPU integration tests for the production 32K Qwen3.5 context canary."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tools.qwen35_long_context_canary import (
    DEFAULT_MODEL_DIR,
    EXPECTED_MAX_MODEL_LEN,
    EXPECTED_MAX_PROMPT_TOKENS,
    EXPECTED_RESPONSE_LENGTH,
    PROFILE_SPECS,
    load_profile_contract,
    run_canary,
)


@pytest.fixture(scope="module")
def qwen35_4b_tokenizer():
    if not DEFAULT_MODEL_DIR.is_dir():
        pytest.skip(f"Qwen3.5 tokenizer not found: {DEFAULT_MODEL_DIR}")
    return AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )


@pytest.mark.parametrize("profile", sorted(PROFILE_SPECS))
def test_production_profile_declares_exact_32k_partition(profile: str):
    contract = load_profile_contract(profile)
    assert contract.max_prompt_tokens == EXPECTED_MAX_PROMPT_TOKENS
    assert contract.response_length == EXPECTED_RESPONSE_LENGTH
    assert contract.max_model_len == EXPECTED_MAX_MODEL_LEN
    assert contract.max_prompt_tokens + contract.response_length == 32_768
    assert contract.recent_observation_max_tokens == -1
    assert contract.allow_current_observation_truncation is False
    assert contract.snapshot_training is True


@pytest.mark.parametrize("profile", sorted(PROFILE_SPECS))
def test_real_tokenizer_long_context_canary(qwen35_4b_tokenizer, profile: str):
    report = run_canary(qwen35_4b_tokenizer, profile=profile)

    assert report.raw_prompt_tokens > report.max_prompt_tokens
    assert report.managed_prompt_tokens <= report.max_prompt_tokens
    assert report.snapshot_prompt_tokens == report.managed_prompt_tokens
    assert report.dropped_turns > 0
    assert report.clipped_observations > 0
    assert report.retained_clipped_history_observations > 0
    assert report.first_retained_turn == report.dropped_turns
    assert report.retained_turns + report.dropped_turns == report.turn_count
    assert report.retained_turns >= report.recent_turns
    assert report.deterministic
    assert report.whole_turn_eviction_verified
    assert report.initialization_protected
    assert report.recent_turns_protected
    assert report.current_observation_protected
    assert report.snapshot_exact

    # The report is intended to be archived alongside experiment evidence.
    assert json.loads(json.dumps(asdict(report)))["profile"] == profile
    assert Path(report.config_path).is_file()
