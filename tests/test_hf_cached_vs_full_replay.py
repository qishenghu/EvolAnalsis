from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.hf_cached_vs_full_replay import (
    IdentitySample,
    ReplayContractError,
    compare_logprobs,
    load_identity_sample,
    run_replay,
)


class _SyntheticCausalLM(torch.nn.Module):
    """Tiny causal model whose full and cached arithmetic is deterministic."""

    def __init__(
        self,
        *,
        vocab_size: int = 17,
        cached_bias_context_length: int | None = None,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.vocab_size = vocab_size
        self.cached_bias_context_length = cached_bias_context_length

    def forward(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        **_kwargs,
    ):
        if past_key_values is None:
            prefix = input_ids
            past_length = 0
        else:
            prefix = torch.cat((past_key_values, input_ids), dim=1)
            past_length = past_key_values.shape[1]

        rows = []
        vocab_axis = torch.arange(
            self.vocab_size, device=input_ids.device, dtype=torch.float32
        )
        for local_index in range(input_ids.shape[1]):
            absolute_index = past_length + local_index
            context_sum = int(prefix[0, : absolute_index + 1].sum().item())
            logits = (
                vocab_axis * (0.03125 + context_sum * 0.0009765625)
                + (vocab_axis.square() % 7) * 0.0078125
                + self.anchor
            )
            if (
                past_key_values is not None
                and self.cached_bias_context_length == absolute_index + 1
            ):
                logits = logits.clone()
                logits[4] += 0.5
            rows.append(logits)
        output_logits = torch.stack(rows, dim=0).unsqueeze(0)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            output_logits = output_logits[:, -logits_to_keep:]
        return SimpleNamespace(
            logits=output_logits,
            past_key_values=prefix.detach().clone() if use_cache else None,
        )


def _sample(response_ids=(3, 4, 5)) -> IdentitySample:
    return IdentitySample(
        prompt_token_ids=torch.tensor([1, 2], dtype=torch.long),
        response_token_ids=torch.tensor(response_ids, dtype=torch.long),
        original_response_tokens=len(response_ids),
        sample_index=0,
        model_path=None,
        artifact_metadata={},
    )


def test_full_and_cached_replay_are_strictly_shift_aligned():
    comparison, full, cached = run_replay(
        _SyntheticCausalLM(),
        _sample(),
        device=torch.device("cpu"),
        logprob_chunk_size=2,
        divergence_atol=0.0,
    )

    assert torch.equal(full, cached)
    assert comparison["tokens"] == 3
    assert comparison["abs_mean"] == 0.0
    assert comparison["first_exact_divergence"] is None
    assert comparison["first_divergence_over_atol"] is None


def test_first_cached_divergence_reports_response_token_alignment():
    # Prompt length is 2.  Feeding response token 0 makes context length 3 and
    # produces the cached logits that score response token 1 (token ID 4).
    model = _SyntheticCausalLM(cached_bias_context_length=3)
    comparison, _full, _cached = run_replay(
        model,
        _sample(),
        device=torch.device("cpu"),
        logprob_chunk_size=4,
        divergence_atol=1e-6,
    )

    first = comparison["first_divergence_over_atol"]
    assert first is not None
    assert first["response_index"] == 1
    assert first["token_id"] == 4
    assert first["abs_delta"] > 0.1


def test_load_identity_sample_unpads_and_caps_response(tmp_path):
    prompts = torch.tensor([[0, 0, 7, 8], [0, 4, 5, 6]])
    responses = torch.tensor([[9, 10, 11, 0], [12, 13, 0, 0]])
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0, 0]]
    )
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    artifact = {
        "schema_version": 1,
        "metadata": {"model_path": "/models/qwen"},
        "tensors": {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=1),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
    }
    path = tmp_path / "identity.pt"
    torch.save(artifact, path)

    sample = load_identity_sample(
        path,
        sample_index=0,
        max_response_tokens=2,
    )

    assert sample.prompt_token_ids.tolist() == [7, 8]
    assert sample.response_token_ids.tolist() == [9, 10]
    assert sample.original_response_tokens == 3
    assert sample.model_path == "/models/qwen"


def test_load_identity_sample_rejects_noncontiguous_response_mask(tmp_path):
    artifact = {
        "schema_version": 1,
        "metadata": {},
        "tensors": {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 1]]),
            "response_mask": torch.tensor([[1, 0, 1]]),
        },
    }
    path = tmp_path / "bad.pt"
    torch.save(artifact, path)

    with pytest.raises(ReplayContractError, match="contiguous valid prefix"):
        load_identity_sample(path, sample_index=0, max_response_tokens=0)


def test_compare_logprobs_rejects_token_length_mismatch():
    with pytest.raises(ReplayContractError, match="same positive length"):
        compare_logprobs(
            torch.tensor([-1.0, -2.0]),
            torch.tensor([-1.0]),
            torch.tensor([1, 2]),
            divergence_atol=1e-5,
        )
