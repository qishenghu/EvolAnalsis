from types import SimpleNamespace

import pytest
import torch

from agentevolver.module.exp_manager.het_actor import (
    HETDataParallelPPOActor,
    _crop_single_item_left_padding,
    _entropy_with_fp32_temperature,
    _logprobs_with_fp32_temperature,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.6, 0.9, 1.0])
def test_fp32_temperature_logprobs_match_reference(dtype, temperature):
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(9, 127, generator=generator).to(dtype)
    labels = torch.randint(0, logits.shape[-1], (logits.shape[0],), generator=generator)

    actual = _logprobs_with_fp32_temperature(
        logits,
        labels,
        temperature=temperature,
        inplace_backward=False,
    )
    reference = torch.log_softmax(logits.float() / temperature, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0.6, 0.9, 1.0])
def test_fp32_temperature_entropy_is_chunk_invariant(temperature):
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(11, 251, generator=generator).to(torch.bfloat16)

    actual = _entropy_with_fp32_temperature(
        logits,
        temperature=temperature,
        chunk_size=3,
    )
    scaled = logits.float() / temperature
    probabilities = torch.softmax(scaled, dim=-1)
    reference = torch.logsumexp(scaled, dim=-1) - torch.sum(
        probabilities * scaled, dim=-1
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class _TailLogitsModule:
    def __init__(self, logits):
        self.logits = logits
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        keep = kwargs["logits_to_keep"]
        active_length = kwargs["input_ids"].size(-1)
        return SimpleNamespace(
            logits=self.logits[:, :active_length, :][:, -keep:, :]
        )


def _bare_precise_actor(module):
    actor = object.__new__(HETDataParallelPPOActor)
    actor.config = {
        "use_dr3": False,
        "behavior_logprob_fp32_temperature": True,
    }
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.device_name = "cpu"
    actor.actor_module = module
    return actor


def test_non_rmpad_tail_logits_align_with_response_tokens():
    generator = torch.Generator().manual_seed(29)
    batch_size, prompt_length, response_length, vocab_size = 1, 5, 3, 17
    sequence_length = prompt_length + response_length
    logits = torch.randn(
        batch_size,
        sequence_length,
        vocab_size,
        generator=generator,
    ).to(torch.bfloat16)
    responses = torch.randint(
        0,
        vocab_size,
        (batch_size, response_length),
        generator=generator,
    )
    module = _TailLogitsModule(logits)
    actor = _bare_precise_actor(module)
    input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            generator=generator,
        )
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    attention_mask[:, -1] = 0
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(sequence_length).repeat(batch_size, 1),
        "responses": responses,
    }

    entropy, actual = actor._forward_micro_batch(
        micro_batch,
        temperature=0.6,
        calculate_entropy=True,
    )

    # The final prompt logit predicts response token 0; the final input logit
    # has no next response target and must be excluded.
    valid_response_length = response_length - 1
    response_logits = logits[
        :, -(response_length + 1) : -(response_length - valid_response_length + 1), :
    ]
    scaled = response_logits.float() / 0.6
    reference = torch.log_softmax(scaled, dim=-1).gather(
        -1, responses[:, :valid_response_length].unsqueeze(-1)
    ).squeeze(-1)
    probabilities = torch.softmax(scaled, dim=-1)
    entropy_reference = torch.logsumexp(scaled, dim=-1) - torch.sum(
        probabilities * scaled,
        dim=-1,
    )

    assert module.last_kwargs["logits_to_keep"] == valid_response_length + 1
    assert module.last_kwargs["use_cache"] is False
    torch.testing.assert_close(
        actual[:, :valid_response_length], reference, rtol=0, atol=0
    )
    torch.testing.assert_close(
        entropy[:, :valid_response_length], entropy_reference, rtol=0, atol=0
    )
    assert torch.equal(actual[:, valid_response_length:], torch.zeros(1, 1))
    assert torch.equal(entropy[:, valid_response_length:], torch.zeros(1, 1))


def test_precise_qwen35_path_rejects_remove_padding():
    module = _TailLogitsModule(torch.zeros(1, 3, 7, dtype=torch.bfloat16))
    actor = _bare_precise_actor(module)
    actor.use_remove_padding = True
    micro_batch = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(3).unsqueeze(0),
        "responses": torch.ones(1, 2, dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="use_remove_padding=false"):
        actor._forward_micro_batch(micro_batch, temperature=0.9)


def test_crop_single_item_left_padding_preserves_real_and_right_pad_suffix():
    input_ids = torch.tensor([[99, 99, 10, 11, 12, 13, 99]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    position_ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4]])

    cropped_ids, cropped_mask, cropped_positions, removed = (
        _crop_single_item_left_padding(input_ids, attention_mask, position_ids)
    )

    assert removed == 2
    assert cropped_ids.tolist() == [[10, 11, 12, 13, 99]]
    assert cropped_mask.tolist() == [[1, 1, 1, 1, 0]]
    assert cropped_positions.tolist() == [[0, 1, 2, 3, 4]]


def test_exact_qwen35_scorer_rejects_multi_item_microbatch():
    input_ids = torch.ones(2, 4, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(4).repeat(2, 1)

    with pytest.raises(RuntimeError, match="micro_batch_size_per_gpu=1"):
        _crop_single_item_left_padding(input_ids, attention_mask, position_ids)
