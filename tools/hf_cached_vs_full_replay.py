#!/usr/bin/env python3
"""Offline HF full-teacher-force versus cached-decode identity replay.

This diagnostic reads one sample from a schema-v1
``identity_gate_failure_step_*.pt`` artifact and scores the exact same response
tokens in two Hugging Face execution modes:

* A: one unpadded ``use_cache=False`` full teacher-forced forward;
* B: one prompt prefill followed by single-token cached forwards.

Both sides report raw temperature-1 log-probabilities.  The tool never imports
vLLM and never performs network I/O.  It loads all floating model parameters as
BF16 while preserving floating model buffers as FP32, matching the actor's
mixed-precision contract.

Example::

    python tools/hf_cached_vs_full_replay.py \
      --artifact /path/to/identity_gate_failure_step_0.pt \
      --sample-index 0 --max-response-tokens 128 \
      --output /tmp/hf_cached_vs_full_sample0.json

``--model`` is optional when the artifact metadata contains ``model_path``.
The default response cap is deliberately 128 tokens; pass 0 to score all
response tokens after the short diagnostic is known to fit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ContextManager, Sequence

import torch
import torch.nn.functional as F


class ReplayContractError(ValueError):
    """Raised when an artifact or model output violates replay alignment."""


@dataclass(frozen=True)
class IdentitySample:
    """One strictly unpadded identity-gate sample on CPU."""

    prompt_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    original_response_tokens: int
    sample_index: int
    model_path: str | None
    artifact_metadata: dict[str, Any]


def _require_2d_tensor(tensors: dict[str, Any], name: str) -> torch.Tensor:
    value = tensors.get(name)
    if not torch.is_tensor(value):
        raise ReplayContractError(f"artifact tensor {name!r} is missing")
    value = value.detach().cpu()
    if value.ndim != 2:
        raise ReplayContractError(
            f"artifact tensor {name!r} must be 2D, got {value.ndim}D"
        )
    return value


def _binary_mask(value: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.all((value == 0) | (value == 1))):
        raise ReplayContractError(f"{name} must contain only zero and one")
    return value.bool()


def _left_padded_valid_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    valid = int(mask.sum().item())
    if valid <= 0:
        raise ReplayContractError(f"{name} contains no valid tokens")
    expected = torch.arange(mask.numel()) >= mask.numel() - valid
    if not torch.equal(mask.cpu(), expected):
        raise ReplayContractError(
            f"{name} must be contiguous left-padding followed by valid tokens"
        )
    return mask


def _right_padded_valid_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    valid = int(mask.sum().item())
    if valid <= 0:
        raise ReplayContractError(f"{name} contains no valid tokens")
    expected = torch.arange(mask.numel()) < valid
    if not torch.equal(mask.cpu(), expected):
        raise ReplayContractError(
            f"{name} must be a contiguous valid prefix followed by padding"
        )
    return mask


def load_identity_sample(
    artifact_path: Path,
    *,
    sample_index: int,
    max_response_tokens: int,
) -> IdentitySample:
    """Load and strictly unpad one schema-v1 identity artifact sample."""

    if max_response_tokens < 0:
        raise ReplayContractError("max_response_tokens must be non-negative")
    try:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ReplayContractError(
            f"cannot safely load identity artifact {artifact_path}: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise ReplayContractError("artifact root must be a dictionary")
    if artifact.get("schema_version") != 1:
        raise ReplayContractError(
            "unsupported artifact schema_version "
            f"{artifact.get('schema_version')!r}; expected 1"
        )
    tensors = artifact.get("tensors")
    metadata = artifact.get("metadata", {})
    if not isinstance(tensors, dict):
        raise ReplayContractError("artifact has no tensors dictionary")
    if not isinstance(metadata, dict):
        raise ReplayContractError("artifact metadata must be a dictionary")

    prompts = _require_2d_tensor(tensors, "prompts")
    responses = _require_2d_tensor(tensors, "responses")
    attention_mask = _require_2d_tensor(tensors, "attention_mask")
    response_mask = _require_2d_tensor(tensors, "response_mask")
    batch_size, prompt_width = prompts.shape
    response_batch, response_width = responses.shape
    if response_batch != batch_size:
        raise ReplayContractError("prompts and responses have different batch sizes")
    if not 0 <= sample_index < batch_size:
        raise ReplayContractError(
            f"sample index {sample_index} is outside batch size {batch_size}"
        )
    if tuple(response_mask.shape) != (batch_size, response_width):
        raise ReplayContractError(
            "response_mask shape does not match responses: "
            f"{tuple(response_mask.shape)} != {(batch_size, response_width)}"
        )
    if tuple(attention_mask.shape) != (
        batch_size,
        prompt_width + response_width,
    ):
        raise ReplayContractError(
            "attention_mask must cover padded prompts plus responses: "
            f"{tuple(attention_mask.shape)} != "
            f"{(batch_size, prompt_width + response_width)}"
        )

    prompt_mask = _left_padded_valid_mask(
        _binary_mask(
            attention_mask[sample_index, :prompt_width],
            "prompt attention_mask",
        ),
        "prompt attention_mask",
    )
    selected_response_mask = _right_padded_valid_mask(
        _binary_mask(response_mask[sample_index], "response_mask"),
        "response_mask",
    )
    attention_response_mask = _binary_mask(
        attention_mask[sample_index, prompt_width:],
        "response attention_mask",
    )
    if not torch.equal(selected_response_mask, attention_response_mask):
        raise ReplayContractError(
            "response_mask differs from the response slice of attention_mask"
        )

    if "input_ids" in tensors:
        input_ids = _require_2d_tensor(tensors, "input_ids")
        expected_shape = (batch_size, prompt_width + response_width)
        if tuple(input_ids.shape) != expected_shape:
            raise ReplayContractError(
                f"input_ids shape {tuple(input_ids.shape)} != {expected_shape}"
            )
        padded_join = torch.cat(
            (prompts[sample_index], responses[sample_index]), dim=0
        )
        if not torch.equal(input_ids[sample_index].to(padded_join), padded_join):
            raise ReplayContractError(
                "input_ids is not the exact padded prompts/responses concatenation"
            )

    prompt_ids = prompts[sample_index][prompt_mask].long().contiguous()
    response_ids = responses[sample_index][selected_response_mask].long().contiguous()
    if bool((prompt_ids < 0).any()) or bool((response_ids < 0).any()):
        raise ReplayContractError("prompt/response token IDs must be non-negative")
    original_response_tokens = int(response_ids.numel())
    if max_response_tokens > 0:
        response_ids = response_ids[:max_response_tokens].contiguous()

    model_path_value = metadata.get("model_path")
    model_path = str(model_path_value) if model_path_value else None
    return IdentitySample(
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        original_response_tokens=original_response_tokens,
        sample_index=sample_index,
        model_path=model_path,
        artifact_metadata=dict(metadata),
    )


def _normalize_floating_buffers_to_fp32(model: torch.nn.Module) -> None:
    """Keep parameter dtype untouched while moving floating buffers to FP32."""

    for module in model.modules():
        for name, buffer in tuple(module._buffers.items()):
            if buffer is not None and buffer.is_floating_point():
                module._buffers[name] = buffer.to(dtype=torch.float32)


def _dtype_contract(model: torch.nn.Module) -> dict[str, Any]:
    floating_parameters = [
        value for value in model.parameters() if value.is_floating_point()
    ]
    floating_buffers = [
        value for value in model.buffers() if value.is_floating_point()
    ]
    parameter_dtypes = Counter(str(value.dtype) for value in floating_parameters)
    buffer_dtypes = Counter(str(value.dtype) for value in floating_buffers)
    if not floating_parameters:
        raise ReplayContractError("model contains no floating parameters")
    if set(parameter_dtypes) != {str(torch.bfloat16)}:
        raise ReplayContractError(
            "expected every floating model parameter to be BF16, got "
            f"{dict(parameter_dtypes)}"
        )
    if any(value.dtype != torch.float32 for value in floating_buffers):
        raise ReplayContractError(
            "expected every floating model buffer to be FP32, got "
            f"{dict(buffer_dtypes)}"
        )
    return {
        "floating_parameter_tensors": len(floating_parameters),
        "floating_parameter_elements": sum(
            value.numel() for value in floating_parameters
        ),
        "floating_parameter_dtypes": dict(parameter_dtypes),
        "floating_buffer_tensors": len(floating_buffers),
        "floating_buffer_elements": sum(value.numel() for value in floating_buffers),
        "floating_buffer_dtypes": dict(buffer_dtypes),
    }


def load_hf_model(
    model_path: str,
    *,
    device: torch.device,
    attention_implementation: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a local HF checkpoint without quantizing its FP32 buffers."""

    resolved_model_path = Path(model_path).expanduser()
    if not resolved_model_path.exists():
        raise ReplayContractError(
            f"model path must exist locally; refusing remote lookup: {model_path}"
        )
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ReplayContractError("transformers is required to load the model") from exc

    config = AutoConfig.from_pretrained(
        str(resolved_model_path),
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=attention_implementation,
    )
    load_kwargs = {
        "pretrained_model_name_or_path": str(resolved_model_path),
        "config": config,
        "dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    except (KeyError, ValueError):
        architectures = getattr(config, "architectures", None) or []
        if not architectures or not hasattr(transformers, architectures[0]):
            raise
        model_class = getattr(transformers, architectures[0])
        model = model_class.from_pretrained(**load_kwargs)

    # Passing dtype to Module.to() would also cast inv_freq and other numerical
    # buffers.  from_pretrained(dtype=...) has already created BF16 parameters;
    # normalize only buffers before moving everything to the requested device.
    _normalize_floating_buffers_to_fp32(model)
    model = model.to(device=device).eval()
    model.requires_grad_(False)
    dtype_contract = _dtype_contract(model)
    return model, dtype_contract


def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Gather raw T=1 selected-token logprobs using FP32 log-softmax."""

    if logits.ndim != 2:
        raise ReplayContractError(f"logits must be 2D, got {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.numel() != logits.shape[0]:
        raise ReplayContractError(
            "labels must be 1D and align with logits rows: "
            f"{tuple(labels.shape)} vs {tuple(logits.shape)}"
        )
    if chunk_size <= 0:
        raise ReplayContractError("logprob chunk_size must be positive")
    pieces = []
    for start in range(0, labels.numel(), chunk_size):
        end = min(start + chunk_size, labels.numel())
        chunk_logits = logits[start:end].float()
        chunk_labels = labels[start:end].long()
        pieces.append(
            F.log_softmax(chunk_logits, dim=-1).gather(
                -1, chunk_labels.unsqueeze(-1)
            )[:, 0]
        )
    values = torch.cat(pieces)
    if not bool(torch.isfinite(values).all()):
        raise ReplayContractError("selected token logprobs contain non-finite values")
    return values


def _autocast_context(device: torch.device) -> ContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _require_logits(output: Any, *, expected_batch: int = 1) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ReplayContractError("model output must expose 3D .logits")
    if logits.shape[0] != expected_batch:
        raise ReplayContractError(
            f"model output batch {logits.shape[0]} != {expected_batch}"
        )
    return logits


def full_teacher_forced_logprobs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    device: torch.device,
    logprob_chunk_size: int,
) -> torch.Tensor:
    """A: score response tokens in one unpadded, cache-free forward."""

    prompt = prompt_ids.to(device=device, dtype=torch.long)
    response = response_ids.to(device=device, dtype=torch.long)
    combined = torch.cat((prompt, response)).unsqueeze(0)
    positions = torch.arange(
        combined.shape[1], device=device, dtype=torch.long
    ).unsqueeze(0)
    attention_mask = torch.ones_like(combined)
    with _autocast_context(device):
        output = model(
            input_ids=combined,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
            logits_to_keep=response.numel() + 1,
        )
    output_logits = _require_logits(output)[0]
    if output_logits.shape[0] == response.numel() + 1:
        aligned_logits = output_logits[:-1]
    elif output_logits.shape[0] == combined.shape[1]:
        prompt_length = prompt.numel()
        aligned_logits = output_logits[
            prompt_length - 1 : prompt_length + response.numel() - 1
        ]
    else:
        raise ReplayContractError(
            "full forward returned an unexpected number of logits: "
            f"{output_logits.shape[0]} (combined={combined.shape[1]}, "
            f"response={response.numel()})"
        )
    values = selected_token_logprobs(
        aligned_logits,
        response,
        chunk_size=logprob_chunk_size,
    )
    return values.detach().float().cpu()


def _cache_state_dtypes(past_key_values: Any) -> dict[str, dict[str, int]]:
    """Summarize HF hybrid-cache tensor dtypes without assuming a cache class."""

    summary: dict[str, Counter[str]] = {
        "conv_states": Counter(),
        "recurrent_states": Counter(),
        "keys": Counter(),
        "values": Counter(),
    }
    for layer in getattr(past_key_values, "layers", ()):
        for field, output_name in (
            ("conv_states", "conv_states"),
            ("recurrent_states", "recurrent_states"),
            ("keys", "keys"),
            ("values", "values"),
        ):
            value = getattr(layer, field, None)
            if torch.is_tensor(value):
                summary[output_name][str(value.dtype)] += 1
    return {name: dict(counts) for name, counts in summary.items() if counts}


def cached_tokenwise_logprobs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    device: torch.device,
    logprob_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """B: score fixed response tokens after prefill via one-token cache steps."""

    prompt = prompt_ids.to(device=device, dtype=torch.long)
    response = response_ids.to(device=device, dtype=torch.long)
    prompt_input = prompt.unsqueeze(0)
    prompt_positions = torch.arange(
        prompt.numel(), device=device, dtype=torch.long
    ).unsqueeze(0)
    with _autocast_context(device):
        output = model(
            input_ids=prompt_input,
            attention_mask=torch.ones_like(prompt_input),
            position_ids=prompt_positions,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    past_key_values = getattr(output, "past_key_values", None)
    if past_key_values is None:
        raise ReplayContractError("cached prompt prefill returned no past_key_values")
    prefill_cache_dtypes = _cache_state_dtypes(past_key_values)
    next_logits = _require_logits(output)[0, -1]
    values = []

    for response_index in range(response.numel()):
        label = response[response_index : response_index + 1]
        value = selected_token_logprobs(
            next_logits.unsqueeze(0),
            label,
            chunk_size=logprob_chunk_size,
        )[0]
        values.append(value)
        if response_index + 1 == response.numel():
            break

        decode_input = label.view(1, 1)
        absolute_position = prompt.numel() + response_index
        position_ids = torch.tensor(
            [[absolute_position]], device=device, dtype=torch.long
        )
        # Full-attention cache masking sees the complete prefix length, while
        # Qwen3.5's linear-attention path recognizes this as an all-valid mask.
        attention_mask = torch.ones(
            (1, absolute_position + 1), device=device, dtype=torch.long
        )
        with _autocast_context(device):
            output = model(
                input_ids=decode_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        next_past = getattr(output, "past_key_values", None)
        if next_past is None:
            raise ReplayContractError(
                f"cached step {response_index} returned no past_key_values"
            )
        past_key_values = next_past
        step_logits = _require_logits(output)
        if step_logits.shape[1] != 1:
            raise ReplayContractError(
                f"cached step {response_index} returned {step_logits.shape[1]} logits"
            )
        next_logits = step_logits[0, 0]

    final_cache_dtypes = _cache_state_dtypes(past_key_values)
    result = torch.stack(values).detach().float().cpu()
    if not bool(torch.isfinite(result).all()):
        raise ReplayContractError("cached token logprobs contain non-finite values")
    return result, prefill_cache_dtypes, final_cache_dtypes


def _first_divergence(
    mask: torch.Tensor,
    *,
    response_token_ids: torch.Tensor,
    full: torch.Tensor,
    cached: torch.Tensor,
) -> dict[str, Any] | None:
    locations = torch.nonzero(mask, as_tuple=False).flatten()
    if locations.numel() == 0:
        return None
    index = int(locations[0].item())
    signed_delta = float((cached[index] - full[index]).item())
    return {
        "response_index": index,
        "token_id": int(response_token_ids[index].item()),
        "full_logprob": float(full[index].item()),
        "cached_logprob": float(cached[index].item()),
        "signed_delta_cached_minus_full": signed_delta,
        "abs_delta": abs(signed_delta),
    }


def compare_logprobs(
    full: torch.Tensor,
    cached: torch.Tensor,
    response_token_ids: torch.Tensor,
    *,
    divergence_atol: float,
) -> dict[str, Any]:
    """Compute aligned drift metrics and exact/tolerance first divergences."""

    if divergence_atol < 0:
        raise ReplayContractError("divergence_atol must be non-negative")
    full = full.detach().float().cpu().flatten()
    cached = cached.detach().float().cpu().flatten()
    response_token_ids = response_token_ids.detach().long().cpu().flatten()
    if not (
        full.numel() == cached.numel() == response_token_ids.numel()
        and full.numel() > 0
    ):
        raise ReplayContractError(
            "full/cached/token IDs must have the same positive length: "
            f"{full.numel()}/{cached.numel()}/{response_token_ids.numel()}"
        )
    if not bool(torch.isfinite(full).all() and torch.isfinite(cached).all()):
        raise ReplayContractError("full/cached logprobs must be finite")

    signed_delta = cached - full
    abs_delta = signed_delta.abs()
    quantiles = torch.quantile(
        abs_delta,
        torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float32),
    )
    return {
        "tokens": int(full.numel()),
        "delta_definition": "cached_minus_full",
        "signed_mean": float(signed_delta.mean().item()),
        "abs_mean": float(abs_delta.mean().item()),
        "abs_max": float(abs_delta.max().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p90": float(quantiles[1].item()),
        "abs_p95": float(quantiles[2].item()),
        "abs_p99": float(quantiles[3].item()),
        "divergence_atol": float(divergence_atol),
        "first_exact_divergence": _first_divergence(
            abs_delta != 0,
            response_token_ids=response_token_ids,
            full=full,
            cached=cached,
        ),
        "first_divergence_over_atol": _first_divergence(
            abs_delta > divergence_atol,
            response_token_ids=response_token_ids,
            full=full,
            cached=cached,
        ),
        "all_within_atol": bool(torch.all(abs_delta <= divergence_atol)),
    }


def _ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(token)) for token in token_ids).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_replay(
    model: torch.nn.Module,
    sample: IdentitySample,
    *,
    device: torch.device,
    logprob_chunk_size: int,
    divergence_atol: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Run A/B on one already-loaded model and return summary plus raw rows."""

    if sample.response_token_ids.numel() <= 0:
        raise ReplayContractError("selected response is empty")
    with torch.inference_mode():
        full = full_teacher_forced_logprobs(
            model,
            sample.prompt_token_ids,
            sample.response_token_ids,
            device=device,
            logprob_chunk_size=logprob_chunk_size,
        )
        cached, prefill_cache_dtypes, final_cache_dtypes = (
            cached_tokenwise_logprobs(
                model,
                sample.prompt_token_ids,
                sample.response_token_ids,
                device=device,
                logprob_chunk_size=logprob_chunk_size,
            )
        )
    comparison = compare_logprobs(
        full,
        cached,
        sample.response_token_ids,
        divergence_atol=divergence_atol,
    )
    comparison["prefill_cache_dtypes"] = prefill_cache_dtypes
    comparison["final_cache_dtypes"] = final_cache_dtypes
    return comparison, full, cached


def _format_divergence(value: dict[str, Any] | None) -> str:
    if value is None:
        return "none"
    return (
        f"index={value['response_index']} token_id={value['token_id']} "
        f"abs={value['abs_delta']:.9g} "
        f"signed={value['signed_delta_cached_minus_full']:.9g}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline raw/T=1 HF replay: full teacher force versus prompt "
            "prefill + one-token cached decode"
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--model",
        help="Local HF model path; defaults to artifact metadata.model_path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=128,
        help="Score this many response-prefix tokens; 0 means all (default: 128)",
    )
    parser.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=32,
        help="Token rows per FP32 log-softmax chunk (default: 32)",
    )
    parser.add_argument(
        "--divergence-atol",
        type=float,
        default=1e-5,
        help="Absolute threshold for the reported first material divergence",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="HF attention implementation (default: flash_attention_2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for per-token logprobs and summary",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sample = load_identity_sample(
        args.artifact,
        sample_index=args.sample_index,
        max_response_tokens=args.max_response_tokens,
    )
    model_path = args.model or sample.model_path
    if not model_path:
        raise ReplayContractError(
            "--model is required because artifact metadata has no model_path"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ReplayContractError("CUDA device requested but torch.cuda is unavailable")

    model, dtype_contract = load_hf_model(
        model_path,
        device=device,
        attention_implementation=args.attn_implementation,
    )
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None:
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None:
        largest_token = int(
            torch.cat((sample.prompt_token_ids, sample.response_token_ids))
            .max()
            .item()
        )
        if largest_token >= int(vocab_size):
            raise ReplayContractError(
                f"artifact token ID {largest_token} exceeds model vocab {vocab_size}"
            )

    comparison, full, cached = run_replay(
        model,
        sample,
        device=device,
        logprob_chunk_size=args.logprob_chunk_size,
        divergence_atol=args.divergence_atol,
    )
    # Verify forward did not mutate the persistent model dtype contract.
    post_replay_dtype_contract = _dtype_contract(model)
    if post_replay_dtype_contract != dtype_contract:
        raise ReplayContractError("model parameter/buffer dtype contract changed")

    response_ids = [int(value) for value in sample.response_token_ids.tolist()]
    prompt_ids = [int(value) for value in sample.prompt_token_ids.tolist()]
    result = {
        "schema_version": 1,
        "semantics": "raw_logprobs_temperature_1",
        "source_artifact": str(args.artifact.resolve()),
        "source_artifact_sha256": _file_sha256(args.artifact),
        "sample_index": sample.sample_index,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_class": type(model).__name__,
        "device": str(device),
        "versions": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "flash-linear-attention": _version("flash-linear-attention"),
            "causal-conv1d": _version("causal-conv1d"),
        },
        "dtype_contract": dtype_contract,
        "prompt_tokens": len(prompt_ids),
        "original_response_tokens": sample.original_response_tokens,
        "scored_response_tokens": len(response_ids),
        "response_truncated": len(response_ids) < sample.original_response_tokens,
        "prompt_sha256": _ids_sha256(prompt_ids),
        "response_sha256": _ids_sha256(response_ids),
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "comparison": comparison,
        "full_teacher_forced_raw_logprobs": [float(value) for value in full.tolist()],
        "cached_tokenwise_raw_logprobs": [float(value) for value in cached.tolist()],
        "signed_delta_cached_minus_full": [
            float(value) for value in (cached - full).tolist()
        ],
    }

    print(
        f"sample={sample.sample_index} prompt_tokens={len(prompt_ids)} "
        f"response_tokens={len(response_ids)}/{sample.original_response_tokens} "
        "semantics=raw_logprobs_temperature_1",
        flush=True,
    )
    print(
        "cached-minus-full: "
        f"signed_mean={comparison['signed_mean']:.9g} "
        f"abs_mean={comparison['abs_mean']:.9g} "
        f"p50/p90/p95/p99={comparison['abs_p50']:.9g}/"
        f"{comparison['abs_p90']:.9g}/{comparison['abs_p95']:.9g}/"
        f"{comparison['abs_p99']:.9g} max={comparison['abs_max']:.9g}",
        flush=True,
    )
    print(
        "first_exact_divergence: "
        + _format_divergence(comparison["first_exact_divergence"]),
        flush=True,
    )
    print(
        f"first_divergence_over_atol({args.divergence_atol:g}): "
        + _format_divergence(comparison["first_divergence_over_atol"]),
        flush=True,
    )
    print(
        "cache_dtypes_after_prefill="
        + json.dumps(comparison["prefill_cache_dtypes"], sort_keys=True),
        flush=True,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
