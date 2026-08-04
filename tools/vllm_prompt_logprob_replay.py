#!/usr/bin/env python3
"""Build and parse vLLM prompt-logprob replay artifacts without networking.

This helper deliberately does not contain an HTTP client.  ``build`` converts a
saved decision snapshot into a request body for ``POST /v1/completions``;
``parse`` validates a response captured separately and aligns teacher-forced
prompt logprobs with the saved response tokens.

Important vLLM 0.19.1 limitation
--------------------------------
The standard prompt-logprob path computes log-softmax directly from model
logits.  It does not apply sampling temperature or the other decode-time logit
processors, even when the server was launched with
``--logprobs-mode processed_logprobs``.  Therefore the extracted values have
``raw_logprobs`` semantics.  They are not directly comparable with processed
rollout logprobs when, for example, rollout temperature is 0.6.

For the cached-decode diagnostic, ``ignore_eos=true`` and ``min_tokens=0`` are
intentional.  ``ignore_eos`` alone makes the request run to ``max_tokens``.
Setting ``min_tokens`` to the same value is *not* an identity operation in
vLLM 0.19.1: the engine still adds the tokenizer EOS ID to
``all_stop_token_ids`` and masks its logit until ``min_tokens`` is reached.
That changes the decode log-softmax normalization and would confound a raw
cached-decode versus teacher-prefill comparison.

Example (all steps here are offline except the intentionally separate curl):

  python tools/vllm_prompt_logprob_replay.py build \
      --snapshot snapshot.json --output /tmp/replay-request.json

  # Run only while the rollout engine is awake and before its weights change:
  # curl -sS -H 'Content-Type: application/json' \
  #   --data-binary @/tmp/replay-request.json \
  #   http://127.0.0.1:PORT/v1/completions > /tmp/replay-response.json

  python tools/vllm_prompt_logprob_replay.py parse \
      --snapshot snapshot.json --response /tmp/replay-response.json \
      --rollout-semantics processed_logprobs --temperature 0.6

Accepted snapshot token fields are ``prompt_token_ids`` and either
``completion_token_ids`` or ``response_token_ids``.  Optional rollout
logprobs may be stored as ``completion_log_probs``, ``rollout_logprobs``, or
``response_logprobs``.  ``build`` and ``parse`` also accept a schema-v1
``identity_gate_failure_step_*.pt`` directly via ``--snapshot``.  The
``export-pt`` command exports every selected tensor sample as a JSON snapshot
and matching request while removing prompt left-padding with
``attention_mask`` and response right-padding with ``response_mask``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL = "shared_models/Qwen3.5-4B-think"
ENDPOINT = "/v1/completions"


class ReplayContractError(ValueError):
    """Raised when a saved snapshot or vLLM response violates the contract."""


def _read_json_record(path: Path, index: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        root = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        root = records

    if isinstance(root, dict) and isinstance(root.get("decision_snapshots"), list):
        root = root["decision_snapshots"]
    if isinstance(root, list):
        try:
            root = root[index]
        except IndexError as exc:
            raise ReplayContractError(
                f"record index {index} is outside a collection of size {len(root)}"
            ) from exc
    elif index != 0:
        raise ReplayContractError("--index is only valid for a record collection")

    if not isinstance(root, dict):
        raise ReplayContractError("selected snapshot must be a JSON object")
    return root


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(token) for token in token_ids).encode("ascii")
    ).hexdigest()


def _load_gate_artifact(path: Path) -> tuple[dict[str, Any], Any]:
    """Load the tensor-only gate artifact on CPU with safe torch unpickling."""
    try:
        import torch
    except ImportError as exc:
        raise ReplayContractError(
            "reading a .pt gate artifact requires PyTorch"
        ) from exc

    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ReplayContractError(f"cannot load gate artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ReplayContractError("gate artifact root must be a dictionary")
    if artifact.get("schema_version") != 1:
        raise ReplayContractError(
            "unsupported gate artifact schema_version: "
            f"{artifact.get('schema_version')!r}; expected 1"
        )
    if not isinstance(artifact.get("tensors"), dict):
        raise ReplayContractError("gate artifact has no tensors dictionary")
    if not isinstance(artifact.get("metadata", {}), dict):
        raise ReplayContractError("gate artifact metadata must be a dictionary")
    return artifact, torch


def _require_tensor(
    tensors: dict[str, Any], name: str, torch: Any, ndim: int
) -> Any:
    value = tensors.get(name)
    if not torch.is_tensor(value):
        raise ReplayContractError(f"gate artifact tensor {name!r} is missing")
    if value.ndim != ndim:
        raise ReplayContractError(
            f"gate artifact tensor {name!r} must be {ndim}D, got {value.ndim}D"
        )
    return value.detach().cpu()


def _binary_mask(value: Any, name: str, torch: Any) -> Any:
    if not bool(torch.all((value == 0) | (value == 1))):
        raise ReplayContractError(f"{name} must contain only zero/one values")
    return value.to(dtype=torch.bool)


def _finite_float_list(value: Any, name: str, torch: Any) -> list[float]:
    value = value.to(dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ReplayContractError(f"{name} contains non-finite values")
    return [float(item) for item in value.tolist()]


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _gate_sample_from_loaded(
    artifact: dict[str, Any],
    torch: Any,
    *,
    source_path: Path,
    sample_index: int,
) -> dict[str, Any]:
    tensors = artifact["tensors"]
    prompts = _require_tensor(tensors, "prompts", torch, 2)
    responses = _require_tensor(tensors, "responses", torch, 2)
    attention_mask = _require_tensor(tensors, "attention_mask", torch, 2)
    response_mask = _require_tensor(tensors, "response_mask", torch, 2)
    rollout_logprobs = _require_tensor(tensors, "rollout_log_probs", torch, 2)
    actor_logprobs = _require_tensor(tensors, "actor_old_log_probs", torch, 2)

    batch_size, prompt_width = prompts.shape
    response_batch, response_width = responses.shape
    if response_batch != batch_size:
        raise ReplayContractError("prompts/responses batch sizes differ")
    if not 0 <= sample_index < batch_size:
        raise ReplayContractError(
            f"sample index {sample_index} outside artifact batch size {batch_size}"
        )
    expected_response_shape = (batch_size, response_width)
    for name, value in (
        ("response_mask", response_mask),
        ("rollout_log_probs", rollout_logprobs),
        ("actor_old_log_probs", actor_logprobs),
    ):
        if tuple(value.shape) != expected_response_shape:
            raise ReplayContractError(
                f"{name} shape {tuple(value.shape)} != {expected_response_shape}"
            )
    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] < (
        prompt_width + response_width
    ):
        raise ReplayContractError(
            "attention_mask does not cover padded prompts plus responses"
        )

    prompt_mask = _binary_mask(
        attention_mask[sample_index, :prompt_width],
        "prompt attention_mask",
        torch,
    )
    prompt_valid = int(prompt_mask.sum().item())
    if prompt_valid <= 0:
        raise ReplayContractError("sample has no valid prompt tokens")
    prompt_start = prompt_width - prompt_valid
    if bool(prompt_mask[:prompt_start].any()) or not bool(
        prompt_mask[prompt_start:].all()
    ):
        raise ReplayContractError(
            "prompt attention_mask is not contiguous left-padding followed by tokens"
        )

    valid_response_mask = _binary_mask(
        response_mask[sample_index], "response_mask", torch
    )
    response_valid = int(valid_response_mask.sum().item())
    if response_valid <= 0:
        raise ReplayContractError("sample has no valid response tokens")
    if not bool(valid_response_mask[:response_valid].all()) or bool(
        valid_response_mask[response_valid:].any()
    ):
        raise ReplayContractError(
            "response_mask is not contiguous valid tokens followed by right-padding"
        )

    attention_response_mask = _binary_mask(
        attention_mask[
            sample_index, prompt_width : prompt_width + response_width
        ],
        "response portion of attention_mask",
        torch,
    )
    if not torch.equal(attention_response_mask, valid_response_mask):
        raise ReplayContractError(
            "response_mask differs from the response portion of attention_mask"
        )

    if "rollout_log_probs_mask" in tensors:
        rollout_mask = _require_tensor(
            tensors, "rollout_log_probs_mask", torch, 2
        )
        if tuple(rollout_mask.shape) != expected_response_shape or not torch.equal(
            _binary_mask(
                rollout_mask[sample_index], "rollout_log_probs_mask", torch
            ),
            valid_response_mask,
        ):
            raise ReplayContractError(
                "rollout_log_probs_mask differs from response_mask"
            )

    if "input_ids" in tensors:
        input_ids = _require_tensor(tensors, "input_ids", torch, 2)
        if input_ids.shape[0] != batch_size or input_ids.shape[1] < (
            prompt_width + response_width
        ):
            raise ReplayContractError("input_ids has an incompatible shape")
        if not torch.equal(
            input_ids[sample_index, :prompt_width].to(prompts.dtype),
            prompts[sample_index],
        ) or not torch.equal(
            input_ids[
                sample_index, prompt_width : prompt_width + response_width
            ].to(responses.dtype),
            responses[sample_index],
        ):
            raise ReplayContractError(
                "input_ids does not equal padded prompts concatenated with responses"
            )

    prompt_ids = [
        int(token)
        for token in prompts[sample_index][prompt_mask].to(torch.int64).tolist()
    ]
    response_ids = [
        int(token)
        for token in responses[sample_index][valid_response_mask]
        .to(torch.int64)
        .tolist()
    ]
    rollout_values = _finite_float_list(
        rollout_logprobs[sample_index][valid_response_mask],
        "valid rollout_log_probs",
        torch,
    )
    actor_values = _finite_float_list(
        actor_logprobs[sample_index][valid_response_mask],
        "valid actor_old_log_probs",
        torch,
    )

    metadata = {
        str(key): _json_scalar(value)
        for key, value in artifact.get("metadata", {}).items()
    }
    per_sample: dict[str, Any] = {}
    for name in (
        "group_ids",
        "context_raw_prompt_tokens",
        "context_managed_prompt_tokens",
        "context_compressed_turns",
        "context_dropped_turns",
        "context_clipped_observations",
        "context_selected_decision_step",
    ):
        value = tensors.get(name)
        if torch.is_tensor(value) and value.ndim == 1 and value.shape[0] == batch_size:
            per_sample[name] = _json_scalar(value[sample_index])

    return {
        "schema_version": 1,
        "source": {
            "artifact": str(source_path.resolve()),
            "artifact_schema_version": 1,
            "sample_index": sample_index,
            "batch_size": batch_size,
            "padded_prompt_width": prompt_width,
            "padded_response_width": response_width,
            "removed_prompt_left_padding": prompt_width - prompt_valid,
            "removed_response_right_padding": response_width - response_valid,
        },
        "metadata": metadata,
        "temperature": metadata.get("temperature"),
        "sample_metadata": per_sample,
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": response_ids,
        "completion_log_probs": rollout_values,
        "actor_old_log_probs": actor_values,
        "prompt_sha256": _ids_sha256(prompt_ids),
        "completion_sha256": _ids_sha256(response_ids),
        "combined_sha256": _ids_sha256([*prompt_ids, *response_ids]),
    }


def read_gate_sample(path: Path, sample_index: int) -> dict[str, Any]:
    artifact, torch = _load_gate_artifact(path)
    return _gate_sample_from_loaded(
        artifact,
        torch,
        source_path=path,
        sample_index=sample_index,
    )


def _read_snapshot_record(path: Path, index: int) -> dict[str, Any]:
    if path.suffix.lower() == ".pt":
        return read_gate_sample(path, index)
    return _read_json_record(path, index)


def _first_field(record: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _token_ids(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ReplayContractError(f"{field_name} must be a non-empty JSON list")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise ReplayContractError(
            f"{field_name} must contain only non-negative integer token IDs"
        )
    return list(value)


def snapshot_arrays(
    record: dict[str, Any],
) -> tuple[list[int], list[int], list[float] | None]:
    prompt_ids = _token_ids(record.get("prompt_token_ids"), "prompt_token_ids")
    response_value = _first_field(
        record, ("completion_token_ids", "response_token_ids")
    )
    response_ids = _token_ids(response_value, "completion/response_token_ids")

    rollout_value = _first_field(
        record,
        ("completion_log_probs", "rollout_logprobs", "response_logprobs"),
    )
    rollout_logprobs: list[float] | None = None
    if rollout_value is not None:
        if not isinstance(rollout_value, list):
            raise ReplayContractError("saved rollout logprobs must be a JSON list")
        rollout_logprobs = []
        for value in rollout_value:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReplayContractError("saved rollout logprobs must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ReplayContractError("saved rollout logprobs must be finite")
            rollout_logprobs.append(value)
        if len(rollout_logprobs) != len(response_ids):
            raise ReplayContractError(
                "rollout logprob count does not match response token count: "
                f"{len(rollout_logprobs)} != {len(response_ids)}"
            )
    return prompt_ids, response_ids, rollout_logprobs


def build_request(record: dict[str, Any], model: str) -> dict[str, Any]:
    prompt_ids, response_ids, _ = snapshot_arrays(record)
    combined_ids = [*prompt_ids, *response_ids]
    digest = hashlib.sha256(
        ",".join(str(token) for token in combined_ids).encode("ascii")
    ).hexdigest()[:20]
    return {
        "model": model,
        # Integer prompts bypass tokenization; no chat-template rebuild occurs.
        "prompt": combined_ids,
        # vLLM 0.19.1 explicitly supports echo + max_tokens=0 by internally
        # running one decode step and discarding it from the response.
        "echo": True,
        "max_tokens": 0,
        # Zero means chosen/target token only (not zero returned positions).
        "prompt_logprobs": 0,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "add_special_tokens": False,
        "stream": False,
        "request_id": f"teacher-replay-{digest}",
    }


def _snapshot_max_model_len(record: dict[str, Any]) -> int | None:
    value = record.get("max_model_len")
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get("max_model_len")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayContractError("snapshot max_model_len must be a positive integer")
    return value


def build_cached_decode_request(
    record: dict[str, Any], model: str, num_tokens: int = 128
) -> dict[str, Any]:
    """Build a fixed-length, raw-comparable cached-decode request.

    ``min_tokens`` must remain zero.  With vLLM 0.19.1, setting it to
    ``num_tokens`` masks EOS during every decode step even when
    ``ignore_eos`` is true, so the returned processed logprob would no longer
    equal the raw logits log-softmax used by prompt-logprob replay.
    """
    if num_tokens <= 0:
        raise ReplayContractError("decode diagnostic token count must be positive")
    prompt_ids, _, _ = snapshot_arrays(record)
    max_model_len = _snapshot_max_model_len(record)
    if max_model_len is not None and len(prompt_ids) + num_tokens > max_model_len:
        raise ReplayContractError(
            "cached-decode diagnostic would exceed max_model_len: "
            f"{len(prompt_ids)} + {num_tokens} > {max_model_len}"
        )
    digest = _ids_sha256(prompt_ids)[:20]
    return {
        "model": model,
        "prompt": prompt_ids,
        "echo": False,
        "max_tokens": num_tokens,
        "min_tokens": 0,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "stop": [],
        "stop_token_ids": [],
        "logit_bias": None,
        "allowed_token_ids": None,
        "logprobs": 0,
        "n": 1,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "stream": False,
        "request_id": f"cached-decode-diagnostic-{digest}",
    }


def _numeric_logprob_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ReplayContractError(f"{name} must be a list")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReplayContractError(f"{name}[{index}] must be numeric")
        item = float(item)
        if not math.isfinite(item):
            raise ReplayContractError(f"{name}[{index}] must be finite")
        result.append(item)
    return result


def parse_cached_decode_response(
    record: dict[str, Any], response: dict[str, Any], expected_tokens: int = 128
) -> dict[str, Any]:
    """Strictly parse chosen IDs/logprobs from a cached-decode response."""
    if expected_tokens <= 0:
        raise ReplayContractError("expected decode token count must be positive")
    prompt_ids, _, _ = snapshot_arrays(record)
    if not isinstance(response, dict):
        raise ReplayContractError("cached-decode response must be a JSON object")
    if response.get("error") is not None:
        raise ReplayContractError(
            f"cached-decode request returned an error: {response['error']!r}"
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ReplayContractError(
            "cached-decode response must contain exactly one choice"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ReplayContractError("cached-decode choice must be an object")
    if choice.get("index") != 0:
        raise ReplayContractError("cached-decode choice index must be zero")
    if choice.get("prompt_token_ids") != prompt_ids:
        raise ReplayContractError(
            "cached-decode response prompt_token_ids differ from the exact snapshot"
        )

    generated_ids = _token_ids(choice.get("token_ids"), "decode choice.token_ids")
    if len(generated_ids) != expected_tokens:
        raise ReplayContractError(
            "cached-decode token count differs from max_tokens/min_tokens: "
            f"{len(generated_ids)} != {expected_tokens}"
        )
    if choice.get("finish_reason") != "length":
        raise ReplayContractError(
            "cached-decode finish_reason must be 'length', got "
            f"{choice.get('finish_reason')!r}"
        )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ReplayContractError("cached-decode choice.logprobs is missing")
    chosen_logprobs = _numeric_logprob_list(
        logprobs.get("token_logprobs"), "decode token_logprobs"
    )
    if len(chosen_logprobs) != expected_tokens:
        raise ReplayContractError(
            "cached-decode logprob count differs from token count: "
            f"{len(chosen_logprobs)} != {expected_tokens}"
        )

    token_strings = logprobs.get("tokens")
    if not isinstance(token_strings, list) or len(token_strings) != expected_tokens:
        raise ReplayContractError(
            "cached-decode logprobs.tokens length differs from token count"
        )
    expected_strings = [f"token_id:{token_id}" for token_id in generated_ids]
    if token_strings != expected_strings:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(token_strings, expected_strings)
                )
                if actual != expected
            ),
            0,
        )
        raise ReplayContractError(
            "cached-decode logprobs.tokens does not encode choice.token_ids at "
            f"index {mismatch}"
        )

    text_offsets = logprobs.get("text_offset")
    if not isinstance(text_offsets, list) or len(text_offsets) != expected_tokens:
        raise ReplayContractError(
            "cached-decode text_offset length differs from token count"
        )
    top_logprobs = logprobs.get("top_logprobs")
    if not isinstance(top_logprobs, list) or len(top_logprobs) != expected_tokens:
        raise ReplayContractError(
            "cached-decode top_logprobs length differs from token count"
        )
    for index, (token_string, chosen_logprob, top_entry) in enumerate(
        zip(token_strings, chosen_logprobs, top_logprobs)
    ):
        if not isinstance(top_entry, dict) or token_string not in top_entry:
            raise ReplayContractError(
                f"decode top_logprobs[{index}] omits chosen token {token_string}"
            )
        top_value = top_entry[token_string]
        if isinstance(top_value, bool) or not isinstance(top_value, (int, float)):
            raise ReplayContractError(
                f"decode top_logprobs[{index}] chosen value is not numeric"
            )
        if float(top_value) != chosen_logprob:
            raise ReplayContractError(
                f"decode chosen logprob disagrees with top_logprobs at index {index}"
            )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ReplayContractError("cached-decode response usage is missing")
    expected_usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": expected_tokens,
        "total_tokens": len(prompt_ids) + expected_tokens,
    }
    for key, expected in expected_usage.items():
        if usage.get(key) != expected:
            raise ReplayContractError(
                f"cached-decode usage.{key}={usage.get(key)!r}, expected {expected}"
            )

    return {
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "decode_logprobs": chosen_logprobs,
        "finish_reason": choice.get("finish_reason"),
        "request_id": response.get("id"),
    }


def build_prefill_replay_request(
    record: dict[str, Any],
    decode_response: dict[str, Any],
    model: str,
    expected_tokens: int = 128,
) -> dict[str, Any]:
    decode = parse_cached_decode_response(record, decode_response, expected_tokens)
    diagnostic_record = {
        "prompt_token_ids": decode["prompt_token_ids"],
        "completion_token_ids": decode["generated_token_ids"],
    }
    max_model_len = _snapshot_max_model_len(record)
    combined_len = len(decode["prompt_token_ids"]) + len(
        decode["generated_token_ids"]
    )
    if max_model_len is not None and combined_len > max_model_len:
        raise ReplayContractError(
            f"teacher-forced replay length {combined_len} exceeds {max_model_len}"
        )
    return build_request(diagnostic_record, model)


def compare_cached_decode_to_prefill(
    record: dict[str, Any],
    decode_response: dict[str, Any],
    prefill_response: dict[str, Any],
    expected_tokens: int = 128,
) -> dict[str, Any]:
    """Compare identity-processed cached decode with raw teacher prefill."""
    decode = parse_cached_decode_response(record, decode_response, expected_tokens)
    diagnostic_record = {
        "prompt_token_ids": decode["prompt_token_ids"],
        "completion_token_ids": decode["generated_token_ids"],
        "completion_log_probs": decode["decode_logprobs"],
    }
    parsed_prefill = parse_replay(
        diagnostic_record,
        prefill_response,
        rollout_semantics="processed_logprobs",
        temperature=1.0,
        processed_is_identity=True,
    )
    if parsed_prefill["response_tokens"] != expected_tokens:
        raise ReplayContractError(
            "teacher-prefill response length differs from cached-decode length"
        )

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    prompt_len = len(decode["prompt_token_ids"])
    for index, (token_id, decode_logprob, teacher_logprob) in enumerate(
        zip(
            decode["generated_token_ids"],
            decode["decode_logprobs"],
            parsed_prefill["teacher_response_logprobs_raw"],
        )
    ):
        delta = teacher_logprob - decode_logprob
        deltas.append(delta)
        rows.append(
            {
                "generated_index": index,
                "combined_index": prompt_len + index,
                "logits_index": prompt_len + index - 1,
                "token_id": token_id,
                "cached_decode_logprob": decode_logprob,
                "teacher_prefill_raw_logprob": teacher_logprob,
                "signed_delta_teacher_minus_decode": delta,
                "abs_delta": abs(delta),
            }
        )
    summary = _delta_summary(deltas)
    summary["signed_mean_teacher_minus_decode"] = summary.pop(
        "signed_mean_teacher_minus_rollout"
    )
    return {
        "diagnostic": "cached_decode_vs_teacher_forced_prefill",
        "comparison_valid": True,
        "semantic_contract": {
            "cached_decode": (
                "processed_logprobs with identity processors: temperature=1, "
                "top_p=1, top_k=0, min_p=0, penalties disabled, min_tokens=0, "
                "and ignore_eos=true"
            ),
            "teacher_prefill": "raw_logprobs",
            "equivalence": (
                "processed decode equals raw logits log-softmax because every "
                "configured processor is identity"
            ),
        },
        "prompt_tokens": prompt_len,
        "generated_tokens": expected_tokens,
        "decode_request_id": decode["request_id"],
        "prefill_request_id": prefill_response.get("id"),
        "generated_token_ids": decode["generated_token_ids"],
        "cached_decode_logprobs": decode["decode_logprobs"],
        "teacher_prefill_raw_logprobs": parsed_prefill[
            "teacher_response_logprobs_raw"
        ],
        "drift_summary": summary,
        "rows": rows,
    }


def _lookup_target_logprob(entry: Any, token_id: int, position: int) -> float:
    if not isinstance(entry, dict):
        raise ReplayContractError(
            f"prompt_logprobs[{position}] must be an object, got {type(entry).__name__}"
        )
    value = entry.get(str(token_id))
    if value is None:
        # Useful when parse_replay is exercised directly before JSON serialization.
        value = entry.get(token_id)
    if value is None:
        raise ReplayContractError(
            f"prompt_logprobs[{position}] has no entry for target token {token_id}"
        )
    if isinstance(value, dict):
        value = value.get("logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayContractError(
            f"target token {token_id} at position {position} has no numeric logprob"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ReplayContractError(
            f"target token {token_id} at position {position} has non-finite logprob"
        )
    return value


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _delta_summary(deltas: Sequence[float]) -> dict[str, float]:
    absolute = [abs(value) for value in deltas]
    return {
        "signed_mean_teacher_minus_rollout": sum(deltas) / len(deltas),
        "abs_mean": sum(absolute) / len(absolute),
        "abs_max": max(absolute),
        "abs_p50": _percentile(absolute, 0.50),
        "abs_p90": _percentile(absolute, 0.90),
        "abs_p95": _percentile(absolute, 0.95),
        "abs_p99": _percentile(absolute, 0.99),
    }


def parse_replay(
    record: dict[str, Any],
    response: dict[str, Any],
    *,
    rollout_semantics: str,
    temperature: float,
    processed_is_identity: bool,
) -> dict[str, Any]:
    prompt_ids, response_ids, rollout_logprobs = snapshot_arrays(record)
    combined_ids = [*prompt_ids, *response_ids]

    if not isinstance(response, dict):
        raise ReplayContractError("vLLM response must be a JSON object")
    if response.get("error") is not None:
        raise ReplayContractError(f"vLLM returned an error: {response['error']!r}")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ReplayContractError("expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ReplayContractError("completion choice must be a JSON object")

    returned_ids = choice.get("prompt_token_ids")
    if returned_ids != combined_ids:
        mismatch = next(
            (
                i
                for i, (expected, actual) in enumerate(
                    zip(combined_ids, returned_ids or [])
                )
                if expected != actual
            ),
            min(len(combined_ids), len(returned_ids or [])),
        )
        raise ReplayContractError(
            "server prompt_token_ids differ from exact saved IDs at index "
            f"{mismatch}; expected_len={len(combined_ids)} "
            f"returned_len={len(returned_ids or [])}"
        )

    prompt_logprobs = choice.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list):
        raise ReplayContractError("choice.prompt_logprobs is missing or not a list")
    if len(prompt_logprobs) != len(combined_ids):
        raise ReplayContractError(
            "prompt_logprobs length does not match combined token count: "
            f"{len(prompt_logprobs)} != {len(combined_ids)}"
        )
    if prompt_logprobs[0] is not None:
        raise ReplayContractError("prompt_logprobs[0] must be null (no BOS predecessor)")

    response_rows: list[dict[str, Any]] = []
    teacher_logprobs: list[float] = []
    prompt_len = len(prompt_ids)
    for response_index, token_id in enumerate(response_ids):
        combined_index = prompt_len + response_index
        teacher_logprob = _lookup_target_logprob(
            prompt_logprobs[combined_index], token_id, combined_index
        )
        teacher_logprobs.append(teacher_logprob)
        row: dict[str, Any] = {
            "response_index": response_index,
            "combined_index": combined_index,
            "logits_index": combined_index - 1,
            "token_id": token_id,
            "teacher_raw_logprob": teacher_logprob,
        }
        if rollout_logprobs is not None:
            rollout_logprob = rollout_logprobs[response_index]
            delta = teacher_logprob - rollout_logprob
            row.update(
                rollout_logprob=rollout_logprob,
                signed_delta_teacher_minus_rollout=delta,
                abs_delta=abs(delta),
            )
        response_rows.append(row)

    comparable = rollout_semantics == "raw_logprobs" or (
        rollout_semantics == "processed_logprobs" and processed_is_identity
    )
    if rollout_semantics == "processed_logprobs" and temperature != 1.0:
        comparable = False

    if comparable:
        comparison_reason = "teacher and rollout logprobs have the same semantics"
    elif rollout_semantics == "processed_logprobs" and temperature != 1.0:
        comparison_reason = (
            "vLLM 0.19.1 prompt_logprobs are raw, but rollout logprobs were "
            f"computed after temperature={temperature:g}; exact scalar conversion "
            "requires the full vocabulary logits"
        )
    else:
        comparison_reason = (
            "processed decode logprobs are only comparable after explicitly proving "
            "all decode-time processors are identity"
        )

    result: dict[str, Any] = {
        "endpoint": ENDPOINT,
        "teacher_semantics": "raw_logprobs",
        "rollout_semantics": rollout_semantics,
        "temperature": temperature,
        "comparison_valid": comparable,
        "comparison_reason": comparison_reason,
        "prompt_tokens": prompt_len,
        "response_tokens": len(response_ids),
        "combined_tokens": len(combined_ids),
        "alignment": {
            "first_response_combined_index": prompt_len,
            "first_response_logits_index": prompt_len - 1,
            "rule": (
                "response[j] is combined[prompt_len+j] and is scored by "
                "logits[combined_index-1]"
            ),
        },
        "teacher_response_logprobs_raw": teacher_logprobs,
        "rows": response_rows,
    }
    if rollout_logprobs is not None:
        deltas = [
            teacher - rollout
            for teacher, rollout in zip(teacher_logprobs, rollout_logprobs)
        ]
        result["rollout_logprobs"] = rollout_logprobs
        result["delta_summary"] = _delta_summary(deltas)
        result["delta_summary_valid"] = comparable
    return result


def _write_json(value: Any, output: str) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        Path(output).write_text(text, encoding="utf-8")


def _record_temperature(record: dict[str, Any]) -> float:
    value = record.get("temperature")
    if value is None and isinstance(record.get("metadata"), dict):
        value = record["metadata"].get("temperature")
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayContractError("snapshot temperature must be numeric")
    value = float(value)
    if value <= 0 or not math.isfinite(value):
        raise ReplayContractError("snapshot temperature must be finite and positive")
    return value


def _build_command(args: argparse.Namespace) -> None:
    record = _read_snapshot_record(Path(args.snapshot), args.index)
    _write_json(build_request(record, args.model), args.output)


def _build_cached_decode_command(args: argparse.Namespace) -> None:
    record = _read_snapshot_record(Path(args.snapshot), args.index)
    request = build_cached_decode_request(record, args.model, args.tokens)
    _write_json(request, args.output)


def _build_prefill_replay_command(args: argparse.Namespace) -> None:
    record = _read_snapshot_record(Path(args.snapshot), args.index)
    decode_response = json.loads(
        Path(args.decode_response).read_text(encoding="utf-8")
    )
    request = build_prefill_replay_request(
        record,
        decode_response,
        args.model,
        args.tokens,
    )
    _write_json(request, args.output)


def _compare_decode_prefill_command(args: argparse.Namespace) -> None:
    record = _read_snapshot_record(Path(args.snapshot), args.index)
    decode_response = json.loads(
        Path(args.decode_response).read_text(encoding="utf-8")
    )
    prefill_response = json.loads(
        Path(args.prefill_response).read_text(encoding="utf-8")
    )
    result = compare_cached_decode_to_prefill(
        record,
        decode_response,
        prefill_response,
        args.tokens,
    )
    _write_json(result, args.output)


def _parse_command(args: argparse.Namespace) -> None:
    record = _read_snapshot_record(Path(args.snapshot), args.index)
    response = json.loads(Path(args.response).read_text(encoding="utf-8"))
    temperature = (
        _record_temperature(record) if args.temperature is None else args.temperature
    )
    result = parse_replay(
        record,
        response,
        rollout_semantics=args.rollout_semantics,
        temperature=temperature,
        processed_is_identity=args.processed_is_identity,
    )
    _write_json(result, args.output)
    if args.require_comparable and not result["comparison_valid"]:
        raise ReplayContractError(result["comparison_reason"])


def _export_pt_command(args: argparse.Namespace) -> None:
    source_path = Path(args.artifact)
    artifact, torch = _load_gate_artifact(source_path)
    prompts = _require_tensor(artifact["tensors"], "prompts", torch, 2)
    batch_size = int(prompts.shape[0])
    selected = list(range(batch_size)) if args.sample is None else args.sample
    if len(set(selected)) != len(selected):
        raise ReplayContractError("--sample contains duplicate indices")
    for sample_index in selected:
        if not 0 <= sample_index < batch_size:
            raise ReplayContractError(
                f"sample index {sample_index} outside artifact batch size {batch_size}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(max(batch_size - 1, 0))))
    planned: list[tuple[Path, Path, dict[str, Any], dict[str, Any]]] = []
    manifest_samples: list[dict[str, Any]] = []
    for sample_index in selected:
        snapshot = _gate_sample_from_loaded(
            artifact,
            torch,
            source_path=source_path,
            sample_index=sample_index,
        )
        request = build_request(snapshot, args.model)
        stem = f"sample_{sample_index:0{width}d}"
        snapshot_path = output_dir / f"{stem}.snapshot.json"
        request_path = output_dir / f"{stem}.request.json"
        planned.append((snapshot_path, request_path, snapshot, request))
        manifest_samples.append(
            {
                "sample_index": sample_index,
                "snapshot": snapshot_path.name,
                "request": request_path.name,
                "prompt_tokens": len(snapshot["prompt_token_ids"]),
                "response_tokens": len(snapshot["completion_token_ids"]),
                "combined_tokens": len(request["prompt"]),
                "combined_sha256": snapshot["combined_sha256"],
            }
        )

    manifest_path = output_dir / "manifest.json"
    targets = [manifest_path]
    for snapshot_path, request_path, _, _ in planned:
        targets.extend((snapshot_path, request_path))
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise ReplayContractError(
            "refusing to overwrite existing export files; use --force: "
            + ", ".join(existing)
        )

    for snapshot_path, request_path, snapshot, request in planned:
        _write_json(snapshot, str(snapshot_path))
        _write_json(request, str(request_path))
    manifest = {
        "schema_version": 1,
        "offline_only": True,
        "endpoint_for_later_use": ENDPOINT,
        "source_artifact": str(source_path.resolve()),
        "source_artifact_sha256": _file_sha256(source_path),
        "artifact_metadata": {
            str(key): _json_scalar(value)
            for key, value in artifact.get("metadata", {}).items()
        },
        "model": args.model,
        "batch_size": batch_size,
        "exported_samples": manifest_samples,
    }
    _write_json(manifest, str(manifest_path))
    if not getattr(args, "quiet", False):
        _write_json(
            {
                "exported": len(planned),
                "output_dir": str(output_dir.resolve()),
                "manifest": str(manifest_path.resolve()),
                "network_used": False,
            },
            "-",
        )


def _selftest_command(_: argparse.Namespace) -> None:
    record = {
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [12, 13],
        "completion_log_probs": [-1.25, -2.5],
    }
    response = {
        "choices": [
            {
                "prompt_token_ids": [10, 11, 12, 13],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.5, "rank": 1}},
                    {"12": {"logprob": -1.25, "rank": 2}},
                    {"13": {"logprob": -2.5, "rank": 3}},
                ],
            }
        ]
    }
    result = parse_replay(
        record,
        response,
        rollout_semantics="raw_logprobs",
        temperature=0.6,
        processed_is_identity=False,
    )
    assert result["comparison_valid"] is True
    assert result["teacher_response_logprobs_raw"] == [-1.25, -2.5]
    assert result["delta_summary"]["abs_max"] == 0.0

    decode_request = build_cached_decode_request(record, DEFAULT_MODEL, 128)
    assert decode_request["prompt"] == [10, 11]
    assert decode_request["max_tokens"] == 128
    assert decode_request["min_tokens"] == 0
    assert decode_request["temperature"] == 1.0
    assert decode_request["top_p"] == 1.0
    assert decode_request["top_k"] == 0
    assert decode_request["min_p"] == 0.0
    assert decode_request["repetition_penalty"] == 1.0
    assert decode_request["frequency_penalty"] == 0.0
    assert decode_request["presence_penalty"] == 0.0
    assert decode_request["stop"] == []
    assert decode_request["stop_token_ids"] == []
    assert decode_request["logit_bias"] is None
    assert decode_request["allowed_token_ids"] is None
    assert decode_request["ignore_eos"] is True
    assert decode_request["logprobs"] == 0

    generated_ids = [20, 21, 22]
    decode_logprobs = [-0.70, -0.80, -0.90]
    decode_token_strings = [f"token_id:{token}" for token in generated_ids]
    decode_response = {
        "id": "cmpl-decode-selftest",
        "choices": [
            {
                "index": 0,
                "text": "",
                "finish_reason": "length",
                "stop_reason": None,
                "prompt_token_ids": [10, 11],
                "token_ids": generated_ids,
                "logprobs": {
                    "tokens": decode_token_strings,
                    "token_logprobs": decode_logprobs,
                    "top_logprobs": [
                        {token: logprob}
                        for token, logprob in zip(
                            decode_token_strings, decode_logprobs
                        )
                    ],
                    "text_offset": [0, 11, 22],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    parsed_decode = parse_cached_decode_response(record, decode_response, 3)
    assert parsed_decode["generated_token_ids"] == generated_ids
    assert parsed_decode["decode_logprobs"] == decode_logprobs
    prefill_request = build_prefill_replay_request(
        record, decode_response, DEFAULT_MODEL, 3
    )
    assert prefill_request["prompt"] == [10, 11, 20, 21, 22]
    assert prefill_request["echo"] is True
    assert prefill_request["max_tokens"] == 0
    assert prefill_request["prompt_logprobs"] == 0

    prefill_response = {
        "id": "cmpl-prefill-selftest",
        "choices": [
            {
                "prompt_token_ids": [10, 11, 20, 21, 22],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.5}},
                    {"20": {"logprob": -0.69}},
                    {"21": {"logprob": -0.82}},
                    {"22": {"logprob": -0.90}},
                ],
            }
        ],
    }
    drift = compare_cached_decode_to_prefill(
        record, decode_response, prefill_response, 3
    )
    assert drift["comparison_valid"] is True
    assert drift["generated_token_ids"] == generated_ids
    assert math.isclose(drift["drift_summary"]["abs_max"], 0.02)
    assert drift["rows"][0]["combined_index"] == 2
    assert drift["rows"][0]["logits_index"] == 1

    malformed_decode = json.loads(json.dumps(decode_response))
    malformed_decode["choices"][0]["token_ids"][1] = 999
    try:
        parse_cached_decode_response(record, malformed_decode, 3)
    except ReplayContractError:
        pass
    else:
        raise AssertionError("decode ID mismatch was not rejected")

    import tempfile

    try:
        import torch
    except ImportError as exc:
        raise ReplayContractError("selftest requires PyTorch") from exc
    with tempfile.TemporaryDirectory(prefix="vllm-replay-selftest-") as tmp:
        tmp_path = Path(tmp)
        artifact_path = tmp_path / "identity_gate_failure_step_1.pt"
        export_dir = tmp_path / "export"
        prompts = torch.tensor([[0, 0, 10, 11], [20, 21, 22, 23]])
        responses = torch.tensor([[12, 13, 0], [24, 0, 0]])
        prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        artifact = {
            "schema_version": 1,
            "metadata": {"temperature": 0.6, "max_model_len": 32768},
            "tensors": {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=1),
                "attention_mask": torch.cat((prompt_mask, response_mask), dim=1),
                "response_mask": response_mask,
                "rollout_log_probs_mask": response_mask.clone(),
                "rollout_log_probs": torch.tensor(
                    [[-1.25, -2.5, 0.0], [-0.75, 0.0, 0.0]]
                ),
                "actor_old_log_probs": torch.tensor(
                    [[-1.20, -2.45, 0.0], [-0.70, 0.0, 0.0]]
                ),
                "group_ids": torch.tensor([7, 8]),
            },
        }
        torch.save(artifact, artifact_path)
        sample = read_gate_sample(artifact_path, 0)
        assert sample["prompt_token_ids"] == [10, 11]
        assert sample["completion_token_ids"] == [12, 13]
        assert sample["completion_log_probs"] == [-1.25, -2.5]
        assert sample["source"]["removed_prompt_left_padding"] == 2
        assert sample["source"]["removed_response_right_padding"] == 1
        export_args = argparse.Namespace(
            artifact=str(artifact_path),
            output_dir=str(export_dir),
            sample=None,
            model=DEFAULT_MODEL,
            force=False,
            quiet=True,
        )
        _export_pt_command(export_args)
        exported_snapshot = json.loads(
            (export_dir / "sample_0000.snapshot.json").read_text(encoding="utf-8")
        )
        exported_request = json.loads(
            (export_dir / "sample_0000.request.json").read_text(encoding="utf-8")
        )
        assert exported_snapshot["prompt_token_ids"] == [10, 11]
        assert exported_request["prompt"] == [10, 11, 12, 13]
        assert len(json.loads((export_dir / "manifest.json").read_text())["exported_samples"]) == 2

    _write_json(
        {
            "selftest": "ok",
            "cached_decode_prefill_drift": "ok",
            "strict_decode_id_validation": "ok",
            "pt_left_right_padding": "ok",
            "pt_per_sample_export": "ok",
            "network_used": False,
        },
        "-",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline builder/parser for vLLM /v1/completions prompt-logprob replay; "
            "this program never opens a network connection."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build an HTTP request JSON body")
    build.add_argument("--snapshot", required=True)
    build.add_argument("--index", type=int, default=0)
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--output", default="-")
    build.set_defaults(func=_build_command)

    cached_decode = subparsers.add_parser(
        "build-cached-decode",
        help="build a fixed-length raw-comparable cached-decode request",
    )
    cached_decode.add_argument("--snapshot", required=True)
    cached_decode.add_argument("--index", type=int, default=0)
    cached_decode.add_argument("--tokens", type=int, default=128)
    cached_decode.add_argument("--model", default=DEFAULT_MODEL)
    cached_decode.add_argument("--output", default="-")
    cached_decode.set_defaults(func=_build_cached_decode_command)

    prefill_replay = subparsers.add_parser(
        "build-prefill-replay",
        help="validate a decode response and build its teacher-prefill request",
    )
    prefill_replay.add_argument("--snapshot", required=True)
    prefill_replay.add_argument("--decode-response", required=True)
    prefill_replay.add_argument("--index", type=int, default=0)
    prefill_replay.add_argument("--tokens", type=int, default=128)
    prefill_replay.add_argument("--model", default=DEFAULT_MODEL)
    prefill_replay.add_argument("--output", default="-")
    prefill_replay.set_defaults(func=_build_prefill_replay_command)

    compare_decode = subparsers.add_parser(
        "compare-decode-prefill",
        help="strictly compare cached-decode and teacher-prefill logprobs",
    )
    compare_decode.add_argument("--snapshot", required=True)
    compare_decode.add_argument("--decode-response", required=True)
    compare_decode.add_argument("--prefill-response", required=True)
    compare_decode.add_argument("--index", type=int, default=0)
    compare_decode.add_argument("--tokens", type=int, default=128)
    compare_decode.add_argument("--output", default="-")
    compare_decode.set_defaults(func=_compare_decode_prefill_command)

    export_pt = subparsers.add_parser(
        "export-pt",
        help="export schema-v1 identity-gate tensor samples to snapshot/request JSON",
    )
    export_pt.add_argument("--artifact", required=True)
    export_pt.add_argument("--output-dir", required=True)
    export_pt.add_argument(
        "--sample",
        type=int,
        action="append",
        help="sample index to export; repeat as needed (default: every sample)",
    )
    export_pt.add_argument("--model", default=DEFAULT_MODEL)
    export_pt.add_argument("--force", action="store_true")
    export_pt.set_defaults(func=_export_pt_command)

    parse = subparsers.add_parser("parse", help="parse a separately captured response")
    parse.add_argument("--snapshot", required=True)
    parse.add_argument("--response", required=True)
    parse.add_argument("--index", type=int, default=0)
    parse.add_argument(
        "--rollout-semantics",
        choices=("raw_logprobs", "processed_logprobs"),
        default="processed_logprobs",
    )
    parse.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="rollout temperature (default: snapshot metadata, otherwise 1.0)",
    )
    parse.add_argument(
        "--processed-is-identity",
        action="store_true",
        help="assert all decode processors are identity (temperature must also be 1)",
    )
    parse.add_argument(
        "--require-comparable",
        action="store_true",
        help="exit nonzero if teacher and rollout semantics differ",
    )
    parse.add_argument("--output", default="-")
    parse.set_defaults(func=_parse_command)

    selftest = subparsers.add_parser("selftest", help="run a synthetic offline test")
    selftest.set_defaults(func=_selftest_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    temperature = getattr(args, "temperature", None)
    if temperature is not None and temperature <= 0:
        parser.error("--temperature must be positive")
    try:
        args.func(args)
    except (OSError, json.JSONDecodeError, ReplayContractError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
