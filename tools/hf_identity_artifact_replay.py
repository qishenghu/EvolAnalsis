#!/usr/bin/env python3
"""Recompute raw and temperature-processed HF logprobs for a gate artifact.

The trainer's strict identity failure artifact contains the exact padded batch.
This tool scores each sample as an independent, fully unpadded Qwen3.5 request,
matching ``HETDataParallelPPOActor`` while also emitting temperature=1 raw
logprobs for comparison with vLLM's teacher-forced prompt-logprob endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def _load_model(model_path: str, device: torch.device):
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
    )
    kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch.bfloat16,
        "config": config,
        "trust_remote_code": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    except (KeyError, ValueError):
        import transformers

        architectures = getattr(config, "architectures", None) or []
        if not architectures or not hasattr(transformers, architectures[0]):
            raise
        model = getattr(transformers, architectures[0]).from_pretrained(**kwargs)
    # ``from_pretrained(torch_dtype=bf16)`` already casts checkpoint
    # parameters.  Do not pass ``dtype`` to ``Module.to`` here: doing so also
    # quantizes non-persistent numerical buffers such as Qwen3.5 RoPE
    # ``inv_freq``.  The trainer's FSDP mixed-precision contract uses bf16
    # parameters but keeps buffers in fp32.
    model = model.to(device=device).eval()
    floating_param_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_param_dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "expected all floating checkpoint parameters to be bf16, got "
            f"{sorted(map(str, floating_param_dtypes))}"
        )
    return model


def _selected_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    pieces = []
    for start in range(0, labels.numel(), chunk_size):
        end = min(start + chunk_size, labels.numel())
        chunk_logits = logits[start:end].float() / float(temperature)
        pieces.append(
            -F.cross_entropy(
                chunk_logits,
                labels[start:end],
                reduction="none",
            )
        )
    return torch.cat(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    artifact = torch.load(
        args.artifact, map_location="cpu", weights_only=False
    )
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported identity artifact schema")
    tensors = artifact["tensors"]
    temperature = float(artifact["metadata"]["temperature"])
    device = torch.device(args.device)
    model = _load_model(args.model, device)

    prompt_width = tensors["prompts"].shape[1]
    response_width = tensors["responses"].shape[1]
    attention_mask = tensors["attention_mask"].bool()
    prompt_mask = attention_mask[:, :prompt_width]
    response_mask = tensors["response_mask"].bool()
    actor_saved = tensors["actor_old_log_probs"]

    raw_rows: list[torch.Tensor] = []
    processed_rows: list[torch.Tensor] = []
    prompt_ids_rows: list[torch.Tensor] = []
    response_ids_rows: list[torch.Tensor] = []

    with torch.inference_mode():
        for sample_index in range(tensors["prompts"].shape[0]):
            prompt_ids = tensors["prompts"][sample_index][
                prompt_mask[sample_index]
            ].long()
            response_ids = tensors["responses"][sample_index][
                response_mask[sample_index]
            ].long()
            input_ids = torch.cat((prompt_ids, response_ids)).unsqueeze(0).to(device)
            position_ids = torch.arange(
                input_ids.shape[1], device=device, dtype=torch.long
            ).unsqueeze(0)
            dense_mask = torch.ones_like(input_ids)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    attention_mask=dense_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=response_ids.numel() + 1,
                )
            logits = output.logits[0, :-1]
            if logits.shape[0] != response_ids.numel():
                raise RuntimeError(
                    f"sample {sample_index}: unexpected logits shape {tuple(logits.shape)}"
                )
            labels = response_ids.to(device)
            raw = _selected_logprobs(logits, labels, temperature=1.0).cpu()
            processed = _selected_logprobs(
                logits, labels, temperature=temperature
            ).cpu()
            saved = actor_saved[sample_index, : response_ids.numel()].float()
            error = (processed.float() - saved).abs()
            print(
                f"sample={sample_index} prompt={prompt_ids.numel()} "
                f"response={response_ids.numel()} "
                f"saved_processed_abs_mean={error.mean().item():.9f} "
                f"max={error.max().item():.9f}",
                flush=True,
            )
            raw_rows.append(raw)
            processed_rows.append(processed)
            prompt_ids_rows.append(prompt_ids)
            response_ids_rows.append(response_ids)

    result = {
        "schema_version": 1,
        "source_artifact": str(args.artifact.resolve()),
        "model_path": str(Path(args.model).resolve()),
        "temperature": temperature,
        "prompt_token_ids": prompt_ids_rows,
        "response_token_ids": response_ids_rows,
        "hf_raw_logprobs": raw_rows,
        "hf_processed_logprobs": processed_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
