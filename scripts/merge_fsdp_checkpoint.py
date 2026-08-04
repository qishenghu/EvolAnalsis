#!/usr/bin/env python3
"""Merge FSDP sharded checkpoint into HuggingFace format.

Usage:
    python scripts/merge_fsdp_checkpoint.py \
        --ckpt_dir checkpoints/agentevolver/alfworld_qwen1.5b_sft/global_step_50/actor \
        --base_model /data/shared_models/Qwen2.5-1.5B-Instruct \
        --output_dir checkpoints/agentevolver/alfworld_qwen1.5b_sft/global_step_50/actor_hf
"""
import argparse
import os
import torch
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from accelerate import init_empty_weights


def merge_fsdp_checkpoint(ckpt_dir: str, base_model: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Detect world_size from files
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_world_size_")]
    if not files:
        raise FileNotFoundError(f"No FSDP shard files in {ckpt_dir}")
    # filename: model_world_size_4_rank_0.pt → split by "_" → ['model', 'world', 'size', '4', 'rank', '0.pt']
    world_size = int(files[0].split("_")[3])
    print(f"Detected world_size={world_size}")

    # Load reference model shapes
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    with init_empty_weights():
        ref_model = AutoModelForCausalLM.from_config(config)
    ref_sd = ref_model.state_dict()

    # Load all shards
    shards = []
    for rank in range(world_size):
        path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        s = torch.load(path, map_location="cpu", weights_only=False)
        sd = OrderedDict()
        for k, v in s.items():
            sd[k] = v._local_tensor.clone() if hasattr(v, "_local_tensor") else v.clone()
        shards.append(sd)
        del s
    print(f"Loaded {world_size} shards")

    # Merge by concatenating along sharded dimension
    merged = OrderedDict()
    for k in shards[0].keys():
        parts = [shards[r][k] for r in range(world_size)]
        expected = ref_sd[k].shape

        if parts[0].shape == expected:
            merged[k] = parts[0]
        else:
            for dim in range(len(expected)):
                if parts[0].shape[dim] * world_size == expected[dim]:
                    merged[k] = torch.cat(parts, dim=dim)
                    break
            else:
                merged[k] = torch.cat(parts, dim=0)

    # Verify
    for k in ref_sd:
        assert merged[k].shape == ref_sd[k].shape, \
            f"Shape mismatch for {k}: expected {ref_sd[k].shape}, got {merged[k].shape}"
    print("All shapes verified")

    # Save
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.load_state_dict(merged, strict=True, assign=True)
    model.save_pretrained(output_dir)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(output_dir)
    print(f"Saved HF checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    merge_fsdp_checkpoint(args.ckpt_dir, args.base_model, args.output_dir)
