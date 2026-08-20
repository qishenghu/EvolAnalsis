#!/usr/bin/env python3
"""Merge a verl FSDP actor checkpoint (DTensor shards) into a servable HF dir.

Why this exists
---------------
verl's FSDP checkpoint manager writes ``model_world_size_4_rank_{0..3}.pt``,
each an ``OrderedDict[str, DTensor]`` whose tensors are ``Shard(dim=0)`` over a
4-way ``fsdp`` mesh, in **float32**, under *text-model* key names
(``model.embed_tokens.weight``, ``model.layers.N.*``, ``model.norm.weight``,
``lm_head.weight`` — 427 keys).  The sidecar ``config.json`` it drops next to
them is the **text-only** config (``model_type: qwen3_5_text``) and carries no
``architectures`` field.

vLLM 0.21.0 (the pinned ``vllm2`` runtime) registers Qwen3.5 only as the
multimodal ``Qwen3_5ForConditionalGeneration`` — there is **no** text-only
``qwen3_5_text`` entry in its model registry.  Serving the raw sidecar config
therefore cannot work.  What production actually serves is the base
``/projects_vol/gp_wangwy/models/Qwen3.5-4B`` dir with the trained tensors
hot-swapped in by ``external/verl_t5x_patches/duet_vllm_worker_ext.py``, which
re-prefixes ``model.*`` -> ``model.language_model.*``.

So this merger reconstructs exactly that served object:

  * base dir supplies config/tokenizer/vision(297)/MTP(15) tensors verbatim;
  * the 426 trained language-model tensors are gathered from the four rank
    shards (``cat`` along dim 0, narrowed to the DTensor's full shape), cast to
    bfloat16, and written under ``model.language_model.*``;
  * ``lm_head.weight`` is dropped after asserting it still equals
    ``model.embed_tokens.weight`` (``tie_word_embeddings: true``); the base
    checkpoint has no ``lm_head`` entry either.

Output layout mirrors the base dir byte-for-byte except for the two safetensors
shards (same file names, same tensor->shard assignment as the base index).

Usage
-----
  python scripts/merge_fsdp_ckpt_to_hf.py \
      --ckpt-dir  .../global_step_30/actor \
      --base-model /projects_vol/gp_wangwy/models/Qwen3.5-4B \
      --out-dir   $SCRATCH/ckpt_hf/p0_catalyst_af_s0/step_30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Files copied verbatim from the base model dir (everything the rollout-server
# contract hashes, minus the weight shards we rewrite).
COPY_FROM_BASE = [
    "config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


def gather_actor_state(ckpt_dir: Path, world_size: int) -> Dict[str, torch.Tensor]:
    shards = []
    for rank in range(world_size):
        path = ckpt_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(torch.load(path, map_location="cpu", mmap=True, weights_only=False))
    keys = list(shards[0].keys())
    for rank, shard in enumerate(shards[1:], start=1):
        if list(shard.keys()) != keys:
            raise RuntimeError(f"rank {rank} key set differs from rank 0")

    merged: Dict[str, torch.Tensor] = {}
    for key in keys:
        parts: List[torch.Tensor] = []
        full_shape = None
        for shard in shards:
            value = shard[key]
            if hasattr(value, "_local_tensor"):
                placements = getattr(value, "placements", None)
                if placements is None or len(placements) != 1 or not placements[0].is_shard(0):
                    raise RuntimeError(f"{key}: unsupported placements {placements}")
                if full_shape is None:
                    full_shape = tuple(value.shape)
                parts.append(value._local_tensor)
            else:  # replicated plain tensor
                if full_shape is None:
                    full_shape = tuple(value.shape)
                parts.append(value)
                break
        tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if full_shape is not None and tuple(tensor.shape) != full_shape:
            if tensor.shape[0] < full_shape[0]:
                raise RuntimeError(
                    f"{key}: gathered {tuple(tensor.shape)} smaller than {full_shape}"
                )
            tensor = tensor.narrow(0, 0, full_shape[0]).contiguous()
        merged[key] = tensor
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=Path,
                    default=Path("/projects_vol/gp_wangwy/models/Qwen3.5-4B"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    done_marker = out_dir / "MERGE_DONE.json"
    if done_marker.is_file() and not args.force:
        print(f"[merge] already done: {done_marker}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    print(f"[merge] gathering {args.ckpt_dir}", flush=True)
    actor = gather_actor_state(args.ckpt_dir, args.world_size)
    print(f"[merge] gathered {len(actor)} tensors in {time.time()-t0:.0f}s", flush=True)

    embed = actor.get("model.embed_tokens.weight")
    lm_head = actor.pop("lm_head.weight", None)
    if lm_head is not None:
        if embed is None or lm_head.shape != embed.shape:
            raise RuntimeError("lm_head present but embed_tokens missing/mismatched")
        if not torch.equal(lm_head, embed):
            raise RuntimeError(
                "lm_head.weight != embed_tokens.weight although the config declares "
                "tie_word_embeddings=true; refusing to silently drop it"
            )
        print("[merge] lm_head.weight == embed_tokens.weight (tied) -> dropped")

    renamed: Dict[str, torch.Tensor] = {}
    for key, value in actor.items():
        if not key.startswith("model."):
            raise RuntimeError(f"unexpected non-'model.' key in actor state: {key}")
        renamed["model.language_model." + key[len("model."):]] = value
    del actor

    index_path = args.base_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index["weight_map"]

    missing = [k for k in renamed if k not in weight_map]
    if missing:
        raise RuntimeError(f"{len(missing)} merged keys absent from base index: {missing[:5]}")
    base_only = [k for k in weight_map if k not in renamed]
    print(f"[merge] replacing {len(renamed)} language tensors; "
          f"copying {len(base_only)} base tensors (vision/MTP)")

    shard_files = sorted(set(weight_map.values()))
    total_bytes = 0
    for shard_file in shard_files:
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(args.base_model / shard_file), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                if key in renamed:
                    tensors[key] = renamed.pop(key).to(dtype).contiguous()
                else:
                    tensors[key] = fh.get_tensor(key)
        save_file(tensors, str(out_dir / shard_file), metadata={"format": "pt"})
        size = (out_dir / shard_file).stat().st_size
        total_bytes += size
        print(f"[merge] wrote {shard_file} ({size/2**30:.2f} GiB, {len(tensors)} tensors)",
              flush=True)
        del tensors
    if renamed:
        raise RuntimeError(f"{len(renamed)} merged tensors were never written: "
                           f"{list(renamed)[:5]}")

    shutil.copyfile(index_path, out_dir / index_path.name)
    for name in COPY_FROM_BASE:
        src = args.base_model / name
        if src.is_file():
            shutil.copyfile(src, out_dir / name)
        else:
            print(f"[merge] WARNING base file missing, skipped: {name}")

    manifest = {
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ckpt_dir": str(args.ckpt_dir.resolve()),
        "base_model": str(args.base_model.resolve()),
        "world_size": args.world_size,
        "dtype": args.dtype,
        "rank_shard_sha256_first64mb": {
            f"rank_{r}": sha256_file(
                args.ckpt_dir / f"model_world_size_{args.world_size}_rank_{r}.pt",
                limit=64 << 20)
            for r in range(args.world_size)
        },
        "output_bytes": total_bytes,
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("text-only actor tensors re-prefixed model.* -> "
                 "model.language_model.* to match the multimodal "
                 "Qwen3_5ForConditionalGeneration checkpoint vLLM 0.21 expects "
                 "(same rename the production duet_vllm_worker_ext performs)"),
    }
    done_marker.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[merge] DONE {out_dir} in {manifest['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
