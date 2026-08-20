#!/usr/bin/env python3
"""Held-out **val** rollout collector for the checkpoint sweep (2026-08-12).

Thin wrapper around ``scripts/collect_student_rollouts.py``.  Everything that
touches the model, the environment, the context contract, the record schema and
the resume/attempt ledger is the *unmodified* collector — this file only swaps
the task-selection gate.

Why a wrapper is needed
-----------------------
``collect_student_rollouts.load_and_validate_task_file`` hard-validates the task
file against ``expected_curriculum()``, whose pool is hardwired to
``AgentGym/agentenv-alfworld/configs/mappings_train.json`` (item_ids 0..2419).
The trainer's held-out validation set is drawn from a *different* split, so any
val task file is rejected by that gate.  ``--skip-live-profile-check`` does not
help: it only skips ``verify_live_task_profile`` (which is itself hardwired to
``split="train"``).

The frozen val curriculum this wrapper reproduces
-------------------------------------------------
Derived from the trainer, not guessed (``ae_ray_trainer._create_dataloader_from_manager``):

  * ``data.val_files: null`` and ``env_type == "alfworld"`` -> the val task list
    comes from ``env_service`` ``/get_env_profile {"split": "val"}``, which
    returns ``env_service/environments/alfworld/alfworld_test.json`` in file
    order: game indices 2420..2619.
  * ``load_tasks_from_environment(..., shuffle=False, max_tasks=128)`` keeps the
    server order and takes the **prefix**: 2420..2547.
  * The val ``FullDataset`` is then mixed by ``OriginalOnlyStrategy(shuffle=True,
    seed=None)``, which falls through to the **global** ``random`` module whose
    state was just reset by ``random.seed(data.seed)`` (=2025) a few lines
    earlier.  So the evaluation order is
    ``ids = list(range(2420, 2548)); random.seed(2025); random.shuffle(ids)``.
  * ``data.validation_shuffle: false`` only pins the DataLoader; it does not undo
    the mixture shuffle.

Verified byte-exact against the trainer's own
``experiments/alfworld/p0_catalyst_af_s0/validation_log/{10..70}.jsonl``
(128 rows each, identical ``task_id`` order).
ordered-newline sha256 = d90efe607c6d63c518968d8ab6a10cb2575d3dff2b4ea6d4e6609ee041680187

Extra flags on top of the base collector
----------------------------------------
``--val-split-count``  size of the val prefix (default 128 = data.max_val_tasks)
``--val-shuffle-seed`` data.seed used for the mixture shuffle (default 2025)
``--shard-index/--shard-count``
    Deterministic round-robin sharding of the ordered task list so N collector
    processes can each drive one TP1 vLLM port.  Sharding never changes *which*
    tasks run, only who runs them; each shard writes its own --output file.

Everything else (``--config``, ``--env-url``, ``--output``, ``--model``,
``--api-base``, ``--rollouts-per-task``, ``--temperature``, ``--top-p``,
``--max-workers``, ``--resume``, ...) is the base collector's own CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.collect_student_rollouts as csr  # noqa: E402  (path bootstrap first)


# The env_service val/dev/test split for ALFWorld is this cached eval index file.
ALFWORLD_EVAL_POOL = (
    PROJECT_ROOT / "env_service/environments/alfworld/alfworld_test.json"
)


def val_curriculum(
    env_name: str, *, count: int, shuffle_seed: int
) -> Dict[str, Any]:
    """Reproduce the trainer's held-out val task order (see module docstring)."""
    if env_name != "alfworld":
        raise RuntimeError(
            f"val curriculum is only derived for alfworld, got {env_name!r}"
        )
    import json

    payload = json.loads(ALFWORLD_EVAL_POOL.read_text(encoding="utf-8"))
    pool = [csr._canonical_task_id(env_name, item["item_id"].split("_")[-1])
            for item in payload]
    if len(pool) != len(set(pool)):
        raise RuntimeError("alfworld eval pool contains duplicates")
    if count <= 0 or count > len(pool):
        raise ValueError(f"invalid val count {count} for pool size {len(pool)}")
    # trainer: prefix first (shuffle=False at load time), THEN mixture shuffle.
    ordered = pool[:count]
    random.Random(shuffle_seed).shuffle(ordered)
    newline_payload = ("\n".join(ordered) + "\n").encode("utf-8")
    sorted_payload = (
        "\n".join(sorted(ordered, key=csr._membership_sort_key(env_name))) + "\n"
    ).encode("utf-8")
    return {
        "environment": env_name,
        "algorithm": "env_val_split_prefix_then_python_random_seed_shuffle",
        "split": "val",
        "val_shuffle_seed": shuffle_seed,
        "pool_count": len(pool),
        "pool_unique_count": len(set(pool)),
        "count": count,
        "source_path": str(ALFWORLD_EVAL_POOL.relative_to(PROJECT_ROOT)),
        "source_sha256": csr.sha256_file(ALFWORLD_EVAL_POOL),
        "ordered_newline_sha256": csr.sha256_bytes(newline_payload),
        "ordered_json_sha256": csr.canonical_json_hash(ordered),
        "sorted_membership_sha256": csr.sha256_bytes(sorted_payload),
        "task_ids": ordered,
    }


def install_val_gate(
    *, count: int, shuffle_seed: int, shard_index: int, shard_count: int
) -> Dict[str, Any]:
    """Monkeypatch the two train-only gates in the base collector."""

    expected = val_curriculum("alfworld", count=count, shuffle_seed=shuffle_seed)
    full_ids: List[str] = list(expected["task_ids"])
    shard_ids = full_ids[shard_index::shard_count]

    def _load_and_validate_task_file(
        path: Path, *, env_name: str, task_seed: int, expected_count: int
    ) -> tuple[List[str], Dict[str, Any]]:
        raw = Path(path).resolve().read_text(encoding="utf-8")
        if not raw.endswith("\n"):
            raise RuntimeError(f"task file must end with a newline: {path}")
        lines = raw.splitlines()
        if any(not line.strip() or line.strip() != line for line in lines):
            raise RuntimeError(f"task file has blank/non-canonical lines: {path}")
        file_ids = [csr._canonical_task_id(env_name, line) for line in lines]
        if file_ids != full_ids:
            raise RuntimeError(
                "task file is not the frozen ALFWorld val curriculum: "
                f"expected_sha256={expected['ordered_newline_sha256']}, "
                f"actual_sha256={csr.sha256_bytes(raw.encode('utf-8'))}"
            )
        if len(file_ids) != expected_count:
            raise RuntimeError(
                f"--expected-task-count must be {len(file_ids)}, got {expected_count}"
            )
        manifest = {k: v for k, v in expected.items() if k != "task_ids"}
        shard_payload = ("\n".join(shard_ids) + "\n").encode("utf-8")
        manifest.update(
            {
                "task_file": str(Path(path).resolve()),
                "task_file_sha256": csr.sha256_bytes(raw.encode("utf-8")),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "shard_task_count": len(shard_ids),
                "shard_ordered_newline_sha256": csr.sha256_bytes(shard_payload),
            }
        )
        return shard_ids, manifest

    def _verify_live_task_profile(
        env_url: str, env_name: str, expected_ids: Sequence[str], task_seed: int
    ) -> None:
        import requests

        response = requests.post(
            f"{env_url.rstrip('/')}/get_env_profile",
            json={"env_type": env_name, "params": {"split": "val"}},
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise RuntimeError(f"live val profile request failed: {body}")
        live = [csr._canonical_task_id(env_name, item) for item in body["data"]]
        if live[:count] != full_ids_unshuffled(count):
            raise RuntimeError(
                "live env val split does not match the frozen eval pool prefix"
            )
        missing = [tid for tid in expected_ids if tid not in set(live)]
        if missing:
            raise RuntimeError(f"live val split is missing task ids: {missing[:8]}")

    def full_ids_unshuffled(n: int) -> List[str]:
        import json

        payload = json.loads(ALFWORLD_EVAL_POOL.read_text(encoding="utf-8"))
        return [
            csr._canonical_task_id("alfworld", item["item_id"].split("_")[-1])
            for item in payload
        ][:n]

    csr.load_and_validate_task_file = _load_and_validate_task_file
    csr.verify_live_task_profile = _verify_live_task_profile
    return {"expected": expected, "shard_ids": shard_ids}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--val-split-count", type=int, default=128)
    extra.add_argument("--val-shuffle-seed", type=int, default=2025)
    extra.add_argument("--shard-index", type=int, default=0)
    extra.add_argument("--shard-count", type=int, default=1)
    ours, rest = extra.parse_known_args(argv)
    if ours.shard_count <= 0 or not (0 <= ours.shard_index < ours.shard_count):
        raise SystemExit("invalid --shard-index/--shard-count")

    info = install_val_gate(
        count=ours.val_split_count,
        shuffle_seed=ours.val_shuffle_seed,
        shard_index=ours.shard_index,
        shard_count=ours.shard_count,
    )
    print(
        f"[eval-collector] val curriculum sha={info['expected']['ordered_newline_sha256']} "
        f"count={ours.val_split_count} shard={ours.shard_index}/{ours.shard_count} "
        f"shard_tasks={len(info['shard_ids'])}",
        flush=True,
    )
    args = csr.parse_args(rest)
    return csr.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
