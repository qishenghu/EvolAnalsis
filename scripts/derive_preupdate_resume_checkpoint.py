#!/usr/bin/env python3
"""Derive the audited pre-update resume checkpoint for the Qwen3.5 v4 run.

The identity gate at trainer ``global_step_2`` fired before its optimizer
update.  Consequently its actor/optimizer/RNG state is the state after step 1,
while its dataloader state has already yielded the second batch.  This tool
creates an independent recovery root that names the checkpoint
``global_step_1`` and rewinds only those dataloader counters.

This is deliberately a narrow, fail-closed migration rather than a general
checkpoint editor.  The source checkpoint, tracker, actor layout, and
dataloader state must exactly match the observed v4 failure artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import uuid
from typing import Any

import torch


SOURCE_STEP = 2
TARGET_STEP = 1
SOURCE_CHECKPOINT_NAME = f"global_step_{SOURCE_STEP}"
TARGET_CHECKPOINT_NAME = f"global_step_{TARGET_STEP}"
TRACKER_NAME = "latest_checkpointed_iteration.txt"
MANIFEST_NAME = "derivation_manifest.json"

_STATIC_ACTOR_FILES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
_SHARD_KINDS = ("extra_state", "model", "optim")
EXPECTED_ACTOR_FILES = _STATIC_ACTOR_FILES | {
    f"{kind}_world_size_4_rank_{rank}.pt"
    for kind in _SHARD_KINDS
    for rank in range(4)
}

EXPECTED_DATA_STATE = {
    "_index_sampler_state": None,
    "_sampler_iter_state": {"samples_yielded": 16},
    "_sampler_iter_yielded": 2,
    "_num_yielded": 2,
    "_IterableDataset_len_called": None,
    "_shared_seed": None,
    "fetcher_state": None,
    "dataset_state": None,
    "_iterator_finished": False,
}
TARGET_DATA_STATE = {
    **EXPECTED_DATA_STATE,
    "_sampler_iter_state": {"samples_yielded": 8},
    "_sampler_iter_yielded": 1,
    "_num_yielded": 1,
}


class DerivationError(RuntimeError):
    """A source or derived-checkpoint invariant was not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_plain_regular_file(path: Path, description: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise DerivationError(f"missing {description}: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise DerivationError(
            f"{description} must be a regular non-symlink file: {path}"
        )
    return info


def _load_data_state(path: Path) -> dict[str, Any]:
    _assert_plain_regular_file(path, "data.pt")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise DerivationError(f"cannot safely load {path}: {error}") from error
    if not isinstance(state, dict):
        raise DerivationError(f"data.pt must contain a dict, got {type(state)!r}")
    return state


def _assert_exact_state(
    actual: dict[str, Any], expected: dict[str, Any], description: str
) -> None:
    # Equality is sufficient here because this migration accepts only the
    # primitive, fully enumerated schema above.  Key order is also checked so
    # that the derived payload preserves the source's field ordering.
    if list(actual) != list(expected):
        raise DerivationError(
            f"{description} keys/order mismatch: expected {list(expected)!r}, "
            f"got {list(actual)!r}"
        )
    if actual != expected:
        raise DerivationError(
            f"{description} values mismatch: expected {expected!r}, got {actual!r}"
        )

    def assert_types(value: Any, template: Any, field: str) -> None:
        if type(value) is not type(template):
            raise DerivationError(
                f"{description} {field} type mismatch: expected "
                f"{type(template).__name__}, got {type(value).__name__}"
            )
        if isinstance(template, dict):
            for child_key in template:
                assert_types(
                    value[child_key], template[child_key], f"{field}.{child_key}"
                )

    assert_types(actual, expected, "root")


def _actor_inventory(actor_dir: Path) -> dict[str, os.stat_result]:
    if not actor_dir.is_dir() or actor_dir.is_symlink():
        raise DerivationError(
            f"actor must be a real directory (not a symlink): {actor_dir}"
        )
    entries = list(actor_dir.iterdir())
    names = {entry.name for entry in entries}
    if names != EXPECTED_ACTOR_FILES:
        missing = sorted(EXPECTED_ACTOR_FILES - names)
        extra = sorted(names - EXPECTED_ACTOR_FILES)
        raise DerivationError(
            f"actor layout mismatch; missing={missing!r}, extra={extra!r}"
        )
    return {
        entry.name: _assert_plain_regular_file(entry, f"actor/{entry.name}")
        for entry in entries
    }


def _validate_source(source_checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_checkpoint.name != SOURCE_CHECKPOINT_NAME:
        raise DerivationError(
            f"source must be named {SOURCE_CHECKPOINT_NAME!r}: {source_checkpoint}"
        )
    if not source_checkpoint.is_dir() or source_checkpoint.is_symlink():
        raise DerivationError(
            f"source checkpoint must be a real directory: {source_checkpoint}"
        )
    source_entries = {entry.name for entry in source_checkpoint.iterdir()}
    if source_entries != {"actor", "data.pt"}:
        raise DerivationError(
            "source checkpoint layout mismatch; expected only actor/ and data.pt, "
            f"got {sorted(source_entries)!r}"
        )

    tracker = source_checkpoint.parent / TRACKER_NAME
    _assert_plain_regular_file(tracker, "source latest tracker")
    if tracker.read_bytes() != b"2":
        raise DerivationError(
            f"source latest tracker must contain exactly b'2': {tracker}"
        )

    state = _load_data_state(source_checkpoint / "data.pt")
    _assert_exact_state(state, EXPECTED_DATA_STATE, "source data.pt")

    inventory = _actor_inventory(source_checkpoint / "actor")
    actor_records: dict[str, Any] = {}
    for name in sorted(inventory):
        info = inventory[name]
        actor_records[name] = {
            "size": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "source_device": info.st_dev,
            "source_inode": info.st_ino,
            "sha256": _sha256(source_checkpoint / "actor" / name),
        }
    return state, actor_records


def _link_or_copy(source: Path, target: Path) -> str:
    """Prefer a hard link; copy only when the filesystem refuses linking."""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as link_error:
        try:
            shutil.copy2(source, target, follow_symlinks=False)
        except Exception as copy_error:
            raise DerivationError(
                f"cannot hard-link or copy {source} to {target}; "
                f"link error={link_error}; copy error={copy_error}"
            ) from copy_error
        return "copy"


def _verify_actor(
    source_actor: Path,
    target_actor: Path,
    source_records: dict[str, Any],
    methods: dict[str, str],
) -> list[dict[str, Any]]:
    target_inventory = _actor_inventory(target_actor)
    if set(target_inventory) != set(source_records):
        raise DerivationError("target actor inventory differs from source inventory")

    verified = []
    for name in sorted(source_records):
        before = source_records[name]
        source_now = _assert_plain_regular_file(
            source_actor / name, f"source actor/{name} after derivation"
        )
        target_now = target_inventory[name]
        if (source_now.st_dev, source_now.st_ino) != (
            before["source_device"],
            before["source_inode"],
        ):
            raise DerivationError(f"source actor/{name} inode changed during derivation")
        if source_now.st_size != before["size"]:
            raise DerivationError(f"source actor/{name} size changed during derivation")
        if stat.S_IMODE(source_now.st_mode) != before["mode"]:
            raise DerivationError(f"source actor/{name} mode changed during derivation")
        if target_now.st_size != before["size"]:
            raise DerivationError(f"target actor/{name} size differs from source")
        if stat.S_IMODE(target_now.st_mode) != before["mode"]:
            raise DerivationError(f"target actor/{name} mode differs from source")

        same_inode = (
            source_now.st_dev == target_now.st_dev
            and source_now.st_ino == target_now.st_ino
        )
        method = methods[name]
        if method == "hardlink" and not same_inode:
            raise DerivationError(f"actor/{name} was not actually hard-linked")
        if method == "copy" and same_inode:
            raise DerivationError(f"actor/{name} copy unexpectedly shares source inode")
        # A hard link names the exact same inode, so the source SHA-256 already
        # authenticates the target bytes; hashing that ~53 GB inode a second
        # time would add no evidence.  A copy has distinct storage and must be
        # hashed independently.
        target_hash = (
            before["sha256"]
            if same_inode
            else _sha256(target_actor / name)
        )
        if target_hash != before["sha256"]:
            raise DerivationError(f"target actor/{name} SHA-256 differs from source")

        verified.append(
            {
                "path": name,
                "method": method,
                "size": before["size"],
                "mode": oct(before["mode"]),
                "sha256": target_hash,
                "source_device": source_now.st_dev,
                "source_inode": source_now.st_ino,
                "target_device": target_now.st_dev,
                "target_inode": target_now.st_ino,
                "same_inode": same_inode,
            }
        )
    return verified


def derive_checkpoint(source_checkpoint: Path, recovery_root: Path) -> dict[str, Any]:
    """Create and atomically publish a verified ``recovery/global_step_1``."""
    source_checkpoint = source_checkpoint.expanduser().resolve(strict=True)
    recovery_root = recovery_root.expanduser().absolute()
    if recovery_root.exists() or recovery_root.is_symlink():
        raise DerivationError(f"target recovery root already exists: {recovery_root}")
    if recovery_root == source_checkpoint or recovery_root in source_checkpoint.parents:
        raise DerivationError("target recovery root may not contain the source")
    if source_checkpoint in recovery_root.parents:
        raise DerivationError("target recovery root may not be inside the source checkpoint")

    source_state, source_records = _validate_source(source_checkpoint)
    source_data_hash_before = _sha256(source_checkpoint / "data.pt")
    source_tracker = source_checkpoint.parent / TRACKER_NAME
    source_tracker_hash_before = _sha256(source_tracker)

    recovery_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = recovery_root.with_name(
        f".{recovery_root.name}.tmp-{uuid.uuid4().hex}"
    )
    if staging_root.exists() or staging_root.is_symlink():
        raise DerivationError(f"refusing to reuse staging path: {staging_root}")

    try:
        target_checkpoint = staging_root / TARGET_CHECKPOINT_NAME
        target_actor = target_checkpoint / "actor"
        target_actor.mkdir(parents=True, exist_ok=False)

        methods: dict[str, str] = {}
        for name in sorted(source_records):
            methods[name] = _link_or_copy(
                source_checkpoint / "actor" / name, target_actor / name
            )

        target_state = copy.deepcopy(source_state)
        target_state["_sampler_iter_state"]["samples_yielded"] = 8
        target_state["_sampler_iter_yielded"] = 1
        target_state["_num_yielded"] = 1
        _assert_exact_state(target_state, TARGET_DATA_STATE, "derived data.pt")
        torch.save(target_state, target_checkpoint / "data.pt")

        (staging_root / TRACKER_NAME).write_bytes(b"1")
        actor_files = _verify_actor(
            source_checkpoint / "actor", target_actor, source_records, methods
        )

        loaded_target_state = _load_data_state(target_checkpoint / "data.pt")
        _assert_exact_state(
            loaded_target_state, TARGET_DATA_STATE, "reloaded derived data.pt"
        )
        if (staging_root / TRACKER_NAME).read_bytes() != b"1":
            raise DerivationError("derived latest tracker is not exactly b'1'")
        if _sha256(source_checkpoint / "data.pt") != source_data_hash_before:
            raise DerivationError("source data.pt changed during derivation")
        if _sha256(source_tracker) != source_tracker_hash_before:
            raise DerivationError("source latest tracker changed during derivation")

        hardlinks = sum(item["method"] == "hardlink" for item in actor_files)
        copies = len(actor_files) - hardlinks
        manifest = {
            "schema_version": 1,
            "source_checkpoint": str(source_checkpoint),
            "source_step": SOURCE_STEP,
            "target_checkpoint": str(recovery_root / TARGET_CHECKPOINT_NAME),
            "target_step": TARGET_STEP,
            "source_data_sha256": source_data_hash_before,
            "target_data_sha256": _sha256(target_checkpoint / "data.pt"),
            "source_tracker_sha256": source_tracker_hash_before,
            "target_tracker_sha256": _sha256(staging_root / TRACKER_NAME),
            "state_changes": {
                "_sampler_iter_state.samples_yielded": {"from": 16, "to": 8},
                "_sampler_iter_yielded": {"from": 2, "to": 1},
                "_num_yielded": {"from": 2, "to": 1},
            },
            "actor_file_count": len(actor_files),
            "actor_hardlink_count": hardlinks,
            "actor_copy_count": copies,
            "actor_files": actor_files,
        }
        manifest_path = staging_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Publication happens only after every content, layout, inode, and state
        # check has succeeded.  The destination was checked absent above and
        # os.rename is atomic within this filesystem.
        try:
            staging_root.rename(recovery_root)
        except FileExistsError as error:
            raise DerivationError(
                f"target recovery root appeared during derivation: {recovery_root}"
            ) from error
        return manifest
    except Exception:
        if staging_root.exists() and staging_root != recovery_root:
            shutil.rmtree(staging_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        type=Path,
        help="exact v4 global_step_2 checkpoint directory",
    )
    parser.add_argument(
        "--recovery-root",
        required=True,
        type=Path,
        help="new recovery root; it must not already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = derive_checkpoint(args.source_checkpoint, args.recovery_root)
    except (DerivationError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
