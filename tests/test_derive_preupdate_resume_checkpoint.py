"""Focused tests for the audited v4 -> v5 recovery checkpoint derivation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.derive_preupdate_resume_checkpoint import (
    DerivationError,
    EXPECTED_ACTOR_FILES,
    EXPECTED_DATA_STATE,
    MANIFEST_NAME,
    TARGET_DATA_STATE,
    derive_checkpoint,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source(tmp_path: Path) -> Path:
    root = tmp_path / "v4"
    source = root / "global_step_2"
    actor = source / "actor"
    actor.mkdir(parents=True)
    for index, name in enumerate(sorted(EXPECTED_ACTOR_FILES)):
        (actor / name).write_bytes(f"actor-file-{index}-{name}".encode())
    torch.save(EXPECTED_DATA_STATE, source / "data.pt")
    (root / "latest_checkpointed_iteration.txt").write_bytes(b"2")
    return source


def test_derives_exact_independent_recovery_checkpoint(tmp_path):
    source = _make_source(tmp_path)
    source_hashes = {
        path.name: _hash(path) for path in (source / "actor").iterdir()
    }
    source_data_hash = _hash(source / "data.pt")
    recovery = tmp_path / "v5" / "recovery"

    manifest = derive_checkpoint(source, recovery)

    target = recovery / "global_step_1"
    assert target.is_dir()
    assert (recovery / "latest_checkpointed_iteration.txt").read_bytes() == b"1"
    assert torch.load(
        target / "data.pt", map_location="cpu", weights_only=True
    ) == TARGET_DATA_STATE
    assert torch.load(
        source / "data.pt", map_location="cpu", weights_only=True
    ) == EXPECTED_DATA_STATE
    assert _hash(source / "data.pt") == source_data_hash
    assert (source.parent / "latest_checkpointed_iteration.txt").read_bytes() == b"2"

    assert {path.name for path in (target / "actor").iterdir()} == set(
        source_hashes
    )
    for name, expected_hash in source_hashes.items():
        source_stat = (source / "actor" / name).stat()
        target_stat = (target / "actor" / name).stat()
        assert (source_stat.st_dev, source_stat.st_ino) == (
            target_stat.st_dev,
            target_stat.st_ino,
        )
        assert _hash(target / "actor" / name) == expected_hash

    assert manifest["actor_file_count"] == len(EXPECTED_ACTOR_FILES)
    assert manifest["actor_hardlink_count"] == len(EXPECTED_ACTOR_FILES)
    assert manifest["actor_copy_count"] == 0
    assert (recovery / MANIFEST_NAME).is_file()


def test_rejects_existing_target_without_touching_source(tmp_path):
    source = _make_source(tmp_path)
    source_data_hash = _hash(source / "data.pt")
    recovery = tmp_path / "v5" / "recovery"
    recovery.mkdir(parents=True)

    with pytest.raises(DerivationError, match="already exists"):
        derive_checkpoint(source, recovery)

    assert _hash(source / "data.pt") == source_data_hash
    assert list(recovery.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state["_sampler_iter_state"].update(samples_yielded=15), "values mismatch"),
        (lambda state: state.update(_num_yielded=3), "values mismatch"),
        (lambda state: state.update(unexpected=True), "keys/order mismatch"),
        (lambda state: state.update(_iterator_finished=0), "type mismatch"),
    ],
)
def test_rejects_unexpected_source_data_and_publishes_nothing(
    tmp_path, mutation, message
):
    source = _make_source(tmp_path)
    state = torch.load(source / "data.pt", map_location="cpu", weights_only=True)
    mutation(state)
    torch.save(state, source / "data.pt")
    recovery = tmp_path / "v5" / "recovery"

    with pytest.raises(DerivationError, match=message):
        derive_checkpoint(source, recovery)

    assert not recovery.exists()


def test_rejects_incomplete_actor_layout_and_publishes_nothing(tmp_path):
    source = _make_source(tmp_path)
    (source / "actor" / "model_world_size_4_rank_3.pt").unlink()
    recovery = tmp_path / "v5" / "recovery"

    with pytest.raises(DerivationError, match="actor layout mismatch"):
        derive_checkpoint(source, recovery)

    assert not recovery.exists()


def test_rejects_wrong_source_tracker(tmp_path):
    source = _make_source(tmp_path)
    (source.parent / "latest_checkpointed_iteration.txt").write_bytes(b"1")

    with pytest.raises(DerivationError, match="must contain exactly b'2'"):
        derive_checkpoint(source, tmp_path / "v5" / "recovery")
