import json
import random
from pathlib import Path

import pytest

from scripts import materialize_teacher_curricula as curricula


def _write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def _student_config() -> str:
    return """\
defaults:
  - _self_

data:
  train_batch_size: 16
  max_train_tasks: 1600
  shuffle: true
  seed: 2025
  task_seed: 2026

trainer:
  total_training_steps: 100
"""


def _make_repo(tmp_path: Path) -> Path:
    alf_source = tmp_path / curricula.BENCHMARKS[0].source_relative_path
    ws_source = tmp_path / curricula.BENCHMARKS[1].source_relative_path
    _write(
        alf_source,
        json.dumps([{"item_id": index} for index in range(1700)]) + "\n",
    )
    _write(
        ws_source,
        json.dumps([{"item_id": f"webshop_{index}"} for index in range(1700)])
        + "\n",
    )

    # The old files intentionally use a different membership contract.  Their
    # provenance must be recorded, never silently treated as today's prefix.
    legacy_ids = [str(index) for index in range(curricula.LEGACY_COUNT)]
    for spec in curricula.BENCHMARKS:
        _write(tmp_path / spec.legacy_relative_path, "\n".join(legacy_ids) + "\n")
        _write(tmp_path / spec.student_config_relative_path, _student_config())
    return tmp_path


def _expected_prefix() -> list[str]:
    values = [str(index) for index in range(1700)]
    previous_state = random.getstate()
    try:
        random.seed(curricula.TASK_SEED)
        random.shuffle(values)
    finally:
        random.setstate(previous_state)
    return values[: curricula.TARGET_COUNT]


def test_materialize_and_verify_exact_order_and_manifest(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    manifest = curricula.materialize(repo)
    expected = _expected_prefix()

    for spec in curricula.BENCHMARKS:
        output = repo / spec.output_relative_path
        assert output.read_bytes() == ("\n".join(expected) + "\n").encode()
        entry = manifest["benchmarks"][spec.name]
        assert entry["curriculum"]["count"] == 1600
        assert entry["curriculum"]["unique_count"] == 1600
        assert entry["source"]["pool_count"] == 1700
        assert entry["student_config"]["verified_data_fields"] == {
            "max_train_tasks": 1600,
            "shuffle": True,
            "task_seed": 2026,
        }
        assert entry["legacy_800_audit"]["status"] == "legacy_not_current_curriculum"
        assert entry["legacy_800_audit"]["ordered_prefix_match"] is False
        assert entry["legacy_800_audit"]["prefix_membership_match"] is False

    assert curricula.verify(repo) == manifest


def test_verify_rejects_legacy_file_substituted_for_current(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    curricula.materialize(repo)
    spec = curricula.BENCHMARKS[0]
    (repo / spec.output_relative_path).write_bytes(
        (repo / spec.legacy_relative_path).read_bytes()
    )

    with pytest.raises(curricula.ContractError, match="has 800 rows, expected 1600"):
        curricula.verify(repo)


def test_verify_rejects_student_task_seed_drift(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    curricula.materialize(repo)
    spec = curricula.BENCHMARKS[1]
    config = repo / spec.student_config_relative_path
    config.write_text(
        config.read_text(encoding="utf-8").replace("task_seed: 2026", "task_seed: 2027"),
        encoding="utf-8",
    )

    with pytest.raises(curricula.ContractError, match="data.task_seed must be 2026"):
        curricula.verify(repo)


def test_verify_rejects_source_drift_even_when_outputs_are_untouched(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    curricula.materialize(repo)
    spec = curricula.BENCHMARKS[0]
    source = repo / spec.source_relative_path
    records = json.loads(source.read_text(encoding="utf-8"))
    records.reverse()
    source.write_text(json.dumps(records) + "\n", encoding="utf-8")

    with pytest.raises(curricula.ContractError, match="does not exactly equal"):
        curricula.verify(repo)


def test_verify_rejects_stale_or_tampered_manifest(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    curricula.materialize(repo)
    manifest_path = repo / curricula.MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmarks"]["alfworld"]["sampling"]["task_seed"] = 2027
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(curricula.ContractError, match="manifest is stale"):
        curricula.verify(repo)


def test_webshop_loader_rejects_noncanonical_numeric_id() -> None:
    bad = json.dumps([{"item_id": "webshop_0007"}]).encode()
    with pytest.raises(curricula.ContractError, match="invalid item_id"):
        curricula._load_webshop_pool(bad, Path("webshop_train.json"))
