#!/usr/bin/env python3
"""Materialize and audit the fixed 1,600-task teacher curricula.

This tool intentionally uses only the Python standard library.  It mirrors the
student task selection contract exactly:

    random.seed(2026)
    random.shuffle(full_train_pool)
    curriculum = full_train_pool[:1600]

The canonical pools and ID conversions are the same ones implemented by
``AlfworldEnv.get_query_list("train")`` and
``WebshopEnv.get_query_list("train")``.  Verification is fail-closed: it
recomputes both curricula from the source bytes, checks the batch-16 student
configs, checks exact output bytes (including the final newline), and requires
the committed manifest to equal the newly derived manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


TASK_SEED = 2026
TARGET_COUNT = 1600
LEGACY_COUNT = 800
ALGORITHM = (
    "python-stdlib-random-v1: random.seed(2026); "
    "random.shuffle(pool); pool[:1600]"
)
MANIFEST_RELATIVE_PATH = Path(
    "data/teacher_curricula_train1600_seed2026.manifest.json"
)


class ContractError(RuntimeError):
    """Raised when any curriculum provenance or compatibility check fails."""


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source_relative_path: Path
    output_relative_path: Path
    legacy_relative_path: Path
    student_config_relative_path: Path
    id_semantics: str


BENCHMARKS: Tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="alfworld",
        source_relative_path=Path(
            "AgentGym/agentenv-alfworld/configs/mappings_train.json"
        ),
        output_relative_path=Path(
            "data/alfworld/task_ids_train1600_seed2026.txt"
        ),
        legacy_relative_path=Path("data/alfworld/task_ids_800_seed2026.txt"),
        student_config_relative_path=Path(
            "config/duet_paper_experiments_configs/iclr2027/alfworld/"
            "alfworld_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml"
        ),
        id_semantics="decimal string of mappings_train.json item_id",
    ),
    BenchmarkSpec(
        name="webshop",
        source_relative_path=Path(
            "env_service/environments/webshop/webshop_train.json"
        ),
        output_relative_path=Path(
            "data/webshop/task_ids_train1600_seed2026.txt"
        ),
        legacy_relative_path=Path("data/webshop/task_ids_800_seed2026.txt"),
        student_config_relative_path=Path(
            "config/duet_paper_experiments_configs/iclr2027/webshop/"
            "webshop_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml"
        ),
        id_semantics="decimal session ID parsed from webshop_<integer> item_id",
    ),
)


_CANONICAL_DECIMAL_RE = re.compile(r"0|[1-9][0-9]*")
_WEBSHOP_ITEM_RE = re.compile(r"webshop_(0|[1-9][0-9]*)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ordered_newline_bytes(task_ids: Sequence[str]) -> bytes:
    return ("\n".join(task_ids) + "\n").encode("utf-8")


def _read_required_bytes(path: Path, label: str) -> bytes:
    _require(path.is_file(), f"missing {label}: {path}")
    return path.read_bytes()


def _decode_json_list(source_bytes: bytes, path: Path) -> List[Any]:
    try:
        decoded = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"source is not UTF-8: {path}: {exc}") from exc
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ContractError(f"source is not valid JSON: {path}: {exc}") from exc
    _require(isinstance(value, list), f"source root must be a list: {path}")
    return value


def _load_alfworld_pool(source_bytes: bytes, path: Path) -> List[str]:
    records = _decode_json_list(source_bytes, path)
    result: List[str] = []
    for index, record in enumerate(records):
        _require(
            isinstance(record, dict),
            f"ALFWorld source record {index} is not an object",
        )
        item_id = record.get("item_id")
        _require(
            isinstance(item_id, int) and not isinstance(item_id, bool),
            f"ALFWorld source record {index} has non-integer item_id: {item_id!r}",
        )
        _require(item_id >= 0, f"ALFWorld item_id must be non-negative: {item_id}")
        result.append(str(item_id))
    return result


def _load_webshop_pool(source_bytes: bytes, path: Path) -> List[str]:
    records = _decode_json_list(source_bytes, path)
    result: List[str] = []
    for index, record in enumerate(records):
        _require(
            isinstance(record, dict),
            f"WebShop source record {index} is not an object",
        )
        item_id = record.get("item_id")
        _require(
            isinstance(item_id, str),
            f"WebShop source record {index} has non-string item_id: {item_id!r}",
        )
        match = _WEBSHOP_ITEM_RE.fullmatch(item_id)
        _require(
            match is not None,
            f"WebShop source record {index} has invalid item_id: {item_id!r}",
        )
        # WebshopEnv._parse_session_id returns the numeric component as an int;
        # converting it back to str reproduces get_query_list's public ID form.
        result.append(str(int(match.group(1))))
    return result


def _load_pool(spec: BenchmarkSpec, repo_root: Path) -> Tuple[List[str], bytes]:
    path = repo_root / spec.source_relative_path
    source_bytes = _read_required_bytes(path, f"{spec.name} canonical source")
    if spec.name == "alfworld":
        pool = _load_alfworld_pool(source_bytes, path)
    elif spec.name == "webshop":
        pool = _load_webshop_pool(source_bytes, path)
    else:  # Defensive: adding a benchmark requires choosing explicit ID semantics.
        raise ContractError(f"unsupported benchmark: {spec.name}")

    _require(len(pool) >= TARGET_COUNT, f"{spec.name} pool has fewer than 1600 tasks")
    _require(
        len(set(pool)) == len(pool),
        f"{spec.name} canonical source contains duplicate numeric task IDs",
    )
    return pool, source_bytes


def _sample_curriculum(pool: Sequence[str]) -> List[str]:
    shuffled = list(pool)
    # Use the module-level API deliberately: this is the exact API called by
    # TaskManager.load_tasks_from_environment.  Restore state so importing this
    # audit helper does not perturb an embedding process's RNG.
    previous_state = random.getstate()
    try:
        random.seed(TASK_SEED)
        random.shuffle(shuffled)
    finally:
        random.setstate(previous_state)
    return shuffled[:TARGET_COUNT]


def _parse_task_id_bytes(data: bytes, path: Path, expected_count: int) -> List[str]:
    _require(data.endswith(b"\n"), f"task file must end with one newline: {path}")
    _require(b"\r" not in data, f"task file must use LF rather than CRLF: {path}")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ContractError(f"task file must be ASCII: {path}: {exc}") from exc
    task_ids = text[:-1].split("\n")
    _require(
        len(task_ids) == expected_count,
        f"task file {path} has {len(task_ids)} rows, expected {expected_count}",
    )
    for index, task_id in enumerate(task_ids):
        _require(
            _CANONICAL_DECIMAL_RE.fullmatch(task_id) is not None,
            f"task file {path} row {index + 1} is not a canonical numeric ID: {task_id!r}",
        )
    _require(
        len(set(task_ids)) == expected_count,
        f"task file {path} must contain {expected_count} unique IDs",
    )
    return task_ids


def _read_data_section_scalars(config_path: Path) -> Mapping[str, str]:
    """Read the explicit top-level ``data`` scalars without a YAML dependency."""

    raw = _read_required_bytes(config_path, "batch-16 student config")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError(f"student config is not UTF-8: {config_path}") from exc

    section_starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"data:\s*(?:#.*)?", line) is not None
    ]
    _require(
        len(section_starts) == 1,
        f"student config must contain exactly one explicit top-level data section: {config_path}",
    )

    scalars: Dict[str, str] = {}
    for line in lines[section_starts[0] + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indentation = len(line) - len(line.lstrip(" "))
        _require("\t" not in line[: len(line) - len(line.lstrip())], f"tabs in YAML indentation: {config_path}")
        if indentation == 0:
            break
        if indentation != 2:
            continue
        match = re.fullmatch(r"  ([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*", line)
        if match is None:
            continue
        key, value = match.groups()
        value = value.split(" #", 1)[0].strip()
        _require(key not in scalars, f"duplicate data.{key} in {config_path}")
        scalars[key] = value
    return scalars


def _verify_student_config(spec: BenchmarkSpec, repo_root: Path) -> Dict[str, Any]:
    path = repo_root / spec.student_config_relative_path
    config_bytes = _read_required_bytes(path, f"{spec.name} batch-16 student config")
    scalars = _read_data_section_scalars(path)

    for key in ("max_train_tasks", "task_seed", "shuffle"):
        _require(key in scalars, f"missing explicit data.{key} in {path}")
    _require(
        scalars["max_train_tasks"] == str(TARGET_COUNT),
        f"{path}: data.max_train_tasks must be {TARGET_COUNT}",
    )
    _require(
        scalars["task_seed"] == str(TASK_SEED),
        f"{path}: data.task_seed must be {TASK_SEED}",
    )
    _require(
        scalars["shuffle"].lower() == "true",
        f"{path}: data.shuffle must be true",
    )
    return {
        "repository_relative_path": spec.student_config_relative_path.as_posix(),
        "source_bytes": len(config_bytes),
        "source_bytes_sha256": _sha256(config_bytes),
        "verified_data_fields": {
            "max_train_tasks": TARGET_COUNT,
            "shuffle": True,
            "task_seed": TASK_SEED,
        },
    }


def _legacy_audit(
    spec: BenchmarkSpec,
    repo_root: Path,
    curriculum: Sequence[str],
) -> Dict[str, Any]:
    legacy_path = repo_root / spec.legacy_relative_path
    output_path = repo_root / spec.output_relative_path
    _require(
        legacy_path.resolve() != output_path.resolve(),
        f"legacy and current curriculum paths alias each other: {legacy_path}",
    )
    legacy_bytes = _read_required_bytes(legacy_path, f"{spec.name} legacy 800-task file")
    legacy_ids = _parse_task_id_bytes(legacy_bytes, legacy_path, LEGACY_COUNT)
    current_prefix = list(curriculum[:LEGACY_COUNT])
    ordered_prefix_match = legacy_ids == current_prefix
    prefix_membership_match = set(legacy_ids) == set(current_prefix)

    # These committed legacy files came from a different sampling contract.
    # Refuse to silently relabel either one as the current fixed-seed prefix.
    _require(
        not ordered_prefix_match,
        f"{legacy_path} unexpectedly equals the ordered current 800-task prefix; review provenance explicitly",
    )
    _require(
        not prefix_membership_match,
        f"{legacy_path} unexpectedly has the current 800-task prefix membership; review provenance explicitly",
    )

    overlap_count = len(set(legacy_ids).intersection(curriculum))
    return {
        "repository_relative_path": spec.legacy_relative_path.as_posix(),
        "status": "legacy_not_current_curriculum",
        "count": len(legacy_ids),
        "unique_count": len(set(legacy_ids)),
        "file_bytes_sha256": _sha256(legacy_bytes),
        "ordered_prefix_match": ordered_prefix_match,
        "prefix_membership_match": prefix_membership_match,
        "overlap_with_current_1600_count": overlap_count,
        "overlap_with_current_1600_fraction": f"{overlap_count}/{LEGACY_COUNT}",
    }


def _curriculum_hashes(curriculum: Sequence[str]) -> Dict[str, Any]:
    ordered_bytes = _ordered_newline_bytes(curriculum)
    canonical_json = _canonical_json_bytes(list(curriculum))
    sorted_membership = sorted(curriculum, key=int)
    return {
        "count": len(curriculum),
        "unique_count": len(set(curriculum)),
        "ordered_newline_with_final_newline_sha256": _sha256(ordered_bytes),
        "canonical_json_sha256": _sha256(canonical_json),
        "sorted_numeric_membership_canonical_json_sha256": _sha256(
            _canonical_json_bytes(sorted_membership)
        ),
    }


def _derive_benchmark_entry(
    spec: BenchmarkSpec,
    repo_root: Path,
    *,
    require_output: bool,
) -> Tuple[Dict[str, Any], List[str], bytes]:
    pool, source_bytes = _load_pool(spec, repo_root)
    curriculum = _sample_curriculum(pool)
    expected_output_bytes = _ordered_newline_bytes(curriculum)
    _require(len(curriculum) == TARGET_COUNT, f"{spec.name}: short sampled curriculum")
    _require(
        len(set(curriculum)) == TARGET_COUNT,
        f"{spec.name}: sampled curriculum is not 1,600 unique tasks",
    )

    if require_output:
        output_path = repo_root / spec.output_relative_path
        actual_output_bytes = _read_required_bytes(
            output_path, f"{spec.name} fixed 1,600-task curriculum"
        )
        _parse_task_id_bytes(actual_output_bytes, output_path, TARGET_COUNT)
        _require(
            actual_output_bytes == expected_output_bytes,
            f"{output_path} does not exactly equal the canonical seed-2026 shuffled prefix",
        )

    entry: Dict[str, Any] = {
        "benchmark": spec.name,
        "source": {
            "repository_relative_path": spec.source_relative_path.as_posix(),
            "source_bytes": len(source_bytes),
            "source_bytes_sha256": _sha256(source_bytes),
            "id_semantics": spec.id_semantics,
            "pool_count": len(pool),
            "pool_unique_count": len(set(pool)),
        },
        "sampling": {
            "task_seed": TASK_SEED,
            "algorithm": ALGORITHM,
            "selected_count": TARGET_COUNT,
            "without_replacement": True,
            "ordered_sequence_preserved": True,
        },
        "curriculum": {
            "repository_relative_path": spec.output_relative_path.as_posix(),
            **_curriculum_hashes(curriculum),
        },
        "student_config": _verify_student_config(spec, repo_root),
        "legacy_800_audit": _legacy_audit(spec, repo_root, curriculum),
    }
    return entry, curriculum, expected_output_bytes


def _derive_manifest(
    repo_root: Path,
    *,
    require_outputs: bool,
) -> Tuple[Dict[str, Any], Mapping[str, bytes]]:
    entries: Dict[str, Any] = {}
    output_bytes: Dict[str, bytes] = {}
    for spec in BENCHMARKS:
        entry, _curriculum, serialized = _derive_benchmark_entry(
            spec, repo_root, require_output=require_outputs
        )
        entries[spec.name] = entry
        output_bytes[spec.name] = serialized

    manifest = {
        "schema_version": 1,
        "generator": "scripts/materialize_teacher_curricula.py",
        "contract": {
            "purpose": "teacher/student fixed training-task membership parity",
            "task_seed": TASK_SEED,
            "selected_count_per_benchmark": TARGET_COUNT,
            "algorithm": ALGORITHM,
            "output_serialization": "one canonical decimal task ID per LF line, with final LF",
            "canonical_json_serialization": (
                "UTF-8 JSON; ensure_ascii=false; allow_nan=false; sort_keys=true; "
                "separators=(',', ':'); no trailing LF"
            ),
        },
        "benchmarks": entries,
    }
    return manifest, output_bytes


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), 0o664)
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def materialize(repo_root: Path) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest, output_bytes = _derive_manifest(repo_root, require_outputs=False)
    for spec in BENCHMARKS:
        _atomic_write(repo_root / spec.output_relative_path, output_bytes[spec.name])
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(repo_root / MANIFEST_RELATIVE_PATH, manifest_bytes)
    # Post-write verification makes materialization transactional at the
    # contract level: success is reported only after all artifacts re-derive.
    return verify(repo_root)


def verify(repo_root: Path) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    expected_manifest, _ = _derive_manifest(repo_root, require_outputs=True)
    manifest_path = repo_root / MANIFEST_RELATIVE_PATH
    manifest_bytes = _read_required_bytes(manifest_path, "teacher curriculum manifest")
    _require(
        manifest_bytes.endswith(b"\n") and b"\r" not in manifest_bytes,
        f"manifest must be UTF-8 JSON with LF final newline: {manifest_path}",
    )
    try:
        actual_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid curriculum manifest {manifest_path}: {exc}") from exc
    _require(
        actual_manifest == expected_manifest,
        f"manifest is stale or does not exactly match recomputed provenance: {manifest_path}",
    )
    return expected_manifest


def _summary(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": "verified",
        "manifest": MANIFEST_RELATIVE_PATH.as_posix(),
        "benchmarks": {
            name: {
                "count": entry["curriculum"]["count"],
                "ordered_sha256": entry["curriculum"][
                    "ordered_newline_with_final_newline_sha256"
                ],
                "legacy_overlap": entry["legacy_800_audit"][
                    "overlap_with_current_1600_fraction"
                ],
            }
            for name, entry in manifest["benchmarks"].items()
        },
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("materialize", "verify"),
        help="write all artifacts and verify them, or perform read-only verification",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="repository root (defaults to the parent of scripts/)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = (
            materialize(args.repo_root)
            if args.mode == "materialize"
            else verify(args.repo_root)
        )
    except (ContractError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
