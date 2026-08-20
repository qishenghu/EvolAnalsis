#!/usr/bin/env python
"""Convert openrouter_teacher_trajectory_v2 jsonl into a training-side teacher
experience file for ExperienceManager (ICLR 2027 / DUET-H200).

Why the output is still v2 jsonl
--------------------------------
``agentevolver/module/exp_manager/exp_manager.py::_dict_to_teacher_trajectory``
natively parses records whose ``schema_version`` is
``openrouter_teacher_trajectory_v2`` (validated fields: contract_sha256,
rollout_id, messages, decision_trace; reward/success/success_rate/query/data_id
are consumed directly).  The training loader therefore does NOT need the legacy
NeurIPS pkl format — the correct "conversion" is a *merge + dedupe + success
filter + fail-fast structural audit + tokenizer sanity gate*, emitting a clean
single-file v2 jsonl plus a manifest.  Emitting legacy pkl would silently drop
the v2 audit fields (decision_trace, contract_sha256) that the v2 loader keeps
in ``traj.metadata`` for replay under StructuredContextPolicy.

Usage (run with the vllm2 env python — its transformers can load the
Qwen3.5-4B tokenizer; this is pure CPU, no model weights are loaded):

  /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/vllm2/bin/python \
    scripts/convert_teacher_v2_to_training.py \
    --env alfworld \
    --input /path/alfworld_..._r1.jsonl /path/alfworld_..._topup8.jsonl \
    --output data/teacher_trajectories/iclr2027_flash/alfworld_dsv4flash_success_dedup.jsonl \
    --expected-tasks 1437

Guarantees:
  * one successful trajectory per task (dedupe keeps the FIRST occurrence in
    the given --input order; pass r1 before topup);
  * every record passes the same structural contract exp_manager enforces,
    plus stricter audit checks (assistant/decision alignment, step ordering);
  * before anything is written, a seeded sample of records is re-tokenized
    with the student tokenizer and byte-compared against the recorded
    completion_token_count / completion_token_ids_sha256 (rendering
    consistency gate);
  * a manifest records source shas, task counts, decision stats, sanity
    results and the converter version.

Fail-fast policy: any structural anomaly, tokenizer mismatch or task-count
mismatch raises immediately — nothing is silently skipped.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

CONVERTER_VERSION = "convert_teacher_v2_to_training/1.0.0 (2026-08-08)"
SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROLES = {"system", "user", "assistant"}


class ConversionError(RuntimeError):
    """Raised on any structural or consistency failure (fail-fast)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(token_ids: List[int]) -> str:
    # Byte-identical recipe to scripts/collect_openrouter_teacher_trajectories_dsv4.py
    return hashlib.sha256(
        ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    ).hexdigest()


def _fail(source: str, line_no: int, message: str) -> None:
    raise ConversionError(f"{source}:{line_no}: {message}")


def validate_record(rec: Dict[str, Any], env: str, source: str, line_no: int) -> None:
    """Structural audit. Superset of exp_manager's v2 validation."""
    if rec.get("schema_version") != SCHEMA_VERSION:
        _fail(source, line_no, f"schema_version != {SCHEMA_VERSION!r}: "
              f"{rec.get('schema_version')!r}")
    if rec.get("environment") != env:
        _fail(source, line_no, f"environment {rec.get('environment')!r} != --env {env!r}")

    task_id = rec.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        _fail(source, line_no, "task_id must be a non-empty string")
    if "data_id" not in rec:
        _fail(source, line_no, "data_id missing")
    if not isinstance(rec.get("rollout_id"), str) or not rec["rollout_id"]:
        _fail(source, line_no, "rollout_id must be a non-empty string")

    contract = rec.get("contract_sha256")
    if not isinstance(contract, str) or not _SHA256_RE.fullmatch(contract):
        _fail(source, line_no, "contract_sha256 must be a lowercase sha256 hex digest")

    if rec.get("success") is not True:
        _fail(source, line_no, f"record is not a success trajectory "
              f"(success={rec.get('success')!r}); the v2 collection protocol only "
              f"persists successes — refusing to continue")
    reward = rec.get("reward")
    if not isinstance(reward, (int, float)) or not reward > 0:
        _fail(source, line_no, f"success record must carry reward > 0, got {reward!r} "
              "(exp_manager sets Reward.outcome from this field)")

    messages = rec.get("messages")
    if not isinstance(messages, list) or not messages:
        _fail(source, line_no, "messages must be a non-empty list")
    for i, msg in enumerate(messages):
        if (not isinstance(msg, dict) or msg.get("role") not in _ALLOWED_ROLES
                or not isinstance(msg.get("content"), str)):
            _fail(source, line_no, f"messages[{i}] must be a role/content mapping with "
                  f"role in {sorted(_ALLOWED_ROLES)}")
    if messages[0]["role"] != "system":
        _fail(source, line_no, f"messages[0].role must be 'system', got {messages[0]['role']!r}")
    if messages[-1]["role"] != "assistant":
        _fail(source, line_no, "last message must be the terminal assistant decision")

    trace = rec.get("decision_trace")
    if not isinstance(trace, list) or not trace:
        _fail(source, line_no, "decision_trace must be a non-empty list")
    for i, dec in enumerate(trace):
        if not isinstance(dec, dict):
            _fail(source, line_no, f"decision_trace[{i}] must be a mapping")
        if dec.get("step_index") != i:
            _fail(source, line_no, f"decision_trace[{i}].step_index == "
                  f"{dec.get('step_index')!r}, expected {i}")
        content = dec.get("completion_content")
        if not isinstance(content, str) or not content:
            _fail(source, line_no, f"decision_trace[{i}].completion_content must be a "
                  "non-empty string")
        count = dec.get("completion_token_count")
        if not isinstance(count, int) or count <= 0:
            _fail(source, line_no, f"decision_trace[{i}].completion_token_count must be "
                  f"a positive int, got {count!r}")
        ids_sha = dec.get("completion_token_ids_sha256")
        if not isinstance(ids_sha, str) or not _SHA256_RE.fullmatch(ids_sha):
            _fail(source, line_no, f"decision_trace[{i}].completion_token_ids_sha256 "
                  "must be a lowercase sha256 hex digest")

    # Assistant/decision alignment: the message list is
    # [system, ack-assistant, (user, assistant) * n_decisions] — every decision
    # completion must appear verbatim as the trailing assistant messages.
    assistant_contents = [m["content"] for m in messages if m["role"] == "assistant"]
    n_dec = len(trace)
    if len(assistant_contents) != n_dec + 1:
        _fail(source, line_no, f"expected {n_dec + 1} assistant messages "
              f"(1 acknowledgement + {n_dec} decisions), found {len(assistant_contents)}")
    for i, dec in enumerate(trace):
        expected = assistant_contents[i + 1]
        if dec["completion_content"] != expected:
            _fail(source, line_no, f"decision_trace[{i}].completion_content does not "
                  f"match assistant message #{i + 1} — transcript/trace divergence")

    metadata = rec.get("metadata")
    if not isinstance(metadata, dict):
        _fail(source, line_no, "metadata must be a mapping")
    tok_info = metadata.get("student_tokenizer")
    if not isinstance(tok_info, dict) or not tok_info.get("path"):
        _fail(source, line_no, "metadata.student_tokenizer.path missing")


def load_and_merge(inputs: List[Path], env: str) -> Dict[str, Any]:
    """Read all inputs in order, validate every record, dedupe by task_id."""
    merged: Dict[str, Dict[str, Any]] = {}
    sources = []
    dedupe_dropped = 0
    for path in inputs:
        if not path.is_file():
            raise ConversionError(f"input file not found: {path}")
        records = 0
        tasks_in_file = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    _fail(str(path), line_no, f"invalid JSON: {exc}")
                validate_record(rec, env, str(path), line_no)
                records += 1
                task_id = rec["task_id"]
                if task_id in tasks_in_file:
                    _fail(str(path), line_no,
                          f"duplicate task_id {task_id!r} within one file — the v2 "
                          "collector writes one success per task; refusing to guess")
                tasks_in_file.add(task_id)
                if task_id in merged:
                    dedupe_dropped += 1  # first occurrence (earlier file) wins
                else:
                    merged[task_id] = rec
        if records == 0:
            raise ConversionError(f"input file contains no records: {path}")
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "records": records,
            "unique_tasks": len(tasks_in_file),
        })
    return {"merged": merged, "sources": sources, "dedupe_dropped": dedupe_dropped}


def run_sanity(
    merged: Dict[str, Dict[str, Any]],
    tokenizer_path: str,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Rendering-consistency gate: re-tokenize sampled completions with the
    student tokenizer and compare count AND token-id sha against the trace."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ConversionError(
            "transformers is required for the sanity gate; run this script with "
            "/projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/vllm2/bin/python"
        ) from exc

    # Cross-check the tokenizer artifacts against what the collector recorded.
    tok_json = Path(tokenizer_path) / "tokenizer.json"
    if not tok_json.is_file():
        raise ConversionError(f"tokenizer.json not found under {tokenizer_path}")
    local_tok_sha = sha256_file(tok_json)
    recorded_shas = {
        rec["metadata"]["student_tokenizer"]["artifact_sha256"].get("tokenizer.json")
        for rec in merged.values()
    }
    if recorded_shas != {local_tok_sha}:
        raise ConversionError(
            "tokenizer.json sha mismatch: records recorded "
            f"{sorted(recorded_shas)} but {tok_json} has {local_tok_sha} — "
            "the sanity gate would not test the collection-time tokenizer"
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    task_ids = sorted(merged.keys())
    rng = random.Random(seed)
    sampled = task_ids if len(task_ids) <= sample_size else rng.sample(task_ids, sample_size)

    decisions_checked = 0
    for task_id in sampled:
        rec = merged[task_id]
        for i, dec in enumerate(rec["decision_trace"]):
            ids = tokenizer.encode(dec["completion_content"], add_special_tokens=False)
            if len(ids) != dec["completion_token_count"]:
                raise ConversionError(
                    f"sanity FAIL task {task_id} decision {i}: retokenized count "
                    f"{len(ids)} != recorded {dec['completion_token_count']}"
                )
            if sha256_token_ids(ids) != dec["completion_token_ids_sha256"]:
                raise ConversionError(
                    f"sanity FAIL task {task_id} decision {i}: token-id sha mismatch "
                    "(same count but different ids — tokenizer drift)"
                )
            decisions_checked += 1
    return {
        "tokenizer_path": tokenizer_path,
        "tokenizer_json_sha256": local_tok_sha,
        "seed": seed,
        "sampled_tasks": len(sampled),
        "decisions_checked": decisions_checked,
        "result": "pass",
    }


def compute_stats(merged: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    n_tasks = len(merged)
    decision_counts = [len(rec["decision_trace"]) for rec in merged.values()]
    completion_tokens = [
        sum(d["completion_token_count"] for d in rec["decision_trace"])
        for rec in merged.values()
    ]
    truncated = sum(
        1
        for rec in merged.values()
        for d in rec["decision_trace"]
        if d.get("truncated_by_length")
    )
    rewards = [float(rec["reward"]) for rec in merged.values()]
    return {
        "tasks": n_tasks,
        "total_decisions": sum(decision_counts),
        "avg_decisions_per_task": round(sum(decision_counts) / n_tasks, 3),
        "min_decisions": min(decision_counts),
        "max_decisions": max(decision_counts),
        "total_completion_tokens": sum(completion_tokens),
        "avg_completion_tokens_per_task": round(sum(completion_tokens) / n_tasks, 1),
        "length_truncated_decisions": truncated,
        "avg_reward": round(sum(rewards) / n_tasks, 4),
        "min_reward": min(rewards),
        "contract_sha256_set": sorted({rec["contract_sha256"] for rec in merged.values()}),
        "teacher_model_set": sorted({rec.get("teacher_model", "unknown") for rec in merged.values()}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", required=True,
                        help="environment name every record must declare (e.g. alfworld)")
    parser.add_argument("--input", nargs="+", required=True, type=Path,
                        help="v2 jsonl files in priority order (r1 before topup); "
                             "on duplicate task_id the earlier file wins")
    parser.add_argument("--output", required=True, type=Path,
                        help="output jsonl (teacher_experience.data_path target); "
                             "a .manifest.json is written next to it")
    parser.add_argument("--expected-tasks", type=int, default=None,
                        help="fail unless the merged unique task count equals this")
    parser.add_argument("--tokenizer", default=None,
                        help="student tokenizer path for the sanity gate "
                             "(default: metadata.student_tokenizer.path from the records)")
    parser.add_argument("--sanity-sample", type=int, default=20,
                        help="number of records to re-tokenize (default 20)")
    parser.add_argument("--sanity-seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing output file")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise ConversionError(f"output already exists: {args.output} (use --overwrite)")

    loaded = load_and_merge(list(args.input), args.env)
    merged = loaded["merged"]

    if args.expected_tasks is not None and len(merged) != args.expected_tasks:
        raise ConversionError(
            f"merged unique task count {len(merged)} != --expected-tasks "
            f"{args.expected_tasks}"
        )

    tokenizer_paths = {
        rec["metadata"]["student_tokenizer"]["path"] for rec in merged.values()
    }
    if args.tokenizer is None:
        if len(tokenizer_paths) != 1:
            raise ConversionError(
                f"records disagree on student tokenizer path {sorted(tokenizer_paths)}; "
                "pass --tokenizer explicitly"
            )
        tokenizer_path = next(iter(tokenizer_paths))
    else:
        tokenizer_path = args.tokenizer

    sanity = run_sanity(merged, tokenizer_path, args.sanity_sample, args.sanity_seed)
    stats = compute_stats(merged)

    # Deterministic output order: numeric task_id sort where possible.
    def _task_key(task_id: str):
        return (0, int(task_id), task_id) if task_id.isdigit() else (1, 0, task_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for task_id in sorted(merged.keys(), key=_task_key):
            handle.write(json.dumps(merged[task_id], ensure_ascii=False) + "\n")
    tmp_path.replace(args.output)

    manifest = {
        "converter_version": CONVERTER_VERSION,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "environment": args.env,
        "command": sys.argv,
        "sources": loaded["sources"],
        "dedupe_dropped_records": loaded["dedupe_dropped"],
        "expected_tasks": args.expected_tasks,
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "records": len(merged),
        },
        "stats": stats,
        "sanity": sanity,
        "loader_contract": (
            "agentevolver/module/exp_manager/exp_manager.py::"
            "load_teacher_trajectories(.jsonl) -> _dict_to_teacher_trajectory "
            "(native openrouter_teacher_trajectory_v2 support; has_log_prob "
            "forced False)"
        ),
    }
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[ok] wrote {len(merged)} tasks -> {args.output}")
    print(f"[ok] manifest -> {manifest_path}")
    print(f"[ok] sanity: {sanity['sampled_tasks']} tasks / "
          f"{sanity['decisions_checked']} decisions retokenized, all consistent")
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConversionError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
