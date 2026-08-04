#!/usr/bin/env python3
"""Fail-closed orchestration for the two fixed DeepSeek teacher datasets.

Safety boundary
---------------
``plan`` and ``audit`` are read-only and never invoke an environment service,
W&B, or OpenRouter.  The only code path that may invoke the collector is the
``run`` subcommand, which additionally requires an exact informed-consent
phrase.  The orchestrator never reads an API key; the child collector parses
the fixed credential source itself without importing it.

The run order is deliberately strict: both one-task/one-rollout canaries must
finish and pass an independent artifact audit before the two 1,600-task full
collections may run concurrently.  The collector journals each successful
trajectory and attempt with fsync, so interrupted runs are resumed explicitly
with ``--resume`` and never overwritten implicitly.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PROJECT_ROOT / "scripts/collect_openrouter_teacher_trajectories.py"
AUX_VERIFIER = PROJECT_ROOT / "scripts/verify_teacher_aux_env.sh"
DEFAULT_COLLECTOR_PYTHON = Path(
    "/data/home/qisheng/miniconda3/envs/duet2/bin/python"
)
API_KEY_SOURCE = Path("/data/home/qisheng/test_openrouter.py")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/openrouter_teacher/deepseek_v4_flash_fixed1600_seed2026"
)

TEACHER_MODEL = "deepseek/deepseek-v4-flash"
API_BASE = "https://openrouter.ai/api/v1"
WANDB_PROJECT = "agentevolver"
TASK_SEED = 2026
TASK_COUNT = 1600
FULL_ROLLOUTS_PER_TASK = 10
MAX_ATTEMPTS_PER_ROLLOUT = 3
FULL_MAX_WORKERS = 4
CANARY_MAX_WORKERS = 1
API_MAX_RETRIES = 5
API_TIMEOUT_SECONDS = 1200.0
SCHEMA_VERSION = "openrouter_teacher_trajectory_v2"
ATTEMPT_SCHEMA_VERSION = "openrouter_teacher_attempt_v1"
ORCHESTRATOR_SCHEMA_VERSION = "openrouter_teacher_orchestration_v1"
CONSENT_PHRASE = "I_AUTHORIZE_BENCHMARK_CONTEXT_TO_OPENROUTER_AND_API_COST"
DISCLOSURE = (
    "OpenRouter/DeepSeek will receive benchmark prompts, environment "
    "observations, and managed conversation context; API use may incur cost."
)


@dataclass(frozen=True)
class Benchmark:
    name: str
    env_url: str
    config: Path
    task_file: Path
    task_sha256: str


BENCHMARKS: Dict[str, Benchmark] = {
    "alfworld": Benchmark(
        name="alfworld",
        env_url="http://127.0.0.1:18091",
        config=PROJECT_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/alfworld/"
        "alfworld_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml",
        task_file=PROJECT_ROOT
        / "data/alfworld/task_ids_train1600_seed2026.txt",
        task_sha256=(
            "38373eb25d63affb26f672dbfec83820731b586cc22c1db832a04303b7b58c39"
        ),
    ),
    "webshop": Benchmark(
        name="webshop",
        env_url="http://127.0.0.1:18093",
        config=PROJECT_ROOT
        / "config/duet_paper_experiments_configs/iclr2027/webshop/"
        "webshop_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml",
        task_file=PROJECT_ROOT
        / "data/webshop/task_ids_train1600_seed2026.txt",
        task_sha256=(
            "bd235d350a18e2a69bc80281e650f3320d1000e54b622abeb4df800d791a31ac"
        ),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def task_ids(spec: Benchmark) -> list[str]:
    raw = spec.task_file.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise RuntimeError(f"task file lacks final newline: {spec.task_file}")
    values = raw.splitlines()
    if len(values) != TASK_COUNT or len(set(values)) != TASK_COUNT:
        raise RuntimeError(
            f"{spec.name} task file is not exactly {TASK_COUNT} unique IDs"
        )
    if sha256_file(spec.task_file) != spec.task_sha256:
        raise RuntimeError(f"{spec.name} fixed curriculum hash drift")
    return values


def validate_static_inputs(
    collector_python: Path, *, require_private_credential: bool
) -> None:
    required_files = [COLLECTOR, AUX_VERIFIER]
    for spec in BENCHMARKS.values():
        required_files.extend([spec.config, spec.task_file])
    for path in required_files:
        if not path.is_file():
            raise RuntimeError(f"required file is missing: {path}")
    if not collector_python.is_file() or not os.access(collector_python, os.X_OK):
        raise RuntimeError(f"collector Python is not executable: {collector_python}")
    for spec in BENCHMARKS.values():
        task_ids(spec)

    # Deliberately inspect only file metadata.  This process must never read
    # credential contents; the collector owns secure AST parsing at run time.
    if require_private_credential:
        try:
            credential_stat = API_KEY_SOURCE.stat()
        except OSError as error:
            raise RuntimeError(f"credential source is unavailable: {API_KEY_SOURCE}") from error
        if not stat.S_ISREG(credential_stat.st_mode):
            raise RuntimeError(f"credential source is not a regular file: {API_KEY_SOURCE}")
        if credential_stat.st_uid != os.getuid():
            raise RuntimeError("credential source must be owned by the collector user")
        if stat.S_IMODE(credential_stat.st_mode) & 0o077:
            raise RuntimeError(
                f"credential source permissions are not private: {API_KEY_SOURCE}; "
                "set mode 0600 before an authorized run"
            )


def artifact_paths(output_dir: Path, environment: str, phase: str) -> Dict[str, Path]:
    suffix = "canary_t1_r1" if phase == "canary" else "fixed1600_r10"
    output = output_dir / f"{environment}_deepseek_v4_flash_{suffix}.jsonl"
    return {
        "output": output,
        "attempts": Path(str(output) + ".attempts.jsonl"),
        "manifest": Path(str(output) + ".manifest.json"),
        "lock": Path(str(output) + ".lock"),
        "log": output_dir / "logs" / f"{environment}_{phase}.log",
    }


def phase_shape(phase: str) -> tuple[int, int, int]:
    if phase == "canary":
        return 1, 1, CANARY_MAX_WORKERS
    if phase == "full":
        return TASK_COUNT, FULL_ROLLOUTS_PER_TASK, FULL_MAX_WORKERS
    raise ValueError(f"unsupported phase: {phase}")


def wandb_run_name(environment: str, phase: str) -> str:
    suffix = "canary_t1_r1" if phase == "canary" else "fixed1600_r10"
    return f"{environment}_deepseek_v4_flash_teacher_{suffix}_seed2026"


def collector_command(
    *,
    collector_python: Path,
    output_dir: Path,
    spec: Benchmark,
    phase: str,
    resume: bool,
    contract_only: bool = False,
) -> list[str]:
    selected_tasks, rollouts, workers = phase_shape(phase)
    paths = artifact_paths(output_dir, spec.name, phase)
    command = [
        str(collector_python),
        str(COLLECTOR),
        "--config",
        str(spec.config),
        "--env-url",
        spec.env_url,
        "--task-file",
        str(spec.task_file),
        "--output",
        str(paths["output"]),
        "--expected-task-count",
        str(TASK_COUNT),
        "--task-seed",
        str(TASK_SEED),
        "--max-tasks",
        str(selected_tasks),
        "--rollouts-per-task",
        str(rollouts),
        "--max-attempts-per-rollout",
        str(MAX_ATTEMPTS_PER_ROLLOUT),
        "--max-workers",
        str(workers),
        "--model",
        TEACHER_MODEL,
        "--api-base",
        API_BASE,
        "--api-key-source",
        str(API_KEY_SOURCE),
        # Prevent an inherited secret from silently replacing the audited file
        # source.  The orchestrator removes OPENROUTER_API_KEY from child env too.
        "--api-key-env",
        "__OPENROUTER_KEY_ENV_DISABLED_BY_ORCHESTRATOR__",
        "--api-timeout",
        str(API_TIMEOUT_SECONDS),
        "--api-max-retries",
        str(API_MAX_RETRIES),
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-run-name",
        wandb_run_name(spec.name, phase),
        "--wandb-mode",
        "online",
    ]
    if contract_only:
        command.extend(["--contract-only", "--skip-live-profile-check"])
    elif resume:
        command.append("--resume")
    return command


def redacted_plan(output_dir: Path, collector_python: Path) -> Dict[str, Any]:
    phases: Dict[str, Any] = {}
    for phase in ("canary", "full"):
        phases[phase] = {}
        for name, spec in BENCHMARKS.items():
            paths = artifact_paths(output_dir, name, phase)
            command = collector_command(
                collector_python=collector_python,
                output_dir=output_dir,
                spec=spec,
                phase=phase,
                resume=False,
            )
            phases[phase][name] = {
                "environment_url": spec.env_url,
                "config": str(spec.config),
                "task_file": str(spec.task_file),
                "task_sha256": spec.task_sha256,
                "output": str(paths["output"]),
                "command_argv": command,
            }
    return {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "safe_default": "plan/audit never call services or external APIs",
        "run_order": "both audited canaries, then parallel full collections",
        "external_data_disclosure": DISCLOSURE,
        "required_consent_phrase": CONSENT_PHRASE,
        "teacher_model": TEACHER_MODEL,
        "api_base": API_BASE,
        "wandb_mode": "online",
        "wandb_project": WANDB_PROJECT,
        "task_seed": TASK_SEED,
        "full_task_count": TASK_COUNT,
        "full_rollouts_per_task": FULL_ROLLOUTS_PER_TASK,
        "max_attempts_per_rollout_across_resumes": MAX_ATTEMPTS_PER_ROLLOUT,
        "full_max_workers_per_benchmark": FULL_MAX_WORKERS,
        "store_prompt_messages": False,
        "phases": phases,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"artifact contract drift for {label}: {actual!r} != {expected!r}")


def audit_artifact(output_dir: Path, spec: Benchmark, phase: str) -> Dict[str, Any]:
    selected_count, rollouts_per_task, expected_workers = phase_shape(phase)
    expected_total = selected_count * rollouts_per_task
    expected_tasks = task_ids(spec)[:selected_count]
    expected_task_set = set(expected_tasks)
    paths = artifact_paths(output_dir, spec.name, phase)
    for key in ("output", "attempts", "manifest"):
        if not paths[key].is_file():
            raise RuntimeError(f"missing {spec.name} {phase} {key}: {paths[key]}")

    manifest = load_json(paths["manifest"])
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"missing collector contract in {paths['manifest']}")
    contract_sha = manifest.get("contract_sha256")
    if not isinstance(contract_sha, str) or len(contract_sha) != 64:
        raise RuntimeError("collector manifest has no valid contract hash")
    _require_equal("teacher model", contract.get("teacher", {}).get("model"), TEACHER_MODEL)
    _require_equal("API base", contract.get("teacher", {}).get("api_base"), API_BASE)
    _require_equal("teacher max tokens", contract.get("teacher", {}).get("max_tokens"), 10240)
    _require_equal("environment URL", contract.get("collection_override", {}).get("env_service.env_url"), spec.env_url)
    _require_equal("snapshot override", contract.get("collection_override", {}).get("context_management.snapshot_training"), False)
    _require_equal("selected task count", contract.get("selected_task_count"), selected_count)
    _require_equal("selected task hash", contract.get("selected_tasks_ordered_newline_sha256"), sha256_lines(expected_tasks))
    _require_equal("task seed", contract.get("task_manifest", {}).get("task_seed"), TASK_SEED)
    _require_equal("task count", contract.get("task_manifest", {}).get("count"), TASK_COUNT)
    _require_equal("task curriculum hash", contract.get("task_manifest", {}).get("ordered_newline_sha256"), spec.task_sha256)
    _require_equal("rollouts per task", contract.get("rollouts_per_task"), rollouts_per_task)
    _require_equal("prompt storage", contract.get("store_prompt_messages"), False)
    _require_equal("prompt length", contract.get("student_contract", {}).get("prompt_length"), 22528)
    _require_equal("response length", contract.get("student_contract", {}).get("response_length"), 10240)
    _require_equal("max model length", contract.get("student_contract", {}).get("max_model_len"), 32768)
    policy = contract.get("collection_policy", {})
    _require_equal("attempt cap", policy.get("max_attempts_per_rollout_total_across_resumes"), MAX_ATTEMPTS_PER_ROLLOUT)
    _require_equal("worker count", policy.get("max_workers"), expected_workers)
    _require_equal("API retry count", policy.get("api_max_retries_per_decision"), API_MAX_RETRIES)
    _require_equal("W&B mode", policy.get("wandb_mode"), "online")
    _require_equal("W&B project", manifest.get("wandb_project"), WANDB_PROJECT)
    _require_equal("W&B run name", manifest.get("wandb_run_name"), wandb_run_name(spec.name, phase))
    if not manifest.get("wandb_run_id"):
        raise RuntimeError("collector manifest lacks W&B run ID")
    audit = manifest.get("audit", {})
    _require_equal("audit complete", audit.get("complete"), True)
    _require_equal("audit trajectory target", audit.get("target_trajectory_count"), expected_total)
    _require_equal("audit trajectory count", audit.get("trajectory_count"), expected_total)
    _require_equal("audit task target", audit.get("target_task_count"), selected_count)
    _require_equal("audit task coverage", audit.get("task_coverage_count"), selected_count)
    _require_equal("audit missing rollouts", audit.get("missing_rollouts"), 0)
    _require_equal("audit missing tasks", audit.get("missing_tasks"), 0)

    rollout_counts: Counter[str] = Counter()
    seen_rollout_ids: set[str] = set()
    with paths["output"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError(f"malformed success JSONL at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid success JSON at line {line_number}: {error}") from error
            _require_equal("success schema", record.get("schema_version"), SCHEMA_VERSION)
            _require_equal("success contract", record.get("contract_sha256"), contract_sha)
            _require_equal("success environment", record.get("environment"), spec.name)
            _require_equal("success flag", record.get("success"), True)
            _require_equal("success teacher", record.get("teacher_model"), TEACHER_MODEL)
            task_id = str(record.get("task_id"))
            if task_id not in expected_task_set:
                raise RuntimeError(f"extraneous task {task_id} at success line {line_number}")
            rollout_index = record.get("rollout_index")
            if not isinstance(rollout_index, int) or not 0 <= rollout_index < rollouts_per_task:
                raise RuntimeError(f"invalid rollout index at success line {line_number}")
            expected_rollout_id = f"{spec.name}:{task_id}:deepseek-v4-flash:{rollout_index}"
            _require_equal("rollout ID", record.get("rollout_id"), expected_rollout_id)
            if expected_rollout_id in seen_rollout_ids:
                raise RuntimeError(f"duplicate rollout ID {expected_rollout_id}")
            seen_rollout_ids.add(expected_rollout_id)
            rollout_counts[task_id] += 1
            trace = record.get("decision_trace")
            if not isinstance(trace, list) or not trace:
                raise RuntimeError(f"empty decision trace at success line {line_number}")
            for decision in trace:
                prompt_tokens = decision.get("prompt_token_count")
                completion_tokens = decision.get("completion_token_count")
                if not isinstance(prompt_tokens, int) or prompt_tokens > 22528:
                    raise RuntimeError(f"invalid prompt budget at success line {line_number}")
                if not isinstance(completion_tokens, int) or completion_tokens > 10240:
                    raise RuntimeError(f"invalid completion budget at success line {line_number}")
                if decision.get("truncated_by_length") is not False:
                    raise RuntimeError(f"truncated success decision at line {line_number}")
                if "prompt_messages" in decision:
                    raise RuntimeError(f"unexpected duplicated prompt messages at line {line_number}")
    _require_equal("success record count", len(seen_rollout_ids), expected_total)
    for task_id in expected_tasks:
        _require_equal(f"rollout count for task {task_id}", rollout_counts[task_id], rollouts_per_task)

    attempt_indices: Dict[str, set[int]] = defaultdict(set)
    finished_successes: set[str] = set()
    with paths["attempts"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise RuntimeError(f"malformed attempt JSONL at line {line_number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid attempt JSON at line {line_number}: {error}") from error
            _require_equal("attempt schema", event.get("schema_version"), ATTEMPT_SCHEMA_VERSION)
            _require_equal("attempt contract", event.get("contract_sha256"), contract_sha)
            rollout_id = str(event.get("rollout_id", ""))
            if rollout_id not in seen_rollout_ids:
                raise RuntimeError(f"attempt ledger has extraneous rollout ID {rollout_id}")
            attempt_index = event.get("attempt_index")
            if not isinstance(attempt_index, int) or not 0 <= attempt_index < MAX_ATTEMPTS_PER_ROLLOUT:
                raise RuntimeError(f"attempt budget exceeded at line {line_number}")
            attempt_indices[rollout_id].add(attempt_index)
            if event.get("event") == "attempt_finished" and event.get("status") == "success":
                finished_successes.add(rollout_id)
    for rollout_id in seen_rollout_ids:
        if not attempt_indices[rollout_id]:
            raise RuntimeError(f"no attempt ledger entries for {rollout_id}")
        if len(attempt_indices[rollout_id]) > MAX_ATTEMPTS_PER_ROLLOUT:
            raise RuntimeError(f"too many attempts for {rollout_id}")
        if rollout_id not in finished_successes:
            raise RuntimeError(f"no successful finished attempt for {rollout_id}")

    return {
        "environment": spec.name,
        "phase": phase,
        "contract_sha256": contract_sha,
        "trajectory_count": len(seen_rollout_ids),
        "task_coverage_count": len(rollout_counts),
        "wandb_run_id": manifest["wandb_run_id"],
        "complete": True,
    }


class ExclusiveOrchestratorLock:
    def __init__(self, output_dir: Path):
        self.path = output_dir / ".teacher_collection_orchestrator.lock"
        self.handle: Optional[Any] = None

    def __enter__(self) -> "ExclusiveOrchestratorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another orchestrator owns {self.path}") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


class ProcessRecord:
    def __init__(self, output_dir: Path, run_id: str, stage: str):
        self.path = output_dir / ".teacher_collection_orchestrator.pid.json"
        self.value: Dict[str, Any] = {
            "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
            "run_id": run_id,
            "orchestrator_pid": os.getpid(),
            "stage": stage,
            "started_at": utc_now(),
            "active": True,
            "children": {},
        }
        atomic_write_json(self.path, self.value)

    def child_started(self, label: str, pid: int, log: Path) -> None:
        self.value["children"][label] = {
            "pid": pid,
            "log": str(log),
            "started_at": utc_now(),
            "status": "running",
        }
        atomic_write_json(self.path, self.value)

    def child_finished(self, label: str, returncode: int) -> None:
        self.value["children"][label].update(
            {"finished_at": utc_now(), "returncode": returncode, "status": "exited"}
        )
        atomic_write_json(self.path, self.value)

    def finish(self, status_text: str) -> None:
        self.value.update(
            {"active": False, "finished_at": utc_now(), "status": status_text}
        )
        atomic_write_json(self.path, self.value)


def run_logged_processes(
    entries: Sequence[tuple[str, list[str], Path]], process_record: ProcessRecord
) -> Dict[str, int]:
    processes: Dict[str, tuple[subprocess.Popen[Any], Any]] = {}
    child_env = os.environ.copy()
    child_env.pop("OPENROUTER_API_KEY", None)
    child_env["WANDB_MODE"] = "online"
    child_env.pop("WANDB_DISABLED", None)
    try:
        for label, command, log_path in entries:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(
                f"\n[{utc_now()}] launch label={label} argv="
                + json.dumps(command, ensure_ascii=False)
                + "\n"
            )
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            processes[label] = (process, log_handle)
            process_record.child_started(label, process.pid, log_path)

        returncodes: Dict[str, int] = {}
        while len(returncodes) < len(processes):
            for label, (process, log_handle) in processes.items():
                if label in returncodes:
                    continue
                returncode = process.poll()
                if returncode is None:
                    continue
                returncodes[label] = returncode
                log_handle.write(f"[{utc_now()}] exit label={label} rc={returncode}\n")
                log_handle.flush()
                process_record.child_finished(label, returncode)
            if len(returncodes) < len(processes):
                time.sleep(0.5)
        return returncodes
    except BaseException:
        for process, _ in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20.0
        for process, _ in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    # Leave the exact PID recorded for operator inspection.  A
                    # broad or forceful cleanup is intentionally not attempted.
                    pass
        raise
    finally:
        for _, log_handle in processes.values():
            log_handle.close()


def verify_aux_stacks(names: Sequence[str], process_record: ProcessRecord) -> None:
    entries = []
    output_dir = process_record.path.parent
    for name in names:
        log_path = output_dir / "logs" / f"preflight_aux_{name}.log"
        entries.append((f"aux-{name}", ["bash", str(AUX_VERIFIER), name], log_path))
    returncodes = run_logged_processes(entries, process_record)
    failed = {label: rc for label, rc in returncodes.items() if rc != 0}
    if failed:
        raise RuntimeError(f"auxiliary environment preflight failed: {failed}")


def verify_collector_contracts(
    names: Sequence[str],
    phase: str,
    output_dir: Path,
    collector_python: Path,
    process_record: ProcessRecord,
) -> None:
    entries = []
    for name in names:
        spec = BENCHMARKS[name]
        command = collector_command(
            collector_python=collector_python,
            output_dir=output_dir,
            spec=spec,
            phase=phase,
            resume=False,
            contract_only=True,
        )
        log = output_dir / "logs" / f"preflight_contract_{name}_{phase}.log"
        entries.append((f"contract-{name}-{phase}", command, log))
    returncodes = run_logged_processes(entries, process_record)
    failed = {label: rc for label, rc in returncodes.items() if rc != 0}
    if failed:
        raise RuntimeError(f"collector contract preflight failed: {failed}")


def artifact_state(output_dir: Path, environment: str, phase: str) -> str:
    paths = artifact_paths(output_dir, environment, phase)
    material = [paths[key].exists() for key in ("output", "attempts", "manifest")]
    if not any(material):
        return "absent"
    try:
        audit_artifact(output_dir, BENCHMARKS[environment], phase)
    except Exception:
        return "partial_or_invalid"
    return "complete"


def prepare_launch_set(
    output_dir: Path, names: Sequence[str], phase: str, resume: bool
) -> list[str]:
    launch: list[str] = []
    for name in names:
        state = artifact_state(output_dir, name, phase)
        if state == "absent":
            launch.append(name)
        elif state == "complete" and resume:
            continue
        elif state == "complete":
            raise RuntimeError(
                f"{name} {phase} is already complete; use --resume to adopt it"
            )
        elif resume:
            launch.append(name)
        else:
            raise RuntimeError(
                f"{name} {phase} has existing partial/invalid artifacts; "
                "refusing overwrite without --resume"
            )
    return launch


def launch_phase(
    *,
    output_dir: Path,
    collector_python: Path,
    names: Sequence[str],
    phase: str,
    resume: bool,
    process_record: ProcessRecord,
    parallel: bool,
) -> list[Dict[str, Any]]:
    launch_names = prepare_launch_set(output_dir, names, phase, resume)
    if launch_names:
        verify_collector_contracts(
            launch_names, phase, output_dir, collector_python, process_record
        )
    if parallel:
        entries = []
        for name in launch_names:
            paths = artifact_paths(output_dir, name, phase)
            entries.append(
                (
                    f"collect-{name}-{phase}",
                    collector_command(
                        collector_python=collector_python,
                        output_dir=output_dir,
                        spec=BENCHMARKS[name],
                        phase=phase,
                        resume=resume,
                    ),
                    paths["log"],
                )
            )
        returncodes = run_logged_processes(entries, process_record) if entries else {}
        failed = {label: rc for label, rc in returncodes.items() if rc != 0}
        if failed:
            raise RuntimeError(
                f"{phase} collection incomplete; safely resumable with --resume: {failed}"
            )
    else:
        for name in launch_names:
            paths = artifact_paths(output_dir, name, phase)
            entry = (
                f"collect-{name}-{phase}",
                collector_command(
                    collector_python=collector_python,
                    output_dir=output_dir,
                    spec=BENCHMARKS[name],
                    phase=phase,
                    resume=resume,
                ),
                paths["log"],
            )
            returncodes = run_logged_processes([entry], process_record)
            if returncodes[entry[0]] != 0:
                raise RuntimeError(
                    f"{name} {phase} incomplete; safely resumable with --resume"
                )
            audit_artifact(output_dir, BENCHMARKS[name], phase)
    return [audit_artifact(output_dir, BENCHMARKS[name], phase) for name in names]


def require_canaries(output_dir: Path) -> list[Dict[str, Any]]:
    return [
        audit_artifact(output_dir, spec, "canary")
        for spec in BENCHMARKS.values()
    ]


def orchestration_manifest_path(output_dir: Path) -> Path:
    return output_dir / "orchestration.manifest.json"


def load_or_create_state(output_dir: Path) -> Dict[str, Any]:
    path = orchestration_manifest_path(output_dir)
    if path.exists():
        state = load_json(path)
        _require_equal(
            "orchestration schema",
            state.get("schema_version"),
            ORCHESTRATOR_SCHEMA_VERSION,
        )
        return state
    return {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "created_at": utc_now(),
        "plan": redacted_plan(output_dir, DEFAULT_COLLECTOR_PYTHON),
        "runs": [],
    }


def execute_run(args: argparse.Namespace) -> int:
    if args.consent != CONSENT_PHRASE:
        raise RuntimeError(
            f"external API authorization missing. Disclosure: {DISCLOSURE} "
            f"Re-run with --consent {CONSENT_PHRASE} only after explicit approval."
        )
    output_dir = args.output_dir.resolve()
    collector_python = args.collector_python.resolve()
    validate_static_inputs(collector_python, require_private_credential=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExclusiveOrchestratorLock(output_dir):
        run_id = uuid.uuid4().hex
        process_record = ProcessRecord(output_dir, run_id, args.stage)
        state = load_or_create_state(output_dir)
        run_state: Dict[str, Any] = {
            "run_id": run_id,
            "stage": args.stage,
            "resume": bool(args.resume),
            "started_at": utc_now(),
            "status": "running",
            "external_api_authorized": True,
            "external_data_disclosure": DISCLOSURE,
            "orchestrator_pid": os.getpid(),
        }
        state.setdefault("runs", []).append(run_state)
        state["updated_at"] = utc_now()
        atomic_write_json(orchestration_manifest_path(output_dir), state)
        try:
            names = list(BENCHMARKS)
            # A direct full-stage request must fail before touching any local
            # service unless both canaries already satisfy the independent gate.
            preaudited_canaries: Optional[list[Dict[str, Any]]] = None
            if args.stage == "full":
                preaudited_canaries = require_canaries(output_dir)
            # In a dual-benchmark stage, validate both local stacks before the
            # first paid request so a missing second stack cannot waste a canary.
            verify_aux_stacks(names, process_record)
            audits: Dict[str, Any] = {}
            if args.stage in {"canary", "all"}:
                audits["canary"] = launch_phase(
                    output_dir=output_dir,
                    collector_python=collector_python,
                    names=names,
                    phase="canary",
                    resume=args.resume,
                    process_record=process_record,
                    parallel=False,
                )
            if args.stage in {"full", "all"}:
                audits["canary_gate"] = (
                    preaudited_canaries
                    if preaudited_canaries is not None
                    else require_canaries(output_dir)
                )
                audits["full"] = launch_phase(
                    output_dir=output_dir,
                    collector_python=collector_python,
                    names=names,
                    phase="full",
                    resume=args.resume,
                    process_record=process_record,
                    parallel=True,
                )
            run_state.update(
                {"status": "complete", "finished_at": utc_now(), "audits": audits}
            )
            process_record.finish("complete")
            return 0
        except BaseException as error:
            run_state.update(
                {
                    "status": "failed_or_interrupted",
                    "finished_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                }
            )
            process_record.finish("failed_or_interrupted")
            raise
        finally:
            state["updated_at"] = utc_now()
            atomic_write_json(orchestration_manifest_path(output_dir), state)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--collector-python", type=Path, default=DEFAULT_COLLECTOR_PYTHON
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate fixed1600 OpenRouter teacher collection safely"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="print the exact plan; never call services or external APIs"
    )
    add_common_arguments(plan)

    audit = subparsers.add_parser(
        "audit", help="read and independently validate existing artifacts"
    )
    add_common_arguments(audit)
    audit.add_argument("--stage", choices=["canary", "full", "all"], default="all")

    run = subparsers.add_parser(
        "run", help="explicitly execute collectors after informed authorization"
    )
    add_common_arguments(run)
    run.add_argument("--stage", choices=["canary", "full", "all"], required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--consent",
        required=True,
        help=f"exact required phrase: {CONSENT_PHRASE}",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            validate_static_inputs(
                args.collector_python.resolve(), require_private_credential=False
            )
            print(
                json.dumps(
                    redacted_plan(
                        args.output_dir.resolve(), args.collector_python.resolve()
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "audit":
            output_dir = args.output_dir.resolve()
            phases = (
                ["canary", "full"] if args.stage == "all" else [args.stage]
            )
            results = [
                audit_artifact(output_dir, spec, phase)
                for phase in phases
                for spec in BENCHMARKS.values()
            ]
            print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            return execute_run(args)
        raise AssertionError(f"unhandled command: {args.command}")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
