import json
import os
from pathlib import Path
import sys

import pytest

from scripts import orchestrate_openrouter_teacher_collection as orchestration
from scripts.collect_openrouter_teacher_trajectories import (
    ATTEMPT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TeacherCollector,
    WorkItem,
    parse_args as parse_collector_args,
    scan_attempt_ledger,
)


def _arg(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize(
    ("environment", "port"), [("alfworld", "18091"), ("webshop", "18093")]
)
def test_full_command_is_exact_online_and_secret_free(tmp_path, environment, port):
    command = orchestration.collector_command(
        collector_python=Path(sys.executable),
        output_dir=tmp_path,
        spec=orchestration.BENCHMARKS[environment],
        phase="full",
        resume=False,
    )

    assert _arg(command, "--env-url") == f"http://127.0.0.1:{port}"
    assert _arg(command, "--model") == "deepseek/deepseek-v4-flash"
    assert _arg(command, "--max-tasks") == "1600"
    assert _arg(command, "--rollouts-per-task") == "10"
    assert _arg(command, "--max-attempts-per-rollout") == "3"
    assert _arg(command, "--max-workers") == "4"
    assert _arg(command, "--wandb-mode") == "online"
    assert "--store-prompt-messages" not in command
    assert "--resume" not in command
    assert not any(value.startswith("sk-") for value in command)


def test_plan_path_cannot_spawn_processes(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not spawn a subprocess")

    monkeypatch.setattr(orchestration.subprocess, "Popen", forbidden)
    result = orchestration.main(
        [
            "plan",
            "--output-dir",
            str(tmp_path),
            "--collector-python",
            sys.executable,
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["safe_default"].startswith("plan/audit never")
    assert payload["required_consent_phrase"] == orchestration.CONSENT_PHRASE


def test_wrong_consent_fails_before_any_process(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("unauthorized run must not spawn a subprocess")

    monkeypatch.setattr(orchestration.subprocess, "Popen", forbidden)
    result = orchestration.main(
        [
            "run",
            "--stage",
            "all",
            "--consent",
            "not-authorized",
            "--output-dir",
            str(tmp_path),
            "--collector-python",
            sys.executable,
        ]
    )

    assert result == 1
    assert list(tmp_path.iterdir()) == []


def test_collector_cli_rejects_offline_wandb():
    with pytest.raises(SystemExit):
        parse_collector_args(
            [
                "--config",
                "config.yaml",
                "--env-url",
                "http://127.0.0.1:1",
                "--task-file",
                "tasks.txt",
                "--output",
                "output.jsonl",
                "--wandb-mode",
                "offline",
            ]
        )


def test_run_requires_private_credential_metadata(monkeypatch, tmp_path):
    credential = tmp_path / "credential.py"
    credential.write_text("raise AssertionError('must not be read')\n", encoding="utf-8")
    credential.chmod(0o644)
    monkeypatch.setattr(orchestration, "API_KEY_SOURCE", credential)

    with pytest.raises(RuntimeError, match="mode 0600"):
        orchestration.validate_static_inputs(
            Path(sys.executable), require_private_credential=True
        )
    credential.chmod(0o600)
    orchestration.validate_static_inputs(
        Path(sys.executable), require_private_credential=True
    )


def _write_canary(tmp_path, environment):
    spec = orchestration.BENCHMARKS[environment]
    paths = orchestration.artifact_paths(tmp_path, environment, "canary")
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    first_task = orchestration.task_ids(spec)[0]
    rollout_id = f"{environment}:{first_task}:deepseek-v4-flash:0"
    contract_sha = "a" * 64
    contract = {
        "teacher": {
            "model": orchestration.TEACHER_MODEL,
            "api_base": orchestration.API_BASE,
            "max_tokens": 10240,
        },
        "collection_override": {
            "env_service.env_url": spec.env_url,
            "context_management.snapshot_training": False,
        },
        "selected_task_count": 1,
        "selected_tasks_ordered_newline_sha256": orchestration.sha256_lines(
            [first_task]
        ),
        "task_manifest": {
            "task_seed": 2026,
            "count": 1600,
            "ordered_newline_sha256": spec.task_sha256,
        },
        "rollouts_per_task": 1,
        "store_prompt_messages": False,
        "student_contract": {
            "prompt_length": 22528,
            "response_length": 10240,
            "max_model_len": 32768,
        },
        "collection_policy": {
            "max_attempts_per_rollout_total_across_resumes": 3,
            "max_workers": 1,
            "api_max_retries_per_decision": 5,
            "wandb_mode": "online",
        },
    }
    manifest = {
        "contract_sha256": contract_sha,
        "contract": contract,
        "wandb_project": "agentevolver",
        "wandb_run_name": orchestration.wandb_run_name(environment, "canary"),
        "wandb_run_id": "wandb-test-id",
        "audit": {
            "complete": True,
            "target_trajectory_count": 1,
            "trajectory_count": 1,
            "target_task_count": 1,
            "task_coverage_count": 1,
            "missing_rollouts": 0,
            "missing_tasks": 0,
        },
    }
    paths["manifest"].write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    record = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_sha,
        "environment": environment,
        "success": True,
        "teacher_model": orchestration.TEACHER_MODEL,
        "task_id": first_task,
        "rollout_index": 0,
        "rollout_id": rollout_id,
        "decision_trace": [
            {
                "prompt_token_count": 10,
                "completion_token_count": 8,
                "truncated_by_length": False,
            }
        ],
    }
    paths["output"].write_text(json.dumps(record) + "\n", encoding="utf-8")
    attempt = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "contract_sha256": contract_sha,
        "event": "attempt_finished",
        "status": "success",
        "rollout_id": rollout_id,
        "attempt_index": 0,
    }
    paths["attempts"].write_text(json.dumps(attempt) + "\n", encoding="utf-8")
    return paths


@pytest.mark.parametrize("environment", ["alfworld", "webshop"])
def test_independent_canary_audit_and_attempt_cap(tmp_path, environment):
    paths = _write_canary(tmp_path, environment)
    result = orchestration.audit_artifact(
        tmp_path, orchestration.BENCHMARKS[environment], "canary"
    )
    assert result["trajectory_count"] == 1
    assert result["complete"] is True

    event = json.loads(paths["attempts"].read_text(encoding="utf-8"))
    event["attempt_index"] = 3
    paths["attempts"].write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="attempt budget exceeded"):
        orchestration.audit_artifact(
            tmp_path, orchestration.BENCHMARKS[environment], "canary"
        )


def test_started_attempt_consumes_resume_budget_and_cap_is_total(tmp_path):
    ledger = tmp_path / "attempts.jsonl"
    rollout_id = "alfworld:1:deepseek-v4-flash:0"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "contract_sha256": "contract",
                "event": "attempt_started",
                "rollout_id": rollout_id,
                "attempt_index": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert scan_attempt_ledger(ledger, "contract")[rollout_id] == 3

    collector = object.__new__(TeacherCollector)
    collector.max_attempts_per_rollout = 3
    result = collector.collect(
        WorkItem(
            task_id="1", rollout_index=0, rollout_id=rollout_id, next_attempt=3
        )
    )
    assert result.attempts == 0
    assert result.final_error == "attempt_budget_exhausted"
