"""Contracts for the isolated teacher environment-service launch path."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "start_env_webshop_aux.sh"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_teacher_aux_env.sh"
HANDOFF_EXEC = ROOT / "scripts" / "teacher_aux_setsid_exec.sh"

RAY_RESOURCE_ENV = {
    "ENV_SERVICE_RAY_NUM_CPUS",
    "ENV_SERVICE_RAY_OBJECT_STORE_MEMORY",
    "ENV_SERVICE_RAY_INCLUDE_DASHBOARD",
}


def _import_env_service_with_fake_ray(extra_env):
    """Import env_service in a subprocess without starting a real Ray runtime."""
    program = r"""
import json
import sys
import types

calls = []
fake_ray = types.ModuleType("ray")
fake_ray.is_initialized = lambda: False
fake_ray.init = lambda **kwargs: calls.append(kwargs)
fake_ray.remote = lambda *args, **kwargs: (lambda value: value)
fake_ray.kill = lambda *args, **kwargs: None
sys.modules["ray"] = fake_ray

import env_service.env_service  # noqa: F401
print(json.dumps(calls[-1], sort_keys=True))
"""
    environment = os.environ.copy()
    for name in RAY_RESOURCE_ENV:
        environment.pop(name, None)
    environment.update(extra_env)
    environment["RAY_TMPDIR"] = "/tmp/evolanalysis-env-service-ray-test"
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_env_service_ray_resources_are_opt_in_and_default_is_compatible():
    result = _import_env_service_with_fake_ray({})

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "_temp_dir": "/tmp/evolanalysis-env-service-ray-test"
    }


def test_env_service_applies_explicit_aux_ray_resource_limits():
    result = _import_env_service_with_fake_ray(
        {
            "ENV_SERVICE_RAY_NUM_CPUS": "8",
            "ENV_SERVICE_RAY_OBJECT_STORE_MEMORY": str(2 * 1024**3),
            "ENV_SERVICE_RAY_INCLUDE_DASHBOARD": "false",
        }
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "_temp_dir": "/tmp/evolanalysis-env-service-ray-test",
        "include_dashboard": False,
        "num_cpus": 8,
        "object_store_memory": 2 * 1024**3,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ENV_SERVICE_RAY_NUM_CPUS", "0"),
        ("ENV_SERVICE_RAY_NUM_CPUS", "nan"),
        ("ENV_SERVICE_RAY_OBJECT_STORE_MEMORY", "2GiB"),
        ("ENV_SERVICE_RAY_INCLUDE_DASHBOARD", "maybe"),
    ],
)
def test_env_service_rejects_invalid_opt_in_ray_values(name, value):
    result = _import_env_service_with_fake_ray({name: value})

    assert result.returncode != 0
    assert name in result.stderr


def _write_fake_curl(path: Path):
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

url = sys.argv[-1]
if url.endswith(":18013/"):
    print(json.dumps("ok"))
elif url.endswith(":18011/"):
    print(json.dumps("This is environment AlfWorld."))
elif url.endswith("/healthz"):
    print("OK")
elif url.endswith("/get_env_profile"):
    size = 6710 if ":18093/" in url else 2420
    size += int(os.environ.get("FAKE_PROFILE_DELTA", "0"))
    print(json.dumps({"success": True, "data": list(range(size))}))
else:
    raise SystemExit(f"unexpected fake curl URL: {url}")
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ("webshop", "engine=18013 wrapper=18093 profile=6710"),
        ("alfworld", "engine=18011 wrapper=18091 profile=2420"),
    ],
)
def test_read_only_aux_verifier_checks_exact_stack_contract(
    tmp_path, environment, expected
):
    fake_curl = tmp_path / "curl"
    _write_fake_curl(fake_curl)
    env = os.environ.copy()
    env["TEACHER_AUX_CURL_BIN"] = str(fake_curl)
    env["TEACHER_AUX_PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), environment],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_read_only_aux_verifier_rejects_profile_size_drift(tmp_path):
    fake_curl = tmp_path / "curl"
    _write_fake_curl(fake_curl)
    env = os.environ.copy()
    env["TEACHER_AUX_CURL_BIN"] = str(fake_curl)
    env["TEACHER_AUX_PYTHON_BIN"] = sys.executable
    env["FAKE_PROFILE_DELTA"] = "-1"

    result = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "webshop"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "6709 tasks; expected 6710" in result.stderr


def test_webshop_aux_launcher_is_fixed_isolated_and_fail_closed():
    source = START_SCRIPT.read_text(encoding="utf-8")

    required_fragments = [
        'source "$SCRIPT_DIR/env_config.sh"',
        "AGENTGYM_PORT=18013",
        "ENVSERVICE_PORT=18093",
        "export RAY_TMPDIR=/data/ray/envwsaux",
        'export CUDA_VISIBLE_DEVICES=""',
        "unset RAY_ADDRESS",
        "export ENV_SERVICE_RAY_NUM_CPUS=8",
        "export ENV_SERVICE_RAY_OBJECT_STORE_MEMORY=2147483648",
        "export ENV_SERVICE_RAY_INCLUDE_DASHBOARD=false",
        "flock -n 9",
        'nohup setsid "$HANDOFF_EXEC" "$handoff_file" "$launch_token"',
        'bash "$VERIFY_SCRIPT" webshop',
        "validate_owned_state",
        "validate_handoff_identity",
        "discover_launched_listener",
        'kill -TERM -- "-$pgid"',
    ]
    for fragment in required_fragments:
        assert fragment in source

    forbidden_fragments = [
        "kill_port",
        "pkill",
        "ray stop",
        'rm -rf "${RAY_TMPDIR}',
        "session_\"*",
        "start_env_alfworld_aux.sh",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source

    assert source.index("expected_process engine") < source.index(
        'echo "Adopting healthy existing WebShop engine'
    )
    assert source.index("expected_process wrapper") < source.index(
        'echo "Adopted the complete healthy WebShop aux stack'
    )


def test_sets_id_fork_uses_actual_child_handoff_not_transient_launcher_pid():
    source = START_SCRIPT.read_text(encoding="utf-8")
    handoff_source = HANDOFF_EXEC.read_text(encoding="utf-8")

    assert 'printf \'pid=%s\\n\' "$$"' in handoff_source
    assert 'printf \'pgid=%s\\n\' "$PGID"' in handoff_source
    assert 'printf \'start_ticks=%s\\n\' "$START_TICKS"' in handoff_source
    assert 'mv -f "$HANDOFF_TMP" "$HANDOFF_FILE"' in handoff_source
    assert 'exec "$@"' in handoff_source

    handoff_wait = source.index('wait_for_handoff "$component" "$handoff_file"')
    listener_discovery = source.index(
        'actual_pid="$(discover_launched_listener "$component" "$port" "$handoff_file")"'
    )
    ownership_write = source.index(
        'write_owned_state "$component" "$actual_pid" "$port" "$launch_token"'
    )
    assert handoff_wait < listener_discovery < ownership_write
    assert 'write_owned_state "$component" "$launcher_pid"' not in source
    assert 'kill -TERM "$launcher_pid"' not in source
    assert 'listener" != "$handoff_pid"' in source


def test_dedicated_ray_root_check_cannot_match_its_own_checker_argv():
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert 'ps -eo pid=,args=' not in source
    assert 'awk -v marker="$RAY_TMPDIR/"' not in source
    assert "find_dedicated_ray_root_processes()" in source
    assert "for cmdline_file in /proc/[0-9]*/cmdline" in source
    assert 'current_pid="${BASHPID:-$$}"' in source
    assert "checker_ancestry[$$]=1" in source
    assert "checker_ancestry[$current_pid]=1" in source
    assert 'process_references_ray_root "$pid" "$RAY_TMPDIR"' in source
    assert 'ray_root_processes="$(find_dedicated_ray_root_processes)"' in source
    assert 'if [ -n "$ray_root_processes" ]' in source


def test_aux_scripts_are_shell_syntax_valid_and_verifier_has_no_signal_commands():
    for path in (START_SCRIPT, VERIFY_SCRIPT, HANDOFF_EXEC):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    verifier_source = VERIFY_SCRIPT.read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s*(kill|pkill|killall|ray\s+stop)\b", verifier_source) is None
    assert "start_env_alfworld_aux.sh" not in verifier_source
