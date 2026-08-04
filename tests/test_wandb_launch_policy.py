from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agentevolver.utils.tracking import (
    Tracking,
    enforce_online_wandb_config,
    get_wandb_runtime_env,
    preflight_wandb_online,
    require_wandb_online_environment,
)
from launcher import (
    _ensure_required_experiment_loggers,
    _preflight_required_wandb,
    _require_canonical_training_target,
)


@pytest.fixture(autouse=True)
def _online_wandb_environment(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


def test_launcher_adds_wandb_to_legacy_console_only_config():
    config = {"trainer": {"logger": ["console"]}}

    result = _ensure_required_experiment_loggers(deepcopy(config))

    assert result["trainer"]["logger"] == ["console", "wandb"]


def test_launcher_tracking_policy_is_idempotent():
    config = {"trainer": {"logger": ["wandb", "console"]}}

    result = _ensure_required_experiment_loggers(deepcopy(config))
    result = _ensure_required_experiment_loggers(result)

    assert result["trainer"]["logger"] == ["wandb", "console"]


def test_launcher_normalizes_string_logger_and_keeps_other_backends():
    config = {"trainer": {"logger": "tensorboard"}}

    result = _ensure_required_experiment_loggers(deepcopy(config))

    assert result["trainer"]["logger"] == ["tensorboard", "console", "wandb"]


def test_launcher_canonical_path_enforces_project_and_runs_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "launcher.preflight_wandb_online",
        lambda **kwargs: calls.append(kwargs),
    )
    config = {
        "trainer": {
            "experiment_name": "custom_target_run",
            "project_name": "legacy-project",
            "logger": ["console"],
        }
    }

    result = _preflight_required_wandb(config)

    assert result["trainer"]["project_name"] == "agentevolver"
    assert result["trainer"]["logger"] == ["console", "wandb"]
    assert calls == [
        {
            "project_name": "agentevolver",
            "experiment_name": "custom_target_run",
            "https_proxy": None,
        }
    ]


def test_launcher_rejects_noncanonical_target_before_side_effects(monkeypatch):
    import launcher

    side_effects = []
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: SimpleNamespace(
            target="custom.training_main",
            python_killer=True,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_fast_kill_by_keyword_bash",
        lambda *_args, **_kwargs: side_effects.append("kill"),
    )

    with pytest.raises(RuntimeError, match="custom target cannot guarantee"):
        launcher.main()

    assert side_effects == []


def test_canonical_target_check_accepts_only_main_ppo():
    _require_canonical_training_target("agentevolver.main_ppo")
    with pytest.raises(RuntimeError, match="mandatory online W&B run"):
        _require_canonical_training_target("agentevolver.other_main")


def test_direct_config_enforcement_normalizes_string_logger_and_project():
    config = {
        "trainer": {
            "experiment_name": "direct_run",
            "project_name": "wrong-project",
            "logger": "tensorboard",
        }
    }

    enforce_online_wandb_config(config)

    assert config["trainer"]["project_name"] == "agentevolver"
    assert config["trainer"]["logger"] == ["tensorboard", "console", "wandb"]


def test_struct_dictconfig_allows_missing_logger_and_project():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {"trainer": {"experiment_name": "structured_run"}}
    )
    OmegaConf.set_struct(config, True)

    enforce_online_wandb_config(config)

    assert OmegaConf.is_struct(config) is True
    assert list(config.trainer.logger) == ["console", "wandb"]
    assert config.trainer.project_name == "agentevolver"


@pytest.mark.parametrize(
    "mode", ["offline", "disabled", "dryrun", "shared", "run", ""]
)
def test_non_online_wandb_modes_are_rejected(mode):
    env = {"WANDB_MODE": mode}

    with pytest.raises(RuntimeError, match="Online W&B is mandatory"):
        require_wandb_online_environment(env)


def test_wandb_disabled_switch_is_rejected():
    with pytest.raises(RuntimeError, match="WANDB_DISABLED"):
        require_wandb_online_environment({"WANDB_DISABLED": "true"})


def test_plaintext_wandb_api_key_is_rejected_without_echoing_secret():
    secret = "never-serialize-this-key"

    with pytest.raises(RuntimeError) as exc_info:
        require_wandb_online_environment({"WANDB_API_KEY": secret})

    message = str(exc_info.value)
    assert "WANDB_API_KEY" in message
    assert "wandb login --verify" in message
    assert "0600" in message
    assert secret not in message


def test_unset_mode_is_made_explicit_and_forwarded():
    env = {}

    runtime_env = get_wandb_runtime_env(env)

    assert env["WANDB_MODE"] == "online"
    assert runtime_env == {"WANDB_MODE": "online"}


def test_wandb_preflight_is_authenticated_read_only_request(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, timeout):
            calls.append(("api", timeout))

        @property
        def viewer(self):
            calls.append(("viewer", None))
            return SimpleNamespace(entity="test-team")

    fake_wandb = SimpleNamespace(
        login=lambda **kwargs: calls.append(("login", kwargs)),
        Api=FakeApi,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    preflight_wandb_online(
        project_name="agentevolver",
        experiment_name="online_run",
        timeout=7,
    )

    assert calls == [
        ("login", {"force": True, "timeout": 1, "verify": True}),
        ("api", 7),
        ("viewer", None),
    ]


def test_wandb_preflight_error_redacts_sdk_message(monkeypatch):
    secret = "must-not-appear"

    class FakeApi:
        def __init__(self, timeout):
            self.timeout = timeout

        @property
        def viewer(self):
            raise ValueError(f"request contained API key {secret}")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(login=lambda **_kwargs: True, Api=FakeApi),
    )

    with pytest.raises(RuntimeError) as exc_info:
        preflight_wandb_online(
            project_name="agentevolver",
            experiment_name="failing_run",
        )

    assert "ValueError" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_wandb_preflight_uses_configured_proxy_and_restores_environment(monkeypatch):
    observed = []
    monkeypatch.setenv("HTTPS_PROXY", "http://inherited.invalid:1")
    monkeypatch.delenv("https_proxy", raising=False)

    class FakeApi:
        def __init__(self, timeout):
            observed.append(("api", timeout, os.environ["HTTPS_PROXY"]))

        @property
        def viewer(self):
            return SimpleNamespace(entity="test-team")

    def fake_login(**_kwargs):
        observed.append(("login", os.environ["HTTPS_PROXY"], os.environ["https_proxy"]))

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(login=fake_login, Api=FakeApi),
    )

    preflight_wandb_online(
        project_name="agentevolver",
        experiment_name="proxied_preflight",
        https_proxy="http://configured.invalid:2",
    )

    assert observed[0] == (
        "login",
        "http://configured.invalid:2",
        "http://configured.invalid:2",
    )
    assert observed[1] == ("api", 20, "http://configured.invalid:2")
    assert os.environ["HTTPS_PROXY"] == "http://inherited.invalid:1"
    assert "https_proxy" not in os.environ


def test_wandb_tracking_disables_stdout_capture_but_keeps_online_init(monkeypatch):
    calls = {}

    def settings(**kwargs):
        calls["settings"] = kwargs
        return SimpleNamespace(**kwargs)

    def init(**kwargs):
        calls["init"] = kwargs

    fake_wandb = SimpleNamespace(
        Settings=settings,
        init=init,
        finish=lambda **_kwargs: None,
        log=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = Tracking(
        project_name="agentevolver",
        experiment_name="future_run",
        default_backend=["wandb"],
        config={"trainer": {}},
    )

    assert calls["settings"] == {"console": "off"}
    assert calls["init"]["project"] == "agentevolver"
    assert calls["init"]["name"] == "future_run"
    assert calls["init"]["mode"] == "online"
    assert calls["init"]["settings"].console == "off"
    tracker.log({"metric": 1.0}, step=1)
    tracker.finish(exit_code=0)


def test_wandb_tracking_preserves_configured_proxy(monkeypatch):
    calls = {}
    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: calls.setdefault("settings", kwargs)
        or kwargs,
        init=lambda **kwargs: calls.setdefault("init", kwargs),
        finish=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = Tracking(
        project_name="agentevolver",
        experiment_name="proxied_run",
        default_backend=["wandb"],
        config={"trainer": {"wandb_proxy": "http://proxy.invalid:8080"}},
    )

    assert calls["settings"] == {
        "console": "off",
        "https_proxy": "http://proxy.invalid:8080",
    }
    tracker.finish(exit_code=0)


def test_tracking_forces_project_and_finishes_failure_once(monkeypatch):
    calls = {"finish": []}
    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        init=lambda **kwargs: calls.setdefault("init", kwargs),
        finish=lambda **kwargs: calls["finish"].append(kwargs),
        log=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    tracker = Tracking(
        project_name="legacy-project",
        experiment_name="failed_run",
        default_backend=["console"],
        config={"trainer": {}},
    )
    tracker.finish(exit_code=1)
    tracker.finish(exit_code=0)

    assert calls["init"]["project"] == "agentevolver"
    assert calls["init"]["name"] == "failed_run"
    assert calls["init"]["mode"] == "online"
    assert calls["finish"] == [{"exit_code": 1}]


@pytest.mark.parametrize("ray_already_initialized", [False, True])
def test_main_ppo_preflight_precedes_ray_and_forwards_only_file_credentials(
    monkeypatch, tmp_path, ray_already_initialized
):
    # Imported lazily because main_ppo loads the training stack.
    from omegaconf import OmegaConf
    import agentevolver.main_ppo as main_ppo

    events = []
    actor_options = {}
    ray_init_kwargs = {}

    config = OmegaConf.create(
        {
            "trainer": {
                "experiment_name": "direct_main_run",
                "project_name": "legacy-project",
                "logger": ["console"],
            },
            "ray_init": {"num_cpus": 2},
            "actor_rollout_ref": {"rollout": {"max_model_len": 32}},
            "data": {"max_prompt_length": 16, "max_response_length": 16},
        }
    )
    credentials_file = str(tmp_path / "wandb-credentials.json")
    monkeypatch.setenv("WANDB_CREDENTIALS_FILE", credentials_file)
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "ray"))

    def fake_preflight(**kwargs):
        events.append("preflight")
        assert kwargs == {
            "project_name": "agentevolver",
            "experiment_name": "direct_main_run",
            "https_proxy": None,
        }
        assert list(config.trainer.logger) == ["console", "wandb"]

    def fake_is_initialized():
        events.append("ray.is_initialized")
        return ray_already_initialized

    def fake_ray_init(**kwargs):
        events.append("ray.init")
        assert "WANDB_CREDENTIALS_FILE" not in os.environ
        ray_init_kwargs.update(kwargs)

    class FakeRunMethod:
        def remote(self, remote_config):
            events.append("task.run.remote")
            assert remote_config is config
            return "task-result"

    class FakeRunner:
        run = FakeRunMethod()

    class FakeConfiguredTaskRunner:
        @staticmethod
        def remote():
            events.append("taskrunner.remote")
            return FakeRunner()

    class FakeTaskRunner:
        @staticmethod
        def options(**kwargs):
            events.append("taskrunner.options")
            actor_options.update(kwargs)
            return FakeConfiguredTaskRunner

    monkeypatch.setattr(main_ppo, "preflight_wandb_online", fake_preflight)
    monkeypatch.setattr(main_ppo.ray, "is_initialized", fake_is_initialized)
    monkeypatch.setattr(main_ppo.ray, "init", fake_ray_init)
    monkeypatch.setattr(main_ppo.ray, "get", lambda result: events.append(("ray.get", result)))
    monkeypatch.setattr(main_ppo, "TaskRunner", FakeTaskRunner)

    main_ppo.run_ppo(config)

    assert events.index("preflight") < events.index("ray.is_initialized")
    if ray_already_initialized:
        assert "ray.init" not in events
    else:
        assert events.index("ray.is_initialized") < events.index("ray.init")
        assert "WANDB_API_KEY" not in ray_init_kwargs["runtime_env"]["env_vars"]
        assert "WANDB_CREDENTIALS_FILE" not in ray_init_kwargs["runtime_env"]["env_vars"]
        assert "WANDB_MODE" not in ray_init_kwargs["runtime_env"]["env_vars"]
    assert "WANDB_API_KEY" not in actor_options["runtime_env"]["env_vars"]
    assert actor_options["runtime_env"]["env_vars"]["WANDB_MODE"] == "online"
    assert (
        actor_options["runtime_env"]["env_vars"]["WANDB_CREDENTIALS_FILE"]
        == credentials_file
    )
    assert os.environ["WANDB_CREDENTIALS_FILE"] == credentials_file


def test_main_ppo_preflight_failure_starts_no_ray(monkeypatch):
    from omegaconf import OmegaConf
    import agentevolver.main_ppo as main_ppo

    config = OmegaConf.create(
        {
            "trainer": {"experiment_name": "blocked", "logger": ["console"]},
            "ray_init": {"num_cpus": 1},
            "actor_rollout_ref": {"rollout": {"max_model_len": 2}},
            "data": {"max_prompt_length": 1, "max_response_length": 1},
        }
    )
    ray_touched = []
    monkeypatch.setattr(
        main_ppo,
        "preflight_wandb_online",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )
    monkeypatch.setattr(
        main_ppo.ray,
        "is_initialized",
        lambda: ray_touched.append(True),
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        main_ppo.run_ppo(config)

    assert ray_touched == []


def test_main_ppo_rejects_plaintext_api_key_before_ray(monkeypatch):
    from omegaconf import OmegaConf
    import agentevolver.main_ppo as main_ppo

    config = OmegaConf.create(
        {
            "trainer": {"experiment_name": "env_key_blocked"},
            "ray_init": {"num_cpus": 1},
            "actor_rollout_ref": {"rollout": {"max_model_len": 2}},
            "data": {"max_prompt_length": 1, "max_response_length": 1},
        }
    )
    ray_touched = []
    monkeypatch.setenv("WANDB_API_KEY", "do-not-serialize")
    monkeypatch.setattr(
        main_ppo.ray,
        "is_initialized",
        lambda: ray_touched.append(True),
    )

    with pytest.raises(RuntimeError, match="Plaintext WANDB_API_KEY"):
        main_ppo.run_ppo(config)

    assert ray_touched == []


def test_final_colocated_gpu_worker_options_scrub_wandb_secrets():
    from agentevolver.module.trainer import ae_ray_trainer

    class FakeRayClassWithInit:
        def __init__(self):
            self.options = {}

        def update_options(self, options):
            self.options.update(options)

    worker = FakeRayClassWithInit()
    ae_ray_trainer._install_wandb_secret_scrub(worker)
    ae_ray_trainer._install_wandb_secret_scrub(worker)
    worker.update_options(
        {
            "runtime_env": {
                "env_vars": {
                    "RANK": "3",
                    "WANDB_API_KEY": "must-be-removed",
                }
            }
        }
    )

    env_vars = worker.options["runtime_env"]["env_vars"]
    assert env_vars["RANK"] == "3"
    assert env_vars["WANDB_API_KEY"] == ""
    assert env_vars["WANDB_IDENTITY_TOKEN_FILE"] == ""


def test_trainer_fit_rejects_missing_canonical_tracker():
    from agentevolver.module.trainer.ae_ray_trainer import (
        AgentEvolverRayPPOTrainer,
    )

    trainer = object.__new__(AgentEvolverRayPPOTrainer)

    with pytest.raises(RuntimeError, match="canonical TaskRunner"):
        trainer.fit()


def test_rollout_server_shell_scrubs_wandb_before_vllm_processes():
    script_path = Path(__file__).resolve().parents[1] / "start_rollout_servers.sh"
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    script = script_path.read_text(encoding="utf-8")

    unset_line = (
        "unset WANDB_API_KEY WANDB_IDENTITY_TOKEN_FILE "
        "WANDB_CREDENTIALS_FILE"
    )
    unset_index = script.index(unset_line)
    assert unset_index < script.index('source "${SCRIPT_DIR}/env_config.sh"')
    assert unset_index < script.index('nohup "${VLLM_BIN}" serve')


def test_taskrunner_tracking_finalizer_preserves_failure_exit_code():
    import agentevolver.main_ppo as main_ppo

    calls = []
    trainer = SimpleNamespace(
        _tracking=SimpleNamespace(
            finish=lambda **kwargs: calls.append(kwargs),
        )
    )

    main_ppo._finish_trainer_tracking(
        trainer, exit_code=1, suppress_errors=True
    )

    assert calls == [{"exit_code": 1}]


def test_taskrunner_tracking_finalizer_preserves_success_exit_code():
    import agentevolver.main_ppo as main_ppo

    calls = []
    trainer = SimpleNamespace(
        _tracking=SimpleNamespace(
            finish=lambda **kwargs: calls.append(kwargs),
        )
    )

    main_ppo._finish_trainer_tracking(
        trainer, exit_code=0, suppress_errors=False
    )

    assert calls == [{"exit_code": 0}]
