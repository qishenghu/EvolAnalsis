# -*- coding: utf-8 -*-
"""Tests for the optional shared-actor recycling in ``env_service``.

Landmine L2 (docs/infra/LANDMINES.md): a long-lived environment process that
dlopens a fresh copy of a shared object per episode eventually dies with
"failed to map segment from shared object".  Only a new process recovers, so
``EnvService`` can retire a shared actor after N instances.

**The safety property these tests exist for**: with
``DUET_ENV_ACTOR_MAX_INSTANCES`` unset/0 and ``DUET_ENV_ACTOR_SELF_HEAL`` unset,
the service must behave exactly as it did before the feature existed — one
shared actor per env_type, created once, never killed, never retried.  A live
job (e.g. the 47511 ckpt sweep) restarts this stack between checkpoints and must
not notice the new code at all.

Ray is stubbed out so the tests are fast and hermetic; ``test_live_service`` is
the opt-in integration check that runs a real uvicorn env_service on a spare
port (never 8081).
"""

import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# A fake Ray just rich enough for EnvService: remote classes, actor handles,
# awaitable remote calls, and a kill counter.
# --------------------------------------------------------------------------- #
class FakeActor:
    """Stands in for one Ray actor process."""

    _next_id = 0

    def __init__(self, behaviour):
        FakeActor._next_id += 1
        self.actor_id = FakeActor._next_id
        self.killed = False
        self.instances = {}
        self.creates = 0
        self._behaviour = behaviour

    def _method(self, name):
        class _Remote:
            def __init__(self, actor, method_name):
                self._actor = actor
                self._name = method_name

            def remote(self, *args, **kwargs):
                return self._actor._call(self._name, *args, **kwargs)

        return _Remote(self, name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._method(name)

    async def _call(self, name, *args, **kwargs):
        if self.killed:
            raise RuntimeError(f"actor {self.actor_id} is dead")
        if self._behaviour is not None:
            self._behaviour(self, name, args)
        if name == "create_env_instance":
            task_id, instance_id, _params = args
            self.creates += 1
            self.instances[instance_id] = task_id
            return None
        if name == "get_init_state":
            return {"obs": f"init from actor {self.actor_id}"}
        if name == "step":
            return {"obs": "ok", "done": False}
        if name == "close":
            self.instances.pop(args[0], None)
            return None
        return None


class FakeRemoteClass:
    def __init__(self, registry, behaviour=None):
        self._registry = registry
        self._behaviour = behaviour

    def remote(self, *args, **kwargs):
        actor = FakeActor(self._behaviour)
        self._registry.append(actor)
        return actor


@pytest.fixture
def svc(monkeypatch):
    """A fresh EnvService with Ray stubbed and both knobs at their defaults."""
    monkeypatch.delenv("DUET_ENV_ACTOR_MAX_INSTANCES", raising=False)
    monkeypatch.delenv("DUET_ENV_ACTOR_SELF_HEAL", raising=False)
    FakeActor._next_id = 0  # actor_id is per-test, so tests can name "the first one"

    fake_ray = types.ModuleType("ray")
    killed = []

    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda **kwargs: None

    def _kill(actor):
        actor.killed = True
        killed.append(actor)

    fake_ray.kill = _kill

    def _remote(*dargs, **dkwargs):
        def _wrap(cls):
            return cls
        if dargs and callable(dargs[0]):
            return dargs[0]
        return _wrap

    fake_ray.remote = _remote
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    mod = importlib.import_module("env_service.env_service")
    importlib.reload(mod)

    service = mod.EnvService()
    service._test_actors = []          # every actor the fake class handed out
    service._test_killed = killed
    service._test_module = mod
    return service


def _install_env_cls(service, behaviour=None):
    """Make get_remote_env_cls hand back our fake class for 'alfworld'."""
    cls = FakeRemoteClass(service._test_actors, behaviour)
    service.remote_env["alfworld"] = cls
    return cls


def _create(service, n, first=0):
    for i in range(first, first + n):
        asyncio.run(
            service.create_instance("alfworld", task_id=str(i), instance_id=f"inst{i}"),
        )


# --------------------------------------------------------------------------- #
# 1. DEFAULT OFF — the red line.
# --------------------------------------------------------------------------- #
def test_default_is_off_no_recycling(svc):
    """Unset knobs => one actor forever, nothing killed. Byte-for-byte legacy."""
    _install_env_cls(svc)
    _create(svc, 25)

    assert len(svc._test_actors) == 1, "a second actor was created with recycling off"
    assert svc._test_killed == [], "an actor was killed with recycling off"
    assert svc._test_actors[0].creates == 25

    stats = svc.actor_stats()
    assert stats["max_instances"] == 0
    assert stats["self_heal"] is False
    assert stats["recycle_events"] == []


def test_default_off_does_not_retry_failures(svc, monkeypatch):
    """With self-heal off a map-segment error propagates on the first attempt."""
    def boom(actor, name, args):
        if name == "create_env_instance":
            raise OSError("failed to map segment from shared object: /tmp/x/libdownward.so")

    _install_env_cls(svc, behaviour=boom)
    with pytest.raises(OSError):
        _create(svc, 1)

    assert len(svc._test_actors) == 1, "self-heal rebuilt an actor while disabled"
    assert svc._test_killed == []


def test_env_knob_parsing_rejects_garbage(svc, monkeypatch):
    mod = svc._test_module
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "banana")
    with pytest.raises(ValueError):
        mod._actor_max_instances()
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "-3")
    with pytest.raises(ValueError):
        mod._actor_max_instances()
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "  0 ")
    assert mod._actor_max_instances() == 0
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "50")
    assert mod._actor_max_instances() == 50


# --------------------------------------------------------------------------- #
# 2. Quota recycling.
# --------------------------------------------------------------------------- #
def test_quota_retires_actor_after_n_instances(svc, monkeypatch):
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "5")
    _install_env_cls(svc)

    # 5 instances all land on generation 0 and retire it; nothing is killed yet
    # because all five are still live.
    _create(svc, 5)
    assert len(svc._test_actors) == 1
    assert svc._test_killed == []
    assert svc.shared_actor_generation["alfworld"] == 1

    # The 6th create builds a fresh actor process.
    _create(svc, 1, first=5)
    assert len(svc._test_actors) == 2
    assert svc._test_actors[1].creates == 1

    events = svc.actor_stats()["recycle_events"]
    assert len(events) == 1
    assert events[0]["instances_served"] == 5
    assert events[0]["forced"] is False


def test_retired_actor_is_reaped_only_after_last_release(svc, monkeypatch):
    """A retired actor keeps serving its in-flight episodes, then dies."""
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "3")
    _install_env_cls(svc)
    _create(svc, 3)
    gen0 = svc._test_actors[0]

    assert not gen0.killed, "retired actor killed while episodes were live"

    # Stepping an instance that lives on the retired actor still works.
    out = asyncio.run(svc.step("inst0", {"action": "look"}))
    assert out["obs"] == "ok"

    asyncio.run(svc.release_instance("inst0"))
    asyncio.run(svc.release_instance("inst1"))
    assert not gen0.killed, "reaped before the last episode finished"

    asyncio.run(svc.release_instance("inst2"))
    assert gen0.killed, "retired actor was never reaped"


def test_quota_recycles_repeatedly(svc, monkeypatch):
    monkeypatch.setenv("DUET_ENV_ACTOR_MAX_INSTANCES", "2")
    _install_env_cls(svc)
    for i in range(6):
        _create(svc, 1, first=i)
        asyncio.run(svc.release_instance(f"inst{i}"))
    # 6 instances / quota 2 => generations 0,1,2 used and retired.
    assert svc.shared_actor_generation["alfworld"] == 3
    assert len(svc._test_actors) == 3
    assert sum(1 for a in svc._test_actors if a.killed) == 3


# --------------------------------------------------------------------------- #
# 3. Self-heal on map-segment errors.
# --------------------------------------------------------------------------- #
def test_self_heal_rebuilds_and_retries_create(svc, monkeypatch):
    monkeypatch.setenv("DUET_ENV_ACTOR_SELF_HEAL", "true")

    state = {"failed": False}

    def flaky(actor, name, args):
        # Only the very first actor is poisoned; the rebuilt one works.
        if name == "create_env_instance" and actor.actor_id == 1 and not state["failed"]:
            state["failed"] = True
            raise OSError("failed to map segment from shared object")

    _install_env_cls(svc, behaviour=flaky)
    asyncio.run(svc.create_instance("alfworld", task_id="7", instance_id="inst7"))

    assert len(svc._test_actors) == 2, "self-heal did not rebuild the actor"
    assert svc._test_actors[0].killed, "the poisoned actor was not force-killed"
    events = svc.actor_stats()["recycle_events"]
    assert events and events[0]["forced"] is True


def test_self_heal_ignores_ordinary_errors(svc, monkeypatch):
    """A normal task failure must NOT trigger a rebuild — that would mask bugs."""
    monkeypatch.setenv("DUET_ENV_ACTOR_SELF_HEAL", "true")

    def boom(actor, name, args):
        if name == "create_env_instance":
            raise ValueError("task_id 999 is not a valid game index")

    _install_env_cls(svc, behaviour=boom)
    with pytest.raises(ValueError):
        _create(svc, 1)
    assert len(svc._test_actors) == 1
    assert svc._test_killed == []


def test_self_heal_on_step_rebuilds_for_future_episodes(svc, monkeypatch):
    monkeypatch.setenv("DUET_ENV_ACTOR_SELF_HEAL", "true")

    def flaky(actor, name, args):
        if name == "step":
            raise OSError("failed to map segment from shared object")

    _install_env_cls(svc, behaviour=flaky)
    _create(svc, 1)
    with pytest.raises(OSError):
        asyncio.run(svc.step("inst0", {"action": "look"}))

    # The episode is lost (its state died with the process) but the NEXT create
    # must land on a fresh actor rather than repeating the failure forever.
    assert svc._test_actors[0].killed
    _create(svc, 1, first=1)
    assert len(svc._test_actors) == 2


def test_marker_detection():
    mod = importlib.import_module("env_service.env_service")
    assert mod.looks_like_mmap_exhaustion(
        OSError("libdownward.so: failed to map segment from shared object"),
    )
    assert mod.looks_like_mmap_exhaustion(
        OSError("cannot allocate memory in static TLS block"),
    )
    assert not mod.looks_like_mmap_exhaustion(ValueError("Instance foo not found!"))
    assert not mod.looks_like_mmap_exhaustion(RuntimeError("connection refused"))


# --------------------------------------------------------------------------- #
# 4. Opt-in live integration check.
#    Runs a real env_service on a spare port — NEVER 8081, which belongs to the
#    production AlfWorld stack — and stops it again.
#      DUET_TEST_LIVE_ENV_SERVICE=1 pytest tests/test_env_actor_recycle.py -k live
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("DUET_TEST_LIVE_ENV_SERVICE") != "1",
    reason="set DUET_TEST_LIVE_ENV_SERVICE=1 to run the live env_service check",
)
def test_live_service_reports_actor_stats():
    import json
    import signal
    import subprocess
    import time
    import urllib.request

    port = int(os.environ.get("DUET_TEST_ENV_SERVICE_PORT", "8091"))
    assert port != 8081, "refusing to bind the production env_service port"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DUET_ENV_ACTOR_MAX_INSTANCES"] = "2"
    env["RAY_TMPDIR"] = env.get("RAY_TMPDIR", f"/tmp/dray_{os.getlogin()}") + "/envsvc_test"
    # Be a good citizen: this may run on a shared login node.
    env["ENV_SERVICE_RAY_NUM_CPUS"] = "2"
    env["ENV_SERVICE_RAY_INCLUDE_DASHBOARD"] = "false"

    proc = subprocess.Popen(
        [sys.executable, "-m", "env_service.env_service",
         "--env", "alfworld", "--portal", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        url = f"http://127.0.0.1:{port}/admin/env_actor_stats"
        deadline = time.time() + 120
        payload = None
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"env_service exited early:\n{proc.stdout.read().decode()[-4000:]}")
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    payload = json.loads(resp.read())
                break
            except Exception:
                time.sleep(2)
        assert payload is not None, "env_service never answered /admin/env_actor_stats"
        assert payload["success"] is True
        assert payload["data"]["max_instances"] == 2
        assert payload["data"]["actors"] == []   # nothing created yet
    finally:
        # Only ever signal the process group we started ourselves.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
