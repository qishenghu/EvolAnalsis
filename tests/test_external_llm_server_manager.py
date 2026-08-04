"""Unit tests for the external vLLM server correctness contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentevolver.module.trainer import external_llm_server_manager as manager_module
from agentevolver.module.trainer.external_llm_server_manager import (
    ExternalLLMServerManager,
)


SERVER_ADDRESSES = [f"server-{index}:820{index}" for index in range(1, 5)]


class _Response:
    def __init__(self, payload=None, *, text="", json_error=None):
        self.status_code = 200
        self.text = text
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def _bare_manager(*, sleeping=False, weights_dirty=False):
    """Build a manager without starting scheduler threads or doing HTTP I/O."""
    manager = ExternalLLMServerManager.__new__(ExternalLLMServerManager)
    manager.server_addresses = list(SERVER_ADDRESSES)
    manager._admin_timeout = 10
    manager._request_timeout = 10
    manager._sleep_between_steps = True
    manager._sync_enabled = True
    manager._sync_dir = "/unused/test-sync-dir"
    manager._weights_dirty = weights_dirty
    manager._sync_metrics = {}
    manager._unsupported_admin_routes = set()
    manager._is_sleeping = sleeping
    manager.worker_group = Mock()
    manager.config = SimpleNamespace(rollout={})
    return manager


def test_constructor_treats_preexisting_server_state_as_unknown(monkeypatch):
    health_check = Mock()

    def init_chat_scheduler(manager):
        manager.chat_scheduler = SimpleNamespace(model_name="test-model")
        manager.chat_scheduler_ready.set()

    monkeypatch.setattr(
        ExternalLLMServerManager, "_check_servers_alive", health_check
    )
    monkeypatch.setattr(
        ExternalLLMServerManager, "_init_chat_scheduler", init_chat_scheduler
    )
    config = manager_module.DictConfig(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "external_server_addresses": SERVER_ADDRESSES,
                    "external_sync_dir": "/unused/test-sync-dir",
                    "external_weight_sync": True,
                }
            },
            "trainer": {"experiment_name": "test"},
        }
    )

    manager = ExternalLLMServerManager(config, worker_group=Mock())

    assert manager._is_sleeping is None
    health_check.assert_called_once_with()


@pytest.mark.parametrize(
    ("response", "error_match"),
    [
        (
            _Response(text="not json", json_error=ValueError("invalid JSON")),
            "returned non-JSON",
        ),
        (_Response({"results": []}), "returned no worker results"),
        (_Response({}), "returned no worker results"),
    ],
)
def test_collective_rpc_rejects_non_json_or_empty_results(
    monkeypatch, response, error_match
):
    manager = _bare_manager()
    monkeypatch.setattr(manager_module.requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match=error_match):
        manager._collective_rpc(SERVER_ADDRESSES[0], "param_checksums", [[]])


def test_sync_rejects_incomplete_reload_statistics(monkeypatch):
    manager = _bare_manager()
    manager.reset_prefix_cache = Mock()
    manager._check_servers_alive = Mock()
    monkeypatch.setattr(manager_module.os, "makedirs", Mock())
    manager._collective_rpc = Mock(
        return_value=[
            {
                "num_files": 1,
                "num_exported_tensors": 426,
                "num_loaded_params": 0,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="reported incomplete load"):
        manager.sync_rollout_weights()

    manager.reset_prefix_cache.assert_not_called()
    manager._check_servers_alive.assert_not_called()


def test_sync_rejects_reload_fingerprint_mismatch(monkeypatch):
    manager = _bare_manager()
    manager.reset_prefix_cache = Mock()
    manager._check_servers_alive = Mock()
    monkeypatch.setattr(manager_module.os, "makedirs", Mock())

    def collective_rpc(address, method, args):
        assert method == "reload_weights_from_disk"
        loaded = 426 if address != SERVER_ADDRESSES[-1] else 425
        return [
            {
                "num_files": 1,
                "num_exported_tensors": 426,
                "num_loaded_params": loaded,
            }
        ]

    manager._collective_rpc = Mock(side_effect=collective_rpc)

    with pytest.raises(RuntimeError, match="different weight reload contracts"):
        manager.sync_rollout_weights()

    manager.reset_prefix_cache.assert_not_called()
    manager._check_servers_alive.assert_not_called()


def test_sync_rejects_cross_server_checksum_mismatch(monkeypatch):
    manager = _bare_manager()
    manager.reset_prefix_cache = Mock()
    manager._check_servers_alive = Mock()
    monkeypatch.setattr(manager_module.os, "makedirs", Mock())

    def collective_rpc(address, method, args):
        if method == "reload_weights_from_disk":
            return [
                {
                    "num_files": 1,
                    "num_exported_tensors": 426,
                    "num_loaded_params": 426,
                }
            ]
        assert method == "param_checksums"
        checksum = "same" if address != SERVER_ADDRESSES[-1] else "different"
        return [{"model.layers.0.weight": checksum}]

    manager._collective_rpc = Mock(side_effect=collective_rpc)

    with pytest.raises(RuntimeError, match="parameter checksums differ"):
        manager.sync_rollout_weights()

    manager.reset_prefix_cache.assert_not_called()
    manager._check_servers_alive.assert_not_called()


def test_wake_precedes_reload_when_servers_are_sleeping():
    manager = _bare_manager(sleeping=True, weights_dirty=True)
    calls = []

    def admin_post(address, route, **kwargs):
        calls.append((route, address))
        return True

    manager._admin_post = Mock(side_effect=admin_post)
    manager._check_servers_alive = Mock(side_effect=lambda: calls.append(("health", None)))
    manager.sync_rollout_weights = Mock(
        side_effect=lambda: calls.append(("reload", None))
    )

    synced = manager.wake_up()

    assert calls == [
        *(('/wake_up', address) for address in SERVER_ADDRESSES),
        ("health", None),
        ("reload", None),
        ("health", None),
    ]
    assert manager._is_sleeping is False
    assert manager._weights_dirty is False
    assert synced is True


def test_unknown_initial_state_wakes_before_reload_even_without_step_sleep():
    manager = _bare_manager(sleeping=None, weights_dirty=True)
    manager._sleep_between_steps = False
    calls = []

    def admin_post(address, route, **kwargs):
        calls.append((route, address))
        return True

    manager._admin_post = Mock(side_effect=admin_post)
    manager._check_servers_alive = Mock(
        side_effect=lambda: calls.append(("health", None))
    )
    manager.sync_rollout_weights = Mock(
        side_effect=lambda: calls.append(("reload", None))
    )

    synced = manager.wake_up()

    assert calls == [
        *(("/wake_up", address) for address in SERVER_ADDRESSES),
        ("health", None),
        ("reload", None),
        ("health", None),
    ]
    assert manager._is_sleeping is False
    assert manager._weights_dirty is False
    assert synced is True


def test_unknown_initial_state_uses_idempotent_wake_for_awake_servers():
    manager = _bare_manager(sleeping=None, weights_dirty=False)
    manager._admin_post = Mock(return_value=True)
    manager._check_servers_alive = Mock()
    manager.sync_rollout_weights = Mock()

    synced = manager.wake_up()

    assert synced is False
    assert manager._admin_post.call_args_list == [
        ((address, "/wake_up"),) for address in SERVER_ADDRESSES
    ]
    assert manager._is_sleeping is False
    manager.sync_rollout_weights.assert_not_called()
    assert manager._check_servers_alive.call_count == 2


def test_wake_reports_when_generation_reuses_current_weights():
    manager = _bare_manager(sleeping=False, weights_dirty=False)
    manager._check_servers_alive = Mock()
    manager.sync_rollout_weights = Mock()

    synced = manager.wake_up()

    assert synced is False
    manager.sync_rollout_weights.assert_not_called()
    manager._check_servers_alive.assert_called_once()
