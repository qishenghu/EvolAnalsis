"""CPU-only tests for the local verl checkpoint compatibility layer."""

from types import SimpleNamespace

import pytest
from transformers import GenerationConfig, PretrainedConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

from agentevolver.module.exp_manager import fsdp_checkpoint_manager as manager_module
from agentevolver.module.exp_manager.fsdp_checkpoint_manager import (
    SafeFSDPCheckpointManager,
)


def _model_config(model_dir, **kwargs):
    config = PretrainedConfig(**kwargs)
    config.save_pretrained(model_dir)
    config.name_or_path = str(model_dir)
    return config


def _bare_manager(config):
    manager = SafeFSDPCheckpointManager.__new__(SafeFSDPCheckpointManager)
    manager.model = SimpleNamespace(config=config, can_generate=lambda: True)
    return manager


def test_existing_generation_config_uses_original_path(monkeypatch, tmp_path):
    expected = GenerationConfig(max_new_tokens=37)
    expected.save_pretrained(tmp_path)
    config = _model_config(tmp_path, bos_token_id=1, eos_token_id=2)
    original_name_or_path = config.name_or_path
    observed = {}

    def fake_save(self, **kwargs):
        observed["name_or_path"] = self.model.config.name_or_path
        observed["generation_config"] = GenerationConfig.from_pretrained(
            self.model.config.name_or_path
        )
        return "saved"

    monkeypatch.setattr(manager_module, "fsdp_version", lambda model: 0)
    monkeypatch.setattr(FSDPCheckpointManager, "save_checkpoint", fake_save)

    assert _bare_manager(config).save_checkpoint("/unused") == "saved"
    assert observed["name_or_path"] == original_name_or_path
    assert observed["generation_config"].max_new_tokens == 37
    assert config.name_or_path == original_name_or_path


def test_malformed_existing_generation_config_error_is_not_hidden(
    monkeypatch, tmp_path
):
    config = _model_config(tmp_path, bos_token_id=1, eos_token_id=2)
    original_name_or_path = config.name_or_path
    (tmp_path / manager_module._GENERATION_CONFIG_FILENAME).write_text(
        "{invalid", encoding="utf-8"
    )

    def fake_save(self, **kwargs):
        assert self.model.config.name_or_path == original_name_or_path
        GenerationConfig.from_pretrained(self.model.config.name_or_path)

    monkeypatch.setattr(manager_module, "fsdp_version", lambda model: 0)
    monkeypatch.setattr(FSDPCheckpointManager, "save_checkpoint", fake_save)

    with pytest.raises(OSError, match="not a valid JSON file"):
        _bare_manager(config).save_checkpoint("/unused")

    assert config.name_or_path == original_name_or_path


def test_missing_generation_config_uses_model_config_fallback(monkeypatch, tmp_path):
    config = _model_config(tmp_path, bos_token_id=11, eos_token_id=12)
    original_name_or_path = config.name_or_path
    observed = {}

    def fake_save(self, **kwargs):
        fallback_path = self.model.config.name_or_path
        observed["fallback_path"] = fallback_path
        observed["exists_during_save"] = (
            manager_module.os.path.isfile(
                manager_module.os.path.join(
                    fallback_path, manager_module._GENERATION_CONFIG_FILENAME
                )
            )
        )
        observed["generation_config"] = GenerationConfig.from_pretrained(
            fallback_path
        )
        return "saved"

    monkeypatch.setattr(manager_module, "fsdp_version", lambda model: 0)
    monkeypatch.setattr(FSDPCheckpointManager, "save_checkpoint", fake_save)

    assert _bare_manager(config).save_checkpoint("/unused") == "saved"
    assert observed["fallback_path"] != original_name_or_path
    assert observed["exists_during_save"] is True
    assert observed["generation_config"].bos_token_id == 11
    assert observed["generation_config"].eos_token_id == 12
    assert config.name_or_path == original_name_or_path
    assert not manager_module.os.path.exists(observed["fallback_path"])


def test_original_name_or_path_is_restored_when_save_raises(monkeypatch, tmp_path):
    config = _model_config(tmp_path, bos_token_id=1, eos_token_id=2)
    original_name_or_path = config.name_or_path
    observed = {}

    def failing_save(self, **kwargs):
        observed["fallback_path"] = self.model.config.name_or_path
        raise RuntimeError("unrelated shard failure")

    monkeypatch.setattr(manager_module, "fsdp_version", lambda model: 0)
    monkeypatch.setattr(FSDPCheckpointManager, "save_checkpoint", failing_save)

    with pytest.raises(RuntimeError, match="unrelated shard failure"):
        _bare_manager(config).save_checkpoint("/unused")

    assert observed["fallback_path"] != original_name_or_path
    assert config.name_or_path == original_name_or_path
    assert not manager_module.os.path.exists(observed["fallback_path"])
