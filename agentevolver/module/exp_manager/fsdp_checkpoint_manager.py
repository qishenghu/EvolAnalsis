"""Workspace-local fixes for verl FSDP checkpoint serialization."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from typing import Iterator

from transformers import GenerationConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import fsdp_version


_GENERATION_CONFIG_FILENAME = "generation_config.json"
_MODEL_CONFIG_FILENAME = "config.json"

# ``name_or_path`` is mutable model-wide state.  Serializing checkpoint saves in
# one process prevents two managers sharing a config from observing one another's
# temporary fallback directory.  Distributed ranks live in separate processes.
_GENERATION_CONFIG_FALLBACK_LOCK = threading.RLock()


def _needs_local_generation_config_fallback(model_config, *, can_generate: bool) -> bool:
    """Return whether a local model directory is missing only its generation config."""
    name_or_path = getattr(model_config, "name_or_path", None)
    return bool(
        can_generate
        and name_or_path
        and os.path.isdir(name_or_path)
        and os.path.isfile(os.path.join(name_or_path, _MODEL_CONFIG_FILENAME))
        and not os.path.exists(os.path.join(name_or_path, _GENERATION_CONFIG_FILENAME))
    )


@contextmanager
def local_generation_config_fallback(
    model_config, *, can_generate: bool
) -> Iterator[bool]:
    """Temporarily redirect a missing local generation config to a safe fallback.

    verl loads ``GenerationConfig`` from ``model_config.name_or_path`` near the
    end of an FSDP save.  Some valid model directories do not ship
    ``generation_config.json``.  In that one case, construct the same fallback
    Transformers uses for a newly instantiated generative model, save it to a
    unique temporary directory, and let verl consume it through its normal path.

    Non-local identifiers, missing model directories, and existing (including
    malformed) generation config files retain verl's original behavior, so this
    compatibility layer cannot mask unrelated checkpoint errors.
    """
    original_name_or_path = getattr(model_config, "name_or_path", None)
    if not _needs_local_generation_config_fallback(
        model_config, can_generate=can_generate
    ):
        yield False
        return

    fallback_config = GenerationConfig.from_model_config(model_config)
    with tempfile.TemporaryDirectory(prefix="duet-generation-config-") as fallback_dir:
        fallback_config.save_pretrained(fallback_dir)
        try:
            model_config.name_or_path = fallback_dir
            yield True
        finally:
            model_config.name_or_path = original_name_or_path


class SafeFSDPCheckpointManager(FSDPCheckpointManager):
    """FSDP manager that supplies a model-derived generation-config fallback."""

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep=None,
    ):
        # Preserve verl's explicit no-op contract without requiring a model.
        if local_path is None:
            return super().save_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )

        with _GENERATION_CONFIG_FALLBACK_LOCK:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            can_generate = bool(unwrap_model.can_generate())
            with local_generation_config_fallback(
                unwrap_model.config, can_generate=can_generate
            ):
                return super().save_checkpoint(
                    local_path=local_path,
                    hdfs_path=hdfs_path,
                    global_step=global_step,
                    max_ckpt_to_keep=max_ckpt_to_keep,
                )
