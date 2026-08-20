#!/usr/bin/env python3
"""Patch vendored verl 0.4.0.dev0 so it is importable under transformers 5.5.1 WITHOUT vllm.

Target break points (verified against the duet env copy of verl 0.4.0.dev):
  P1 verl/__init__.py            : top-level `import pkg_resources` (removed in setuptools>=81)
  P2 verl/models/transformers/monkey_patch.py
                                 : top-level `from transformers.modeling_flash_attention_utils
                                   import _flash_attention_forward`; also make apply_monkey_patch
                                   no-op with a warning for qwen3_5* (GDN) model types.
  P3 verl/utils/fsdp_utils.py    : `from transformers.trainer_pt_utils import
                                   get_module_class_from_name` (moved/removed in 5.x)
  P4 verl/workers/fsdp_workers.py: AutoModelForVision2Seq removed in transformers 5.x +
                                   `_model_mapping` private API; add robust fallback so a
                                   *ForConditionalGeneration architecture (e.g.
                                   Qwen3_5ForConditionalGeneration) still loads.

All replacements are idempotent: running twice is a no-op.

Usage:
    python patch_verl.py /path/to/site-packages/verl [--check]
"""

import sys
from pathlib import Path

PATCHES = []


def patch(path, old, new, name):
    PATCHES.append((path, old, new, name))


# ---------------------------------------------------------------------------
# P1: verl/__init__.py — guard pkg_resources (only needed on the NPU path)
# ---------------------------------------------------------------------------
patch(
    "__init__.py",
    """import pkg_resources
from packaging.version import parse as parse_version
from pkg_resources import DistributionNotFound
""",
    """from packaging.version import parse as parse_version

try:  # [t5x-patch] pkg_resources is gone from setuptools>=81; only used on the NPU path
    import pkg_resources
    from pkg_resources import DistributionNotFound
except ImportError:  # pragma: no cover
    pkg_resources = None

    class DistributionNotFound(Exception):
        pass
""",
    "P1 pkg_resources guard",
)

# ---------------------------------------------------------------------------
# P2a: monkey_patch.py — lazy/guarded flash-attn import
# ---------------------------------------------------------------------------
patch(
    "models/transformers/monkey_patch.py",
    """from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel
""",
    """from transformers.modeling_utils import PreTrainedModel

try:  # [t5x-patch] moved/refactored in transformers 5.x; GDN models never reach these patches
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
except ImportError:  # pragma: no cover
    _flash_attention_forward = None
""",
    "P2a lazy _flash_attention_forward import",
)

# ---------------------------------------------------------------------------
# P2b: monkey_patch.py — no-op with warning for qwen3_5* (GDN) model types
# ---------------------------------------------------------------------------
patch(
    "models/transformers/monkey_patch.py",
    '''    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]
''',
    '''    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    # [t5x-patch] Qwen3.5 (GDN / hybrid linear attention) is incompatible with the
    # ulysses/rmpad flash-attention patches; skip entirely.
    _model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    if _model_type.startswith("qwen3_5"):
        import warnings as _warnings

        _warnings.warn(
            f"apply_monkey_patch: skipped for model_type={_model_type} "
            "(GDN/hybrid-attention model; ulysses SP and remove-padding are unsupported)."
        )
        return
    if _flash_attention_forward is None and (use_remove_padding or ulysses_sp_size > 1):
        raise ImportError(
            "transformers.modeling_flash_attention_utils._flash_attention_forward is "
            "unavailable in this transformers version; disable use_remove_padding and "
            "ulysses_sequence_parallel_size or install transformers<5."
        )
    module = sys.modules[model.__module__]
''',
    "P2b apply_monkey_patch qwen3_5 no-op",
)

# ---------------------------------------------------------------------------
# P3: fsdp_utils.py — vendored get_module_class_from_name
# ---------------------------------------------------------------------------
patch(
    "utils/fsdp_utils.py",
    """from transformers.trainer_pt_utils import get_module_class_from_name
""",
    """try:  # [t5x-patch] moved/removed in transformers 5.x
    from transformers.trainer_pt_utils import get_module_class_from_name
except ImportError:  # pragma: no cover

    def get_module_class_from_name(module, name):
        \"\"\"Vendored from transformers.trainer_pt_utils (removed in transformers 5.x).\"\"\"
        if module.__class__.__name__ == name:
            return module.__class__
        for child_module in module.children():
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
        return None
""",
    "P3 get_module_class_from_name fallback",
)

# ---------------------------------------------------------------------------
# P4a: fsdp_workers.py — guard AutoModelForVision2Seq import
# ---------------------------------------------------------------------------
patch(
    "workers/fsdp_workers.py",
    """        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
""",
    """        from transformers import AutoConfig, AutoModelForCausalLM

        try:  # [t5x-patch] AutoModelForVision2Seq removed in transformers 5.x
            from transformers import AutoModelForVision2Seq
        except ImportError:  # pragma: no cover
            AutoModelForVision2Seq = None
""",
    "P4a AutoModelForVision2Seq import guard",
)

# ---------------------------------------------------------------------------
# P4b: fsdp_workers.py — robust model class selection + ConditionalGeneration
# fallback (e.g. Qwen3_5ForConditionalGeneration not registered under
# AutoModelForCausalLM).
# ---------------------------------------------------------------------------
patch(
    "workers/fsdp_workers.py",
    """            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
""",
    """            # [t5x-patch] `_model_mapping` is private API and AutoModelForVision2Seq is
            # removed in transformers 5.x; fall back to AutoModelForCausalLM, then to the
            # class named in config.architectures (e.g. Qwen3_5ForConditionalGeneration),
            # then to AutoModel.
            actor_module_class = AutoModelForCausalLM
            if AutoModelForVision2Seq is not None:
                try:
                    if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                        actor_module_class = AutoModelForVision2Seq
                except AttributeError:
                    pass

            _from_pretrained_kwargs = dict(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            try:
                actor_module = actor_module_class.from_pretrained(**_from_pretrained_kwargs)
            except (ValueError, KeyError):
                import transformers as _hf_transformers

                _archs = getattr(actor_model_config, "architectures", None) or []
                _fallback_cls = getattr(_hf_transformers, _archs[0], None) if _archs else None
                if _fallback_cls is None:
                    from transformers import AutoModel as _fallback_cls
                print(f"[t5x-patch] AutoModelForCausalLM cannot load {_archs}, falling back to {_fallback_cls.__name__}")
                actor_module = _fallback_cls.from_pretrained(**_from_pretrained_kwargs)
""",
    "P4b model class fallback",
)

# ---------------------------------------------------------------------------
# P5: fsdp_utils.py — tolerate wrap-policy class names that are absent from the
# loaded module. Qwen3.5 declares _no_split_modules = {Qwen3_5DecoderLayer,
# Qwen3_5VisionBlock}; a text-only load has no vision blocks, and verl aborts as
# soon as one name is missing. Only fail when NONE of the names resolve.
# ---------------------------------------------------------------------------
patch(
    "utils/fsdp_utils.py",
    """        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
""",
    """        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                # [t5x-patch] absent sub-module class (e.g. vision blocks on a
                # text-only load) — skip instead of aborting.
                print(f"[t5x-patch] wrap policy: class {layer_class} not present in model, skipping")
            else:
                transformer_cls_to_wrap.add(transformer_cls)
        if not transformer_cls_to_wrap:
            raise Exception("Could not find the transformer layer class to wrap in the model.")
""",
    "P5 wrap-policy tolerate missing classes",
)

# ---------------------------------------------------------------------------
# P6: chat_scheduler.py — never send tools=[]/None to the rollout server.
# vLLM >= 0.21 rejects an empty tools array with HTTP 400 ("`tools` must not
# be an empty array"), which older servers (<= 0.19) silently accepted. The
# scheduler always forwards completion_callback.tool_schemas, which is [] for
# tool-free agents (multi_turn.tool_config_path == '').
# ---------------------------------------------------------------------------
patch(
    "workers/rollout/chat_scheduler.py",
    """            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")""",
    """            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            if not chat_complete_request.get("tools"):  # [t5x-patch] P6: vllm>=0.21 400s on tools=[]
                chat_complete_request.pop("tools", None)
            extra_headers = chat_complete_request.pop("extra_headers")""",
    "P6 drop empty tools field for vllm 0.21+",
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    verl_root = Path(sys.argv[1])
    check_only = "--check" in sys.argv[2:]
    assert (verl_root / "version").exists() or (verl_root / "protocol.py").exists(), f"{verl_root} does not look like a verl package root"

    n_applied = n_skipped = n_failed = 0
    for rel_path, old, new, name in PATCHES:
        target = verl_root / rel_path
        if not target.exists():
            print(f"[FAIL] {name}: {target} missing")
            n_failed += 1
            continue
        text = target.read_text()
        if new in text:
            print(f"[skip] {name}: already applied")
            n_skipped += 1
            continue
        if old not in text:
            print(f"[FAIL] {name}: anchor not found in {target}")
            n_failed += 1
            continue
        if check_only:
            print(f"[ok  ] {name}: would apply")
            n_applied += 1
            continue
        target.write_text(text.replace(old, new, 1))
        print(f"[ok  ] {name}: applied to {target}")
        n_applied += 1

    print(f"\napplied={n_applied} skipped={n_skipped} failed={n_failed}")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
