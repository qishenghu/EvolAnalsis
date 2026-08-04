"""vLLM worker extension for DUET external rollout servers.

Loaded into `vllm serve` via
    --worker-extension-cls duet_vllm_worker_ext.RolloutWeightReloadExtension
(start_rollout_servers.sh puts this directory on PYTHONPATH). Methods defined
here are attached to the vLLM Worker and are callable from the trainer through
the dev-mode HTTP endpoint POST /collective_rpc.

Deliberately standalone: must not import agentevolver/verl (the vllm2 env has
neither).

NOTE on Qwen3.5 (GDN hybrid) weight names: the trainer exports plain HF names
(e.g. `model.language_model.layers.3.self_attn.q_proj.weight`). vLLM's
Qwen3_5ForConditionalGeneration.load_weights() applies hf_to_vllm_mapper
(`model.language_model.` -> `language_model.model.`, `model.visual.` ->
`visual.`) and fuses stacked params (q/k/v -> qkv_proj, gate/up ->
gate_up_proj, in_proj_qkv/in_proj_z -> in_proj_qkvz, in_proj_b/in_proj_a ->
in_proj_ba), so `num_loaded_params` is SMALLER than the number of exported
tensors even when everything loads (3 fused names per decoder layer).
`mtp.*` draft-head tensors are skipped by design (AutoWeightsLoader
skip_prefixes) — the rollout server never runs the MTP head.
"""

import glob
import os


class RolloutWeightReloadExtension:
    """Reload model weights in-place from a directory of safetensors files.

    Used for per-step weight sync: the trainer exports the FSDP actor state as
    bf16 safetensors (HF weight names), then calls
    collective_rpc("reload_weights_from_disk", args=[load_dir]).
    """

    def reload_weights_from_disk(self, load_dir: str):
        from safetensors.torch import safe_open

        files = sorted(glob.glob(os.path.join(load_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors under {load_dir}")

        model = self.model_runner.model

        exported_names = []

        # The trainer loads the text-only class, so it exports "model.layers.*",
        # while a multimodal checkpoint (Qwen3.5) is served under a wrapper that
        # wants "model.language_model.*". Re-prefix only when the served model
        # really has that sub-module, so text-only serving is untouched.
        # vLLM's multimodal wrapper exposes `language_model` at the top level
        # (HF nests it as model.language_model); either shape means the loader
        # expects checkpoint-style "model.language_model.*" names.
        inner = getattr(model, "model", None)
        needs_lm_prefix = hasattr(model, "language_model") or (
            inner is not None and hasattr(inner, "language_model"))

        def _remap(name):
            if (needs_lm_prefix and name.startswith("model.")
                    and not name.startswith(("model.language_model.", "model.visual."))):
                return "model.language_model." + name[len("model."):]
            return name

        def _weights_iter():
            for f in files:
                with safe_open(f, framework="pt", device="cpu") as sf:
                    for name in sf.keys():
                        exported_names.append(name)
                        yield _remap(name), sf.get_tensor(name)

        loaded = model.load_weights(_weights_iter())
        n_loaded = len(loaded) if loaded is not None else -1

        # Diagnostics: which exported top-level prefixes exist, and how many
        # tensors were intentionally skipped (mtp.* draft head).
        n_mtp_skipped = sum(1 for n in exported_names if n.startswith("mtp."))
        prefix_counts = {}
        for n in exported_names:
            top = ".".join(n.split(".")[:2])
            prefix_counts[top] = prefix_counts.get(top, 0) + 1

        return {
            "num_files": len(files),
            "num_exported_tensors": len(exported_names),
            "num_loaded_params": n_loaded,
            "num_mtp_skipped": n_mtp_skipped,
            "exported_prefix_counts": prefix_counts,
        }

    def param_checksums(self, patterns):
        """Debug/proof tool: {vllm_param_name: float32 sum} for every model
        parameter whose name contains any of the given substrings.

        Lets the trainer/tests PROVE a reload actually changed GPU tensors:
        collective_rpc("param_checksums", args=[["layers.3.self_attn.qkv_proj"]]).
        Accepts a list of substrings or a single comma-separated string.
        """
        import torch

        if isinstance(patterns, str):
            patterns = [p for p in patterns.split(",") if p]
        model = self.model_runner.model
        out = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if any(pat in name for pat in patterns):
                    out[name] = float(p.detach().float().sum().item())
        return out

    def param_names_sample(self, pattern: str = "", limit: int = 50):
        """Debug: list up to `limit` parameter names containing `pattern`."""
        model = self.model_runner.model
        names = [n for n, _ in model.named_parameters() if pattern in n]
        return {"num_matching": len(names), "sample": names[:limit]}
