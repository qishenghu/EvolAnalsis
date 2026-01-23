# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Modifications copyright 2025 Alibaba Tongyi EconML Lab. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

__all__ = ['HETDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor


class HETDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, **kwargs):
        """
        Initializes the HETDataParallelPPOActor with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments passed to the superclass constructor.
        """
        super().__init__(**kwargs)
        # 7.8: gap->beta scheduler state (kept inside actor; default disabled)
        self._gap_beta_ema: float | None = None
        self._gap_beta_state: float | None = None  # hysteresis state: current beta value
        self._gap_beta_updates: int = 0
        # Teacher-vs-onpolicy gradient direction diagnostics (default disabled)
        self._teacher_grad_dir_diag_updates: int = 0

    def _maybe_log_teacher_onpolicy_grad_dir(
        self,
        *,
        on_pg_loss: torch.Tensor,
        teacher_off_pg_loss: torch.Tensor,
        has_teacher_data: bool,
        append_fn,
    ) -> None:
        """
        Optional diagnostics: compare gradient directions induced by
        - on-policy policy gradient loss (on_pg_loss)
        - teacher policy gradient loss (teacher_off_pg_loss)

        We approximate the full gradient vector by selecting a small subset of parameters
        (e.g., lm_head) to keep overhead manageable under FSDP/large models.
        """
        try:
            # Default ON: this is an analysis-oriented repo; allow users to disable if needed.
            if not bool(self.config.get("teacher_grad_dir_diag_enable", True)):
                return
            if (not has_teacher_data) or (teacher_off_pg_loss is None) or (on_pg_loss is None):
                return
            if (not torch.is_tensor(on_pg_loss)) or (not torch.is_tensor(teacher_off_pg_loss)):
                return

            # Computing autograd.grad twice can be expensive; default to a moderate interval.
            # (Metrics can still be logged every step by reusing the most recent computed values if desired.)
            interval = int(self.config.get("teacher_grad_dir_diag_interval", 20))
            interval = max(1, interval)
            if (self._teacher_grad_dir_diag_updates % interval) != 0:
                self._teacher_grad_dir_diag_updates += 1
                return
            self._teacher_grad_dir_diag_updates += 1

            # Avoid vocab-tied / embedding / lm_head parameters by default:
            # they are huge and can be dominated by token-frequency effects, and are expensive under FSDP.
            name_excludes = self.config.get(
                "teacher_grad_dir_diag_param_name_excludes",
                [
                    "lm_head",
                    "embed",
                    "wte",
                    "word_embeddings",
                    "tok_embeddings",
                    "token_embeddings",
                    "shared",
                    "output_embedding",
                ],
            )
            if isinstance(name_excludes, str):
                name_excludes = [name_excludes]
            name_excludes = [str(s) for s in name_excludes if str(s)]

            max_param_numel = int(self.config.get("teacher_grad_dir_diag_max_param_numel", 200_000))
            max_param_numel = max(1, max_param_numel)
            max_params = int(self.config.get("teacher_grad_dir_diag_max_params", 4))
            max_params = max(1, min(32, max_params))

            named_params = [(n, p) for n, p in self.actor_module.named_parameters() if p is not None and p.requires_grad]

            # Parse layer index from common LLM naming patterns:
            # - model.layers.{i}.*
            # - layers.{i}.*
            def _extract_layer_idx(name: str) -> int | None:
                try:
                    import re
                    m = re.search(r"(?:^|\\.)model\\.layers\\.(\\d+)\\.", name)
                    if m:
                        return int(m.group(1))
                    m = re.search(r"(?:^|\\.)layers\\.(\\d+)\\.", name)
                    if m:
                        return int(m.group(1))
                except Exception:
                    return None
                return None

            # Determine last layer index present in this model (best-effort).
            layer_idxs = []
            for n, _p in named_params:
                li = _extract_layer_idx(str(n))
                if li is not None:
                    layer_idxs.append(li)
            max_layer_idx = max(layer_idxs) if layer_idxs else None

            def _select_params(
                include_substr: list[str],
                probe_max_params: int,
                *,
                layer_filter_mode: str = "all",
                last_k: int = 2,
            ) -> list[tuple[str, torch.Tensor]]:
                include_substr = [str(s) for s in include_substr if str(s)]
                # size + exclude filters
                cand = []
                for n, p in named_params:
                    try:
                        if int(p.numel()) > max_param_numel:
                            continue
                    except Exception:
                        continue
                    if name_excludes and any(x in n for x in name_excludes):
                        continue
                    # optional layer filtering
                    lf = str(layer_filter_mode).lower().strip()
                    if lf in ("last_k", "last", "tail") and max_layer_idx is not None:
                        k = int(last_k) if int(last_k) > 0 else 2
                        li = _extract_layer_idx(str(n))
                        if li is not None:
                            if li < (int(max_layer_idx) - k + 1):
                                continue
                        else:
                            # keep non-layer params (e.g., final norms) by default
                            pass
                    cand.append((n, p))
                # include filter
                if include_substr:
                    cand2 = [(n, p) for (n, p) in cand if any(s in n for s in include_substr)]
                else:
                    cand2 = cand
                if not cand2:
                    cand2 = cand
                # pick smallest
                cand2 = sorted(cand2, key=lambda x: int(x[1].numel()))[:probe_max_params]
                return cand2

            def _compute_probe(probe_name: str, include_substr: list[str]) -> dict[str, float] | None:
                # Per-probe layer filtering
                if probe_name == "proj":
                    lf = self.config.get("teacher_grad_dir_diag_layer_filter_mode_proj", "last_k")
                    lk = int(self.config.get("teacher_grad_dir_diag_last_k_proj", 2))
                else:
                    lf = self.config.get("teacher_grad_dir_diag_layer_filter_mode_ln", "all")
                    lk = int(self.config.get("teacher_grad_dir_diag_last_k_ln", 2))
                sel = _select_params(
                    include_substr=include_substr,
                    probe_max_params=max_params,
                    layer_filter_mode=lf,
                    last_k=lk,
                )
                params = [p for (_, p) in sel]
                if not params:
                    return None
                g_on = torch.autograd.grad(on_pg_loss, params, retain_graph=True, allow_unused=True)
                g_te = torch.autograd.grad(teacher_off_pg_loss, params, retain_graph=True, allow_unused=True)

                def _pack(gs):
                    chunks = []
                    for g in gs:
                        if g is None:
                            continue
                        chunks.append(g.detach().reshape(-1).float())
                    if not chunks:
                        return None
                    return torch.cat(chunks, dim=0)

                v_on = _pack(g_on)
                v_te = _pack(g_te)
                if v_on is None or v_te is None:
                    return None
                eps = 1e-12
                dot = (v_on * v_te).sum()
                n_on = torch.linalg.vector_norm(v_on)
                n_te = torch.linalg.vector_norm(v_te)
                cos = dot / (n_on * n_te + eps)
                out = {
                    f"grad_dir/{probe_name}/cos": float(cos.item()),
                    f"grad_dir/{probe_name}/dot": float(dot.item()),
                    f"grad_dir/{probe_name}/norm_on": float(n_on.item()),
                    f"grad_dir/{probe_name}/norm_teacher": float(n_te.item()),
                    f"grad_dir/{probe_name}/param_count": float(len(params)),
                    f"grad_dir/{probe_name}/vec_dim": float(v_on.numel()),
                }
                return out

            # Two-probe default:
            # - ln: cheap stability probe
            # - proj: more behavior-related small matrices (still filtered by max_param_numel)
            ln_contains = self.config.get("teacher_grad_dir_diag_param_name_contains_ln", ["norm", "ln", "layernorm"])
            # Behavior-related lightweight probe:
            # - For many LLMs (incl. Qwen2.5-3B), projection weights (o_proj/down_proj) are huge and may be filtered by
            #   max_param_numel. Biases are much smaller and still reflect behavior-related updates.
            proj_contains = self.config.get("teacher_grad_dir_diag_param_name_contains_proj", ["q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias", "down_proj.bias"])
            if isinstance(ln_contains, str):
                ln_contains = [ln_contains]
            if isinstance(proj_contains, str):
                proj_contains = [proj_contains]

            out_all: dict[str, float] = {
                "grad_dir/interval": float(interval),
                "grad_dir/max_param_numel": float(max_param_numel),
            }
            o1 = _compute_probe("ln", list(ln_contains))
            if o1:
                out_all.update(o1)
            o2 = _compute_probe("proj", list(proj_contains))
            if o2:
                out_all.update(o2)
            if len(out_all) > 2:
                append_fn(out_all)
        except Exception:
            # diagnostics must never break training
            return

    def _gap_beta_scheduler(
        self,
        *,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        exp_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        update_state: bool,
    ) -> tuple[float | None, dict[str, float]]:
        """
        7.8: Adjust teacher policy shaping beta based on advantage gap.

        We define per-sample scalar advantage as the mean over response tokens:
          adv_scalar[i] = mean_t advantages[i,t] over response_mask.

        Then advantage gap (scalar) is:
          gap = mean(adv_scalar[teacher_samples]) - mean(adv_scalar[onpolicy_samples])

        NOTE:
        - This is a scheduler (no grad), intended to be stable even when tasks differ each step.
        - The result is a *scalar* beta override applied to teacher shaping only.
        """
        diag: dict[str, float] = {}
        if not self.config.get("teacher_gap_beta_enable", False):
            return None, diag

        if advantages is None or (not torch.is_tensor(advantages)):
            return None, diag
        if teacher_mask is None or (not torch.is_tensor(teacher_mask)):
            return None, diag
        if exp_mask is None or (not torch.is_tensor(exp_mask)):
            return None, diag

        # per-sample scalar advantage (token-mean on response tokens)
        rm = response_mask.float()
        denom = rm.sum(dim=-1).clamp(min=1.0)
        adv_scalar = (advantages * rm).sum(dim=-1) / denom  # (bs,)

        is_teacher = (teacher_mask.sum(dim=-1) > 0)
        is_off = (exp_mask.sum(dim=-1) > 0)
        is_on = ~is_off

        if (not is_teacher.any()) or (not is_on.any()):
            return None, diag

        teacher_mean = adv_scalar[is_teacher].mean()
        on_mean = adv_scalar[is_on].mean()
        gap = (teacher_mean - on_mean).detach().float().item()
        diag["adv_gap"] = float(gap)

        use_ema = bool(self.config.get("teacher_gap_beta_use_ema", True))
        use_hyst = bool(self.config.get("teacher_gap_beta_use_hysteresis", True))

        # EMA update
        gap_used = gap
        if use_ema:
            decay = float(self.config.get("teacher_gap_beta_ema_decay", 0.9))
            decay = min(max(decay, 0.0), 0.9999)
            if self._gap_beta_ema is None:
                ema = gap
            else:
                ema = decay * float(self._gap_beta_ema) + (1.0 - decay) * gap
            diag["adv_gap_ema"] = float(ema)
            gap_used = float(ema)
            if update_state:
                self._gap_beta_ema = float(ema)

        beta_strong = float(self.config.get("teacher_gap_beta_beta_strong", 0.05))  # stronger teacher
        beta_weak = float(self.config.get("teacher_gap_beta_beta_weak", 0.10))      # weaker teacher
        # normalize ordering
        beta_low = min(beta_strong, beta_weak)
        beta_high = max(beta_strong, beta_weak)

        beta_out: float
        switched = 0.0

        if use_hyst:
            hi = float(self.config.get("teacher_gap_beta_hysteresis_hi", 0.55))
            lo = float(self.config.get("teacher_gap_beta_hysteresis_lo", 0.45))
            if lo > hi:
                lo, hi = hi, lo

            # state init
            cur = self._gap_beta_state
            if cur is None:
                # choose initial state by comparing to mid threshold
                mid = 0.5 * (hi + lo)
                cur = beta_low if gap_used > mid else beta_high
                if update_state:
                    self._gap_beta_state = float(cur)

            # hysteresis transitions:
            # - if gap is high (teacher much better), strengthen teacher => smaller beta
            # - if gap is low, weaken teacher => larger beta
            if float(cur) == beta_high and gap_used > hi:
                beta_out = beta_low
                switched = 1.0
            elif float(cur) == beta_low and gap_used < lo:
                beta_out = beta_high
                switched = 1.0
            else:
                beta_out = float(cur)

            if update_state:
                self._gap_beta_state = float(beta_out)
        else:
            thr = float(self.config.get("teacher_gap_beta_threshold", 0.5))
            beta_out = beta_low if gap_used > thr else beta_high

        if update_state:
            self._gap_beta_updates += 1

        diag["beta"] = float(beta_out)
        diag["switched"] = float(switched)
        diag["use_ema"] = 1.0 if use_ema else 0.0
        diag["use_hysteresis"] = 1.0 if use_hyst else 0.0
        diag["updates"] = float(self._gap_beta_updates)
        return float(beta_out), diag

    def update_policy(self, data: DataProto):
        """
        Updates the policy of the reinforcement learning model using the Proximal Policy Optimization (PPO) algorithm.
        Handles data in mini-batches and micro-batches, and computes various losses including policy loss, entropy loss,
        and KL divergence loss.

        Args:
            data (DataProto): The data containing the necessary information for updating the policy.
        """
        # make sure we are in training mode
        self.actor_module.train()  # ⭐ Ensure the actor module is in training mode

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        ##################
        # ANNI
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
            select_keys.append("exp_mask")
        # ⭐ Teacher replay / gating extras (keep backward compatible)
        if isinstance(data, DataProto):
            if "teacher_mask" in data.batch:
                select_keys.append("teacher_mask")
            if "teacher_loss_scale" in data.batch:
                select_keys.append("teacher_loss_scale")
        # if multi_turn:
        #     select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        ##################

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # ------------------------------------------------------------------
        # Gradient-direction diagnostics (teacher vs on-policy), per-step aggregated.
        #
        # Definition:
        # - g_on_step      = sum over all micro-batches (epoch==0) of ∇θ on_pg_loss
        # - g_teacher_step = sum over all micro-batches (epoch==0) of ∇θ teacher_off_pg_loss
        # Then we log cosine/dot/norm on a small parameter subset (two probes: ln / proj).
        #
        # Rationale:
        # - Stable and matches "per-step" semantics; independent of micro-batch ordering.
        # - Avoids the earlier pitfall where teacher might not be in micro-batch 0.
        # ------------------------------------------------------------------
        grad_dir_enable = bool(self.config.get("teacher_grad_dir_diag_enable", True))
        grad_dir_interval = int(self.config.get("teacher_grad_dir_diag_interval", 1))
        grad_dir_interval = max(1, grad_dir_interval)
        grad_dir_should_run = False
        if grad_dir_enable:
            try:
                grad_dir_should_run = (int(self._teacher_grad_dir_diag_updates) % grad_dir_interval) == 0
            except Exception:
                grad_dir_should_run = True
            try:
                self._teacher_grad_dir_diag_updates += 1
            except Exception:
                pass

        grad_dir_step = {
            "run": bool(grad_dir_enable and grad_dir_should_run),
            "teacher_samples": 0,
            "on_samples": 0,
            "missing_reason": 0.0,  # 0=ok, 2=no_teacher, 3=interval_skip, 4=disabled, 5=failed
        }

        grad_dir_acc_ln = None
        grad_dir_acc_proj = None
        grad_dir_params_ln = None
        grad_dir_params_proj = None
        grad_dir_union_params = None
        grad_dir_union_idx_ln = None
        grad_dir_union_idx_proj = None
        grad_dir_union_idx_flat = None
        grad_dir_union_acc = None  # {"on":[...], "te":[...]}

        if grad_dir_step["run"]:
            try:
                import re
                from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP

                def _extract_layer_idx(name: str) -> int | None:
                    m = re.search(r"(?:^|\\.)model\\.layers\\.(\\d+)\\.", name)
                    if m:
                        return int(m.group(1))
                    m = re.search(r"(?:^|\\.)layers\\.(\\d+)\\.", name)
                    if m:
                        return int(m.group(1))
                    return None

                name_excludes = self.config.get(
                    "teacher_grad_dir_diag_param_name_excludes",
                    ["lm_head", "embed", "wte", "word_embeddings", "tok_embeddings", "token_embeddings", "shared"],
                )
                if isinstance(name_excludes, str):
                    name_excludes = [name_excludes]
                name_excludes = [str(s) for s in list(name_excludes) if str(s)]

                max_param_numel = int(self.config.get("teacher_grad_dir_diag_max_param_numel", 200_000))
                max_param_numel = max(1, max_param_numel)
                max_params = int(self.config.get("teacher_grad_dir_diag_max_params", 4))
                max_params = max(1, min(32, max_params))

                ln_contains = self.config.get("teacher_grad_dir_diag_param_name_contains_ln", ["norm", "ln", "layernorm"])
                if isinstance(ln_contains, str):
                    ln_contains = [ln_contains]
                ln_contains = [str(s) for s in list(ln_contains) if str(s)]

                proj_contains = self.config.get(
                    "teacher_grad_dir_diag_param_name_contains_proj",
                    ["q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias", "down_proj.bias"],
                )
                if isinstance(proj_contains, str):
                    proj_contains = [proj_contains]
                proj_contains = [str(s) for s in list(proj_contains) if str(s)]

                lf_ln = str(self.config.get("teacher_grad_dir_diag_layer_filter_mode_ln", "all")).lower().strip()
                lf_proj = str(self.config.get("teacher_grad_dir_diag_layer_filter_mode_proj", "last_k")).lower().strip()
                last_k_proj = int(self.config.get("teacher_grad_dir_diag_last_k_proj", 2))
                last_k_proj = max(1, last_k_proj)

                named_params = [(n, p) for n, p in self.actor_module.named_parameters() if p is not None and p.requires_grad]
                layer_idxs = []
                for n, _p in named_params:
                    li = _extract_layer_idx(str(n))
                    if li is not None:
                        layer_idxs.append(li)
                max_layer_idx = max(layer_idxs) if layer_idxs else None

                def _select_params(include_substr: list[str], *, layer_filter_mode: str, last_k: int) -> list[tuple[str, torch.Tensor]]:
                    cand: list[tuple[str, torch.Tensor]] = []
                    for n, p in named_params:
                        n = str(n)
                        try:
                            if int(p.numel()) > max_param_numel:
                                continue
                        except Exception:
                            continue
                        if name_excludes and any(x in n for x in name_excludes):
                            continue
                        if layer_filter_mode in ("last_k", "last", "tail") and max_layer_idx is not None:
                            li = _extract_layer_idx(n)
                            if li is not None and li < (int(max_layer_idx) - int(last_k) + 1):
                                continue
                        if include_substr and (not any(s in n for s in include_substr)):
                            continue
                        cand.append((n, p))
                    cand = sorted(cand, key=lambda x: int(x[1].numel()))[:max_params]
                    return cand

                sel_ln = _select_params(ln_contains, layer_filter_mode=lf_ln, last_k=last_k_proj)
                sel_proj = _select_params(proj_contains, layer_filter_mode=lf_proj, last_k=last_k_proj)
                grad_dir_params_ln = [p for (_n, p) in sel_ln]
                grad_dir_params_proj = [p for (_n, p) in sel_proj]

                # Build union param list to avoid repeated autograd.grad calls per probe.
                # Keep stable order: ln params then proj params, de-duplicated by id().
                union: list[torch.Tensor] = []
                idx_ln: list[int] = []
                idx_proj: list[int] = []
                seen: dict[int, int] = {}

                def _add(ps: list[torch.Tensor], out_idx: list[int]):
                    for p in ps:
                        pid = id(p)
                        if pid in seen:
                            out_idx.append(seen[pid])
                            continue
                        seen[pid] = len(union)
                        out_idx.append(len(union))
                        union.append(p)

                _add(list(grad_dir_params_ln or []), idx_ln)
                _add(list(grad_dir_params_proj or []), idx_proj)

                grad_dir_union_params = union
                grad_dir_union_idx_ln = idx_ln
                grad_dir_union_idx_proj = idx_proj
                grad_dir_union_idx_flat = []

                # Best-effort mapping for FSDP flat_param -> which original param FQNs / layer indices it contains.
                # Works even with use_orig_params=False, but relies on internal FSDP handle attrs (may vary by torch version).
                flat_meta: dict[int, dict[str, float]] = {}
                try:
                    fsdp_mods = [m for m in self.actor_module.modules() if isinstance(m, _FSDP)]
                    for m in fsdp_mods:
                        h = getattr(m, "_handle", None)
                        fp = None
                        if h is not None:
                            fp = getattr(h, "flat_param", None)
                        if fp is None:
                            # some versions attach flat_param directly
                            fp = getattr(m, "flat_param", None)
                        if fp is None:
                            continue
                        fqns = None
                        if h is not None:
                            fqns = getattr(h, "_fqns", None) or getattr(h, "fqns", None)
                        if fqns is None:
                            fqns = getattr(fp, "_fqns", None)
                        li_list: list[int] = []
                        if fqns is not None:
                            try:
                                for q in list(fqns):
                                    li = _extract_layer_idx(str(q))
                                    if li is not None:
                                        li_list.append(int(li))
                            except Exception:
                                li_list = []
                        if li_list:
                            flat_meta[id(fp)] = {
                                "layer_min": float(min(li_list)),
                                "layer_max": float(max(li_list)),
                                "layer_count": float(len(set(li_list))),
                            }
                except Exception:
                    flat_meta = {}

                # Fallback for FSDP use_orig_params=False:
                # named_parameters may only expose flat_param(s), making ln/proj substring matching impossible.
                # In that case, use last-K flat_param tensors as a "flat" probe.
                if (not grad_dir_union_params) and bool(self.config.get("teacher_grad_dir_diag_allow_flat_param", True)):
                    flat = [(n, p) for (n, p) in named_params if ("flat_param" in str(n)) and p is not None and p.requires_grad]
                    if flat:
                        k = int(self.config.get("teacher_grad_dir_diag_flat_last_k", 2))
                        k = max(1, k)
                        pick_mode = str(self.config.get("teacher_grad_dir_diag_flat_pick_mode", "last_k")).lower().strip()
                        if pick_mode in ("smallest_k", "min_k"):
                            flat_sel = sorted(flat, key=lambda x: int(x[1].numel()))[:k]
                        elif pick_mode in ("layer_last_k", "layer_tail_k"):
                            # Prefer flat_params that contain the last-K transformer layers, if we can infer layer indices.
                            last_layers_k = int(self.config.get("teacher_grad_dir_diag_flat_last_k_layers", 2))
                            last_layers_k = max(1, last_layers_k)
                            # estimate model max layer idx from whatever names we have
                            max_li = max_layer_idx
                            if max_li is None and flat_meta:
                                max_li = int(max(v.get("layer_max", 0.0) for v in flat_meta.values()))
                            if max_li is not None and flat_meta:
                                keep_min = int(max_li) - int(last_layers_k) + 1
                                cand = []
                                for n, p in flat:
                                    meta = flat_meta.get(id(p), None)
                                    if meta is None:
                                        continue
                                    if int(meta.get("layer_max", -1.0)) >= keep_min:
                                        cand.append((n, p))
                                flat_sel = cand[-k:] if cand else flat[-k:]
                            else:
                                flat_sel = flat[-k:]
                        else:
                            # default: last_k
                            flat_sel = flat[-k:]
                        for (_n, p) in flat_sel:
                            _add([p], grad_dir_union_idx_flat)
                        grad_dir_union_params = union

                        # store a compact summary for logging (layer min/max/count across selected flat params)
                        try:
                            mins = []
                            maxs = []
                            cnts = []
                            for (_n, p) in flat_sel:
                                meta = flat_meta.get(id(p), None)
                                if meta is None:
                                    continue
                                mins.append(int(meta.get("layer_min", 0.0)))
                                maxs.append(int(meta.get("layer_max", 0.0)))
                                cnts.append(int(meta.get("layer_count", 0.0)))
                            if mins and maxs:
                                grad_dir_step["flat_layer_min"] = float(min(mins))
                                grad_dir_step["flat_layer_max"] = float(max(maxs))
                                grad_dir_step["flat_layer_count"] = float(sum(cnts))
                        except Exception:
                            pass

                if grad_dir_union_params:
                    grad_dir_union_acc = {
                        "on": [torch.zeros_like(p, dtype=torch.float32) for p in grad_dir_union_params],
                        "te": [torch.zeros_like(p, dtype=torch.float32) for p in grad_dir_union_params],
                    }
            except Exception:
                grad_dir_step["run"] = False
                grad_dir_step["missing_reason"] = 5.0

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()  # ⭐ Zero the gradients before computing the new ones
                # Accumulate per-trajectory teacher diagnostics across micro-batches (for correct step-level stats)
                teacher_traj_value_acc: dict[str, list[torch.Tensor]] = {}

                # ------------------------------------------------------------------
                # 7.8: gap->beta (compute once per mini-batch; apply to all micro-batches)
                # ------------------------------------------------------------------
                gap_beta_override: float | None = None
                gap_beta_diag: dict[str, float] = {}
                try:
                    # Build masks on (likely) CPU tensors to avoid GPU sync.
                    mb = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    mb_responses = mb["responses"]
                    mb_response_length = mb_responses.size(1)
                    mb_attention_mask = mb["attention_mask"]
                    if multi_turn:
                        mb_response_mask = mb["loss_mask"][:, -mb_response_length:]
                    else:
                        mb_response_mask = mb_attention_mask[:, -mb_response_length:]
                    mb_adv = mb["advantages"][:, -mb_response_length:]
                    mb_exp = mb["exp_mask"][:, -mb_response_length:]
                    mb_teacher = mb.get("teacher_mask", None)
                    if mb_teacher is not None:
                        mb_teacher = mb_teacher[:, -mb_response_length:]
                    # only update EMA/state on the first PPO epoch for this batch
                    gap_beta_override, gap_beta_diag = self._gap_beta_scheduler(
                        advantages=mb_adv,
                        response_mask=mb_response_mask,
                        exp_mask=mb_exp,
                        teacher_mask=mb_teacher,
                        update_state=(epoch == 0),
                    )
                except Exception:
                    gap_beta_override, gap_beta_diag = None, {}

                for micro_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]


                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)  # ⭐ Forward pass to get entropy and log probabilities

                    ##################
                    # ANNI 0814
                    from .het_core_algos import het_compute_token_on_off_policy_loss, het_compute_teacher_aware_loss, dapo_compute_policy_loss
                    off_cliprange_high = self.config.get("off_cliprange_high", 1.0)
                    exp_mask = data["exp_mask"][:, -response_length:]
                    
                    # ⭐ Off-policy policy shaping configuration
                    off_policy_shaping_mode = self.config.get("off_policy_shaping_mode", "higher_clip_bound")
                    off_policy_shaping_beta = self.config.get("off_policy_shaping_beta", 0.1)
                    
                    # ⭐ DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) configuration
                    use_dapo = self.config.get("use_dapo", False)
                    
                    # ⭐ Teacher Experience configuration
                    teacher_mask = data.get("teacher_mask", None)
                    if teacher_mask is not None:
                        teacher_mask = teacher_mask[:, -response_length:]
                        # 检查是否有任何 teacher 数据
                        has_teacher_data = teacher_mask.sum() > 0
                    else:
                        has_teacher_data = False
                    teacher_loss_scale = data.get("teacher_loss_scale", None)
                    if teacher_loss_scale is not None and torch.is_tensor(teacher_loss_scale):
                        teacher_loss_scale = teacher_loss_scale[:, -response_length:]
                    
                    if use_dapo:
                        # Use DAPO's decoupled asymmetric clipping mechanism
                        # ⭐ Experience-Replay compatible: pass off-policy shaping parameters
                        ret_dict = dapo_compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            exp_mask=exp_mask,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            # Off-policy (Experience Replay) settings
                            off_policy_shaping_mode=off_policy_shaping_mode,
                            off_policy_shaping_beta=off_policy_shaping_beta,
                        )  # ⭐ Compute policy loss using DAPO's Clip-Higher mechanism (Experience-Replay compatible)
                    elif has_teacher_data:
                        # ⭐ Teacher Experience: 使用 het_compute_teacher_aware_loss
                        # 获取 teacher 相关配置
                        teacher_use_log_prob = self.config.get("teacher_use_log_prob", False)
                        teacher_policy_shaping_enable = self.config.get("teacher_policy_shaping_enable", True)
                        teacher_policy_shaping_mode = self.config.get("teacher_policy_shaping_mode", "p_div_p_beta")
                        teacher_policy_shaping_beta = self.config.get("teacher_policy_shaping_beta", 0.1)
                        teacher_use_clip = self.config.get("teacher_use_clip", False)
                        # ⭐ 7.7: TER sequence-level β schedule (soft-min teacher confidence)
                        teacher_seq_beta_enable = self.config.get("teacher_seq_beta_enable", False)
                        teacher_seq_beta_alpha = self.config.get("teacher_seq_beta_alpha", -5.0)
                        teacher_seq_beta_c0 = self.config.get("teacher_seq_beta_c0", 0.25)
                        teacher_seq_beta_temperature = self.config.get("teacher_seq_beta_temperature", 0.05)
                        teacher_seq_beta_gate_space = self.config.get("teacher_seq_beta_gate_space", "prob")
                        teacher_seq_beta_logc0 = self.config.get("teacher_seq_beta_logc0", None)
                        teacher_seq_beta_log_temperature = self.config.get("teacher_seq_beta_log_temperature", 0.5)
                        teacher_seq_beta_conf_mode = self.config.get("teacher_seq_beta_conf_mode", "gen_mean_prob")
                        teacher_seq_beta_logp_q = self.config.get("teacher_seq_beta_logp_q", 0.10)
                        teacher_seq_beta_max_tokens_per_traj = self.config.get("teacher_seq_beta_max_tokens_per_traj", 4096)
                        teacher_seq_beta_beta_min = self.config.get("teacher_seq_beta_beta_min", 0.05)
                        teacher_seq_beta_beta_max = self.config.get("teacher_seq_beta_beta_max", 0.30)
                        teacher_seq_beta_p_min = self.config.get("teacher_seq_beta_p_min", 1e-4)
                        teacher_seq_beta_stop_grad = self.config.get("teacher_seq_beta_stop_grad", True)
                        # ⭐ 7.6 AG-PM: Advantage-Gated Probability Margin (optional; default disabled)
                        # 双门控机制：只向"好老师"学习，但只学到"懂了为止"
                        teacher_ag_pm_enable = self.config.get("teacher_ag_pm_enable", False)
                        teacher_ag_pm_adv_threshold = self.config.get("teacher_ag_pm_adv_threshold", 0.4)
                        teacher_ag_pm_adv_temperature = self.config.get("teacher_ag_pm_adv_temperature", 0.2)
                        teacher_ag_pm_adv_min = self.config.get("teacher_ag_pm_adv_min", 0.5)
                        teacher_ag_pm_adv_max = self.config.get("teacher_ag_pm_adv_max", 1.0)
                        teacher_ag_pm_prob_max = self.config.get("teacher_ag_pm_prob_max", 0.9)
                        teacher_ag_pm_prob_temperature = self.config.get("teacher_ag_pm_prob_temperature", 0.02)
                        teacher_ag_pm_prob_min = self.config.get("teacher_ag_pm_prob_min", 0.0)
                        teacher_ag_pm_prob_max_gate = self.config.get("teacher_ag_pm_prob_max_gate", 1.0)
                        teacher_ag_pm_stop_grad = self.config.get("teacher_ag_pm_stop_grad", True)
                        
                        # ⭐ 7.8: advantage-gap scheduled beta override (only for LUFFY + p_div_p_beta)
                        if (
                            gap_beta_override is not None
                            and (not teacher_use_log_prob)
                            and teacher_policy_shaping_enable
                            and str(teacher_policy_shaping_mode).lower().strip() == "p_div_p_beta"
                            and (not self.config.get("teacher_seq_beta_enable", False))  # 7.7 has higher priority
                        ):
                            teacher_policy_shaping_beta = float(gap_beta_override)
                        
                        ret_dict = het_compute_teacher_aware_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            exp_mask=exp_mask,
                            teacher_mask=teacher_mask,  # ⭐ 传入 teacher_mask
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            off_cliprange_high=off_cliprange_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            off_policy_shaping_mode=off_policy_shaping_mode,
                            off_policy_shaping_beta=off_policy_shaping_beta,
                            # Teacher-specific settings
                            teacher_use_log_prob=teacher_use_log_prob,
                            teacher_policy_shaping_enable=teacher_policy_shaping_enable,
                            teacher_policy_shaping_mode=teacher_policy_shaping_mode,
                            teacher_policy_shaping_beta=teacher_policy_shaping_beta,
                            teacher_use_clip=teacher_use_clip,
                            teacher_loss_scale=teacher_loss_scale,
                            # 7.7: sequence-level beta schedule
                            teacher_seq_beta_enable=teacher_seq_beta_enable,
                            teacher_seq_beta_alpha=teacher_seq_beta_alpha,
                            teacher_seq_beta_c0=teacher_seq_beta_c0,
                            teacher_seq_beta_temperature=teacher_seq_beta_temperature,
                            teacher_seq_beta_gate_space=teacher_seq_beta_gate_space,
                            teacher_seq_beta_logc0=teacher_seq_beta_logc0,
                            teacher_seq_beta_log_temperature=teacher_seq_beta_log_temperature,
                            teacher_seq_beta_conf_mode=teacher_seq_beta_conf_mode,
                            teacher_seq_beta_logp_q=teacher_seq_beta_logp_q,
                            teacher_seq_beta_max_tokens_per_traj=teacher_seq_beta_max_tokens_per_traj,
                            teacher_seq_beta_beta_min=teacher_seq_beta_beta_min,
                            teacher_seq_beta_beta_max=teacher_seq_beta_beta_max,
                            teacher_seq_beta_p_min=teacher_seq_beta_p_min,
                            teacher_seq_beta_stop_grad=teacher_seq_beta_stop_grad,
                            # 7.6 AG-PM: Advantage-Gated Probability Margin
                            teacher_ag_pm_enable=teacher_ag_pm_enable,
                            teacher_ag_pm_adv_threshold=teacher_ag_pm_adv_threshold,
                            teacher_ag_pm_adv_temperature=teacher_ag_pm_adv_temperature,
                            teacher_ag_pm_adv_min=teacher_ag_pm_adv_min,
                            teacher_ag_pm_adv_max=teacher_ag_pm_adv_max,
                            teacher_ag_pm_prob_max=teacher_ag_pm_prob_max,
                            teacher_ag_pm_prob_temperature=teacher_ag_pm_prob_temperature,
                            teacher_ag_pm_prob_min=teacher_ag_pm_prob_min,
                            teacher_ag_pm_prob_max_gate=teacher_ag_pm_prob_max_gate,
                            teacher_ag_pm_stop_grad=teacher_ag_pm_stop_grad,
                        )  # ⭐ Compute teacher-aware loss (LUFFY + ExGRPO + 7.6 AG-PM)
                        # Collect raw per-trajectory values for correct step-level aggregation
                        traj_vals = ret_dict.get("teacher_diag_traj_values")
                        if isinstance(traj_vals, dict) and traj_vals:
                            for k, v in traj_vals.items():
                                if v is None or (not torch.is_tensor(v)) or v.numel() == 0:
                                    continue
                                teacher_traj_value_acc.setdefault(str(k), []).append(v.detach())
                    else:
                        # Use original het_compute_token_on_off_policy_loss
                        ret_dict = het_compute_token_on_off_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            exp_mask=exp_mask,   # (bs, response_length) ANNI add: 1 w/ exp(off-policy); 0 w/o exp(on-policy)
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            off_cliprange_high=off_cliprange_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                            off_policy_shaping_mode=off_policy_shaping_mode,
                            off_policy_shaping_beta=off_policy_shaping_beta,
                        )  # ⭐ Compute on-policy and off-policy losses
                    pg_loss = ret_dict["pg_loss"]
                    pg_losses = ret_dict["pg_losses"]
                    on_pg_losses = ret_dict["on_pg_losses"]
                    off_pg_losses = ret_dict["off_pg_losses"]
                    on_pg_loss = ret_dict["on_pg_loss"]
                    off_pg_loss = ret_dict["off_pg_loss"]
                    on_pg_clipfrac = ret_dict["on_pg_clipfrac"]
                    on_pg_clipfrac_lower = ret_dict["on_pg_clipfrac_lower"]
                    on_pg_cliphit_rate = ret_dict.get("on_pg_cliphit_rate", None)
                    off_pg_cliphit_rate = ret_dict.get("off_pg_cliphit_rate", None)
                    self_off_pg_cliphit_rate = ret_dict.get("self_off_pg_cliphit_rate", None)
                    teacher_off_pg_cliphit_rate = ret_dict.get("teacher_off_pg_cliphit_rate", None)
                    ppo_kl = ret_dict["ppo_kl"]
                    # ⭐ Teacher Experience: 提取 teacher-specific metrics（如果有）
                    self_off_pg_loss = ret_dict.get("self_off_pg_loss")
                    teacher_off_pg_loss = ret_dict.get("teacher_off_pg_loss")
                    teacher_diag_stats = ret_dict.get("teacher_diag_stats")  # ratio/adv 分布统计
                    exp_replay_diag_stats = ret_dict.get("exp_replay_diag_stats")  # endo replay: ratio/adv/shaping 统计
                    ##################
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate entropy loss

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)  # ⭐ Compute KL divergence penalty
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate KL divergence loss

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    # Optional: teacher vs on-policy gradient direction diagnostics (run once per mini-batch)
                    # Per-step aggregated grad_dir: accumulate gradients across ALL micro-batches (epoch==0 only).
                    if grad_dir_step["run"] and epoch == 0:
                        try:
                            # count samples (sanity check)
                            em = exp_mask
                            if em is not None and torch.is_tensor(em):
                                is_off = (em.sum(dim=-1) > 0)  # (bs,)
                                is_on = ~is_off
                                grad_dir_step["on_samples"] += int(is_on.sum().item())
                            if has_teacher_data and (teacher_mask is not None) and torch.is_tensor(teacher_mask):
                                grad_dir_step["teacher_samples"] += int((teacher_mask.sum(dim=-1) > 0).sum().item())

                            # Accumulate per-step gradients on a small union parameter subset:
                            # - Only accumulate g_on on ON-policy samples (avoid undefined on_pg_loss on teacher-only micro-batches).
                            # - Only accumulate g_teacher on TEACHER samples.
                            if grad_dir_union_acc is not None and grad_dir_union_params is not None and len(grad_dir_union_params) > 0:
                                has_on = False
                                try:
                                    has_on = bool((is_on.sum().item() > 0))
                                except Exception:
                                    has_on = False
                                has_te = bool(has_teacher_data and (teacher_off_pg_loss is not None) and torch.is_tensor(teacher_off_pg_loss))

                                if has_on and (on_pg_loss is not None) and torch.is_tensor(on_pg_loss):
                                    g_on = torch.autograd.grad(on_pg_loss, grad_dir_union_params, retain_graph=True, allow_unused=True)
                                    for i, go in enumerate(g_on):
                                        if go is not None:
                                            grad_dir_union_acc["on"][i].add_(go.detach().float())

                                if has_te:
                                    g_te = torch.autograd.grad(teacher_off_pg_loss, grad_dir_union_params, retain_graph=True, allow_unused=True)
                                    for i, gt in enumerate(g_te):
                                        if gt is not None:
                                            grad_dir_union_acc["te"][i].add_(gt.detach().float())
                        except Exception:
                            # don't break training; mark as failed
                            grad_dir_step["run"] = False
                            grad_dir_step["missing_reason"] = 5.0
                    loss.backward()  # ⭐ Backpropagate the loss

                    ##################
                    # ANNI TODO: add metric
                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/on_pg_clipfrac": on_pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/on_pg_clipfrac_lower": on_pg_clipfrac_lower.detach().item(),
                        "actor/on_pg_loss": on_pg_loss.detach().item(),
                        "actor/off_pg_loss": off_pg_loss.detach().item(),
                    }
                    # ⭐ Surrogate clipping diagnostics (clip hit rate / distortion strength proxy)
                    if on_pg_cliphit_rate is not None and torch.is_tensor(on_pg_cliphit_rate):
                        data["actor/on_pg_cliphit_rate"] = on_pg_cliphit_rate.detach().float().item()
                    if off_pg_cliphit_rate is not None and torch.is_tensor(off_pg_cliphit_rate):
                        data["actor/off_pg_cliphit_rate"] = off_pg_cliphit_rate.detach().float().item()
                    if self_off_pg_cliphit_rate is not None and torch.is_tensor(self_off_pg_cliphit_rate):
                        data["actor/self_off_pg_cliphit_rate"] = self_off_pg_cliphit_rate.detach().float().item()
                    if teacher_off_pg_cliphit_rate is not None and torch.is_tensor(teacher_off_pg_cliphit_rate):
                        data["actor/teacher_off_pg_cliphit_rate"] = teacher_off_pg_cliphit_rate.detach().float().item()
                    # ⭐ Teacher Experience: 添加 teacher 专属 metrics
                    if self_off_pg_loss is not None:
                        data["actor/self_off_pg_loss"] = self_off_pg_loss.detach().item()
                    if teacher_off_pg_loss is not None:
                        data["actor/teacher_off_pg_loss"] = teacher_off_pg_loss.detach().item()
                    # ⭐ Teacher Experience: 诊断指标（ratio / advantage 分布）
                    if isinstance(teacher_diag_stats, dict) and teacher_diag_stats:
                        for k, v in teacher_diag_stats.items():
                            try:
                                if torch.is_tensor(v):
                                    data[f"teacher_diag/{k}"] = v.detach().float().item()
                                else:
                                    data[f"teacher_diag/{k}"] = float(v)
                            except Exception:
                                # skip non-numeric
                                pass
                    # ⭐ 7.8: log scheduler diagnostics (step-level; duplicated across micro-batches is OK)
                    if gap_beta_diag:
                        for k, v in gap_beta_diag.items():
                            try:
                                data[f"teacher_diag/gap_beta/{k}"] = float(v)
                            except Exception:
                                pass
                    # ⭐ Endogenous replay diagnostics (importance ratio / shaped ratio / adv split)
                    if isinstance(exp_replay_diag_stats, dict) and exp_replay_diag_stats:
                        for k, v in exp_replay_diag_stats.items():
                            try:
                                if torch.is_tensor(v):
                                    data[f"exp_replay_diag/{k}"] = v.detach().float().item()
                                else:
                                    data[f"exp_replay_diag/{k}"] = float(v)
                            except Exception:
                                pass
                    ##################
                    append_to_dict(metrics, data)

                # ------------------------------------------------------------------
                # Step-level traj statistics (aggregated across micro-batches)
                # ------------------------------------------------------------------
                def _stats_from_vals(v: torch.Tensor) -> dict[str, float]:
                    v = v.flatten()
                    out: dict[str, float] = {}
                    if v.numel() == 0:
                        out["count"] = 0.0
                        return out
                    out["count"] = float(v.numel())
                    out["mean"] = float(v.mean().item())
                    out["std"] = float(v.std().item()) if v.numel() > 1 else 0.0
                    out["min"] = float(v.min().item())
                    out["max"] = float(v.max().item())
                    try:
                        out["p50"] = float(torch.quantile(v, 0.50).item())
                        out["p90"] = float(torch.quantile(v, 0.90).item())
                        out["p99"] = float(torch.quantile(v, 0.99).item())
                    except Exception:
                        pass
                    return out

                if teacher_traj_value_acc:
                    step_traj_metrics = {}
                    for k, chunks in teacher_traj_value_acc.items():
                        try:
                            vv = torch.cat([c.flatten() for c in chunks if torch.is_tensor(c) and c.numel() > 0], dim=0)
                        except Exception:
                            continue
                        # Log as teacher_diag/seq_beta/*_traj/*
                        if k.startswith("seq_beta/"):
                            base = k.split("/", 1)[1]  # e.g. "beta" or "logC_alpha"
                            stats = _stats_from_vals(vv.float())
                            for sk, sv in stats.items():
                                step_traj_metrics[f"teacher_diag/seq_beta/{base}_traj/{sk}"] = sv
                    if step_traj_metrics:
                        append_to_dict(metrics, step_traj_metrics)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)

        # ------------------------------------------------------------------
        # Finalize per-step aggregated grad_dir metrics (log once per update_policy call)
        # ------------------------------------------------------------------
        try:
            # Always emit the full grad_dir schema every step so that distributed metric reduction
            # (which may keep only intersecting keys) won't drop ln/proj fields when some ranks
            # have no teacher samples.
            grad_dir_default = {
                # IMPORTANT: keep defaults finite. Some distributed reducers drop NaN keys entirely.
                # Use `valid` flags to indicate whether the values are meaningful.
                "grad_dir/ln/valid": 0.0,
                "grad_dir/ln/cos": 0.0,
                "grad_dir/ln/dot": 0.0,
                "grad_dir/ln/norm_on": 0.0,
                "grad_dir/ln/norm_teacher": 0.0,
                "grad_dir/ln/param_count": 0.0,
                "grad_dir/ln/vec_dim": 0.0,
                "grad_dir/ln/abs_mean_on": 0.0,
                "grad_dir/ln/abs_mean_teacher": 0.0,
                "grad_dir/proj/valid": 0.0,
                "grad_dir/proj/cos": 0.0,
                "grad_dir/proj/dot": 0.0,
                "grad_dir/proj/norm_on": 0.0,
                "grad_dir/proj/norm_teacher": 0.0,
                "grad_dir/proj/param_count": 0.0,
                "grad_dir/proj/vec_dim": 0.0,
                "grad_dir/proj/abs_mean_on": 0.0,
                "grad_dir/proj/abs_mean_teacher": 0.0,
                # FSDP flat-param probe (works even when use_orig_params=False)
                "grad_dir/flat/valid": 0.0,
                "grad_dir/flat/cos": 0.0,
                "grad_dir/flat/dot": 0.0,
                "grad_dir/flat/norm_on": 0.0,
                "grad_dir/flat/norm_teacher": 0.0,
                "grad_dir/flat/param_count": 0.0,
                "grad_dir/flat/vec_dim": 0.0,
                "grad_dir/flat/abs_mean_on": 0.0,
                "grad_dir/flat/abs_mean_teacher": 0.0,
                "grad_dir/flat/layer_min": 0.0,
                "grad_dir/flat/layer_max": 0.0,
                "grad_dir/flat/layer_count": 0.0,
            }
            append_to_dict(metrics, grad_dir_default)

            # Compute global sample counts first (avoid per-rank dilution in distributed metric reduction).
            # These are used to decide whether cosine is meaningful at the *global step* level.
            teacher_samples_global = float(grad_dir_step.get("teacher_samples", 0))
            on_samples_global = float(grad_dir_step.get("on_samples", 0))
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    _dev = None
                    try:
                        _dev = (
                            grad_dir_union_acc["on"][0].device
                            if (grad_dir_union_acc is not None and len(grad_dir_union_acc.get("on", [])) > 0)
                            else None
                        )
                    except Exception:
                        _dev = None
                    if _dev is None:
                        _dev = torch.device("cpu")
                    ts = torch.tensor(float(grad_dir_step.get("teacher_samples", 0)), device=_dev)
                    os_ = torch.tensor(float(grad_dir_step.get("on_samples", 0)), device=_dev)
                    dist.all_reduce(ts, op=dist.ReduceOp.SUM)
                    dist.all_reduce(os_, op=dist.ReduceOp.SUM)
                    teacher_samples_global = float(ts.detach().cpu().item())
                    on_samples_global = float(os_.detach().cpu().item())
            except Exception:
                pass
            append_to_dict(
                metrics,
                {
                    "grad_dir/teacher_samples_global": float(teacher_samples_global),
                    "grad_dir/on_samples_global": float(on_samples_global),
                },
            )

            if not grad_dir_enable:
                append_to_dict(metrics, {"grad_dir/recorded": 0.0, "grad_dir/missing_reason": 4.0})  # disabled
            elif not grad_dir_should_run:
                append_to_dict(metrics, {"grad_dir/recorded": 0.0, "grad_dir/missing_reason": 3.0})  # interval skip
            elif not grad_dir_step.get("run", False):
                # selection/accumulation failed
                mr = float(grad_dir_step.get("missing_reason", 5.0))
                append_to_dict(metrics, {"grad_dir/recorded": 0.0, "grad_dir/missing_reason": mr})
            elif teacher_samples_global <= 0.0:
                append_to_dict(
                    metrics,
                    {
                        "grad_dir/recorded": 0.0,
                        "grad_dir/missing_reason": 2.0,  # no teacher in this step
                        "grad_dir/aggregate_micro_batches": 1.0,
                    },
                )
            else:
                def _finalize_probe(prefix: str, acc, params):
                    if acc is None or params is None or len(params) == 0:
                        return {}
                    dot = torch.tensor(0.0, device=acc["on"][0].device, dtype=torch.float32)
                    n_on2 = torch.tensor(0.0, device=acc["on"][0].device, dtype=torch.float32)
                    n_te2 = torch.tensor(0.0, device=acc["on"][0].device, dtype=torch.float32)
                    vec_dim = 0
                    for go, gt in zip(acc["on"], acc["te"]):
                        dot = dot + (go * gt).sum()
                        n_on2 = n_on2 + (go * go).sum()
                        n_te2 = n_te2 + (gt * gt).sum()
                        vec_dim += int(go.numel())
                    n_on = torch.sqrt(n_on2 + 1e-12)
                    n_te = torch.sqrt(n_te2 + 1e-12)
                    cos = dot / (n_on * n_te + 1e-12)
                    return {
                        f"grad_dir/{prefix}/cos": float(cos.detach().cpu().item()),
                        f"grad_dir/{prefix}/dot": float(dot.detach().cpu().item()),
                        f"grad_dir/{prefix}/norm_on": float(n_on.detach().cpu().item()),
                        f"grad_dir/{prefix}/norm_teacher": float(n_te.detach().cpu().item()),
                        f"grad_dir/{prefix}/param_count": float(len(params)),
                        f"grad_dir/{prefix}/vec_dim": float(vec_dim),
                    }

                out = {
                    "grad_dir/recorded": 1.0,
                    "grad_dir/missing_reason": 0.0,
                    "grad_dir/aggregate_micro_batches": 1.0,
                }
                # Finalize from union accumulators using index maps.
                # Under FSDP, parameters are sharded; the correct global dot/norm is obtained by
                # summing per-rank shard contributions (scalar all_reduce).
                def _finalize_from_union(prefix: str, idxs: list[int]):
                    if grad_dir_union_acc is None or grad_dir_union_params is None or (not idxs):
                        return {}
                    dot = torch.tensor(0.0, device=grad_dir_union_acc["on"][0].device, dtype=torch.float32)
                    n_on2 = torch.tensor(0.0, device=grad_dir_union_acc["on"][0].device, dtype=torch.float32)
                    n_te2 = torch.tensor(0.0, device=grad_dir_union_acc["on"][0].device, dtype=torch.float32)
                    abs_on = torch.tensor(0.0, device=grad_dir_union_acc["on"][0].device, dtype=torch.float32)
                    abs_te = torch.tensor(0.0, device=grad_dir_union_acc["on"][0].device, dtype=torch.float32)
                    vec_dim = 0
                    for j in idxs:
                        go = grad_dir_union_acc["on"][j]
                        gt = grad_dir_union_acc["te"][j]
                        dot = dot + (go * gt).sum()
                        n_on2 = n_on2 + (go * go).sum()
                        n_te2 = n_te2 + (gt * gt).sum()
                        abs_on = abs_on + go.abs().sum()
                        abs_te = abs_te + gt.abs().sum()
                        vec_dim += int(go.numel())

                    # Global reduction across ranks (sum shard contributions)
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(dot, op=dist.ReduceOp.SUM)
                            dist.all_reduce(n_on2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(n_te2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(abs_on, op=dist.ReduceOp.SUM)
                            dist.all_reduce(abs_te, op=dist.ReduceOp.SUM)
                    except Exception:
                        pass

                    n_on = torch.sqrt(n_on2 + 1e-12)
                    n_te = torch.sqrt(n_te2 + 1e-12)
                    cos = dot / (n_on * n_te + 1e-12)
                    return {
                        f"grad_dir/{prefix}/valid": 1.0,
                        f"grad_dir/{prefix}/cos": float(cos.detach().cpu().item()),
                        f"grad_dir/{prefix}/dot": float(dot.detach().cpu().item()),
                        f"grad_dir/{prefix}/norm_on": float(n_on.detach().cpu().item()),
                        f"grad_dir/{prefix}/norm_teacher": float(n_te.detach().cpu().item()),
                        f"grad_dir/{prefix}/param_count": float(len(idxs)),
                        f"grad_dir/{prefix}/vec_dim": float(vec_dim),
                        # extra sanity: mean absolute grad magnitude in probe subspace
                        f"grad_dir/{prefix}/abs_mean_on": float((abs_on / max(1, vec_dim)).detach().cpu().item()),
                        f"grad_dir/{prefix}/abs_mean_teacher": float((abs_te / max(1, vec_dim)).detach().cpu().item()),
                    }

                out.update(_finalize_from_union("ln", list(grad_dir_union_idx_ln or [])))
                out.update(_finalize_from_union("proj", list(grad_dir_union_idx_proj or [])))
                out.update(_finalize_from_union("flat", list(grad_dir_union_idx_flat or [])))
                # Attach best-effort flat probe layer summary (helps interpret flat_param probe)
                out["grad_dir/flat/layer_min"] = float(grad_dir_step.get("flat_layer_min", 0.0))
                out["grad_dir/flat/layer_max"] = float(grad_dir_step.get("flat_layer_max", 0.0))
                out["grad_dir/flat/layer_count"] = float(grad_dir_step.get("flat_layer_count", 0.0))

                # Attach already-computed global sample counts (so panels stay simple)
                out["grad_dir/teacher_samples_global"] = float(teacher_samples_global)
                out["grad_dir/on_samples_global"] = float(on_samples_global)

                append_to_dict(metrics, out)
        except Exception:
            # never break training
            pass
        self.actor_optimizer.zero_grad()
        return metrics