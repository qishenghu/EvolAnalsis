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

                for data in micro_batches:
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
        self.actor_optimizer.zero_grad()
        return metrics