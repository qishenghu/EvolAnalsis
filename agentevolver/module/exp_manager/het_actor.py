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

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics