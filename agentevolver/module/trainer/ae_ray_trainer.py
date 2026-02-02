# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from pprint import pprint
from typing import List, Optional, Any, Dict
import warnings

from loguru import logger
import numpy as np
import ray
import torch
import random
import json
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from torch.utils.data import SequentialSampler,IterableDataset,Dataset,Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from agentevolver.client.env_client import EnvClient
from agentevolver.module.task_manager.task_manager import AutoReloadDataset, FullDataset
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, create_colocated_worker_cls
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from agentevolver.utils.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer, ResourcePoolManager, WorkerType,
                                          _timer, apply_kl_penalty,
                                          compute_response_mask, Role)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.metric import reduce_metrics

from agentevolver.client.llm_client import DashScopeClient
from agentevolver.client.em_client import EMClient
from agentevolver.module.env_manager.env_manager import ParallelEnvManager
from agentevolver.module.task_manager import adapter as task_adapter
from agentevolver.module.task_manager import TaskManager,NaiveTaskObjectiveRetrieval
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from agentevolver.utils.tracking import ValidationGenerationsLogger

from agentevolver.module.adv_processor.adca_grpo_pipeline import apply_adca_grpo

from agentevolver.module.exp_manager.exp_manager import ExperienceManager
from agentevolver.module.exp_manager.experience_collate import (
    ExperienceMixCollateFn, 
    TeacherExperienceMixCollateFn,
    LUFFYTeacherRolloutMixer,
    RePOReplayManager,
    RePORolloutMixer,
)
from agentevolver.module.exp_manager.het_core_algos import (
    repo_compute_grpo_advantage,
    repo_compute_token_loss,
)


def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Return masked mean; None if empty."""
    m = mask.bool()
    if m.sum() == 0:
        return None
    return x[m].mean()


def _safe_masked_std(x: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Return masked std; None if empty or single element."""
    m = mask.bool()
    if m.sum() == 0:
        return None
    vals = x[m]
    if vals.numel() <= 1:
        return torch.tensor(0.0, device=x.device)
    return vals.std()


def compute_teacher_effect_metrics(batch: DataProto, entropys: Optional[torch.Tensor] = None) -> dict:
    """
    Compute batch-level diagnostics to interpret teacher experience effects.
    This is designed to be cheap and robust (works even if some masks are absent).
    """
    metrics: dict[str, float] = {}

    if "responses" not in batch.batch:
        return metrics

    response_len = batch.batch["responses"].shape[-1]

    # response mask: for multi-turn, trainer uses compute_response_mask(batch) which should be response_len mask
    response_mask = batch.batch.get("response_mask", None)
    if response_mask is None:
        response_mask = compute_response_mask(batch)
        batch.batch["response_mask"] = response_mask

    # Ensure response_mask is (bs, response_len)
    if response_mask.dim() == 2 and response_mask.shape[-1] != response_len and response_mask.shape[-1] > response_len:
        response_mask = response_mask[:, -response_len:]

    # exp_mask is full seq; slice response part
    exp_mask_full = batch.batch.get("exp_mask", None)
    exp_mask = None
    if exp_mask_full is not None and exp_mask_full.dim() == 2:
        if exp_mask_full.shape[-1] == response_len:
            exp_mask = exp_mask_full
        else:
            exp_mask = exp_mask_full[:, -response_len:]

    teacher_mask = batch.batch.get("teacher_mask", None)
    if teacher_mask is not None and teacher_mask.dim() == 2 and teacher_mask.shape[-1] != response_len:
        teacher_mask = teacher_mask[:, -response_len:]

    # token-level rewards (response_len)
    token_level_rewards = batch.batch.get("token_level_rewards", None)
    if token_level_rewards is None:
        token_level_rewards = batch.batch.get("token_level_scores", None)
    rewards_sum = None
    if token_level_rewards is not None and token_level_rewards.dim() == 2:
        rewards_sum = (token_level_rewards[:, :response_len] * response_mask).sum(dim=-1)  # (bs,)

    # token-level advantages (response_len)
    advantages = batch.batch.get("advantages", None)
    adv_scalar = None
    if advantages is not None and advantages.dim() == 2:
        denom = response_mask.sum(dim=-1).clamp_min(1.0)
        adv_scalar = (advantages[:, :response_len] * response_mask).sum(dim=-1) / denom

    # derive per-token category masks
    if exp_mask is None:
        exp_mask = torch.zeros_like(response_mask)
    if teacher_mask is None:
        teacher_mask = torch.zeros_like(response_mask)

    on_token = (1.0 - exp_mask) * response_mask
    off_token = exp_mask * response_mask
    teacher_token = exp_mask * teacher_mask * response_mask
    self_off_token = exp_mask * (1.0 - teacher_mask) * response_mask

    # ratios (token level)
    total_tokens = response_mask.sum().clamp_min(1.0)
    metrics["diag/exp_token_ratio"] = (off_token.sum() / total_tokens).detach().item()
    metrics["diag/teacher_token_ratio"] = (teacher_token.sum() / total_tokens).detach().item()

    # ⭐ Extra ratios to avoid confusion:
    # - teacher_token_ratio above uses total response tokens as denominator (includes env/tool tokens in multi-turn)
    # - the following metrics provide more intuitive denominators
    loss_mask_full = batch.batch.get("loss_mask", None)
    loss_mask = None
    if loss_mask_full is not None and torch.is_tensor(loss_mask_full) and loss_mask_full.dim() == 2:
        if loss_mask_full.shape[-1] == response_len:
            loss_mask = loss_mask_full
        elif loss_mask_full.shape[-1] > response_len:
            loss_mask = loss_mask_full[:, -response_len:]
    if loss_mask is not None:
        llm_token = loss_mask * response_mask  # only LLM positions within response
        llm_total = llm_token.sum().clamp_min(1.0)
        metrics["diag/llm_token_ratio_in_response"] = (llm_total / total_tokens).detach().item()
        metrics["diag/teacher_llm_token_ratio"] = (teacher_token.sum() / llm_total).detach().item()

    # sample-level ratios (rollout-level intuition)
    # Prefer teacher_mask to detect teacher samples (works even if exp_mask/loss_mask has edge cases)
    is_teacher_sample = (teacher_mask.sum(dim=-1) > 0) if teacher_mask is not None else (teacher_token.sum(dim=-1) > 0)
    metrics["diag/teacher_sample_ratio"] = is_teacher_sample.float().mean().detach().item()
    is_off_sample = (off_token.sum(dim=-1) > 0)
    metrics["diag/offpolicy_sample_ratio"] = is_off_sample.float().mean().detach().item()

    # advantage means by type (token level + per-sample scalar)
    if advantages is not None:
        v = _safe_masked_mean(advantages[:, :response_len], teacher_token.bool())
        if v is not None:
            metrics["diag/adv_teacher_token_mean"] = v.detach().item()
        v = _safe_masked_mean(advantages[:, :response_len], self_off_token.bool())
        if v is not None:
            metrics["diag/adv_self_off_token_mean"] = v.detach().item()
        v = _safe_masked_mean(advantages[:, :response_len], on_token.bool())
        if v is not None:
            metrics["diag/adv_onpolicy_token_mean"] = v.detach().item()

    if adv_scalar is not None:
        # sample-level teacher/offpolicy flags
        is_teacher_sample = (teacher_token.sum(dim=-1) > 0)
        is_off_sample = (off_token.sum(dim=-1) > 0)
        is_on_sample = ~is_off_sample
        v = _safe_masked_mean(adv_scalar, is_teacher_sample)
        if v is not None:
            metrics["diag/adv_teacher_sample_mean"] = v.detach().item()
        v = _safe_masked_mean(adv_scalar, is_on_sample)
        if v is not None:
            metrics["diag/adv_onpolicy_sample_mean"] = v.detach().item()

        # sign ratios
        if is_on_sample.any():
            metrics["diag/onpolicy_adv_pos_ratio"] = (adv_scalar[is_on_sample] > 0).float().mean().detach().item()
        if is_teacher_sample.any():
            metrics["diag/teacher_adv_pos_ratio"] = (adv_scalar[is_teacher_sample] > 0).float().mean().detach().item()

    # reward means by type (sample level)
    if rewards_sum is not None:
        is_teacher_sample = (teacher_token.sum(dim=-1) > 0)
        is_off_sample = (off_token.sum(dim=-1) > 0)
        is_on_sample = ~is_off_sample
        v = _safe_masked_mean(rewards_sum, is_on_sample)
        if v is not None:
            metrics["diag/reward_onpolicy_mean"] = v.detach().item()
        v = _safe_masked_mean(rewards_sum, is_teacher_sample)
        if v is not None:
            metrics["diag/reward_teacher_mean"] = v.detach().item()

        # group gap: teacher reward - onpolicy reward within same uid group
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is not None:
            try:
                # uids: np array of strings
                uids_list = [str(u) for u in list(uids)]
                rs = rewards_sum.detach().cpu().tolist()
                is_t = is_teacher_sample.detach().cpu().tolist()
                # group -> (teacher_rewards, on_rewards)
                grp: dict[str, dict[str, list]] = {}
                for i, gid in enumerate(uids_list):
                    if gid not in grp:
                        grp[gid] = {"t": [], "o": []}
                    if is_t[i]:
                        grp[gid]["t"].append(rs[i])
                    else:
                        grp[gid]["o"].append(rs[i])
                gaps = []
                nt_means = []
                all_means = []
                for gid, d in grp.items():
                    if d["t"] and d["o"]:
                        gaps.append(float(np.mean(d["t"]) - np.mean(d["o"])))
                        nt_means.append(float(np.mean(d["o"])))
                        all_means.append(float(np.mean(d["t"] + d["o"])))
                if gaps:
                    metrics["diag/group_teacher_minus_on_reward_mean"] = float(np.mean(gaps))
                    metrics["diag/group_teacher_minus_on_reward_std"] = float(np.std(gaps))
                    metrics["diag/group_gap_count"] = float(len(gaps))
                if nt_means:
                    # Useful for 2.1 teacher-baseline separation: this approximates the non-teacher baseline.
                    metrics["diag/group_non_teacher_reward_mean"] = float(np.mean(nt_means))
                if all_means:
                    # Useful to compare against original GRPO mixed baseline proxy.
                    metrics["diag/group_all_reward_mean"] = float(np.mean(all_means))
            except Exception:
                pass

    # entropy diagnostics by type (token level)
    if entropys is not None and entropys.dim() == 2 and entropys.shape[-1] >= response_len:
        ent = entropys[:, :response_len]
        v = _safe_masked_mean(ent, teacher_token.bool())
        if v is not None:
            metrics["diag/entropy_teacher_token_mean"] = v.detach().item()
        v = _safe_masked_mean(ent, on_token.bool())
        if v is not None:
            metrics["diag/entropy_onpolicy_token_mean"] = v.detach().item()
        v = _safe_masked_mean(ent, off_token.bool())
        if v is not None:
            metrics["diag/entropy_offpolicy_token_mean"] = v.detach().item()
        s = _safe_masked_std(ent, on_token.bool())
        if s is not None:
            metrics["diag/entropy_onpolicy_token_std"] = s.detach().item()

    return metrics


def parse_reward_from_dataproto(data: DataProto, return_dict=False) -> dict | torch.Tensor:
    """
    Compute reward for a batch of data.

    Args:
        data: DataProto object containing the input data.
        return_dict: Whether to return a dictionary or just the reward tensor.

    Returns:
        Tensor of shape (bs, response_len) if return_dict is False,
        or a dict with 'reward_tensor' and 'reward_extra_info'.
    """
    # Within DataFlow, world.execute() will pass a float score, which will be contained in the DataProto.non_tensor_batch('reward_scores')

    # Initialize reward tensor
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (bs, reslen)  # ⭐ Initialize the reward tensor
    reward_extra_info = defaultdict(list)

    # Batch-level processing
    prompt_ids_batch = data.batch["prompts"]  # (bs, prompt_len)
    prompt_lengths = prompt_ids_batch.shape[-1]

    # Get attention masks for all items
    attention_masks = data.batch["attention_mask"]  # (bs, total_len)
    response_lengths = attention_masks[:, prompt_lengths:].sum(dim=1)  # (bs, )

    # Get reward scores
    reward_scores_list = [item["outcome"] for item in data.non_tensor_batch["reward_scores"]]
    reward_scores = torch.tensor(reward_scores_list, device=reward_tensor.device, dtype=torch.float32)  # (bs, )  # ⭐ Convert reward scores to a tensor

    # Use advanced indexing to assign rewards
    reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores

    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }
    else:
        return reward_tensor


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def union_gen_batch_via_task_id(tasks, batch: DataProto, gen_batch_output: DataProto):
    """
    Merges the `gen_batch_output` with the `batch` based on the `task_id`.

    Args:
        tasks (list): A list of task objects, each containing a `task_id`.
        batch (DataProto): The original batch of data.
        gen_batch_output (DataProto): The generated batch output that needs to be merged.

    Returns:
        DataProto: The final merged batch.
    """
    map_task_id_to_index = {t.task_id:i for i, t in enumerate(tasks)}  # ⭐ Create a mapping from task_id to its index in tasks
    gen_task_task_ids = gen_batch_output.non_tensor_batch['task_ids']
    indices = [map_task_id_to_index[tid] for tid in gen_task_task_ids]
    batch_extend = batch.select_idxs(indices)
    batch_final = batch_extend.union(gen_batch_output)  # ⭐ Merge the selected part of the batch with the gen_batch_output
    return batch_final


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    if scores.dim()!=1:
        logger.warning("scores.dim()!=1")

    with torch.no_grad():
        bsz = scores.shape[0]
        
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                # no std
                # if llm judge output similar rewards for undistinguishable samples, we may want to reduce its weight according to the batch std
                # scores[i] = scores[i] / (batch_std + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_grpo_outcome_advantage_teacher_baseline_separated(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    teacher_mask: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    teacher_baseline: str = "all_mean",
    non_teacher_baseline: str = "non_teacher_mean",
    std_source: str = "non_teacher",
):
    """
    GRPO outcome-advantage with teacher-separated baseline (2.1 improvement).

    Goal:
    - Do NOT let teacher rollouts (typically reward=1) lift the baseline used for non-teacher samples.
    - Keep original behavior by default (feature is off unless enabled by config).

    Notes:
    - "teacher" is inferred by per-sample teacher_mask (any token marked -> teacher sample).
    - "non-teacher" corresponds to the remaining samples in the same GRPO group.
    - We only separate teacher vs non-teacher; we do NOT change self-generated experience replay semantics.
    """
    # Keep original score definition for comparability
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Align teacher_mask to response length and derive per-sample teacher flag
    resp_len = response_mask.size(1)
    if teacher_mask.dim() == 2 and teacher_mask.size(1) != resp_len and teacher_mask.size(1) > resp_len:
        teacher_mask = teacher_mask[:, -resp_len:]
    is_teacher = (teacher_mask * response_mask).sum(dim=-1) > 0  # (bs,)

    id2idxs: dict[Any, list[int]] = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2idxs[index[i]].append(i)

        id2_mean_all: dict[Any, torch.Tensor] = {}
        id2_mean_non_teacher: dict[Any, torch.Tensor] = {}
        id2_std: dict[Any, torch.Tensor] = {}

        for gid, idxs in id2idxs.items():
            idxs_t = torch.tensor(idxs, device=scores.device, dtype=torch.long)
            s_all = scores[idxs_t]
            t_all = is_teacher[idxs_t]
            s_nt = s_all[~t_all]

            mean_all = s_all.mean() if s_all.numel() > 0 else torch.tensor(0.0, device=scores.device)
            mean_nt = s_nt.mean() if s_nt.numel() > 0 else mean_all

            id2_mean_all[gid] = mean_all
            id2_mean_non_teacher[gid] = mean_nt

            if std_source == "all":
                s_for_std = s_all
            else:  # "non_teacher"
                s_for_std = s_nt if s_nt.numel() > 0 else s_all

            if s_for_std.numel() <= 1:
                std = torch.tensor(1.0, device=scores.device)
            else:
                std = s_for_std.std()
                if torch.isnan(std).item() or std.item() == 0.0:
                    std = torch.tensor(1.0, device=scores.device)
            id2_std[gid] = std

        adv = scores.clone()
        for i in range(bsz):
            gid = index[i]

            if is_teacher[i]:
                if teacher_baseline == "fixed_one":
                    base = torch.tensor(1.0, device=scores.device)
                elif teacher_baseline == "non_teacher_mean":
                    base = id2_mean_non_teacher[gid]
                else:  # "all_mean"
                    base = id2_mean_all[gid]
            else:
                if non_teacher_baseline == "all_mean":
                    base = id2_mean_all[gid]
                else:  # "non_teacher_mean"
                    base = id2_mean_non_teacher[gid]

            adv_i = scores[i] - base
            if norm_adv_by_std_in_grpo:
                adv_i = adv_i / (id2_std[gid] + epsilon)
            adv[i] = adv_i

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """
    Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]

        # 2.1 improvement (optional): teacher-separated baseline for GRPO advantage.
        # Default remains original GRPO behavior.
        teacher_sep_cfg = None
        try:
            teacher_sep_cfg = (config.get("grpo", {}) or {}).get("teacher_baseline_separation", None) if config else None
        except Exception:
            teacher_sep_cfg = None

        # NOTE: config may be OmegaConf DictConfig; treat Mapping as config-like
        enable_teacher_sep = bool(teacher_sep_cfg.get("enable", False)) if isinstance(teacher_sep_cfg, Mapping) else False
        teacher_mask = data.batch.get("teacher_mask", None)
        has_teacher = (
            teacher_mask is not None
            and torch.is_tensor(teacher_mask)
            and teacher_mask.numel() > 0
            and teacher_mask.sum().item() > 0
        )

        if enable_teacher_sep and has_teacher:
            resp_len = grpo_calculation_mask.size(1)
            if teacher_mask.dim() == 2 and teacher_mask.size(1) != resp_len and teacher_mask.size(1) > resp_len:
                teacher_mask = teacher_mask[:, -resp_len:]
            advantages, returns = compute_grpo_outcome_advantage_teacher_baseline_separated(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                teacher_mask=teacher_mask,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                teacher_baseline=str(teacher_sep_cfg.get("teacher_baseline", "all_mean")),
                non_teacher_baseline=str(teacher_sep_cfg.get("non_teacher_baseline", "non_teacher_mean")),
                std_source=str(teacher_sep_cfg.get("std_source", "non_teacher")),
            )
        else:
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )  # ⭐ Compute advantages and returns for GRPO
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)  # ⭐ Compute advantages and returns for other estimators
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class AgentEvolverRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        train_task_manager:TaskManager,
        val_task_manager:TaskManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, # type: ignore
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        collate_fn=None,
        shuffle_trainset:bool=False,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.

        Args:
            config: Configuration object containing various settings.
            tokenizer: Tokenizer used for processing text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping of roles to worker types.
            resource_pool_manager (ResourcePoolManager): Manager for resource pools.
            train_task_manager (TaskManager): Task manager for training tasks.
            val_task_manager (TaskManager): Task manager for validation tasks.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor (optional): Processor for additional data processing.
            reward_fn (optional): Function to compute rewards.
            val_reward_fn (optional): Function to compute validation rewards.
            collate_fn (optional): Function to collate data.
            shuffle_trainset (bool, optional): Whether to shuffle the training dataset. Defaults to False.
            device_name (str, optional): Name of the device to use. Defaults to "cuda".
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"  # ⭐ Ensure the hybrid engine is supported

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"  # ⭐ Ensure ActorRollout role is present in the mapping

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        # 2.2 adaptive teacher gating (EMA state for reward gap)
        self._teacher_gap_ema: Optional[float] = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not supported
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()

        self.env_manager: ParallelEnvManager | None = None
        self.thread_pool: ThreadPoolExecutor | None = None

        self.train_task_manager=train_task_manager
        self.val_task_manager=val_task_manager
        self._collate_fn=collate_fn

        self._create_dataloader_from_manager(collate_fn, shuffle_trainset)  # ⭐ Create dataloader from the provided manager


    def init_workers(self):
        """
        Initializes distributed training workers using the Ray backend.

        This function creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Args:
            None

        Returns:
            None
        """
        self.resource_pool_manager.create_resource_pool()  # ⭐ Initialize the resource pools

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls,
                                                device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()  # ⭐ Initialize the critic model

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()  # ⭐ Initialize the reference policy model

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()  # ⭐ Initialize the reward model

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()  # ⭐ Initialize the actor rollout model

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
            self.async_rollout_mode = True
            self.async_rollout_manager = BaAsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg)  # ⭐ Create the asynchronous rollout manager

        self.reward_fn = parse_reward_from_dataproto
        self.val_reward_fn = parse_reward_from_dataproto

        self.env_manager = ParallelEnvManager(config=self.config, async_rollout_manager=self.async_rollout_manager, max_parallel=self.config.actor_rollout_ref.rollout.max_env_worker)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool.max_workers)
        self.exp_manager = ExperienceManager(config=self.config)


    def _create_dataloader_from_manager(self, collate_fn, shuffle_trainset: bool = True):
        """
        Creates the train and validation dataloaders.

        1. Check the existence of train and val files and load local tasks from them. If no files given, load tasks from environment (train and val/dev splits).
        2. Use task manager to generate synthetic tasks for trainset, and load the original val dataset.
        3. Use task manager to mix tasks from different sources.
        4. Adapt datasets and create dataloaders used in the trainer.

        Args:
            collate_fn (callable): The function to use for collating data into batches.
            shuffle_trainset (bool, optional): Whether to shuffle the training set. Defaults to True.

        Returns:
            None
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn


        from verl.trainer.main_ppo import create_rl_dataset
        # load train dataset from files or environment
        env_client=EnvClient(self.config.env_service.env_url)
        env_type = str(self.config.env_service.env_type)
        if self.config.data.train_files is not None:
            train_seed_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(train_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.train_task_manager.load_tasks_from_dataset(train_seed_dataset,env_type=env_type)
            train_source = f"files({self.config.data.train_files})"
        else:
            # self.train_task_manager.load_tasks_from_environment(env_client,env_type=self.config.env_service.env_type,split="train")
            max_train_tasks = self.config.data.get("max_train_tasks", None)
            shuffle = self.config.data.get("shuffle", True) # by default, shuffle the train tasks
            seed = self.config.data.get("seed", 2026)
            self.train_task_manager.load_tasks_from_environment(
                env_client,
                env_type=env_type,
                split="train",
                max_tasks=max_train_tasks,
                shuffle=shuffle,
                seed=seed,
            )
            train_source = f"env(split=train, max_train_tasks={max_train_tasks}, shuffle={shuffle}, seed={seed})"
        
        # load val dataset
        if self.config.data.val_files is not None:
            val_seed_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
            assert isinstance(val_seed_dataset,RLHFDataset), "train_dataset must be RLHFDataset"
            self.val_task_manager.load_tasks_from_dataset(val_seed_dataset,env_type=env_type)
            val_source = f"files({self.config.data.val_files})"
        else:
            num_loaded_val_tasks = 0
            val_source = "env(unknown)"
            if 'val_on_test' in os.environ.get("DEBUG_ARG",'') or (self.config.data.val_type == 'test_normal' and self.config.env_service.env_type == "appworld"):
                logger.warning("using test_normal as val dataset")
                num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=env_type,split="test_normal", shuffle=False)
                val_source = "env(split=test_normal)"
            else:
                # For AlfWorld, val and dev both return test set (200 tasks)
                # So we only need to load once to avoid duplicates
                if env_type == "alfworld":
                    # Only load from 'val' split (which now returns test set)
                    try:
                        num_loaded_val_tasks += self.val_task_manager.load_tasks_from_environment(env_client,env_type=env_type,split="val", shuffle=False)
                        val_source = "env(split=val)"
                    except:
                        logger.warning(f"failed to load val dataset from environment, split=val")
                else:
                    # For other environments:
                    # - Prefer a single split to avoid accidentally loading the same IDs twice (common in programmatic envs).
                    # - Fallback to dev only if val is unavailable.
                    for split in ["val", "dev"]:
                        try:
                            loaded = self.val_task_manager.load_tasks_from_environment(
                                env_client,
                                env_type=env_type,
                                split=split,
                                shuffle=False,
                            )
                            num_loaded_val_tasks += loaded
                            if loaded > 0:
                                val_source = f"env(split={split})"
                                break
                        except Exception:
                            logger.warning(
                                f"failed to load val dataset from environment, split={split}. "
                                f"this may be *normal* if your dataset is split into train/dev"
                            )
            
            assert num_loaded_val_tasks > 0, "failed to load val/dev dataset from environment"

        # ------------------------------------------------------------------
        # 🔥 Very visible dataset size logging (train/val) for sanity checks
        # ------------------------------------------------------------------
        try:
            train_tasks = list(self.train_task_manager.seed_tasks)
            val_tasks = list(self.val_task_manager.seed_tasks)

            train_ids = [t.task_id for t in train_tasks]
            val_ids = [t.task_id for t in val_tasks]
            train_unique = len(set(train_ids))
            val_unique = len(set(val_ids))
            overlap = len(set(train_ids) & set(val_ids))

            banner = "=" * 88
            msg = (
                f"\n{banner}\n"
                f"📌 DATASET SIZES (env={env_type})\n"
                f"  - train: {len(train_tasks)} tasks (unique task_id={train_unique})  source={train_source}\n"
                f"  -   val: {len(val_tasks)} tasks (unique task_id={val_unique})  source={val_source}\n"
                f"  - train∩val overlap (by task_id): {overlap}\n"
                f"{banner}\n"
            )
            # Use both logger and print to ensure visibility in different launch modes.
            logger.warning(msg)
            print(msg, flush=True)
        except Exception as e:
            logger.warning(f"failed to print dataset sizes: {e}")
        
        self.train_dataset = FullDataset(
            self.train_task_manager,
            self.train_task_manager._mixture_strategy,
            self.train_task_manager._reward_config,
            self.config.task_manager.train_data_path,
            tokenizer=self.tokenizer,
            config=self.config.data,
            processor=self.processor,
        )
        self.val_dataset = FullDataset(
            self.val_task_manager,
            self.val_task_manager._mixture_strategy,
            self.val_task_manager._reward_config,
            cache_path=None,
            tokenizer=self.tokenizer,
            config=self.config.data,
            processor=self.processor,
        )

        assert not isinstance(self.train_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        assert not isinstance(self.val_dataset,AutoReloadDataset), "please disable multiple workers for AutoReloadDataset"
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data,self.train_dataset),
        )  # ⭐ Create the train dataloader with specified parameters

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset) # type: ignore

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )  # ⭐ Create the validation dataloader with specified parameters

        # train dataloader is on-the-fly, so we don't need to check the size
        # assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if not isinstance(self.train_dataset,IterableDataset):
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        else:
            # FIXME: need a elegant way to set total_training_steps
            total_training_steps = len(self.train_task_manager.seed_tasks)*self.config.trainer.total_epochs
            print(f"Size of train dataloader: unknown, Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


    def _get_attribution_config(self):
        """
        Retrieves and validates the configuration for attribution-driven credit assignment, including the setup for API retry attempts.

        Returns:
            dict: The validated and possibly updated configuration dictionary.

        Raises:
            ValueError: If the required 'attribution_driven_credit_assignment' block is missing from the configuration.
        """
        if not hasattr(self.config, 'attribution_driven_credit_assignment'):
            raise ValueError("attribution_driven_credit_assignment configuration block is required")

        config = self.config.attribution_driven_credit_assignment

        # set the default api_max_retries
        if not hasattr(config, 'api_max_retries'):
            config.api_max_retries = 200  # ⭐ Set the default number of API retries to 200
            print(f"[attribution_config] Using default api_max_retries: {config.api_max_retries}")

        return config


    def _validate_config(self):
        """
        Validates the configuration settings to ensure they are consistent and meet the necessary requirements for the training process.

        This function checks:
        - The total number of GPUs and their allocation.
        - The total batch size and its divisibility by the minimal possible batch size.
        - Mutual exclusivity of certain micro-batch size parameters.
        - Consistency in actor, critic, and reward model configurations.
        - Other critical settings such as loss aggregation mode and sequence parallelism.

        Raises:
            AssertionError: If any of the configuration settings do not meet the required conditions.
            ValueError: If mutually exclusive parameters are both set or neither is set.
        """
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size  # ⭐ Ensure train_batch_size is at least as large as ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0  # ⭐ Ensure ppo_mini_batch_size is divisible by ppo_micro_batch_size
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ Ensure sufficient GPU allocation for micro-batch size and sequence parallelism

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size  # ⭐ Ensure train_batch_size is at least as large as ppo_mini_batch_size for critic
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0  # ⭐ Ensure ppo_mini_batch_size is divisible by ppo_micro_batch_size for critic
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus  # ⭐ Ensure sufficient GPU allocation for micro-batch size and sequence parallelism for critic

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            # 0623 yunpeng comment: no need this tool_config_path
            # assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    ##################
    # Experience Replay Methods
    def _replace_recorded_old_log_probs(
        self,
        batch: DataProto,
        current_old_log_prob: DataProto,
        entropys: torch.Tensor,
    ) -> DataProto:
        """
        替换 off-policy 数据的 old_log_prob 为 recorded_old_log_probs。
        
        这是 Experience Replay 的关键步骤：
        - 对于 on-policy 数据：使用当前 policy 计算的 old_log_prob
        - 对于 off-policy 数据：使用收集经验时保存的 recorded_old_log_probs
        
        Args:
            batch: 当前 batch 数据，包含 exp_mask 和 recorded_old_log_probs
            current_old_log_prob: 当前 policy 计算的 old_log_prob
            entropys: 当前 policy 计算的 entropy
            
        Returns:
            更新后的 batch
        """
        exp_mask = batch.batch["exp_mask"]  # [batch, seq_len]
        response_mask = batch.batch["response_mask"]  # [batch, seq_len]
        recorded_old_log_probs = batch.batch["recorded_old_log_probs"]  # [batch, response_len]
        current_old_log_probs = current_old_log_prob.batch["old_log_probs"]  # [batch, response_len]
        
        # 获取 response 部分的 exp_mask
        # exp_mask 是完整序列，我们需要提取 response 部分
        # response_mask 标记了 response 部分的位置
        prompt_length = batch.batch["prompts"].shape[-1]
        response_exp_mask = exp_mask[:, prompt_length:]  # [batch, response_len]
        
        # 对齐长度
        min_len = min(response_exp_mask.shape[-1], recorded_old_log_probs.shape[-1], current_old_log_probs.shape[-1])
        response_exp_mask = response_exp_mask[:, :min_len]
        recorded_old_log_probs = recorded_old_log_probs[:, :min_len]
        
        # 创建替换后的 old_log_probs
        # off-policy (exp_mask=1): 使用 recorded_old_log_probs
        # on-policy (exp_mask=0): 使用 current_old_log_probs
        new_old_log_probs = torch.where(
            response_exp_mask.bool(),
            recorded_old_log_probs,
            current_old_log_probs[:, :min_len]
        )
        
        # 如果长度不够，用 current_old_log_probs 补齐
        if min_len < current_old_log_probs.shape[-1]:
            new_old_log_probs = torch.cat([
                new_old_log_probs,
                current_old_log_probs[:, min_len:]
            ], dim=-1)
        
        # 更新 current_old_log_prob（后续会 union 到 batch）
        current_old_log_prob.batch["old_log_probs"] = new_old_log_probs
        
        # 返回 batch（虽然 batch 本身没有改变，但 current_old_log_prob 已经更新）
        return batch

    def _select_best_offpolicy_by_current_entropy(
        self,
        task_to_candidates: Dict[str, List],
        tasks: List,
        num_trajectories_per_task: int = 1,
    ) -> List:
        """
        使用当前 policy 计算 entropy，选择每个 task 的最优 off-policy 轨迹。
        
        ⭐ ExGRPO 方式：在选择时使用当前 policy 重新计算 entropy，而不是使用保存时的 entropy。
        ⭐ Multi-turn 关键：只对 LLM 响应部分（loss_mask=1）计算 entropy。
        
        Args:
            task_to_candidates: task_id -> 候选轨迹列表的映射
            tasks: experience task 列表
            num_trajectories_per_task: 每个 task 选择的轨迹数量
            
        Returns:
            List[Trajectory]: 选中的 off-policy 轨迹列表
        """
        if not task_to_candidates:
            return []
        
        selected_trajectories = []
        
        for task in tasks:
            task_id = task.task_id
            candidates = task_to_candidates.get(task_id, [])
            
            if not candidates:
                # 没有候选轨迹，设置 n_offpolicy_trajectories 为 0
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = 0
                else:
                    task.metadata = {"n_offpolicy_trajectories": 0}
                continue
            
            if len(candidates) == 1:
                # 只有一个候选，直接选择
                selected_trajectories.append(candidates[0])
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = 1
                else:
                    task.metadata = {"n_offpolicy_trajectories": 1}
                continue
            
            # 多个候选：将候选轨迹转换为 CMT，计算 entropy
            try:
                candidate_cmts = self.env_manager.convert_offpolicy_to_cmt(
                    offpolicy_trajectories=candidates,
                    config=self.config,
                    tokenizer=self.tokenizer
                )
                
                # ⭐ 检查转换后的 CMT 是否为空
                if not candidate_cmts:
                    raise ValueError("convert_offpolicy_to_cmt returned empty list")
                
                # 转换为 samples，跳过 world_size 对齐（因为只是用于 entropy 计算，不需要对齐）
                samples = []
                for cmt in candidate_cmts:
                    extras = self.env_manager.get_extra(cmt)
                    sample_arr = cmt.group_tokenize()
                    for sample in sample_arr:
                        sample.extras = extras
                    samples.extend(sample_arr)
                
                # ⭐ 检查 samples 是否为空
                if not samples:
                    raise ValueError("No samples generated from candidate CMTs")
                
                # ⭐ 为了支持分布式计算，需要将样本数量对齐到 world_size
                world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                original_num_samples = len(samples)
                remainder = original_num_samples % world_size
                if remainder != 0:
                    # 需要添加 padding 样本（复制已有样本）
                    padding_needed = world_size - remainder
                    for i in range(padding_needed):
                        samples.append(samples[i % original_num_samples])
                
                # 转换为 DataProto
                candidate_batch = self.env_manager.samples_to_dataproto(samples)
                
                # 计算 entropy
                log_prob_result = self.actor_rollout_wg.compute_log_prob(candidate_batch)
                # ⭐ 只取原始样本的 entropy（去除 padding）
                entropys = log_prob_result.batch["entropys"][:original_num_samples]  # [num_candidates, response_len]
                
                # ⭐ Multi-turn 关键：使用 loss_mask 计算 LLM 响应部分的平均 entropy
                # 对于 multi-turn，loss_mask 只标记 LLM 响应位置（不包含 environment 响应）
                response_length = candidate_batch.batch["responses"].shape[-1]
                # ⭐ 只取原始样本的 response_mask（去除 padding）
                response_masks = candidate_batch.batch["loss_mask"][:original_num_samples, -response_length:]  # [num_candidates, response_len]
                
                # 计算每个候选的平均 entropy（只考虑 LLM 响应部分）
                avg_entropys = []
                for i in range(len(candidates)):
                    if i < entropys.shape[0]:
                        traj_entropy = entropys[i].cpu().numpy()
                        traj_response_mask = response_masks[i].cpu().numpy()
                        
                        # 只计算 response_mask=1 的位置（LLM 响应）
                        valid_entropys = traj_entropy[traj_response_mask.astype(bool)]
                        avg_entropy = float(np.mean(valid_entropys)) if len(valid_entropys) > 0 else float('inf')
                    else:
                        avg_entropy = float('inf')
                    avg_entropys.append(avg_entropy)
                
                # 根据 exp_select_mode 选择最优轨迹
                exp_select_mode = self.exp_manager.exp_select_mode
                if exp_select_mode == "argmin":
                    # 选择 entropy 最低的 num_trajectories_per_task 个
                    sorted_indices = np.argsort(avg_entropys)
                elif exp_select_mode == "argmax":
                    # 选择 entropy 最高的 num_trajectories_per_task 个
                    sorted_indices = np.argsort(avg_entropys)[::-1]
                else:
                    # 随机选择
                    sorted_indices = np.random.permutation(len(candidates))
                
                # 选择 num_trajectories_per_task 个轨迹
                num_to_select = min(num_trajectories_per_task, len(candidates))
                for idx in sorted_indices[:num_to_select]:
                    selected_trajectories.append(candidates[idx])
                
                # 更新 task 的 n_offpolicy_trajectories
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = num_to_select
                else:
                    task.metadata = {"n_offpolicy_trajectories": num_to_select}
                
                logger.debug(
                    f"Task {task_id}: selected {num_to_select} off-policy trajectories "
                    f"(avg_entropys: {[f'{e:.4f}' for e in avg_entropys]})"
                )
                
            except Exception as e:
                logger.warning(f"Failed to compute entropy for task {task_id}: {e}")
                # 回退到使用保存时的 entropy
                if self.exp_manager.exp_select_mode == "argmin":
                    candidates.sort(key=lambda t: t.metadata.get("entropy", float('inf')))
                elif self.exp_manager.exp_select_mode == "argmax":
                    candidates.sort(key=lambda t: t.metadata.get("entropy", float('-inf')), reverse=True)
                
                num_to_select = min(num_trajectories_per_task, len(candidates))
                selected_trajectories.extend(candidates[:num_to_select])
                
                if hasattr(task, 'metadata') and task.metadata:
                    task.metadata["n_offpolicy_trajectories"] = num_to_select
                else:
                    task.metadata = {"n_offpolicy_trajectories": num_to_select}
        
        return selected_trajectories

    def _select_best_teacher_by_current_entropy(
        self,
        teacher_trajectories: List,
        tasks: List,
        num_trajectories_per_task: int = 1,
    ) -> List:
        """
        使用当前 policy 计算 entropy，选择每个 task 的最优 teacher 轨迹。
        
        ⭐ 对于有 log_prob 的轨迹：已经由 get_teacher_trajectories 按 entropy 排序
        ⭐ 对于没有 log_prob 的轨迹：使用当前 policy 计算 entropy 并排序
        ⭐ Multi-turn 关键：只对 LLM 响应部分（loss_mask=1）计算 entropy。
        
        Args:
            teacher_trajectories: Teacher 轨迹列表（可能包含有/无 log_prob 的混合）
            tasks: teacher experience task 列表
            num_trajectories_per_task: 每个 task 选择的轨迹数量
            
        Returns:
            List[Trajectory]: 选中的 teacher 轨迹列表（按 entropy 从低到高排序）
        """
        if not teacher_trajectories:
            return []
        
        # 按 task_id 分组轨迹
        task_id_to_trajs = defaultdict(list)
        for traj in teacher_trajectories:
            task_id = traj.task_id if hasattr(traj, 'task_id') else traj.metadata.get('task_id')
            if task_id:
                task_id_to_trajs[task_id].append(traj)
        
        selected_trajectories = []
        
        for task in tasks:
            task_id = task.task_id
            candidates = task_id_to_trajs.get(task_id, [])
            
            if not candidates:
                continue
            
            if len(candidates) == 1:
                selected_trajectories.append(candidates[0])
                continue
            
            # 分离有 log_prob 和没有 log_prob 的轨迹
            trajs_with_logprob = []
            trajs_without_logprob = []
            
            for traj in candidates:
                if traj.metadata.get("has_log_prob", False) and "old_log_probs" in traj.metadata:
                    trajs_with_logprob.append(traj)
                else:
                    trajs_without_logprob.append(traj)
            
            # 对于没有 log_prob 的轨迹，使用当前 policy 计算 entropy
            if trajs_without_logprob:
                try:
                    # 转换为 CMT
                    candidate_cmts = self.env_manager.convert_offpolicy_to_cmt(
                        offpolicy_trajectories=trajs_without_logprob,
                        config=self.config,
                        tokenizer=self.tokenizer
                    )
                    
                    if not candidate_cmts:
                        logger.warning(f"Task {task_id}: Failed to convert teacher trajectories to CMT")
                        # 回退：只使用有 log_prob 的轨迹
                        selected_trajectories.extend(trajs_with_logprob[:num_trajectories_per_task])
                        continue
                    
                    # 转换为 samples
                    samples = []
                    for cmt in candidate_cmts:
                        extras = self.env_manager.get_extra(cmt)
                        sample_arr = cmt.group_tokenize()
                        for sample in sample_arr:
                            sample.extras = extras
                        samples.extend(sample_arr)
                    
                    if not samples:
                        logger.warning(f"Task {task_id}: No samples generated from teacher CMTs")
                        selected_trajectories.extend(trajs_with_logprob[:num_trajectories_per_task])
                        continue
                    
                    # 对齐到 world_size（分布式计算需要）
                    world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                    original_num_samples = len(samples)
                    remainder = original_num_samples % world_size
                    if remainder != 0:
                        padding_needed = world_size - remainder
                        for i in range(padding_needed):
                            samples.append(samples[i % original_num_samples])
                    
                    # 转换为 DataProto 并计算 entropy
                    candidate_batch = self.env_manager.samples_to_dataproto(samples)
                    log_prob_result = self.actor_rollout_wg.compute_log_prob(candidate_batch)
                    entropys = log_prob_result.batch["entropys"][:original_num_samples]
                    
                    # 使用 loss_mask 计算 LLM 响应部分的平均 entropy
                    response_length = candidate_batch.batch["responses"].shape[-1]
                    response_masks = candidate_batch.batch["loss_mask"][:original_num_samples, -response_length:]
                    
                    # 计算每个候选的平均 entropy
                    traj_entropys = []
                    for i, traj in enumerate(trajs_without_logprob):
                        if i < entropys.shape[0]:
                            traj_entropy = entropys[i].cpu().numpy()
                            traj_response_mask = response_masks[i].cpu().numpy()
                            valid_entropys = traj_entropy[traj_response_mask.astype(bool)]
                            avg_entropy = float(np.mean(valid_entropys)) if len(valid_entropys) > 0 else float('inf')
                        else:
                            avg_entropy = float('inf')
                        traj_entropys.append((avg_entropy, traj))
                    
                    # 按 entropy 从低到高排序
                    trajs_without_logprob = [traj for _, traj in sorted(traj_entropys, key=lambda x: x[0])]
                    
                    logger.debug(
                        f"Task {task_id}: Computed entropy for {len(trajs_without_logprob)} teacher trajectories "
                        f"without log_prob (avg_entropys: {[f'{e:.4f}' for e, _ in sorted(traj_entropys, key=lambda x: x[0])]})"
                    )
                    
                except Exception as e:
                    logger.warning(f"Task {task_id}: Failed to compute entropy for teacher trajectories without log_prob: {e}")
                    # 回退：只使用有 log_prob 的轨迹
                    trajs_without_logprob = []
            
            # 合并：先是有 log_prob 的（已排序），然后是没有 log_prob 的（已排序）
            # 两者都按 entropy 从低到高排序
            all_sorted = trajs_with_logprob + trajs_without_logprob
            num_to_select = min(num_trajectories_per_task, len(all_sorted))
            selected_trajectories.extend(all_sorted[:num_to_select])
        
        return selected_trajectories

    ##################
    # ANNI
    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """
        Dumps rollout/validation samples as JSONL.

        Args:
            inputs (list): List of input data.
            outputs (list): List of output data.
            scores (list): List of score data.
            reward_extra_infos_dict (dict): Dictionary containing additional reward information.
            dump_path (str): Path to the directory where the JSONL file will be saved.

        Returns:
            None
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")  # ⭐ Create the filename for the JSONL file

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")  # ⭐ Write the data to the JSONL file

        print(f"Dumped generations to {filename}")

    def _save_trajectories_for_analysis(
        self,
        trajectories: List[Trajectory],
        tasks: List[Task],
        entropys: torch.Tensor,
        response_masks: torch.Tensor,
        global_steps: int,
        output_dir: str,
        batch_task_ids: Optional[np.ndarray] = None,
        batch_rollout_ids: Optional[np.ndarray] = None,
        batch_messages: Optional[np.ndarray] = None,
        batch_group_ids: Optional[torch.Tensor] = None,
        batch_reward_scores: Optional[np.ndarray] = None,
        # ⭐ New: batch-level tensors for deeper analysis
        batch_advantages: Optional[torch.Tensor] = None,          # (bs, response_len)
        batch_exp_mask: Optional[torch.Tensor] = None,            # (bs, full_len) or (bs, response_len)
        batch_teacher_mask: Optional[torch.Tensor] = None,         # (bs, response_len)
        batch_old_log_probs: Optional[torch.Tensor] = None,        # (bs, response_len)
        batch_token_level_rewards: Optional[torch.Tensor] = None,  # (bs, response_len)
        batch_uid: Optional[np.ndarray] = None,                   # (bs,)
        batch_diag_metrics: Optional[Dict[str, Any]] = None,       # aggregated metrics for this step
    ):
        """
        保存 Trajectory 信息用于后续分析。
        
        保存的信息包括：
        - messages: 从 trajectory.steps 中提取的消息列表
        - reward: trajectory.reward.outcome 和完整的 reward 信息
        - entropy: 从 entropys tensor 中提取的平均 entropy（只计算有效 token）
        - step: 训练步数 (global_steps)
        - task_id: 从 trajectory.metadata 或 tasks 中获取
        - data_id, rollout_id: 用于匹配和标识
        - is_terminated: 是否终止
        - 其他 metadata 信息
        
        Args:
            trajectories: Trajectory 对象列表
            tasks: Task 对象列表，用于获取 task_id
            entropys: Entropy tensor，shape (batch_size, response_len)
            response_masks: Response mask tensor，shape (batch_size, response_len)
            global_steps: 当前训练步数
            output_dir: 输出目录路径
        """
        if not trajectories:
            return
        
        # 创建 Trajectory 文件夹
        trajectory_dir = os.path.join(output_dir, "Trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # 创建 task_id 到 Task 的映射（用于获取 task_id）
        task_id_map = {}
        if tasks:
            # 如果 tasks 列表长度与 trajectories 匹配，可以直接索引
            # 否则需要通过 data_id 匹配
            for task in tasks:
                task_id_map[task.task_id] = task
        
        # 准备保存的数据
        saved_data = []

        # ---------------------------------------------------------------------
        # Debug printing (sampled): print some rollouts' dialogues during training
        # to help online inspection without opening JSONL files.
        # ---------------------------------------------------------------------
        print_prob = 0.1
        max_print_per_call = 3
        max_msg_chars = 1024  # truncate each message to avoid spamming logs

        def _truncate_text(s: str, n: int) -> str:
            if s is None:
                return ""
            s = str(s)
            return s if len(s) <= n else s[:n] + "\n…(truncated)…"

        def _format_messages(msgs: list) -> str:
            lines = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "unknown")
                content = _truncate_text(m.get("content", ""), max_msg_chars)
                lines.append(f"[{role}]: {content}")
            return "\n".join(lines)

        printed_cnt = 0
        
        # ---------------------------------------------------------------------
        # IMPORTANT (multi-gpu / balance_batch):
        # - `trainer.balance_batch=True` will reorder samples inside `batch`.
        # - `trajectories` keeps the original rollout order.
        # If we index-align `entropys[idx]` with `trajectories[idx]`, we may mix
        # stats from one rollout with messages from another.
        #
        # Fix: if batch-level identifiers are provided, align by (group_id, rollout_id)
        # where group_id == int(data_id) used in env_manager.samples_to_dataproto().
        # ---------------------------------------------------------------------

        traj_map: dict[tuple[int, str], Trajectory] = {}
        for t in trajectories:
            try:
                gid = int(t.data_id)
            except Exception:
                continue
            traj_map[(gid, str(t.rollout_id))] = t

        use_batch_alignment = (
            batch_rollout_ids is not None
            and batch_messages is not None
            and batch_group_ids is not None
            and entropys is not None
        )

        if use_batch_alignment:
            # Iterate in batch order (aligned with entropys/response_masks) and only
            # keep entries that correspond to on-policy trajectories we just rolled out.
            # NOTE: batch_group_ids is a tensor on device; bring to cpu once.
            batch_gids = batch_group_ids.detach().cpu().tolist()
            # prepare optional tensors in CPU/numpy for cheap per-sample stats
            resp_len = entropys.shape[1] if entropys is not None else None
            adv_cpu = batch_advantages.detach().cpu() if (batch_advantages is not None and torch.is_tensor(batch_advantages)) else None
            exp_mask_cpu = None
            if batch_exp_mask is not None and torch.is_tensor(batch_exp_mask) and resp_len is not None:
                if batch_exp_mask.dim() == 2 and batch_exp_mask.shape[-1] == resp_len:
                    exp_mask_cpu = batch_exp_mask.detach().cpu()
                elif batch_exp_mask.dim() == 2 and batch_exp_mask.shape[-1] > resp_len:
                    exp_mask_cpu = batch_exp_mask[:, -resp_len:].detach().cpu()
            teacher_mask_cpu = None
            if batch_teacher_mask is not None and torch.is_tensor(batch_teacher_mask) and resp_len is not None:
                if batch_teacher_mask.dim() == 2 and batch_teacher_mask.shape[-1] == resp_len:
                    teacher_mask_cpu = batch_teacher_mask.detach().cpu()
                elif batch_teacher_mask.dim() == 2 and batch_teacher_mask.shape[-1] > resp_len:
                    teacher_mask_cpu = batch_teacher_mask[:, -resp_len:].detach().cpu()
            oldlp_cpu = batch_old_log_probs.detach().cpu() if (batch_old_log_probs is not None and torch.is_tensor(batch_old_log_probs)) else None
            tlr_cpu = batch_token_level_rewards.detach().cpu() if (batch_token_level_rewards is not None and torch.is_tensor(batch_token_level_rewards)) else None
            rm_cpu = response_masks.detach().cpu() if (response_masks is not None and torch.is_tensor(response_masks)) else None
            uid_list = [str(u) for u in list(batch_uid)] if batch_uid is not None else None

            for bidx in range(min(len(batch_gids), entropys.shape[0])):
                gid = int(batch_gids[bidx])
                rollout_id = str(batch_rollout_ids[bidx])
                traj = traj_map.get((gid, rollout_id))
                if traj is None:
                    continue

                data_id = traj.data_id

                # 获取 task_id（优先使用 batch_task_ids，其次 fallback 到 traj/tasks）
                task_id = None
                if batch_task_ids is not None and bidx < len(batch_task_ids):
                    task_id = str(batch_task_ids[bidx])
                elif hasattr(traj, "task_id"):
                    task_id = traj.task_id
                elif traj.metadata and "task_id" in traj.metadata:
                    task_id = traj.metadata["task_id"]
                elif tasks and bidx < len(tasks):
                    task_id = tasks[bidx].task_id
                if task_id is None:
                    task_id = f"unknown_{data_id}_{rollout_id}"

                # messages: from batch (aligned) if possible
                messages = []
                bm = batch_messages[bidx]
                if isinstance(bm, dict) and "messages" in bm:
                    messages = bm["messages"]
                elif isinstance(bm, list):
                    messages = bm

                # reward: prefer batch reward_scores if provided (aligned)
                reward_info = None
                if batch_reward_scores is not None and bidx < len(batch_reward_scores):
                    rs = batch_reward_scores[bidx]
                    if isinstance(rs, dict):
                        reward_info = rs
                elif traj.reward:
                    reward_info = {
                        "outcome": traj.reward.outcome,
                        "success_rate": traj.reward.success_rate,
                        "madness": traj.reward.madness,
                        "description": traj.reward.description,
                    }
                    if hasattr(traj.reward, "metadata") and traj.reward.metadata:
                        reward_info["metadata"] = traj.reward.metadata

                # entropy stats
                entropy_info = None
                traj_entropy = entropys[bidx].detach().cpu().numpy()
                if response_masks is not None and bidx < response_masks.shape[0]:
                    traj_response_mask = response_masks[bidx].detach().cpu().numpy()
                    valid_entropys = traj_entropy[traj_response_mask.astype(bool)]
                    entropy_info = {
                        "mean": float(np.mean(valid_entropys)) if len(valid_entropys) > 0 else None,
                        "std": float(np.std(valid_entropys)) if len(valid_entropys) > 0 else None,
                        "min": float(np.min(valid_entropys)) if len(valid_entropys) > 0 else None,
                        "max": float(np.max(valid_entropys)) if len(valid_entropys) > 0 else None,
                        "num_valid_tokens": int(len(valid_entropys)),
                        "total_tokens": int(len(traj_entropy)),
                    }
                else:
                    entropy_info = {
                        "mean": float(np.mean(traj_entropy)),
                        "std": float(np.std(traj_entropy)),
                        "min": float(np.min(traj_entropy)),
                        "max": float(np.max(traj_entropy)),
                        "num_valid_tokens": int(len(traj_entropy)),
                        "total_tokens": int(len(traj_entropy)),
                    }

                # ---- extra per-rollout diagnostics (from batch tensors) ----
                extra_diag: dict[str, Any] = {}
                if rm_cpu is not None and bidx < rm_cpu.shape[0]:
                    rm = rm_cpu[bidx].numpy().astype(bool)
                    extra_diag["response_valid_tokens"] = int(rm.sum())
                if exp_mask_cpu is not None and bidx < exp_mask_cpu.shape[0]:
                    em = exp_mask_cpu[bidx].numpy().astype(bool)
                    extra_diag["exp_tokens"] = int(em.sum())
                    extra_diag["offpolicy_ratio"] = float(em.sum() / max(1, extra_diag.get("response_valid_tokens", int(em.size))))
                if teacher_mask_cpu is not None and bidx < teacher_mask_cpu.shape[0]:
                    tm = teacher_mask_cpu[bidx].numpy().astype(bool)
                    extra_diag["teacher_tokens"] = int(tm.sum())
                    extra_diag["is_teacher"] = bool(tm.any())
                if adv_cpu is not None and rm_cpu is not None and bidx < adv_cpu.shape[0] and bidx < rm_cpu.shape[0]:
                    a = adv_cpu[bidx].numpy()
                    rm = rm_cpu[bidx].numpy().astype(bool)
                    extra_diag["adv_mean"] = float(a[rm].mean()) if rm.any() else None
                    extra_diag["adv_std"] = float(a[rm].std()) if rm.any() else None
                if oldlp_cpu is not None and rm_cpu is not None and bidx < oldlp_cpu.shape[0] and bidx < rm_cpu.shape[0]:
                    olp = oldlp_cpu[bidx].numpy()
                    rm = rm_cpu[bidx].numpy().astype(bool)
                    extra_diag["old_log_prob_mean"] = float(olp[rm].mean()) if rm.any() else None
                    # --- teacher-token-only logp distribution (critical for TER gating debugging) ---
                    if teacher_mask_cpu is not None and bidx < teacher_mask_cpu.shape[0]:
                        tm = teacher_mask_cpu[bidx].numpy().astype(bool)
                        m = rm & tm
                        if m.any():
                            v = olp[m].astype(float)
                            extra_diag["teacher_old_logp_mean"] = float(v.mean())
                            extra_diag["teacher_old_logp_min"] = float(v.min())
                            # percentiles are robust summaries for tail behavior
                            try:
                                extra_diag["teacher_old_logp_p10"] = float(np.quantile(v, 0.10))
                                extra_diag["teacher_old_logp_p50"] = float(np.quantile(v, 0.50))
                            except Exception:
                                pass

                            # --- seq_beta(7.7) replay diagnostics (approx): compute using old_log_probs ---
                            # This enables case-level postmortem even if wandb only has aggregated teacher_diag.
                            try:
                                seq_cfg = getattr(self.config.exp_manager.teacher_experience, "seq_beta_77", None)
                            except Exception:
                                seq_cfg = None
                            if seq_cfg is not None and getattr(seq_cfg, "enable", False):
                                conf_mode = str(getattr(seq_cfg, "conf_mode", "gen_mean_prob")).lower().strip()
                                q = float(getattr(seq_cfg, "logp_q", 0.10))
                                if not (0.0 < q < 1.0):
                                    q = 0.10
                                max_tok = int(getattr(seq_cfg, "max_tokens_per_traj", 2048))
                                if max_tok <= 0:
                                    max_tok = 2048
                                vv = v
                                if vv.size > max_tok:
                                    idx = np.random.permutation(vv.size)[:max_tok]
                                    vv = vv[idx]
                                vv_sorted = np.sort(vv)
                                k = max(1, int(round(q * vv_sorted.size)))
                                if conf_mode == "quantile_logp":
                                    logC = float(vv_sorted[k - 1])
                                elif conf_mode == "cvar_logp":
                                    logC = float(vv_sorted[:k].mean())
                                else:
                                    # legacy: generalized mean on prob space using alpha<0
                                    alpha = float(getattr(seq_cfg, "alpha", -5.0))
                                    if alpha >= 0:
                                        alpha = -5.0
                                    p_min = float(getattr(seq_cfg, "p_min", 1e-8))
                                    if p_min <= 0:
                                        p_min = 1e-8
                                    # logC = (1/alpha) * log(mean(exp(alpha*logp))) ; logp<=0
                                    x = np.clip(alpha * np.log(np.clip(np.exp(vv), p_min, 1.0)), -80.0, 80.0)
                                    mean_pa = float(np.mean(np.exp(x)))
                                    mean_pa = max(mean_pa, 1e-20)
                                    logC = float((1.0 / alpha) * np.log(mean_pa))

                                extra_diag["seq_beta_logC_old"] = float(logC)
                                # gate & beta (logprob space preferred)
                                gate_space = str(getattr(seq_cfg, "gate_space", "logprob")).lower().strip()
                                beta_min = float(getattr(seq_cfg, "beta_min", 0.05))
                                beta_max = float(getattr(seq_cfg, "beta_max", 0.35))
                                if beta_min > beta_max:
                                    beta_min, beta_max = beta_max, beta_min
                                if gate_space == "logprob":
                                    logc0 = getattr(seq_cfg, "logc0", None)
                                    if logc0 is None:
                                        # fallback to log(c0) if provided
                                        c0 = float(getattr(seq_cfg, "c0", 0.25))
                                        logc0 = float(np.log(c0)) if (0.0 < c0 < 1.0) else -6.0
                                    Tlog = float(getattr(seq_cfg, "log_temperature", 1.0))
                                    if Tlog <= 0:
                                        Tlog = 1.0
                                    gate = 1.0 / (1.0 + np.exp(-(logC - float(logc0)) / Tlog))
                                else:
                                    # prob-space (legacy)
                                    c0 = float(getattr(seq_cfg, "c0", 0.25))
                                    T = float(getattr(seq_cfg, "temperature", 0.05))
                                    if T <= 0:
                                        T = 0.05
                                    C = float(np.exp(logC))
                                    gate = 1.0 / (1.0 + np.exp(-(C - c0) / T))
                                beta = beta_min + (beta_max - beta_min) * float(gate)
                                extra_diag["seq_beta_beta_old"] = float(beta)
                                extra_diag["seq_beta_gate_old"] = float(gate)
                        else:
                            extra_diag["teacher_old_logp_mean"] = None
                if tlr_cpu is not None and rm_cpu is not None and bidx < tlr_cpu.shape[0] and bidx < rm_cpu.shape[0]:
                    r = tlr_cpu[bidx].numpy()
                    rm = rm_cpu[bidx].numpy().astype(bool)
                    extra_diag["reward_sum"] = float(r[rm].sum()) if rm.any() else float(r.sum())
                if uid_list is not None and bidx < len(uid_list):
                    extra_diag["uid"] = uid_list[bidx]

                traj_data = {
                    "data_id": data_id,
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "step": global_steps,
                    "query": traj.query,
                    "messages": messages,
                    "reward": reward_info,
                    "entropy": entropy_info,
                    "diag": extra_diag,
                    "is_terminated": traj.is_terminated,
                    "success": traj.success if hasattr(traj, "success") else (traj.reward is not None and traj.reward.outcome > 0),
                }

                # metadata (same behavior as before)
                if traj.metadata:
                    metadata_copy = {}
                    for k, v in traj.metadata.items():
                        if k not in ["task_id", "old_log_probs", "response_mask"]:
                            try:
                                json.dumps(v, ensure_ascii=False)
                                metadata_copy[k] = v
                            except (TypeError, ValueError):
                                try:
                                    metadata_copy[k] = str(v)
                                except Exception:
                                    pass
                    if metadata_copy:
                        traj_data["metadata"] = metadata_copy

                saved_data.append(traj_data)

                # sampled printing
                if printed_cnt < max_print_per_call and random.random() < print_prob:
                    try:
                        logger.info(
                            "\n"
                            f"========== [Trajectory Sample] step={global_steps} ==========\n"
                            f"task_id={task_id} data_id={data_id} rollout_id={rollout_id} "
                            f"success={traj_data.get('success')} terminated={traj_data.get('is_terminated')}\n"
                            f"entropy_valid/total={entropy_info.get('num_valid_tokens') if entropy_info else None}/"
                            f"{entropy_info.get('total_tokens') if entropy_info else None}\n"
                            f"{_format_messages(messages)}\n"
                            f"========== [End Trajectory Sample] =========="
                        )
                        printed_cnt += 1
                    except Exception as _e:
                        logger.warning(f"Failed to print sampled trajectory: {_e}")
        else:
            # Fallback to original (index-based) behavior
            for idx, traj in enumerate(trajectories):
                # 提取基本信息
                data_id = traj.data_id
                rollout_id = traj.rollout_id
            
            # 获取 task_id
            task_id = None
            if hasattr(traj, 'task_id'):
                task_id = traj.task_id
            elif traj.metadata and "task_id" in traj.metadata:
                task_id = traj.metadata["task_id"]
            elif tasks and idx < len(tasks):
                task_id = tasks[idx].task_id
            else:
                # 尝试从 data_id 匹配
                try:
                    data_id_int = int(data_id)
                    if data_id_int < len(tasks):
                        task_id = tasks[data_id_int].task_id
                except (ValueError, TypeError, IndexError):
                    pass
            
            if task_id is None:
                task_id = f"unknown_{data_id}_{rollout_id}"
            
            # 提取 messages（从 steps 中）
            messages = []
            if traj.steps:
                for step in traj.steps:
                    if isinstance(step, dict):
                        # steps 已经是消息格式
                        msg = {
                            "role": step.get("role", "unknown"),
                            "content": step.get("content", ""),
                        }
                        # 保留其他可能的字段
                        if "tool_calls" in step:
                            msg["tool_calls"] = step["tool_calls"]
                        if "timestamp" in step:
                            msg["timestamp"] = step["timestamp"]
                        messages.append(msg)
            
            # 提取 reward 信息
            reward_info = None
            if traj.reward:
                reward_info = {
                    "outcome": traj.reward.outcome,
                    "success_rate": traj.reward.success_rate,
                    "madness": traj.reward.madness,
                    "description": traj.reward.description,
                }
                if hasattr(traj.reward, 'metadata') and traj.reward.metadata:
                    reward_info["metadata"] = traj.reward.metadata
            
            # 提取 entropy（从 entropys tensor 中）
            entropy_info = None
            if entropys is not None and idx < entropys.shape[0]:
                traj_entropy = entropys[idx].cpu().numpy()  # (response_len,)
                
                # 获取对应的 response_mask
                if response_masks is not None and idx < response_masks.shape[0]:
                    traj_response_mask = response_masks[idx].cpu().numpy()  # (response_len,)
                    
                    # 只计算有效 token 的 entropy
                    valid_entropys = traj_entropy[traj_response_mask.astype(bool)]
                    
                    if len(valid_entropys) > 0:
                        entropy_info = {
                            "mean": float(np.mean(valid_entropys)),
                            "std": float(np.std(valid_entropys)),
                            "min": float(np.min(valid_entropys)),
                            "max": float(np.max(valid_entropys)),
                            "num_valid_tokens": int(len(valid_entropys)),
                            "total_tokens": int(len(traj_entropy)),
                        }
                    else:
                        entropy_info = {
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None,
                            "num_valid_tokens": 0,
                            "total_tokens": int(len(traj_entropy)),
                        }
                else:
                    # 如果没有 mask，计算所有 token 的统计
                    entropy_info = {
                        "mean": float(np.mean(traj_entropy)),
                        "std": float(np.std(traj_entropy)),
                        "min": float(np.min(traj_entropy)),
                        "max": float(np.max(traj_entropy)),
                        "num_valid_tokens": int(len(traj_entropy)),
                        "total_tokens": int(len(traj_entropy)),
                    }
            
            # 构建保存的数据结构
            traj_data = {
                "data_id": data_id,
                "rollout_id": rollout_id,
                "task_id": task_id,
                "step": global_steps,
                "query": traj.query,
                "messages": messages,
                "reward": reward_info,
                "entropy": entropy_info,
                "is_terminated": traj.is_terminated,
                "success": traj.success if hasattr(traj, 'success') else (traj.reward is not None and traj.reward.outcome > 0),
            }
            
            # 添加 metadata（排除已经单独提取的字段）
            if traj.metadata:
                metadata_copy = {}
                for k, v in traj.metadata.items():
                    # 跳过已经单独提取的字段
                    if k not in ["task_id", "old_log_probs", "response_mask"]:
                        # 尝试序列化，如果失败则跳过
                        try:
                            json.dumps(v, ensure_ascii=False)
                            metadata_copy[k] = v
                        except (TypeError, ValueError):
                            # 如果无法序列化，尝试转换为字符串
                            try:
                                metadata_copy[k] = str(v)
                            except:
                                pass
                if metadata_copy:
                    traj_data["metadata"] = metadata_copy
            
                saved_data.append(traj_data)

                # sampled printing (fallback)
                if printed_cnt < max_print_per_call and random.random() < print_prob:
                    try:
                        entropy_info = traj_data.get("entropy") or {}
                        logger.info(
                            "\n"
                            f"========== [Trajectory Sample] step={global_steps} ==========\n"
                            f"task_id={traj_data.get('task_id')} data_id={traj_data.get('data_id')} "
                            f"rollout_id={traj_data.get('rollout_id')} "
                            f"success={traj_data.get('success')} terminated={traj_data.get('is_terminated')}\n"
                            f"entropy_valid/total={entropy_info.get('num_valid_tokens')}/{entropy_info.get('total_tokens')}\n"
                            f"{_format_messages(traj_data.get('messages', []))}\n"
                            f"========== [End Trajectory Sample] =========="
                        )
                        printed_cnt += 1
                    except Exception as _e:
                        logger.warning(f"Failed to print sampled trajectory: {_e}")
        
        # 保存为 JSONL 文件
        filename = os.path.join(trajectory_dir, f"trajectories_step_{global_steps}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for data in saved_data:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(saved_data)} trajectories to {filename}")

        # ⭐ 同步保存 batch 级诊断指标（便于后续整体解读）
        if batch_diag_metrics:
            summary_path = os.path.join(trajectory_dir, f"batch_diag_step_{global_steps}.json")
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(batch_diag_metrics, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved batch diagnostics to {summary_path}")
            except Exception as e:
                logger.warning(f"Failed to save batch diagnostics: {e}")


    def _validate(self):
        """
        Validates the model by generating sequences, collecting samples, and storing the results.

        This function processes each batch of validation data, generates outputs, and collects
        input, output, and experience information for further analysis.

        Args:
            None

        Returns:
            None
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_success_rates = []
        # Summary counters (help interpret what val metrics actually average over)
        n_val_tasks = 0

        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                            # evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'], # avoid potential bugs
                         ) for i in range(len(test_gen_batch))]
                n_val_tasks += len(tasks)
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ Execute the rollout to generate trajectories
                print("=" * 10 + "end validate rollout" + "=" * 10)
                test_output_gen_batch = self.env_manager.to_dataproto(trajectories)
                # test_output_gen_batch_padded = self.explorer_manager.rollout(test_gen_batch_padded)
                # test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store original inputs
            input_ids = test_output_gen_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # repeat test batch
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            test_batch = union_gen_batch_via_task_id(tasks, test_batch, test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)  # ⭐ Evaluate the test batch using the reward function
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            # Also collect success_rate from env rollouts (if available)
            try:
                rs_list = test_batch.non_tensor_batch.get("reward_scores", None)
                if rs_list is None and hasattr(test_output_gen_batch, "non_tensor_batch"):
                    rs_list = test_output_gen_batch.non_tensor_batch.get("reward_scores", None)
                if rs_list is not None:
                    sr_list: list[float] = []
                    for rs in list(rs_list):
                        if isinstance(rs, dict):
                            sr = rs.get("success_rate", None)
                            if sr is None:
                                # fallback: treat outcome>0 as success
                                sr = 1.0 if float(rs.get("outcome", 0.0)) > 0.0 else 0.0
                            sr_list.append(float(sr))
                        else:
                            # unknown type; fallback to 0
                            sr_list.append(0.0)
                    if len(sr_list) == len(scores):
                        sample_success_rates.extend(sr_list)
                    else:
                        logger.warning(
                            f"validate: success_rate length mismatch (success_rate={len(sr_list)}, scores={len(scores)}); skip val-summary success_rate for this batch"
                        )
            except Exception as e:
                logger.warning(f"validate: failed to collect success_rate: {e}")

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # val_data_dir = "experiments/validation_log"
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)  # ⭐ Process the validation metrics for different data sources
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # ------------------------------------------------------------------
        # ✅ Add "true" aggregate validation metrics over ALL validation outputs
        # ------------------------------------------------------------------
        # These metrics are what people usually mean by "average val performance":
        # average reward over all validation trajectories produced in this _validate() call.
        try:
            import numpy as _np

            env_name = str(self.config.env_service.env_type)
            scores_arr = _np.asarray(sample_scores, dtype=float)
            metric_dict[f"val-summary/{env_name}/reward_mean_all"] = float(scores_arr.mean()) if scores_arr.size else 0.0
            metric_dict[f"val-summary/{env_name}/reward_std_all"] = float(scores_arr.std()) if scores_arr.size else 0.0
            metric_dict[f"val-summary/{env_name}/n_outputs"] = int(scores_arr.size)
            metric_dict[f"val-summary/{env_name}/n_tasks"] = int(n_val_tasks)
            metric_dict[f"val-summary/{env_name}/n_unique_prompts"] = int(len(set(sample_inputs)))
            sr_arr = _np.asarray(sample_success_rates, dtype=float)
            metric_dict[f"val-summary/{env_name}/success_rate_mean_all"] = float(sr_arr.mean()) if sr_arr.size else 0.0
        except Exception as e:
            logger.warning(f"failed to compute val-summary metrics: {e}")

        return metric_dict
    
    def initialize_exp_pool(self):
        """
        """
        for i, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "extras" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("extras")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                raise NotImplementedError

            else:
                self.async_rollout_manager.wake_up()
                tasks = [Task(
                            task_id=test_gen_batch.non_tensor_batch["extras"][i]["task_id"],
                            query=test_gen_batch.non_tensor_batch["extras"][i]['new_query'],
                            env_type=self.config.env_service.env_type,
                            open_query=test_gen_batch.non_tensor_batch["extras"][i]['open_query'],
                            # evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'], # avoid potential bugs
                         ) for i in range(len(test_gen_batch))]
                task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="validate")
                print("=" * 10 + "start validate rollout" + "=" * 10)
                trajectories = self.env_manager.rollout(tasks, task_exp_configs, mode="validate", epoch=f"test.1.{i}")  # ⭐ Execute the rollout to generate trajectories
                print("=" * 10 + "end validate rollout" + "=" * 10)
                self.async_rollout_manager.sleep()

            # summarize in batch: updating experience pool
            self.exp_manager.summarize_in_batch(trajectories)
        
        return


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from agentevolver.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # ⭐ Experience Replay: 加载 experience pool（如果有）
        exp_replay_config = self.config.exp_manager.get("experience_replay", {})
        if exp_replay_config.get("enable", False):
            # 尝试从最新的 checkpoint 目录加载 experience pool
            exp_pool_base_dir = os.path.join(self.config.trainer.default_local_dir, "experience_pool")
            if os.path.exists(exp_pool_base_dir):
                # 找到最新的 step 目录
                step_dirs = [d for d in os.listdir(exp_pool_base_dir) if d.startswith("step_")]
                if step_dirs:
                    latest_step_dir = max(step_dirs, key=lambda x: int(x.split("_")[1]))
                    exp_pool_load_dir = os.path.join(exp_pool_base_dir, latest_step_dir)
                    self.exp_manager.load_experience_pool_from_disk(exp_pool_load_dir)
                    logger.info(f"Loaded experience pool from {exp_pool_load_dir}")
        
        # spread parameters to vllm
        self.async_rollout_manager.wake_up()
        self.async_rollout_manager.sleep()

        # initialize experience pool
        if self.config.exp_manager.get("init_exp_before_training", False):
            self.initialize_exp_pool()
            if self.config.exp_manager.get("init_exp_only", False):
                return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()  # ⭐ Perform initial validation and get the validation metrics
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # [0616] qingxu: add `RAY_DEBUG_POST_MORTEM` env var to activate breakpoint debugging
        # vscode_conditional_breakpoint()
        # breakpoint()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            for i, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "extras" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("extras")
                    batch_extras = deepcopy(batch.non_tensor_batch["extras"])
                else:
                    batch_extras = None
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        trajectories: List[Trajectory] = []
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            # gen_batch_output = self.explorer_manager.rollout(gen_batch)

                            tasks = [Task(
                                        task_id=gen_batch.non_tensor_batch["extras"][i]["task_id"],
                                        query=gen_batch.non_tensor_batch["extras"][i]['new_query'],
                                        env_type=self.config.env_service.env_type,
                                        open_query=gen_batch.non_tensor_batch["extras"][i]['open_query'],
                                        evaluator=gen_batch.non_tensor_batch['extras'][i]['evaluator'],
                                        ground_truth=gen_batch.non_tensor_batch['extras'][i]['ground_truth']
                                    ) for i in range(len(gen_batch))
                            ]
                            
                            # ⭐ Experience Replay: 混合 on-policy 和 off-policy tasks
                            exp_replay_config = self.config.exp_manager.get("experience_replay", {})
                            teacher_exp_config = self.config.exp_manager.get("teacher_experience", {})
                            enable_exp_replay = exp_replay_config.get("enable", False)
                            enable_teacher_exp = teacher_exp_config.get("enable", False) and self.exp_manager.teacher_enabled
                            
                            # ⭐ LUFFY vs ExGRPO: 检查混合模式
                            # - "rollout_level": LUFFY 风格，每个 task 内混合 on-policy 和 teacher rollouts
                            # - "task_level": ExGRPO 风格，batch 中某些 task 完全是 teacher experience
                            teacher_mix_mode = teacher_exp_config.get("mix_mode", "rollout_level")
                            use_luffy_rollout_level = (
                                enable_teacher_exp and 
                                teacher_mix_mode == "rollout_level"
                            )
                            
                            experience_tasks = []
                            teacher_exp_tasks = []  # ⭐ Task 级别：Teacher experience tasks
                            offpolicy_cmt_array = []
                            teacher_offpolicy_cmt_array = []  # ⭐ Task 级别：Teacher off-policy trajectories
                            luffy_mixer = None  # ⭐ Rollout 级别：LUFFY mixer
                            
                            if use_luffy_rollout_level:
                                # ⭐⭐⭐ LUFFY 风格：Rollout 级别混合 ⭐⭐⭐
                                # 每个 task 的 n 个 rollouts 中混合 on-policy 和 teacher rollouts
                                n_teacher_rollouts = teacher_exp_config.get("n_teacher_rollouts_per_task", 1)
                                n_rollout = self.config.actor_rollout_ref.rollout.n
                                
                                logger.info(
                                    f"[LUFFY] Rollout-level mixing: "
                                    f"n_rollout={n_rollout}, n_teacher_per_task={n_teacher_rollouts}, "
                                    f"n_onpolicy_per_task={n_rollout - n_teacher_rollouts}"
                                )
                                
                                # 创建 LUFFY mixer
                                luffy_mixer = LUFFYTeacherRolloutMixer(
                                    exp_manager=self.exp_manager,
                                    n_teacher_rollouts_per_task=n_teacher_rollouts,
                                    n_rollout=n_rollout,
                                )
                                
                                # ⭐ 所有 tasks 都进行 on-policy rollout，但数量减少
                                # 不需要修改 tasks 列表，因为我们会在 rollout 后混入 teacher
                                # 注意：需要修改 rollout 的 n 参数
                                
                            elif enable_exp_replay or enable_teacher_exp:
                                # ⭐⭐⭐ ExGRPO 风格：Task 级别混合（向后兼容）⭐⭐⭐
                                # 计算当前训练进度
                                training_progress = self.global_steps / self.total_training_steps
                                replay_start_ratio = exp_replay_config.get("replay_start_ratio", 0.35)
                                
                                if training_progress >= replay_start_ratio:
                                    # ⭐ 判断是否使用 Teacher Experience
                                    if enable_teacher_exp and teacher_mix_mode == "task_level":
                                        # ⭐ Task 级别：Teacher 从总 off-policy 比例中分走一部分
                                        total_exp_ratio = exp_replay_config.get("exp_ratio", 0.5)
                                        teacher_exp_ratio = teacher_exp_config.get("exp_ratio", 0.2)
                                        teacher_exp_ratio = min(teacher_exp_ratio, total_exp_ratio)
                                        self_exp_ratio = max(0.0, total_exp_ratio - teacher_exp_ratio)
                                        
                                        logger.info(
                                            f"[Task-level] 比例分配: "
                                            f"total_exp={total_exp_ratio:.2f}, "
                                            f"self_exp={self_exp_ratio:.2f}, "
                                            f"teacher_exp={teacher_exp_ratio:.2f}, "
                                            f"on_policy={1-total_exp_ratio:.2f}"
                                        )
                                        
                                        experience_mix_collate = TeacherExperienceMixCollateFn(
                                            exp_manager=self.exp_manager,
                                            train_task_manager=self.train_task_manager,
                                            self_exp_ratio=self_exp_ratio,
                                            teacher_exp_ratio=teacher_exp_ratio,
                                            teacher_exp_enabled=True,
                                            replay_start_ratio=replay_start_ratio,
                                            offpolicy_trajectories_per_task=exp_replay_config.get("offpolicy_trajectories_per_task", 1),
                                            n_rollout=self.config.actor_rollout_ref.rollout.n,
                                        )
                                        
                                        experience_tasks, teacher_exp_tasks, on_policy_tasks = experience_mix_collate(
                                            training_tasks=tasks,
                                            training_progress=training_progress,
                                            enable_replay=True,
                                        )
                                    elif enable_exp_replay:
                                        # 使用原有的 ExperienceMixCollateFn（向后兼容）
                                        experience_mix_collate = ExperienceMixCollateFn(
                                            exp_manager=self.exp_manager,
                                            train_task_manager=self.train_task_manager,
                                            exp_ratio=exp_replay_config.get("exp_ratio", 0.5),
                                            replay_start_ratio=replay_start_ratio,
                                            offpolicy_trajectories_per_task=exp_replay_config.get("offpolicy_trajectories_per_task", 1),
                                            n_rollout=self.config.actor_rollout_ref.rollout.n,
                                        )
                                        
                                        experience_tasks, on_policy_tasks = experience_mix_collate(
                                            training_tasks=tasks,
                                            training_progress=training_progress,
                                            enable_replay=True,
                                        )
                                        # 统一变量名，便于后续合并逻辑复用
                                        teacher_exp_tasks = []
                                    
                                    # 合并 tasks（experience_tasks + teacher_exp_tasks + on_policy_tasks）
                                    tasks = experience_tasks + teacher_exp_tasks + on_policy_tasks
                                    
                                    # 为 self-generated experience tasks 获取 off-policy trajectories
                                    if experience_tasks:
                                        use_current_policy_entropy = exp_replay_config.get("use_current_policy_entropy", True)
                                        num_trajectories_per_task = exp_replay_config.get("offpolicy_trajectories_per_task", 1)
                                        
                                        if use_current_policy_entropy:
                                            task_to_candidates = self.exp_manager.get_all_candidates_batch(
                                                tasks=experience_tasks
                                            )
                                            offpolicy_trajectories = self._select_best_offpolicy_by_current_entropy(
                                                task_to_candidates=task_to_candidates,
                                                tasks=experience_tasks,
                                                num_trajectories_per_task=num_trajectories_per_task,
                                            )
                                        else:
                                            offpolicy_trajectories = self.exp_manager.get_offpolicy_batch(
                                                tasks=experience_tasks,
                                                num_trajectories_per_task=num_trajectories_per_task,
                                                use_saved_entropy=True
                                            )
                                        
                                        if offpolicy_trajectories:
                                            task_id_to_data_id = {
                                                task.task_id: idx
                                                for idx, task in enumerate(tasks)
                                            }
                                            
                                            offpolicy_cmt_array = self.env_manager.convert_offpolicy_to_cmt(
                                                offpolicy_trajectories=offpolicy_trajectories,
                                                config=self.config,
                                                tokenizer=self.tokenizer,
                                                task_id_to_data_id=task_id_to_data_id
                                            )
                                            logger.info(f"Got {len(offpolicy_cmt_array)} self-generated off-policy trajectories")
                                    
                                    # ⭐ Task 级别：为 teacher experience tasks 获取 off-policy trajectories
                                    if teacher_exp_tasks:
                                        teacher_num_per_task = teacher_exp_config.get("max_trajectories_per_task", 1)
                                        teacher_offpolicy_trajectories = self.exp_manager.get_teacher_offpolicy_batch(
                                            tasks=teacher_exp_tasks,
                                            num_trajectories_per_task=teacher_num_per_task,
                                        )
                                        
                                        if teacher_offpolicy_trajectories:
                                            teacher_select_mode = teacher_exp_config.get("select_mode", "random")
                                            if teacher_select_mode == "entropy":
                                                teacher_offpolicy_trajectories = self._select_best_teacher_by_current_entropy(
                                                    teacher_trajectories=teacher_offpolicy_trajectories,
                                                    tasks=teacher_exp_tasks,
                                                    num_trajectories_per_task=teacher_num_per_task,
                                                )
                                            
                                            task_id_to_data_id = {
                                                task.task_id: idx
                                                for idx, task in enumerate(tasks)
                                            }
                                            
                                            teacher_offpolicy_cmt_array = self.env_manager.convert_offpolicy_to_cmt(
                                                offpolicy_trajectories=teacher_offpolicy_trajectories,
                                                config=self.config,
                                                tokenizer=self.tokenizer,
                                                task_id_to_data_id=task_id_to_data_id
                                            )
                                            for cmt in teacher_offpolicy_cmt_array:
                                                cmt.metadata["is_teacher"] = True
                                            logger.info(f"Got {len(teacher_offpolicy_cmt_array)} teacher off-policy trajectories")
                            
                            task_exp_configs = self.exp_manager.get_complete_exp_configs(tasks, mode="sample")
                            assert len(task_exp_configs)==len(tasks), "{len(task_exp_configs)=}, {len(gen_batch)=}"

                            # TODO enable tracing by jinli 0619
                            print("=" * 10 + "start fit rollout" + "=" * 10)
                            
                            # ⭐ LUFFY 模式：生成完整的 n_rollout on-policy，然后用 teacher 替换部分
                            # 不再减少 on-policy 数量，这样可以自动处理 teacher 不足的情况
                            trajectories = self.env_manager.rollout(
                                tasks, 
                                task_exp_configs, 
                                mode="sample", 
                                epoch=f"train.{epoch}.{i}",
                            )
                            assert len(trajectories)>0, "{len(trajectories)=}?"
                            print("=" * 10 + "end fit rollout" + "=" * 10)
                            
                            # ⭐ Experience Replay: 更新 difficulty2task_dict 并合并轨迹
                            if use_luffy_rollout_level and luffy_mixer:
                                # ⭐⭐⭐ LUFFY 风格：Rollout 级别混合（替换模式）⭐⭐⭐
                                # 1. 更新 difficulty2task_dict（用完整的 on-policy 轨迹）
                                self.exp_manager.update_difficulty2task_dict(trajectories)
                                
                                # 2. 使用 LUFFY mixer 用 teacher rollouts 替换部分 on-policy
                                # 如果 teacher 不足，会自动保留更多 on-policy（总数保持 n_rollout）
                                # trajectories 是 CMT 对象列表（不是 Trajectory）
                                all_trajectories, luffy_stats = luffy_mixer.mix_trajectories(
                                    on_policy_cmt_array=trajectories,
                                    tasks=tasks,
                                    env_manager=self.env_manager,
                                    config=self.config,
                                    tokenizer=self.tokenizer,
                                )
                                
                                # 更新 metrics
                                metrics.update({
                                    "luffy/tasks_with_full_teacher": luffy_stats.get("tasks_with_full_teacher", 0),
                                    "luffy/tasks_with_partial_teacher": luffy_stats.get("tasks_with_partial_teacher", 0),
                                    "luffy/tasks_without_teacher": luffy_stats.get("tasks_without_teacher", 0),
                                    "luffy/total_teacher_rollouts": luffy_stats["total_teacher"],
                                    "luffy/total_onpolicy_kept": luffy_stats.get("total_onpolicy_kept", 0),
                                    "luffy/onpolicy_replaced_by_teacher": luffy_stats.get("total_onpolicy_replaced", 0),
                                })
                                # ⭐ 7.9: difficulty-aware teacher mixing diagnostics (optional)
                                if isinstance(luffy_stats, dict) and luffy_stats.get("difficulty_gate_79_enable", 0):
                                    metrics.update({
                                        "luffy79/threshold": luffy_stats.get("difficulty_gate_79_threshold", 0.0),
                                        "luffy79/min_onpolicy_rollouts": luffy_stats.get("difficulty_gate_79_min_onpolicy_rollouts", 1),
                                        "luffy79/tasks_skipped_teacher": luffy_stats.get("difficulty_gate_79_tasks_skipped_teacher", 0),
                                        # robustness / availability
                                        "luffy79/tasks_need_teacher": luffy_stats.get("difficulty_gate_79_tasks_need_teacher", 0),
                                        "luffy79/tasks_no_teacher_available": luffy_stats.get("difficulty_gate_79_tasks_no_teacher_available", 0),
                                        # per-task on-policy avg reward distribution (computed on tasks with valid rewards)
                                        "luffy79/onpolicy_avg_reward_count": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_count", 0),
                                        "luffy79/onpolicy_avg_reward_mean": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_mean", float("nan")),
                                        "luffy79/onpolicy_avg_reward_p50": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_p50", float("nan")),
                                        "luffy79/onpolicy_avg_reward_p90": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_p90", float("nan")),
                                        "luffy79/onpolicy_avg_reward_min": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_min", float("nan")),
                                        "luffy79/onpolicy_avg_reward_max": luffy_stats.get("difficulty_gate_79_onpolicy_avg_reward_max", float("nan")),
                                })

                                # ⭐ 7.10: step-based hard disable teacher rollouts after a given step (optional)
                                if isinstance(luffy_stats, dict) and luffy_stats.get("disable_teacher_710_enable", 0):
                                    metrics.update({
                                        "luffy710/after_step": luffy_stats.get("disable_teacher_710_after_step", 50),
                                        "luffy710/cur_step": luffy_stats.get("disable_teacher_710_cur_step", -1),
                                        "luffy710/disabled": luffy_stats.get("disable_teacher_710_disabled", 0),
                                    })
                                
                            elif enable_exp_replay or enable_teacher_exp:
                                # ⭐⭐⭐ ExGRPO 风格：Task 级别混合（向后兼容）⭐⭐⭐
                                # 更新 difficulty2task_dict（只用 on-policy 轨迹）
                                self.exp_manager.update_difficulty2task_dict(trajectories)
                                
                                # 合并 on-policy、self-generated off-policy、teacher off-policy 轨迹
                                all_trajectories = trajectories.copy()
                                
                                if offpolicy_cmt_array:
                                    all_trajectories.extend(offpolicy_cmt_array)
                                    logger.info(f"Added {len(offpolicy_cmt_array)} self-generated off-policy trajectories")
                                
                                if teacher_offpolicy_cmt_array:
                                    all_trajectories.extend(teacher_offpolicy_cmt_array)
                                    logger.info(f"Added {len(teacher_offpolicy_cmt_array)} teacher off-policy trajectories")
                                
                                if offpolicy_cmt_array or teacher_offpolicy_cmt_array:
                                    logger.info(
                                        f"Merged {len(trajectories)} on-policy + "
                                        f"{len(offpolicy_cmt_array)} self-off-policy + "
                                        f"{len(teacher_offpolicy_cmt_array)} teacher-off-policy = "
                                        f"{len(all_trajectories)} total trajectories"
                                    )
                            else:
                                all_trajectories = trajectories
                            
                            gen_batch_output = self.env_manager.to_dataproto(all_trajectories)
                            
                            # update metrics about experience manager
                            exp_mask_ratio = gen_batch_output.batch["exp_mask"].float().mean()
                            metrics.update({"exp_mask_ratio": exp_mask_ratio.detach().item()})
                            context_time_cost = [x.metadata["context_time_cost"] for x in trajectories if "context_time_cost" in x.metadata]
                            if context_time_cost:
                                metrics.update({
                                  "exp_manager/context_cost_avg":   np.mean(context_time_cost),
                                  "exp_manager/context_cost_max":   np.max(context_time_cost),
                                  "exp_manager/context_cost_min":   np.min(context_time_cost),
                                })

                            print(f"gen_batch_output.info batch.keys={gen_batch_output.batch.keys()}")
                            num_term_traj = sum([traj.is_terminated  for traj in trajectories])
                            num_not_none_traj = sum([len(traj.steps)>0  for traj in trajectories])

                            # gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)  # ⭐ Generate baseline sequences for advantage estimation

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor  # ⭐ Add reward baselines to the batch

                            del gen_baseline_batch, gen_baseline_output

                    # in the new code, the rollout process generates new extras, which should be merged with the original extra.
                    # by now, they are stored separately.
                    # assert len(gen_batch_output.non_tensor_batch["extras"].keys()&batch_extras.keys())==0, "extra of extra should not overlap with existing extra...how funny..."
                    batch.non_tensor_batch['original_extras']=batch_extras  # ⭐ Store original extras before scaling
                    batch = union_gen_batch_via_task_id(tasks, batch, gen_batch_output)  # ⭐ Merge generated batch with the current batch
                    
                    # ⭐ GRPO 分组关键：uid 必须基于 data_id（group_ids）来设置，而不是随机 UUID
                    # GRPO 使用 uid 来分组计算 advantage，同一 data_id 的轨迹应该在同一组
                    # 从 group_ids 获取 data_id，转换为字符串作为 uid
                    if "group_ids" in batch.batch:
                        # group_ids 是 tensor，shape: (batch_size,)
                        group_ids = batch.batch["group_ids"].cpu().numpy()
                        batch.non_tensor_batch["uid"] = np.array([str(int(gid)) for gid in group_ids], dtype=object)
                    else:
                        # 如果没有 group_ids，使用随机 UUID（向后兼容）
                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    batch.batch["response_mask"] = compute_response_mask(batch)  # ⭐ Compute and add response mask to the batch

                    # update experience pool
                    summary_task = self.exp_manager.submit_summary_task(trajectories, self.global_steps)


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)  # ⭐ Balance the batch to distribute valid tokens evenly

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()  # ⭐ Compute and store the global token numbers

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)  # ⭐ Compute reward scores using the reward model
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)  # ⭐ Compute rewards and extra information

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # ⭐ Compute old log probabilities
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        # ⭐ Endogenous replay: split entropy by on/off-policy (LLM tokens only, in multi-turn)
                        # This is critical to analyze entropy collapse / exploration suppression caused by replay.
                        try:
                            if "responses" in batch.batch and "exp_mask" in batch.batch and "loss_mask" in batch.batch:
                                resp_len = batch.batch["responses"].shape[-1]
                                exp_mask_full = batch.batch["exp_mask"]
                                exp_mask = exp_mask_full if exp_mask_full.shape[-1] == resp_len else exp_mask_full[:, -resp_len:]
                                loss_mask_full = batch.batch["loss_mask"]
                                loss_mask = loss_mask_full if loss_mask_full.shape[-1] == resp_len else loss_mask_full[:, -resp_len:]
                                llm_mask = loss_mask * response_masks
                                denom_all = llm_mask.sum().clamp_min(1.0)
                                denom_off = (exp_mask * llm_mask).sum().clamp_min(1.0)
                                denom_on = ((1.0 - exp_mask) * llm_mask).sum().clamp_min(1.0)

                                # entropys is (bs, resp_len)
                                ent_all = (entropys * llm_mask).sum() / denom_all
                                ent_off = (entropys * exp_mask * llm_mask).sum() / denom_off
                                ent_on = (entropys * (1.0 - exp_mask) * llm_mask).sum() / denom_on

                                metrics.update(
                                    {
                                        "exp_replay/entropy_llm_mean": ent_all.detach().float().item(),
                                        "exp_replay/entropy_llm_onpolicy_mean": ent_on.detach().float().item(),
                                        "exp_replay/entropy_llm_offpolicy_mean": ent_off.detach().float().item(),
                                    }
                                )
                        except Exception:
                            pass
                        
                        # ⭐ Experience Replay: 替换 off-policy 数据的 old_log_prob
                        if enable_exp_replay and "recorded_old_log_probs" in batch.batch:
                            exp_is_correct = exp_replay_config.get("exp_is_correct", True)
                            if exp_is_correct:
                                batch = self._replace_recorded_old_log_probs(
                                    batch=batch,
                                    current_old_log_prob=old_log_prob,
                                    entropys=entropys,
                                )
                                logger.info("Replaced off-policy old_log_probs with recorded ones")
                        
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
                        # ⭐ Experience Replay: 保存成功轨迹到内存
                        if enable_exp_replay:
                            n_rollout = self.config.actor_rollout_ref.rollout.n
                            # 只处理 on-policy 轨迹（前 len(trajectories) 个）
                            num_on_policy = len(trajectories)
                            on_policy_entropys = entropys[:num_on_policy] if entropys is not None else None
                            on_policy_response_mask = response_masks[:num_on_policy]
                            
                            # 更新 skip_uid_set 并筛选轨迹
                            filtered_trajectories = self.exp_manager.update_skip_uid_set_and_filter_trajectories(
                                trajectories=trajectories,
                                n_rollout=n_rollout,
                                entropys=on_policy_entropys,
                                response_mask=on_policy_response_mask,
                            )
                            
                            # 将 old_log_probs 保存到轨迹 metadata
                            # ⭐ Multi-turn 关键：保存完整的 response 部分的 old_log_probs（不过滤）
                            # 这样在加载时可以正确对齐，避免位置错位
                            old_log_probs_tensor = old_log_prob.batch["old_log_probs"]
                            for idx, traj in enumerate(trajectories):
                                if idx < old_log_probs_tensor.shape[0]:
                                    traj_old_log_prob = old_log_probs_tensor[idx].cpu().numpy()
                                    traj_response_mask = response_masks[idx].cpu().numpy()
                                    # ⭐ 保存完整的 old_log_probs 和 response_mask
                                    # 不过滤，保持位置对齐
                                    traj.metadata["old_log_probs"] = traj_old_log_prob.tolist()
                                    traj.metadata["response_mask"] = traj_response_mask.tolist()  # 保存 mask 用于后续对齐
                                    traj.metadata["policy_version"] = self.global_steps
                            
                            # 保存筛选后的轨迹
                            if filtered_trajectories:
                                self.exp_manager.save_trajectories_to_memory(filtered_trajectories)
                                logger.info(
                                    f"Saved {len(filtered_trajectories)} trajectories to memory "
                                    f"(skip_uid_set size: {len(self.exp_manager.skip_uid_set)})"
                                )
                        
                        # ⭐ RePO: 保存所有 on-policy 轨迹到 RePO buffer（不只是成功的）
                        if self.exp_manager.repo_enabled:
                            repo_saved = self.exp_manager.repo_save_all_trajectories(
                                trajectories=trajectories,
                                global_step=self.global_steps,
                            )
                            logger.info(f"[RePO] Saved {repo_saved} trajectories to buffer at step {self.global_steps}")
                            
                            # 添加 experience replay 相关指标
                            metrics.update({
                                "exp_replay/skip_uid_set_size": len(self.exp_manager.skip_uid_set),
                                "exp_replay/total_tasks_in_pool": len(self.exp_manager.get_valid_replay_task_ids()),
                                "exp_replay/num_experience_tasks": len(experience_tasks),
                                "exp_replay/num_offpolicy_trajectories": len(offpolicy_cmt_array),
                                # ⭐ 更直观的 replay 强度（rollout 级别）
                                "exp_replay/offpolicy_rollout_ratio": (
                                    float(len(offpolicy_cmt_array)) / float(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n)
                                    if (self.config.data.train_batch_size and self.config.actor_rollout_ref.rollout.n)
                                    else 0.0
                                ),
                                # ⭐ 新增：Teacher Experience Metrics
                                "teacher_exp/num_teacher_tasks": len(teacher_exp_tasks),
                                "teacher_exp/num_teacher_trajectories": len(teacher_offpolicy_cmt_array),
                                "teacher_exp/total_teacher_pool_size": len(self.exp_manager.get_valid_teacher_task_ids()) if self.exp_manager.teacher_enabled else 0,
                            })

                            # ⭐ Endogenous replay：pool difficulty 分布（便于分析 replay 主要集中在哪些难度）
                            try:
                                n_rollout = int(self.config.actor_rollout_ref.rollout.n)
                                diff_dict = getattr(self.exp_manager, "difficulty2task_dict", {}) or {}
                                total_cnt = 0
                                weighted_sum = 0
                                for d in range(n_rollout + 1):
                                    c = len(diff_dict.get(d, []) or [])
                                    total_cnt += c
                                    weighted_sum += d * c
                                    metrics[f"exp_replay/pool_count/d{d}"] = c
                                metrics["exp_replay/pool_count_total"] = total_cnt
                                metrics["exp_replay/pool_difficulty_mean"] = (weighted_sum / total_cnt) if total_cnt > 0 else 0.0
                            except Exception:
                                pass

                            # ⭐ Endogenous replay：token-level off-policy 占比（对齐 response_mask/loss_mask，避免 exp_mask_ratio 受 padding 干扰）
                            try:
                                if "responses" in batch.batch and "response_mask" in batch.batch and "exp_mask" in batch.batch:
                                    resp_len = batch.batch["responses"].shape[-1]
                                    response_mask = batch.batch["response_mask"]
                                    exp_mask_full = batch.batch["exp_mask"]
                                    exp_mask = exp_mask_full if exp_mask_full.shape[-1] == resp_len else exp_mask_full[:, -resp_len:]
                                    resp_total = response_mask.sum().clamp_min(1.0)
                                    off_resp = (exp_mask * response_mask).sum()
                                    metrics["exp_replay/offpolicy_token_ratio_response"] = (off_resp / resp_total).detach().float().item()

                                    loss_mask_full = batch.batch.get("loss_mask", None)
                                    if loss_mask_full is not None and loss_mask_full.shape[-1] >= resp_len:
                                        loss_mask = loss_mask_full[:, -resp_len:]
                                        llm_mask = loss_mask * response_mask
                                        llm_total = llm_mask.sum().clamp_min(1.0)
                                        off_llm = (exp_mask * llm_mask).sum()
                                        metrics["exp_replay/offpolicy_token_ratio_llm"] = (off_llm / llm_total).detach().float().item()
                            except Exception:
                                pass

                            # ⭐ Endogenous replay：经验陈旧度（age = global_step - policy_version）
                            try:
                                extras = batch.non_tensor_batch.get("extras", None)
                                if extras is not None:
                                    ages = []
                                    for ex in extras:
                                        if not isinstance(ex, dict):
                                            continue
                                        if not ex.get("is_experience_replay", False):
                                            continue
                                        pv = ex.get("policy_version", None)
                                        if pv is None:
                                            continue
                                        try:
                                            ages.append(float(self.global_steps - int(pv)))
                                        except Exception:
                                            continue
                                    if ages:
                                        metrics["exp_replay/offpolicy_age_mean"] = float(np.mean(ages))
                                        metrics["exp_replay/offpolicy_age_max"] = float(np.max(ages))
                                        metrics["exp_replay/offpolicy_age_min"] = float(np.min(ages))
                                        metrics["exp_replay/offpolicy_age_count"] = float(len(ages))
                            except Exception:
                                pass

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # ⭐ Compute reference log probabilities
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)  # ⭐ Compute values using the critic model
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)  # ⭐ Get the reward tensor and extra info from the async call
                        batch.batch["token_level_scores"] = reward_tensor

                        # ============================================================================
                        # 🔍 DEBUG: Check ORIGINAL reward tensor BEFORE overlong_reward_shaping
                        # ============================================================================
                        _orig_reward_sums = reward_tensor.sum(dim=-1)
                        if _orig_reward_sums.min().item() < -10:  # Threshold for abnormal values
                            logger.warning(
                                f"🚨 ABNORMAL REWARD DETECTED BEFORE overlong_reward_shaping!\n"
                                f"  reward_sums: min={_orig_reward_sums.min().item():.2f}, max={_orig_reward_sums.max().item():.2f}, mean={_orig_reward_sums.mean().item():.2f}\n"
                                f"  reward_tensor shape: {reward_tensor.shape}\n"
                                f"  Non-zero positions per sample (first 5):"
                            )
                            for i in range(min(5, reward_tensor.shape[0])):
                                non_zero_count = (reward_tensor[i] != 0).sum().item()
                                logger.warning(f"    Sample {i}: non-zero positions={non_zero_count}, sum={_orig_reward_sums[i].item():.2f}")

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        # Extra debug: if reward manager provides overlong-related fields, summarize them.
                        if reward_extra_infos_dict:
                            try:
                                if "overlong_reward" in reward_extra_infos_dict:
                                    _ov = torch.tensor(reward_extra_infos_dict["overlong_reward"], dtype=torch.float32)
                                    metrics.update({
                                        "reward/overlong_reward/mean": _ov.mean().item(),
                                        "reward/overlong_reward/max": _ov.max().item(),
                                        "reward/overlong_reward/min": _ov.min().item(),
                                    })
                                if "overlong" in reward_extra_infos_dict:
                                    _ovb = torch.tensor(reward_extra_infos_dict["overlong"], dtype=torch.float32)
                                    metrics.update({
                                        "reward/overlong/ratio": _ovb.mean().item(),
                                    })
                            except Exception as _e:
                                logger.warning(f"Failed to summarize overlong reward extra info: {_e}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # ============================================================================
                        # ⭐ DAPO Overlong Reward Shaping: Apply soft penalty to truncated samples
                        # ============================================================================
                        dapo_config = self.config.algorithm.get("dapo", {})
                        use_dapo = dapo_config.get("enable", False)
                        
                        if use_dapo and dapo_config.get("overlong_reward_shaping", {}).get("enable", False):
                            from agentevolver.module.exp_manager.het_core_algos import dapo_overlong_reward_shaping
                            
                            overlong_config = dapo_config.get("overlong_reward_shaping", {})
                            truncation_penalty = overlong_config.get("truncation_penalty", -0.5)
                            soft_penalty_mode = overlong_config.get("soft_penalty_mode", "additive")
                            
                            # ============================================================================
                            # Multi-turn aware truncation detection
                            # ============================================================================
                            # In multi-turn scenarios, we need to check:
                            # 1. Response length hitting max_response_length
                            # 2. Trajectory is_terminated flag (if available)
                            # 3. For multi-turn: max_steps reached without completion
                            # ============================================================================
                            
                            responses = batch.batch["responses"]
                            attention_mask = batch.batch["attention_mask"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]
                            
                            # Method 1: Check if response reached max length
                            max_response_length = self.config.data.max_response_length
                            actual_response_lengths = response_mask.sum(dim=-1)
                            is_truncated_by_length = (actual_response_lengths >= max_response_length - 1)
                            
                            # Method 2: Check trajectory is_terminated flag (for multi-turn)
                            # If trajectory is not terminated, it was likely truncated
                            is_truncated_by_termination = torch.zeros_like(is_truncated_by_length)
                            if self.async_rollout_mode and trajectories:
                                # Only check on-policy trajectories
                                num_on_policy = len(trajectories)
                                for idx, traj in enumerate(trajectories):
                                    if idx < len(is_truncated_by_termination):
                                        # Not terminated means truncated in multi-turn
                                        if not getattr(traj, 'is_terminated', True):
                                            is_truncated_by_termination[idx] = True
                            
                            # Combine both methods: truncated if either condition is met
                            is_truncated = is_truncated_by_length | is_truncated_by_termination.to(is_truncated_by_length.device)
                            
                            # Apply overlong reward shaping
                            original_reward_tensor = reward_tensor.clone()
                            reward_tensor = dapo_overlong_reward_shaping(
                                rewards=reward_tensor,
                                is_truncated=is_truncated,
                                truncation_penalty=truncation_penalty,
                                soft_penalty_mode=soft_penalty_mode,
                                response_mask=response_mask,
                            )
                            batch.batch["token_level_scores"] = reward_tensor
                            
                            # ============================================================================
                            # 🔍 DEBUG: Check reward tensor values after overlong_reward_shaping
                            # ============================================================================
                            reward_sums = reward_tensor.sum(dim=-1)
                            original_reward_sums = original_reward_tensor.sum(dim=-1)
                            
                            # Check for abnormally negative rewards
                            if reward_sums.min().item() < -10:  # Threshold for abnormal values
                                logger.warning(
                                    f"🚨 ABNORMAL REWARD DETECTED after overlong_reward_shaping!\n"
                                    f"  reward_sums: min={reward_sums.min().item():.2f}, max={reward_sums.max().item():.2f}, mean={reward_sums.mean().item():.2f}\n"
                                    f"  original_reward_sums: min={original_reward_sums.min().item():.2f}, max={original_reward_sums.max().item():.2f}, mean={original_reward_sums.mean().item():.2f}\n"
                                    f"  is_truncated count: {is_truncated.sum().item()}/{len(is_truncated)}\n"
                                    f"  truncation_penalty: {truncation_penalty}, soft_penalty_mode: {soft_penalty_mode}"
                                )
                                # Check non-zero count per sample
                                for i in range(min(5, reward_tensor.shape[0])):
                                    non_zero_count = (reward_tensor[i] != 0).sum().item()
                                    if non_zero_count > 1:
                                        logger.warning(
                                            f"  Sample {i}: non-zero positions={non_zero_count}, sum={reward_sums[i].item():.2f}, "
                                            f"is_truncated={is_truncated[i].item()}"
                                        )
                            
                            # Log metrics with multi-turn aware information
                            num_truncated = is_truncated.sum().item()
                            num_truncated_by_length = is_truncated_by_length.sum().item()
                            num_truncated_by_termination = is_truncated_by_termination.sum().item()
                            metrics.update({
                                "dapo/num_truncated_samples": num_truncated,
                                "dapo/truncation_ratio": num_truncated / len(is_truncated),
                                "dapo/num_truncated_by_length": num_truncated_by_length,
                                "dapo/num_truncated_by_termination": num_truncated_by_termination,
                                # 🔍 DEBUG: Add reward statistics (after DAPO overlong penalty)
                                "dapo/reward_sum_min": reward_sums.min().item(),
                                "dapo/reward_sum_max": reward_sums.max().item(),
                                "dapo/reward_sum_mean": reward_sums.mean().item(),
                                # ⭐ Record original environment reward (before DAPO overlong penalty)
                                "env_reward/original_reward_min": original_reward_sums.min().item(),
                                "env_reward/original_reward_max": original_reward_sums.max().item(),
                                "env_reward/original_reward_mean": original_reward_sums.mean().item(),
                                "env_reward/original_reward_std": original_reward_sums.std().item(),
                            })
                            if num_truncated > 0:
                                reward_diff = (original_reward_tensor.sum(dim=-1) - reward_tensor.sum(dim=-1))[is_truncated].mean().item()
                                metrics.update({
                                    "dapo/avg_truncation_penalty_applied": reward_diff,
                                })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)  # ⭐ Apply KL divergence penalty
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        
                        # ============================================================================
                        # 🔍 DEBUG: Detailed reward tensor analysis to diagnose negative rewards
                        # ============================================================================
                        _tlr = batch.batch["token_level_rewards"]
                        _tlr_sums = _tlr.sum(dim=-1)
                        _nonzero_counts = (_tlr != 0).sum(dim=-1)
                        
                        logger.info(
                            f"📊 REWARD TENSOR DEBUG:\n"
                            f"  Shape: {_tlr.shape}\n"
                            f"  Sum stats: min={_tlr_sums.min().item():.4f}, max={_tlr_sums.max().item():.4f}, mean={_tlr_sums.mean().item():.4f}\n"
                            f"  Non-zero counts: min={_nonzero_counts.min().item()}, max={_nonzero_counts.max().item()}, mean={_nonzero_counts.float().mean().item():.2f}"
                        )
                        
                        # Check for samples with multiple non-zero positions (should be 0 or 1)
                        _multi_nonzero_mask = _nonzero_counts > 1
                        if _multi_nonzero_mask.any():
                            logger.warning(
                                f"⚠️ MULTIPLE NON-ZERO REWARD POSITIONS DETECTED!\n"
                                f"  {_multi_nonzero_mask.sum().item()} samples have >1 non-zero reward positions\n"
                                f"  This could cause sum to be unexpectedly large!"
                            )
                            # Print details for first few problematic samples
                            for idx in torch.where(_multi_nonzero_mask)[0][:5]:
                                _positions = (_tlr[idx] != 0).nonzero(as_tuple=True)[0]
                                _values = _tlr[idx, _positions]
                                logger.warning(
                                    f"    Sample {idx.item()}: {_nonzero_counts[idx].item()} non-zero positions at {_positions.tolist()}, "
                                    f"values={_values.tolist()}, sum={_tlr_sums[idx].item():.4f}"
                                )

                        # ============================================================================
                        # ⭐ DAPO Dynamic Sampling: Filter samples with all-correct or all-incorrect outcomes
                        # ============================================================================
                        if use_dapo and dapo_config.get("dynamic_sampling", {}).get("enable", False):
                            from agentevolver.module.exp_manager.het_core_algos import dapo_filter_samples
                            
                            dynamic_sampling_config = dapo_config.get("dynamic_sampling", {})
                            filter_mode = dynamic_sampling_config.get("filter_mode", "strict")
                            
                            # Get group IDs for filtering
                            if "uid" in batch.non_tensor_batch:
                                group_ids = batch.non_tensor_batch["uid"]
                            elif "group_ids" in batch.batch:
                                group_ids = batch.batch["group_ids"].cpu().numpy()
                            else:
                                group_ids = np.array([str(i) for i in range(len(batch))])
                            
                            # Get rewards for filtering
                            filter_rewards = batch.batch["token_level_rewards"].sum(dim=-1)
                            
                            # Apply dynamic sampling filter
                            keep_mask = dapo_filter_samples(
                                rewards=filter_rewards,
                                group_ids=group_ids,
                                n_rollout=self.config.actor_rollout_ref.rollout.n,
                                filter_mode=filter_mode,
                            )
                            
                            num_filtered = (~keep_mask).sum().item()
                            if num_filtered > 0:
                                logger.info(f"DAPO Dynamic Sampling: Filtered {num_filtered} samples with {filter_mode} mode")
                                
                                # Instead of removing samples (which would break GRPO grouping),
                                # we zero out the advantages for filtered samples after advantage computation
                                batch.batch["dapo_keep_mask"] = keep_mask.float()
                                
                            metrics.update({
                                "dapo/num_filtered_samples": num_filtered,
                                "dapo/filter_ratio": num_filtered / len(keep_mask),
                            })

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        if os.environ.get("DEBUG_ARG","").find("disable_adv_std")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change norm_adv_by_std_in_grpo from True to False, using batch std!")
                            norm_adv_by_std_in_grpo = False

                        # call the original compute_advantage for compatibility
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                        # ⭐ Teacher effect diagnostics: batch-level logging + save per-rollout analysis
                        try:
                            diag_metrics = compute_teacher_effect_metrics(batch=batch, entropys=entropys)
                            # prefix already in keys
                            metrics.update(diag_metrics)
                        except Exception as _e:
                            logger.warning(f"Failed to compute teacher effect metrics: {_e}")
                            diag_metrics = None

                        # ============================================================================
                        # ⭐ 2.2 Improvement (optional): adaptive teacher loss gating/annealing
                        # ============================================================================
                        # This writes `teacher_loss_scale` into batch.batch so actor can scale teacher loss.
                        # Safety:
                        # - Only active if exp_manager.teacher_experience.adaptive_weight.enable=true
                        # - Only applies when teacher_mask exists and has teacher tokens
                        # - Does NOT affect ExGRPO self-generated experience replay (teacher_mask=0)
                        try:
                            te_cfg = getattr(getattr(self.config, "exp_manager", None), "teacher_experience", None)
                            aw_cfg = getattr(te_cfg, "adaptive_weight", None) if te_cfg is not None else None
                            aw_enable = bool(getattr(aw_cfg, "enable", False)) if aw_cfg is not None else False
                        except Exception:
                            aw_cfg, aw_enable = None, False

                        if aw_enable and diag_metrics and isinstance(diag_metrics, dict):
                            gap = diag_metrics.get("diag/group_teacher_minus_on_reward_mean", None)
                            teacher_mask_full = batch.batch.get("teacher_mask", None)
                            has_teacher = (
                                teacher_mask_full is not None
                                and torch.is_tensor(teacher_mask_full)
                                and teacher_mask_full.numel() > 0
                                and teacher_mask_full.sum().item() > 0
                            )
                            if gap is not None and has_teacher:
                                mode = str(getattr(aw_cfg, "mode", "gap_linear"))
                                gate_level = str(getattr(aw_cfg, "gate_level", "batch")).lower().strip()
                                eps = float(getattr(aw_cfg, "epsilon", 0.0))
                                tau = float(getattr(aw_cfg, "tau", 1.0))
                                a_min = float(getattr(aw_cfg, "min", 0.0))
                                a_max = float(getattr(aw_cfg, "max", 1.0))

                                # optional EMA on gap
                                ema_cfg = getattr(aw_cfg, "ema", None)
                                ema_enable = bool(getattr(ema_cfg, "enable", False)) if ema_cfg is not None else False
                                ema_beta = float(getattr(ema_cfg, "beta", 0.9)) if ema_cfg is not None else 0.9
                                denom = tau if tau > 0 else 1.0

                                # fill per-token scale aligned to actor's response_length
                                resp_len = batch.batch["responses"].shape[-1]
                                bs = batch.batch["responses"].shape[0]
                                device = batch.batch["responses"].device

                                # Use response mask aligned rewards for group-level gap if needed
                                response_mask_local = batch.batch.get("response_mask", None)
                                if response_mask_local is None:
                                    response_mask_local = compute_response_mask(batch)
                                    batch.batch["response_mask"] = response_mask_local
                                if (
                                    response_mask_local.dim() == 2
                                    and response_mask_local.shape[-1] != resp_len
                                    and response_mask_local.shape[-1] > resp_len
                                ):
                                    response_mask_local = response_mask_local[:, -resp_len:]

                                token_level_rewards_local = batch.batch.get("token_level_rewards", None)
                                if token_level_rewards_local is None:
                                    token_level_rewards_local = batch.batch.get("token_level_scores", None)
                                rewards_sum_local = None
                                if token_level_rewards_local is not None and torch.is_tensor(token_level_rewards_local) and token_level_rewards_local.dim() == 2:
                                    rewards_sum_local = (token_level_rewards_local[:, :resp_len] * response_mask_local).sum(dim=-1)  # (bs,)

                                # Derive is_teacher_sample from teacher_mask within response span
                                teacher_mask_local = teacher_mask_full
                                if teacher_mask_local is not None and teacher_mask_local.dim() == 2 and teacher_mask_local.shape[-1] != resp_len and teacher_mask_local.shape[-1] > resp_len:
                                    teacher_mask_local = teacher_mask_local[:, -resp_len:]
                                is_teacher_sample_local = (teacher_mask_local.sum(dim=-1) > 0) if teacher_mask_local is not None else None

                                def _alpha_from_gap(g_eff: float) -> float:
                                    a = (g_eff - eps) / denom
                                    a = max(a_min, min(a_max, a))
                                    return float(a)

                                if gate_level == "group" and rewards_sum_local is not None and is_teacher_sample_local is not None:
                                    # v3: per-task(per-group) gate.
                                    # Group by uid if present; fallback to task_ids.
                                    group_ids = None
                                    try:
                                        group_ids = batch.non_tensor_batch.get("uid", None)
                                    except Exception:
                                        group_ids = None
                                    if group_ids is None:
                                        try:
                                            group_ids = batch.non_tensor_batch.get("task_ids", None)
                                        except Exception:
                                            group_ids = None

                                    if group_ids is None or len(group_ids) != bs:
                                        # fallback to batch-level if grouping not available
                                        gate_level = "batch"
                                    else:
                                        # build group -> indices mapping (string key for safety)
                                        id2idx = defaultdict(list)
                                        for i, gid in enumerate(group_ids):
                                            id2idx[str(gid)].append(i)

                                        # compute per-group gap and alpha
                                        alpha_per_sample = torch.empty((bs,), device=device, dtype=torch.float32)
                                        gap_used_list: List[float] = []
                                        alpha_list: List[float] = []

                                        # Optional per-group EMA (stateful), isolated to this trainer instance.
                                        ema_by_group = getattr(self, "_teacher_gap_ema_by_group", None)
                                        if ema_enable and ema_by_group is None:
                                            ema_by_group = {}
                                            setattr(self, "_teacher_gap_ema_by_group", ema_by_group)

                                        for gid, idxs in id2idx.items():
                                            idx_t = [i for i in idxs if bool(is_teacher_sample_local[i].item())]
                                            idx_o = [i for i in idxs if not bool(is_teacher_sample_local[i].item())]
                                            if len(idx_t) == 0 or len(idx_o) == 0:
                                                # if missing teacher or on-policy samples in group, fall back to batch gap
                                                g_raw = float(gap)
                                            else:
                                                rt = float(rewards_sum_local[idx_t].mean().item())
                                                ro = float(rewards_sum_local[idx_o].mean().item())
                                                g_raw = rt - ro

                                            if ema_enable:
                                                prev = ema_by_group.get(gid, None)
                                                if prev is None:
                                                    ema_by_group[gid] = g_raw
                                                else:
                                                    ema_by_group[gid] = ema_beta * float(prev) + (1.0 - ema_beta) * g_raw
                                                g_eff = float(ema_by_group[gid])
                                            else:
                                                g_eff = float(g_raw)

                                            a = _alpha_from_gap(g_eff)
                                            for i in idxs:
                                                alpha_per_sample[i] = a
                                            gap_used_list.append(g_eff)
                                            alpha_list.append(a)

                                        batch.batch["teacher_loss_scale"] = alpha_per_sample.unsqueeze(-1).expand(bs, resp_len).contiguous()

                                        # logging: keep original keys as MEAN to stay compatible with existing analysis scripts
                                        if len(alpha_list) > 0:
                                            metrics["diag/teacher_loss_scale"] = float(np.mean(alpha_list))
                                            metrics["diag/teacher_loss_scale_min"] = float(np.min(alpha_list))
                                            metrics["diag/teacher_loss_scale_max"] = float(np.max(alpha_list))
                                        if len(gap_used_list) > 0:
                                            metrics["diag/teacher_gap_used"] = float(np.mean(gap_used_list))
                                            metrics["diag/teacher_gap_used_p90"] = float(np.quantile(np.array(gap_used_list), 0.90))
                                            metrics["diag/teacher_gap_used_max"] = float(np.max(gap_used_list))
                                        metrics["diag/teacher_gate_level"] = 1.0  # 1=group, 0=batch (for quick filtering)
                                        # Done for group gate
                                        gate_level = "group"

                                if gate_level != "group":
                                    # batch-level gate (backward compatible)
                                    g = float(gap)
                                    if ema_enable:
                                        if getattr(self, "_teacher_gap_ema", None) is None:
                                            self._teacher_gap_ema = g
                                        else:
                                            self._teacher_gap_ema = ema_beta * float(self._teacher_gap_ema) + (1.0 - ema_beta) * g
                                        g_eff = float(self._teacher_gap_ema)
                                    else:
                                        g_eff = g

                                    alpha = _alpha_from_gap(g_eff)
                                    batch.batch["teacher_loss_scale"] = torch.full(
                                        (bs, resp_len), float(alpha), device=device, dtype=torch.float32
                                    )
                                    metrics["diag/teacher_loss_scale"] = float(alpha)
                                    metrics["diag/teacher_gap_used"] = float(g_eff)
                                    metrics["diag/teacher_gate_level"] = 0.0  # 1=group, 0=batch
                                    if ema_enable:
                                        metrics["diag/teacher_gap_ema"] = float(self._teacher_gap_ema) if self._teacher_gap_ema is not None else float("nan")

                        # ⭐ 保存 Trajectory 信息用于后续分析（放到 compute_advantage 后，包含 advantages）
                        if self.async_rollout_mode and trajectories:
                            trajectory_save_dir = self.config.trainer.get("trajectory_save_dir", None)
                            if trajectory_save_dir is None:
                                trajectory_save_dir = self.config.trainer.get("default_local_dir", "checkpoints")
                            if trajectory_save_dir:
                                try:
                                    self._save_trajectories_for_analysis(
                                        trajectories=trajectories,
                                        tasks=tasks,
                                        entropys=entropys,
                                        response_masks=response_masks,
                                        global_steps=self.global_steps,
                                        output_dir=trajectory_save_dir,
                                        # batch identifiers (aligned)
                                        batch_task_ids=batch.non_tensor_batch.get("task_ids", None),
                                        batch_rollout_ids=batch.non_tensor_batch.get("rollout_ids", None),
                                        batch_messages=batch.non_tensor_batch.get("messages", None),
                                        batch_group_ids=batch.batch.get("group_ids", None),
                                        batch_reward_scores=batch.non_tensor_batch.get("reward_scores", None),
                                        # batch tensors for deeper analysis
                                        batch_advantages=batch.batch.get("advantages", None),
                                        batch_exp_mask=batch.batch.get("exp_mask", None),
                                        batch_teacher_mask=batch.batch.get("teacher_mask", None),
                                        batch_old_log_probs=batch.batch.get("old_log_probs", None),
                                        batch_token_level_rewards=batch.batch.get("token_level_rewards", None),
                                        batch_uid=batch.non_tensor_batch.get("uid", None),
                                        batch_diag_metrics=diag_metrics,
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to save trajectories for analysis: {e}")
                        
                        # ============================================================================
                        # ⭐ DAPO Dynamic Sampling: Zero out advantages for filtered samples
                        # ============================================================================
                        if "dapo_keep_mask" in batch.batch:
                            dapo_keep_mask = batch.batch["dapo_keep_mask"]
                            # Expand mask to match advantage shape (batch_size, seq_len)
                            if dapo_keep_mask.dim() == 1:
                                dapo_keep_mask = dapo_keep_mask.unsqueeze(-1)
                            # Zero out advantages for filtered samples
                            batch.batch["advantages"] = batch.batch["advantages"] * dapo_keep_mask
                            logger.info(f"DAPO: Applied keep_mask to zero out {(dapo_keep_mask == 0).sum().item()} sample advantages")
                        
                        # shuchang
                        # ==================== Begin ADCA GRPO  ====================
                        attribution_cfg = self._get_attribution_config()
                        if getattr(attribution_cfg, 'enable', False):
                            batch, adca_metrics = apply_adca_grpo(
                                batch=batch,
                                attribution_cfg=attribution_cfg,
                                tokenizer=self.tokenizer,
                                global_steps=self.global_steps,
                                epoch=epoch,
                                i=i,
                            )
                            metrics.update(adca_metrics)
                        # ==================== End ADCA GRPO ====================
                        # Apply decay factor of 0.5 to non_tensor_batch['extras'][i]['evaluator'] != 'env'
                        if os.environ.get("DEBUG_ARG","").find("synth_decay")!=-1:
                            if epoch==0 and i==0:
                                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                            assert 'extras' in batch.non_tensor_batch
                            if 'extras' in batch.non_tensor_batch:
                                for i in range(len(batch.non_tensor_batch['extras'])):
                                    assert 'evaluator' in batch.non_tensor_batch['extras'][i]
                                    evaluator = batch.non_tensor_batch['extras'][i]['evaluator']
                                    if evaluator != 'env':
                                        batch.batch["advantages"][i] *= 0.5  # ⭐ Apply decay factor to synthetic data

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)  # ⭐ Update the critic model
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # ⭐ CHORD: 传递 global_step 用于 μ 三阶段调度（Warmup → Decay → 稳定）
                            batch.meta_info["global_step"] = self.global_steps
                            actor_output = self.actor_rollout_wg.update_actor(batch)  # ⭐ Update the actor with the new batch
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # ------------------------------------------------------------------
                        # ⭐ Gradient-direction diagnostics (teacher vs on-policy)
                        # Ensure it is visible both in WandB (via metrics) and in local analysis logs.
                        # - WandB: already handled by tracker.log(data=metrics)
                        # - Local: write/merge into Trajectory/batch_diag_step_{step}.json
                        #          and also dump a small grad_dir_{step}.json under rollout_log.
                        # ------------------------------------------------------------------
                        try:
                            grad_dir_metrics = {k: v for k, v in metrics.items() if isinstance(k, str) and k.startswith("grad_dir/")}
                            # Surrogate distortion diagnostics (clip hit / clipfrac)
                            clip_metrics = {
                                k: v
                                for k, v in metrics.items()
                                if isinstance(k, str)
                                and (
                                    k.startswith("actor/on_pg_clip")
                                    or k.startswith("actor/off_pg_clip")
                                    or k.startswith("actor/self_off_pg_clip")
                                    or k.startswith("actor/teacher_off_pg_clip")
                                )
                            }
                            local_metrics = {}
                            local_metrics.update(grad_dir_metrics)
                            local_metrics.update(clip_metrics)

                            if local_metrics:
                                # 1) trajectory analysis summary (overwrite/merge)
                                trajectory_save_dir = self.config.trainer.get("trajectory_save_dir", None)
                                if trajectory_save_dir is None:
                                    trajectory_save_dir = self.config.trainer.get("default_local_dir", "checkpoints")
                                if trajectory_save_dir:
                                    trajectory_dir = os.path.join(str(trajectory_save_dir), "Trajectory")
                                    os.makedirs(trajectory_dir, exist_ok=True)
                                    summary_path = os.path.join(trajectory_dir, f"batch_diag_step_{self.global_steps}.json")
                                    merged = {}
                                    if os.path.exists(summary_path):
                                        try:
                                            with open(summary_path, "r", encoding="utf-8") as f:
                                                merged = json.load(f) or {}
                                        except Exception:
                                            merged = {}
                                    if not isinstance(merged, dict):
                                        merged = {}
                                    merged.update(local_metrics)
                                    with open(summary_path, "w", encoding="utf-8") as f:
                                        json.dump(merged, f, ensure_ascii=False, indent=2)

                                # 2) rollout_log dumps (easy to find alongside traj_{step}.jsonl)
                                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                                if rollout_data_dir:
                                    os.makedirs(rollout_data_dir, exist_ok=True)
                                    if grad_dir_metrics:
                                        gd_path = os.path.join(str(rollout_data_dir), f"grad_dir_{self.global_steps}.json")
                                        with open(gd_path, "w", encoding="utf-8") as f:
                                            json.dump(grad_dir_metrics, f, ensure_ascii=False, indent=2)
                                    if clip_metrics:
                                        cm_path = os.path.join(str(rollout_data_dir), f"clip_metrics_{self.global_steps}.json")
                                        with open(cm_path, "w", encoding="utf-8") as f:
                                            json.dump(clip_metrics, f, ensure_ascii=False, indent=2)
                        except Exception:
                            # must never break training
                            pass
                    
                    # collect summary tasks
                    if summary_task is not None:
                        time_cost = self.exp_manager.collect_summary_result(summary_task)
                        metrics.update({"exp_manager/summary": time_cost})


                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )  # ⭐ Dump the generated experiences and trajectories

                            # save original trajectory
                            filename = os.path.join(rollout_data_dir, f"traj_{self.global_steps}.jsonl")
                            with open(filename, "w") as f:
                                for traj in trajectories:
                                    f.write(traj.json() + "\n")
                            # save tasks
                            filename = os.path.join(rollout_data_dir, f"task_{self.global_steps}.jsonl")
                            with open(filename,"w") as f:
                                for task in tasks: # this must be bounded # type: ignore
                                    f.write(task.json() + "\n")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()  # ⭐ Validate the model and collect validation metrics
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()  # ⭐ Save the current state of the model as a checkpoint
                            
                            # ⭐ Experience Replay: 保存 experience pool 到磁盘
                            if enable_exp_replay:
                                exp_pool_save_dir = os.path.join(
                                    self.config.trainer.default_local_dir,
                                    "experience_pool",
                                    f"step_{self.global_steps}"
                                )
                                self.exp_manager.save_experience_pool_to_disk(exp_pool_save_dir)
                                logger.info(f"Saved experience pool to {exp_pool_save_dir}")

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/num_not_none_traj": num_not_none_traj,
                        "training/num_term_traj": num_term_traj
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                tracker.log(data=metrics, step=self.global_steps)  # ⭐ Log the collected metrics

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            # we expect the train dataset is fully explored at the beginning, no reload needed.
            # if isinstance(self.train_dataset, FullDataset):
            #     self.train_dataset.reload()
            if os.environ.get("DEBUG_ARG",'').find("ratio_decay")!=-1:
                from agentevolver.module.task_manager.data_mixture import UnifiedMixtureStrategy
                print("DEBUG: change ratio of synthetic data from 1 to 0.5")
                assert isinstance(self.train_dataset._mixture_strategy,UnifiedMixtureStrategy)
                self.train_dataset._mixture_strategy._synthetic_ratio-=1/5 # initial 1, 0 at about epoch 5 (about step 30)
            self.train_dataset.update()  # ⭐ Update the training dataset for the next iteration


