from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch


def _masked_mean_abs_diff(current_lp: torch.Tensor, recorded_lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (current_lp - recorded_lp).abs() * mask
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return diff.sum(dim=-1, keepdim=True) / denom


def _masked_mean(current_lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = current_lp * mask
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return vals.sum(dim=-1, keepdim=True) / denom


def compute_frontier_repair_weights(
    batch,
    current_old_log_probs: torch.Tensor,
    frontier_cfg,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute per-sample repair weights for frontier replay samples.

    当前实现目标：
    - `none`: no scaling
    - `similarity_gated`: 根据当前 logprob 与 recorded logprob 的接近程度进行 gating
    - `mixture`: 固定混合系数
    - `dr3_local`: 轻量 local ratio proxy，在 cell 内按 group 归一化
    """
    response_len = current_old_log_probs.shape[-1]
    device = current_old_log_probs.device
    weights = torch.ones((current_old_log_probs.shape[0], 1), dtype=torch.float32, device=device)

    extras = list(batch.non_tensor_batch.get("extras", []))
    if not extras:
        return weights, {}

    response_mask = batch.batch.get("response_mask", None)
    if response_mask is None:
        response_mask = batch.batch["loss_mask"][:, -response_len:]
    else:
        response_mask = response_mask[:, :response_len]
    response_mask = response_mask.float()

    recorded = batch.batch.get("recorded_old_log_probs", None)
    if recorded is None:
        recorded = torch.zeros_like(current_old_log_probs)
    else:
        recorded = recorded[:, :response_len].to(device=device, dtype=current_old_log_probs.dtype)

    mode = str(frontier_cfg.get("repair_mode", "none")).lower().strip()
    sim_temp = float(frontier_cfg.get("similarity_temperature", 1.0))
    if sim_temp <= 0:
        sim_temp = 1.0
    mixture_alpha = float(frontier_cfg.get("mixture_alpha", 0.5))
    local_clip = float(frontier_cfg.get("dr3_local_log_ratio_clip", 2.0))
    min_w = float(frontier_cfg.get("repair_min", 0.1))
    max_w = float(frontier_cfg.get("repair_max", 2.0))

    frontier_indices = []
    group_ids = batch.batch.get("group_ids", None)
    group_ids_cpu = group_ids.detach().cpu().tolist() if torch.is_tensor(group_ids) else list(range(len(extras)))

    for i, extra in enumerate(extras):
        if isinstance(extra, dict) and extra.get("is_frontier_replay", False):
            frontier_indices.append(i)

    if not frontier_indices or mode == "none":
        metrics = {"frc/repair_num_samples": float(len(frontier_indices))}
        return weights, metrics

    frontier_weights = {}
    for idx in frontier_indices:
        mask = response_mask[idx : idx + 1]
        cur_lp = current_old_log_probs[idx : idx + 1]
        rec_lp = recorded[idx : idx + 1]
        valid_mask = mask.clone()

        rec_nonzero = (rec_lp.abs() > 1e-8).float()
        if rec_nonzero.sum() > 0:
            valid_mask = valid_mask * rec_nonzero

        if mode == "similarity_gated":
            if valid_mask.sum().item() > 0:
                mean_abs_diff = _masked_mean_abs_diff(cur_lp, rec_lp, valid_mask)
                w = torch.exp(-mean_abs_diff / sim_temp)
            else:
                w = torch.ones((1, 1), device=device, dtype=torch.float32)
        elif mode == "mixture":
            if valid_mask.sum().item() > 0:
                mean_abs_diff = _masked_mean_abs_diff(cur_lp, rec_lp, valid_mask)
                similarity = torch.exp(-mean_abs_diff / sim_temp)
            else:
                similarity = torch.ones((1, 1), device=device, dtype=torch.float32)
            w = mixture_alpha + (1.0 - mixture_alpha) * similarity
        elif mode == "dr3_local":
            if valid_mask.sum().item() > 0:
                log_ratio = _masked_mean(cur_lp - rec_lp, valid_mask).clamp(min=-local_clip, max=local_clip)
                w = torch.exp(log_ratio)
            else:
                # teacher/no-logprob fallback: use current confidence as local recoverability proxy
                conf = _masked_mean(cur_lp, mask).clamp(min=-local_clip, max=0.0)
                w = torch.exp(conf)
        else:
            w = torch.ones((1, 1), device=device, dtype=torch.float32)

        frontier_weights[idx] = w.squeeze().detach()

    # Group-normalize `dr3_local` so weight semantics are local to each frontier cell.
    if mode == "dr3_local":
        gid_to_vals = defaultdict(list)
        for idx in frontier_indices:
            gid_to_vals[group_ids_cpu[idx]].append(float(frontier_weights[idx].item()))
        gid_to_mean = {
            gid: max(float(np.mean(vals)), 1e-6)
            for gid, vals in gid_to_vals.items()
        }
        for idx in frontier_indices:
            gid = group_ids_cpu[idx]
            frontier_weights[idx] = torch.tensor(
                float(frontier_weights[idx].item()) / gid_to_mean[gid],
                device=device,
                dtype=torch.float32,
            )

    for idx in frontier_indices:
        weights[idx, 0] = frontier_weights[idx].clamp(min=min_w, max=max_w)

    selected = weights[frontier_indices, 0]
    metrics = {
        "frc/repair_num_samples": float(len(frontier_indices)),
        "frc/repair_weight_mean": float(selected.mean().detach().item()),
        "frc/repair_weight_min": float(selected.min().detach().item()),
        "frc/repair_weight_max": float(selected.max().detach().item()),
    }
    return weights, metrics
