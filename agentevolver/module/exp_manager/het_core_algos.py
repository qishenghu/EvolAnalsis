"""
Heterogeneous Core Algorithms for Experience Replay.

This module provides loss computation functions for:
- het_compute_token_on_off_policy_loss: Original ExGRPO self-generated experience
- het_compute_teacher_aware_loss: Extended version supporting teacher experience (LUFFY)
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Optional

import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss



def het_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    off_policy_shaping_mode: str = "higher_clip_bound",  # "higher_clip_bound" or "exgrpo_policy_shaping"
    off_policy_shaping_beta: float = 0.1,  # β for ExGRPO policy shaping: f(x) = x/(x+β)
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning using PPO.

    Args:
        old_log_prob (Tensor): Log probabilities of the actions under the old policy.
        log_prob (Tensor): Log probabilities of the actions under the new policy.
        advantages (Tensor): Advantage values for the actions.
        response_mask (Tensor): Mask indicating which tokens are part of the response.
        exp_mask (Tensor): Mask indicating which tokens are part of the experience replay.
        cliprange (float, optional): Clipping range for the policy ratio.
        cliprange_low (float, optional): Lower bound for the clipping range.
        cliprange_high (float, optional): Upper bound for the clipping range.
        off_cliprange_high (float, optional): Upper bound for the off-policy clipping range.
        clip_ratio_c (float, optional): Constant used in the clipping mechanism.
        loss_agg_mode (str, optional): Mode for aggregating the losses. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing various computed losses and metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    ratio = torch.exp(negative_approx_kl)

    # =========================
    # Diagnostics (endogenous replay)
    # =========================
    def _masked_stats(x: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict:
        out: dict[str, torch.Tensor] = {}
        if x is None or mask is None:
            out[f"{prefix}/count"] = torch.tensor(0.0, device=log_prob.device)
            return out
        if not torch.is_tensor(mask):
            out[f"{prefix}/count"] = torch.tensor(0.0, device=log_prob.device)
            return out
        mask_b = mask.bool()
        if mask_b.sum().item() == 0:
            out[f"{prefix}/count"] = torch.tensor(0.0, device=x.device)
            return out

        vals = x[mask_b]
        out[f"{prefix}/count"] = torch.tensor(float(vals.numel()), device=x.device)
        out[f"{prefix}/mean"] = vals.mean()
        out[f"{prefix}/std"] = vals.std() if vals.numel() > 1 else torch.tensor(0.0, device=x.device)
        out[f"{prefix}/min"] = vals.min()
        out[f"{prefix}/max"] = vals.max()

        # quantiles (sample if too many)
        max_q_samples = 4096
        if vals.numel() > max_q_samples:
            idx = torch.randperm(vals.numel(), device=vals.device)[:max_q_samples]
            vals_q = vals[idx]
        else:
            vals_q = vals
        try:
            out[f"{prefix}/p90"] = torch.quantile(vals_q, 0.90)
            out[f"{prefix}/p99"] = torch.quantile(vals_q, 0.99)
        except Exception:
            pass
        return out

    off_token_mask = exp_mask * response_mask
    on_token_mask = (1.0 - exp_mask) * response_mask
    exp_replay_diag_stats: dict[str, torch.Tensor] = {}
    # Importance ratio stats (off-policy tokens only)
    exp_replay_diag_stats.update(_masked_stats(ratio, off_token_mask, "importance_ratio/off"))
    # Advantage stats split by on/off tokens (helps detect baseline pollution / exploration suppression)
    exp_replay_diag_stats.update(_masked_stats(advantages, off_token_mask, "adv/off"))
    exp_replay_diag_stats.update(_masked_stats(advantages, on_token_mask, "adv/on"))

    def compute_pg_losses(cliprange_low, cliprange_high):
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        # NOTE:
        # - clipfrac (upper): where the clipped surrogate is chosen over unclipped (PPO-style max)
        # - clipfrac_lower: where the "extreme" safety clipping is active for negative advantages
        upper_hit = torch.gt(pg_losses2, pg_losses1)
        lower_hit = torch.gt(clip_pg_losses1, pg_losses3) & (advantages < 0)
        clipfrac = verl_F.masked_mean(upper_hit.float(), response_mask)
        clipfrac_lower = verl_F.masked_mean(lower_hit.float(), response_mask)
        return pg_losses, clipfrac, clipfrac_lower

    def compute_cliphit_rate(cliprange_low, cliprange_high, token_mask: torch.Tensor) -> torch.Tensor:
        """
        Combined clip hit rate = P(upper clip hit OR lower/extreme clip hit) over tokens in token_mask.
        """
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        upper_hit = torch.gt(pg_losses2, pg_losses1)
        lower_hit = torch.gt(clip_pg_losses1, pg_losses3) & (advantages < 0)
        cliphit = upper_hit | lower_hit
        return verl_F.masked_mean(cliphit.float(), token_mask)

    # On-policy calculations
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses, on_pg_clipfrac, on_pg_clipfrac_lower = compute_pg_losses(cliprange_low, cliprange_high)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)  # ⭐ Compute the on-policy loss
    # Cliphit rate for ON-policy tokens only (avoid off-policy dilution)
    on_policy_mask = (1.0 - exp_mask) * response_mask
    on_pg_cliphit_rate = compute_cliphit_rate(cliprange_low, cliprange_high, on_policy_mask)

    # Off-policy calculations
    if off_policy_shaping_mode == "exgrpo_policy_shaping":
        # ⭐ ExGRPO Policy Shaping: Replace CLIP term with f(w*(θ)) = w*(θ) / (w*(θ) + β)
        # where w*(θ) = exp(log_prob - old_log_prob) is the importance sampling ratio
        # This amplifies low-probability signals and dampens high-probability ones
        negative_approx_kl_off = log_prob - old_log_prob
        off_ratio = torch.exp(negative_approx_kl_off)  # w*(θ) = π_new / π_old
        
        # Apply policy shaping: f(x) = x / (x + β)
        off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)
        # shaped ratio stats (off-policy tokens only)
        exp_replay_diag_stats.update(_masked_stats(off_ratio_shaped, off_token_mask, "importance_ratio_shaped/off"))
        
        # Replace CLIP term with shaped ratio: -advantages * f(w*(θ))
        off_pg_losses = -advantages * off_ratio_shaped
        
        # No clipping for off-policy when using policy shaping
        off_pg_clipfrac = torch.tensor(0.0)
        off_pg_clipfrac_lower = torch.tensor(0.0)
        off_pg_cliphit_rate = torch.tensor(0.0)
        
        # Compute off-policy loss (only on LLM response tokens in multi-turn)
        off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
        off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss
        
    elif off_policy_shaping_mode == "higher_clip_bound":
        # ⭐ AgentEvolver original: Use higher clip_upper_bound for off-policy data
        off_cliprange_low = cliprange_low
        off_pg_losses, off_pg_clipfrac, off_pg_clipfrac_lower = compute_pg_losses(off_cliprange_low, off_cliprange_high)
        off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)  # ⭐ Compute the off-policy loss
        off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss
        off_policy_mask = exp_mask * response_mask
        off_pg_cliphit_rate = compute_cliphit_rate(off_cliprange_low, off_cliprange_high, off_policy_mask)
    else:
        raise ValueError(f"Invalid off_policy_shaping_mode: {off_policy_shaping_mode}. Must be 'exgrpo_policy_shaping' or 'higher_clip_bound'")

    # Combine on-policy and off-policy losses
    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)  # ⭐ Aggregate the combined losses

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        # ⭐ New: combined clip-hit rate (upper OR lower) on on/off tokens
        "on_pg_cliphit_rate": on_pg_cliphit_rate,
        "off_pg_cliphit_rate": off_pg_cliphit_rate,
        "ppo_kl": ppo_kl,
        # ⭐ Endogenous replay diagnostics (logged by het_actor.py)
        "exp_replay_diag_stats": exp_replay_diag_stats,
    }


def het_compute_teacher_aware_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,
    teacher_mask: Optional[torch.Tensor] = None,  # ⭐ 新增：标记 Teacher 轨迹
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    off_cliprange_high=1.0,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    off_policy_shaping_mode: str = "higher_clip_bound",
    off_policy_shaping_beta: float = 0.1,
    # ⭐ Teacher 专用配置
    teacher_use_log_prob: bool = False,  # 是否使用 log_prob（有则用 ExGRPO，无则用 LUFFY）
    teacher_policy_shaping_enable: bool = True,  # 是否启用 policy shaping
    teacher_policy_shaping_mode: str = "p_div_p_beta",  # Policy shaping 模式
    teacher_policy_shaping_beta: float = 0.1,  # Policy shaping 参数 β
    teacher_use_clip: bool = False,  # Teacher 轨迹是否使用 clipping
    teacher_loss_scale: Optional[torch.Tensor] = None,  # ⭐ 2.2: scale teacher loss (0..1), broadcastable to (bs, resp_len)
    # ⭐ 7.7: TER sequence-level β schedule (soft-min teacher confidence)
    # 仅用于 teacher_use_log_prob=False 且 policy_shaping_mode == "p_div_p_beta"
    teacher_seq_beta_enable: bool = False,
    teacher_seq_beta_alpha: float = -5.0,          # α < 0; more negative => more tail/min sensitive
    teacher_seq_beta_c0: float = 0.25,             # center of sigmoid in confidence space (prob-space by default)
    teacher_seq_beta_temperature: float = 0.05,    # transition smoothness; smaller => sharper
    # ⭐ 7.7-v2: gate in log-prob space to fix scale mismatch for LLM token probabilities
    # If enabled, use logC_alpha and logc0 (typically negative, e.g., -6) for gating.
    teacher_seq_beta_gate_space: str = "prob",     # "prob" | "logprob"
    teacher_seq_beta_logc0: Optional[float] = None,# if None and gate_space=="logprob", fall back to log(c0) when 0<c0<1
    teacher_seq_beta_log_temperature: float = 0.5, # temperature in log-space
    # ⭐ 7.7-v3: robust confidence mode to avoid extreme tail token pinning logC to very negative values
    # - "gen_mean_prob": original generalized-mean on prob space (alpha<0)
    # - "quantile_logp": per-trajectory quantile of teacher-token logp (e.g., q=0.1)
    # - "cvar_logp": mean of the lowest-q fraction of teacher-token logp (CVaR; more stable than min)
    teacher_seq_beta_conf_mode: str = "gen_mean_prob",
    teacher_seq_beta_logp_q: float = 0.10,        # used by quantile_logp / cvar_logp
    teacher_seq_beta_max_tokens_per_traj: int = 4096,  # sampling cap for robust stats
    teacher_seq_beta_beta_min: float = 0.05,       # >=0; stronger teacher (since r=p/(p+β))
    teacher_seq_beta_beta_max: float = 0.30,       # >= beta_min; weaker teacher
    teacher_seq_beta_p_min: float = 1e-4,          # clamp prob to avoid overflow for alpha<0
    teacher_seq_beta_stop_grad: bool = True,
    # ⭐ 7.6 AG-PM: Advantage-Gated Probability Margin
    # 双门控机制：(1) Advantage Gate + (2) Probability Margin Gate
    teacher_ag_pm_enable: bool = False,
    # Advantage Gate 参数（继承 7.4 思想）
    teacher_ag_pm_adv_threshold: float = 0.4,
    teacher_ag_pm_adv_temperature: float = 0.2,
    teacher_ag_pm_adv_min: float = 0.5,
    teacher_ag_pm_adv_max: float = 1.0,
    # Probability Margin Gate 参数（新机制）
    teacher_ag_pm_prob_max: float = 0.9,  # 概率上界，达到后停止学习
    teacher_ag_pm_prob_temperature: float = 0.02,  # 非常陡峭，近似硬截断
    teacher_ag_pm_prob_min: float = 0.0,
    teacher_ag_pm_prob_max_gate: float = 1.0,
    teacher_ag_pm_stop_grad: bool = True,
):
    """
    计算混合 on-policy、self-generated off-policy 和 teacher off-policy 的 loss。
    
    ⭐ 扩展自 het_compute_token_on_off_policy_loss，增加 teacher_mask 支持
    
    Teacher 轨迹的两种处理模式：
    1. teacher_use_log_prob=True: 使用 ExGRPO 形式，ratio = π_current / π_old（需要 log_prob）
    2. teacher_use_log_prob=False: 使用 LUFFY 形式，ratio = π_current（分母=1）
    
    Args:
        old_log_prob: [batch, seq_len] - 旧策略的 log prob
        log_prob: [batch, seq_len] - 当前策略的 log prob
        advantages: [batch, seq_len] - 优势值
        response_mask: [batch, seq_len] - 响应 token mask
        exp_mask: [batch, seq_len] - Off-policy mask (1=off-policy, 0=on-policy)
        teacher_mask: [batch, seq_len] - Teacher 轨迹 mask (1=teacher, 0=非 teacher)
        teacher_use_log_prob: 是否使用 log_prob 进行重要性采样
        teacher_policy_shaping_enable: 是否启用 policy shaping（teacher_use_log_prob=False 时推荐）
        teacher_policy_shaping_mode: Policy shaping 模式 ("p_div_p_beta", "sqrt", "no_shaping")
        teacher_policy_shaping_beta: β 参数
        teacher_use_clip: Teacher 轨迹是否使用 clipping
        
    Returns:
        dict: 包含各种 loss 和 metrics
    """
    # 如果没有 teacher_mask，直接调用原函数
    if teacher_mask is None:
        return het_compute_token_on_off_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            exp_mask=exp_mask,
            cliprange=cliprange,
            cliprange_low=cliprange_low,
            cliprange_high=cliprange_high,
            off_cliprange_high=off_cliprange_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
            off_policy_shaping_mode=off_policy_shaping_mode,
            off_policy_shaping_beta=off_policy_shaping_beta,
        )
    
    # =============== 基础计算 ===============
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    ratio = torch.exp(negative_approx_kl)  # 标准 ratio = π_new / π_old
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    # =============== On-policy loss（与现有逻辑相同） ===============
    on_pg_losses1 = -advantages * ratio
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)
    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    
    on_upper_hit = torch.gt(on_pg_losses2, on_pg_losses1)
    on_lower_hit = torch.gt(on_clip_pg_losses1, on_pg_losses3) & (advantages < 0)
    on_pg_clipfrac = verl_F.masked_mean(on_upper_hit.float(), response_mask)
    on_pg_clipfrac_lower = verl_F.masked_mean(on_lower_hit.float(), response_mask)
    # combined (upper OR lower) clip-hit rate
    on_pg_cliphit_rate_all = on_upper_hit | on_lower_hit
    
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)
    on_policy_mask = (1.0 - exp_mask) * response_mask
    on_pg_cliphit_rate = verl_F.masked_mean(on_pg_cliphit_rate_all.float(), on_policy_mask)
    
    # =============== Self-generated off-policy loss ===============
    # 使用现有的 off_policy_shaping_mode
    if off_policy_shaping_mode == "exgrpo_policy_shaping":
        self_off_ratio = ratio
        self_off_ratio_shaped = self_off_ratio / (self_off_ratio + off_policy_shaping_beta)
        self_off_pg_losses = -advantages * self_off_ratio_shaped
        self_off_pg_cliphit_rate = torch.tensor(0.0, device=log_prob.device)
    else:  # higher_clip_bound
        self_off_pg_losses1 = -advantages * ratio
        self_off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + off_cliprange_high)
        self_off_pg_losses = torch.maximum(self_off_pg_losses1, self_off_pg_losses2)
        self_upper_hit = torch.gt(self_off_pg_losses2, self_off_pg_losses1)
        self_exp_mask_local = exp_mask * (1.0 - teacher_mask_float)
        self_off_pg_cliphit_rate = verl_F.masked_mean(self_upper_hit.float(), self_exp_mask_local * response_mask)
    
    # =============== Teacher off-policy loss（LUFFY） ===============
    teacher_mask_float = teacher_mask.float()
    
    if teacher_use_log_prob:
        # ========== 模式 1: 使用 log_prob（ExGRPO 形式） ==========
        # Teacher 轨迹有 log_prob，使用标准重要性采样
        # ratio = π_current / π_old = exp(log_prob - old_log_prob)
        teacher_ratio = ratio  # 直接使用标准 ratio
    else:
        # ========== 模式 2: 无 log_prob（LUFFY 形式） ==========
        # Teacher 轨迹无 log_prob，使用简化计算
        # 假设 π_old = 1，ratio = π_current / 1 = π_current
        teacher_ratio = torch.exp(log_prob)
    
    # ⭐ 7.7: TER sequence-level β schedule (soft-min / tail-aware)
    # 目的：避免 token-level gate 的误伤；在“仍有短板 token”时不削弱 teacher，在“整体已学会”时逐步减弱 teacher。
    # 说明：这里只改变 β（shaping 强度），不改变 7.1 的 baseline separation 或其他 7.3/7.6 功能。
    teacher_seq_beta_C = None
    teacher_seq_beta_logC = None  # shape: (bs, 1)
    teacher_seq_beta_logC_robust = None  # shape: (bs, 1), for quantile/cvar modes
    teacher_seq_beta_beta = None  # shape: (bs, 1)
    if (
        teacher_seq_beta_enable
        and (not teacher_use_log_prob)
        and teacher_policy_shaping_enable
        and str(teacher_policy_shaping_mode).lower().strip() == "p_div_p_beta"
    ):
        # mask teacher tokens inside response
        _tm = (teacher_mask_float * response_mask).float()  # (bs, resp_len)
        _cnt = _tm.sum(dim=-1, keepdim=True)  # (bs, 1)
        if torch.any(_cnt > 0):
            conf_mode = str(teacher_seq_beta_conf_mode).lower().strip()
            max_tok = int(teacher_seq_beta_max_tokens_per_traj) if int(teacher_seq_beta_max_tokens_per_traj) > 0 else 4096
            # teacher token logp (<=0) for robust modes
            token_logp = log_prob.clamp(max=0)  # (bs, resp_len)

            if conf_mode in ("quantile_logp", "cvar_logp"):
                # per-trajectory robust log-confidence based on teacher-token logp distribution
                q = float(teacher_seq_beta_logp_q)
                if not (0.0 < q < 1.0):
                    q = 0.10
                vals = []
                bs = token_logp.shape[0]
                for i in range(bs):
                    mi = _tm[i].bool()
                    if mi.sum().item() == 0:
                        vals.append(torch.tensor([-1e9], device=token_logp.device, dtype=token_logp.dtype))
                        continue
                    vi = token_logp[i][mi]
                    # cap for speed
                    if vi.numel() > max_tok:
                        idx = torch.randperm(vi.numel(), device=vi.device)[:max_tok]
                        vi = vi[idx]
                    vi_sorted, _ = torch.sort(vi)
                    k = int(max(1, round(q * vi_sorted.numel())))
                    if conf_mode == "quantile_logp":
                        # quantile at q (e.g., p10 of logp)
                        vals.append(vi_sorted[k - 1].view(1))
                    else:
                        # CVaR (mean of worst-q fraction)
                        vals.append(vi_sorted[:k].mean().view(1))
                teacher_seq_beta_logC_robust = torch.stack(vals, dim=0).view(-1, 1)  # (bs,1)
                # define C for logging parity (not used for gating unless gate_space=="prob")
                teacher_seq_beta_logC = teacher_seq_beta_logC_robust
                teacher_seq_beta_C = torch.exp(teacher_seq_beta_logC_robust)
            else:
                # legacy: generalized-mean on prob space
                # teacher token probability p_t = exp(log_prob) ∈ (0,1]
                # clamp max=0 ensures p<=1; clamp min avoids overflow when alpha<0
                p_min = float(teacher_seq_beta_p_min) if float(teacher_seq_beta_p_min) > 0 else 1e-4
                p = torch.exp(token_logp).clamp(min=p_min)  # (bs, resp_len)

                alpha = float(teacher_seq_beta_alpha)
                if alpha >= 0:
                    # enforce alpha<0; fallback to -5.0
                    alpha = -5.0

                # generalized mean for alpha<0:
                # C_alpha = (mean(p^alpha))^(1/alpha)
                # compute p^alpha in log-space for stability: exp(alpha * log p)
                logp_prob = torch.log(p)
                pa = torch.exp((alpha * logp_prob).clamp(min=-80.0, max=80.0))  # (bs, resp_len)
                mean_pa = (pa * _tm).sum(dim=-1, keepdim=True) / (_cnt + 1e-8)  # (bs,1)
                mean_pa = mean_pa.clamp(min=1e-20)
                # logC_alpha = (1/alpha) * log(mean(p^alpha))
                teacher_seq_beta_logC = (1.0 / alpha) * torch.log(mean_pa)  # (bs,1), typically negative
                teacher_seq_beta_C = torch.exp(teacher_seq_beta_logC)  # (bs,1), in (0,1]

            # beta schedule: beta in [beta_min, beta_max]
            beta_min = float(teacher_seq_beta_beta_min) if float(teacher_seq_beta_beta_min) >= 0 else 0.0
            beta_max = float(teacher_seq_beta_beta_max) if float(teacher_seq_beta_beta_max) >= 0 else beta_min
            if beta_min > beta_max:
                beta_min, beta_max = beta_max, beta_min

            gate_space = str(teacher_seq_beta_gate_space).lower().strip()
            if gate_space == "logprob":
                # log-space gating: gate = sigmoid((logC - logc0) / Tlog)
                Tlog = float(teacher_seq_beta_log_temperature) if float(teacher_seq_beta_log_temperature) > 0 else 0.5
                logc0 = teacher_seq_beta_logc0
                if logc0 is None:
                    # fallback: if 0<c0<1, use log(c0); else use a conservative default
                    c0 = float(teacher_seq_beta_c0)
                    if 0.0 < c0 < 1.0:
                        logc0 = float(torch.log(torch.tensor(c0)).item())
                    else:
                        logc0 = -6.0
                gate = torch.sigmoid((teacher_seq_beta_logC - float(logc0)) / Tlog)  # (bs,1)
            else:
                # prob-space gating (legacy): gate = sigmoid((C - c0) / T)
                c0 = float(teacher_seq_beta_c0)
                temp = float(teacher_seq_beta_temperature) if float(teacher_seq_beta_temperature) > 0 else 0.05
                gate = torch.sigmoid((teacher_seq_beta_C - c0) / temp)  # (bs,1)
            teacher_seq_beta_beta = beta_min + (beta_max - beta_min) * gate  # (bs,1)

            if teacher_seq_beta_stop_grad:
                teacher_seq_beta_C = teacher_seq_beta_C.detach()
                teacher_seq_beta_logC = teacher_seq_beta_logC.detach() if teacher_seq_beta_logC is not None else None
                teacher_seq_beta_logC_robust = teacher_seq_beta_logC_robust.detach() if teacher_seq_beta_logC_robust is not None else None
                teacher_seq_beta_beta = teacher_seq_beta_beta.detach()
    
    # Policy shaping（LUFFY 推荐使用）
    if teacher_policy_shaping_enable:
        beta_for_shaping = teacher_policy_shaping_beta
        if teacher_seq_beta_beta is not None and torch.is_tensor(teacher_seq_beta_beta):
            beta_for_shaping = teacher_seq_beta_beta  # (bs,1), broadcastable
        teacher_ratio = _apply_policy_shaping(
            teacher_ratio,
            mode=teacher_policy_shaping_mode,
            beta=beta_for_shaping,
        )

    # ⭐ 7.6 AG-PM: Advantage-Gated Probability Margin
    # 双门控机制：只向"好老师"学习，但只学到"懂了为止"
    teacher_ag_pm_adv_gate_w = None
    teacher_ag_pm_prob_gate_w = None
    teacher_prob = None  # 用于统计
    if teacher_ag_pm_enable:
        # === 第一道门：Advantage Gate (继承 7.4 思想) ===
        # 只学习高 advantage 的 teacher token
        _adv_thr = float(teacher_ag_pm_adv_threshold)
        _adv_temp = float(teacher_ag_pm_adv_temperature) if float(teacher_ag_pm_adv_temperature) > 0 else 0.2
        _adv_min = float(teacher_ag_pm_adv_min)
        _adv_max = float(teacher_ag_pm_adv_max)
        if _adv_min > _adv_max:
            _adv_min, _adv_max = _adv_max, _adv_min
        
        teacher_ag_pm_adv_gate_w = torch.sigmoid((advantages - _adv_thr) / _adv_temp)
        teacher_ag_pm_adv_gate_w = torch.clamp(teacher_ag_pm_adv_gate_w, min=_adv_min, max=_adv_max)
        
        # === 第二道门：Probability Margin Gate (新机制) ===
        # 当 π(a_T|s) >= prob_max 时，停止学习（防止熵坍塌）
        _prob_max = float(teacher_ag_pm_prob_max)
        _prob_temp = float(teacher_ag_pm_prob_temperature) if float(teacher_ag_pm_prob_temperature) > 0 else 0.02
        _prob_min = float(teacher_ag_pm_prob_min)
        _prob_max_gate = float(teacher_ag_pm_prob_max_gate)
        if _prob_min > _prob_max_gate:
            _prob_min, _prob_max_gate = _prob_max_gate, _prob_min
        
        # 计算 π(a_T|s) = exp(log_prob)，clamp 避免数值问题
        teacher_prob = torch.exp(log_prob.clamp(max=0))  # log_prob <= 0, 所以 prob <= 1
        
        # Bang-Bang Control: 用陡峭 sigmoid 近似硬截断 I(π < prob_max)
        # 当 prob < prob_max 时 gate ≈ 1，当 prob >= prob_max 时 gate ≈ 0
        teacher_ag_pm_prob_gate_w = torch.sigmoid((_prob_max - teacher_prob) / _prob_temp)
        teacher_ag_pm_prob_gate_w = torch.clamp(teacher_ag_pm_prob_gate_w, min=_prob_min, max=_prob_max_gate)
        
        # Stop gradient（将 gate 视为调度器，不反传梯度）
        if teacher_ag_pm_stop_grad:
            teacher_ag_pm_adv_gate_w = teacher_ag_pm_adv_gate_w.detach()
            teacher_ag_pm_prob_gate_w = teacher_ag_pm_prob_gate_w.detach()
        
        # 组合双门控
        teacher_ratio = teacher_ratio * teacher_ag_pm_adv_gate_w * teacher_ag_pm_prob_gate_w

    # ⭐ 2.2: Optional adaptive scaling for teacher loss
    if teacher_loss_scale is not None and torch.is_tensor(teacher_loss_scale):
        # Ensure broadcastable: (bs, resp_len) preferred; accept (bs, 1) or scalar-like
        if teacher_loss_scale.dim() == 1:
            teacher_loss_scale = teacher_loss_scale.unsqueeze(-1)
        teacher_ratio = teacher_ratio * teacher_loss_scale
    
    # Teacher loss 计算
    teacher_off_pg_losses_raw = -advantages * teacher_ratio
    
    if teacher_use_clip:
        teacher_off_pg_losses2 = -advantages * torch.clamp(teacher_ratio, 1 - cliprange_low, 1 + off_cliprange_high)
        teacher_off_pg_losses = torch.maximum(teacher_off_pg_losses_raw, teacher_off_pg_losses2)
        teacher_upper_hit = torch.gt(teacher_off_pg_losses2, teacher_off_pg_losses_raw)
        teacher_off_pg_cliphit_mask = teacher_upper_hit
    else:
        teacher_off_pg_losses = teacher_off_pg_losses_raw  # 不 clip
        teacher_off_pg_cliphit_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    
    # =============== 混合 off-policy loss ===============
    # 根据 teacher_mask 选择使用哪种 off-policy loss
    # - teacher_mask=1: Teacher 轨迹，使用 teacher_off_pg_losses
    # - teacher_mask=0 且 exp_mask=1: Self-generated，使用 self_off_pg_losses
    off_pg_losses = torch.where(
        teacher_mask_float.bool(),
        teacher_off_pg_losses,
        self_off_pg_losses
    )
    
    # 计算各类型的 off-policy loss（用于 logging）
    # Self-generated off-policy loss（排除 teacher）
    self_exp_mask = exp_mask * (1.0 - teacher_mask_float)
    self_off_pg_loss = verl_F.masked_mean(self_off_pg_losses, self_exp_mask * response_mask)
    self_off_pg_loss = torch.tensor(0.0, device=log_prob.device) if self_off_pg_loss.isnan().item() else self_off_pg_loss
    
    # Teacher off-policy loss
    teacher_exp_mask = exp_mask * teacher_mask_float
    teacher_off_pg_loss = verl_F.masked_mean(teacher_off_pg_losses, teacher_exp_mask * response_mask)
    teacher_off_pg_loss = torch.tensor(0.0, device=log_prob.device) if teacher_off_pg_loss.isnan().item() else teacher_off_pg_loss
    teacher_off_pg_cliphit_rate = verl_F.masked_mean(
        teacher_off_pg_cliphit_mask.float(),
        teacher_exp_mask * response_mask,
    )

    # =============== 诊断指标：ratio / advantage 分布（用于分析 teacher 影响） ===============
    # 说明：
    # - 这些指标用于判断 teacher 对更新的“有效权重”是否过小/过大，以及是否导致训练不稳定。
    # - 为避免在超长 response 上计算分位数过慢，这里对样本数做上限采样。
    def _masked_stats(x: torch.Tensor, m: torch.Tensor, prefix: str) -> dict:
        # x, m: (bs, response_len)
        out = {}
        mask = m.bool()
        if mask.sum() == 0:
            out[f"{prefix}/count"] = torch.tensor(0.0, device=x.device)
            return out

        vals = x[mask]
        out[f"{prefix}/count"] = torch.tensor(float(vals.numel()), device=x.device)
        out[f"{prefix}/mean"] = vals.mean()
        out[f"{prefix}/std"] = vals.std() if vals.numel() > 1 else torch.tensor(0.0, device=x.device)
        out[f"{prefix}/min"] = vals.min()
        out[f"{prefix}/max"] = vals.max()

        # quantiles (sample if too many)
        max_q_samples = 4096
        if vals.numel() > max_q_samples:
            idx = torch.randperm(vals.numel(), device=vals.device)[:max_q_samples]
            vals_q = vals[idx]
        else:
            vals_q = vals
        try:
            out[f"{prefix}/p50"] = torch.quantile(vals_q, 0.50)
            out[f"{prefix}/p90"] = torch.quantile(vals_q, 0.90)
            out[f"{prefix}/p99"] = torch.quantile(vals_q, 0.99)
        except Exception:
            # quantile may fail on some backends; keep mean/min/max only
            pass
        return out

    def _vector_stats(v: torch.Tensor, prefix: str) -> dict:
        """
        v: 1D tensor of per-trajectory values (teacher-only).
        This avoids token-weighting artifacts when a scalar (e.g., beta) is broadcast to tokens.
        """
        out: dict[str, torch.Tensor] = {}
        if v is None or (not torch.is_tensor(v)) or v.numel() == 0:
            out[f"{prefix}/count"] = torch.tensor(0.0, device=log_prob.device)
            return out
        vv = v.flatten()
        out[f"{prefix}/count"] = torch.tensor(float(vv.numel()), device=vv.device)
        out[f"{prefix}/mean"] = vv.mean()
        out[f"{prefix}/std"] = vv.std() if vv.numel() > 1 else torch.tensor(0.0, device=vv.device)
        out[f"{prefix}/min"] = vv.min()
        out[f"{prefix}/max"] = vv.max()
        try:
            out[f"{prefix}/p50"] = torch.quantile(vv, 0.50)
            out[f"{prefix}/p90"] = torch.quantile(vv, 0.90)
            out[f"{prefix}/p99"] = torch.quantile(vv, 0.99)
        except Exception:
            pass
        return out

    # masks
    teacher_token_mask = teacher_exp_mask * response_mask
    self_token_mask = self_exp_mask * response_mask
    on_token_mask = (1.0 - exp_mask) * response_mask
    # per-trajectory teacher presence mask (bool): does this sequence contain any teacher tokens?
    teacher_traj_mask = (teacher_token_mask.sum(dim=-1) > 0)  # (bs,)

    ratio_stats = {}
    # teacher_ratio: after (optional) shaping, used for teacher loss
    ratio_stats.update(_masked_stats(teacher_ratio, teacher_token_mask, "teacher_ratio"))
    # 7.7: sequence-level beta schedule diagnostics (token-level)
    if teacher_seq_beta_C is not None and torch.is_tensor(teacher_seq_beta_C):
        # broadcast to token shape for masked stats
        ratio_stats.update(
            _masked_stats(teacher_seq_beta_C.expand_as(teacher_ratio), teacher_token_mask, "seq_beta/C_alpha")
        )
    if teacher_seq_beta_logC is not None and torch.is_tensor(teacher_seq_beta_logC):
        ratio_stats.update(
            _masked_stats(teacher_seq_beta_logC.expand_as(teacher_ratio), teacher_token_mask, "seq_beta/logC_alpha")
        )
    if teacher_seq_beta_logC_robust is not None and torch.is_tensor(teacher_seq_beta_logC_robust):
        ratio_stats.update(
            _masked_stats(teacher_seq_beta_logC_robust.expand_as(teacher_ratio), teacher_token_mask, "seq_beta/logC_robust")
        )
    if teacher_seq_beta_beta is not None and torch.is_tensor(teacher_seq_beta_beta):
        ratio_stats.update(
            _masked_stats(teacher_seq_beta_beta.expand_as(teacher_ratio), teacher_token_mask, "seq_beta/beta")
        )

    # NOTE:
    # - We intentionally do NOT log traj-level stats here.
    # - Actor update uses micro-batches (often bsz=1), which makes traj-level stats degenerate (count=1).
    # - We instead return raw per-traj values for step-level aggregation in het_actor.
    teacher_diag_traj_values = {}
    if teacher_seq_beta_beta is not None and torch.is_tensor(teacher_seq_beta_beta):
        teacher_diag_traj_values["seq_beta/beta"] = teacher_seq_beta_beta.squeeze(-1)[teacher_traj_mask]  # (n_teacher_traj,)
    if teacher_seq_beta_logC is not None and torch.is_tensor(teacher_seq_beta_logC):
        teacher_diag_traj_values["seq_beta/logC_alpha"] = teacher_seq_beta_logC.squeeze(-1)[teacher_traj_mask]
    if teacher_seq_beta_logC_robust is not None and torch.is_tensor(teacher_seq_beta_logC_robust):
        teacher_diag_traj_values["seq_beta/logC_robust"] = teacher_seq_beta_logC_robust.squeeze(-1)[teacher_traj_mask]
    # 7.6 AG-PM: gate weights and probability
    if teacher_ag_pm_adv_gate_w is not None and torch.is_tensor(teacher_ag_pm_adv_gate_w):
        ratio_stats.update(_masked_stats(teacher_ag_pm_adv_gate_w, teacher_token_mask, "ag_pm_adv_gate_w"))
    if teacher_ag_pm_prob_gate_w is not None and torch.is_tensor(teacher_ag_pm_prob_gate_w):
        ratio_stats.update(_masked_stats(teacher_ag_pm_prob_gate_w, teacher_token_mask, "ag_pm_prob_gate_w"))
    if teacher_prob is not None and torch.is_tensor(teacher_prob):
        ratio_stats.update(_masked_stats(teacher_prob, teacher_token_mask, "teacher_prob"))
    # self_off ratio: exp(log_prob - old_log_prob)
    ratio_stats.update(_masked_stats(ratio, self_token_mask, "self_off_ratio"))
    # on-policy ratio is always 1 in PPO surrogate, but we can still log on-policy advantage stats
    ratio_stats.update(_masked_stats(advantages, teacher_token_mask, "adv/teacher"))
    ratio_stats.update(_masked_stats(advantages, self_token_mask, "adv/self_off"))
    ratio_stats.update(_masked_stats(advantages, on_token_mask, "adv/onpolicy"))
    
    # 总 off-policy loss
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    off_pg_loss = torch.tensor(0.0, device=log_prob.device) if off_pg_loss.isnan().item() else off_pg_loss
    
    # =============== 合并 loss ===============
    exp_mask_float = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask_float + on_pg_losses * (1.0 - exp_mask_float)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        # ⭐ 新增：分开统计 self-generated 和 teacher loss
        "self_off_pg_loss": self_off_pg_loss,
        "teacher_off_pg_loss": teacher_off_pg_loss,
        # ⭐ 诊断指标：用于分析 teacher 影响（ratio / advantage 分布）
        "teacher_diag_stats": ratio_stats,
        # ⭐ raw per-trajectory values for correct step-level aggregation (across micro-batches)
        "teacher_diag_traj_values": teacher_diag_traj_values,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        # ⭐ New: combined clip-hit rates
        "on_pg_cliphit_rate": on_pg_cliphit_rate,
        "self_off_pg_cliphit_rate": self_off_pg_cliphit_rate,
        "teacher_off_pg_cliphit_rate": teacher_off_pg_cliphit_rate,
        "ppo_kl": ppo_kl,
    }


def _apply_policy_shaping(ratio: torch.Tensor, mode: str, beta=0.1) -> torch.Tensor:
    """
    应用 Policy Shaping。
    
    Args:
        ratio: 原始 ratio
        mode: shaping 模式
            - "p_div_p_beta": LUFFY 的 f(x) = x / (x + β)，放大低概率信号
            - "sqrt": f(x) = sqrt(x)
            - "no_shaping": 不使用 shaping
        beta: β 参数（float 或 broadcastable tensor，例如 shape=(bs,1)）
    
    Returns:
        shaped_ratio: 处理后的 ratio
    """
    if mode == "p_div_p_beta":
        # LUFFY 的 policy shaping: f(x) = x / (x + β)
        # 放大低概率信号，抑制高概率信号
        return ratio / (ratio + beta)
    elif mode == "sqrt":
        return torch.sqrt(ratio + 1e-8)
    elif mode == "no_shaping":
        return ratio
    else:
        raise ValueError(f"Unknown policy_shaping_mode: {mode}. "
                        f"Supported: 'p_div_p_beta', 'sqrt', 'no_shaping'")


def bam_compute_token_on_off_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning.

    Args:
        old_log_prob (Tensor): Log probabilities of the old policy.
        log_prob (Tensor): Log probabilities of the current policy.
        advantages (Tensor): Advantage values.
        response_mask (Tensor): Mask indicating valid response tokens.
        exp_mask (Tensor): Mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): Clipping range for PPO. Defaults to None.
        cliprange_low (float, optional): Lower clipping range for PPO. Defaults to None.
        cliprange_high (float, optional): Upper clipping range for PPO. Defaults to None.
        clip_ratio_c (float, optional): Clipping ratio constant. Defaults to 3.0.
        loss_agg_mode (str, optional): Mode for aggregating the loss. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing various computed losses and metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: denominator=1 + reshape + no clipping

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    off_ratio = off_ratio / (off_ratio + 0.1)   # ⭐ Reshape the off-policy ratio to stabilize the loss
    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def bam_compute_token_on_off_policy_loss_v2(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning, using various methods to handle clipping and masking.

    Args:
        old_log_prob (Tensor): The old log probabilities of the actions.
        log_prob (Tensor): The new log probabilities of the actions.
        advantages (Tensor): The advantage values.
        response_mask (Tensor): A mask indicating which tokens are part of the response.
        exp_mask (Tensor): A mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): The range for clipping the ratio. Defaults to None.
        cliprange_low (float, optional): The lower bound for clipping the ratio. Defaults to None.
        cliprange_high (float, optional): The upper bound for clipping the ratio. Defaults to None.
        clip_ratio_c (float, optional): The clipping ratio constant. Defaults to 3.0.
        loss_agg_mode (str, optional): The mode for aggregating the loss. Defaults to "token-mean".

    Returns:
        dict: A dictionary containing the computed losses and other relevant metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: denominator=1 + reshape + no clipping

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length)
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_ratio = torch.exp(log_prob)     #(bs, response_length)
    off_ratio = off_ratio / (off_ratio + 0.1)   # ⭐ Reshape the off-policy ratio
    off_pg_losses = -advantages * off_ratio
    ############
    # ANNI add 0728: For negative samples with A<0, do not compute loss gradients, mask them out
    off_positive_mask = (exp_mask > 0) & (advantages >=0) & (response_mask > 0) # mask containing only off-policy data with advantages>=0
    adjusted_off_pg_losses = torch.where(off_positive_mask, off_pg_losses, torch.zeros_like(off_pg_losses))
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_positive_mask)
    if torch.isnan(off_pg_loss).item():
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = adjusted_off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        # "adjusted_off_pg_losses": adjusted_off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict

def dapo_compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask=None,
    cliprange_low=0.2,
    cliprange_high=0.28,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    # Off-policy (Experience Replay) settings
    off_policy_shaping_mode: str = "exgrpo_policy_shaping",
    off_policy_shaping_beta: float = 0.1,
):
    """
    Computes the policy loss using DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization).
    
    DAPO introduces asymmetric clipping (Clip-Higher) to encourage exploration:
    - For positive advantages (A > 0): Use clip_high to limit probability increase
    - For negative advantages (A < 0): Remove upper clipping to allow low-probability tokens to decrease further
    
    This decoupled clipping mechanism helps prevent entropy collapse by allowing the model
    to explore low-probability tokens while still constraining high-probability updates.
    
    ⭐ Experience-Replay Compatible:
    - On-policy data: Uses DAPO's Clip-Higher mechanism
    - Off-policy data: Uses ExGRPO policy shaping (f(x) = x/(x+β)) to handle importance sampling

    Args:
        old_log_prob (Tensor): Log probabilities of the actions under the old policy.
            Shape: (batch_size, response_length)
        log_prob (Tensor): Log probabilities of the actions under the new policy.
            Shape: (batch_size, response_length)
        advantages (Tensor): Advantage values for the actions.
            Shape: (batch_size, response_length) or (batch_size,)
        response_mask (Tensor): Mask indicating which tokens are part of the response.
            Shape: (batch_size, response_length)
        exp_mask (Tensor, optional): Mask indicating off-policy data (1) vs on-policy data (0).
            Shape: (batch_size, response_length). Defaults to None (all on-policy).
        cliprange_low (float): Lower bound for clipping range. Default: 0.2 (DAPO default: ε_low)
        cliprange_high (float): Upper bound for clipping range. Default: 0.28 (DAPO default: ε_high)
        clip_ratio_c (float): Maximum ratio for extreme clipping. Default: 3.0
        loss_agg_mode (str): Mode for aggregating the loss. Defaults to "token-mean".
        off_policy_shaping_mode (str): How to handle off-policy data from Experience Replay.
            - "exgrpo_policy_shaping": Use f(x) = x/(x+β) shaping (default, recommended)
            - "dapo_clip_higher": Apply DAPO Clip-Higher to off-policy data too
        off_policy_shaping_beta (float): β for ExGRPO policy shaping. Default: 0.1

    Returns:
        dict: A dictionary containing:
            - pg_loss: Aggregated policy gradient loss
            - pg_losses: Per-token policy gradient losses
            - pg_clipfrac: Fraction of tokens that were clipped (upper bound)
            - pg_clipfrac_lower: Fraction of tokens clipped at lower bound
            - ppo_kl: Approximate KL divergence between old and new policies
            - entropy_bonus_tokens: Count of tokens where clipping was relaxed (A < 0)
            - on_pg_loss, off_pg_loss: Separate losses for on/off-policy data
    """
    # Compute importance sampling ratio
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    ratio = torch.exp(negative_approx_kl)  # π_new / π_old

    # Handle exp_mask: default to all on-policy if not provided
    if exp_mask is None:
        exp_mask = torch.zeros_like(response_mask)
    exp_mask = exp_mask.float()

    # ============================================================================
    # ON-POLICY: DAPO Clip-Higher (Decoupled asymmetric clipping)
    # ============================================================================
    # For A > 0 (encouraging actions): clip ratio to [1-ε_low, 1+ε_high]
    # For A < 0 (discouraging actions): clip ratio to [1-ε_low, ∞) - remove upper bound
    #   This allows low-probability tokens to be further reduced without constraint
    # ============================================================================
    
    # Standard PPO loss without clipping
    on_pg_losses1 = -advantages * ratio
    
    # Clipped PPO loss for positive advantages (A > 0)
    # Use standard clip: [1 - ε_low, 1 + ε_high]
    ratio_clipped_pos = torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    on_pg_losses_clipped_pos = -advantages * ratio_clipped_pos
    
    # Clipped PPO loss for negative advantages (A < 0)
    # DAPO: Remove upper clip bound - only clip lower bound
    # This is equivalent to: clip ratio to [1 - ε_low, +∞)
    # In practice, we use a very large upper bound (clip_ratio_c)
    ratio_clipped_neg = torch.clamp(ratio, 1 - cliprange_low, clip_ratio_c)
    on_pg_losses_clipped_neg = -advantages * ratio_clipped_neg
    
    # Select clipped loss based on advantage sign
    on_pg_losses_clipped = torch.where(advantages >= 0, on_pg_losses_clipped_pos, on_pg_losses_clipped_neg)
    
    # PPO-style max: take the more conservative (larger) loss
    on_pg_losses = torch.maximum(on_pg_losses1, on_pg_losses_clipped)
    
    # Additional safety clipping for extreme ratios (prevents gradient explosion)
    on_pg_losses_extreme = -advantages * clip_ratio_c
    on_pg_losses = torch.where(
        advantages < 0,
        torch.min(on_pg_losses, on_pg_losses_extreme),
        on_pg_losses
    )
    
    # Compute on-policy loss (only where exp_mask == 0)
    on_policy_mask = (1.0 - exp_mask) * response_mask
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_policy_mask)
    
    # ============================================================================
    # OFF-POLICY: Experience Replay handling
    # ============================================================================
    # Off-policy data from Experience Replay needs special handling due to
    # distribution shift. We use ExGRPO's policy shaping by default.
    # ============================================================================
    
    if off_policy_shaping_mode == "exgrpo_policy_shaping":
        # ⭐ ExGRPO Policy Shaping: Replace CLIP term with f(w*(θ)) = w*(θ) / (w*(θ) + β)
        # This amplifies low-probability signals and dampens high-probability ones
        off_ratio = ratio  # Same ratio, but with different shaping
        off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)
        off_pg_losses = -advantages * off_ratio_shaped
        off_pg_clipfrac = torch.tensor(0.0)
    elif off_policy_shaping_mode == "dapo_clip_higher":
        # Apply DAPO Clip-Higher to off-policy data too (same as on-policy)
        off_pg_losses = on_pg_losses
        off_pg_clipfrac = torch.tensor(0.0)
    else:
        raise ValueError(f"Invalid off_policy_shaping_mode: {off_policy_shaping_mode}")
    
    # Compute off-policy loss (only where exp_mask == 1)
    off_policy_mask = exp_mask * response_mask
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_policy_mask)
    off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss
    
    # ============================================================================
    # Combine on-policy and off-policy losses
    # ============================================================================
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # Compute clipping statistics (only for on-policy data)
    # Upper bound clipping: when clipped loss > unclipped loss (for A > 0)
    clipfrac_upper = verl_F.masked_mean(
        (torch.gt(on_pg_losses_clipped_pos, on_pg_losses1) & (advantages >= 0)).float(), 
        on_policy_mask
    )
    clipfrac_upper = torch.tensor(0.0) if clipfrac_upper.isnan().item() else clipfrac_upper
    
    # Lower bound clipping: when extreme clipping was applied (for A < 0)
    clipfrac_lower = verl_F.masked_mean(
        (torch.gt(on_pg_losses, on_pg_losses_extreme) & (advantages < 0)).float(), 
        on_policy_mask
    )
    clipfrac_lower = torch.tensor(0.0) if clipfrac_lower.isnan().item() else clipfrac_lower
    cliphit = torch.tensor(0.0, device=log_prob.device)
    try:
        cliphit = torch.clamp(clipfrac_upper + clipfrac_lower, 0.0, 1.0)
    except Exception:
        cliphit = torch.tensor(0.0, device=log_prob.device)
    
    # Count tokens where DAPO relaxed clipping (A < 0, allowing more freedom)
    entropy_bonus_tokens = verl_F.masked_mean((advantages < 0).float(), response_mask)

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac_upper,
        "pg_clipfrac_lower": clipfrac_lower,
        "pg_cliphit_rate": cliphit,
        "ppo_kl": ppo_kl,
        "entropy_bonus_tokens": entropy_bonus_tokens,
        # Separate on/off-policy losses for monitoring
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_clipfrac": clipfrac_upper,
        "on_pg_clipfrac_lower": clipfrac_lower,
    }


def dapo_filter_samples(
    rewards: torch.Tensor,
    group_ids: np.ndarray,
    n_rollout: int,
    filter_mode: str = "strict",
) -> torch.Tensor:
    """
    DAPO Dynamic Sampling: Filter samples where all rollouts have same outcome.
    
    This implements DAPO's Dynamic Sampling mechanism which filters out prompts
    where either:
    - All rollouts are correct (accuracy = 1.0) - too easy
    - All rollouts are incorrect (accuracy = 0.0) - too hard or no learning signal
    
    These prompts don't contribute useful gradients for learning because the
    advantage normalization within the group will result in zero variance.

    Args:
        rewards (Tensor): Reward tensor, shape (batch_size,) or (batch_size, seq_len)
            For outcome rewards, this is typically the final reward per trajectory.
        group_ids (np.ndarray): Group IDs (uid) for GRPO grouping, shape (batch_size,)
            Samples with the same group_id belong to the same prompt/task.
        n_rollout (int): Expected number of rollouts per prompt.
        filter_mode (str): Filtering mode:
            - "strict": Remove if all same (acc=0 or acc=1)
            - "remove_all_correct": Only remove if all correct (acc=1)
            - "remove_all_incorrect": Only remove if all incorrect (acc=0)
            - "none": No filtering

    Returns:
        torch.Tensor: Boolean mask indicating which samples to KEEP (True = keep, False = filter out)
            Shape: (batch_size,)
    """
    if filter_mode == "none":
        return torch.ones(len(rewards), dtype=torch.bool, device=rewards.device)
    
    # Get scalar rewards if needed
    if rewards.dim() > 1:
        rewards = rewards.sum(dim=-1)
    
    # Convert rewards to binary success/failure
    # Assume reward > 0 means success
    successes = (rewards > 0).float()
    
    # Group by group_id and compute success rate
    group_to_indices = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_to_indices[gid].append(i)
    
    keep_mask = torch.ones(len(rewards), dtype=torch.bool, device=rewards.device)
    
    for gid, indices in group_to_indices.items():
        if len(indices) == 0:
            continue
        
        group_successes = successes[indices]
        success_rate = group_successes.mean().item()
        
        should_filter = False
        
        if filter_mode == "strict":
            # Filter if all same (acc=0 or acc=1)
            should_filter = (success_rate == 0.0) or (success_rate == 1.0)
        elif filter_mode == "remove_all_correct":
            # Only remove if all correct
            should_filter = (success_rate == 1.0)
        elif filter_mode == "remove_all_incorrect":
            # Only remove if all incorrect
            should_filter = (success_rate == 0.0)
        
        if should_filter:
            for idx in indices:
                keep_mask[idx] = False
    
    return keep_mask


def dapo_overlong_reward_shaping(
    rewards: torch.Tensor,
    is_truncated: torch.Tensor,
    truncation_penalty: float = -0.5,
    soft_penalty_mode: str = "additive",
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DAPO Overlong Reward Shaping: Apply soft penalty to truncated samples.
    
    Instead of treating truncated samples as complete failures (reward=0),
    DAPO applies a soft penalty that:
    1. Preserves some learning signal from partially correct trajectories
    2. Discourages overly long responses that get truncated
    3. Reduces reward noise from arbitrary truncation points

    Args:
        rewards (Tensor): Original reward tensor, shape (batch_size,) or (batch_size, seq_len)
            For 2D tensors, reward is typically placed at the last valid token position.
        is_truncated (Tensor): Boolean tensor indicating which samples were truncated
            Shape: (batch_size,)
        truncation_penalty (float): Penalty for truncated samples. Default: -0.5
            - For "additive": Added to the reward
            - For "multiplicative": Multiplies the reward
            - For "replace_if_positive": Replaces positive rewards with this value
        soft_penalty_mode (str): How to apply the penalty:
            - "additive": reward = reward + penalty
            - "multiplicative": reward = reward * (1 + penalty)  [penalty < 0]
            - "replace_if_positive": If truncated and reward > 0, set reward = penalty
            - "cap": Cap positive rewards at penalty value
        response_mask (Tensor, optional): Response mask indicating valid response tokens.
            Shape: (batch_size, seq_len). If provided, when the original reward tensor is all-zeros
            (e.g., reward==0 for a trajectory), we will place the shaped penalty at the last valid
            response token instead of the last index (which may be padding).

    Returns:
        torch.Tensor: Modified rewards with overlong penalty applied
            Same shape as input rewards
    """
    modified_rewards = rewards.clone()
    
    # Handle both 1D and 2D reward tensors
    if rewards.dim() > 1:
        # For 2D reward tensors (batch_size, seq_len), the reward is typically placed
        # at the last valid token position. We should only apply penalty to the 
        # trajectory-level reward, not to every token.
        # 
        # ⭐ FIX: Sum rewards to get trajectory-level reward, apply penalty, then 
        # find the last non-zero position and apply the penalty there.
        # 
        # Alternative simpler approach: Apply penalty only to the last token position
        # where reward was originally placed.
        
        # Find positions where reward is non-zero (typically last valid token)
        reward_positions = (rewards != 0)
        
        for i in range(rewards.shape[0]):
            if not is_truncated[i]:
                continue
                
            # Get trajectory-level reward
            traj_reward = rewards[i].sum()
            
            # Apply penalty based on mode
            if soft_penalty_mode == "additive":
                new_traj_reward = traj_reward + truncation_penalty
            elif soft_penalty_mode == "multiplicative":
                new_traj_reward = traj_reward * (1 + truncation_penalty)
            elif soft_penalty_mode == "replace_if_positive":
                new_traj_reward = truncation_penalty if traj_reward > 0 else traj_reward
            elif soft_penalty_mode == "cap":
                new_traj_reward = min(traj_reward.item(), truncation_penalty) if traj_reward > truncation_penalty else traj_reward
            else:
                raise ValueError(f"Invalid soft_penalty_mode: {soft_penalty_mode}")
            
            # Find the position where reward was placed (last non-zero or last position)
            non_zero_positions = reward_positions[i].nonzero(as_tuple=True)[0]
            if len(non_zero_positions) > 0:
                # Reward was placed at a specific position
                reward_pos = non_zero_positions[-1]
                modified_rewards[i] = 0  # Clear all positions
                modified_rewards[i, reward_pos] = new_traj_reward
            else:
                # No reward was assigned (all zeros, e.g. reward==0). Place shaped reward at the last
                # valid response token if response_mask is available; otherwise fall back to last index.
                if response_mask is not None:
                    # last valid token index in the response
                    last_valid = response_mask[i].nonzero(as_tuple=True)[0]
                    if len(last_valid) > 0:
                        reward_pos = last_valid[-1]
                        modified_rewards[i] = 0
                        modified_rewards[i, reward_pos] = new_traj_reward
                    else:
                        # Degenerate: no valid response tokens. Fall back to last index,
                        # but still place the correctly *shaped* trajectory reward.
                        modified_rewards[i] = 0
                        modified_rewards[i, -1] = new_traj_reward
                else:
                    # No response_mask provided. Fall back to last index,
                    # but still place the correctly *shaped* trajectory reward.
                    modified_rewards[i] = 0
                    modified_rewards[i, -1] = new_traj_reward
    else:
        # 1D case: simple element-wise penalty application
        if soft_penalty_mode == "additive":
            modified_rewards = torch.where(
                is_truncated,
                rewards + truncation_penalty,
                rewards
            )
        elif soft_penalty_mode == "multiplicative":
            modified_rewards = torch.where(
                is_truncated,
                rewards * (1 + truncation_penalty),
                rewards
            )
        elif soft_penalty_mode == "replace_if_positive":
            modified_rewards = torch.where(
                is_truncated & (rewards > 0),
                torch.full_like(rewards, truncation_penalty),
                rewards
            )
        elif soft_penalty_mode == "cap":
            modified_rewards = torch.where(
                is_truncated & (rewards > truncation_penalty),
                torch.full_like(rewards, truncation_penalty),
                rewards
            )
        else:
            raise ValueError(f"Invalid soft_penalty_mode: {soft_penalty_mode}")
    
    return modified_rewards


def bam_compute_token_on_off_policy_loss_v3(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    exp_mask,   # (bs, response_length) ANNI add: 1 indicates off-policy data; 0 indicates on-policy data
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Computes the on-policy and off-policy losses for reinforcement learning, using various methods to handle clipping and masking.

    Args:
        old_log_prob (Tensor): The log probability of the old policy.
        log_prob (Tensor): The log probability of the new policy.
        advantages (Tensor): The advantage values.
        response_mask (Tensor): A mask indicating which tokens are part of the response.
        exp_mask (Tensor): A mask indicating whether the data is from an off-policy (1) or on-policy (0) source.
        cliprange (float, optional): The range for clipping the ratio.
        cliprange_low (float, optional): The lower bound for clipping the ratio.
        cliprange_high (float, optional): The upper bound for clipping the ratio.
        clip_ratio_c (float, optional): The constant used for clipping the ratio.
        loss_agg_mode (str, optional): The mode for aggregating the loss. Default is "token-mean".

    Returns:
        dict: A dictionary containing the computed losses and other relevant metrics.
    """
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # on-policy: no changes
    # off-policy: cliphigh=1, rest kept consistent with on-policy

    # on-policy: keep unchanged
    ratio = torch.exp(negative_approx_kl)   # (bs, response_length) ⭐ Compute the ratio of new to old policy probabilities
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    on_clip_pg_losses1 = torch.maximum(on_pg_losses1, on_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    on_clip_pg_losses2 = torch.min(on_pg_losses3, on_clip_pg_losses1)
    on_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(on_clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask)

    on_pg_losses = torch.where(advantages < 0, on_clip_pg_losses2, on_clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0-exp_mask) * response_mask)

    # off-policy
    off_pg_losses1 = -advantages * ratio
    off_cliprange_low = cliprange_low
    off_cliprange_high = 1.0
    off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - off_cliprange_low, 1 + off_cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    off_clip_pg_losses1 = torch.maximum(off_pg_losses1, off_pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    off_pg_clipfrac = verl_F.masked_mean(torch.gt(off_pg_losses2, off_pg_losses1).float(), response_mask)
    off_pg_losses3 = -advantages * clip_ratio_c
    off_clip_pg_losses2 = torch.min(off_pg_losses3, off_clip_pg_losses1)
    off_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(off_clip_pg_losses1, off_pg_losses3) * (advantages < 0).float(), response_mask)

    off_pg_losses = torch.where(advantages < 0, off_clip_pg_losses2, off_clip_pg_losses1)
    off_pg_loss = verl_F.masked_mean(off_pg_losses, (1.0-exp_mask) * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)

    exp_mask = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ret_dict = {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_losses":  on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_clipfrac": on_pg_clipfrac,
        "on_pg_clipfrac_lower": on_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
    }

    return ret_dict


# ==============================================================================
# RePO (Replay-Enhanced Policy Optimization) Advantage Computation
# ==============================================================================

def repo_compute_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exp_mask: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
    advantage_mode: str = "separate",
) -> torch.Tensor:
    """
    Compute GRPO advantages with RePO's separate baseline for on/off-policy.
    
    ⭐ Key difference from standard GRPO:
    - Standard GRPO: All samples use the same baseline (mean of all rewards)
    - RePO (separate): On-policy uses on-policy baseline, off-policy uses all-samples baseline
    
    From RePO paper Section 3.2:
    - For on-policy samples: A_i = (R_i - mean(R_on)) / std(R_on)
    - For off-policy samples: A_j = (R_j - mean(R_all)) / std(R_all)
    
    Args:
        token_level_rewards: Token-level rewards, shape (bs, response_len)
            Typically reward is placed at the last valid token position
        response_mask: Response mask, shape (bs, response_len)
        index: Group indices for GRPO grouping, shape (bs,)
            Samples with the same index belong to the same prompt/task
        exp_mask: Experience mask, shape (bs, response_len) or (bs,)
            1 = off-policy, 0 = on-policy
        epsilon: Small value for numerical stability
        norm_adv_by_std: Whether to normalize advantage by std
            - True: A = (R - mean) / std (standard GRPO)
            - False: A = R - mean (Dr.GRPO style)
        advantage_mode: How to compute advantages
            - "separate": RePO style, on/off use different baselines (recommended)
            - "unified": ExGRPO style, all samples use same baseline
            
    Returns:
        advantages: Token-level advantages, shape (bs, response_len)
    """
    # Get sequence-level rewards (sum over response tokens)
    # Typically reward is placed at the last valid token, so sum works
    seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (bs,)
    
    # Get sequence-level exp_mask
    # If exp_mask is (bs, response_len), take max to get (bs,)
    if exp_mask.dim() > 1:
        seq_exp_mask = exp_mask.max(dim=-1).values  # (bs,)
    else:
        seq_exp_mask = exp_mask  # Already (bs,)
    
    # Convert to boolean masks
    is_offpolicy = seq_exp_mask > 0.5  # (bs,) bool
    is_onpolicy = ~is_offpolicy  # (bs,) bool
    
    # Group by index (task/prompt)
    unique_indices = np.unique(index)
    
    # Initialize advantages
    advantages = torch.zeros_like(token_level_rewards)
    
    for idx in unique_indices:
        # Get samples belonging to this group
        group_mask = (index == idx)  # (bs,) bool numpy
        group_mask_tensor = torch.tensor(group_mask, device=token_level_rewards.device)
        
        # Get rewards for this group
        group_rewards = seq_rewards[group_mask]  # (n_group,)
        group_is_offpolicy = is_offpolicy[group_mask]  # (n_group,) bool
        group_is_onpolicy = is_onpolicy[group_mask]  # (n_group,) bool
        
        if len(group_rewards) == 0:
            continue
        
        if advantage_mode == "unified":
            # ========== Unified mode (ExGRPO style) ==========
            # All samples use the same baseline
            mean_all = group_rewards.mean()
            std_all = group_rewards.std() + epsilon
            
            if norm_adv_by_std:
                group_advantages = (group_rewards - mean_all) / std_all
            else:
                group_advantages = group_rewards - mean_all
        
        elif advantage_mode == "separate":
            # ========== Separate mode (RePO style) ==========
            # On-policy: use on-policy baseline
            # Off-policy: use all-samples baseline
            
            # Compute on-policy statistics
            on_rewards = group_rewards[group_is_onpolicy]
            if len(on_rewards) > 0:
                mean_on = on_rewards.mean()
                std_on = on_rewards.std() + epsilon
            else:
                # No on-policy samples, fallback to all
                mean_on = group_rewards.mean()
                std_on = group_rewards.std() + epsilon
            
            # Compute all-samples statistics
            mean_all = group_rewards.mean()
            std_all = group_rewards.std() + epsilon
            
            # Compute advantages
            group_advantages = torch.zeros_like(group_rewards)
            
            # On-policy: use on-policy baseline
            if norm_adv_by_std:
                group_advantages[group_is_onpolicy] = (
                    (group_rewards[group_is_onpolicy] - mean_on) / std_on
                )
            else:
                group_advantages[group_is_onpolicy] = group_rewards[group_is_onpolicy] - mean_on
            
            # Off-policy: use all-samples baseline
            if norm_adv_by_std:
                group_advantages[group_is_offpolicy] = (
                    (group_rewards[group_is_offpolicy] - mean_all) / std_all
                )
            else:
                group_advantages[group_is_offpolicy] = group_rewards[group_is_offpolicy] - mean_all
        
        else:
            raise ValueError(f"Unknown advantage_mode: {advantage_mode}. Use 'separate' or 'unified'.")
        
        # Broadcast sequence-level advantages to token level
        # advantages[group_mask] = group_advantages (broadcasted)
        group_indices = torch.where(group_mask_tensor)[0]
        for i, gi in enumerate(group_indices):
            # Set advantage for all response tokens in this sequence
            advantages[gi] = group_advantages[i]
    
    return advantages


# ==============================================================================
# CHORD (Controllable Harmonization of On- and Off-Policy RL via Dynamic Weighting)
# ==============================================================================

def chord_mu_scheduler(
    training_progress: float,
    mu_max: float = 0.5,
    lambda_decay: float = 2.0,
) -> float:
    """
    CHORD 全局系数 μ(t) 调度器。
    
    公式：μ(t) = μ_max * (1 - t/T)^λ
    
    Args:
        training_progress: 训练进度 t/T，范围 [0, 1]
        mu_max: 初始 SFT 权重（论文默认 0.5）
        lambda_decay: 衰减指数（论文默认 2.0）
        
    Returns:
        float: 当前的 μ 值
        
    Example:
        >>> chord_mu_scheduler(0.0, mu_max=0.5, lambda_decay=2.0)
        0.5  # 训练开始，μ = μ_max
        >>> chord_mu_scheduler(0.5, mu_max=0.5, lambda_decay=2.0)
        0.125  # 训练中期，μ = 0.5 * 0.5^2 = 0.125
        >>> chord_mu_scheduler(1.0, mu_max=0.5, lambda_decay=2.0)
        0.0  # 训练结束，μ = 0
    """
    # 确保 training_progress 在 [0, 1] 范围内
    training_progress = max(0.0, min(1.0, training_progress))
    
    # μ(t) = μ_max * (1 - t/T)^λ
    mu = mu_max * ((1.0 - training_progress) ** lambda_decay)
    
    return mu


def compute_chord_token_weights(
    log_prob: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    计算 CHORD 的 token-wise 加权 φ(d)。
    
    公式：φ(d) = d / (d + δ)
    其中 d = exp(log_prob) = π_θ(y_i | x, y_{<i})
    
    Args:
        log_prob: 当前策略对 expert token 的 log 概率，shape (bs, seq_len)
        delta: 平滑参数（论文默认 0.1）
        
    Returns:
        torch.Tensor: Token-wise 权重 φ(d)，shape (bs, seq_len)
        
    Example:
        >>> log_prob = torch.tensor([[-0.1, -1.0, -5.0]])  # 概率: [0.90, 0.37, 0.007]
        >>> compute_chord_token_weights(log_prob, delta=0.1)
        tensor([[0.900, 0.787, 0.065]])  # 高概率 token 权重高，低概率 token 权重低
    """
    # d = exp(log_prob) = π_θ(y_i | x, y_{<i})
    # Clamp log_prob to avoid numerical issues
    d = torch.exp(log_prob.clamp(max=0))
    
    # φ(d) = d / (d + δ)
    phi = d / (d + delta)
    
    return phi


def compute_chord_sft_loss(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    exp_mask: torch.Tensor,
    delta: float = 0.1,
    use_token_weighting: bool = True,
    loss_agg_mode: str = "token-mean",
) -> dict:
    """
    计算 CHORD 的加权 SFT Loss。
    
    公式：L_sft = -∑_i φ(d_i) * log π_θ(y_i | x, y_{<i})
    
    ⭐ 关键设计：
    1. 只对 expert 数据 (exp_mask=1) 计算 SFT loss
    2. 使用 φ(d) 加权，避免强制学习已掌握或无法学习的 token
    3. 与 GRPO loss 分开计算，通过 μ(t) 动态加权合并
    
    Args:
        log_prob: 当前策略的 log 概率，shape (bs, seq_len)
        response_mask: Response 部分的 mask，shape (bs, seq_len)
        exp_mask: Expert 数据的 mask，shape (bs, seq_len)，1=expert, 0=on-policy
        delta: φ(d) 的平滑参数
        use_token_weighting: 是否使用 token-wise 加权
        loss_agg_mode: Loss 聚合模式
        
    Returns:
        dict: 包含 sft_loss 和诊断指标
    """
    # 确保 exp_mask 是 float
    exp_mask = exp_mask.float()
    
    # Expert token 的 mask
    expert_mask = exp_mask * response_mask
    
    # 计算 SFT loss: -log_prob
    sft_losses = -log_prob
    
    if use_token_weighting:
        # 计算 token-wise 权重 φ(d)
        phi = compute_chord_token_weights(log_prob, delta=delta)
        
        # 加权 SFT loss
        weighted_sft_losses = phi * sft_losses
    else:
        # 不加权，使用原始 SFT loss
        weighted_sft_losses = sft_losses
        phi = torch.ones_like(sft_losses)
    
    # 聚合 loss（只对 expert token）
    sft_loss = agg_loss(
        loss_mat=weighted_sft_losses,
        loss_mask=expert_mask,
        loss_agg_mode=loss_agg_mode,
    )
    
    # 处理 NaN（当没有 expert 数据时）
    if sft_loss.isnan().item():
        sft_loss = torch.tensor(0.0, device=log_prob.device)
    
    # 诊断指标
    chord_diag = {}
    with torch.no_grad():
        # Expert token 数量
        n_expert_tokens = expert_mask.sum().item()
        chord_diag["n_expert_tokens"] = n_expert_tokens
        
        # φ(d) 统计
        if n_expert_tokens > 0:
            phi_expert = phi[expert_mask.bool()]
            chord_diag["phi_mean"] = phi_expert.mean().item()
            chord_diag["phi_std"] = phi_expert.std().item() if phi_expert.numel() > 1 else 0.0
            chord_diag["phi_min"] = phi_expert.min().item()
            chord_diag["phi_max"] = phi_expert.max().item()
            
            # log_prob 统计
            log_prob_expert = log_prob[expert_mask.bool()]
            chord_diag["log_prob_mean"] = log_prob_expert.mean().item()
            chord_diag["log_prob_std"] = log_prob_expert.std().item() if log_prob_expert.numel() > 1 else 0.0
            
            # 原始 SFT loss（未加权）统计
            sft_losses_expert = sft_losses[expert_mask.bool()]
            chord_diag["sft_loss_unweighted_mean"] = sft_losses_expert.mean().item()
        else:
            chord_diag["phi_mean"] = 0.0
            chord_diag["phi_std"] = 0.0
            chord_diag["phi_min"] = 0.0
            chord_diag["phi_max"] = 0.0
            chord_diag["log_prob_mean"] = 0.0
            chord_diag["log_prob_std"] = 0.0
            chord_diag["sft_loss_unweighted_mean"] = 0.0
    
    return {
        "sft_loss": sft_loss,
        "weighted_sft_losses": weighted_sft_losses,
        "chord_diag": chord_diag,
    }


def repo_compute_token_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    exp_mask: torch.Tensor,
    cliprange: float = 0.2,
    clip_eps: float = 0.2,
    use_importance_clipping: bool = True,
    loss_agg_mode: str = "token-mean",
) -> dict:
    """
    Compute RePO policy loss with importance ratio clipping for off-policy data.
    
    ⭐ Key features:
    1. On-policy: Standard PPO clipped objective
    2. Off-policy: Importance ratio + clip to handle distribution shift
    
    From RePO paper:
    - ratio = π_θ(o|q) / π_θ_old(o|q)
    - L_off = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    
    Args:
        old_log_prob: Log probs from behavior policy (stored), shape (bs, seq_len)
        log_prob: Log probs from current policy, shape (bs, seq_len)
        advantages: Advantages (from repo_compute_grpo_advantage), shape (bs, seq_len)
        response_mask: Response mask, shape (bs, seq_len)
        exp_mask: Experience mask (1=off-policy, 0=on-policy), shape (bs, seq_len)
        cliprange: Clip range for on-policy PPO
        clip_eps: Clip epsilon for off-policy importance ratio
        use_importance_clipping: Whether to clip importance ratio for off-policy
        loss_agg_mode: Loss aggregation mode
        
    Returns:
        dict with pg_loss, on_pg_loss, off_pg_loss, and diagnostics
    """
    # Compute importance ratio
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    # ========== On-policy loss (standard PPO) ==========
    on_pg_losses1 = -advantages * ratio
    on_pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    on_pg_losses = torch.maximum(on_pg_losses1, on_pg_losses2)
    
    on_policy_mask = (1.0 - exp_mask) * response_mask
    on_pg_loss = verl_F.masked_mean(on_pg_losses, on_policy_mask)
    
    # Clip fraction for on-policy
    on_clipfrac = verl_F.masked_mean(
        torch.gt(on_pg_losses2, on_pg_losses1).float(), 
        on_policy_mask
    )
    
    # ========== Off-policy loss (with importance ratio clipping) ==========
    if use_importance_clipping:
        # PPO-style clipped objective for off-policy
        off_pg_losses1 = -advantages * ratio
        off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        off_pg_losses = torch.maximum(off_pg_losses1, off_pg_losses2)
        
        off_clipfrac = verl_F.masked_mean(
            torch.gt(off_pg_losses2, off_pg_losses1).float(),
            exp_mask * response_mask
        )
    else:
        # No clipping, just use ratio directly
        off_pg_losses = -advantages * ratio
        off_clipfrac = torch.tensor(0.0, device=log_prob.device)
    
    off_policy_mask = exp_mask * response_mask
    off_pg_loss = verl_F.masked_mean(off_pg_losses, off_policy_mask)
    off_pg_loss = torch.tensor(0.0, device=log_prob.device) if off_pg_loss.isnan().item() else off_pg_loss
    
    # ========== Combine losses ==========
    exp_mask_float = exp_mask.float()
    pg_losses = off_pg_losses * exp_mask_float + on_pg_losses * (1.0 - exp_mask_float)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # ========== Diagnostics ==========
    def _masked_stats(x: torch.Tensor, m: torch.Tensor, prefix: str) -> dict:
        out = {}
        mask = m.bool()
        if mask.sum() == 0:
            out[f"{prefix}/count"] = torch.tensor(0.0, device=x.device)
            return out
        vals = x[mask]
        out[f"{prefix}/count"] = torch.tensor(float(vals.numel()), device=x.device)
        out[f"{prefix}/mean"] = vals.mean()
        out[f"{prefix}/std"] = vals.std() if vals.numel() > 1 else torch.tensor(0.0, device=x.device)
        out[f"{prefix}/min"] = vals.min()
        out[f"{prefix}/max"] = vals.max()
        return out
    
    diag_stats = {}
    diag_stats.update(_masked_stats(ratio, on_policy_mask, "ratio/on"))
    diag_stats.update(_masked_stats(ratio, off_policy_mask, "ratio/off"))
    diag_stats.update(_masked_stats(advantages, on_policy_mask, "adv/on"))
    diag_stats.update(_masked_stats(advantages, off_policy_mask, "adv/off"))
    
    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "on_pg_loss": on_pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_losses": on_pg_losses,
        "off_pg_losses": off_pg_losses,
        "on_pg_clipfrac": on_clipfrac,
        "off_pg_clipfrac": off_clipfrac,
        "ppo_kl": ppo_kl,
        "repo_diag_stats": diag_stats,
    }