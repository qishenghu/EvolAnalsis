import numpy as np
import torch
from collections import defaultdict

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

    def compute_pg_losses(cliprange_low, cliprange_high):
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
        clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)
        return pg_losses, clipfrac, clipfrac_lower

    # On-policy calculations
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    on_pg_losses, on_pg_clipfrac, on_pg_clipfrac_lower = compute_pg_losses(cliprange_low, cliprange_high)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)  # ⭐ Compute the on-policy loss

    # Off-policy calculations
    if off_policy_shaping_mode == "exgrpo_policy_shaping":
        # ⭐ ExGRPO Policy Shaping: Replace CLIP term with f(w*(θ)) = w*(θ) / (w*(θ) + β)
        # where w*(θ) = exp(log_prob - old_log_prob) is the importance sampling ratio
        # This amplifies low-probability signals and dampens high-probability ones
        negative_approx_kl_off = log_prob - old_log_prob
        off_ratio = torch.exp(negative_approx_kl_off)  # w*(θ) = π_new / π_old
        
        # Apply policy shaping: f(x) = x / (x + β)
        off_ratio_shaped = off_ratio / (off_ratio + off_policy_shaping_beta)
        
        # Replace CLIP term with shaped ratio: -advantages * f(w*(θ))
        off_pg_losses = -advantages * off_ratio_shaped
        
        # No clipping for off-policy when using policy shaping
        off_pg_clipfrac = torch.tensor(0.0)
        off_pg_clipfrac_lower = torch.tensor(0.0)
        
        # Compute off-policy loss (only on LLM response tokens in multi-turn)
        off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
        off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss
        
    elif off_policy_shaping_mode == "higher_clip_bound":
        # ⭐ AgentEvolver original: Use higher clip_upper_bound for off-policy data
        off_cliprange_low = cliprange_low
        off_pg_losses, off_pg_clipfrac, off_pg_clipfrac_lower = compute_pg_losses(off_cliprange_low, off_cliprange_high)
        off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)  # ⭐ Compute the off-policy loss
        off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss
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
        "ppo_kl": ppo_kl,
    }



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
    
    # Count tokens where DAPO relaxed clipping (A < 0, allowing more freedom)
    entropy_bonus_tokens = verl_F.masked_mean((advantages < 0).float(), response_mask)

    return {
        "pg_loss": pg_loss,
        "pg_losses": pg_losses,
        "pg_clipfrac": clipfrac_upper,
        "pg_clipfrac_lower": clipfrac_lower,
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
                # No reward was assigned (all zeros), assign penalty to last position
                modified_rewards[i, -1] = truncation_penalty if soft_penalty_mode == "additive" else 0
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