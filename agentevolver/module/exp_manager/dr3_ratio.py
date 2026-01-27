"""
DR³ (Density-Ratio-Repair) utilities.

Goal:
- Estimate relative density ratio w_alpha = pi_theta(y|x) / m_alpha(y|x)
  where m_alpha = (1-alpha) pi_theta + alpha q (q is unknown behavior/offline distribution).
- Do NOT require teacher logits / teacher tokenizer alignment.

Implementation strategy (practical, low-overhead):
- Train a small discriminator D_psi(x,y) to classify samples as
  on-policy (label=1) vs off-policy/teacher (label=0) within each update step.
- Under mixture prior P(z=1)=1-alpha, Bayes optimal discriminator satisfies:
    D*(x,y) = (1-alpha) * w_alpha(x,y)
  => w_alpha = D*/(1-alpha)

We use sequence-level features derived from tensors already available in actor update:
- masked mean/std/min of log_prob
- masked mean abs(advantage)
- response length

This module is intentionally lightweight and fully self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


def _masked_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = m.float()
    denom = m.sum(dim=-1).clamp_min(eps)
    return (x * m).sum(dim=-1) / denom


def _masked_std(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = _masked_mean(x, m, eps=eps)
    m = m.float()
    denom = m.sum(dim=-1).clamp_min(eps)
    var = (m * (x - mu.unsqueeze(-1)) ** 2).sum(dim=-1) / denom
    return torch.sqrt(var.clamp_min(0.0))


def _masked_min(x: torch.Tensor, m: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    # If mask is all-zero for a row, return fill.
    # We use +inf outside mask then take min.
    m_bool = m.bool()
    if x.dim() != 2 or m_bool.dim() != 2:
        raise ValueError("Expected 2D tensors for _masked_min")
    x2 = x.clone()
    x2[~m_bool] = float("inf")
    v, _ = x2.min(dim=-1)
    v = torch.where(torch.isfinite(v), v, torch.full_like(v, float(fill)))
    return v


def _masked_max(x: torch.Tensor, m: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    # If mask is all-zero for a row, return fill.
    # We use -inf outside mask then take max.
    m_bool = m.bool()
    if x.dim() != 2 or m_bool.dim() != 2:
        raise ValueError("Expected 2D tensors for _masked_max")
    x2 = x.clone()
    x2[~m_bool] = float("-inf")
    v, _ = x2.max(dim=-1)
    v = torch.where(torch.isfinite(v), v, torch.full_like(v, float(fill)))
    return v


def compute_sequence_features(
    *,
    log_prob: torch.Tensor,        # (bs, T)
    advantages: torch.Tensor,      # (bs, T)
    response_mask: torch.Tensor,   # (bs, T)
    ref_log_prob: Optional[torch.Tensor] = None,  # (bs, T) optional
    feature_mode: str = "v2",      # "v1" | "v2"
    extra_seq_features: Optional[torch.Tensor] = None,  # (bs, K) optional
) -> torch.Tensor:
    """
    Build a compact per-sequence feature vector from already-available tensors.
    Returns: (bs, F)
    """
    with torch.no_grad():
        m = response_mask
        feature_mode = str(feature_mode).lower().strip()
        resp_len = m.float().sum(dim=-1)

        # v1: original minimal feature set (5 dims)
        if feature_mode in ("v1", "min", "minimal"):
            lp_mean = _masked_mean(log_prob, m)
            lp_std = _masked_std(log_prob, m)
            lp_min = _masked_min(log_prob, m, fill=0.0)
            adv_abs_mean = _masked_mean(advantages.abs(), m)
            feats = torch.stack([lp_mean, lp_std, lp_min, adv_abs_mean, resp_len], dim=-1)
        elif feature_mode in ("v3", "v2_noadv", "noadv", "no_adv"):
            # v3: advantage-free features (closer to density-ratio objective; avoids reward/adv leakage)
            lp_mean = _masked_mean(log_prob, m)
            lp_std = _masked_std(log_prob, m)
            lp_min = _masked_min(log_prob, m, fill=0.0)
            lp_max = _masked_max(log_prob, m, fill=0.0)

            m_bool = m.bool()
            denom = m_bool.sum(dim=-1).float().clamp_min(1.0)
            low_thr = -10.0
            lp_low_ratio = ((log_prob < low_thr) & m_bool).sum(dim=-1).float() / denom

            if ref_log_prob is not None and torch.is_tensor(ref_log_prob) and ref_log_prob.shape == log_prob.shape:
                kl_ref_mean = _masked_mean((log_prob - ref_log_prob), m)
            else:
                kl_ref_mean = torch.zeros_like(lp_mean)

            feats = torch.stack(
                [
                    lp_mean,
                    lp_std,
                    lp_min,
                    lp_max,
                    lp_low_ratio,
                    resp_len,
                    kl_ref_mean,
                ],
                dim=-1,
            )
        else:
            # v2: still cheap, but adds tail/shape signals and optional KL-to-ref
            lp_mean = _masked_mean(log_prob, m)
            lp_std = _masked_std(log_prob, m)
            lp_min = _masked_min(log_prob, m, fill=0.0)
            lp_max = _masked_max(log_prob, m, fill=0.0)

            # fraction of very-low-prob tokens (captures mismatch / long-tail)
            m_bool = m.bool()
            denom = m_bool.sum(dim=-1).float().clamp_min(1.0)
            low_thr = -10.0
            lp_low_ratio = ((log_prob < low_thr) & m_bool).sum(dim=-1).float() / denom

            adv_mean = _masked_mean(advantages, m)
            adv_std = _masked_std(advantages, m)
            adv_abs_mean = _masked_mean(advantages.abs(), m)
            adv_pos_ratio = ((advantages > 0) & m_bool).sum(dim=-1).float() / denom

            # KL-to-ref proxy: mean(logp - ref_logp) on response tokens if available
            if ref_log_prob is not None and torch.is_tensor(ref_log_prob) and ref_log_prob.shape == log_prob.shape:
                kl_ref_mean = _masked_mean((log_prob - ref_log_prob), m)
            else:
                kl_ref_mean = torch.zeros_like(lp_mean)

            feats = torch.stack(
                [
                    lp_mean,
                    lp_std,
                    lp_min,
                    lp_max,
                    lp_low_ratio,
                    adv_mean,
                    adv_std,
                    adv_abs_mean,
                    adv_pos_ratio,
                    resp_len,
                    kl_ref_mean,
                ],
                dim=-1,
            )
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if extra_seq_features is not None and torch.is_tensor(extra_seq_features):
            try:
                extra = extra_seq_features.detach().float()
                if extra.dim() == 1:
                    extra = extra.view(-1, 1)
                if extra.dim() == 2 and extra.shape[0] == feats.shape[0]:
                    extra = torch.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
                    feats = torch.cat([feats, extra], dim=-1)
            except Exception:
                pass
        return feats


class DR3Discriminator(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden: int = 64,
        stats_dim: int = 0,
        hidden_proj_dim: int = 0,
        hidden_proj_dropout: float = 0.0,
    ):
        super().__init__()
        h = int(hidden)
        self.in_dim = int(in_dim)
        self.stats_dim = int(stats_dim)
        self.hidden_proj_dim = int(hidden_proj_dim)
        self.hidden_proj_dropout = float(hidden_proj_dropout)

        # Optional projection for high-dimensional hidden features:
        # input x is assumed to be [stats (first stats_dim dims), hidden (remaining dims)].
        # If hidden_proj_dim>0, we map hidden -> hidden_proj_dim before feeding MLP.
        self.hidden_proj: Optional[nn.Module] = None
        self._use_hidden_proj = False
        if self.hidden_proj_dim > 0 and self.stats_dim > 0 and self.stats_dim < self.in_dim:
            hidden_in = int(self.in_dim - self.stats_dim)
            self.hidden_proj = nn.Sequential(
                nn.Linear(hidden_in, int(self.hidden_proj_dim)),
                nn.LayerNorm(int(self.hidden_proj_dim)),
                nn.Dropout(float(self.hidden_proj_dropout)) if float(self.hidden_proj_dropout) > 0 else nn.Identity(),
            )
            self._use_hidden_proj = True

        mlp_in = int(self.in_dim)
        if self._use_hidden_proj:
            mlp_in = int(self.stats_dim + self.hidden_proj_dim)

        self.net = nn.Sequential(
            nn.Linear(mlp_in, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_hidden_proj and self.hidden_proj is not None:
            try:
                stats = x[:, : self.stats_dim]
                hid = x[:, self.stats_dim :]
                hid2 = self.hidden_proj(hid)
                x = torch.cat([stats, hid2], dim=-1)
            except Exception:
                # fall back to raw input
                pass
        return self.net(x).squeeze(-1)  # (bs,)


@dataclass
class DR3DualESS:
    enable: bool = True
    ess_target_ratio: float = 0.5  # κ
    lr: float = 0.05
    lam: float = 0.0

    def update(self, *, ess: float, n: int) -> float:
        if (not self.enable) or (n <= 0):
            return float(self.lam)
        target = float(self.ess_target_ratio) * float(n)
        self.lam = float(max(0.0, self.lam + float(self.lr) * (target - float(ess))))
        return float(self.lam)

    def clip_upper(self, *, base_upper: float, clip_max: float) -> float:
        """
        Map dual variable to a conservative clip upper bound.
        - lam increases when ESS is below target -> stronger clipping (smaller upper bound)
        """
        cu = float(min(base_upper, float(clip_max)))
        if not self.enable:
            return cu
        # monotone decreasing in lam; bounded away from 0 by a small epsilon in caller
        return cu / (1.0 + float(self.lam))


class DR3RatioEstimator:
    """
    Stateful helper owned by each actor rank.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        lr: float = 5e-4,
        disc_steps_per_call: int = 1,
        buffer_size: int = 2048,
        train_batch_size: int = 128,
        ess_window: int = 32,
        sync_across_ranks: bool = False,
        sync_every_n_calls: int = 1,
        broadcast_params: bool = False,
        broadcast_every_n_calls: int = 1,
        clip_max: float = 10.0,
        dual_enable: bool = True,
        ess_target_ratio: float = 0.5,
        dual_lr: float = 0.05,
        dual_init: float = 0.0,
        disc_weight_decay: float = 0.0,
        disc_stats_dim: int = 0,
        disc_hidden_proj_dim: int = 0,
        disc_hidden_proj_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.disc_weight_decay = float(disc_weight_decay)
        self.disc_stats_dim = int(disc_stats_dim)
        self.disc_hidden_proj_dim = int(disc_hidden_proj_dim)
        self.disc_hidden_proj_dropout = float(disc_hidden_proj_dropout)
        self.disc_steps_per_call = max(0, int(disc_steps_per_call))
        self.clip_max = float(clip_max)
        self.eps = float(eps)

        self._disc: Optional[DR3Discriminator] = None
        self._opt: Optional[torch.optim.Optimizer] = None
        self._bce = nn.BCEWithLogitsLoss()

        self.dual = DR3DualESS(enable=bool(dual_enable), ess_target_ratio=float(ess_target_ratio), lr=float(dual_lr), lam=float(dual_init))

        # ------------------------------------------------------------------
        # ⭐ Make DR³ robust for ppo_micro_batch_size_per_gpu=1
        # ------------------------------------------------------------------
        # We maintain:
        # - a rolling feature/label buffer for discriminator training across micro-batches
        # - a rolling window of off-policy weights for ESS/dual update (so ESS != 1 forever)
        self.buffer_size = max(32, int(buffer_size))
        self.train_batch_size = max(8, int(train_batch_size))
        self.ess_window = max(1, int(ess_window))
        self.sync_across_ranks = bool(sync_across_ranks)
        self.sync_every_n_calls = max(1, int(sync_every_n_calls))
        self._calls: int = 0
        self.broadcast_params = bool(broadcast_params)
        self.broadcast_every_n_calls = max(1, int(broadcast_every_n_calls))

        self._buf_x: Optional[torch.Tensor] = None  # (N,F) on device
        self._buf_y: Optional[torch.Tensor] = None  # (N,) labels_on in {0,1} on device
        self._w_off_hist: list[torch.Tensor] = []   # list of 1D CPU tensors
        self._alpha_ema: Optional[float] = None

    @staticmethod
    def _clip_alpha(alpha: float, eps: float) -> float:
        # Avoid extreme alpha causing 1/(1-alpha) blow-ups and unstable weights.
        return float(min(max(float(alpha), float(eps)), 1.0 - float(eps)))

    def _estimate_alpha_from_labels(self, labels_on: torch.Tensor) -> Optional[float]:
        """
        labels_on: (n,) float tensor in {0,1} (1=on, 0=off/teacher)
        returns off_ratio in [0,1] or None if empty
        """
        try:
            if labels_on is None or (not torch.is_tensor(labels_on)) or labels_on.numel() <= 0:
                return None
            off_cnt = float((1.0 - labels_on.float()).sum().item())
            tot = float(labels_on.numel())
            if tot <= 0:
                return None
            return off_cnt / tot
        except Exception:
            return None

    def _update_alpha_ema(self, alpha_raw: float, beta: float) -> float:
        beta = float(min(max(beta, 0.0), 0.999))
        a = float(alpha_raw)
        if self._alpha_ema is None:
            self._alpha_ema = a
        else:
            self._alpha_ema = float(beta) * float(self._alpha_ema) + (1.0 - float(beta)) * a
        return float(self._alpha_ema)

    def _maybe_init(self, *, device: torch.device, in_dim: int) -> None:
        if self._disc is not None:
            return
        self._disc = DR3Discriminator(
            in_dim=in_dim,
            hidden=self.hidden,
            stats_dim=self.disc_stats_dim,
            hidden_proj_dim=self.disc_hidden_proj_dim,
            hidden_proj_dropout=self.disc_hidden_proj_dropout,
        ).to(device)
        self._opt = torch.optim.Adam(self._disc.parameters(), lr=self.lr, weight_decay=float(self.disc_weight_decay))
        self._buf_x = torch.empty((0, in_dim), device=device, dtype=torch.float32)
        self._buf_y = torch.empty((0,), device=device, dtype=torch.float32)
        # Optional: ensure all ranks start from identical discriminator params
        if self.broadcast_params:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    for p in self._disc.parameters():
                        dist.broadcast(p.data, src=0)
            except Exception:
                pass

    def _get_rank_world(self) -> tuple[int, int]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank()), int(dist.get_world_size())
        except Exception:
            pass
        return 0, 1

    def _broadcast_model_params(self) -> None:
        if not self.broadcast_params or self._disc is None:
            return
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                for p in self._disc.parameters():
                    dist.broadcast(p.data, src=0)
        except Exception:
            pass

    def _push_buffer(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self._buf_x is None or self._buf_y is None:
            return
        x = x.detach().float()
        y = y.detach().float().view(-1)
        if x.numel() == 0 or y.numel() == 0:
            return
        self._buf_x = torch.cat([self._buf_x, x], dim=0)
        self._buf_y = torch.cat([self._buf_y, y], dim=0)
        if self._buf_x.shape[0] > self.buffer_size:
            extra = int(self._buf_x.shape[0] - self.buffer_size)
            self._buf_x = self._buf_x[extra:]
            self._buf_y = self._buf_y[extra:]

    def _can_train(self) -> bool:
        if self._buf_y is None:
            return False
        if int(self._buf_y.numel()) < 8:
            return False
        try:
            return bool((self._buf_y.min() < 0.5) and (self._buf_y.max() > 0.5))
        except Exception:
            return False

    def _sample_train_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._buf_x is None or self._buf_y is None or (not self._can_train()):
            return None
        n = int(self._buf_y.numel())
        bs = min(self.train_batch_size, n)

        # Prefer class-balanced sampling for better discriminator training/calibration.
        try:
            y = self._buf_y
            idx_on = (y > 0.5).nonzero(as_tuple=False).view(-1)
            idx_off = (y <= 0.5).nonzero(as_tuple=False).view(-1)
            if idx_on.numel() > 0 and idx_off.numel() > 0 and bs >= 2:
                bs_on = bs // 2
                bs_off = bs - bs_on
                sel_on = idx_on[torch.randint(low=0, high=int(idx_on.numel()), size=(bs_on,), device=y.device)]
                sel_off = idx_off[torch.randint(low=0, high=int(idx_off.numel()), size=(bs_off,), device=y.device)]
                idx = torch.cat([sel_on, sel_off], dim=0)
                # shuffle
                idx = idx[torch.randperm(idx.numel(), device=idx.device)]
                return self._buf_x[idx], self._buf_y[idx]
        except Exception:
            pass

        # Fallback: uniform sampling
        idx = torch.randint(low=0, high=n, size=(bs,), device=self._buf_y.device)
        return self._buf_x[idx], self._buf_y[idx]

    @staticmethod
    def effective_sample_size(w: torch.Tensor, eps: float = 1e-8) -> float:
        """
        w: (n,) non-negative weights
        ESS = (sum w)^2 / sum w^2
        """
        w = w.detach().float()
        if w.numel() == 0:
            return 0.0
        s1 = float(w.sum().item())
        s2 = float((w * w).sum().item())
        if s2 <= eps:
            return 0.0
        return (s1 * s1) / s2

    def step(
        self,
        *,
        features: torch.Tensor,  # (bs, F), detached
        is_offpolicy: torch.Tensor,  # (bs,) bool; off-policy/teacher label=0
        alpha: Optional[float] = None,
        alpha_mode: str = "prior_or_micro",  # "prior_or_micro" | "sync_batch_ema" | "sync_batch" | "micro" | "buffer_ema" | "buffer"
        alpha_ema_beta: float = 0.9,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
          w_hat: (bs,) estimated relative density ratio w_alpha = D/(1-alpha).
          metrics: dict
        """
        device = features.device
        feats = features.detach().float()
        labels_on = (~is_offpolicy).float().detach()  # on-policy=1, off-policy=0

        self._maybe_init(device=device, in_dim=int(feats.shape[-1]))
        assert self._disc is not None
        assert self._opt is not None

        # --------------------------------------------------------------
        # Optional: multi-GPU synchronization for robust discriminator training
        # --------------------------------------------------------------
        self._calls += 1
        rank, world_size = self._get_rank_world()
        sync_metrics = {
            "dr3/sync_enabled": 1.0 if self.sync_across_ranks else 0.0,
            "dr3/sync_world_size": float(world_size),
            "dr3/sync_rank": float(rank),
            "dr3/sync_batch": float(feats.shape[0]),
            "dr3/sync_has_both_classes": 0.0,
            "dr3/sync_on_count": float(labels_on.sum().item()) if labels_on.numel() else 0.0,
            "dr3/sync_off_count": float((1.0 - labels_on).sum().item()) if labels_on.numel() else 0.0,
            # robustness: all_gather requires equal batch shapes across ranks
            "dr3/sync_shape_ok": 1.0,
            "dr3/sync_skipped_shape_mismatch": 0.0,
        }
        bcast_metrics = {
            "dr3/bcast_enabled": 1.0 if self.broadcast_params else 0.0,
            "dr3/bcast_world_size": float(world_size),
            "dr3/bcast_rank": float(rank),
            "dr3/bcast_happened": 0.0,
        }

        feats_for_buffer = feats
        labels_for_buffer = labels_on
        if self.sync_across_ranks and (self._calls % self.sync_every_n_calls == 0):
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    ws = int(dist.get_world_size())
                    sync_metrics["dr3/sync_world_size"] = float(ws)

                    # Gather batch sizes first (works even when per-rank micro-batch sizes differ).
                    bs_local = torch.tensor([int(feats.shape[0])], device=feats.device, dtype=torch.int32)
                    bs_list = [torch.zeros_like(bs_local) for _ in range(ws)]
                    dist.all_gather(bs_list, bs_local)
                    bss = [int(x.item()) for x in bs_list]
                    bs_max = int(max(bss)) if bss else int(feats.shape[0])
                    bs_sum = int(sum(bss)) if bss else int(feats.shape[0])
                    sync_metrics["dr3/sync_bs_max"] = float(bs_max)
                    sync_metrics["dr3/sync_bs_sum"] = float(bs_sum)
                    if len(set(bss)) != 1:
                        sync_metrics["dr3/sync_shape_ok"] = 0.0
                        sync_metrics["dr3/sync_skipped_shape_mismatch"] = 1.0

                    # Robust all_gather with padding to max batch size.
                    # This avoids deadlocks and lets rank0 build a "global" buffer even when some ranks have bs=0/1/2...
                    if bs_max <= 0:
                        # nothing to sync
                        feats_for_buffer = feats
                        labels_for_buffer = labels_on
                    else:
                        fdim = int(feats.shape[-1])
                        # pad feats/labels to (bs_max, ...)
                        if int(feats.shape[0]) < bs_max:
                            pad_n = bs_max - int(feats.shape[0])
                            feats_pad = torch.cat(
                                [feats, torch.zeros((pad_n, fdim), device=feats.device, dtype=feats.dtype)],
                                dim=0,
                            )
                            labels_pad = torch.cat(
                                [labels_on, torch.zeros((pad_n,), device=labels_on.device, dtype=labels_on.dtype)],
                                dim=0,
                            )
                        else:
                            feats_pad = feats
                            labels_pad = labels_on

                        feat_list = [torch.zeros_like(feats_pad) for _ in range(ws)]
                        lab_list = [torch.zeros_like(labels_pad) for _ in range(ws)]
                        dist.all_gather(feat_list, feats_pad)
                        dist.all_gather(lab_list, labels_pad)

                        # trim to real sizes then concat
                        feats_cat = []
                        labs_cat = []
                        for i in range(ws):
                            bi = int(bss[i]) if i < len(bss) else bs_max
                            if bi <= 0:
                                continue
                            feats_cat.append(feat_list[i][:bi])
                            labs_cat.append(lab_list[i][:bi])
                        if feats_cat:
                            feats_for_buffer = torch.cat(feats_cat, dim=0)
                            labels_for_buffer = torch.cat(labs_cat, dim=0)
                            sync_metrics["dr3/sync_batch"] = float(feats_for_buffer.shape[0])
                    try:
                        on_cnt = float(labels_for_buffer.sum().item())
                        off_cnt = float((1.0 - labels_for_buffer).sum().item())
                        sync_metrics["dr3/sync_on_count"] = on_cnt
                        sync_metrics["dr3/sync_off_count"] = off_cnt
                        sync_metrics["dr3/sync_has_both_classes"] = 1.0 if (on_cnt > 0 and off_cnt > 0) else 0.0
                    except Exception:
                        pass
            except Exception:
                # never break training
                feats_for_buffer = feats
                labels_for_buffer = labels_on

        # Push into rolling buffer so discriminator can be trained across micro-batches (and optionally across ranks).
        self._push_buffer(feats_for_buffer, labels_for_buffer)

        # Train discriminator a few steps (low overhead, local to rank)
        # With micro_batch_size=1, current micro-batch is often single-class.
        # We train on the rolling buffer (which mixes classes over time).
        disc_loss_val = 0.0
        disc_acc_val = 0.0
        disc_trained_steps = 0.0
        # If using rank0-train + broadcast, only rank0 performs optimizer steps.
        can_optimize = (not self.broadcast_params) or (rank == 0)
        if self.disc_steps_per_call > 0 and self._can_train() and can_optimize:
            batch = self._sample_train_batch()
            if batch is not None:
                xb, yb = batch
                for _ in range(self.disc_steps_per_call):
                    logits = self._disc(xb)
                    loss = self._bce(logits, yb)
                    self._opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self._opt.step()
                    disc_loss_val = float(loss.detach().item())
                    with torch.no_grad():
                        pred = (torch.sigmoid(logits) > 0.5).float()
                        disc_acc_val = float((pred == yb).float().mean().item())
                    disc_trained_steps += 1.0

        # Broadcast discriminator parameters from rank0 to all ranks (optional).
        if self.broadcast_params and (self._calls % self.broadcast_every_n_calls == 0):
            self._broadcast_model_params()
            bcast_metrics["dr3/bcast_happened"] = 1.0

        with torch.no_grad():
            logits = self._disc(feats)
            d = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)  # (bs,)
            # --------------------------------------------------------------
            # Decide alpha_used:
            # - If alpha is provided (alpha_prior): use it.
            # - Else estimate from actual on/off ratio (micro/sync/buffer) and optionally EMA smooth it.
            # --------------------------------------------------------------
            alpha_mode_norm = str(alpha_mode).lower().strip()
            alpha_raw: Optional[float] = None
            if alpha is not None:
                alpha_used = float(alpha)
                alpha_raw = float(alpha)
                alpha_mode_used = "prior"
            else:
                # auto estimation
                if alpha_mode_norm in ("sync_batch_ema", "sync_ema", "auto", "sync"):
                    alpha_raw = self._estimate_alpha_from_labels(labels_for_buffer)
                    alpha_mode_used = "sync_batch_ema"
                elif alpha_mode_norm in ("sync_batch", "sync_noema", "sync_raw"):
                    alpha_raw = self._estimate_alpha_from_labels(labels_for_buffer)
                    alpha_mode_used = "sync_batch"
                elif alpha_mode_norm in ("buffer_ema", "buf_ema"):
                    alpha_raw = self._estimate_alpha_from_labels(self._buf_y) if self._buf_y is not None else None
                    alpha_mode_used = "buffer_ema"
                elif alpha_mode_norm in ("buffer", "buf"):
                    alpha_raw = self._estimate_alpha_from_labels(self._buf_y) if self._buf_y is not None else None
                    alpha_mode_used = "buffer"
                elif alpha_mode_norm in ("micro", "micro_batch", "batch"):
                    alpha_raw = self._estimate_alpha_from_labels(labels_on)
                    alpha_mode_used = "micro"
                else:
                    # backward-compatible default: prior_or_micro
                    alpha_raw = self._estimate_alpha_from_labels(labels_on)
                    alpha_mode_used = "prior_or_micro"

                if alpha_raw is None:
                    alpha_raw = float(0.5)  # safe fallback (won't be applied if no off-policy anyway)
                    alpha_mode_used = f"{alpha_mode_used}_fallback"

                if "ema" in alpha_mode_used:
                    alpha_used = self._update_alpha_ema(alpha_raw, beta=float(alpha_ema_beta))
                else:
                    alpha_used = float(alpha_raw)

            alpha_used = self._clip_alpha(alpha_used, eps=self.eps)
            one_minus_alpha = float(max(self.eps, 1.0 - alpha_used))

            # Convert discriminator posterior to likelihood ratio r = p/q.
            # With class-balanced training batches, the implicit training prior is ~0.5,
            # so r_hat ≈ d / (1-d).
            r_hat = d / (1.0 - d)

            # Relative density ratio w_alpha = p / ((1-α)p + αq) = r / ((1-α)r + α)
            w = r_hat / (one_minus_alpha * r_hat + alpha_used)

            # Base theoretical upper bound for relative ratio
            base_upper = 1.0 / one_minus_alpha

            # Dual-controlled clipping
            # - only meaningful for off-policy samples in practice, but we clip all for safety.
            clip_upper = self.dual.clip_upper(base_upper=base_upper, clip_max=self.clip_max)
            clip_upper = float(max(self.eps, clip_upper))
            w_clipped = torch.clamp(w, 0.0, clip_upper)

            # Diagnostics on off-policy subset (rolling window)
            off_idx = is_offpolicy.bool()
            if off_idx.any():
                self._w_off_hist.append(w_clipped[off_idx].detach().float().cpu().flatten())
                if len(self._w_off_hist) > self.ess_window:
                    self._w_off_hist = self._w_off_hist[-self.ess_window :]

            w_hist = None
            try:
                if self._w_off_hist:
                    w_hist = torch.cat([t for t in self._w_off_hist if torch.is_tensor(t) and t.numel() > 0], dim=0)
            except Exception:
                w_hist = None
            ess = self.effective_sample_size(w_hist, eps=self.eps) if (w_hist is not None) else 0.0
            self.dual.update(ess=ess, n=int(w_hist.numel()) if (w_hist is not None) else 0)

            clipfrac = 0.0
            if off_idx.any():
                clipfrac = float((w[off_idx] > clip_upper).float().mean().item())

        metrics = {
            # keep "dr3/alpha" as the alpha actually used for weighting/clipping
            "dr3/alpha": float(alpha_used),
            "dr3/alpha_mode": float(
                0.0
                if alpha_mode_used.startswith("prior")
                else (1.0 if "sync" in alpha_mode_used else (2.0 if "buffer" in alpha_mode_used else 3.0))
            ),
            "dr3/alpha_raw": float(alpha_raw) if alpha_raw is not None else 0.0,
            "dr3/alpha_ema": float(self._alpha_ema) if self._alpha_ema is not None else 0.0,
            "dr3/disc_loss": float(disc_loss_val),
            "dr3/disc_acc": float(disc_acc_val),
            "dr3/disc_trained_steps": float(disc_trained_steps),
            # How many samples were pushed into the buffer in THIS call (after optional all_gather).
            "dr3/buf_pushed": float(feats_for_buffer.shape[0]) if torch.is_tensor(feats_for_buffer) else 0.0,
            "dr3/buf_pushed_on": float(labels_for_buffer.sum().item()) if torch.is_tensor(labels_for_buffer) else 0.0,
            "dr3/buf_pushed_off": float((1.0 - labels_for_buffer).sum().item()) if torch.is_tensor(labels_for_buffer) else 0.0,
            "dr3/buf_size": float(self._buf_y.numel()) if self._buf_y is not None else 0.0,
            "dr3/w_mean": float(w_clipped.mean().item()) if w_clipped.numel() else 0.0,
            "dr3/w_std": float(w_clipped.std().item()) if w_clipped.numel() > 1 else 0.0,
            "dr3/w_max": float(w_clipped.max().item()) if w_clipped.numel() else 0.0,
            "dr3/w_clip_upper": float(clip_upper),
            "dr3/w_clipfrac_off": float(clipfrac),
            "dr3/ess_off_window": float(ess),
            "dr3/ess_window_len": float(len(self._w_off_hist)),
            "dr3/dual_lambda": float(self.dual.lam),
        }
        metrics.update(sync_metrics)
        metrics.update(bcast_metrics)
        # Off-policy subset distribution (most important for analysis)
        try:
            off_idx = is_offpolicy.bool()
            if off_idx.any():
                vv = w_clipped[off_idx].float().flatten()
                metrics["dr3/w_off_mean"] = float(vv.mean().item()) if vv.numel() else 0.0
                metrics["dr3/w_off_std"] = float(vv.std().item()) if vv.numel() > 1 else 0.0
                metrics["dr3/w_off_max"] = float(vv.max().item()) if vv.numel() else 0.0
                # quantiles
                if vv.numel() >= 4:
                    metrics["dr3/w_off_p50"] = float(torch.quantile(vv, 0.50).item())
                    metrics["dr3/w_off_p90"] = float(torch.quantile(vv, 0.90).item())
                    metrics["dr3/w_off_p99"] = float(torch.quantile(vv, 0.99).item())
        except Exception:
            pass
        return w_clipped, metrics

