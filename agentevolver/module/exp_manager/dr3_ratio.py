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


def compute_sequence_features(
    *,
    log_prob: torch.Tensor,        # (bs, T)
    advantages: torch.Tensor,      # (bs, T)
    response_mask: torch.Tensor,   # (bs, T)
) -> torch.Tensor:
    """
    Build a compact per-sequence feature vector from already-available tensors.
    Returns: (bs, F)
    """
    with torch.no_grad():
        m = response_mask
        lp_mean = _masked_mean(log_prob, m)
        lp_std = _masked_std(log_prob, m)
        lp_min = _masked_min(log_prob, m, fill=0.0)
        adv_abs_mean = _masked_mean(advantages.abs(), m)
        resp_len = m.float().sum(dim=-1)
        feats = torch.stack([lp_mean, lp_std, lp_min, adv_abs_mean, resp_len], dim=-1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats


class DR3Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        h = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        clip_max: float = 10.0,
        dual_enable: bool = True,
        ess_target_ratio: float = 0.5,
        dual_lr: float = 0.05,
        dual_init: float = 0.0,
        eps: float = 1e-6,
    ):
        self.hidden = int(hidden)
        self.lr = float(lr)
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

        self._buf_x: Optional[torch.Tensor] = None  # (N,F) on device
        self._buf_y: Optional[torch.Tensor] = None  # (N,) labels_on in {0,1} on device
        self._w_off_hist: list[torch.Tensor] = []   # list of 1D CPU tensors

    def _maybe_init(self, *, device: torch.device, in_dim: int) -> None:
        if self._disc is not None:
            return
        self._disc = DR3Discriminator(in_dim=in_dim, hidden=self.hidden).to(device)
        self._opt = torch.optim.Adam(self._disc.parameters(), lr=self.lr)
        self._buf_x = torch.empty((0, in_dim), device=device, dtype=torch.float32)
        self._buf_y = torch.empty((0,), device=device, dtype=torch.float32)

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
        alpha: float,
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
        sync_metrics = {
            "dr3/sync_enabled": 1.0 if self.sync_across_ranks else 0.0,
            "dr3/sync_world_size": 1.0,
            "dr3/sync_batch": float(feats.shape[0]),
            "dr3/sync_has_both_classes": 0.0,
            "dr3/sync_on_count": float(labels_on.sum().item()) if labels_on.numel() else 0.0,
            "dr3/sync_off_count": float((1.0 - labels_on).sum().item()) if labels_on.numel() else 0.0,
        }

        feats_for_buffer = feats
        labels_for_buffer = labels_on
        if self.sync_across_ranks and (self._calls % self.sync_every_n_calls == 0):
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    ws = int(dist.get_world_size())
                    sync_metrics["dr3/sync_world_size"] = float(ws)
                    # all_gather current micro-batch features/labels across ranks
                    feat_list = [torch.zeros_like(feats) for _ in range(ws)]
                    lab_list = [torch.zeros_like(labels_on) for _ in range(ws)]
                    dist.all_gather(feat_list, feats)
                    dist.all_gather(lab_list, labels_on)
                    feats_for_buffer = torch.cat(feat_list, dim=0)
                    labels_for_buffer = torch.cat(lab_list, dim=0)
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
        if self.disc_steps_per_call > 0 and self._can_train():
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

        with torch.no_grad():
            logits = self._disc(feats)
            d = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)  # (bs,)
            one_minus_alpha = float(max(self.eps, 1.0 - float(alpha)))
            w = d / one_minus_alpha

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
            "dr3/alpha": float(alpha),
            "dr3/disc_loss": float(disc_loss_val),
            "dr3/disc_acc": float(disc_acc_val),
            "dr3/disc_trained_steps": float(disc_trained_steps),
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

