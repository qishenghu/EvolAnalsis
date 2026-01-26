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

    def _maybe_init(self, *, device: torch.device, in_dim: int) -> None:
        if self._disc is not None:
            return
        self._disc = DR3Discriminator(in_dim=in_dim, hidden=self.hidden).to(device)
        self._opt = torch.optim.Adam(self._disc.parameters(), lr=self.lr)

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

        # Train discriminator a few steps (low overhead, local to rank)
        # IMPORTANT: micro-batches can be single-class (all teacher or all on-policy).
        # In that case, skip training to avoid collapsing the classifier.
        disc_loss_val = 0.0
        disc_acc_val = 0.0
        single_class = False
        try:
            single_class = bool((labels_on.min() == labels_on.max()).item())
        except Exception:
            single_class = False

        if (self.disc_steps_per_call > 0) and (not single_class):
            for _ in range(self.disc_steps_per_call):
                logits = self._disc(feats)
                loss = self._bce(logits, labels_on)
                self._opt.zero_grad(set_to_none=True)
                loss.backward()
                self._opt.step()
                disc_loss_val = float(loss.detach().item())
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    disc_acc_val = float((pred == labels_on).float().mean().item())

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

            # Diagnostics on off-policy subset
            off_idx = is_offpolicy.bool()
            w_off = w_clipped[off_idx]
            ess = self.effective_sample_size(w_off, eps=self.eps)
            self.dual.update(ess=ess, n=int(off_idx.sum().item()))

            clipfrac = 0.0
            if off_idx.any():
                clipfrac = float((w[off_idx] > clip_upper).float().mean().item())

        metrics = {
            "dr3/alpha": float(alpha),
            "dr3/disc_loss": float(disc_loss_val),
            "dr3/disc_acc": float(disc_acc_val),
            "dr3/w_mean": float(w_clipped.mean().item()) if w_clipped.numel() else 0.0,
            "dr3/w_std": float(w_clipped.std().item()) if w_clipped.numel() > 1 else 0.0,
            "dr3/w_max": float(w_clipped.max().item()) if w_clipped.numel() else 0.0,
            "dr3/w_clip_upper": float(clip_upper),
            "dr3/w_clipfrac_off": float(clipfrac),
            "dr3/ess_off": float(ess),
            "dr3/dual_lambda": float(self.dual.lam),
        }
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

