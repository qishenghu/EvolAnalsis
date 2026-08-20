"""T3:w 计算正确性 + stop-grad(∂bc/∂logits 无经 w 的二阶项)。"""

import pytest
import torch

from agentevolver.module.exp_manager.catalyst import compute_replay_bc_terms


def test_w_formula():
    log_prob = torch.tensor([[-0.1, -1.0, -3.0]])
    entropy = torch.tensor([[0.05, 0.5, 0.2]])
    w, losses = compute_replay_bc_terms(log_prob, entropy)
    expected_w = torch.clamp(torch.exp(log_prob + entropy), max=1.0)
    assert torch.allclose(w, expected_w)
    assert torch.allclose(losses, expected_w * (-log_prob))
    # φ>0 的位置被 cap 到 1
    log_prob2 = torch.tensor([[-0.1]])
    entropy2 = torch.tensor([[2.0]])
    w2, _ = compute_replay_bc_terms(log_prob2, entropy2)
    assert w2.item() == pytest.approx(1.0)


def test_w_cap_and_tau():
    log_prob = torch.tensor([[-1.0]])
    entropy = torch.tensor([[0.0]])
    w, _ = compute_replay_bc_terms(log_prob, entropy, w_cap=0.5)
    assert w.item() == pytest.approx(min(0.5, torch.exp(torch.tensor(-1.0)).item()))
    w_tau, _ = compute_replay_bc_terms(log_prob, entropy, phi_tau=2.0)
    assert w_tau.item() == pytest.approx(torch.exp(torch.tensor(-0.5)).item())


def test_w_is_stop_grad_and_gradient_is_minus_w_dlogp():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7, requires_grad=True)
    log_softmax = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([[1, 4, 2]])
    log_prob = torch.gather(log_softmax, -1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_softmax.exp() * log_softmax).sum(-1)

    w, losses = compute_replay_bc_terms(log_prob, entropy)
    # stop-grad:w 不在计算图上
    assert not w.requires_grad and w.grad_fn is None
    loss = losses.sum()
    (grad_actual,) = torch.autograd.grad(loss, logits, retain_graph=True)

    # 手算参照:∂loss/∂logits == Σ_t w_t(detached)·∂(−logπ_t)/∂logits
    (grad_ref,) = torch.autograd.grad(
        (w.detach() * (-log_prob)).sum(), logits, retain_graph=True
    )
    assert torch.allclose(grad_actual, grad_ref, atol=1e-6)

    # 反证:若 w 参与梯度(不 stop-grad),梯度应不同
    with torch.enable_grad():
        w_live = torch.clamp(torch.exp(log_prob + entropy), max=1.0)
        loss_live = (w_live * (-log_prob)).sum()
    (grad_live,) = torch.autograd.grad(loss_live, logits)
    assert not torch.allclose(grad_actual, grad_live, atol=1e-6)


def test_self_written_tokens_have_w_near_one():
    """φ 零点自校准直觉:典型自写 token(p 接近分布典型值)w≈1。"""
    # 均匀分布:logπ = −log V,H = log V → φ = 0 → w = 1
    vocab = 11
    log_prob = torch.full((1, 4), -torch.log(torch.tensor(float(vocab))))
    entropy = torch.full((1, 4), torch.log(torch.tensor(float(vocab))))
    w, _ = compute_replay_bc_terms(log_prob, entropy)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6)
