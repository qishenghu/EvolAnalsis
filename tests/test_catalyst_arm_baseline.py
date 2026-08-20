"""T2:分臂基线(D1 uid 后缀)的正确性。

uid 后缀让现有 compute_grpo_outcome_advantage 按 (task, arm) 分组:
每样本 advantage == 臂内手算 (r − mean_arm)/(std_arm + ε)。
"""

import numpy as np
import pytest
import torch

from agentevolver.module.exp_manager.catalyst import arm_uid_values
from agentevolver.module.trainer.ae_ray_trainer import (
    compute_grpo_outcome_advantage,
)

EPS = 1e-6


def _grpo(rewards, uids, norm_std=True):
    n = len(rewards)
    resp_len = 4
    token_level_rewards = torch.zeros(n, resp_len)
    token_level_rewards[:, -1] = torch.tensor(rewards, dtype=torch.float32)
    response_mask = torch.ones(n, resp_len)
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(uids, dtype=object),
        norm_adv_by_std_in_grpo=norm_std,
    )
    return adv[:, 0]  # 每行 broadcast,同一值


def _expected(rewards):
    t = torch.tensor(rewards, dtype=torch.float32)
    if t.numel() == 1:
        return t  # 单样本组:mean=0/std=1 → adv=raw(F5 病态,别处防御)
    return (t - t.mean()) / (t.std() + EPS)  # torch.std 默认无偏(n−1)


def test_arm_uid_values_marks_hint_rows():
    group_ids = np.array([0, 0, 0, 0, 1])
    extras = [
        {"catalyst_arm": "hint"},
        {"catalyst_arm": "hint"},
        {},
        {"catalyst_arm": "bare"},
        {"catalyst_arm": "hint"},
    ]
    assert arm_uid_values(group_ids, extras) == ["0|h", "0|h", "0", "0", "1|h"]


def test_per_arm_baseline_matches_hand_calc():
    bare_r = [1.0, 0.0, 0.0, 0.0]
    hint_r = [1.0, 1.0, 1.0, 0.0]
    uids = ["0", "0", "0", "0", "0|h", "0|h", "0|h", "0|h"]
    adv = _grpo(bare_r + hint_r, uids)
    exp_bare = _expected(bare_r)
    exp_hint = _expected(hint_r)
    assert torch.allclose(adv[:4], exp_bare, atol=1e-5)
    assert torch.allclose(adv[4:], exp_hint, atol=1e-5)
    # 交叉检查:分臂后裸臂优势不再被提示臂的高 SR 抬升的基线压制
    whole = _expected(bare_r + hint_r)
    assert not torch.allclose(adv, whole, atol=1e-4)


def test_without_suffix_reduces_to_group_baseline():
    rewards = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    adv = _grpo(rewards, ["0"] * 8)
    assert torch.allclose(adv, _expected(rewards), atol=1e-5)


def test_two_tasks_arms_are_independent_groups():
    rewards = [1.0, 0.0, 1.0, 0.0]
    uids = ["0", "0", "1|h", "1|h"]
    adv = _grpo(rewards, uids)
    assert torch.allclose(adv[:2], _expected([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(adv[2:], _expected([1.0, 0.0]), atol=1e-5)


def test_no_norm_variant_centers_only():
    bare_r = [1.0, 0.0]
    hint_r = [1.0, 1.0]
    adv = _grpo(bare_r + hint_r, ["0", "0", "0|h", "0|h"], norm_std=False)
    assert torch.allclose(
        adv[:2], torch.tensor([0.5, -0.5]), atol=1e-5
    )
    # 提示臂全对 → 臂内中心化为 0(教师素材"物有所值"信号交给治理层,而非优势)
    assert torch.allclose(adv[2:], torch.tensor([0.0, 0.0]), atol=1e-5)


def test_singleton_group_pathology_reproduced():
    """F5 再现:单样本组 adv == 原始 score(重放行必须由后处理清零)。"""
    adv = _grpo([1.0], ["100000"])
    assert adv[0] == pytest.approx(1.0)
