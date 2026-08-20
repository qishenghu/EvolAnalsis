"""T7+T8:compute_advantage 之后的 CATALYST 后处理。

T8:单样本 uid 组经 GRPO 得到非零优势(F5 病态再现)→ 后处理清零;
T7:熵恒温器 A′=A+λ(−logπ−b) 的平移数学、mask 剔除重放行、λ 非负投影与封顶。
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto

from agentevolver.module.trainer.ae_ray_trainer import (
    AgentEvolverRayPPOTrainer,
    compute_grpo_outcome_advantage,
)


def make_trainer(*, thermostat=False, lam=0.5, eta=0.1, h_ref=0.4, lam_max=1.0):
    trainer = object.__new__(AgentEvolverRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"multi_turn": {"enable": True}}}}
    )
    trainer._catalyst = SimpleNamespace(
        thermostat_enabled=thermostat,
        thermo_lambda=lam,
        thermo_eta=eta,
        thermo_h_ref=h_ref,
        thermo_lambda_max=lam_max,
    )
    return trainer


def make_batch(
    *,
    bs=3,
    prompt_len=4,
    resp_len=5,
    replay_rows=(2,),
    advantages=None,
):
    adv = (
        advantages
        if advantages is not None
        else torch.arange(bs * resp_len, dtype=torch.float32).reshape(bs, resp_len)
    )
    crm = torch.zeros(bs, prompt_len + resp_len, dtype=torch.int)
    for row in replay_rows:
        crm[row, prompt_len:] = 1
    loss_mask = torch.ones(bs, prompt_len + resp_len, dtype=torch.int)
    loss_mask[:, :prompt_len] = 0
    old_lp = -torch.linspace(0.5, 2.0, bs * resp_len).reshape(bs, resp_len)
    fields = {
        "advantages": adv.clone(),
        "catalyst_replay_mask": crm,
        "loss_mask": loss_mask,
        "old_log_probs": old_lp,
    }
    return (
        DataProto(batch=TensorDict(fields, batch_size=[bs])),
        adv,
        old_lp,
        loss_mask,
    )


# ------------------------------- T8 -------------------------------

def test_singleton_uid_group_grpo_advantage_is_nonzero_then_zeroed():
    # F5 再现:单样本组 mean=0/std=1 → adv == 原始 score ≠ 0
    token_level_rewards = torch.zeros(1, 5)
    token_level_rewards[0, -1] = 1.0
    adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=torch.ones(1, 5),
        index=np.array(["100000"], dtype=object),
    )
    assert adv.abs().sum() > 0

    trainer = make_trainer()
    batch, _, _, _ = make_batch(bs=1, resp_len=5, replay_rows=(0,), advantages=adv)
    metrics = trainer._catalyst_post_advantage(batch, entropys=None)
    assert torch.all(batch.batch["advantages"] == 0)
    assert metrics["catalyst/replay_rows_zeroed"] == 1.0


def test_replay_rows_zeroed_others_untouched():
    trainer = make_trainer()
    batch, adv, _, _ = make_batch(bs=3, replay_rows=(1,))
    trainer._catalyst_post_advantage(batch, entropys=None)
    out = batch.batch["advantages"]
    assert torch.all(out[1] == 0)
    assert torch.equal(out[0], adv[0])
    assert torch.equal(out[2], adv[2])


def test_no_replay_mask_is_noop_without_thermostat():
    trainer = make_trainer()
    bs, prompt_len, resp_len = 2, 3, 4
    adv = torch.randn(bs, resp_len)
    batch = DataProto(
        batch=TensorDict({"advantages": adv.clone()}, batch_size=[bs])
    )
    metrics = trainer._catalyst_post_advantage(batch, entropys=None)
    assert metrics == {}
    assert torch.equal(batch.batch["advantages"], adv)


# ------------------------------- T7 -------------------------------

def test_thermostat_shift_math_and_replay_exclusion():
    trainer = make_trainer(thermostat=True, lam=0.5, eta=0.1, h_ref=0.4)
    bs, prompt_len, resp_len = 3, 4, 5
    batch, adv, old_lp, loss_mask = make_batch(
        bs=bs, prompt_len=prompt_len, resp_len=resp_len, replay_rows=(2,)
    )
    entropys = torch.rand(bs, resp_len)
    metrics = trainer._catalyst_post_advantage(batch, entropys)

    # 手算参照:mask = response 段 loss_mask 且剔除重放行
    mask = loss_mask[:, -resp_len:].float()
    mask[2] = 0.0
    denom = mask.sum()
    neg_lp = -old_lp
    b = (neg_lp * mask).sum() / denom
    h_hat = (entropys * mask).sum() / denom
    expected = adv.clone()
    expected[2] = 0.0  # ① 先清零
    expected = expected + 0.5 * (neg_lp - b) * mask  # ② 再平移(重放行 mask=0)
    assert torch.allclose(batch.batch["advantages"], expected, atol=1e-6)
    assert torch.all(batch.batch["advantages"][2] == 0)

    assert metrics["catalyst/lambda"] == pytest.approx(0.5)
    assert metrics["catalyst/h_hat"] == pytest.approx(h_hat.item(), abs=1e-6)
    assert metrics["catalyst/thermo_b"] == pytest.approx(b.item(), abs=1e-6)
    # λ 对偶更新:λ' = λ + η(H_ref − Ĥ)
    expected_lambda = min(
        max(0.5 + 0.1 * (0.4 - h_hat.item()), 0.0), 1.0
    )
    assert trainer._catalyst.thermo_lambda == pytest.approx(
        expected_lambda, abs=1e-6
    )


def test_thermostat_lambda_projection_and_cap():
    # 投影:Ĥ 远高于 H_ref → λ 落到 0,不为负
    trainer = make_trainer(thermostat=True, lam=0.01, eta=1.0, h_ref=0.0)
    batch, _, _, _ = make_batch(bs=2, replay_rows=())
    entropys = torch.full((2, 5), 5.0)
    trainer._catalyst_post_advantage(batch, entropys)
    assert trainer._catalyst.thermo_lambda == 0.0
    # 封顶:Ĥ 远低于 H_ref → λ 卡在 lambda_max
    trainer2 = make_trainer(thermostat=True, lam=0.9, eta=1.0, h_ref=5.0, lam_max=1.0)
    batch2, _, _, _ = make_batch(bs=2, replay_rows=())
    trainer2._catalyst_post_advantage(batch2, torch.zeros(2, 5))
    assert trainer2._catalyst.thermo_lambda == 1.0


def test_thermostat_requires_aligned_entropys():
    trainer = make_trainer(thermostat=True)
    batch, _, _, _ = make_batch(bs=2, replay_rows=())
    with pytest.raises(RuntimeError):
        trainer._catalyst_post_advantage(batch, entropys=None)


# ------------------------- v3:课程 critic 优势覆写 -------------------------

def make_v3_trainer():
    trainer = object.__new__(AgentEvolverRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"multi_turn": {"enable": True}}}}
    )
    trainer._catalyst = SimpleNamespace(
        thermostat_enabled=False,
        entry_enabled=True,
        entry_mode="interval",
        entry_adv_scale=1.0,
    )
    return trainer


def test_v3_entry_advantage_override_with_critic_baseline():
    trainer = make_v3_trainer()
    bs, plen, rlen = 3, 4, 5
    adv = torch.full((bs, rlen), 7.0)  # GRPO 组内算出的任意值,应被覆写
    batch, _, _, _ = make_batch(
        bs=bs, prompt_len=plen, resp_len=rlen, replay_rows=(), advantages=adv
    )
    tlr = torch.zeros(bs, rlen)
    tlr[0, -1] = 1.0   # row0: entry 成功
    # row1: entry 失败(奖励 0);row2: 裸臂,不覆写
    batch.batch["token_level_rewards"] = tlr
    rmask = torch.ones(bs, rlen)
    rmask[0, -1] = 0   # 造一个 mask 洞验证广播乘 mask
    batch.batch["response_mask"] = rmask
    batch.non_tensor_batch["extras"] = np.array(
        [
            {"catalyst_arm": "entry", "catalyst_entry_vhat": 0.25},
            {"catalyst_arm": "entry", "catalyst_entry_vhat": 0.25},
            {"catalyst_arm": "bare"},
        ],
        dtype=object,
    )
    metrics = trainer._catalyst_post_advantage(batch, entropys=None)
    out = batch.batch["advantages"]
    # row0: (1−0.25)=0.75 × mask;row1: (0−0.25)=−0.25;row2: 原样 7.0
    assert torch.allclose(out[0], torch.tensor([0.75] * 4 + [0.0]))
    assert torch.allclose(out[1], torch.full((rlen,), -0.25))
    assert torch.all(out[2] == 7.0)
    assert metrics["catalyst/entry_adv_rows"] == 2.0
    assert metrics["catalyst/entry_adv_mean"] == pytest.approx(0.25)


def test_v31_hint_advantage_override_gated_by_vhat_presence():
    trainer = make_v3_trainer()
    trainer._catalyst.hint_critic_baseline = True
    bs, plen, rlen = 3, 4, 5
    adv = torch.full((bs, rlen), 7.0)
    batch, _, _, _ = make_batch(
        bs=bs, prompt_len=plen, resp_len=rlen, replay_rows=(), advantages=adv
    )
    tlr = torch.zeros(bs, rlen)
    tlr[0, -1] = 1.0
    batch.batch["token_level_rewards"] = tlr
    batch.batch["response_mask"] = torch.ones(bs, rlen)
    batch.non_tensor_batch["extras"] = np.array(
        [
            {"catalyst_arm": "hint", "catalyst_hint_vhat": 0.8},   # 成:+0.2
            {"catalyst_arm": "hint", "catalyst_hint_vhat": 0.8},   # 败:−0.8
            {"catalyst_arm": "hint"},  # 无 vhat(未启用批次)→ 不覆写
        ],
        dtype=object,
    )
    metrics = trainer._catalyst_post_advantage(batch, entropys=None)
    out = batch.batch["advantages"]
    assert torch.allclose(out[0], torch.full((rlen,), 0.2), atol=1e-6)
    assert torch.allclose(out[1], torch.full((rlen,), -0.8), atol=1e-6)
    assert torch.all(out[2] == 7.0)
    assert metrics["catalyst/hint_adv_rows"] == 2.0
