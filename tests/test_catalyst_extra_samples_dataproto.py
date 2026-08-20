"""T9(A2):混合批(常规样本 + 重放样本)过 to_dataproto 全链。

断言:形状/padding/对齐正确;catalyst_replay_mask 的 prompt 段全 0、response
段 == 各重放样本的 response_loss_mask(padding 后);常规行全 0;无重放样本时
该 key 不存在;exp_mask 把重放行归 off-policy 侧。
"""

import torch

from catalyst_test_utils import (
    FakeTokenizer,
    make_cmt_stub,
    make_env_manager,
    make_sample,
    onpolicy_extras,
    replay_extras,
)


def build_mixed_inputs(tokenizer):
    """1 任务 × 2 常规快照样本 + 2 条重放样本(不同长度,考验 padding)。"""
    cmts = []
    for i in range(2):
        sample = make_sample(
            data_id="0", task_id="tA", rollout_id=str(i),
            prompt_text=f"live prompt {i} with some length padding {'x' * (8 * i)}",
            response_text=f"<action>\ngo {i}\n</action>",
            extras=onpolicy_extras(1),
            tokenizer=tokenizer,
        )
        cmts.append(make_cmt_stub([sample], onpolicy_extras(len(sample.response_ids))))
    replay_samples = [
        make_sample(
            data_id=str(100000 + j), task_id="tA", rollout_id=f"cr1_{j}",
            prompt_text=f"dehinted replay prompt {j}",
            response_text="<think>\nplan\n</think>\n<action>\ntake soap\n</action>"
            + ("!" * 13 * j),
            extras=replay_extras("tA", inserted_step=j),
            tokenizer=tokenizer,
        )
        for j in range(2)
    ]
    return cmts, replay_samples


def test_mixed_batch_shapes_and_replay_mask():
    tokenizer = FakeTokenizer()
    manager = make_env_manager(world_size=1, rollout_n=2)
    manager.get_extra = lambda cmt: dict(cmt.extras)
    cmts, replay_samples = build_mixed_inputs(tokenizer)

    out = manager.to_dataproto(
        cmts, optimizer_batch=True, extra_samples=replay_samples
    )
    bs = out.batch["input_ids"].shape[0]
    assert bs == 4  # 2 常规 + 2 重放,整组对齐无裁剪(world_size=1)

    input_shape = out.batch["input_ids"].shape
    for key in ("attention_mask", "loss_mask", "exp_mask", "catalyst_replay_mask"):
        assert out.batch[key].shape == input_shape, key

    resp_len = out.batch["responses"].shape[-1]
    prompt_len = input_shape[-1] - resp_len
    crm = out.batch["catalyst_replay_mask"]
    # prompt 段恒 0
    assert torch.all(crm[:, :prompt_len] == 0)

    # 行序:trajectories_to_samples 先常规后 extra(对齐保持插入序)
    task_rollout = [
        (str(t), str(r))
        for t, r in zip(
            out.non_tensor_batch["task_ids"], out.non_tensor_batch["rollout_ids"]
        )
    ]
    assert task_rollout == [
        ("tA", "0"), ("tA", "1"), ("tA", "cr1_0"), ("tA", "cr1_1")
    ]

    # 常规行 replay mask 全 0;重放行 response 段 == padded response_loss_mask
    assert torch.all(crm[0] == 0) and torch.all(crm[1] == 0)
    for row, sample in ((2, replay_samples[0]), (3, replay_samples[1])):
        expected = torch.zeros(resp_len, dtype=crm.dtype)
        expected[: len(sample.response_loss_mask)] = torch.tensor(
            sample.response_loss_mask, dtype=crm.dtype
        )
        assert torch.equal(crm[row, prompt_len:], expected)
        # exp_mask:is_experience_replay → response 段同 loss mask(off-policy 侧)
        assert torch.equal(out.batch["exp_mask"][row, prompt_len:], expected)
        # 重放样本无 rollout logprob:mask 行全 0
        assert torch.all(out.batch["rollout_log_probs_mask"][row] == 0)

    # group_ids:常规同组 0,重放行独立成组(A1 基址)
    assert out.batch["group_ids"].tolist() == [0, 0, 100000, 100001]

    # response ids 原样保序进入 padded tensor
    for row, sample in ((2, replay_samples[0]), (3, replay_samples[1])):
        n = len(sample.response_ids)
        assert out.batch["responses"][row, :n].tolist() == sample.response_ids
        assert torch.all(out.batch["responses"][row, n:] == manager.pad_token_id)


def test_no_replay_samples_no_mask_key():
    tokenizer = FakeTokenizer()
    manager = make_env_manager(world_size=1, rollout_n=2)
    manager.get_extra = lambda cmt: dict(cmt.extras)
    cmts, _ = build_mixed_inputs(tokenizer)
    out = manager.to_dataproto(cmts, optimizer_batch=True)
    assert "catalyst_replay_mask" not in out.batch.keys()


def test_replay_rows_relax_pure_grpo_integrity_check():
    """重放样本(is_experience_replay)使批走 mixed 宽松对齐——不触发
    “每 UID 组恰 rollout.n 条”的纯 on-policy 硬校验。"""
    tokenizer = FakeTokenizer()
    manager = make_env_manager(world_size=1, rollout_n=2)
    manager.get_extra = lambda cmt: dict(cmt.extras)
    cmts, replay_samples = build_mixed_inputs(tokenizer)
    # 重放组只有 1 条/组;若被当纯 on-policy 批会被 expected_group_size 拒绝。
    out = manager.to_dataproto(
        cmts, optimizer_batch=True, extra_samples=replay_samples[:1]
    )
    assert out.batch["input_ids"].shape[0] == 3
