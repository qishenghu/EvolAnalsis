"""T1:默认关闭 = 逐字节等价(规格 §0.2 纪律的作证测试)。

覆盖四个默认路径接触点:
  1. TrajExpConfig/TaskExpConfig 新字段的默认值(注入分支恒跳过);
  2. get_extra:无 catalyst metadata 时不加 catalyst_arm 键(D3);
  3. arm_uid_values:无 hint 标记时与原 str(int(gid)) 逐元素相等;
  4. samples_to_dataproto:无 is_catalyst_replay 样本时 batch keys 不含
     catalyst_replay_mask,且输出张量与"显式 False 标记"的批逐字节相等。
"""

from types import SimpleNamespace

import numpy as np
import torch

from catalyst_test_utils import (
    FakeTokenizer,
    make_env_manager,
    make_sample,
    onpolicy_extras,
)
from agentevolver.module.exp_manager.catalyst import arm_uid_values
from agentevolver.module.exp_manager.exp_manager import (
    TaskExpConfig,
    TrajExpConfig,
)


def test_exp_config_defaults_keep_injection_dormant():
    tec = TaskExpConfig(add_exp=[])
    assert tec.catalyst_hint_slots is None
    traj = TrajExpConfig()
    assert traj.catalyst_hint_text is None
    assert traj.catalyst_arm == "bare"
    # AgentFlow 注入块的守卫恰是 `if catalyst_hint_text:` → 默认恒跳过
    assert not traj.catalyst_hint_text


def test_get_extra_adds_no_key_without_catalyst_metadata():
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager

    manager = make_env_manager()
    cmt = SimpleNamespace(
        metadata={"env_type": "alfworld"},
        task_id="t0",
        rollout_id="0",
        decision_snapshots=[],
        context_policy=None,
    )
    extras = ParallelEnvManager.get_extra(manager, cmt)
    assert "catalyst_arm" not in extras
    # 提示臂 metadata 存在时才有键
    cmt.metadata["catalyst_arm"] = "hint"
    extras = ParallelEnvManager.get_extra(manager, cmt)
    assert extras["catalyst_arm"] == "hint"


def test_arm_uid_values_passthrough_without_hint_marks():
    group_ids = np.array([0, 0, 1, 1])
    baseline = [str(int(g)) for g in group_ids]
    assert arm_uid_values(group_ids, None) == baseline
    extras = [{"foo": 1}, {}, {"catalyst_arm": "bare"}, {"catalyst_arm": "replay"}]
    assert arm_uid_values(group_ids, extras) == baseline


def _regular_samples(tokenizer):
    return [
        make_sample(
            data_id="0", task_id="tA", rollout_id=str(i),
            prompt_text=f"prompt for rollout {i}",
            response_text=f"<action>\ngo {i}\n</action>",
            extras=onpolicy_extras(1),
            tokenizer=tokenizer,
        )
        for i in range(2)
    ]


def test_samples_to_dataproto_no_replay_no_key_and_deterministic():
    tokenizer = FakeTokenizer()
    manager = make_env_manager()
    out1 = manager.samples_to_dataproto(_regular_samples(tokenizer))
    assert "catalyst_replay_mask" not in out1.batch.keys()

    # 显式 is_catalyst_replay=False 与"键不存在"逐字节等价
    samples2 = _regular_samples(tokenizer)
    for sample in samples2:
        sample.extras["is_catalyst_replay"] = False
    out2 = manager.samples_to_dataproto(samples2)
    assert sorted(out1.batch.keys()) == sorted(out2.batch.keys())
    for key in out1.batch.keys():
        assert torch.equal(out1.batch[key], out2.batch[key]), key

    # 同输入重跑逐字节确定性
    out3 = manager.samples_to_dataproto(_regular_samples(tokenizer))
    for key in out1.batch.keys():
        assert torch.equal(out1.batch[key], out3.batch[key]), key


def test_to_dataproto_default_extra_samples_is_noop():
    """extra_samples=None(默认)与不传参数完全同路径。"""
    from catalyst_test_utils import make_cmt_stub

    tokenizer = FakeTokenizer()
    manager = make_env_manager()
    manager.get_extra = lambda cmt: dict(cmt.extras)

    def cmts():
        return [
            make_cmt_stub([s], onpolicy_extras(len(s.response_ids)))
            for s in _regular_samples(tokenizer)
        ]

    out_default = manager.to_dataproto(cmts(), optimizer_batch=True)
    out_none = manager.to_dataproto(
        cmts(), optimizer_batch=True, extra_samples=None
    )
    assert sorted(out_default.batch.keys()) == sorted(out_none.batch.keys())
    for key in out_default.batch.keys():
        assert torch.equal(out_default.batch[key], out_none.batch[key]), key
