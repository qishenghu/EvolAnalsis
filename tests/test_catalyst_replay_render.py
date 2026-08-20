"""T5(+A1):重放池入池军规审计 / 去提示重渲染 / TTL / data_id 撞号防御。"""

import copy
from types import SimpleNamespace

import pytest

from catalyst_test_utils import build_hinted_episode
from agentevolver.module.exp_manager.catalyst import (
    CATALYST_REPLAY_DATA_ID_BASE,
    HINT_MARKER,
    CatalystReplayPool,
)


def make_pool(**overrides):
    cfg = {
        "per_task": 1,
        "pool_max_per_task": 4,
        "ttl_steps": 20,
        "audit_on_insert": True,
    }
    cfg.update(overrides)
    return CatalystReplayPool(cfg)


def make_ready_pool(**overrides):
    cmt, policy, tokenizer, rollout_cfg = build_hinted_episode()
    pool = make_pool(**overrides)
    pool.attach_renderer(tokenizer, rollout_cfg)
    return pool, cmt, tokenizer


def test_insert_passes_military_audit():
    pool, cmt, _ = make_ready_pool()
    assert pool.insert_from_cmt(cmt, global_step=1)
    assert pool.size() == 1
    assert pool.audit_failures_total == 0


def test_insert_audit_catches_prompt_drift():
    pool, cmt, _ = make_ready_pool()
    # 篡改快照 prompt ids → 重构链与采样时不一致 → 拒收并计数
    cmt.decision_snapshots[1].prompt_token_ids = (
        list(cmt.decision_snapshots[1].prompt_token_ids) + [7]
    )
    assert not pool.insert_from_cmt(cmt, global_step=1)
    assert pool.size() == 0
    assert pool.audit_failures_total == 1


def test_insert_skips_non_snapshot_and_bad_shape():
    pool, cmt, _ = make_ready_pool()
    no_snap = copy.copy(cmt)
    no_snap.decision_snapshots = []
    assert not pool.insert_from_cmt(no_snap, global_step=1)
    bad_shape = copy.copy(cmt)
    bad_shape.full_context = cmt.full_context[:-1]  # 缺最后的 llm 消息
    assert not pool.insert_from_cmt(bad_shape, global_step=1)
    assert pool.insert_skips_total == 2


def test_dehinted_render_contract():
    pool, cmt, tokenizer = make_ready_pool()
    assert pool.insert_from_cmt(cmt, global_step=1)
    tasks = [SimpleNamespace(task_id=cmt.task_id)]
    samples, metrics = pool.build_replay_samples(
        tasks,
        global_step=2,
        max_prompt_len=8192,
        max_response_len=2048,
        existing_group_ids=[0, 1],
    )
    assert len(samples) == 1 and metrics["replay_samples_in_batch"] == 1.0
    sample = samples[0]

    # 军规:prompt 无 hint 痕迹;messages_raw(去提示全量消息)同样干净
    assert HINT_MARKER not in tokenizer.decode(sample.prompt_ids)
    for message in sample.messages + sample.messages_raw:
        assert HINT_MARKER not in message["content"]

    # response = 原采样 completion ids(零重分词漂移)
    selected = pool._pool[cmt.task_id][0]
    t = pool._select_decision_index(selected, global_step=2)
    assert sample.response_ids == selected.decisions[t]["completion_token_ids"]
    assert sample.minor_index_id == selected.decisions[t]["step_index"]

    # 去提示 prompt ≠ 采样时(带提示)prompt
    assert sample.prompt_ids != list(cmt.decision_snapshots[t].prompt_token_ids)

    # extras 契约(规格 F4)与损失 mask 形状
    extras = sample.extras
    assert extras["is_experience_replay"] is True
    assert extras["is_catalyst_replay"] is True
    assert extras["is_teacher"] is False
    assert extras["snapshot_training"] is False
    assert extras["rollout_log_probs"] is None
    assert extras["catalyst_arm"] == "replay"
    assert sample.prompt_loss_mask == [0] * len(sample.prompt_ids)
    assert sample.response_loss_mask == [1] * len(sample.response_ids)
    assert sample.reward_scores["success_rate"] == 1.0


def test_decision_selection_varies_with_global_step():
    pool, cmt, _ = make_ready_pool()
    pool.insert_from_cmt(cmt, global_step=0)
    entry = pool._pool[cmt.task_id][0]
    picks = {pool._select_decision_index(entry, step) for step in range(24)}
    assert picks == {0, 1}  # token 加权覆盖两个 decision


def test_strip_failure_drops_and_counts():
    # 消息里没有 hint 的条目(异常态)→ 渲染期剥除断言失败 → 弃样计数
    cmt, policy, tokenizer, rollout_cfg = build_hinted_episode(with_hint=False)
    cmt.metadata = {"catalyst_arm": "hint"}
    pool = make_pool()
    pool.attach_renderer(tokenizer, rollout_cfg)
    assert pool.insert_from_cmt(cmt, global_step=1)
    samples, metrics = pool.build_replay_samples(
        [SimpleNamespace(task_id=cmt.task_id)],
        global_step=2,
        max_prompt_len=8192,
        max_response_len=2048,
        existing_group_ids=[0],
    )
    assert samples == [] and metrics["replay_render_drops"] == 1.0


def test_ttl_eviction_and_fifo_capacity():
    pool, cmt, _ = make_ready_pool(ttl_steps=5, pool_max_per_task=2)
    assert pool.insert_from_cmt(cmt, global_step=0)
    # 容量:再插两条(不同 rollout_id),最旧的被 FIFO 挤出
    for rid in ["4", "5"]:
        cmt2 = copy.copy(cmt)
        cmt2.rollout_id = rid
        assert pool.insert_from_cmt(cmt2, global_step=1)
    assert pool.size() == 2
    assert [e.rollout_id for e in pool._pool[cmt.task_id]] == ["4", "5"]
    # TTL:step 7 时(7−1>5)全部代谢
    assert pool.evict_stale(global_step=7) == 2
    assert pool.size() == 0


def test_replay_data_id_collision_defense_a1():
    pool, cmt, _ = make_ready_pool(per_task=1)
    pool.insert_from_cmt(cmt, global_step=1)
    tasks = [SimpleNamespace(task_id=cmt.task_id)]
    # 人为把真实 group_ids 顶到重放基址 → 必须整体偏移且不撞号
    colliding = [CATALYST_REPLAY_DATA_ID_BASE, CATALYST_REPLAY_DATA_ID_BASE + 1]
    samples, _ = pool.build_replay_samples(
        tasks,
        global_step=2,
        max_prompt_len=8192,
        max_response_len=2048,
        existing_group_ids=colliding,
    )
    assert len(samples) == 1
    assert int(samples[0].data_id) not in colliding
    assert int(samples[0].data_id) >= CATALYST_REPLAY_DATA_ID_BASE


def test_build_restricted_to_in_batch_tasks():
    pool, cmt, _ = make_ready_pool()
    pool.insert_from_cmt(cmt, global_step=1)
    samples, metrics = pool.build_replay_samples(
        [SimpleNamespace(task_id="some_other_task")],
        global_step=2,
        max_prompt_len=8192,
        max_response_len=2048,
        existing_group_ids=[0],
    )
    assert samples == []  # 池外任务不产样本(规格 F7)
    assert metrics["replay_pool_entries"] == 1.0


def test_renderer_required():
    pool = make_pool()
    with pytest.raises(RuntimeError):
        pool.build_replay_samples(
            [SimpleNamespace(task_id="t")],
            global_step=1,
            max_prompt_len=10,
            max_response_len=10,
            existing_group_ids=[],
        )
