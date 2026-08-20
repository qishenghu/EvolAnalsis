"""CATALYST v4 单测:估计器闭式解 / 分配性质 / 统一优势 / 回放一致性 / 运行时接线。

设计与回放验证:docs/research/CATALYST_进展汇报_2026-08-17.md §六。
回放一致性测试是本套件的锚:真模块跑历史统计口径,必须复现验证脚本的结论
(携梯度行 ~77%、锚定退火方向),防止实现漂离被验证的设计。
"""
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agentevolver.module.exp_manager.catalyst import CatalystRuntime, HINT_CLEAN_VERSION
from agentevolver.module.exp_manager.catalyst_v4 import (
    CatalystV4State,
    SlotPlan,
    V4Allocator,
    V4ValueTable,
    ctx_key,
    v4_loo_prior_advantage,
)
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig
from test_catalyst_entry import ENTRY_BOOK_VERSION, good_cmt, make_entry_files


# ---------------------------------------------------------------------------
# 值表
# ---------------------------------------------------------------------------
def test_value_table_shrinkage_chain():
    t = V4ValueTable({"alpha": 0.5, "n0": 2.0})
    c = ("bare", None)
    assert t.prior("x", c) == 0.5                      # 双冷启动
    t.update("other", c, True); t.update("other", c, True)
    assert t.prior("x", c) > 0.5                       # 无个人数据 → 听全局
    t.update("x", c, False)
    # 个人 1 条(n_eff=1)+ 先验 2 条全局:(1*0 + 2*g)/(1+2)
    g = t._g[c][0]
    assert t.prior("x", c) == pytest.approx(2 * g / 3)


def test_value_table_bootstrap_only_fills_absent():
    t = V4ValueTable({})
    t.update("a", ("bare", None), True)
    n = t.bootstrap_bare({"a": {"sr_bare": 0.0, "n_bare": 9},
                          "b": {"sr_bare": 0.1, "n_bare": 9}})
    assert n == 1                                       # a 有在线观测,不覆盖
    assert t.prior("b", ("bare", None)) < 0.5


def test_value_table_roundtrip(tmp_path):
    t = V4ValueTable({})
    t.update("a", ("entry", 3), True); t.update("a", ("hint", None), False)
    CatalystV4State.save(str(tmp_path / "s.json"), t)
    t2 = V4ValueTable({})
    assert CatalystV4State.load(str(tmp_path / "s.json"), t2)
    assert t2.prior("a", ("entry", 3)) == t.prior("a", ("entry", 3))


def test_ctx_key_bins():
    assert ctx_key("entry", 0.19) == ("entry", 0)
    assert ctx_key("entry", 0.99) == ("entry", 4)
    assert ctx_key("bare") == ("bare", None)


# ---------------------------------------------------------------------------
# 统一优势(闭式)
# ---------------------------------------------------------------------------
def test_loo_prior_advantage_closed_form():
    # k=4 全成,m=0.25:b = (2*0.25 + 3)/(2+3) = 0.7 → A = +0.3(全同组有梯度)
    a = v4_loo_prior_advantage([1, 1, 1, 1], [0.25] * 4, 2.0)
    assert a == pytest.approx([0.3] * 4)
    # 孤样本:b = m → A = r − m
    assert v4_loo_prior_advantage([1.0], [0.4], 2.0) == pytest.approx([0.6])
    # n0→0、k 大:还原 RLOO 留一均值
    a = v4_loo_prior_advantage([1, 0, 1, 0], [0.9] * 4, 1e-9)
    assert a[0] == pytest.approx(1 - 1 / 3, abs=1e-6)
    # 均值近零性:m 校准时 E[A]≈0
    a = v4_loo_prior_advantage([1, 0], [0.5, 0.5], 2.0)
    assert sum(a) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 分配器
# ---------------------------------------------------------------------------
def alloc_and_table(bare_sr):
    t = V4ValueTable({})
    for _ in range(6):
        t.update("t", ("bare", None), bare_sr > 0.5)
        t.update("t", ("hint", None), True)
        for b in range(5):
            t.update("t", ("entry", b), b >= 2)
    a = V4Allocator({"fbins": 5, "bare_floor": 2})
    return a, t


def test_allocator_deterministic_and_floor():
    a, t = alloc_and_table(0.0)
    p1 = a.allocate("t", 7, 8, t, has_hint=True, has_entry=True)
    p2 = a.allocate("t", 7, 8, t, has_hint=True, has_entry=True)
    assert p1 == p2                                       # 可复现
    assert len(p1) == 8
    assert sum(1 for s in p1 if s.arm == "bare") >= 2      # 裸保底
    fr = [s.frac for s in p1 if s.arm == "entry"]
    assert len(set(fr)) == len(fr)                        # entry 槽起点各异


def test_allocator_anchor_annealing():
    """目标锚定:裸已解决(V̂_bare 高)→ 辅助份额显著低于裸未解决时。"""
    a = V4Allocator({"fbins": 5, "bare_floor": 2})
    def aux_share(bare_sr):
        t = V4ValueTable({})
        # 全局层喂出画像:裸 = bare_sr,辅助全部中带(最大可学习性)
        for _ in range(8):
            t.update("g", ("bare", None), bare_sr)
        for arm in [("hint", None)] + [("entry", b) for b in range(5)]:
            t.update("g", arm, True); t.update("g", arm, False)
        plans = a.allocate("g", 3, 8, t, has_hint=True, has_entry=True)
        return sum(1 for s in plans if s.arm != "bare") / 8
    low = aux_share(False)    # 裸全败(V̂_bare→0)
    high = aux_share(True)    # 裸全成(V̂_bare→1)
    assert low > high         # 裸未解决时辅助多,解决后辅助被锚定压低
    assert high <= 0.25       # 裸解决后辅助近乎退场


def test_allocator_no_signal_reduces_to_grpo():
    t = V4ValueTable({})
    # 所有 context 打满(V̂→1):分数全零 → 8 槽全裸 = GRPO 特例
    for arm in [("bare", None), ("hint", None)] + [("entry", b) for b in range(5)]:
        for _ in range(8):
            t.update("t", arm, True)
        for _ in range(8):
            t.update("g2", arm, True)
    a = V4Allocator({"fbins": 5, "bare_floor": 2})
    plans = a.allocate("t", 1, 8, t, has_hint=True, has_entry=True)
    assert all(s.arm == "bare" for s in plans)


# ---------------------------------------------------------------------------
# 回放一致性(锚测试):真模块复现验证脚本的统计结论
# ---------------------------------------------------------------------------
def test_replay_consistency_signal_density_and_annealing():
    """合成一个缩微回放:三类组(全成 hint 组/全败 entry 组/混合裸组),
    组基线携梯度 1/3,统一优势应 3/3;再验锚定退火方向。"""
    n0 = 2.0
    t = V4ValueTable({"n0": n0})
    t.update("x", ("hint", None), False)   # hint 先验 < 1
    m_h = t.prior("x", ("hint", None))
    groups = [
        ([1, 1, 1, 1], [m_h] * 4),          # 全成 hint 组:组基线 0,统一优势 >0
        ([0, 0, 0, 0], [0.5] * 4),          # 全败 entry 组:组基线 0,统一优势 <0
        ([1, 0, 1, 0], [0.5] * 4),          # 混合:两者都有
    ]
    live_group = live_unified = 0
    for rs, ms in groups:
        if len(set(rs)) > 1:
            live_group += 1
        a = v4_loo_prior_advantage([float(r) for r in rs], ms, n0)
        if any(abs(x) > 0.05 for x in a):
            live_unified += 1
    assert live_group == 1 and live_unified == 3


# ---------------------------------------------------------------------------
# 运行时接线(plan → 槽位/透传;update → 值表/池/校准)
# ---------------------------------------------------------------------------
def make_v4_runtime(tmp_path):
    make_entry_files(tmp_path)
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {"file": str(tmp_path / "hints.json"),
                              "require_manifest": True},
                    "replay": {"enable": False},
                    "thermostat": {"enable": False},
                    "entry": {
                        "enable": False,
                        "book_file": str(tmp_path / "entry_book.json"),
                        "require_manifest": False,
                        "stats_bootstrap_file": str(tmp_path / "stats.json"),
                    },
                    "v4": {"enable": True, "n0": 2.0, "alpha": 0.5,
                           "fbins": 5, "bare_floor": 2},
                }
            },
            "actor_rollout_ref": {"rollout": {"multi_turn": {"max_steps": 30}}},
        }
    )
    runtime = CatalystRuntime(config)
    runtime.load_persistent_state(str(tmp_path / "state" / "gov.json"))
    return runtime


def test_v4_plan_arms_transport_chain(tmp_path):
    runtime = make_v4_runtime(tmp_path)
    tecs = [TaskExpConfig(add_exp=[])]
    metrics = runtime.plan_arms(
        [SimpleNamespace(task_id="corner")], tecs, n_rollout=8, global_step=1
    )
    ms = tecs[0].catalyst_v4_m_slots
    assert len(ms) == 8 and all(isinstance(m, float) for m in ms)
    entry_slots = getattr(tecs[0], "catalyst_entry_slots", None)
    hint_slots = getattr(tecs[0], "catalyst_hint_slots", None)
    n_entry = sum(1 for p in (entry_slots or []) if p)
    n_hint = sum(1 for h in (hint_slots or []) if h)
    n_bare = 8 - n_entry - n_hint
    assert n_bare >= 2                                    # 裸保底
    assert metrics["v4_aux_share"] == pytest.approx((n_entry + n_hint) / 8)
    if entry_slots:
        for i in range(8):
            assert not (entry_slots[i] and (hint_slots or [None] * 8)[i])


def test_v4_update_feeds_table_pool_and_brier(tmp_path):
    runtime = make_v4_runtime(tmp_path)
    trajs = [
        SimpleNamespace(
            task_id="corner", discarded=False,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={"catalyst_arm": "hint", "catalyst_v4_m": 0.3},
            rollout_id="r1", full_context=good_cmt("corner").full_context,
        ),
        SimpleNamespace(
            task_id="corner", discarded=False,
            reward=SimpleNamespace(success_rate=0.0),
            metadata={"catalyst_arm": "entry", "catalyst_entry_frac": 0.7,
                      "catalyst_v4_m": 0.6, "catalyst_entry_divergence": 0},
        ),
    ]
    metrics = runtime.update_after_rollout(trajs, global_step=1)
    assert runtime.v4_table.prior("corner", ("hint", None)) > 0.5
    assert runtime.v4_table.prior("corner", ("entry", 3)) < 0.5
    assert runtime.entry_state_pool.size() == 1           # hint 成功入池
    # Brier = mean((0.3−1)², (0.6−0)²)
    assert metrics["v4_brier"] == pytest.approx((0.49 + 0.36) / 2)


def test_v4_persistent_roundtrip(tmp_path):
    runtime = make_v4_runtime(tmp_path)
    runtime.v4_table.update("corner", ("hint", None), True)
    runtime.entry_state_pool.insert_from_cmt(good_cmt("corner"), global_step=1)
    path = str(tmp_path / "state" / "gov.json")
    runtime.save_persistent_state(path)
    runtime2 = make_v4_runtime(tmp_path)
    runtime2.load_persistent_state(path)
    assert runtime2.v4_table.prior("corner", ("hint", None)) == \
        runtime.v4_table.prior("corner", ("hint", None))
    assert runtime2.entry_state_pool.size() == 1


def test_v4_mutually_exclusive_with_legacy(tmp_path):
    make_entry_files(tmp_path)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        CatalystRuntime(OmegaConf.create({
            "exp_manager": {"catalyst": {
                "enable": True,
                "hints": {"file": str(tmp_path / "hints.json"),
                          "require_manifest": True},
                "replay": {"enable": False},
                "entry": {"enable": True, "mode": "interval",
                          "book_file": str(tmp_path / "entry_book.json"),
                          "require_manifest": False},
                "v4": {"enable": True},
            }},
            "actor_rollout_ref": {"rollout": {"multi_turn": {"max_steps": 30}}},
        }))


# ---------------------------------------------------------------------------
# v5:rescue 第四格(失败前缀 + 提示;2×2 闭合)
# ---------------------------------------------------------------------------
from agentevolver.module.exp_manager.catalyst_entry import CatalystStatePool as _Pool
from test_catalyst_entry import FakeCmt


def failed_cmt(task_id="corner", n_pairs=4, rollout_id="f0", noop=0):
    msgs = [("initialization", "sys"), ("initialization", "task")]
    for i in range(n_pairs):
        msgs.append(("llm", f"<action>\ngo to desk {i}\n</action>"))
        obs = "Nothing happened" if i < noop else f"You see desk {i}."
        msgs.append(("env", obs))
    return FakeCmt(task_id, msgs, rollout_id=rollout_id)


def test_failure_pool_insert_and_rescue_plan():
    pool = _Pool({"pool_max_per_task": 2})
    assert pool.insert_failure_from_cmt(failed_cmt(), global_step=3)
    assert pool.has_failure("corner")
    plan = pool.build_rescue_plan("corner", frac=0.5, max_steps=30)
    assert plan is not None and plan.teacher_rollout_id.startswith("failure:")
    assert plan.k_steps == 2  # floor(0.5 * (4+1))
    # 垃圾滤网:过半无效动作拒收
    assert not pool.insert_failure_from_cmt(failed_cmt(noop=3), global_step=3)
    # 太短拒收
    assert not pool.insert_failure_from_cmt(failed_cmt(n_pairs=2), global_step=3)


def test_ctx_key_rescue_bins():
    assert ctx_key("rescue", 0.55) == ("rescue", 2)


def test_allocator_rescue_contexts():
    t = V4ValueTable({})
    for _ in range(4):
        t.update("t", ("bare", None), False)
    a = V4Allocator({"fbins": 5, "bare_floor": 2})
    plans = a.allocate("t", 3, 8, t, has_hint=True, has_entry=False,
                       has_rescue=True)
    arms = {p.arm for p in plans}
    assert "rescue" in arms          # 裸全败 → 救场格拿到槽
    assert sum(1 for p in plans if p.arm == "bare") >= 2


def make_v5_runtime(tmp_path):
    make_entry_files(tmp_path)
    config = OmegaConf.create({
        "exp_manager": {"catalyst": {
            "enable": True,
            "hints": {"file": str(tmp_path / "hints.json"),
                      "require_manifest": True},
            "replay": {"enable": False},
            "thermostat": {"enable": False},
            "entry": {"enable": False,
                      "book_file": str(tmp_path / "entry_book.json"),
                      "require_manifest": False},
            "v4": {"enable": True, "n0": 2.0, "alpha": 0.5, "fbins": 5,
                   "bare_floor": 2, "rescue": True},
        }},
        "actor_rollout_ref": {"rollout": {"multi_turn": {"max_steps": 30}}},
    })
    runtime = CatalystRuntime(config)
    runtime.load_persistent_state(str(tmp_path / "state" / "gov.json"))
    return runtime


def test_v5_rescue_slot_composition_and_update(tmp_path):
    runtime = make_v5_runtime(tmp_path)
    # 喂一条裸失败 → 失败桶;把裸 V̂ 压低使救场格开闸
    trajs = [SimpleNamespace(
        task_id="corner", discarded=False,
        reward=SimpleNamespace(success_rate=0.0),
        metadata={"catalyst_arm": "bare", "catalyst_v4_m": 0.5},
        rollout_id=f"f{i}",
        full_context=failed_cmt("corner", rollout_id=f"f{i}").full_context,
    ) for i in range(4)]
    runtime.update_after_rollout(trajs, global_step=1)
    assert runtime.entry_state_pool.has_failure("corner")
    tecs = [TaskExpConfig(add_exp=[])]
    runtime.plan_arms([SimpleNamespace(task_id="corner")], tecs,
                      n_rollout=8, global_step=2)
    entry_slots = getattr(tecs[0], "catalyst_entry_slots", None) or [None]*8
    hint_slots = getattr(tecs[0], "catalyst_hint_slots", None) or [None]*8
    combo = [i for i in range(8) if entry_slots[i] and hint_slots[i]]
    assert combo, "应存在 rescue 复合槽(entry payload + hint 同槽)"
    assert entry_slots[combo[0]]["teacher_rollout_id"].startswith("failure:")
    # rescue 结局进值表 ("rescue", bin)
    trajs2 = [SimpleNamespace(
        task_id="corner", discarded=False,
        reward=SimpleNamespace(success_rate=1.0),
        metadata={"catalyst_arm": "rescue", "catalyst_entry_frac": 0.55,
                  "catalyst_v4_m": 0.4, "catalyst_entry_divergence": 0},
    )]
    runtime.update_after_rollout(trajs2, global_step=2)
    assert runtime.v4_table.prior("corner", ("rescue", 2)) > 0.5


def test_v5_persistence_includes_failure_bucket(tmp_path):
    runtime = make_v5_runtime(tmp_path)
    runtime.entry_state_pool.insert_failure_from_cmt(
        failed_cmt("corner"), global_step=1)
    p = str(tmp_path / "state" / "gov.json")
    runtime.save_persistent_state(p)
    r2 = make_v5_runtime(tmp_path)
    r2.load_persistent_state(p)
    assert r2.entry_state_pool.has_failure("corner")
