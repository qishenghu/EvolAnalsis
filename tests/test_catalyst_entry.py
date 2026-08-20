"""CATALYST v2 entry-k 单测:计划构建边界 / 调度促降退 / uid / 资格判定。

设计:docs/research/CATALYST_v2_设计_2026-08-13.md
"""
import json

import pytest

from agentevolver.module.exp_manager.catalyst import arm_uid_values
from agentevolver.module.exp_manager.catalyst_entry import (
    ENTRY_BOOK_VERSION,
    CatalystEntryBook,
    CatalystEntryReplayError,
    CatalystEntryScheduler,
    EntryPlan,
    extract_tagged_action,
    replay_teacher_prefix,
)


# ---------------------------------------------------------------------------
# 素材与工具
# ---------------------------------------------------------------------------
def make_book(tmp_path, n_decisions=10, task_id="42"):
    steps = [
        {"action": f"go to shelf {i}", "observation": f"You see shelf {i}."}
        for i in range(n_decisions - 1)  # builder 只存前 n−1 步
    ]
    payload = {
        "version": ENTRY_BOOK_VERSION,
        "environment": "alfworld",
        "tasks": {
            task_id: {
                "teacher_rollout_id": "t0",
                "n_teacher_decisions": n_decisions,
                "init_messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Your task is to: test."},
                ],
                "steps": steps,
            }
        },
    }
    book_path = tmp_path / "book.json"
    book_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return CatalystEntryBook(str(book_path), require_manifest=False)


def test_extract_tagged_action_takes_last_post_think():
    content = (
        "<think>plan <action>fake</action></think>\n"
        "<action>\ngo to desk 1\n</action>"
    )
    assert extract_tagged_action(content) == "go to desk 1"
    with pytest.raises(RuntimeError):
        extract_tagged_action("<think>no action</think>")


# ---------------------------------------------------------------------------
# 计划构建边界(与试点 build_takeover_plan 同构)
# ---------------------------------------------------------------------------
def test_build_plan_k_bounds(tmp_path):
    book = make_book(tmp_path, n_decisions=10)
    plan = book.build_plan("42", frac=0.75, rung=0, max_steps=30)
    assert plan.k_steps == 7  # floor(0.75*10)
    assert len(plan.replay_actions) == 7
    assert plan.rung == 0
    # k ≤ n−1:frac 高到吃满也要给学生留一步
    plan = book.build_plan("42", frac=0.99, rung=0, max_steps=30)
    assert plan.k_steps == 9
    # k < max_steps 防御
    with pytest.raises(RuntimeError):
        book.build_plan("42", frac=0.75, rung=0, max_steps=5)


def test_plan_payload_roundtrip(tmp_path):
    book = make_book(tmp_path)
    plan = book.build_plan("42", frac=0.5, rung=1, max_steps=30)
    restored = EntryPlan.from_payload(plan.to_payload())
    assert restored == plan


# ---------------------------------------------------------------------------
# walk-back 调度:促升 / 毕业 / 回撤 / 退休
# ---------------------------------------------------------------------------
def sched(**kw):
    cfg = dict(
        fracs=[0.75, 0.5, 0.25],
        slots_per_task=4,
        promote_hi=0.5,
        demote_lo=0.125,
        min_obs=4,
        ema_alpha=0.5,
        retire_windows=3,
    )
    cfg.update(kw)
    return CatalystEntryScheduler(cfg)


def test_scheduler_promote_and_graduate():
    s = sched()
    assert s.current_frac("t") == (0.75, 0)
    s.update("t", [True, True, True, True], global_step=1)
    assert s.current_frac("t") == (0.5, 1)  # 促升清零 EMA
    s.update("t", [True, True, True, True], global_step=2)
    assert s.current_frac("t") == (0.25, 2)
    s.update("t", [True, True, True, True], global_step=3)
    assert s.state("t").graduated
    assert not s.active("t")
    assert s.graduated_total == 1


def test_scheduler_retire_on_rung0():
    s = sched()
    for step in range(1, 4):
        s.update("t", [False, False, False, False], global_step=step)
    assert s.state("t").retired
    assert not s.active("t")
    assert s.retired_total == 1


def test_scheduler_demote_from_higher_rung():
    s = sched()
    s.update("t", [True] * 4, global_step=1)  # rung 0 -> 1
    for step in range(2, 5):
        s.update("t", [False] * 4, global_step=step)
    st = s.state("t")
    assert st.rung == 0 and not st.retired  # 回撤而非退休


def test_scheduler_needs_min_obs():
    s = sched(min_obs=8)
    s.update("t", [True] * 4, global_step=1)
    assert s.current_frac("t") == (0.75, 0)  # 4 obs < 8,不促升
    s.update("t", [True] * 4, global_step=2)
    assert s.current_frac("t") == (0.5, 1)


def test_scheduler_state_roundtrip(tmp_path):
    s = sched()
    s.update("t", [True] * 4, global_step=1)
    path = tmp_path / "entry_state.json"
    s.save_state(str(path))
    s2 = sched()
    assert s2.load_state(str(path))
    assert s2.current_frac("t") == (0.5, 1)


def test_scheduler_descending_fracs_required():
    with pytest.raises(ValueError):
        sched(fracs=[0.25, 0.5, 0.75])


# ---------------------------------------------------------------------------
# uid 分组(entry rung 后缀)
# ---------------------------------------------------------------------------
def test_arm_uid_values_entry_suffix():
    extras = [
        None,
        {"catalyst_arm": "hint"},
        {"catalyst_arm": "entry", "catalyst_entry_rung": 2},
        {"catalyst_arm": "entry"},  # rung 缺省 0
    ]
    uids = arm_uid_values([7, 7, 7, 7], extras)
    assert uids == ["7", "7|h", "7|e2", "7|e0"]


# ---------------------------------------------------------------------------
# 重放:divergence 计数 / 终止 fail-fast / 观测用 live
# ---------------------------------------------------------------------------
class FakeEnv:
    def __init__(self, observations, terminate_at=None, fail_at=None):
        self.observations = observations
        self.terminate_at = terminate_at
        self.fail_at = fail_at
        self.calls = 0

    def step(self, instance_id, message):
        index = self.calls
        self.calls += 1
        if self.fail_at is not None and index == self.fail_at:
            raise ConnectionError("boom")
        return {
            "state": [{"role": "user", "content": self.observations[index]}],
            "is_terminated": (
                self.terminate_at is not None and index == self.terminate_at
            ),
        }


def make_plan(k=3):
    return EntryPlan(
        task_id="42",
        frac=0.5,
        rung=0,
        k_steps=k,
        n_teacher_decisions=k * 2,
        teacher_rollout_id="t0",
        init_messages=[{"role": "user", "content": "task"}],
        replay_actions=[f"a{i}" for i in range(k)],
        expected_observations=[f"o{i}" for i in range(k)],
    )


def test_replay_seeds_live_observations():
    plan = make_plan(k=3)
    env = FakeEnv(["o0", "LIVE-DIFF", "o2"])
    seed_pairs, divergence = replay_teacher_prefix(env, "i", plan, None)
    assert divergence == 1
    assert seed_pairs[1] == ("<action>\na1\n</action>", "LIVE-DIFF")
    assert len(seed_pairs) == 3


def test_replay_termination_raises():
    plan = make_plan(k=3)
    env = FakeEnv(["o0", "o1", "o2"], terminate_at=1)
    with pytest.raises(CatalystEntryReplayError):
        replay_teacher_prefix(env, "i", plan, None)


def test_replay_transport_error_wrapped():
    plan = make_plan(k=3)
    env = FakeEnv(["o0", "o1", "o2"], fail_at=0)
    with pytest.raises(CatalystEntryReplayError):
        replay_teacher_prefix(env, "i", plan, None)


# ---------------------------------------------------------------------------
# runtime 级:v2 分配策略(entry 优先角点 / hint 中带 / 毕业裸跑)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

from agentevolver.module.exp_manager.catalyst import (  # noqa: E402
    HINT_CLEAN_VERSION,
    CatalystRuntime,
)
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig  # noqa: E402


def make_entry_files(tmp_path, *, stats=None, book_tasks=("corner",)):
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                tid: {"raw": f"hint for {tid}"}
                for tid in ("corner", "midband", "graduated", "nobook", "unseen")
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "hints.json.manifest.json").write_text(
        json.dumps({"clean_version": HINT_CLEAN_VERSION}), encoding="utf-8"
    )
    book_payload = {
        "version": ENTRY_BOOK_VERSION,
        "environment": "alfworld",
        "tasks": {
            tid: {
                "teacher_rollout_id": "t0",
                "n_teacher_decisions": 10,
                "init_messages": [{"role": "user", "content": "task"}],
                "steps": [
                    {"action": f"a{i}", "observation": f"o{i}"}
                    for i in range(9)
                ],
            }
            for tid in book_tasks
        },
    }
    book_path = tmp_path / "entry_book.json"
    book_path.write_text(
        json.dumps(book_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    if stats is None:
        stats = {
            "corner": {"sr_bare": 0.0, "n_bare": 16},
            "midband": {"sr_bare": 0.4, "n_bare": 16},
            "graduated": {"sr_bare": 0.9, "n_bare": 16},
            "nobook": {"sr_bare": 0.0, "n_bare": 16},
        }
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps({"schema": "catalyst_task_stats_v1", "tasks": stats}),
        encoding="utf-8",
    )
    return hints_path, book_path, stats_path


def make_v2_runtime(tmp_path, *, stats=None, book_tasks=("corner",)):
    hints_path, book_path, stats_path = make_entry_files(
        tmp_path, stats=stats, book_tasks=book_tasks
    )
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {
                        "file": str(hints_path),
                        "require_manifest": True,
                    },
                    "governance": {
                        "u_bootstrap_min_obs": 8,
                        "max_hint_rollouts": 2,
                    },
                    "arm_baseline": {"enable": True},
                    "replay": {"enable": False},
                    "thermostat": {"enable": False},
                    "entry": {
                        "enable": True,
                        "book_file": str(book_path),
                        "require_manifest": False,
                        "stats_bootstrap_file": str(stats_path),
                        "s_lo": 0.125,
                        "fracs": [0.75, 0.5, 0.25],
                        "slots_per_task": 4,
                        "promote_hi": 0.5,
                        "demote_lo": 0.125,
                        "min_obs": 4,
                        "ema_alpha": 0.5,
                        "retire_windows": 3,
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {"multi_turn": {"max_steps": 30}}
            },
        }
    )
    runtime = CatalystRuntime(config)
    # 自举在 load_persistent_state 里发生(resume 优先);测试直接触发
    runtime.load_persistent_state(str(tmp_path / "state" / "gov.json"))
    return runtime


def test_v2_allocation_routes_arms(tmp_path):
    runtime = make_v2_runtime(tmp_path)
    tasks = [
        SimpleNamespace(task_id=tid)
        for tid in ("corner", "midband", "graduated", "nobook", "unseen")
    ]
    tecs = [TaskExpConfig(add_exp=[]) for _ in tasks]
    metrics = runtime.plan_arms(tasks, tecs, n_rollout=8, global_step=1)

    # corner:入册 + 有观测 + 裸 EMA<0.125 → 4 entry + 4 裸
    entry_slots = tecs[0].catalyst_entry_slots
    assert entry_slots is not None
    assert sum(1 for s in entry_slots if s) == 4
    assert entry_slots[0]["frac"] == 0.75 and entry_slots[0]["k_steps"] == 7
    assert getattr(tecs[0], "catalyst_hint_slots", None) is None

    # midband:hint 臂,税率封顶 2
    hint_slots = tecs[1].catalyst_hint_slots
    assert hint_slots is not None
    assert sum(1 for s in hint_slots if s) == 2
    assert getattr(tecs[1], "catalyst_entry_slots", None) is None

    # graduated:全裸
    assert getattr(tecs[2], "catalyst_hint_slots", None) is None
    assert getattr(tecs[2], "catalyst_entry_slots", None) is None

    # nobook(角点但未入册):落 hint 路径
    assert getattr(tecs[3], "catalyst_entry_slots", None) is None
    assert tecs[3].catalyst_hint_slots is not None

    # unseen(无自举无观测):不吃 entry 槽(n_bare_obs=0),走 hint 路径
    assert getattr(tecs[4], "catalyst_entry_slots", None) is None

    assert metrics["entry_tasks"] == 1.0
    assert metrics["entry_rollouts"] == 4.0


def test_v2_entry_outcomes_feed_scheduler_not_governor(tmp_path):
    runtime = make_v2_runtime(tmp_path)
    bare_obs_before = runtime.governor.state("corner").n_bare_obs
    trajs = [
        SimpleNamespace(
            task_id="corner",
            discarded=False,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={
                "catalyst_arm": "entry",
                "catalyst_entry_rung": 0,
                "catalyst_entry_divergence": 0,
            },
        )
        for _ in range(4)
    ]
    metrics = runtime.update_after_rollout(trajs, global_step=1)
    # entry 成败只进调度器,不进 governor 裸 EMA
    assert runtime.governor.state("corner").n_bare_obs == bare_obs_before
    st = runtime.entry_scheduler.state("corner")
    assert st.rung == 1       # 4/4 促升
    assert st.n_obs == 0      # 促升清零观测(新 rung 重新计数)
    assert metrics["sr_entry_batch"] == 1.0


def test_v2_persistent_state_roundtrip(tmp_path):
    runtime = make_v2_runtime(tmp_path)
    runtime.entry_scheduler.update("corner", [True] * 4, global_step=3)
    state_path = str(tmp_path / "state" / "gov.json")
    runtime.save_persistent_state(state_path)
    runtime2 = make_v2_runtime(tmp_path)
    runtime2.load_persistent_state(state_path)
    assert runtime2.entry_scheduler.current_frac("corner") == (0.5, 1)


# ---------------------------------------------------------------------------
# v3:区间调度 / 课程 critic / 学生状态池 / 优势覆写数据链
# ---------------------------------------------------------------------------
from agentevolver.module.exp_manager.catalyst_entry import (  # noqa: E402
    CatalystEntryIntervalScheduler,
    CatalystStatePool,
    deterministic_frac,
    frac_bin,
)


def isched(**kw):
    cfg = dict(
        f_init=[0.5, 0.9], f_delta=0.05, f_min=0.05, f_max=0.95,
        graduate_f_hi=0.15, retire_f_lo=0.85, retire_windows=3,
        slots_per_task=4, critic_alpha=0.5, critic_min_task_obs=2,
    )
    cfg.update(kw)
    return CatalystEntryIntervalScheduler(cfg)


def test_interval_frac_sampling_deterministic_and_decorrelated():
    s = isched()
    f1 = s.plan_fracs("t", 7, 4)
    f2 = s.plan_fracs("t", 7, 4)
    assert f1 == f2                      # 同步同槽可复现
    assert len(set(round(f, 6) for f in f1)) == 4   # 逐槽不同(去相关)
    assert all(0.5 <= f <= 0.9 for f in f1)
    assert s.plan_fracs("t", 8, 4) != f1  # 跨步变化


def test_interval_moves_up_on_all_fail_and_down_on_all_success():
    s = isched()
    st = s.state("t")
    s.update("t", [(0.6, False)] * 4, global_step=1)
    assert (st.f_lo, st.f_hi) == (0.55, 0.95)
    s2 = isched()
    s2.update("t", [(0.7, True)] * 4, global_step=1)
    st2 = s2.state("t")
    assert st2.f_hi == pytest.approx(0.85) and st2.f_lo == pytest.approx(0.45)
    s2.update("t", [(0.6, True), (0.7, False)], global_step=2)  # 混合不动
    assert (st2.f_lo, st2.f_hi) == (pytest.approx(0.45), pytest.approx(0.85))


def test_interval_graduate_and_retire():
    s = isched()
    for step in range(1, 30):
        s.update("g", [(0.2, True)] * 4, global_step=step)
        if s.state("g").graduated:
            break
    assert s.state("g").graduated
    for step in range(1, 30):
        s.update("r", [(0.9, False)] * 4, global_step=step)
        if s.state("r").retired:
            break
    assert s.state("r").retired


def test_critic_task_then_global_fallback():
    s = isched()
    # 双冷启动 → 0
    assert s.vhat("t", 0.65) == 0.0
    # 别的任务喂了同段 → 全局段退化
    s.update("other", [(0.65, True), (0.65, True)], global_step=1)
    assert s.vhat("t", 0.65) > 0.5
    # 本任务喂满 min_task_obs → 用逐任务段
    s.update("t", [(0.65, False), (0.65, False)], global_step=2)
    assert s.vhat("t", 0.65) == 0.0
    assert frac_bin(0.19) == 0 and frac_bin(0.99) == 4


def test_interval_state_roundtrip():
    s = isched()
    s.update("t", [(0.6, True)] * 4, global_step=3)
    payload = s.save_payload()
    s2 = isched()
    s2.load_payload(payload)
    assert s2.state("t").f_hi == s.state("t").f_hi
    assert s2.global_n == s.global_n


class FakeMsg:
    def __init__(self, author, content):
        self.author = author
        self.content = content


class FakeCmt:
    def __init__(self, task_id, msgs, rollout_id="r0"):
        self.task_id = task_id
        self.rollout_id = rollout_id
        self.full_context = [FakeMsg(a, c) for a, c in msgs]


def good_cmt(task_id="42", n_dec=5, rollout_id="r0"):
    msgs = [("initialization", "sys"), ("initialization", "task")]
    for i in range(n_dec - 1):
        msgs.append(("llm", f"<think>plan</think>\n<action>\ngo to shelf {i}\n</action>"))
        msgs.append(("env", f"You see shelf {i}."))
    msgs.append(("llm", "<action>\ntake x\n</action>"))
    return FakeCmt(task_id, msgs, rollout_id=rollout_id)


def test_state_pool_insert_and_plan():
    pool = CatalystStatePool({"pool_max_per_task": 2, "pool_max_decisions": 30})
    assert pool.insert_from_cmt(good_cmt(), global_step=5)
    assert "42" in pool and pool.size() == 1
    plan = pool.build_plan("42", frac=0.5, max_steps=30)
    assert plan is not None
    assert plan.k_steps == 2  # floor(0.5*5)
    assert plan.replay_actions[0] == "go to shelf 0"
    assert plan.teacher_rollout_id.startswith("student:")
    # 教师 think 不可能入池:存的就是提取后的 action
    assert "<think>" not in plan.replay_actions[0]


def test_state_pool_rejects_malformed():
    pool = CatalystStatePool({})
    bad = FakeCmt("9", [("initialization", "t"), ("llm", "no action here")])
    assert not pool.insert_from_cmt(bad, global_step=1)
    assert pool.rejected_total == 1
    assert pool.build_plan("9", frac=0.5, max_steps=30) is None  # 空池兜底


def test_state_pool_fifo_and_roundtrip():
    pool = CatalystStatePool({"pool_max_per_task": 2})
    for i in range(3):
        pool.insert_from_cmt(good_cmt(rollout_id=f"r{i}"), global_step=i)
    plan = pool.build_plan("42", frac=0.5, max_steps=30)
    assert plan.teacher_rollout_id == "student:r2"  # 最新优先
    pool2 = CatalystStatePool({"pool_max_per_task": 2})
    pool2.load_payload(pool.save_payload())
    assert pool2.size() == 2


def make_v3_runtime(tmp_path, **entry_overrides):
    """复用文件构建 helper(不构造 v2 runtime,避免阶梯调度器误读 v3 状态)。"""
    make_entry_files(tmp_path)
    entry = {
        "enable": True,
        "mode": "interval",
        "book_file": str(tmp_path / "entry_book.json"),
        "require_manifest": False,
        "stats_bootstrap_file": str(tmp_path / "stats.json"),
        "s_lo": 0.125,
        "slots_per_task": 4,
        "f_init": [0.5, 0.9],
        "corner_hint_slots": 2,
        "student_pool": True,
    }
    entry.update(entry_overrides)
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {
                        "file": str(tmp_path / "hints.json"),
                        "require_manifest": True,
                    },
                    "governance": {"u_bootstrap_min_obs": 8},
                    "arm_baseline": {"enable": True},
                    "replay": {"enable": False},
                    "thermostat": {"enable": False},
                    "entry": entry,
                }
            },
            "actor_rollout_ref": {
                "rollout": {"multi_turn": {"max_steps": 30}}
            },
        }
    )
    runtime = CatalystRuntime(config)
    runtime.load_persistent_state(str(tmp_path / "state" / "gov.json"))
    return runtime


def test_v3_allocation_mixed_arms_and_per_slot_fracs(tmp_path):
    runtime = make_v3_runtime(tmp_path)
    tasks = [SimpleNamespace(task_id="corner")]
    tecs = [TaskExpConfig(add_exp=[])]
    metrics = runtime.plan_arms(tasks, tecs, n_rollout=8, global_step=1)
    entry_slots = tecs[0].catalyst_entry_slots
    payloads = [p for p in entry_slots if p]
    assert len(payloads) == 4
    fracs = {round(p["frac"], 6) for p in payloads}
    assert len(fracs) == 4                      # 逐槽独立 frac
    assert all("vhat" in p and p["source"] == "teacher" for p in payloads)
    hint_slots = tecs[0].catalyst_hint_slots
    assert sum(1 for h in hint_slots if h) == 2  # 角点混臂 hint
    assert all(
        not (entry_slots[i] and hint_slots[i]) for i in range(8)
    )  # 槽位互斥
    assert metrics["entry_pool_hit_frac"] == 0.0  # 冷启动全教师兜底


def test_v3_pool_preferred_after_student_success(tmp_path):
    runtime = make_v3_runtime(tmp_path)
    # 一条角点任务的 hint 臂成功轨迹进池(v3 真实燃料流:角点混臂的 hint
    # 造学生成功轨迹;hint 成败进 sr_hint_ema,不动裸 EMA → 任务仍是角点)
    trajs = [
        SimpleNamespace(
            task_id="corner", discarded=False,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={"catalyst_arm": "hint"},
            rollout_id="r7",
            full_context=good_cmt("corner").full_context,
        )
    ]
    runtime.update_after_rollout(trajs, global_step=1)
    assert runtime.entry_state_pool.size() == 1
    tecs = [TaskExpConfig(add_exp=[])]
    runtime.plan_arms(
        [SimpleNamespace(task_id="corner")], tecs, n_rollout=8, global_step=2
    )
    payloads = [p for p in tecs[0].catalyst_entry_slots if p]
    assert all(p["source"] == "student" for p in payloads)  # 池优先


def test_v3_entry_outcomes_update_interval_and_critic(tmp_path):
    runtime = make_v3_runtime(tmp_path)
    trajs = [
        SimpleNamespace(
            task_id="corner", discarded=False,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={
                "catalyst_arm": "entry",
                "catalyst_entry_frac": 0.7,
                "catalyst_entry_divergence": 0,
            },
        )
        for _ in range(4)
    ]
    metrics = runtime.update_after_rollout(trajs, global_step=1)
    st = runtime.entry_scheduler.state("corner")
    assert st.f_hi < 0.9                       # 全成 → 下移
    assert runtime.entry_scheduler.vhat("corner", 0.7) == 1.0
    assert metrics["entry_group_live_frac"] == 0.0  # 全同组
    assert metrics["sr_entry_batch"] == 1.0


def test_v3_persistent_roundtrip(tmp_path):
    runtime = make_v3_runtime(tmp_path)
    runtime.entry_scheduler.update("corner", [(0.7, True)] * 4, global_step=2)
    runtime.entry_state_pool.insert_from_cmt(good_cmt("corner"), global_step=2)
    state_path = str(tmp_path / "state" / "gov.json")
    runtime.save_persistent_state(state_path)
    runtime2 = make_v3_runtime(tmp_path)
    runtime2.load_persistent_state(state_path)
    assert runtime2.entry_scheduler.state("corner").f_hi < 0.9
    assert runtime2.entry_state_pool.size() == 1


def test_v2_replay_and_entry_mutually_exclusive(tmp_path):
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        runtime = make_v2_runtime(tmp_path)  # noqa: F841 - 先建好文件
        config_files_ready = tmp_path / "hints.json"
        assert config_files_ready.is_file()
        CatalystRuntime(
            OmegaConf.create(
                {
                    "exp_manager": {
                        "catalyst": {
                            "enable": True,
                            "hints": {
                                "file": str(tmp_path / "hints.json"),
                                "require_manifest": True,
                            },
                            "replay": {"enable": True},
                            "entry": {
                                "enable": True,
                                "book_file": str(
                                    tmp_path / "entry_book.json"
                                ),
                                "require_manifest": False,
                            },
                        }
                    }
                }
            )
        )
