"""T4:ρ 控制器 / U 门 / EMA / 退休 / 量化 / 状态往返 + plan_arms 槽位填充。"""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agentevolver.module.exp_manager.catalyst import (
    HINT_CLEAN_VERSION,
    CatalystGovernor,
    CatalystRuntime,
)
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig


class StubHintBook:
    def __init__(self, hints):
        self._hints = dict(hints)

    def get(self, task_id):
        return self._hints.get(str(task_id))

    def __len__(self):
        return len(self._hints)


def make_governor(**overrides):
    cfg = {
        "s_hi": 0.8,
        "rho_max": 0.5,
        "delta_u": 0.0,
        "ema_alpha": 0.2,
        "u_bootstrap_min_obs": 8,
        "min_hint_rollouts": 2,
        "max_hint_rollouts": -1,
        "retire_windows": 3,
    }
    cfg.update(overrides)
    return CatalystGovernor(cfg, StubHintBook({"t1": "hint text"}))


# ------------------------------ ρ 公式与量化 ------------------------------

def test_rho_boundaries():
    gov = make_governor()
    st = gov.state("t1")
    st.sr_bare_ema = 0.0
    assert gov.rho("t1") == pytest.approx(0.5)      # 冷启动:ρ=ρ_max
    st.sr_bare_ema = 0.8
    assert gov.rho("t1") == pytest.approx(0.0)      # SR≥s* → 0
    st.sr_bare_ema = 0.4
    assert gov.rho("t1") == pytest.approx(0.5)      # clip 到 ρ_max
    st.sr_bare_ema = 0.72
    assert gov.rho("t1") == pytest.approx(1 - 0.72 / 0.8)


def test_quantize_k():
    gov = make_governor()
    n = 8
    assert gov.quantize_k(0, n) == 0
    assert gov.quantize_k(1, n) == 0        # < min_hint_rollouts → 0
    assert gov.quantize_k(2, n) == 2
    assert gov.quantize_k(7, n) == 6        # cap n−2(裸臂 ≥2)
    assert gov.quantize_k(100, n) == 6
    gov2 = make_governor(max_hint_rollouts=3)
    assert gov2.quantize_k(5, n) == 3


def test_plan_k_hint_routing():
    gov = make_governor()
    # 无素材 → R0
    assert gov.plan_k_hint("unknown_task", 8) == 0
    # 冷启动:ρ=0.5 → k=4
    assert gov.plan_k_hint("t1", 8) == 4
    # SR 达标 → R0
    gov.state("t1").sr_bare_ema = 0.85
    assert gov.plan_k_hint("t1", 8) == 0
    # 退休 → R0
    gov.state("t1").sr_bare_ema = 0.0
    gov.state("t1").retired = True
    assert gov.plan_k_hint("t1", 8) == 0


def test_u_gate_bootstrap_and_close():
    gov = make_governor()
    st = gov.state("t1")
    # bootstrap:提示臂观测不足时门放行(即使 U<=δ)
    st.sr_hint_ema = 0.0
    st.sr_bare_ema = 0.5
    st.n_hint_obs = 7
    assert gov.plan_k_hint("t1", 8) > 0
    # bootstrap 结束且 U<=δ → 门关
    st.n_hint_obs = 8
    assert gov.plan_k_hint("t1", 8) == 0
    # U>δ → 门开
    st.sr_hint_ema = 0.6
    assert gov.plan_k_hint("t1", 8) > 0


# ------------------------------ EMA 与退休 ------------------------------

def test_ema_first_observation_seeds_then_smooths():
    gov = make_governor(ema_alpha=0.2)
    gov.update_from_outcomes({"t1": {"bare": [True, False], "hint": []}}, 1)
    st = gov.state("t1")
    assert st.sr_bare_ema == pytest.approx(0.5)     # 首次观测直接播种
    assert st.n_bare_obs == 2
    gov.update_from_outcomes({"t1": {"bare": [True, True], "hint": []}}, 2)
    assert st.sr_bare_ema == pytest.approx(0.8 * 0.5 + 0.2 * 1.0)


def test_retirement_streak_and_permanence():
    gov = make_governor(u_bootstrap_min_obs=2, retire_windows=3)
    outcomes = {"t1": {"bare": [True, True], "hint": [False, False]}}
    # 窗 1:n_hint_obs 0→2,达到 bootstrap 阈值后才计 streak;
    # update 内先累计 obs 再判 streak,故第 1 窗即开始计数
    for step in range(1, 3):
        gov.update_from_outcomes(outcomes, step)
        assert not gov.state("t1").retired
    metrics = gov.update_from_outcomes(outcomes, 3)
    st = gov.state("t1")
    assert st.retired and st.retired_step == 3
    assert gov.retired_total == 1
    assert metrics["tasks_newly_retired"] == 1.0
    # 退休不可逆(M1):即使 U 回正也不复活
    gov.update_from_outcomes(
        {"t1": {"bare": [False, False], "hint": [True, True]}}, 4
    )
    assert gov.state("t1").retired
    assert gov.plan_k_hint("t1", 8) == 0


def test_positive_u_resets_streak():
    # ema_alpha=1.0 使 EMA 即时跟随本窗读数,单个好窗即可翻正 U
    gov = make_governor(u_bootstrap_min_obs=2, retire_windows=2, ema_alpha=1.0)
    bad = {"t1": {"bare": [True, True], "hint": [False, False]}}
    good = {"t1": {"bare": [False, False], "hint": [True, True]}}
    gov.update_from_outcomes(bad, 1)
    assert gov.state("t1").u_low_streak == 1
    gov.update_from_outcomes(good, 2)
    assert gov.state("t1").u_low_streak == 0
    assert not gov.state("t1").retired


def test_state_save_load_roundtrip(tmp_path):
    gov = make_governor()
    gov.update_from_outcomes(
        {"t1": {"bare": [True], "hint": [False, True]}}, 5
    )
    gov.state("t1").u_low_streak = 2
    path = tmp_path / "gov.json"
    gov.save_state(str(path))
    gov2 = make_governor()
    assert gov2.load_state(str(path))
    assert gov2.per_task_dump() == gov.per_task_dump()
    assert gov2.retired_total == gov.retired_total
    # 不存在的路径 → False(冷启动)
    assert not make_governor().load_state(str(tmp_path / "nope.json"))


# ------------------------------ plan_arms(runtime 级) -----------------------

def make_runtime(tmp_path, *, replay=False):
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps({"t1": {"raw": "check the sink"}, "t2": {"raw": "go north"}}),
        encoding="utf-8",
    )
    (tmp_path / "hints.json.manifest.json").write_text(
        json.dumps({"clean_version": HINT_CLEAN_VERSION}), encoding="utf-8"
    )
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {"file": str(hints_path), "require_manifest": True},
                    "governance": {"u_bootstrap_min_obs": 8},
                    "arm_baseline": {"enable": True},
                    "replay": {"enable": replay},
                    "thermostat": {"enable": False},
                }
            }
        }
    )
    return CatalystRuntime(config)


def test_plan_arms_fills_slots(tmp_path):
    runtime = make_runtime(tmp_path)
    tasks = [SimpleNamespace(task_id=t) for t in ["t1", "t2", "t3"]]
    tecs = [TaskExpConfig(add_exp=[]) for _ in tasks]
    metrics = runtime.plan_arms(tasks, tecs, n_rollout=8, global_step=1)
    # t1/t2 冷启动 R1(k=4);t3 无素材 R0
    assert metrics["tasks_r1"] == 2.0 and metrics["tasks_r0"] == 1.0
    assert metrics["hint_rollouts"] == 8.0
    assert metrics["rho_mean"] == pytest.approx(0.5)
    assert tecs[0].catalyst_hint_slots == ["check the sink"] * 4 + [None] * 4
    assert tecs[1].catalyst_hint_slots == ["go north"] * 4 + [None] * 4
    assert tecs[2].catalyst_hint_slots is None   # R0 任务保持默认(零改动路径)


def test_plan_arms_requires_n_ge_4(tmp_path):
    runtime = make_runtime(tmp_path)
    with pytest.raises(RuntimeError):
        runtime.plan_arms(
            [SimpleNamespace(task_id="t1")], [TaskExpConfig(add_exp=[])],
            n_rollout=2, global_step=1,
        )


def test_update_after_rollout_governance(tmp_path):
    runtime = make_runtime(tmp_path)
    trajs = [
        SimpleNamespace(
            task_id="t1", discarded=False,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={"catalyst_arm": "hint"},
        ),
        SimpleNamespace(
            task_id="t1", discarded=False,
            reward=SimpleNamespace(success_rate=0.0),
            metadata={},          # 裸臂:无 catalyst 标记(D3 语义)
        ),
        SimpleNamespace(  # discarded 轨迹不入统计
            task_id="t1", discarded=True,
            reward=SimpleNamespace(success_rate=1.0),
            metadata={},
        ),
    ]
    metrics = runtime.update_after_rollout(trajs, global_step=7)
    st = runtime.governor.state("t1")
    assert st.n_hint_obs == 1 and st.n_bare_obs == 1
    assert st.sr_hint_ema == pytest.approx(1.0)
    assert st.sr_bare_ema == pytest.approx(0.0)
    assert metrics["sr_hint_batch"] == pytest.approx(1.0)
    assert metrics["sr_bare_batch"] == pytest.approx(0.0)


def test_hint_book_fail_fast(tmp_path):
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {
                        "file": str(tmp_path / "missing.json"),
                        "require_manifest": True,
                    },
                }
            }
        }
    )
    with pytest.raises(FileNotFoundError):
        CatalystRuntime(config)
    # 缺 hints.file 同样 fail-fast
    config2 = OmegaConf.create(
        {"exp_manager": {"catalyst": {"enable": True, "hints": {}}}}
    )
    with pytest.raises(RuntimeError):
        CatalystRuntime(config2)
