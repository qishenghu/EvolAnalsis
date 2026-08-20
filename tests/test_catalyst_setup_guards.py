"""互斥断言与真实素材加载:trainer._catalyst_setup + CatalystRuntime 实件。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agentevolver.module.exp_manager.catalyst import CatalystRuntime
from agentevolver.module.trainer.ae_ray_trainer import AgentEvolverRayPPOTrainer

REPO = Path(__file__).resolve().parents[1]
AF_HINTS = REPO / "data" / "catalyst_hints" / "alfworld_dsv4flash.json"


def make_setup_trainer(*, conflicts=None, replay=True, actor_bc=True):
    conflicts = conflicts or {}
    trainer = object.__new__(AgentEvolverRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "exp_manager": {
                "teacher_experience": {
                    "enable": conflicts.get("teacher", False)
                },
                "experience_replay": {
                    "enable": conflicts.get("exp_replay", False)
                },
                "repo": {"enable": conflicts.get("repo", False)},
                "state_channel": {"enable": conflicts.get("sc", False)},
            },
            "actor_rollout_ref": {
                "rollout": {"foo": 1},
                "actor": {
                    "use_chord": conflicts.get("chord", False),
                    "use_dr3": conflicts.get("dr3", False),
                    "use_dapo": conflicts.get("dapo", False),
                    "catalyst": {"replay_bc": {"enable": actor_bc}},
                },
            },
            "algorithm": {
                "dapo": {"enable": conflicts.get("algo_dapo", False)},
                "grpo": {
                    "teacher_baseline_separation": {
                        "enable": conflicts.get("tbs", False)
                    }
                },
            },
            "trainer": {"default_local_dir": "/tmp/catalyst-test"},
        }
    )
    calls = {}
    trainer.exp_manager = SimpleNamespace(
        catalyst=SimpleNamespace(
            replay_enabled=replay,
            entry_enabled=False,
            arm_baseline_enabled=True,
            thermostat_enabled=False,
            attach_renderer=lambda tok, cfg: calls.setdefault("renderer", (tok, cfg)),
            state_path=lambda base: f"{base}/catalyst_state/governor_latest.json",
            # v2 契约:trainer 统一走 runtime 的 load/save_persistent_state
            load_persistent_state=lambda path: calls.setdefault("loaded", path),
            save_persistent_state=lambda path: calls.setdefault("saved", path),
            governor=SimpleNamespace(
                load_state=lambda path: calls.setdefault("loaded", path) and False
            ),
        )
    )
    trainer.tokenizer = object()
    return trainer, calls


def test_setup_noop_when_disabled():
    trainer = object.__new__(AgentEvolverRayPPOTrainer)
    trainer.exp_manager = SimpleNamespace(catalyst=None)
    trainer._catalyst_setup()
    assert trainer._catalyst is None


@pytest.mark.parametrize(
    "conflict_key",
    ["teacher", "exp_replay", "repo", "sc", "chord", "dr3", "dapo",
     "algo_dapo", "tbs"],
)
def test_setup_mutual_exclusion(conflict_key):
    trainer, _ = make_setup_trainer(conflicts={conflict_key: True})
    with pytest.raises(RuntimeError, match="mutual-exclusion"):
        trainer._catalyst_setup()


def test_setup_replay_flag_consistency():
    trainer, _ = make_setup_trainer(replay=True, actor_bc=False)
    with pytest.raises(RuntimeError, match="replay_bc"):
        trainer._catalyst_setup()
    trainer2, calls2 = make_setup_trainer(replay=False, actor_bc=False)
    trainer2._catalyst_setup()  # 一致 → 通过并挂载渲染器
    assert "renderer" in calls2


def test_setup_happy_path_attaches_renderer_and_state():
    trainer, calls = make_setup_trainer()
    trainer._catalyst_setup()
    assert trainer._catalyst is trainer.exp_manager.catalyst
    assert calls["renderer"][0] is trainer.tokenizer
    assert trainer._catalyst_state_path.endswith(
        "catalyst_state/governor_latest.json"
    )


@pytest.mark.skipif(not AF_HINTS.is_file(), reason="built hints not present")
def test_runtime_loads_real_alfworld_hints():
    config = OmegaConf.create(
        {
            "exp_manager": {
                "catalyst": {
                    "enable": True,
                    "hints": {"file": str(AF_HINTS), "require_manifest": True},
                    "replay": {"enable": False},
                }
            }
        }
    )
    runtime = CatalystRuntime(config)
    assert len(runtime.hint_book) == 1437
    hint = runtime.hint_book.get("1215")
    assert hint and len(hint) <= 5000
    # 试点回归锚(F9):AF 1215 的开头逐字节
    assert hint.startswith("I need to find a soapbar and make it clean")
