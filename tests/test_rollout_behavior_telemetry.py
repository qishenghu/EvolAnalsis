import math
from types import SimpleNamespace

import torch

from agentevolver.module.trainer.ae_ray_trainer import (
    _canonical_task_family,
    _extract_telemetry_action,
    _rollout_behavior_batch_diagnostics,
    _task_family_reward_batch_diagnostics,
)
from agentevolver.module.env_manager.env_manager import ParallelEnvManager


def _tagged_message_row(*actions: str):
    messages = [
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "initial observation"},
    ]
    for action in actions:
        messages.append(
            {
                "role": "assistant",
                "content": f"<action>\n{action}\n</action>",
            }
        )
        messages.append({"role": "user", "content": "observation"})
    return {"messages": messages}


def test_rollout_behavior_metrics_record_additive_denominators():
    messages = [
        _tagged_message_row("look", "look", "look"),
        _tagged_message_row("go to fridge 1", "look", "look"),
        _tagged_message_row("inventory"),
        {"messages": [{"role": "assistant", "content": "no action tag"}]},
        {
            "messages": [
                {"role": "assistant", "content": "<action>look</action>"},
                {"role": "assistant", "content": "malformed selected turn"},
            ]
        },
        _tagged_message_row("look", "look", "look"),
        _tagged_message_row("look", "look", "look"),
    ]
    extras = [
        {},
        {},
        {},
        {},
        {},
        {"is_experience_replay": True},
        {"is_teacher": True},
    ]

    metrics = _rollout_behavior_batch_diagnostics(
        messages=messages,
        sample_extras=extras,
        action_format="react_tags",
    )

    assert metrics["rollout/behavior/onpolicy_sample_count"] == 5.0
    assert metrics["rollout/selected_action/parsed_count"] == 3.0
    assert metrics["rollout/selected_action/unparsed_count"] == 2.0
    assert metrics["rollout/selected_action/look_count"] == 2.0
    assert math.isclose(
        metrics["rollout/selected_action/look_fraction"], 2.0 / 3.0
    )
    assert metrics["rollout/action_history/sample_count"] == 4.0
    assert metrics["rollout/action_history/adjacent_pair_count"] == 4.0
    assert metrics["rollout/action_history/adjacent_repeat_count"] == 3.0
    assert math.isclose(
        metrics["rollout/action_history/adjacent_repeat_fraction"], 0.75
    )
    assert (
        metrics[
            "rollout/action_history/sample_has_repeat_run_ge3_count"
        ]
        == 1.0
    )
    assert math.isclose(
        metrics[
            "rollout/action_history/sample_has_repeat_run_ge3_fraction"
        ],
        1.0 / 4.0,
    )


def test_action_parser_uses_post_think_action_and_strict_formats():
    content = (
        "<think>candidate <action>look</action></think>\n"
        "<action>\nGo   To   Fridge 1\n</action>"
    )
    assert (
        _extract_telemetry_action(content, action_format="react_tags")
        == "go to fridge 1"
    )
    assert (
        _extract_telemetry_action(
            "<action>\nLOOK\n", action_format="react_tags"
        )
        == "look"
    )
    assert (
        _extract_telemetry_action(
            "Thought:\ncheck\nAction:\ngo to shelf 1", action_format="react"
        )
        == "go to shelf 1"
    )
    assert (
        _extract_telemetry_action(
            "arbitrary final line", action_format="react"
        )
        is None
    )


def test_task_family_metrics_use_only_canonical_env_metadata_and_no_task_ids():
    def extra(task_type=None, **flags):
        value = {"env_type": "alfworld", **flags}
        if task_type is not None:
            value["environment_task_type"] = task_type
        return value

    group_ids = [
        "901234",
        "901234",
        "901235",
        "901236",
        "901237",
        "901238",
        "901239",
        "901240",
        "901240",
    ]
    extras = [
        extra("pick_cool_then_place_in_recep"),
        extra("pick_cool_then_place_in_recep"),
        extra("pick_cool_then_place_in_recep"),
        extra("pick_clean_then_place_in_recep"),
        extra("new_unrecognized_family"),
        extra("pick_heat_then_place_in_recep", is_experience_replay=True),
        extra("pick_and_place_simple", is_teacher=True),
        extra("pick_heat_then_place_in_recep"),
        extra("pick_cool_then_place_in_recep"),
    ]
    rewards = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
         [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
         [0.0, 0.0]]
    )

    metrics = _task_family_reward_batch_diagnostics(
        group_ids=group_ids,
        sample_extras=extras,
        sample_rewards=rewards,
    )

    assert metrics["rollout/task_family/reward_sample_count"] == 7.0
    assert metrics["rollout/task_family/metadata_known_sample_count"] == 6.0
    assert metrics["rollout/task_family/known_sample_count"] == 4.0
    assert metrics["rollout/task_family/unknown_sample_count"] == 3.0
    assert math.isclose(
        metrics["rollout/task_family/known_sample_fraction"], 4.0 / 7.0
    )
    assert metrics["rollout/task_family/group_count"] == 5.0
    assert metrics["rollout/task_family/known_group_count"] == 3.0
    assert metrics["rollout/task_family/unknown_group_count"] == 2.0
    assert metrics["rollout/task_family/inconsistent_group_count"] == 1.0

    assert metrics["rollout/task_family/cool/sample_count"] == 3.0
    assert metrics["rollout/task_family/cool/reward_sum"] == 2.0
    assert math.isclose(
        metrics["rollout/task_family/cool/sample_reward_mean"], 2.0 / 3.0
    )
    assert metrics["rollout/task_family/cool/group_count"] == 2.0
    assert metrics["rollout/task_family/cool/group_reward_mean_sum"] == 1.5
    assert metrics["rollout/task_family/cool/group_reward_mean"] == 0.75
    assert metrics["rollout/task_family/clean/group_count"] == 1.0
    assert "rollout/task_family/heat/sample_count" not in metrics
    assert not any(group_id in key for key in metrics for group_id in group_ids)


def test_task_family_mapping_fails_closed_without_exact_alfworld_metadata():
    assert (
        _canonical_task_family(
            {
                "env_type": "alfworld",
                "environment_task_type": "pick_cool_then_place_in_recep",
            }
        )
        == "cool"
    )
    assert (
        _canonical_task_family(
            {
                "env_type": "alfworld",
                "environment_task_type": "cool some object",
            }
        )
        is None
    )
    assert (
        _canonical_task_family(
            {
                "env_type": "webshop",
                "environment_task_type": "pick_cool_then_place_in_recep",
            }
        )
        is None
    )


def test_env_extras_preserve_trusted_task_type_for_future_batches():
    cmt = SimpleNamespace(
        task_id="1371",
        rollout_id="0",
        metadata={
            "env_type": "alfworld",
            "environment_task_type": "pick_cool_then_place_in_recep",
        },
        decision_snapshots=[],
    )

    extras = ParallelEnvManager.get_extra(None, cmt)

    assert extras["env_type"] == "alfworld"
    assert (
        extras["environment_task_type"]
        == "pick_cool_then_place_in_recep"
    )
    assert _canonical_task_family(extras) == "cool"
