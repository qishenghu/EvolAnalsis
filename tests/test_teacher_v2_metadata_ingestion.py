"""Focused contracts for OpenRouter teacher-v2 JSON deserialization.

The production snapshot replay lives at the env-manager boundary.  These tests
cover the loader-side invariant independently: no audit field may be discarded
before that replay sees the trajectory, and API data must never claim behavior
log-probabilities.
"""

import copy

import pytest

from agentevolver.module.exp_manager.exp_manager import ExperienceManager


def _v2_record():
    messages = [
        {"role": "system", "content": "act carefully"},
        {"role": "user", "content": "initial observation"},
        {
            "role": "assistant",
            "content": "<think>reason</think>\n<action>look</action>",
        },
        {"role": "user", "content": "done"},
    ]
    trace = [
        {
            "step_index": 0,
            "prompt_messages_sha256": "b" * 64,
            "prompt_token_ids_sha256": "c" * 64,
            "raw_prompt_token_ids_sha256": "d" * 64,
            "completion_token_ids_sha256": "e" * 64,
            "completion_content": messages[2]["content"],
        }
    ]
    return {
        "schema_version": "openrouter_teacher_trajectory_v2",
        "contract_sha256": "a" * 64,
        "task_id": "17",
        "data_id": "17",
        "rollout_id": "alfworld:17:deepseek-v4-flash:3",
        "messages": messages,
        "query": "put the apple away",
        "reward": 0.75,
        "success": True,
        "success_rate": 0.75,
        "is_terminated": True,
        "teacher_model": "deepseek/deepseek-v4-flash",
        "decision_trace": trace,
        # A v2 API record cannot gain behavior-policy provenance through stale
        # or malicious legacy fields.
        "log_probs": [-0.1],
        "metadata": {
            "has_log_prob": True,
            "old_log_probs": [-0.2],
            "nested": {"kept": True},
        },
    }


def _decode(record):
    return ExperienceManager._dict_to_teacher_trajectory(None, record)


def test_v2_loader_preserves_snapshot_audit_metadata_and_raw_transcript():
    source = _v2_record()
    expected_messages = copy.deepcopy(source["messages"])
    expected_trace = copy.deepcopy(source["decision_trace"])

    trajectory = _decode(source)

    assert trajectory.task_id == "17"
    assert trajectory.data_id == "17"
    assert trajectory.rollout_id == source["rollout_id"]
    assert trajectory.query == source["query"]
    assert trajectory.steps == expected_messages
    assert trajectory.is_terminated is True
    assert trajectory.reward.outcome == pytest.approx(0.75)
    assert trajectory.reward.success_rate == pytest.approx(0.75)

    metadata = trajectory.metadata
    assert metadata["schema_version"] == source["schema_version"]
    assert metadata["contract_sha256"] == source["contract_sha256"]
    assert metadata["contract_sha"] == source["contract_sha256"]
    assert metadata["decision_trace"] == expected_trace
    assert metadata["raw_messages"] == expected_messages
    assert metadata["rollout_id"] == source["rollout_id"]
    assert metadata["is_terminated"] is True
    assert metadata["success_rate"] == pytest.approx(0.75)
    assert metadata["teacher_model"] == "deepseek/deepseek-v4-flash"
    assert metadata["is_teacher"] is True
    assert metadata["is_experience_replay"] is True
    assert metadata["has_log_prob"] is False
    assert "log_probs" not in metadata
    assert "old_log_probs" not in metadata

    # Deserialization freezes the record rather than retaining aliases into a
    # caller-owned JSON object.
    source["messages"][0]["content"] = "tampered"
    source["decision_trace"][0]["completion_content"] = "tampered"
    source["metadata"]["nested"]["kept"] = False
    assert trajectory.steps == expected_messages
    assert metadata["raw_messages"] == expected_messages
    assert metadata["decision_trace"] == expected_trace
    assert metadata["nested"] == {"kept": True}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(contract_sha256="not-a-sha"),
        lambda record: record.update(decision_trace=[]),
        lambda record: record.update(messages=[]),
    ],
)
def test_v2_loader_fails_closed_on_missing_audit_contract(mutation):
    record = _v2_record()
    mutation(record)
    with pytest.raises(ValueError, match="openrouter teacher v2"):
        _decode(record)


def test_legacy_teacher_loader_remains_compatible():
    record = {
        "task_id": "legacy-task",
        "rollout_id": "legacy-rollout",
        "messages": [
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": "Action: look"},
        ],
        "reward": 1.0,
        "success": True,
        "log_probs": [-0.5, -0.25],
        "log_probs_per_turn": [
            {"token_ids": [11, 12], "log_probs": [-0.5, -0.25]}
        ],
        "metadata": {"policy_version": 4},
    }

    trajectory = _decode(record)

    assert trajectory.task_id == "legacy-task"
    assert trajectory.rollout_id == "legacy-rollout"
    assert trajectory.steps == record["messages"]
    assert trajectory.reward.outcome == pytest.approx(1.0)
    assert trajectory.reward.success_rate == pytest.approx(1.0)
    assert trajectory.metadata["is_teacher"] is True
    assert trajectory.metadata["is_experience_replay"] is True
    assert trajectory.metadata["has_log_prob"] is True
    assert trajectory.metadata["log_probs"] == [-0.5, -0.25]
    assert trajectory.metadata["log_probs_per_turn"] == record[
        "log_probs_per_turn"
    ]
    assert "schema_version" not in trajectory.metadata

