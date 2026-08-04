import math
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from agentevolver.module.trainer.ae_ray_trainer import (
    _expected_onpolicy_behavior_mask,
    _length_truncation_batch_diagnostics,
    _rollout_drift_gate_violations,
    _rollout_drift_metrics_by_relative_position,
    _rollout_drift_nonfinite_fields,
    _should_skip_zero_advantage_grpo_actor_update,
    AgentEvolverRayPPOTrainer,
)


def test_identity_failure_persists_actor_export_and_training_checkpoint(tmp_path):
    trainer = AgentEvolverRayPPOTrainer.__new__(AgentEvolverRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "rollout_data_dir": str(tmp_path / "rollouts"),
                "export_actor_weights_on_identity_failure": True,
                "save_checkpoint_on_identity_failure": True,
            }
        }
    )
    trainer.global_steps = 7
    trainer.actor_rollout_wg = Mock()
    trainer._save_checkpoint = Mock()

    trainer._persist_identity_failure_training_state()

    trainer.actor_rollout_wg.save_rollout_weights.assert_called_once_with(
        str(tmp_path / "rollouts/identity_gate_failure_step_7_actor_weights")
    )
    trainer._save_checkpoint.assert_called_once_with()


def test_rollout_drift_metrics_use_per_response_relative_position():
    signed_delta = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [-0.1, -0.2, -0.3, 9.0, 9.0, 9.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    # Excluding row 1's mid token must not shift its late token into mid.
    identity_mask = response_mask.clone()
    identity_mask[1, 1] = False

    metrics = _rollout_drift_metrics_by_relative_position(
        signed_delta,
        response_mask,
        identity_mask,
        clip_low=0.8,
        clip_high=1.2,
    )

    expected = {
        "early": torch.tensor([0.1, 0.2, -0.1]),
        "mid": torch.tensor([0.3, 0.4]),
        "late": torch.tensor([0.5, 0.6, -0.3]),
    }
    expected_outside_clip = {"early": 1.0 / 3.0, "mid": 1.0, "late": 1.0}
    for bucket_name, values in expected.items():
        prefix = f"training/rollout_logprob_drift_{bucket_name}"
        assert metrics[f"{prefix}_tokens"] == float(values.numel())
        assert metrics[f"{prefix}_signed_mean"] == pytest.approx(
            values.mean().item()
        )
        assert metrics[f"{prefix}_abs_mean"] == pytest.approx(
            values.abs().mean().item()
        )
        assert metrics[f"{prefix}_abs_max"] == pytest.approx(
            values.abs().max().item()
        )
        assert metrics[f"{prefix}_ratio_mean"] == pytest.approx(
            torch.exp(values).mean().item()
        )
        assert metrics[f"{prefix}_ratio_outside_clip_fraction"] == pytest.approx(
            expected_outside_clip[bucket_name]
        )


def test_rollout_drift_metrics_short_and_empty_buckets_are_safe():
    signed_delta = torch.tensor([[0.25, 7.0]])
    response_mask = torch.tensor([[1, 0]], dtype=torch.bool)
    identity_mask = response_mask.clone()

    metrics = _rollout_drift_metrics_by_relative_position(
        signed_delta,
        response_mask,
        identity_mask,
        clip_low=0.8,
        clip_high=1.2,
    )

    early = "training/rollout_logprob_drift_early"
    mid = "training/rollout_logprob_drift_mid"
    late = "training/rollout_logprob_drift_late"
    assert metrics[f"{early}_tokens"] == 0.0
    assert metrics[f"{mid}_tokens"] == 1.0
    assert metrics[f"{late}_tokens"] == 0.0
    assert f"{early}_abs_mean" not in metrics
    assert metrics[f"{mid}_signed_mean"] == pytest.approx(0.25)
    assert metrics[f"{mid}_ratio_mean"] == pytest.approx(math.exp(0.25))
    assert f"{late}_abs_mean" not in metrics


def test_rollout_drift_metrics_ignore_nonfinite_values_outside_identity_mask():
    signed_delta = torch.tensor([[0.1, float("nan"), -0.2]])
    response_mask = torch.ones_like(signed_delta, dtype=torch.bool)
    identity_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    metrics = _rollout_drift_metrics_by_relative_position(
        signed_delta,
        response_mask,
        identity_mask,
        clip_low=0.8,
        clip_high=1.2,
    )

    numeric_metrics = [
        value
        for name, value in metrics.items()
        if not name.endswith("_tokens")
    ]
    assert numeric_metrics
    assert all(math.isfinite(value) for value in numeric_metrics)


def test_rollout_drift_metrics_validate_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        _rollout_drift_metrics_by_relative_position(
            torch.zeros(1, 3),
            torch.ones(1, 2),
            torch.ones(1, 2),
            clip_low=0.8,
            clip_high=1.2,
        )


def test_rollout_drift_gate_allows_values_at_threshold_boundary():
    violations = _rollout_drift_gate_violations(
        mean_abs_diff=0.01,
        max_abs_diff=0.25,
        p99_abs_diff=0.1,
        importance_ratio_outside_clip_fraction=0.02,
        mean_threshold=0.01,
        max_threshold=0.25,
        p99_threshold=0.1,
        importance_ratio_outside_clip_threshold=0.02,
    )

    assert violations == []


def test_rollout_drift_gate_negative_thresholds_disable_checks():
    violations = _rollout_drift_gate_violations(
        mean_abs_diff=100.0,
        max_abs_diff=100.0,
        p99_abs_diff=100.0,
        importance_ratio_outside_clip_fraction=1.0,
        mean_threshold=-1.0,
        max_threshold=-1.0,
        p99_threshold=-1.0,
        importance_ratio_outside_clip_threshold=-1.0,
    )

    assert violations == []


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf")])
def test_rollout_drift_gate_rejects_nonfinite_thresholds(bad_threshold):
    with pytest.raises(ValueError, match="thresholds must be finite"):
        _rollout_drift_gate_violations(
            mean_abs_diff=0.0,
            max_abs_diff=0.0,
            p99_abs_diff=0.0,
            importance_ratio_outside_clip_fraction=0.0,
            mean_threshold=bad_threshold,
            max_threshold=1.0,
            p99_threshold=1.0,
            importance_ratio_outside_clip_threshold=1.0,
        )


def test_expected_behavior_mask_excludes_replay_teacher_and_nonloss_tokens():
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.bool
    )
    loss_mask = torch.tensor(
        [[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool
    )
    extras = [
        {"is_experience_replay": False, "is_teacher": False},
        {"is_experience_replay": True, "is_teacher": False},
        {"is_experience_replay": False, "is_teacher": True},
    ]

    expected = _expected_onpolicy_behavior_mask(
        response_mask=response_mask,
        response_loss_mask=loss_mask,
        sample_extras=extras,
    )

    assert expected.tolist() == [
        [True, False, True],
        [False, False, False],
        [False, False, False],
    ]


def test_length_truncation_batch_diagnostics_use_decision_denominator():
    diagnostics = _length_truncation_batch_diagnostics(
        decision_count=torch.tensor([3, 1, 2]),
        length_truncated_decision_count=torch.tensor([1, 0, 0]),
        has_length_truncated_decision=torch.tensor([True, False, False]),
    )

    assert diagnostics["rollout/decision_count"] == 6.0
    assert diagnostics["rollout/length_truncated_decision_count"] == 1.0
    assert diagnostics[
        "rollout/length_truncated_decision_fraction"
    ] == pytest.approx(1.0 / 6.0)
    assert diagnostics[
        "rollout/length_truncated_sample_fraction"
    ] == pytest.approx(1.0 / 3.0)


def test_length_truncation_diagnostics_reject_inconsistent_counts():
    with pytest.raises(ValueError, match="cannot exceed"):
        _length_truncation_batch_diagnostics(
            decision_count=torch.tensor([1]),
            length_truncated_decision_count=torch.tensor([2]),
            has_length_truncated_decision=torch.tensor([True]),
        )


def test_rollout_drift_gate_reports_all_violations_in_stable_order():
    violations = _rollout_drift_gate_violations(
        mean_abs_diff=0.02,
        max_abs_diff=0.3,
        p99_abs_diff=0.15,
        importance_ratio_outside_clip_fraction=0.05,
        mean_threshold=0.01,
        max_threshold=0.25,
        p99_threshold=0.1,
        importance_ratio_outside_clip_threshold=0.02,
    )

    assert violations == [
        ("mean_abs_diff", 0.02, 0.01),
        ("max_abs_diff", 0.3, 0.25),
        ("p99_abs_diff", 0.15, 0.1),
        ("importance_ratio_outside_clip_fraction", 0.05, 0.02),
    ]


def test_rollout_drift_nonfinite_check_ignores_unselected_tokens():
    rollout = torch.tensor([[-1.0, float("nan"), -2.0]])
    current = torch.tensor([[-1.0, float("inf"), -2.0]])
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    fields = _rollout_drift_nonfinite_fields(
        rollout_log_probs=rollout,
        current_log_probs=current,
        identity_mask=mask,
        signed_logprob_delta=torch.zeros(2),
        importance_ratio=torch.ones(2),
        aggregate_statistics={"mean": torch.tensor(0.0)},
    )

    assert fields == {}


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("rollout", {"rollout_log_probs": 1}),
        ("current", {"current_log_probs": 1}),
        ("delta", {"signed_logprob_delta": 1}),
        ("ratio", {"importance_ratio": 1}),
        ("aggregate", {"aggregate/p99": 1}),
    ],
)
def test_rollout_drift_nonfinite_check_is_fail_closed(field, expected):
    rollout = torch.tensor([[-1.0, -2.0]])
    current = torch.tensor([[-1.0, -2.0]])
    delta = torch.zeros(2)
    ratio = torch.ones(2)
    aggregates = {"p99": torch.tensor(0.0)}
    if field == "rollout":
        rollout[0, 0] = float("nan")
    elif field == "current":
        current[0, 0] = float("inf")
    elif field == "delta":
        delta[0] = float("nan")
    elif field == "ratio":
        ratio[0] = float("inf")
    else:
        aggregates["p99"] = torch.tensor(float("nan"))

    fields = _rollout_drift_nonfinite_fields(
        rollout_log_probs=rollout,
        current_log_probs=current,
        identity_mask=torch.ones_like(rollout, dtype=torch.bool),
        signed_logprob_delta=delta,
        importance_ratio=ratio,
        aggregate_statistics=aggregates,
    )

    assert fields == expected


def test_zero_advantage_pure_grpo_guard_skips_optimizer_and_adamw_decay():
    advantages = torch.tensor([[0.0, 0.0, 99.0]])
    effective_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    should_skip, stats = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="grpo",
        advantages=advantages,
        effective_mask=effective_mask,
    )

    # A zero-gradient AdamW step still changes a parameter through weight
    # decay; the trainer-side guard avoids calling that step at all.
    actor = torch.nn.Parameter(torch.tensor([1.0]))
    reference = actor.detach().clone()
    optimizer = torch.optim.AdamW([actor], lr=0.1, weight_decay=0.1)
    if not should_skip:
        actor.grad = torch.zeros_like(actor)
        optimizer.step()

    assert should_skip
    assert stats["is_pure_grpo"] == 1.0
    assert stats["effective_tokens"] == 2.0
    assert stats["effective_advantage_abs_max"] == 0.0
    assert torch.equal(actor.detach(), reference)


def test_zero_advantage_guard_includes_values_equal_to_zero_threshold():
    advantages = torch.tensor([[1.0e-6]])
    mask = torch.ones_like(advantages, dtype=torch.bool)

    at_boundary, _ = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="grpo",
        advantages=advantages,
        effective_mask=mask,
        zero_atol=1.0e-6,
    )
    above_boundary, _ = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="grpo",
        advantages=torch.tensor([[1.0001e-6]]),
        effective_mask=mask,
        zero_atol=1.0e-6,
    )

    assert at_boundary
    assert not above_boundary


@pytest.mark.parametrize("auxiliary_name", ["exp_mask", "teacher_mask"])
def test_zero_advantage_guard_does_not_classify_auxiliary_batch_as_pure(
    auxiliary_name,
):
    kwargs = {auxiliary_name: torch.tensor([[1, 0]], dtype=torch.bool)}
    should_skip, stats = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="grpo",
        advantages=torch.zeros(1, 2),
        effective_mask=torch.ones(1, 2, dtype=torch.bool),
        **kwargs,
    )

    assert not should_skip
    assert stats["is_pure_grpo"] == 0.0
    assert stats["has_auxiliary_tokens"] == 1.0


def test_zero_advantage_guard_allows_informative_grpo_and_other_estimators():
    mask = torch.ones(1, 2, dtype=torch.bool)
    informative, _ = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="grpo",
        advantages=torch.tensor([[0.0, 0.1]]),
        effective_mask=mask,
    )
    other_estimator, _ = _should_skip_zero_advantage_grpo_actor_update(
        adv_estimator="gae",
        advantages=torch.zeros(1, 2),
        effective_mask=mask,
    )

    assert not informative
    assert not other_estimator


def test_zero_advantage_guard_hard_stops_nonfinite_effective_advantage():
    with pytest.raises(RuntimeError, match="non-finite effective advantages"):
        _should_skip_zero_advantage_grpo_actor_update(
            adv_estimator="grpo",
            advantages=torch.tensor([[float("nan")]]),
            effective_mask=torch.ones(1, 1, dtype=torch.bool),
        )
