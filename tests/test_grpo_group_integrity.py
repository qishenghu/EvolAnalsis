from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agentevolver.module.env_manager.env_manager import (
    ParallelEnvManager,
    _align_samples_by_complete_uid_groups,
)


def _sample(
    uid,
    *,
    replay=False,
    teacher=False,
    rollout_id=0,
    snapshot=False,
    expected_rollouts=None,
):
    return SimpleNamespace(
        data_id=str(uid),
        rollout_id=str(rollout_id),
        extras={
            "is_experience_replay": replay,
            "is_teacher": teacher,
            "rollout_mode": None if (replay or teacher) else "sample",
            "snapshot_training": snapshot,
            "expected_group_rollouts": expected_rollouts,
        },
    )


def _group(uid, count, **kwargs):
    return [_sample(uid, rollout_id=i, **kwargs) for i in range(count)]


def _counts(samples):
    result = {}
    for sample in samples:
        result[sample.data_id] = result.get(sample.data_id, 0) + 1
    return result


def _manager(*, world_size=4, rollout_n=2):
    manager = object.__new__(ParallelEnvManager)
    manager.config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": world_size, "nnodes": 1},
            "algorithm": {"adv_estimator": "grpo"},
        }
    )
    manager.rollout_n = rollout_n
    manager.get_extra = lambda cmt: dict(cmt.extras)
    return manager


def _cmt(
    samples,
    *,
    mode="sample",
    replay=False,
    teacher=False,
    snapshot=False,
    expected_rollouts=None,
):
    extras = {
        "rollout_mode": mode,
        "is_experience_replay": replay,
        "is_teacher": teacher,
        "snapshot_training": snapshot,
        "expected_group_rollouts": expected_rollouts,
    }
    return SimpleNamespace(
        extras=extras,
        group_tokenize=lambda: samples,
    )


def test_complete_pure_grpo_groups_are_unchanged():
    samples = _group("a", 2) + _group("b", 2)

    aligned = _align_samples_by_complete_uid_groups(
        samples, world_size=4, expected_group_size=2
    )

    assert aligned == samples
    assert _counts(aligned) == {"a": 2, "b": 2}


def test_pure_grpo_trim_drops_only_a_complete_uid_group():
    samples = _group("a", 2) + _group("b", 2) + _group("c", 2)

    aligned = _align_samples_by_complete_uid_groups(
        samples, world_size=4, expected_group_size=2
    )

    assert [sample.data_id for sample in aligned] == ["a", "a", "b", "b"]
    assert _counts(aligned) == {"a": 2, "b": 2}


def test_multi_group_alignment_keeps_maximum_deterministic_subset():
    # With n=3 and world_size=4, complete groups must be retained four at a
    # time. Five input groups therefore become the first four groups (12
    # samples), never eleven samples from five partial groups.
    samples = []
    for uid in "abcde":
        samples.extend(_group(uid, 3))

    aligned = _align_samples_by_complete_uid_groups(
        samples, world_size=4, expected_group_size=3
    )

    assert len(aligned) == 12
    assert _counts(aligned) == {uid: 3 for uid in "abcd"}


def test_missing_pure_grpo_sample_fails_before_alignment():
    samples = _group("complete", 2) + _group("missing-one", 1)

    with pytest.raises(RuntimeError, match=r"rollout\.n=2.*missing-one.*count=1"):
        _align_samples_by_complete_uid_groups(
            samples, world_size=2, expected_group_size=2
        )


def test_trajectories_to_samples_enforces_n_for_pure_training_batch():
    manager = _manager(world_size=2, rollout_n=2)
    cmts = [
        _cmt(_group("complete", 2)),
        _cmt(_group("missing-one", 1)),
    ]

    with pytest.raises(
        RuntimeError,
        match=r"missing-one.*unique_rollouts=1,expected=2",
    ):
        manager.trajectories_to_samples(cmts)


def test_mixed_teacher_replay_path_does_not_inherit_pure_n_assertion():
    # Heterogeneous group sizes are valid for a mixed path. Alignment may
    # remove groups, but it must retain/drop every UID atomically.
    samples = (
        _group("student-plus-teacher", 3, teacher=True)
        + _group("replay-group", 3, replay=True)
        + _group("student-group", 2)
        + _group("trailing-replay", 1, replay=True)
    )

    aligned = _align_samples_by_complete_uid_groups(
        samples,
        world_size=4,
        expected_group_size=None,
        require_divisible=True,
    )

    assert len(aligned) == 8
    assert _counts(aligned) == {
        "student-plus-teacher": 3,
        "replay-group": 3,
        "student-group": 2,
    }


def test_trajectories_to_samples_does_not_assert_n_for_mixed_batch():
    manager = _manager(world_size=4, rollout_n=2)
    cmts = [
        _cmt(_group("student", 2)),
        _cmt(_group("replay", 1, replay=True), replay=True),
        _cmt(_group("teacher", 1, teacher=True), teacher=True),
    ]

    aligned = manager.trajectories_to_samples(cmts)

    assert _counts(aligned) == {"student": 2, "replay": 1, "teacher": 1}


def test_training_fails_if_no_nonempty_complete_group_subset_can_align():
    samples = _group("a", 2) + _group("b", 2)

    with pytest.raises(RuntimeError, match="refusing to enter the optimizer"):
        _align_samples_by_complete_uid_groups(
            samples,
            world_size=8,
            expected_group_size=2,
            require_divisible=True,
        )


def test_small_validation_batch_remains_supported():
    samples = _group("validation", 1)

    aligned = _align_samples_by_complete_uid_groups(
        samples,
        world_size=4,
        expected_group_size=None,
        require_divisible=False,
    )

    assert aligned == samples


def test_validation_never_trims_when_a_divisible_subset_exists():
    samples = sum((_group(str(i), 1) for i in range(5)), [])

    aligned = _align_samples_by_complete_uid_groups(
        samples,
        world_size=4,
        expected_group_size=None,
        require_divisible=False,
    )

    assert aligned == samples
    assert len(aligned) == 5


def test_explicit_optimizer_mode_rejects_nondivisible_teacher_only_batch():
    manager = _manager(world_size=4, rollout_n=2)
    cmts = [
        _cmt(_group("teacher-a", 1, teacher=True), teacher=True),
        _cmt(_group("teacher-b", 1, teacher=True), teacher=True),
    ]

    with pytest.raises(RuntimeError, match="refusing to enter the optimizer"):
        manager.trajectories_to_samples(cmts, optimizer_batch=True)


def test_legacy_multi_sample_rollouts_check_unique_rollouts_not_sample_count():
    manager = _manager(world_size=4, rollout_n=2)
    samples = [
        _sample("a", rollout_id=0),
        _sample("a", rollout_id=0),
        _sample("a", rollout_id=1),
        _sample("a", rollout_id=1),
    ]

    aligned = manager.trajectories_to_samples(
        [_cmt(samples)], optimizer_batch=True
    )

    assert aligned == samples
