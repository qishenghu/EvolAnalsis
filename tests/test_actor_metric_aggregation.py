from unittest.mock import sentinel

import pytest
import torch
from verl.trainer.ppo.core_algos import agg_loss

from agentevolver.module.exp_manager.het_actor import (
    _distributed_normalized_sum,
    _distributed_weighted_mean,
    _loss_metric_sum_and_weight,
)


def _combine(parts):
    metric_sum = sum((part[0] for part in parts), start=torch.tensor(0.0))
    metric_weight = sum((part[1] for part in parts), start=torch.tensor(0.0))
    return _distributed_weighted_mean(metric_sum, metric_weight)


@pytest.mark.parametrize(
    "mode",
    [
        "token-mean",
        "seq-mean-token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-sum-norm",
    ],
)
def test_metric_aggregation_matches_upstream_full_batch_contract(
    monkeypatch,
    mode,
):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    loss_mat = torch.tensor(
        [
            [2.0, 99.0, 99.0],
            [3.0, 6.0, 9.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 1],
        ]
    )
    metric_sum, metric_normalizer = _loss_metric_sum_and_weight(
        loss_mat,
        loss_mask,
        mode,
    )
    if mode == "seq-mean-token-sum-norm":
        actual = _distributed_normalized_sum(metric_sum, metric_normalizer)
    else:
        actual = _distributed_weighted_mean(metric_sum, metric_normalizer)

    expected = agg_loss(loss_mat, loss_mask, mode).item()
    assert actual == pytest.approx(expected)


def test_token_mean_metric_is_invariant_to_uneven_microbatches(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    short = _loss_metric_sum_and_weight(
        torch.tensor([[1.0, 3.0, 99.0]]),
        torch.tensor([[1, 1, 0]]),
        "token-mean",
    )
    long_value = _loss_metric_sum_and_weight(
        torch.tensor([[10.0, 99.0, 99.0]]),
        torch.tensor([[1, 0, 0]]),
        "token-mean",
    )

    assert _combine([short, long_value]) == pytest.approx(14.0 / 3.0)
    # The old mean-of-microbatch-scalars result was (2 + 10) / 2 == 6.
    assert _combine([short, long_value]) != pytest.approx(6.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("seq-mean-token-mean", 4.0),
        ("seq-mean-token-sum", 10.0),
    ],
)
def test_sequence_metric_modes_use_sequence_weight(monkeypatch, mode, expected):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    stats = _loss_metric_sum_and_weight(
        torch.tensor(
            [
                [2.0, 99.0, 99.0],
                [3.0, 6.0, 9.0],
            ]
        ),
        torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
        mode,
    )

    assert _combine([stats]) == pytest.approx(expected)


def test_sequence_sum_norm_matches_full_batch_agg_loss(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    stats = _loss_metric_sum_and_weight(
        torch.tensor(
            [
                [2.0, 99.0, 99.0],
                [3.0, 6.0, 9.0],
            ]
        ),
        torch.tensor(
            [
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
        "seq-mean-token-sum-norm",
    )

    # Upstream agg_loss is sum(sequence token sums) / response_width:
    # (2 + 3 + 6 + 9) / 3 == 20/3.  It does not divide by sequence count.
    assert _distributed_normalized_sum(*stats) == pytest.approx(20.0 / 3.0)


def test_sequence_sum_norm_is_microbatch_invariant(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    first = _loss_metric_sum_and_weight(
        torch.tensor([[2.0, 99.0, 99.0]]),
        torch.tensor([[1, 0, 0]]),
        "seq-mean-token-sum-norm",
    )
    second = _loss_metric_sum_and_weight(
        torch.tensor([[3.0, 6.0, 9.0]]),
        torch.tensor([[1, 1, 1]]),
        "seq-mean-token-sum-norm",
    )

    assert first[1].item() == second[1].item() == 3.0
    assert _distributed_normalized_sum(
        first[0] + second[0],
        first[1],
    ) == pytest.approx(20.0 / 3.0)


def test_sequence_sum_norm_averages_repeated_ppo_epochs(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    # Two full-batch passes have masked sums 20 and 10 at width=3.  The logged
    # value is mean(20/3, 10/3), not their sum.
    assert _distributed_normalized_sum(
        torch.tensor(30.0),
        torch.tensor(3.0 * 2),
    ) == pytest.approx(5.0)


def test_distributed_weighted_mean_reduces_sum_and_weight_on_dp_group(
    monkeypatch,
):
    observed = {}

    def fake_all_reduce(stats, *, op, group):
        observed["op"] = op
        observed["group"] = group
        # A second DP rank contributed sum=30 over weight=3.
        stats.add_(torch.tensor([30.0, 3.0], dtype=stats.dtype))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    actual = _distributed_weighted_mean(
        torch.tensor(4.0),
        torch.tensor(2.0),
        process_group=sentinel.dp_group,
    )

    assert actual == pytest.approx(34.0 / 5.0)
    assert observed == {
        "op": torch.distributed.ReduceOp.SUM,
        "group": sentinel.dp_group,
    }


def test_distributed_sequence_sum_norm_reduces_only_global_numerator(
    monkeypatch,
):
    observed = {}

    def fake_all_reduce(total, *, op, group):
        observed["op"] = op
        observed["group"] = group
        # A second DP rank contributes another masked token sum of 30.  The
        # shared response width remains 3 and must not be summed across ranks.
        total.add_(30.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    actual = _distributed_normalized_sum(
        torch.tensor(4.0),
        torch.tensor(3.0),
        process_group=sentinel.dp_group,
    )

    assert actual == pytest.approx(34.0 / 3.0)
    assert observed == {
        "op": torch.distributed.ReduceOp.SUM,
        "group": sentinel.dp_group,
    }


def test_loss_metric_stats_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        _loss_metric_sum_and_weight(
            torch.zeros(1, 3),
            torch.ones(1, 2),
            "token-mean",
        )
