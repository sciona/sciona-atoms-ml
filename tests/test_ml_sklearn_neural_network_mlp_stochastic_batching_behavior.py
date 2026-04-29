from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.neural_network.mlp_stochastic_batching import (
    mlp_stochastic_accumulated_loss,
    mlp_stochastic_batch_indices,
    mlp_stochastic_batches_per_epoch,
    mlp_stochastic_sample_indices,
    mlp_stochastic_stratify_targets,
)


def test_mlp_stochastic_stratify_targets_matches_classifier_rule() -> None:
    y = np.array([[True], [False], [True]], dtype=bool)
    observed = mlp_stochastic_stratify_targets(y, is_classifier=True, n_outputs=1)
    assert np.array_equal(observed, y)

    assert mlp_stochastic_stratify_targets(y, is_classifier=False, n_outputs=1) is None
    assert mlp_stochastic_stratify_targets(y, is_classifier=True, n_outputs=2) is None


def test_mlp_stochastic_sample_indices_are_zero_based() -> None:
    observed = mlp_stochastic_sample_indices(5)
    assert np.array_equal(observed, np.arange(5, dtype=np.int64))


def test_mlp_stochastic_batches_per_epoch_uses_ceiling_division() -> None:
    assert mlp_stochastic_batches_per_epoch(10, 4) == 3
    assert mlp_stochastic_batches_per_epoch(12, 4) == 3


def test_mlp_stochastic_batch_indices_follow_shuffle_mode() -> None:
    sample_indices = mlp_stochastic_sample_indices(6)
    observed_unshuffled = mlp_stochastic_batch_indices(
        sample_indices,
        batch_start=2,
        batch_stop=5,
        shuffle=False,
    )
    assert np.array_equal(observed_unshuffled, np.array([2, 3, 4], dtype=np.int64))

    shuffled = np.array([4, 1, 5, 0, 3, 2], dtype=np.int64)
    observed_shuffled = mlp_stochastic_batch_indices(
        shuffled,
        batch_start=1,
        batch_stop=4,
        shuffle=True,
    )
    assert np.array_equal(observed_shuffled, np.array([1, 5, 0], dtype=np.int64))


def test_mlp_stochastic_accumulated_loss_weights_by_batch_size() -> None:
    observed = mlp_stochastic_accumulated_loss(
        3.5,
        0.25,
        batch_start=2,
        batch_stop=6,
    )
    assert observed == pytest.approx(4.5)


def test_mlp_stochastic_batching_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        mlp_stochastic_sample_indices(0)

    with pytest.raises((ViolationError, ValueError)):
        mlp_stochastic_batches_per_epoch(5, 0)

    with pytest.raises((ViolationError, ValueError)):
        mlp_stochastic_batch_indices(
            np.arange(5, dtype=np.int64),
            batch_start=4,
            batch_stop=4,
            shuffle=True,
        )

    with pytest.raises((ViolationError, ValueError)):
        mlp_stochastic_accumulated_loss(
            1.0,
            0.5,
            batch_start=3,
            batch_stop=2,
        )
