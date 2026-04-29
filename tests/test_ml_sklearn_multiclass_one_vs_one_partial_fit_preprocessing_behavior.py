from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import _partial_fit_ovo_binary

from sciona.atoms.ml.sklearn.multiclass.one_vs_one_partial_fit_preprocessing import (
    one_vs_one_partial_fit_binary_targets,
    one_vs_one_partial_fit_estimator_count,
    one_vs_one_partial_fit_pair_mask,
    one_vs_one_partial_fit_subset_indices,
    one_vs_one_partial_fit_unknown_classes,
)


class RecordingBinaryEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, **kwargs) -> "RecordingBinaryEstimator":
        del classes, kwargs
        self.calls.append((np.asarray(X), np.asarray(y)))
        return self


def test_one_vs_one_partial_fit_preprocessing_atoms_import() -> None:
    assert callable(one_vs_one_partial_fit_estimator_count)
    assert callable(one_vs_one_partial_fit_unknown_classes)
    assert callable(one_vs_one_partial_fit_pair_mask)
    assert callable(one_vs_one_partial_fit_subset_indices)
    assert callable(one_vs_one_partial_fit_binary_targets)


def test_estimator_count_matches_one_vs_one_formula() -> None:
    classes = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    assert one_vs_one_partial_fit_estimator_count(classes) == 6


def test_unknown_classes_returns_sorted_unique_difference() -> None:
    y = np.array([20.0, 50.0, 10.0, 50.0, 60.0], dtype=np.float64)
    classes = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    observed = one_vs_one_partial_fit_unknown_classes(y, classes)

    assert np.array_equal(observed, np.array([50.0, 60.0], dtype=np.float64))


def test_pair_mask_subset_indices_and_binary_targets_match_private_helper_logic() -> None:
    X = np.arange(18, dtype=np.float64).reshape(6, 3)
    y = np.array([10.0, 20.0, 30.0, 20.0, 10.0, 40.0], dtype=np.float64)

    pair_mask = one_vs_one_partial_fit_pair_mask(y, 10.0, 20.0)
    subset_indices = one_vs_one_partial_fit_subset_indices(pair_mask)
    binary_targets = one_vs_one_partial_fit_binary_targets(y, 10.0, 20.0)

    estimator = RecordingBinaryEstimator()
    updated = _partial_fit_ovo_binary(estimator, X, y, 10.0, 20.0, partial_fit_params={})

    assert updated is estimator
    assert np.array_equal(pair_mask, np.array([True, True, False, True, True, False], dtype=np.bool_))
    assert np.array_equal(subset_indices, np.array([0, 1, 3, 4], dtype=np.int64))
    assert np.array_equal(binary_targets, np.array([0, 1, 1, 0], dtype=np.int64))
    assert len(estimator.calls) == 1
    seen_x, seen_y = estimator.calls[0]
    assert np.array_equal(seen_x, X[subset_indices])
    assert np.array_equal(seen_y, binary_targets)


def test_binary_targets_can_be_empty_when_pair_is_absent() -> None:
    y = np.array([30.0, 40.0], dtype=np.float64)

    pair_mask = one_vs_one_partial_fit_pair_mask(y, 10.0, 20.0)
    subset_indices = one_vs_one_partial_fit_subset_indices(pair_mask)
    binary_targets = one_vs_one_partial_fit_binary_targets(y, 10.0, 20.0)

    assert np.array_equal(pair_mask, np.array([False, False], dtype=np.bool_))
    assert subset_indices.shape == (0,)
    assert binary_targets.shape == (0,)


def test_contracts_reject_invalid_partial_fit_pair_inputs() -> None:
    with pytest.raises(ViolationError):
        one_vs_one_partial_fit_estimator_count(np.array([10.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        one_vs_one_partial_fit_pair_mask(np.array([1.0, 2.0], dtype=np.float64), 10.0, 10.0)

    with pytest.raises(ViolationError):
        one_vs_one_partial_fit_subset_indices(np.array([1, 0], dtype=np.int64))
