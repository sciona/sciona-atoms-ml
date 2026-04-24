from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.impute._iterative import _assign_where


class _DeterministicEstimator:
    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = np.asarray(predictions, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[0] == self._predictions.shape[0]
        return self._predictions.copy()


class _PosteriorEstimator:
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray) -> None:
        self._mus = np.asarray(mus, dtype=np.float64)
        self._sigmas = np.asarray(sigmas, dtype=np.float64)

    def predict(self, X: np.ndarray, return_std: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        assert X.shape[0] == self._mus.shape[0]
        if return_std:
            return self._mus.copy(), self._sigmas.copy()
        return self._mus.copy()


def test_iterative_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_postprocessing import (
        iterative_assign_feature_values,
        iterative_clipped_imputed_values,
        iterative_posterior_imputed_values,
        iterative_restore_observed_values,
    )

    assert callable(iterative_posterior_imputed_values)
    assert callable(iterative_clipped_imputed_values)
    assert callable(iterative_assign_feature_values)
    assert callable(iterative_restore_observed_values)


def test_iterative_clipped_imputed_values_and_assignment_match_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_postprocessing import (
        iterative_assign_feature_values,
        iterative_clipped_imputed_values,
    )

    X_filled = np.array(
        [
            [1.0, 0.2],
            [2.0, 0.5],
            [3.0, 0.7],
            [4.0, 0.9],
        ],
        dtype=np.float64,
    )
    mask_missing_values = np.array(
        [
            [False, True],
            [False, False],
            [False, True],
            [False, True],
        ],
        dtype=np.bool_,
    )
    feat_idx = 1
    predictions = np.array([-1.0, 0.6, 3.0], dtype=np.float64)

    imputer = IterativeImputer(sample_posterior=False)
    imputer._min_value = np.array([-np.inf, 0.0], dtype=np.float64)
    imputer._max_value = np.array([np.inf, 1.0], dtype=np.float64)
    estimator = _DeterministicEstimator(predictions)
    expected, _ = imputer._impute_one_feature(
        X_filled.copy(),
        mask_missing_values,
        feat_idx,
        np.array([0], dtype=np.int64),
        estimator=estimator,
        fit_mode=False,
        params={},
    )

    clipped = iterative_clipped_imputed_values(predictions, min_value=0.0, max_value=1.0)
    actual = iterative_assign_feature_values(
        X_filled,
        clipped,
        mask_missing_values[:, feat_idx],
        feat_idx=feat_idx,
    )

    assert np.allclose(clipped, np.array([0.0, 0.6, 1.0], dtype=np.float64))
    assert np.allclose(actual, expected)


def test_iterative_posterior_imputed_values_and_assignment_match_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_postprocessing import (
        iterative_assign_feature_values,
        iterative_posterior_imputed_values,
    )

    X_filled = np.array(
        [
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
        ],
        dtype=np.float64,
    )
    mask_missing_values = np.array(
        [
            [False, True],
            [False, True],
            [False, False],
            [False, True],
            [False, True],
        ],
        dtype=np.bool_,
    )
    feat_idx = 1
    mus = np.array([-0.5, 0.3, 1.4, 0.6], dtype=np.float64)
    sigmas = np.array([0.2, 0.1, 0.3, -1.0], dtype=np.float64)

    imputer = IterativeImputer(sample_posterior=True, random_state=13)
    imputer._min_value = np.array([-np.inf, 0.0], dtype=np.float64)
    imputer._max_value = np.array([np.inf, 1.0], dtype=np.float64)
    imputer.random_state_ = np.random.RandomState(13)
    estimator = _PosteriorEstimator(mus, sigmas)
    expected, _ = imputer._impute_one_feature(
        X_filled.copy(),
        mask_missing_values,
        feat_idx,
        np.array([0], dtype=np.int64),
        estimator=estimator,
        fit_mode=False,
        params={},
    )

    sampled = iterative_posterior_imputed_values(
        mus,
        sigmas,
        min_value=0.0,
        max_value=1.0,
        random_state=13,
    )
    actual = iterative_assign_feature_values(
        X_filled,
        sampled,
        mask_missing_values[:, feat_idx],
        feat_idx=feat_idx,
    )

    assert sampled[0] == pytest.approx(0.0)
    assert sampled[2] == pytest.approx(1.0)
    assert sampled[3] == pytest.approx(0.6)
    assert np.allclose(actual, expected)


def test_iterative_restore_observed_values_matches_private_assign_where() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_postprocessing import iterative_restore_observed_values

    X_target = np.array([[0.1, 9.0], [0.2, 8.0], [0.3, 7.0]], dtype=np.float64)
    X_source = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    observed_mask = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)

    expected = X_target.copy()
    _assign_where(expected, X_source, cond=observed_mask)

    actual = iterative_restore_observed_values(X_target, X_source, observed_mask)
    assert np.allclose(actual, expected)


def test_contracts_reject_invalid_iterative_postprocessing_inputs() -> None:
    from sciona.atoms.ml.sklearn.impute.iterative_postprocessing import (
        iterative_assign_feature_values,
        iterative_clipped_imputed_values,
        iterative_posterior_imputed_values,
        iterative_restore_observed_values,
    )

    with pytest.raises(ViolationError):
        iterative_posterior_imputed_values(
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
            min_value=0.0,
            max_value=1.0,
        )

    with pytest.raises(ViolationError):
        iterative_clipped_imputed_values(np.array([0.1, np.nan], dtype=np.float64), min_value=0.0, max_value=1.0)

    with pytest.raises(ViolationError):
        iterative_assign_feature_values(
            np.ones((2, 2), dtype=np.float64),
            np.array([0.5], dtype=np.float64),
            np.array([True, True], dtype=np.bool_),
            feat_idx=1,
        )

    with pytest.raises(ViolationError):
        iterative_restore_observed_values(
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            np.ones((2, 2), dtype=np.int64),
        )
