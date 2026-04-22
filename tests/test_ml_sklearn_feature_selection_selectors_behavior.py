from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.feature_selection._from_model import _calculate_threshold


class CoefEstimator:
    def __init__(self, coef: np.ndarray) -> None:
        self.coef_ = coef


class Lasso:
    def __init__(self) -> None:
        self.coef_ = np.array([0.0, 1.0], dtype=np.float64)


class PlainEstimator:
    def __init__(self) -> None:
        self.coef_ = np.array([0.0, 1.0], dtype=np.float64)


def test_selector_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import (
        feature_importances_transform,
        rfe_elimination_step,
        select_from_model_support_mask,
        select_from_model_threshold,
        sequential_best_feature,
        sequential_candidate_masks,
    )

    assert callable(feature_importances_transform)
    assert callable(select_from_model_threshold)
    assert callable(select_from_model_support_mask)
    assert callable(rfe_elimination_step)
    assert callable(sequential_candidate_masks)
    assert callable(sequential_best_feature)


def test_feature_importances_transform_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import feature_importances_transform

    raw = np.array([-2.0, 3.0, -4.0], dtype=np.float64)
    assert np.array_equal(feature_importances_transform(raw), _get_feature_importances(CoefEstimator(raw), "auto"))
    assert np.array_equal(
        feature_importances_transform(raw, transform_func="norm"),
        _get_feature_importances(CoefEstimator(raw), "auto", transform_func="norm"),
    )
    assert np.array_equal(
        feature_importances_transform(raw, transform_func="square"),
        _get_feature_importances(CoefEstimator(raw), "auto", transform_func="square"),
    )

    matrix = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64)
    assert np.allclose(
        feature_importances_transform(matrix, transform_func="norm", norm_order=2),
        _get_feature_importances(CoefEstimator(matrix), "auto", transform_func="norm", norm_order=2),
    )
    assert np.allclose(
        feature_importances_transform(matrix, transform_func="square"),
        _get_feature_importances(CoefEstimator(matrix), "auto", transform_func="square"),
    )


def test_select_from_model_threshold_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import select_from_model_threshold

    scores = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    for threshold in ("mean", "median", "1.5*mean", "0.5*median", 0.25):
        assert select_from_model_threshold(scores, threshold=threshold) == _calculate_threshold(PlainEstimator(), scores, threshold)

    assert select_from_model_threshold(scores, threshold=None, l1_default=False) == _calculate_threshold(PlainEstimator(), scores, None)
    assert select_from_model_threshold(scores, threshold=None, l1_default=True) == _calculate_threshold(Lasso(), scores, None)


def test_select_from_model_support_mask_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import select_from_model_support_mask

    scores = np.array([0.4, 0.9, 0.7, 0.1, 0.7], dtype=np.float64)
    threshold = 0.5

    expected = np.zeros_like(scores, dtype=bool)
    expected[np.argsort(-scores, kind="mergesort")[:3]] = True
    expected[scores < threshold] = False
    assert np.array_equal(select_from_model_support_mask(scores, threshold=threshold, max_features=3), expected)

    no_cap = np.ones_like(scores, dtype=bool)
    no_cap[scores < threshold] = False
    assert np.array_equal(select_from_model_support_mask(scores, threshold=threshold), no_cap)


def test_rfe_elimination_step_matches_source_update() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import rfe_elimination_step

    support = np.array([True, True, False, True, True], dtype=np.bool_)
    ranking = np.array([1, 1, 2, 1, 1], dtype=np.int64)
    importances = np.array([0.5, 0.1, 0.4, 0.2], dtype=np.float64)

    result_support, result_ranking = rfe_elimination_step(
        support,
        ranking,
        importances,
        n_features_to_select=2,
        step=2,
    )

    expected_support = support.copy()
    expected_ranking = ranking.copy()
    features = np.arange(support.shape[0])[support]
    ranks = np.ravel(np.argsort(importances))
    threshold = min(2, np.sum(support) - 2)
    expected_support[features[ranks][:threshold]] = False
    expected_ranking[np.logical_not(expected_support)] += 1

    assert np.array_equal(result_support, expected_support)
    assert np.array_equal(result_ranking, expected_ranking)


def test_sequential_candidate_masks_forward_backward_and_best_feature() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import sequential_best_feature, sequential_candidate_masks

    current = np.array([True, False, True, False], dtype=np.bool_)
    candidate_indices, forward_masks = sequential_candidate_masks(current, direction="forward")
    assert np.array_equal(candidate_indices, np.array([1, 3], dtype=np.int64))
    assert np.array_equal(
        forward_masks,
        np.array([[True, True, True, False], [True, False, True, True]], dtype=np.bool_),
    )

    backward_indices, backward_masks = sequential_candidate_masks(current, direction="backward")
    assert np.array_equal(backward_indices, candidate_indices)
    assert np.array_equal(
        backward_masks,
        np.array([[False, False, False, True], [False, True, False, False]], dtype=np.bool_),
    )

    assert sequential_best_feature(candidate_indices, np.array([0.2, 0.8], dtype=np.float64)) == 3
    assert sequential_best_feature(candidate_indices, np.array([0.8, 0.8], dtype=np.float64)) == 1


def test_contracts_reject_invalid_selector_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.selectors import (
        feature_importances_transform,
        select_from_model_support_mask,
        sequential_candidate_masks,
    )

    with pytest.raises(ViolationError):
        feature_importances_transform(np.ones((2, 2), dtype=np.float64))

    with pytest.raises(ViolationError):
        select_from_model_support_mask(np.array([0.1, 0.2], dtype=np.float64), threshold=0.0, max_features=3)

    with pytest.raises(ViolationError):
        sequential_candidate_masks(np.array([True, True], dtype=np.bool_))
