from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model._ransac import _dynamic_max_trials


def test_ransac_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import (
        ransac_consensus_is_better,
        ransac_default_residual_threshold,
        ransac_dynamic_max_trials,
        ransac_inlier_mask,
        ransac_loss_residuals,
    )

    assert callable(ransac_default_residual_threshold)
    assert callable(ransac_loss_residuals)
    assert callable(ransac_inlier_mask)
    assert callable(ransac_consensus_is_better)
    assert callable(ransac_dynamic_max_trials)


def test_ransac_default_residual_threshold_matches_source_formula() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import ransac_default_residual_threshold

    y = np.array([1.0, 1.5, 2.0, 100.0, 2.5], dtype=np.float64)
    expected = np.median(np.abs(y - np.median(y)))
    assert ransac_default_residual_threshold(y) == pytest.approx(expected)

    y_multi = np.array([[1.0, 2.0], [1.5, 2.5], [20.0, -5.0]], dtype=np.float64)
    expected_multi = np.median(np.abs(y_multi - np.median(y_multi)))
    assert ransac_default_residual_threshold(y_multi) == pytest.approx(expected_multi)


def test_ransac_loss_residuals_match_builtin_loss_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import ransac_loss_residuals

    y_true = np.array([1.0, -2.0, 4.0], dtype=np.float64)
    y_pred = np.array([0.0, -3.5, 5.0], dtype=np.float64)
    assert np.allclose(ransac_loss_residuals(y_true, y_pred), np.abs(y_true - y_pred))
    assert np.allclose(ransac_loss_residuals(y_true, y_pred, loss="squared_error"), (y_true - y_pred) ** 2)

    y_true_multi = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y_pred_multi = np.array([[0.5, 4.0], [4.5, 1.0]], dtype=np.float64)
    assert np.allclose(
        ransac_loss_residuals(y_true_multi, y_pred_multi),
        np.sum(np.abs(y_true_multi - y_pred_multi), axis=1),
    )
    assert np.allclose(
        ransac_loss_residuals(y_true_multi, y_pred_multi, loss="squared_error"),
        np.sum((y_true_multi - y_pred_multi) ** 2, axis=1),
    )


def test_ransac_inlier_mask_and_consensus_selection_match_source_checks() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import ransac_consensus_is_better, ransac_inlier_mask

    residuals = np.array([0.1, 0.5, 1.5, 0.0, 2.0], dtype=np.float64)
    assert np.array_equal(
        ransac_inlier_mask(residuals, residual_threshold=0.5),
        np.array([True, True, False, True, False], dtype=np.bool_),
    )

    assert ransac_consensus_is_better(2, 0.99, 3, 0.1) is False
    assert ransac_consensus_is_better(3, 0.09, 3, 0.1) is False
    assert ransac_consensus_is_better(3, 0.1, 3, 0.1) is True
    assert ransac_consensus_is_better(4, -10.0, 3, 0.1) is True


def test_ransac_dynamic_max_trials_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import ransac_dynamic_max_trials

    cases = [
        (20, 100, 2, 0.99),
        (90, 100, 5, 0.95),
        (0, 10, 2, 0.99),
        (5, 10, 3, 0.0),
        (10, 10, 2, 0.99),
    ]
    for n_inliers, n_samples, min_samples, probability in cases:
        result = ransac_dynamic_max_trials(n_inliers, n_samples, min_samples, probability)
        expected = _dynamic_max_trials(n_inliers, n_samples, min_samples, probability)
        if np.isinf(expected):
            assert np.isinf(result)
        else:
            assert result == pytest.approx(expected)


def test_contracts_reject_invalid_ransac_helper_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.ransac import (
        ransac_default_residual_threshold,
        ransac_dynamic_max_trials,
        ransac_inlier_mask,
        ransac_loss_residuals,
    )

    with pytest.raises(ViolationError):
        ransac_default_residual_threshold(np.array([1.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        ransac_loss_residuals(np.ones(3, dtype=np.float64), np.ones((3, 1), dtype=np.float64))

    with pytest.raises(ViolationError):
        ransac_inlier_mask(np.ones((2, 2), dtype=np.float64), residual_threshold=1.0)

    with pytest.raises(ViolationError):
        ransac_dynamic_max_trials(11, 10, 2, 0.99)
