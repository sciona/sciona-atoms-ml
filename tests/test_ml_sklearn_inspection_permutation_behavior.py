from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.inspection._permutation_importance import _create_importances_bunch


def test_permutation_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation import (
        permutation_importance_mean,
        permutation_importance_std,
        permutation_importance_summary,
        permutation_importance_values,
    )

    assert callable(permutation_importance_values)
    assert callable(permutation_importance_mean)
    assert callable(permutation_importance_std)
    assert callable(permutation_importance_summary)


def test_permutation_importance_values_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation import permutation_importance_values

    baseline_score = 0.82
    permuted_scores = np.array(
        [
            [0.52, 0.61, 0.58],
            [0.79, 0.80, 0.81],
            [0.20, 0.25, 0.22],
        ],
        dtype=np.float64,
    )

    expected = _create_importances_bunch(baseline_score, permuted_scores).importances
    assert np.allclose(permutation_importance_values(baseline_score, permuted_scores), expected)


def test_permutation_importance_mean_and_std_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation import (
        permutation_importance_mean,
        permutation_importance_std,
        permutation_importance_values,
    )

    baseline_score = -1.25
    permuted_scores = np.array([[0.2, -0.1, -0.3, 0.4], [-1.0, -1.4, -1.2, -1.3]], dtype=np.float64)
    expected = _create_importances_bunch(baseline_score, permuted_scores)
    importances = permutation_importance_values(baseline_score, permuted_scores)

    assert np.allclose(permutation_importance_mean(importances), expected.importances_mean)
    assert np.allclose(permutation_importance_std(importances), expected.importances_std)


def test_permutation_importance_summary_matches_sklearn_bunch_fields() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation import permutation_importance_summary

    baseline_score = 1.5
    permuted_scores = np.array(
        [
            [1.0, 1.1, 0.8],
            [1.45, 1.55, 1.40],
            [0.2, 0.3, 0.4],
            [1.5, 1.5, 1.5],
        ],
        dtype=np.float64,
    )

    expected = _create_importances_bunch(baseline_score, permuted_scores)
    mean, spread, importances = permutation_importance_summary(baseline_score, permuted_scores)

    assert np.allclose(mean, expected.importances_mean)
    assert np.allclose(spread, expected.importances_std)
    assert np.allclose(importances, expected.importances)


def test_contracts_reject_invalid_permutation_importance_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation import (
        permutation_importance_mean,
        permutation_importance_summary,
        permutation_importance_values,
    )

    with pytest.raises(ViolationError):
        permutation_importance_values(float("nan"), np.ones((2, 3), dtype=np.float64))

    with pytest.raises(ViolationError):
        permutation_importance_values(0.5, np.ones(3, dtype=np.float64))

    with pytest.raises(ViolationError):
        permutation_importance_mean(np.array([[0.1, np.nan]], dtype=np.float64))

    with pytest.raises(ViolationError):
        permutation_importance_summary(0.5, np.ones((0, 3), dtype=np.float64))
