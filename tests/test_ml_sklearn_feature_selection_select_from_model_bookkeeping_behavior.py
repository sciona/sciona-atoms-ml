from __future__ import annotations

import numpy as np


def test_select_from_model_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.select_from_model_bookkeeping import (
        select_from_model_candidate_indices,
        select_from_model_checked_max_features,
        select_from_model_prefit_callable_max_features_ready,
        select_from_model_prefit_estimator_valid,
    )

    assert callable(select_from_model_checked_max_features)
    assert callable(select_from_model_prefit_estimator_valid)
    assert callable(select_from_model_prefit_callable_max_features_ready)
    assert callable(select_from_model_candidate_indices)


def test_checked_max_features_matches_sklearn_scalar_constraints() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.select_from_model_bookkeeping import select_from_model_checked_max_features

    assert select_from_model_checked_max_features(0, n_features=5) == 0
    assert select_from_model_checked_max_features(3, n_features=5) == 3
    assert select_from_model_checked_max_features(5, n_features=5) == 5


def test_prefit_estimator_valid_matches_prefit_guard() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.select_from_model_bookkeeping import select_from_model_prefit_estimator_valid

    assert select_from_model_prefit_estimator_valid(prefit=False, estimator_is_fitted=False) is True
    assert select_from_model_prefit_estimator_valid(prefit=True, estimator_is_fitted=True) is True
    assert select_from_model_prefit_estimator_valid(prefit=True, estimator_is_fitted=False) is False


def test_prefit_callable_max_features_ready_matches_transform_guard() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.select_from_model_bookkeeping import (
        select_from_model_prefit_callable_max_features_ready,
    )

    assert select_from_model_prefit_callable_max_features_ready(
        prefit=True,
        max_features_is_callable=True,
        has_fitted_max_features=False,
    ) is False
    assert select_from_model_prefit_callable_max_features_ready(
        prefit=True,
        max_features_is_callable=True,
        has_fitted_max_features=True,
    ) is True
    assert select_from_model_prefit_callable_max_features_ready(
        prefit=False,
        max_features_is_callable=True,
        has_fitted_max_features=False,
    ) is True


def test_candidate_indices_match_sklearn_stable_descending_top_k() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.select_from_model_bookkeeping import select_from_model_candidate_indices

    scores = np.array([0.4, 0.9, 0.9, 0.1, 0.6], dtype=np.float64)
    expected = np.argsort(-scores, kind="mergesort")[:3]

    observed = select_from_model_candidate_indices(scores, max_features=3)

    assert np.array_equal(observed, expected)
    assert np.array_equal(observed, np.array([1, 2, 4], dtype=np.int64))
