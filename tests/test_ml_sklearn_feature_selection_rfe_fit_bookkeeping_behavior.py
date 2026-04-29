from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_rfe_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import (
        rfe_active_feature_indices,
        rfe_resolve_n_features_to_select,
        rfe_resolve_step,
        rfe_step_history_append,
        rfe_warn_too_many_features_to_select,
    )

    assert callable(rfe_resolve_n_features_to_select)
    assert callable(rfe_warn_too_many_features_to_select)
    assert callable(rfe_resolve_step)
    assert callable(rfe_active_feature_indices)
    assert callable(rfe_step_history_append)


def test_rfe_resolve_n_features_to_select_matches_sklearn_fit_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import rfe_resolve_n_features_to_select

    assert rfe_resolve_n_features_to_select(7, n_features_to_select=None) == 3
    assert rfe_resolve_n_features_to_select(7, n_features_to_select=4) == 4
    assert rfe_resolve_n_features_to_select(7, n_features_to_select=0.5) == 3
    assert rfe_resolve_n_features_to_select(7, n_features_to_select=1.0) == 7


def test_rfe_warn_too_many_features_to_select_matches_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import rfe_warn_too_many_features_to_select

    assert rfe_warn_too_many_features_to_select(5, resolved_n_features_to_select=6) is True
    assert rfe_warn_too_many_features_to_select(5, resolved_n_features_to_select=5) is False


def test_rfe_resolve_step_matches_sklearn_fit_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import rfe_resolve_step

    assert rfe_resolve_step(10, step=3) == 3
    assert rfe_resolve_step(10, step=0.2) == 2
    assert rfe_resolve_step(10, step=0.01) == 1
    assert rfe_resolve_step(7, step=0.5) == 3


def test_rfe_active_feature_indices_match_support_mask_indexing() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import rfe_active_feature_indices

    support = np.array([True, False, True, False, True], dtype=np.bool_)
    observed = rfe_active_feature_indices(support)

    assert np.array_equal(observed, np.array([0, 2, 4], dtype=np.int64))


def test_rfe_step_history_append_matches_sklearn_list_append_shape_and_values() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import rfe_step_history_append

    step_n_features = np.array([5, 3], dtype=np.int64)
    step_scores = np.array([0.8, 0.85], dtype=np.float64)

    observed_counts, observed_scores = rfe_step_history_append(
        step_n_features,
        step_scores,
        n_features=2,
        score=0.9,
    )

    assert np.array_equal(observed_counts, np.array([5, 3, 2], dtype=np.int64))
    assert np.allclose(observed_scores, np.array([0.8, 0.85, 0.9], dtype=np.float64))


def test_rfe_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping import (
        rfe_active_feature_indices,
        rfe_resolve_n_features_to_select,
        rfe_resolve_step,
        rfe_step_history_append,
    )

    with pytest.raises(ViolationError):
        rfe_resolve_n_features_to_select(5, n_features_to_select=0.0)

    with pytest.raises(ViolationError):
        rfe_resolve_step(5, step=0.0)

    with pytest.raises(ViolationError):
        rfe_active_feature_indices(np.array([False, False], dtype=np.bool_))

    with pytest.raises(ViolationError):
        rfe_step_history_append(
            np.array([5, 3], dtype=np.int64),
            np.array([0.8], dtype=np.float64),
            n_features=2,
            score=0.9,
        )
