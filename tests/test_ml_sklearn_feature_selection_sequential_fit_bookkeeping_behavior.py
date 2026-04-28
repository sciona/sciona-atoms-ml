from __future__ import annotations

import numpy as np


def test_sequential_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import (
        sequential_auto_select_enabled,
        sequential_direction_tol_valid,
        sequential_finalize_support,
        sequential_iteration_count,
        sequential_resolve_n_features_to_select,
        sequential_tolerance_break,
    )

    assert callable(sequential_resolve_n_features_to_select)
    assert callable(sequential_direction_tol_valid)
    assert callable(sequential_auto_select_enabled)
    assert callable(sequential_iteration_count)
    assert callable(sequential_tolerance_break)
    assert callable(sequential_finalize_support)


def test_resolve_n_features_to_select_matches_sklearn_fit_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_resolve_n_features_to_select

    assert sequential_resolve_n_features_to_select(10, n_features_to_select="auto", tol=None) == 5
    assert sequential_resolve_n_features_to_select(10, n_features_to_select="auto", tol=0.01) == 9
    assert sequential_resolve_n_features_to_select(10, n_features_to_select=3, tol=None) == 3
    assert sequential_resolve_n_features_to_select(10, n_features_to_select=0.4, tol=None) == 4


def test_direction_tol_valid_matches_sklearn_guard() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_direction_tol_valid

    assert sequential_direction_tol_valid(direction="forward", tol=0.1) is True
    assert sequential_direction_tol_valid(direction="forward", tol=-0.1) is False
    assert sequential_direction_tol_valid(direction="backward", tol=-0.1) is True
    assert sequential_direction_tol_valid(direction="backward", tol=None) is True


def test_auto_select_enabled_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_auto_select_enabled

    assert sequential_auto_select_enabled(n_features_to_select="auto", tol=0.01) is True
    assert sequential_auto_select_enabled(n_features_to_select="auto", tol=None) is False
    assert sequential_auto_select_enabled(n_features_to_select=3, tol=0.01) is False


def test_iteration_count_matches_sklearn_fit_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_iteration_count

    assert sequential_iteration_count(10, 5, n_features_to_select="auto", direction="forward") == 5
    assert sequential_iteration_count(10, 4, n_features_to_select=0.4, direction="forward") == 4
    assert sequential_iteration_count(10, 4, n_features_to_select=0.4, direction="backward") == 6


def test_tolerance_break_matches_sklearn_auto_select_condition() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_tolerance_break

    assert sequential_tolerance_break(-np.inf, 0.4, tol=0.01) is False
    assert sequential_tolerance_break(0.4, 0.405, tol=0.01) is True
    assert sequential_tolerance_break(0.4, 0.42, tol=0.01) is False


def test_finalize_support_matches_forward_and_backward_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.sequential_fit_bookkeeping import sequential_finalize_support

    current_mask = np.array([True, False, True, False], dtype=np.bool_)

    assert np.array_equal(sequential_finalize_support(current_mask, direction="forward"), current_mask)
    assert np.array_equal(
        sequential_finalize_support(current_mask, direction="backward"),
        np.array([False, True, False, True], dtype=np.bool_),
    )
