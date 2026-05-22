from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_cv_best_refit_selection_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_best_C_l1_selection,
        logistic_cv_best_flat_index,
        logistic_cv_loop_path_views,
        logistic_cv_multinomial_final_components,
        logistic_cv_nonrefit_average_C,
        logistic_cv_nonrefit_average_l1_ratio,
        logistic_cv_nonrefit_average_w,
        logistic_cv_nonrefit_best_indices,
        logistic_cv_ovr_final_row,
        logistic_cv_refit_coef_init,
    )

    assert callable(logistic_cv_loop_path_views)
    assert callable(logistic_cv_best_flat_index)
    assert callable(logistic_cv_best_C_l1_selection)
    assert callable(logistic_cv_refit_coef_init)
    assert callable(logistic_cv_nonrefit_best_indices)
    assert callable(logistic_cv_nonrefit_average_w)
    assert callable(logistic_cv_nonrefit_average_C)
    assert callable(logistic_cv_nonrefit_average_l1_ratio)
    assert callable(logistic_cv_multinomial_final_components)
    assert callable(logistic_cv_ovr_final_row)


def test_logistic_cv_loop_path_views_match_ovr_and_multinomial_source_views() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import logistic_cv_loop_path_views

    ovr_scores = {"a": np.array([[0.1, 0.2]], dtype=np.float64)}
    ovr_coefs = {"a": np.array([[[1.0, 2.0]]], dtype=np.float32)}
    multinomial_scores = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]], dtype=np.float64)
    multinomial_coefs = np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2)

    scores_view, coefs_view = logistic_cv_loop_path_views("ovr", "a", ovr_scores, ovr_coefs, multinomial_scores, multinomial_coefs)
    assert scores_view is ovr_scores["a"]
    assert coefs_view is ovr_coefs["a"]

    scores_view, coefs_view = logistic_cv_loop_path_views("multinomial", "a", ovr_scores, ovr_coefs, multinomial_scores, multinomial_coefs)
    np.testing.assert_array_equal(scores_view, multinomial_scores[0])
    assert coefs_view is multinomial_coefs


def test_logistic_cv_refit_best_selection_and_coef_init_match_source() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_best_C_l1_selection,
        logistic_cv_best_flat_index,
        logistic_cv_refit_coef_init,
    )

    Cs = np.array([0.1, 1.0], dtype=np.float64)
    l1_ratios = np.array([0.25, 0.75], dtype=np.float64)
    scores = np.array([[1.0, 1.0, 2.0, 1.5], [0.5, 0.6, 0.2, 2.2]], dtype=np.float64)
    coefs_ovr = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    coefs_multi = np.arange(3 * 2 * 4 * 5, dtype=np.float64).reshape(3, 2, 4, 5)

    best_index = logistic_cv_best_flat_index(scores)
    C_value, l1_value = logistic_cv_best_C_l1_selection(best_index, Cs, l1_ratios)
    coef_init_ovr = logistic_cv_refit_coef_init(coefs_ovr, multi_class="ovr", best_index=best_index)
    coef_init_multi = logistic_cv_refit_coef_init(coefs_multi, multi_class="multinomial", best_index=best_index)

    assert best_index == int(scores.sum(axis=0).argmax())
    assert C_value == Cs[best_index % len(Cs)]
    assert l1_value == l1_ratios[best_index // len(Cs)]
    np.testing.assert_array_equal(coef_init_ovr, np.mean(coefs_ovr[:, best_index, :], axis=0))
    np.testing.assert_array_equal(coef_init_multi, np.mean(coefs_multi[:, :, best_index, :], axis=1))
    assert coef_init_ovr.dtype == np.float32
    assert coef_init_multi.dtype == np.float64


def test_logistic_cv_best_flat_index_preserves_numpy_first_tie_behavior() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import logistic_cv_best_flat_index

    scores = np.array([[1.0, 2.0, 2.0], [2.0, 1.0, 1.0]], dtype=np.float64)

    assert logistic_cv_best_flat_index(scores) == 0


def test_logistic_cv_best_selection_allows_non_elasticnet_none_l1_ratio() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_best_C_l1_selection,
        logistic_cv_nonrefit_average_l1_ratio,
    )

    C_value, l1_value = logistic_cv_best_C_l1_selection(1, np.array([0.1, 1.0], dtype=np.float64), [None])

    assert C_value == 1.0
    assert l1_value is None
    assert logistic_cv_nonrefit_average_l1_ratio(np.array([0, 1], dtype=np.int64), np.array([0.1, 1.0]), [None], "l2") is None


def test_logistic_cv_nonrefit_selection_and_averages_match_source() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_nonrefit_average_C,
        logistic_cv_nonrefit_average_l1_ratio,
        logistic_cv_nonrefit_average_w,
        logistic_cv_nonrefit_best_indices,
    )

    Cs = np.array([0.1, 1.0], dtype=np.float64)
    l1_ratios = np.array([0.25, 0.75], dtype=np.float64)
    scores = np.array([[0.0, 4.0, 1.0, 2.0], [3.0, 0.0, 4.0, 1.0], [1.0, 2.0, 3.0, 6.0]], dtype=np.float64)
    coefs_ovr = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    coefs_multi = np.arange(2 * 3 * 4 * 2, dtype=np.float64).reshape(2, 3, 4, 2)

    best_indices = logistic_cv_nonrefit_best_indices(scores)
    w_ovr = logistic_cv_nonrefit_average_w(coefs_ovr, best_indices, multi_class="ovr")
    w_multi = logistic_cv_nonrefit_average_w(coefs_multi, best_indices, multi_class="multinomial")
    C_value = logistic_cv_nonrefit_average_C(best_indices, Cs)
    l1_value = logistic_cv_nonrefit_average_l1_ratio(best_indices, Cs, l1_ratios, "elasticnet")

    expected_best_indices = np.argmax(scores, axis=1)
    np.testing.assert_array_equal(best_indices, expected_best_indices)
    np.testing.assert_array_equal(w_ovr, np.mean([coefs_ovr[i, expected_best_indices[i], :] for i in range(3)], axis=0))
    np.testing.assert_array_equal(w_multi, np.mean([coefs_multi[:, i, expected_best_indices[i], :] for i in range(3)], axis=0))
    assert C_value == np.mean(Cs[expected_best_indices % len(Cs)])
    assert l1_value == np.mean(l1_ratios[expected_best_indices // len(Cs)])
    assert w_ovr.dtype == np.float32
    assert w_multi.dtype == np.float64


def test_logistic_cv_final_component_packaging_matches_source() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_multinomial_final_components,
        logistic_cv_ovr_final_row,
    )

    C_values = [0.1]
    l1_values = [None]
    w_multi = np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 8.0]], dtype=np.float64)
    w_ovr = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    C_tiled, l1_tiled, coef, intercept = logistic_cv_multinomial_final_components(
        C_values,
        l1_values,
        w_multi,
        n_classes=2,
        n_features=2,
        fit_intercept=True,
    )
    coef_row, intercept_value = logistic_cv_ovr_final_row(w_ovr, n_features=2, fit_intercept=True)
    coef_row_no_intercept, intercept_none = logistic_cv_ovr_final_row(w_ovr[:2], n_features=2, fit_intercept=False)

    np.testing.assert_array_equal(C_tiled, np.tile(C_values, 2))
    np.testing.assert_array_equal(l1_tiled, np.tile(l1_values, 2))
    np.testing.assert_array_equal(coef, w_multi[:, :2])
    np.testing.assert_array_equal(intercept, w_multi[:, -1])
    np.testing.assert_array_equal(coef_row, w_ovr[:2])
    assert intercept_value == w_ovr[-1]
    np.testing.assert_array_equal(coef_row_no_intercept, w_ovr[:2])
    assert intercept_none is None


def test_logistic_cv_best_refit_selection_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_best_refit_selection_shell import (
        logistic_cv_best_C_l1_selection,
        logistic_cv_best_flat_index,
        logistic_cv_loop_path_views,
        logistic_cv_multinomial_final_components,
        logistic_cv_nonrefit_average_l1_ratio,
        logistic_cv_nonrefit_average_w,
        logistic_cv_nonrefit_best_indices,
        logistic_cv_ovr_final_row,
        logistic_cv_refit_coef_init,
    )

    with pytest.raises(ViolationError):
        logistic_cv_loop_path_views("bad", "a", {}, {}, np.ones((1, 1)), np.ones((1, 1, 1)))

    with pytest.raises(ViolationError):
        logistic_cv_best_flat_index(np.array([np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        logistic_cv_best_C_l1_selection(4, np.array([0.1, 1.0]), np.array([0.5, 0.9]))

    with pytest.raises(ViolationError):
        logistic_cv_refit_coef_init(np.ones((2, 2), dtype=np.float64), multi_class="ovr", best_index=0)

    with pytest.raises(ViolationError):
        logistic_cv_nonrefit_best_indices(np.array([0.1, 0.2], dtype=np.float64))

    with pytest.raises(ViolationError):
        logistic_cv_nonrefit_average_w(np.ones((2, 2, 3), dtype=np.float64), np.array([0], dtype=np.int64), multi_class="ovr")

    with pytest.raises(ViolationError):
        logistic_cv_nonrefit_average_l1_ratio(np.array([4], dtype=np.int64), np.array([0.1, 1.0]), np.array([0.5, 0.9]), "elasticnet")

    with pytest.raises(ViolationError):
        logistic_cv_multinomial_final_components([0.1], [0.5], np.ones((2, 2), dtype=np.float64), n_classes=2, n_features=2, fit_intercept=True)

    with pytest.raises(ViolationError):
        logistic_cv_ovr_final_row(np.ones((2, 2), dtype=np.float64), n_features=2, fit_intercept=False)
