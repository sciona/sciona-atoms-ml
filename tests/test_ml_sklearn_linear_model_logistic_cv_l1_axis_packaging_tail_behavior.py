from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_cv_l1_axis_packaging_tail_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import (
        logistic_cv_coefs_paths_dict_l1_axis,
        logistic_cv_coefs_paths_l1_axis,
        logistic_cv_l1_axis_enabled,
        logistic_cv_n_iter_l1_axis,
        logistic_cv_scores_dict_l1_axis,
        logistic_cv_scores_l1_axis,
    )

    assert callable(logistic_cv_l1_axis_enabled)
    assert callable(logistic_cv_coefs_paths_l1_axis)
    assert callable(logistic_cv_coefs_paths_dict_l1_axis)
    assert callable(logistic_cv_scores_l1_axis)
    assert callable(logistic_cv_scores_dict_l1_axis)
    assert callable(logistic_cv_n_iter_l1_axis)


def test_logistic_cv_l1_axis_enabled_uses_public_parameter_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import logistic_cv_l1_axis_enabled

    assert logistic_cv_l1_axis_enabled(None) is False
    assert logistic_cv_l1_axis_enabled([0.5]) is True
    assert logistic_cv_l1_axis_enabled([None]) is True


def test_logistic_cv_coefs_and_scores_l1_axis_match_source_transpose() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import (
        logistic_cv_coefs_paths_l1_axis,
        logistic_cv_scores_l1_axis,
    )

    n_folds = 2
    n_Cs = 2
    n_l1_ratios = 3
    width = 4
    coefs = np.arange(n_folds * n_Cs * n_l1_ratios * width, dtype=np.float32).reshape(n_folds, n_Cs * n_l1_ratios, width)
    scores = np.linspace(0.1, 0.9, n_folds * n_Cs * n_l1_ratios, dtype=np.float64).reshape(n_folds, n_Cs * n_l1_ratios)

    coefs_result = logistic_cv_coefs_paths_l1_axis(coefs, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)
    scores_result = logistic_cv_scores_l1_axis(scores, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)

    expected_coefs = coefs.reshape((n_folds, n_l1_ratios, n_Cs, width)).transpose((0, 2, 1, 3))
    expected_scores = scores.reshape((n_folds, n_l1_ratios, n_Cs)).transpose((0, 2, 1))
    np.testing.assert_array_equal(coefs_result, expected_coefs)
    np.testing.assert_array_equal(scores_result, expected_scores)
    assert coefs_result.dtype == np.float32
    assert scores_result.dtype == np.float64


def test_logistic_cv_l1_axis_dict_helpers_preserve_class_order_and_values() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import (
        logistic_cv_coefs_paths_dict_l1_axis,
        logistic_cv_scores_dict_l1_axis,
    )

    n_folds = 2
    n_Cs = 2
    n_l1_ratios = 2
    coefs_by_class = {
        "a": np.arange(24, dtype=np.float64).reshape(n_folds, n_Cs * n_l1_ratios, 3),
        "b": np.arange(24, 48, dtype=np.float64).reshape(n_folds, n_Cs * n_l1_ratios, 3),
    }
    scores_by_class = {
        "a": np.arange(8, dtype=np.float64).reshape(n_folds, n_Cs * n_l1_ratios),
        "b": np.arange(8, 16, dtype=np.float64).reshape(n_folds, n_Cs * n_l1_ratios),
    }

    coefs_result = logistic_cv_coefs_paths_dict_l1_axis(coefs_by_class, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)
    scores_result = logistic_cv_scores_dict_l1_axis(scores_by_class, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)

    assert list(coefs_result) == ["a", "b"]
    assert list(scores_result) == ["a", "b"]
    np.testing.assert_array_equal(coefs_result["a"], coefs_by_class["a"].reshape((n_folds, n_l1_ratios, n_Cs, 3)).transpose((0, 2, 1, 3)))
    np.testing.assert_array_equal(scores_result["b"], scores_by_class["b"].reshape((n_folds, n_l1_ratios, n_Cs)).transpose((0, 2, 1)))


def test_logistic_cv_n_iter_l1_axis_preserves_inferred_class_axis_and_dtype() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import logistic_cv_n_iter_l1_axis

    n_folds = 2
    n_Cs = 2
    n_l1_ratios = 3
    n_iter = np.arange(2 * n_folds * n_Cs * n_l1_ratios, dtype=np.int32).reshape(2, n_folds, n_Cs * n_l1_ratios)

    result = logistic_cv_n_iter_l1_axis(n_iter, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)

    expected = n_iter.reshape((-1, n_folds, n_l1_ratios, n_Cs)).transpose((0, 1, 3, 2))
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2, n_folds, n_Cs, n_l1_ratios)
    assert result.dtype == np.int32


def test_logistic_cv_l1_axis_singleton_axes_are_retained() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import (
        logistic_cv_coefs_paths_l1_axis,
        logistic_cv_n_iter_l1_axis,
        logistic_cv_scores_l1_axis,
    )

    coefs = np.arange(6, dtype=np.float64).reshape(2, 1, 3)
    scores = np.arange(2, dtype=np.float64).reshape(2, 1)
    n_iter = np.arange(2, dtype=np.int32).reshape(1, 2, 1)

    assert logistic_cv_coefs_paths_l1_axis(coefs, n_folds=2, n_Cs=1, n_l1_ratios=1).shape == (2, 1, 1, 3)
    assert logistic_cv_scores_l1_axis(scores, n_folds=2, n_Cs=1, n_l1_ratios=1).shape == (2, 1, 1)
    assert logistic_cv_n_iter_l1_axis(n_iter, n_folds=2, n_Cs=1, n_l1_ratios=1).shape == (1, 2, 1, 1)


def test_logistic_cv_l1_axis_packaging_tail_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_l1_axis_packaging_tail import (
        logistic_cv_coefs_paths_dict_l1_axis,
        logistic_cv_coefs_paths_l1_axis,
        logistic_cv_n_iter_l1_axis,
        logistic_cv_scores_dict_l1_axis,
        logistic_cv_scores_l1_axis,
    )

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_l1_axis(np.ones((2, 3), dtype=np.float64), n_folds=0, n_Cs=1, n_l1_ratios=1)

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_l1_axis(np.ones((5,), dtype=np.float64), n_folds=2, n_Cs=2, n_l1_ratios=2)

    with pytest.raises(ViolationError):
        logistic_cv_scores_l1_axis(np.array([np.nan], dtype=np.float64), n_folds=1, n_Cs=1, n_l1_ratios=1)

    with pytest.raises(ViolationError):
        logistic_cv_n_iter_l1_axis(np.array([-1], dtype=np.int32), n_folds=1, n_Cs=1, n_l1_ratios=1)

    with pytest.raises(ViolationError):
        logistic_cv_coefs_paths_dict_l1_axis({}, n_folds=1, n_Cs=1, n_l1_ratios=1)

    with pytest.raises(ViolationError):
        logistic_cv_scores_dict_l1_axis({"a": np.ones((3,), dtype=np.float64)}, n_folds=2, n_Cs=2, n_l1_ratios=1)
