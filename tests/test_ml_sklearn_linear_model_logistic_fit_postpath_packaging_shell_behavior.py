from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_fit_postpath_packaging_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
        logistic_fit_n_iter_from_path_results,
        logistic_fit_path_results,
    )

    assert callable(logistic_fit_path_results)
    assert callable(logistic_fit_n_iter_from_path_results)
    assert callable(logistic_fit_coef_with_intercept)
    assert callable(logistic_fit_final_coef)
    assert callable(logistic_fit_final_intercept)


def test_logistic_fit_path_results_matches_source_unzip() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import logistic_fit_path_results

    coef_a = np.array([[0.2, -0.4, 1.5]], dtype=np.float32)
    coef_b = np.array([[-0.1, 0.7, -0.3]], dtype=np.float32)
    path_results = (
        (coef_a, np.array([1.0]), np.array([7], dtype=np.int64)),
        (coef_b, np.array([1.0]), np.array([11], dtype=np.int64)),
    )

    fold_coefs, Cs, n_iter = logistic_fit_path_results(path_results)

    assert fold_coefs == (coef_a, coef_b)
    assert len(Cs) == 2
    assert n_iter == (path_results[0][2], path_results[1][2])


def test_logistic_fit_n_iter_from_path_results_matches_source_slice() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_n_iter_from_path_results,
    )

    n_iter = (np.array([7], dtype=np.int64), np.array([11], dtype=np.int64))

    result = logistic_fit_n_iter_from_path_results(n_iter)

    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, np.asarray(n_iter, dtype=np.int32)[:, 0])


def test_logistic_fit_coef_packaging_matches_ovr_multiclass_layout() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
    )

    fold_coefs = (
        np.array([[0.2, -0.4, 1.5]], dtype=np.float32),
        np.array([[-0.1, 0.7, -0.3]], dtype=np.float32),
    )

    coef_with_intercept = logistic_fit_coef_with_intercept(
        fold_coefs,
        multi_class="ovr",
        n_classes=2,
        n_features=2,
        fit_intercept=True,
    )

    np.testing.assert_array_equal(
        coef_with_intercept,
        np.asarray(fold_coefs).reshape(2, 3),
    )
    assert coef_with_intercept.dtype == np.float32
    np.testing.assert_array_equal(logistic_fit_final_coef(coef_with_intercept, fit_intercept=True), coef_with_intercept[:, :-1])
    np.testing.assert_array_equal(
        logistic_fit_final_intercept(coef_with_intercept, n_classes=2, fit_intercept=True),
        coef_with_intercept[:, -1],
    )


def test_logistic_fit_coef_packaging_matches_binary_ovr_effective_one_class() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
    )

    fold_coefs = (np.array([[0.2, -0.4, 1.5]], dtype=np.float64),)

    coef_with_intercept = logistic_fit_coef_with_intercept(
        fold_coefs,
        multi_class="ovr",
        n_classes=1,
        n_features=2,
        fit_intercept=True,
    )

    assert coef_with_intercept.shape == (1, 3)
    np.testing.assert_array_equal(logistic_fit_final_coef(coef_with_intercept, fit_intercept=True), np.array([[0.2, -0.4]]))
    np.testing.assert_array_equal(
        logistic_fit_final_intercept(coef_with_intercept, n_classes=1, fit_intercept=True),
        np.array([1.5]),
    )


def test_logistic_fit_coef_packaging_matches_multinomial_layout() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
    )

    multinomial_path = np.array(
        [
            [
                [0.1, 0.2, 0.3, 1.0],
                [0.4, 0.5, 0.6, -0.5],
                [0.7, 0.8, 0.9, 0.25],
            ]
        ],
        dtype=np.float64,
    )
    fold_coefs = (multinomial_path,)

    coef_with_intercept = logistic_fit_coef_with_intercept(
        fold_coefs,
        multi_class="multinomial",
        n_classes=3,
        n_features=3,
        fit_intercept=True,
    )

    np.testing.assert_array_equal(coef_with_intercept, fold_coefs[0][0])
    np.testing.assert_array_equal(logistic_fit_final_coef(coef_with_intercept, fit_intercept=True), fold_coefs[0][0][:, :-1])
    np.testing.assert_array_equal(
        logistic_fit_final_intercept(coef_with_intercept, n_classes=3, fit_intercept=True),
        fold_coefs[0][0][:, -1],
    )


def test_logistic_fit_no_intercept_branch_returns_zero_intercepts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
    )

    fold_coefs = (
        np.array([[0.2, -0.4]], dtype=np.float64),
        np.array([[-0.1, 0.7]], dtype=np.float64),
    )

    coef_with_intercept = logistic_fit_coef_with_intercept(
        fold_coefs,
        multi_class="ovr",
        n_classes=2,
        n_features=2,
        fit_intercept=False,
    )

    np.testing.assert_array_equal(logistic_fit_final_coef(coef_with_intercept, fit_intercept=False), coef_with_intercept)
    np.testing.assert_array_equal(
        logistic_fit_final_intercept(coef_with_intercept, n_classes=2, fit_intercept=False),
        np.zeros(2),
    )


def test_logistic_fit_postpath_packaging_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_fit_postpath_packaging_shell import (
        logistic_fit_coef_with_intercept,
        logistic_fit_final_coef,
        logistic_fit_final_intercept,
        logistic_fit_n_iter_from_path_results,
        logistic_fit_path_results,
    )

    with pytest.raises(ViolationError):
        logistic_fit_path_results(())

    with pytest.raises(ViolationError):
        logistic_fit_path_results(((np.array([1.0]), np.array([1.0])),))

    with pytest.raises(ViolationError):
        logistic_fit_n_iter_from_path_results(np.array([1, 2], dtype=np.int64))

    with pytest.raises(ViolationError):
        logistic_fit_n_iter_from_path_results(np.array([[-1]], dtype=np.int64))

    with pytest.raises(ViolationError):
        logistic_fit_coef_with_intercept(
            (np.array([[0.1, 0.2]], dtype=np.float64),),
            multi_class="bad",
            n_classes=1,
            n_features=2,
            fit_intercept=False,
        )

    with pytest.raises(ViolationError):
        logistic_fit_coef_with_intercept(
            (np.array([[0.1, 0.2]], dtype=np.float64),),
            multi_class="ovr",
            n_classes=2,
            n_features=2,
            fit_intercept=False,
        )

    with pytest.raises(ViolationError):
        logistic_fit_final_coef(np.array([[0.1]], dtype=np.float64), fit_intercept=True)

    with pytest.raises(ViolationError):
        logistic_fit_final_intercept(np.array([[0.1, 0.2]], dtype=np.float64), n_classes=2, fit_intercept=True)
