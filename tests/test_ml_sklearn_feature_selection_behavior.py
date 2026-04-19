from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import (
    chi2 as sklearn_chi2,
    f_classif as sklearn_f_classif,
    f_regression as sklearn_f_regression,
    r_regression as sklearn_r_regression,
)


def test_all_four_feature_selection_functions_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import chi2, f_classif, f_regression, r_regression

    assert callable(chi2)
    assert callable(f_classif)
    assert callable(r_regression)
    assert callable(f_regression)


def test_chi2_matches_sklearn_dense_and_sparse() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import chi2

    X = np.array(
        [
            [1.0, 1.0, 3.0],
            [0.0, 1.0, 5.0],
            [5.0, 4.0, 1.0],
            [6.0, 6.0, 2.0],
            [1.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 0, 0, 2, 2])

    result = chi2(X, y)
    expected = sklearn_chi2(X, y)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])

    sparse_result = chi2(sp.csr_matrix(X), y)
    sparse_expected = sklearn_chi2(sp.csr_matrix(X), y)
    assert np.allclose(sparse_result[0], sparse_expected[0])
    assert np.allclose(sparse_result[1], sparse_expected[1])


def test_f_classif_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import f_classif

    X = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [5.0, 4.0, 3.0],
            [6.0, 5.0, 5.0],
            [1.0, 5.0, 7.0],
            [2.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    result = f_classif(X, y)
    expected = sklearn_f_classif(X, y)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_r_regression_matches_sklearn_centered_and_uncentered() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import r_regression

    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 2.5],
            [2.0, 4.0, 3.0],
            [3.0, 8.0, 3.5],
            [4.0, 16.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.1, 1.9, 3.2, 4.1], dtype=np.float64)

    assert np.allclose(r_regression(X, y), sklearn_r_regression(X, y))
    assert np.allclose(
        r_regression(X, y, center=False),
        sklearn_r_regression(X, y, center=False),
    )


def test_f_regression_matches_sklearn_and_forces_constant_feature_finite() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import f_regression

    X = np.array(
        [
            [0.0, 1.0, 5.0],
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 5.0],
            [3.0, 8.0, 5.0],
            [4.0, 16.0, 5.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 1.1, 1.9, 3.2, 4.1], dtype=np.float64)

    result = f_regression(X, y)
    expected = sklearn_f_regression(X, y)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.isfinite(result[0]).all()
    assert np.isfinite(result[1]).all()


def test_score_output_shapes_and_probability_bounds() -> None:
    from sciona.atoms.ml.sklearn.feature_selection import chi2, f_classif, f_regression, r_regression

    X_nonnegative = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0], [4.0, 1.0]])
    y_class = np.array([0, 0, 1, 1])
    X_reg = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])
    y_reg = np.array([0.0, 1.0, 2.0, 3.0])

    for scores, p_values in (chi2(X_nonnegative, y_class), f_classif(X_nonnegative, y_class), f_regression(X_reg, y_reg)):
        assert scores.shape == (2,)
        assert p_values.shape == (2,)
        assert np.all((p_values >= 0.0) & (p_values <= 1.0))

    r_values = r_regression(X_reg, y_reg)
    assert r_values.shape == (2,)
    assert np.all(np.abs(r_values) <= 1.0 + 1e-12)
