from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.svm import l1_min_c as sklearn_l1_min_c


def test_svm_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    assert callable(l1_min_c)


def test_l1_min_c_matches_sklearn_squared_hinge_dense() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    X = np.array([[0.0, 1.0], [1.5, -0.5], [0.3, 0.8], [2.0, 0.0]], dtype=np.float64)
    y = np.array([0, 1, 0, 1])
    assert np.allclose(l1_min_c(X, y, loss="squared_hinge"), sklearn_l1_min_c(X, y, loss="squared_hinge"))


def test_l1_min_c_matches_sklearn_log_without_intercept_sparse() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    X = sp.csr_matrix([[0.0, 1.0, 2.0], [1.5, -0.5, 0.0], [0.3, 0.8, 1.0], [2.0, 0.0, -1.0]])
    y = np.array(["a", "b", "a", "b"])
    result = l1_min_c(X, y, loss="log", fit_intercept=False)
    expected = sklearn_l1_min_c(X, y, loss="log", fit_intercept=False)
    assert np.allclose(result, expected)


def test_l1_min_c_intercept_scaling_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    X = np.array([[0.1, 0.2], [0.3, 0.1], [2.0, 1.0], [2.5, 1.5]], dtype=np.float64)
    y = np.array([-1, -1, 1, 1])
    result = l1_min_c(X, y, intercept_scaling=3.0)
    expected = sklearn_l1_min_c(X, y, intercept_scaling=3.0)
    assert np.allclose(result, expected)


def test_l1_min_c_zero_denominator_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    X = np.zeros((4, 2), dtype=np.float64)
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="Ill-posed l1_min_c"):
        l1_min_c(X, y, fit_intercept=False)
    with pytest.raises(ValueError, match="Ill-posed l1_min_c"):
        sklearn_l1_min_c(X, y, fit_intercept=False)


def test_l1_min_c_rejects_inconsistent_lengths() -> None:
    from sciona.atoms.ml.sklearn.svm import l1_min_c

    with pytest.raises(Exception):
        l1_min_c(np.ones((3, 2)), np.array([0, 1]))
