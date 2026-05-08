from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse


def test_coordinate_descent_path_residuals_split_slicing_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_split_slicing_shell import (
        cd_path_residuals_X_test_slice,
        cd_path_residuals_X_train_slice,
        cd_path_residuals_y_test_slice,
        cd_path_residuals_y_train_slice,
    )

    assert callable(cd_path_residuals_X_train_slice)
    assert callable(cd_path_residuals_y_train_slice)
    assert callable(cd_path_residuals_X_test_slice)
    assert callable(cd_path_residuals_y_test_slice)


def test_coordinate_descent_path_residuals_split_slicing_shell_matches_numpy_slices() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_split_slicing_shell import (
        cd_path_residuals_X_test_slice,
        cd_path_residuals_X_train_slice,
        cd_path_residuals_y_test_slice,
        cd_path_residuals_y_train_slice,
    )

    X = np.arange(24, dtype=np.float64).reshape(6, 4)
    y = np.arange(12, dtype=np.float64).reshape(6, 2)
    train = np.array([0, 2, 5])
    test = np.array([1, 3, 4])

    assert np.array_equal(cd_path_residuals_X_train_slice(X, train), X[train])
    assert np.array_equal(cd_path_residuals_y_train_slice(y, train), y[train])
    assert np.array_equal(cd_path_residuals_X_test_slice(X, test), X[test])
    assert np.array_equal(cd_path_residuals_y_test_slice(y, test), y[test])


def test_coordinate_descent_path_residuals_split_slicing_shell_handles_sparse_features() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_split_slicing_shell import (
        cd_path_residuals_X_test_slice,
        cd_path_residuals_X_train_slice,
    )

    X = sparse.csr_matrix(np.arange(24, dtype=np.float64).reshape(6, 4))
    train = np.array([0, 2, 5])
    test = np.array([1, 3, 4])

    assert sparse.issparse(cd_path_residuals_X_train_slice(X, train))
    assert np.array_equal(cd_path_residuals_X_train_slice(X, train).toarray(), X[train].toarray())
    assert sparse.issparse(cd_path_residuals_X_test_slice(X, test))
    assert np.array_equal(cd_path_residuals_X_test_slice(X, test).toarray(), X[test].toarray())


def test_coordinate_descent_path_residuals_split_slicing_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_split_slicing_shell import (
        cd_path_residuals_X_train_slice,
        cd_path_residuals_y_test_slice,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_X_train_slice(object(), [0])

    with pytest.raises(ViolationError):
        cd_path_residuals_y_test_slice(np.arange(3), 1)
