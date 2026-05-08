from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse


def test_coordinate_descent_set_order_helper_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_set_order_helper_shell import (
        cd_set_order_conversion_required,
        cd_set_order_invalid_order_message,
        cd_set_order_invalid_order_required,
        cd_set_order_outputs,
        cd_set_order_sparse_format,
    )

    assert callable(cd_set_order_invalid_order_required)
    assert callable(cd_set_order_invalid_order_message)
    assert callable(cd_set_order_conversion_required)
    assert callable(cd_set_order_sparse_format)
    assert callable(cd_set_order_outputs)


def test_coordinate_descent_set_order_helper_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_set_order_helper_shell import (
        cd_set_order_conversion_required,
        cd_set_order_invalid_order_message,
        cd_set_order_invalid_order_required,
        cd_set_order_outputs,
        cd_set_order_sparse_format,
    )

    assert cd_set_order_invalid_order_required(None) is False
    assert cd_set_order_invalid_order_required("C") is False
    assert cd_set_order_invalid_order_required("F") is False
    assert cd_set_order_invalid_order_required("A") is True
    assert (
        cd_set_order_invalid_order_message("A")
        == "Unknown value for order. Got A instead of None, 'C' or 'F'."
    )

    assert cd_set_order_conversion_required(None) is False
    assert cd_set_order_conversion_required("C") is True
    assert cd_set_order_conversion_required("F") is True
    assert cd_set_order_sparse_format("C") == "csr"
    assert cd_set_order_sparse_format("F") == "csc"

    X = np.array([[1.0, 2.0], [3.0, 4.0]], order="C")
    y = np.array([[5.0, 6.0], [7.0, 8.0]], order="C")
    none_X, none_y = cd_set_order_outputs(X, y, None)
    assert none_X is X
    assert none_y is y

    f_X, f_y = cd_set_order_outputs(X, y, "F")
    assert np.array_equal(f_X, X)
    assert np.array_equal(f_y, y)
    assert f_X.flags["F_CONTIGUOUS"]
    assert f_y.flags["F_CONTIGUOUS"]

    c_X, c_y = cd_set_order_outputs(np.asfortranarray(X), np.asfortranarray(y), "C")
    assert np.array_equal(c_X, X)
    assert np.array_equal(c_y, y)
    assert c_X.flags["C_CONTIGUOUS"]
    assert c_y.flags["C_CONTIGUOUS"]

    sparse_X = sparse.csr_matrix(X)
    sparse_y = sparse.coo_matrix(y)
    f_sparse_X, f_sparse_y = cd_set_order_outputs(sparse_X, sparse_y, "F")
    assert sparse.isspmatrix_csc(f_sparse_X)
    assert sparse.isspmatrix_csc(f_sparse_y)
    assert np.array_equal(f_sparse_X.toarray(), X)
    assert np.array_equal(f_sparse_y.toarray(), y)

    c_sparse_X, c_sparse_y = cd_set_order_outputs(sparse.csc_matrix(X), sparse.coo_matrix(y), "C")
    assert sparse.isspmatrix_csr(c_sparse_X)
    assert sparse.isspmatrix_csr(c_sparse_y)
    assert np.array_equal(c_sparse_X.toarray(), X)
    assert np.array_equal(c_sparse_y.toarray(), y)


def test_coordinate_descent_set_order_helper_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_set_order_helper_shell import (
        cd_set_order_conversion_required,
        cd_set_order_outputs,
        cd_set_order_sparse_format,
    )

    with pytest.raises(ViolationError):
        cd_set_order_conversion_required("A")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_set_order_sparse_format(None)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_set_order_outputs([[1.0]], [[2.0]], "A")  # type: ignore[arg-type]
