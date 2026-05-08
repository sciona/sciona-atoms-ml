from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot


def test_coordinate_descent_alpha_grid_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_prelude_shell import (
        cd_alpha_grid_dense_Xyw,
        cd_alpha_grid_l1_ratio_zero_error_message,
        cd_alpha_grid_l1_ratio_zero_guard_required,
        cd_alpha_grid_precomputed_Xy,
        cd_alpha_grid_preprocess_kwargs,
        cd_alpha_grid_sparse_mono_output_centered_Xyw,
        cd_alpha_grid_yw,
    )

    assert callable(cd_alpha_grid_l1_ratio_zero_guard_required)
    assert callable(cd_alpha_grid_l1_ratio_zero_error_message)
    assert callable(cd_alpha_grid_precomputed_Xy)
    assert callable(cd_alpha_grid_preprocess_kwargs)
    assert callable(cd_alpha_grid_yw)
    assert callable(cd_alpha_grid_dense_Xyw)
    assert callable(cd_alpha_grid_sparse_mono_output_centered_Xyw)


def test_coordinate_descent_alpha_grid_prelude_shell_matches_guard_and_callbacks() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_prelude_shell import (
        cd_alpha_grid_l1_ratio_zero_error_message,
        cd_alpha_grid_l1_ratio_zero_guard_required,
        cd_alpha_grid_precomputed_Xy,
        cd_alpha_grid_preprocess_kwargs,
    )

    Xy = object()
    sample_weight = object()
    assert cd_alpha_grid_l1_ratio_zero_guard_required(0.0) is True
    assert cd_alpha_grid_l1_ratio_zero_guard_required(0.5) is False
    assert (
        cd_alpha_grid_l1_ratio_zero_error_message(0.0)
        == "Automatic alpha grid generation is not supported for l1_ratio=0. "
        "Please supply a grid by providing your estimator with the appropriate `alphas=` argument."
    )
    assert cd_alpha_grid_precomputed_Xy(Xy) is Xy
    assert cd_alpha_grid_preprocess_kwargs(False, True, sample_weight) == {
        "fit_intercept": False,
        "copy": True,
        "sample_weight": sample_weight,
        "check_input": False,
    }


def test_coordinate_descent_alpha_grid_prelude_shell_weighted_targets() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_prelude_shell import cd_alpha_grid_yw

    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    weights = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    assert cd_alpha_grid_yw(y, None) is y
    assert np.allclose(cd_alpha_grid_yw(y, weights), y * weights)

    y_multi = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    assert np.allclose(cd_alpha_grid_yw(y_multi, weights), y_multi * weights.reshape(-1, 1))


def test_coordinate_descent_alpha_grid_prelude_shell_dense_and_sparse_Xyw() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_prelude_shell import (
        cd_alpha_grid_dense_Xyw,
        cd_alpha_grid_sparse_mono_output_centered_Xyw,
    )

    X_dense = np.array([[1.0, 0.0], [2.0, -1.0], [0.0, 3.0]], dtype=np.float64)
    yw = np.array([0.5, 1.5, -2.0], dtype=np.float64)
    assert np.allclose(cd_alpha_grid_dense_Xyw(X_dense, yw), np.dot(X_dense.T, yw))

    X_sparse = sparse.csr_matrix(X_dense)
    X_offset = np.array([0.25, -0.5], dtype=np.float64)
    expected = safe_sparse_dot(X_sparse.T, yw, dense_output=True) - np.sum(yw) * X_offset
    assert np.allclose(cd_alpha_grid_sparse_mono_output_centered_Xyw(X_sparse, yw, X_offset), expected)


def test_coordinate_descent_alpha_grid_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_prelude_shell import (
        cd_alpha_grid_precomputed_Xy,
        cd_alpha_grid_sparse_mono_output_centered_Xyw,
        cd_alpha_grid_yw,
    )

    with pytest.raises(ViolationError):
        cd_alpha_grid_precomputed_Xy(None)

    with pytest.raises(ViolationError):
        cd_alpha_grid_yw(np.ones((2, 2, 2)), None)

    with pytest.raises(ViolationError):
        cd_alpha_grid_yw(np.ones(3), np.ones(2))

    with pytest.raises(ViolationError):
        cd_alpha_grid_sparse_mono_output_centered_Xyw(
            np.ones((2, 2)),  # type: ignore[arg-type]
            np.ones(2),
            np.ones(2),
        )
