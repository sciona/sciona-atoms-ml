from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model._coordinate_descent import _alpha_grid


def test_coordinate_descent_alpha_grid_math_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_math import (
        cd_alpha_grid_alpha_max,
        cd_alpha_grid_sample_count,
        cd_alpha_grid_use_resolution_fallback,
        cd_alpha_grid_values,
        cd_alpha_grid_xyw_matrix,
    )

    assert callable(cd_alpha_grid_xyw_matrix)
    assert callable(cd_alpha_grid_sample_count)
    assert callable(cd_alpha_grid_alpha_max)
    assert callable(cd_alpha_grid_use_resolution_fallback)
    assert callable(cd_alpha_grid_values)


def test_coordinate_descent_alpha_grid_math_matches_sklearn_runtime() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_math import (
        cd_alpha_grid_alpha_max,
        cd_alpha_grid_sample_count,
        cd_alpha_grid_values,
        cd_alpha_grid_xyw_matrix,
    )

    rng = np.random.RandomState(0)
    X = rng.randn(6, 4)
    y = rng.randn(6)
    Xy = X.T @ y

    Xyw = cd_alpha_grid_xyw_matrix(Xy)
    sample_count = cd_alpha_grid_sample_count(n_samples=X.shape[0])
    alpha_max = cd_alpha_grid_alpha_max(Xyw, sample_count=sample_count, l1_ratio=0.7)
    observed = cd_alpha_grid_values(alpha_max=alpha_max, eps=1e-3, n_alphas=5)
    expected = _alpha_grid(X, y, Xy=Xy, l1_ratio=0.7, fit_intercept=False, eps=1e-3, n_alphas=5, copy_X=False)
    assert np.allclose(observed, expected)


def test_coordinate_descent_alpha_grid_math_resolution_fallback() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_math import (
        cd_alpha_grid_use_resolution_fallback,
        cd_alpha_grid_values,
    )

    resolution = np.finfo(np.float64).resolution
    assert cd_alpha_grid_use_resolution_fallback(resolution) is True
    assert np.array_equal(cd_alpha_grid_values(alpha_max=resolution, eps=1e-3, n_alphas=3), np.full(3, resolution))


def test_coordinate_descent_alpha_grid_math_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_alpha_grid_math import (
        cd_alpha_grid_sample_count,
        cd_alpha_grid_values,
        cd_alpha_grid_xyw_matrix,
    )

    with pytest.raises(ViolationError):
        cd_alpha_grid_xyw_matrix(np.array(1.0))

    with pytest.raises(ViolationError):
        cd_alpha_grid_sample_count(n_samples=0)

    with pytest.raises(ViolationError):
        cd_alpha_grid_values(alpha_max=-1.0, eps=1e-3, n_alphas=3)
