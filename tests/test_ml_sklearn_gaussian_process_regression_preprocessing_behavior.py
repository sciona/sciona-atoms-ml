from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _single_output_targets() -> np.ndarray:
    return np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64)


def _multi_output_targets() -> np.ndarray:
    return np.column_stack(
        [
            np.array([0.3, -0.1, 0.7, 1.2], dtype=np.float64),
            np.array([1.0, 0.4, -0.2, 0.5], dtype=np.float64),
        ]
    )


def test_gp_regression_preprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import (
        gp_regression_normalize_targets,
        gp_regression_resolve_alpha,
        gp_regression_target_count,
        gp_regression_target_statistics,
        gp_regression_validate_n_targets,
    )

    assert callable(gp_regression_target_count)
    assert callable(gp_regression_validate_n_targets)
    assert callable(gp_regression_target_statistics)
    assert callable(gp_regression_normalize_targets)
    assert callable(gp_regression_resolve_alpha)


def test_gp_regression_target_count_and_validation_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import (
        gp_regression_target_count,
        gp_regression_validate_n_targets,
    )

    y1 = _single_output_targets()
    y2 = _multi_output_targets()

    assert gp_regression_target_count(y1) == 1
    assert gp_regression_target_count(y2) == 2
    assert gp_regression_validate_n_targets(2, n_targets=2) == 2

    with pytest.raises(ValueError, match="The number of targets seen in `y` is different"):
        gp_regression_validate_n_targets(2, n_targets=1)


def test_gp_regression_target_statistics_and_normalization_match_fit_single_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import (
        gp_regression_normalize_targets,
        gp_regression_target_statistics,
    )

    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = _single_output_targets()
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.4, constant_value_bounds="fixed") * RBF(0.9, length_scale_bounds="fixed"),
        alpha=0.05,
        optimizer=None,
        normalize_y=True,
    ).fit(X, y)

    mean, std = gp_regression_target_statistics(y, normalize_y=True)
    normalized = gp_regression_normalize_targets(y, mean, std)

    assert np.isclose(mean, model._y_train_mean)
    assert np.isclose(std, model._y_train_std)
    assert np.allclose(normalized, model.y_train_)


def test_gp_regression_target_statistics_and_normalization_match_fit_multi_output() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import (
        gp_regression_normalize_targets,
        gp_regression_target_statistics,
    )

    X = np.array([[-1.0], [-0.2], [0.4], [1.1]], dtype=np.float64)
    y = _multi_output_targets()
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.2, constant_value_bounds="fixed") * RBF(0.8, length_scale_bounds="fixed"),
        alpha=np.full(X.shape[0], 0.03),
        optimizer=None,
        normalize_y=True,
    ).fit(X, y)

    mean, std = gp_regression_target_statistics(y, normalize_y=True)
    normalized = gp_regression_normalize_targets(y, mean, std)

    assert np.allclose(mean, model._y_train_mean)
    assert np.allclose(std, model._y_train_std)
    assert np.allclose(normalized, model.y_train_)


def test_gp_regression_target_statistics_without_normalization_matches_fit_defaults() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import gp_regression_target_statistics

    y1 = _single_output_targets()
    y2 = _multi_output_targets()

    mean1, std1 = gp_regression_target_statistics(y1, normalize_y=False)
    mean2, std2 = gp_regression_target_statistics(y2, normalize_y=False)

    assert mean1 == 0.0
    assert std1 == 1.0
    assert np.array_equal(mean2, np.zeros(2, dtype=np.float64))
    assert np.array_equal(std2, np.ones(2, dtype=np.float64))


def test_gp_regression_resolve_alpha_matches_sklearn_fit_branch() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import gp_regression_resolve_alpha

    assert gp_regression_resolve_alpha(0.1, n_samples=4) == 0.1
    assert gp_regression_resolve_alpha(np.array([0.2], dtype=np.float64), n_samples=4) == 0.2

    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    assert np.array_equal(gp_regression_resolve_alpha(vector, n_samples=4), vector)

    with pytest.raises(ValueError, match="alpha must be a scalar or an array with same number of entries as y"):
        gp_regression_resolve_alpha(np.array([0.1, 0.2], dtype=np.float64), n_samples=4)


def test_gp_regression_preprocessing_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_preprocessing import (
        gp_regression_normalize_targets,
        gp_regression_resolve_alpha,
        gp_regression_target_count,
    )

    with pytest.raises(ViolationError):
        gp_regression_target_count(np.array([], dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_regression_normalize_targets(
            _multi_output_targets(),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        gp_regression_resolve_alpha(np.array([-1.0], dtype=np.float64), n_samples=4)
