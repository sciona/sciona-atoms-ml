from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def _kernel_blocks(kernel: object) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[bool, ...], tuple[str, ...]]:
    hyperparameters = tuple(kernel.hyperparameters)
    params = kernel.get_params()
    value_blocks = tuple(np.atleast_1d(np.asarray(params[hyp.name], dtype=np.float64)) for hyp in hyperparameters)
    bounds_blocks = tuple("fixed" if isinstance(hyp.bounds, str) else np.asarray(hyp.bounds, dtype=np.float64) for hyp in hyperparameters)
    fixed = tuple(bool(hyp.fixed) for hyp in hyperparameters)
    names = tuple(hyp.name for hyp in hyperparameters)
    return value_blocks, bounds_blocks, fixed, names


def test_gp_kernel_hyperparameters_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernel_hyperparameters import (
        gp_kernel_bound_warning_records,
        gp_kernel_bounds,
        gp_kernel_theta,
        gp_kernel_values_from_theta,
    )

    assert callable(gp_kernel_theta)
    assert callable(gp_kernel_values_from_theta)
    assert callable(gp_kernel_bounds)
    assert callable(gp_kernel_bound_warning_records)


def test_gp_kernel_theta_and_bounds_match_sklearn_properties() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernel_hyperparameters import (
        gp_kernel_bounds,
        gp_kernel_theta,
    )

    kernel = ConstantKernel(1.4, constant_value_bounds=(1e-3, 10.0)) * RBF(
        [0.9, 1.7],
        length_scale_bounds=[(1e-2, 5.0), (1e-1, 7.0)],
    )
    value_blocks, bounds_blocks, fixed, _ = _kernel_blocks(kernel)

    assert np.allclose(gp_kernel_theta(value_blocks, fixed), np.asarray(kernel.theta, dtype=np.float64))
    assert np.allclose(gp_kernel_bounds(bounds_blocks, fixed), np.asarray(kernel.bounds, dtype=np.float64))


def test_gp_kernel_values_from_theta_round_trips_blocks() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernel_hyperparameters import (
        gp_kernel_theta,
        gp_kernel_values_from_theta,
    )

    kernel = ConstantKernel(2.0, constant_value_bounds="fixed") * RBF(
        [0.5, 1.5],
        length_scale_bounds=[(1e-2, 5.0), (1e-1, 7.0)],
    )
    value_blocks, _bounds_blocks, fixed, _ = _kernel_blocks(kernel)

    theta = gp_kernel_theta(value_blocks, fixed)
    actual = gp_kernel_values_from_theta(theta, value_blocks, fixed)

    for expected_block, actual_block in zip(value_blocks, actual):
        assert np.allclose(actual_block, expected_block)


def test_gp_kernel_bound_warning_records_match_bound_proximity_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernel_hyperparameters import (
        gp_kernel_bound_warning_records,
    )

    kernel = ConstantKernel(1e-5, constant_value_bounds=(1e-5, 10.0)) * RBF(
        5.0,
        length_scale_bounds=(1e-2, 5.0),
    )
    value_blocks, bounds_blocks, fixed, names = _kernel_blocks(kernel)

    actual = gp_kernel_bound_warning_records(np.asarray(kernel.theta, dtype=np.float64), bounds_blocks, fixed, names)

    assert actual == (
        ("k1__constant_value", 0, "lower", 1e-5),
        ("k2__length_scale", 0, "upper", 5.0),
    )


def test_gp_kernel_hyperparameters_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.kernel_hyperparameters import (
        gp_kernel_bound_warning_records,
        gp_kernel_bounds,
        gp_kernel_theta,
        gp_kernel_values_from_theta,
    )

    with pytest.raises(ViolationError):
        gp_kernel_theta((np.array([-1.0], dtype=np.float64),), (False,))

    with pytest.raises(ViolationError):
        gp_kernel_values_from_theta(
            np.array([0.0, 1.0], dtype=np.float64),
            (np.array([1.0], dtype=np.float64),),
            (False,),
        )

    with pytest.raises(ViolationError):
        gp_kernel_bounds((np.array([[0.0, 1.0]], dtype=np.float64),), (False,))

    with pytest.raises(ViolationError):
        gp_kernel_bound_warning_records(
            np.array([0.0], dtype=np.float64),
            (np.array([[1e-5, 1.0]], dtype=np.float64),),
            (False,),
            (),
        )
