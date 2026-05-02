"""Gaussian-process regression sample_y shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.atoms.ml.sklearn.gaussian_process.regression_sampling import (
    gp_sample_y_multi_output,
    gp_sample_y_single_output,
)
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_sample_y_result,
    witness_gp_sample_y_use_multioutput_branch,
)

RandomStateLike = int | np.random.RandomState | None


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_mean(values: object) -> bool:
    return _finite_vector(values) or _finite_matrix(values)


def _valid_n_samples(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _single_output_result_valid(
    result: NDArray[np.float64],
    y_mean: NDArray[np.float64],
    n_samples: int,
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    mean = np.asarray(y_mean, dtype=np.float64)
    return bool(mean.ndim == 1 and values.shape == (mean.shape[0], int(n_samples)) and np.all(np.isfinite(values)))


def _multi_output_result_valid(
    result: NDArray[np.float64],
    y_mean: NDArray[np.float64],
    n_samples: int,
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    mean = np.asarray(y_mean, dtype=np.float64)
    return bool(
        mean.ndim == 2
        and values.shape == (mean.shape[0], mean.shape[1], int(n_samples))
        and np.all(np.isfinite(values))
    )


@register_atom(witness_gp_sample_y_use_multioutput_branch)
@icontract.require(lambda y_mean: _finite_mean(y_mean), "y_mean must be a finite nonempty vector or matrix")
def gp_sample_y_use_multioutput_branch(
    y_mean: NDArray[np.float64],
) -> bool:
    """Decide whether GaussianProcessRegressor.sample_y uses the multi-output sampling branch."""
    return bool(np.asarray(y_mean, dtype=np.float64).ndim > 1)


@register_atom(witness_gp_sample_y_result)
@icontract.require(lambda y_mean: _finite_mean(y_mean), "y_mean must be a finite nonempty vector or matrix")
@icontract.require(lambda n_samples=1: _valid_n_samples(n_samples), "n_samples must be a positive integer")
@icontract.ensure(
    lambda result, y_mean, n_samples=1: (
        _single_output_result_valid(result, y_mean, n_samples)
        if not gp_sample_y_use_multioutput_branch(y_mean)
        else _multi_output_result_valid(result, y_mean, n_samples)
    ),
    "result must match sklearn's single-output or multi-output sample_y shape",
)
def gp_sample_y_result(
    y_mean: NDArray[np.float64],
    y_cov: NDArray[np.float64],
    *,
    n_samples: int = 1,
    random_state: RandomStateLike = 0,
) -> NDArray[np.float64]:
    """Evaluate GaussianProcessRegressor.sample_y from supplied predictive mean and covariance."""
    if gp_sample_y_use_multioutput_branch(y_mean):
        return gp_sample_y_multi_output(
            np.asarray(y_mean, dtype=np.float64),
            np.asarray(y_cov, dtype=np.float64),
            n_samples=int(n_samples),
            random_state=random_state,
        )
    return gp_sample_y_single_output(
        np.asarray(y_mean, dtype=np.float64),
        np.asarray(y_cov, dtype=np.float64),
        n_samples=int(n_samples),
        random_state=random_state,
    )
