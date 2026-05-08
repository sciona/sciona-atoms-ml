"""Gaussian-process classification fit-state shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Callable

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_fit_binary_base_estimator,
    witness_gpc_fit_binary_log_marginal_likelihood_value,
    witness_gpc_fit_multiclass_log_marginal_likelihood_value,
    witness_gpc_fit_one_vs_one_estimator,
    witness_gpc_fit_one_vs_rest_estimator,
    witness_gpc_fit_return_self,
)

OptimizerLike = str | Callable[..., object] | None
RandomStateLike = int | np.random.RandomState | None

def _kernel_or_none(value: object) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    return value is None or isinstance(value, Kernel)

def _optimizer_valid(value: object) -> bool:
    return value is None or isinstance(value, str) or callable(value)

def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1

def _bool(value: object) -> bool:
    return isinstance(value, bool)

def _n_jobs_valid(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))

def _finite_float(value: object) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))

def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and value != ""

def _same_kernel(result_kernel: Kernel | None, source_kernel: Kernel | None) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    if source_kernel is None:
        return result_kernel is None
    if result_kernel is None:
        return False
    return bool(np.array_equal(result_kernel.theta, source_kernel.theta))

def _binary_base_estimator_valid(
    result: _BinaryGaussianProcessClassifierLaplace,
    kernel: Kernel | None,
    optimizer: OptimizerLike,
    n_restarts_optimizer: int,
    max_iter_predict: int,
    warm_start: bool,
    copy_X_train: bool,
    random_state: RandomStateLike,
) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    return bool(
        isinstance(result, _BinaryGaussianProcessClassifierLaplace)
        and _same_kernel(result.kernel, kernel)
        and result.optimizer == optimizer
        and result.n_restarts_optimizer == int(n_restarts_optimizer)
        and result.max_iter_predict == int(max_iter_predict)
        and result.warm_start is warm_start
        and result.copy_X_train is copy_X_train
        and result.random_state == random_state
    )

def _ovr_estimator_valid(
    result: OneVsRestClassifier,
    base_estimator: _BinaryGaussianProcessClassifierLaplace,
    n_jobs: int | None,
) -> bool:
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    return bool(
        isinstance(result, OneVsRestClassifier)
        and result.estimator is base_estimator
        and result.n_jobs == n_jobs
    )

def _ovo_estimator_valid(
    result: OneVsOneClassifier,
    base_estimator: _BinaryGaussianProcessClassifierLaplace,
    n_jobs: int | None,
) -> bool:
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    return bool(
        isinstance(result, OneVsOneClassifier)
        and result.estimator is base_estimator
        and result.n_jobs == n_jobs
    )

@register_atom(witness_gpc_fit_binary_base_estimator)
@icontract.require(lambda kernel: _kernel_or_none(kernel), "kernel must be a sklearn Kernel or None")
@icontract.require(lambda optimizer: _optimizer_valid(optimizer), "optimizer must be a string, callable, or None")
@icontract.require(lambda n_restarts_optimizer: _nonnegative_int(n_restarts_optimizer), "n_restarts_optimizer must be a nonnegative integer")
@icontract.require(lambda max_iter_predict: _positive_int(max_iter_predict), "max_iter_predict must be a positive integer")
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda copy_X_train: _bool(copy_X_train), "copy_X_train must be boolean")
@icontract.ensure(
    lambda result, kernel, optimizer, n_restarts_optimizer, max_iter_predict, warm_start, copy_X_train, random_state:
    _binary_base_estimator_valid(
        result,
        kernel,
        optimizer,
        n_restarts_optimizer,
        max_iter_predict,
        warm_start,
        copy_X_train,
        random_state,
    ),
    "binary base estimator must preserve GaussianProcessClassifier.fit constructor arguments",
)
def gpc_fit_binary_base_estimator(
    kernel: Kernel | None,
    optimizer: OptimizerLike,
    n_restarts_optimizer: int,
    max_iter_predict: int,
    warm_start: bool,
    copy_X_train: bool,
    random_state: RandomStateLike,
) -> _BinaryGaussianProcessClassifierLaplace:
    from sklearn.base import clone
    from sklearn.gaussian_process.kernels import Kernel
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    """Construct the binary base estimator template used by GaussianProcessClassifier.fit."""
    return _BinaryGaussianProcessClassifierLaplace(
        kernel=clone(kernel) if kernel is not None else None,
        optimizer=optimizer,
        n_restarts_optimizer=int(n_restarts_optimizer),
        max_iter_predict=int(max_iter_predict),
        warm_start=warm_start,
        copy_X_train=copy_X_train,
        random_state=random_state,
    )

@register_atom(witness_gpc_fit_one_vs_rest_estimator)
@icontract.require(
    lambda base_estimator: isinstance(base_estimator, _BinaryGaussianProcessClassifierLaplace),
    "base_estimator must be a binary Gaussian-process classifier",
)
@icontract.require(lambda n_jobs=None: _n_jobs_valid(n_jobs), "n_jobs must be an integer or None")
@icontract.ensure(
    lambda result, base_estimator, n_jobs=None: _ovr_estimator_valid(result, base_estimator, n_jobs),
    "one-vs-rest estimator must wrap the supplied binary base estimator and preserve n_jobs",
)
def gpc_fit_one_vs_rest_estimator(
    base_estimator: _BinaryGaussianProcessClassifierLaplace,
    n_jobs: int | None = None,
) -> OneVsRestClassifier:
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    """Wrap GaussianProcessClassifier's binary base estimator in OneVsRestClassifier."""
    return OneVsRestClassifier(base_estimator, n_jobs=n_jobs)

@register_atom(witness_gpc_fit_one_vs_one_estimator)
@icontract.require(
    lambda base_estimator: isinstance(base_estimator, _BinaryGaussianProcessClassifierLaplace),
    "base_estimator must be a binary Gaussian-process classifier",
)
@icontract.require(lambda n_jobs=None: _n_jobs_valid(n_jobs), "n_jobs must be an integer or None")
@icontract.ensure(
    lambda result, base_estimator, n_jobs=None: _ovo_estimator_valid(result, base_estimator, n_jobs),
    "one-vs-one estimator must wrap the supplied binary base estimator and preserve n_jobs",
)
def gpc_fit_one_vs_one_estimator(
    base_estimator: _BinaryGaussianProcessClassifierLaplace,
    n_jobs: int | None = None,
) -> OneVsOneClassifier:
    from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier,
    _BinaryGaussianProcessClassifierLaplace,
    )
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    """Wrap GaussianProcessClassifier's binary base estimator in OneVsOneClassifier."""
    return OneVsOneClassifier(base_estimator, n_jobs=n_jobs)

@register_atom(witness_gpc_fit_binary_log_marginal_likelihood_value)
@icontract.require(
    lambda base_estimator_log_marginal_likelihood_value: _finite_float(base_estimator_log_marginal_likelihood_value),
    "base_estimator_log_marginal_likelihood_value must be finite",
)
@icontract.ensure(
    lambda result, base_estimator_log_marginal_likelihood_value: _finite_float(result)
    and float(result) == float(base_estimator_log_marginal_likelihood_value),
    "binary fit log-marginal-likelihood value must be preserved",
)
def gpc_fit_binary_log_marginal_likelihood_value(
    base_estimator_log_marginal_likelihood_value: float,
) -> float:
    """Expose GaussianProcessClassifier.fit's binary log_marginal_likelihood_value_ assignment."""
    return float(base_estimator_log_marginal_likelihood_value)

@register_atom(witness_gpc_fit_multiclass_log_marginal_likelihood_value)
@icontract.require(
    lambda estimator_log_marginal_likelihood_values: _finite_vector(estimator_log_marginal_likelihood_values),
    "estimator_log_marginal_likelihood_values must be a finite nonempty vector",
)
@icontract.ensure(lambda result: _finite_float(result), "multiclass fit log-marginal-likelihood value must be finite")
def gpc_fit_multiclass_log_marginal_likelihood_value(
    estimator_log_marginal_likelihood_values: NDArray[np.float64],
) -> float:
    """Compute GaussianProcessClassifier.fit's multiclass mean log_marginal_likelihood_value_."""
    return float(np.mean(np.asarray(estimator_log_marginal_likelihood_values, dtype=np.float64)))

@register_atom(witness_gpc_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: result == estimator_token, "fit return value must preserve the estimator token")
def gpc_fit_return_self(estimator_token: str) -> str:
    """Model GaussianProcessClassifier.fit returning self after fitted-state side effects."""
    return estimator_token
