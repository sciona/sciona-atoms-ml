"""Sklearn SGDOneClassSVM fit shell atoms."""

from __future__ import annotations

from numbers import Integral, Real

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sgd_one_class_average_active,
    witness_sgd_one_class_average_buffers,
    witness_sgd_one_class_fit_one_class_payload,
    witness_sgd_one_class_fixed_solver_context,
    witness_sgd_one_class_intercept_from_offset,
    witness_sgd_one_class_offset_from_intercept,
    witness_sgd_one_class_parameter_allocation_payload,
    witness_sgd_one_class_partial_fit_result,
    witness_sgd_one_class_target,
    witness_sgd_one_class_time_step_after_fit,
    witness_sgd_one_class_validation_sample_mask,
)

_FIT_ONE_CLASS_PAYLOAD_KEYS = {
    "X",
    "alpha",
    "C",
    "learning_rate",
    "sample_weight",
    "max_iter",
}

_ALLOCATION_PAYLOAD_KEYS = {
    "n_classes",
    "n_features",
    "input_dtype",
    "coef_init",
    "intercept_init",
    "one_class",
}

_AVERAGE_BUFFER_KEYS = {"average_coef", "average_intercept"}


def _finite_real(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)))


def _finite_nonnegative_real(value: object) -> bool:
    return bool(_finite_real(value) and float(value) >= 0.0)


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _nonnegative_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0)


def _float_dtype(value: object) -> bool:
    try:
        dtype = np.dtype(value)
    except TypeError:
        return False
    return bool(dtype in (np.dtype(np.float32), np.dtype(np.float64)))


def _validated_one_class_X(X: object) -> bool:
    try:
        array = np.asarray(X)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] > 0 and _float_dtype(array.dtype))


def _finite_vector(value: object) -> bool:
    try:
        array = np.atleast_1d(np.asarray(value))
    except (TypeError, ValueError):
        return False
    return bool(array.size > 0 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _finite_sample_weight(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size > 0 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _average_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    return bool(isinstance(value, Real) and np.isfinite(float(value)) and float(value) >= 0.0)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value)


def _target_valid(result: np.ndarray, X: object) -> bool:
    array = np.asarray(X)
    return bool(
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and result.shape == (array.shape[0],)
        and result.dtype == array.dtype
        and result.flags.c_contiguous
        and np.all(result == 1)
    )


def _offset_intercept_valid(result: np.ndarray, source: object) -> bool:
    expected = 1 - np.atleast_1d(source)
    return bool(
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and result.shape == expected.shape
        and np.allclose(result, expected)
        and np.all(np.isfinite(result))
    )


def _sample_mask_valid(result: np.ndarray, sample_weight: object) -> bool:
    expected = np.asarray(sample_weight) > 0
    return bool(
        isinstance(result, np.ndarray)
        and result.dtype == np.dtype(bool)
        and result.shape == expected.shape
        and np.array_equal(result, expected)
    )


def _average_buffers_valid(result: dict[str, np.ndarray], n_features: int, dtype: object) -> bool:
    expected_dtype = np.dtype(dtype)
    return bool(
        set(result) == _AVERAGE_BUFFER_KEYS
        and isinstance(result["average_coef"], np.ndarray)
        and isinstance(result["average_intercept"], np.ndarray)
        and result["average_coef"].shape == (int(n_features),)
        and result["average_intercept"].shape == (1,)
        and result["average_coef"].dtype == expected_dtype
        and result["average_intercept"].dtype == expected_dtype
        and result["average_coef"].flags.c_contiguous
        and result["average_intercept"].flags.c_contiguous
        and np.all(result["average_coef"] == 0)
        and np.all(result["average_intercept"] == 0)
    )


def _allocation_payload_valid(
    result: dict[str, object],
    n_features: int,
    input_dtype: object,
    coef_init: object,
    offset_init: object,
) -> bool:
    return bool(
        set(result) == _ALLOCATION_PAYLOAD_KEYS
        and result["n_classes"] == 1
        and result["n_features"] == int(n_features)
        and result["input_dtype"] == np.dtype(input_dtype)
        and result["coef_init"] is coef_init
        and result["intercept_init"] is offset_init
        and result["one_class"] == 1
    )


def _fit_one_class_payload_valid(
    result: dict[str, object],
    X: object,
    alpha: float,
    C: float,
    learning_rate: str,
    sample_weight: object,
    max_iter: int,
) -> bool:
    return bool(
        set(result) == _FIT_ONE_CLASS_PAYLOAD_KEYS
        and result["X"] is X
        and result["alpha"] == float(alpha)
        and result["C"] == float(C)
        and result["learning_rate"] == learning_rate
        and result["sample_weight"] is sample_weight
        and result["max_iter"] == int(max_iter)
    )


@register_atom(witness_sgd_one_class_target)
@icontract.require(lambda X: _validated_one_class_X(X), "X must be a nonempty 2D float32/float64 array-like")
@icontract.ensure(lambda result, X: _target_valid(result, X), "target must be all ones with X dtype and sample count")
def sgd_one_class_target(X: object) -> np.ndarray:
    """Return the artificial y=np.ones(n_samples) target for SGDOneClassSVM."""
    array = np.asarray(X)
    return np.ones(array.shape[0], dtype=array.dtype, order="C")


@register_atom(witness_sgd_one_class_fixed_solver_context)
@icontract.require(lambda one_class: one_class == 1, "one_class must be the sklearn one-class solver flag")
@icontract.ensure(lambda result: result == (1, 1, 1), "one-class and class-weight flags must be fixed")
def sgd_one_class_fixed_solver_context(one_class: int = 1) -> tuple[int, int, int]:
    """Return the fixed one_class, pos_weight, and neg_weight solver context."""
    del one_class
    return (1, 1, 1)


@register_atom(witness_sgd_one_class_validation_sample_mask)
@icontract.require(lambda sample_weight: _finite_sample_weight(sample_weight), "sample_weight must be a finite 1D vector")
@icontract.ensure(
    lambda result, sample_weight: _sample_mask_valid(result, sample_weight),
    "validation sample mask must mark strictly positive sample weights",
)
def sgd_one_class_validation_sample_mask(sample_weight: object) -> np.ndarray:
    """Return the sample mask passed into SGDOneClassSVM validation splitting."""
    return np.asarray(sample_weight) > 0


@register_atom(witness_sgd_one_class_intercept_from_offset)
@icontract.require(lambda offset: _finite_vector(offset), "offset must be a finite scalar or vector")
@icontract.ensure(
    lambda result, offset: _offset_intercept_valid(result, offset),
    "solver intercept must be 1 - atleast_1d(offset)",
)
def sgd_one_class_intercept_from_offset(offset: object) -> np.ndarray:
    """Return the SGD solver intercept initialized from the one-class offset."""
    return 1 - np.atleast_1d(offset)


@register_atom(witness_sgd_one_class_offset_from_intercept)
@icontract.require(lambda intercept: _finite_vector(intercept), "intercept must be a finite scalar or vector")
@icontract.ensure(
    lambda result, intercept: _offset_intercept_valid(result, intercept),
    "one-class offset must be 1 - atleast_1d(intercept)",
)
def sgd_one_class_offset_from_intercept(intercept: object) -> np.ndarray:
    """Return the public one-class offset written from a solver intercept."""
    return 1 - np.atleast_1d(intercept)


@register_atom(witness_sgd_one_class_time_step_after_fit)
@icontract.require(lambda t_before: _finite_real(t_before) and float(t_before) > 0.0, "t_before must be positive and finite")
@icontract.require(lambda n_iter: _nonnegative_integer(n_iter), "n_iter must be a nonnegative integer")
@icontract.require(lambda n_samples: _positive_integer(n_samples), "n_samples must be a positive integer")
@icontract.ensure(
    lambda result, t_before, n_iter, n_samples: result == float(t_before) + int(n_iter) * int(n_samples),
    "t_ must advance by n_iter * n_samples",
)
def sgd_one_class_time_step_after_fit(t_before: float, n_iter: int, n_samples: int) -> float:
    """Return SGDOneClassSVM.t_ after the delegated solver fit."""
    return float(t_before) + int(n_iter) * int(n_samples)


@register_atom(witness_sgd_one_class_average_active)
@icontract.require(lambda average: _average_value(average), "average must be bool-like or finite nonnegative numeric")
@icontract.require(lambda t_after: _finite_real(t_after) and float(t_after) > 0.0, "t_after must be positive and finite")
@icontract.ensure(
    lambda result, average, t_after: result is bool(average > 0 and average <= float(t_after) - 1.0),
    "average branch must follow sklearn's average <= t_ - 1 threshold",
)
def sgd_one_class_average_active(average: object, t_after: float) -> bool:
    """Return whether averaged coefficients and offset should be used."""
    return bool(average > 0 and average <= float(t_after) - 1.0)


@register_atom(witness_sgd_one_class_average_buffers)
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be a positive integer")
@icontract.require(lambda dtype: _float_dtype(dtype), "dtype must be float32 or float64")
@icontract.ensure(
    lambda result, n_features, dtype: _average_buffers_valid(result, n_features, dtype),
    "average buffers must be zero arrays with sklearn shapes and dtype",
)
def sgd_one_class_average_buffers(n_features: int, dtype: object) -> dict[str, np.ndarray]:
    """Return average coefficient and intercept buffers allocated by _partial_fit."""
    return {
        "average_coef": np.zeros(int(n_features), dtype=np.dtype(dtype), order="C"),
        "average_intercept": np.zeros(1, dtype=np.dtype(dtype), order="C"),
    }


@register_atom(witness_sgd_one_class_parameter_allocation_payload)
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be a positive integer")
@icontract.require(lambda input_dtype: _float_dtype(input_dtype), "input_dtype must be float32 or float64")
@icontract.ensure(
    lambda result, n_features, input_dtype, coef_init, offset_init: _allocation_payload_valid(
        result,
        n_features,
        input_dtype,
        coef_init,
        offset_init,
    ),
    "allocation payload must match SGDOneClassSVM _allocate_parameter_mem arguments",
)
def sgd_one_class_parameter_allocation_payload(
    n_features: int,
    input_dtype: object,
    *,
    coef_init: object = None,
    offset_init: object = None,
) -> dict[str, object]:
    """Return the one-class parameter allocation payload from _partial_fit."""
    return {
        "n_classes": 1,
        "n_features": int(n_features),
        "input_dtype": np.dtype(input_dtype),
        "coef_init": coef_init,
        "intercept_init": offset_init,
        "one_class": 1,
    }


@register_atom(witness_sgd_one_class_fit_one_class_payload)
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda alpha: _finite_nonnegative_real(alpha), "alpha must be finite and nonnegative")
@icontract.require(lambda C: _finite_nonnegative_real(C), "C must be finite and nonnegative")
@icontract.require(lambda learning_rate: _nonempty_string(learning_rate), "learning_rate must be a nonempty string")
@icontract.require(lambda max_iter: _positive_integer(max_iter), "max_iter must be a positive integer")
@icontract.ensure(
    lambda result, X, alpha, C, learning_rate, sample_weight, max_iter: _fit_one_class_payload_valid(
        result,
        X,
        alpha,
        C,
        learning_rate,
        sample_weight,
        max_iter,
    ),
    "_fit_one_class payload must match SGDOneClassSVM._partial_fit delegation",
)
def sgd_one_class_fit_one_class_payload(
    X: object,
    *,
    alpha: float,
    C: float,
    learning_rate: str,
    sample_weight: object,
    max_iter: int,
) -> dict[str, object]:
    """Return the delegated _fit_one_class callback payload from _partial_fit."""
    return {
        "X": X,
        "alpha": float(alpha),
        "C": float(C),
        "learning_rate": learning_rate,
        "sample_weight": sample_weight,
        "max_iter": int(max_iter),
    }


@register_atom(witness_sgd_one_class_partial_fit_result)
@icontract.require(lambda estimator: estimator is not None, "estimator must be provided")
@icontract.ensure(lambda result, estimator: result is estimator, "_partial_fit must return the estimator by identity")
def sgd_one_class_partial_fit_result(estimator: object) -> object:
    """Return the estimator object after delegated SGDOneClassSVM fitting."""
    return estimator
