"""Sklearn LogisticRegression fit post-path packaging atoms."""

from __future__ import annotations

from numbers import Integral

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_fit_coef_with_intercept,
    witness_logistic_fit_final_coef,
    witness_logistic_fit_final_intercept,
    witness_logistic_fit_n_iter_from_path_results,
    witness_logistic_fit_path_results,
)

_MULTI_CLASS_VALUES = {"ovr", "multinomial"}


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _bool_value(value: object) -> bool:
    return bool(isinstance(value, (bool, np.bool_)))


def _multi_class_valid(value: object) -> bool:
    return bool(value in _MULTI_CLASS_VALUES)


def _finite_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _n_iter_input_valid(n_iter: object) -> bool:
    try:
        values = np.asarray(n_iter)
        int_values = values.astype(np.int32)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(int_values >= 0))


def _path_results_valid(path_results: object) -> bool:
    try:
        values = tuple(path_results)  # type: ignore[arg-type]
    except TypeError:
        return False
    return bool(len(values) >= 1 and all(isinstance(item, tuple) and len(item) == 3 for item in values))


def _fold_coefs_valid(
    fold_coefs: object,
    multi_class: str,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> bool:
    width = int(n_features) + int(bool(fit_intercept))
    try:
        if multi_class == "multinomial":
            values = np.asarray(fold_coefs[0][0])  # type: ignore[index]
            return bool(values.shape == (int(n_classes), width) and np.all(np.isfinite(values)))
        values = np.asarray(fold_coefs)
        reshaped = values.reshape(int(n_classes), width)
    except (TypeError, ValueError, IndexError):
        return False
    return bool(reshaped.shape == (int(n_classes), width) and np.all(np.isfinite(reshaped)))


def _coef_width_valid(coef_with_intercept: object, fit_intercept: bool) -> bool:
    try:
        values = np.asarray(coef_with_intercept)
    except (TypeError, ValueError):
        return False
    min_width = 2 if fit_intercept else 1
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= min_width and np.all(np.isfinite(values)))


def _coef_classes_valid(coef_with_intercept: object, n_classes: int) -> bool:
    try:
        values = np.asarray(coef_with_intercept)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] == int(n_classes))


def _n_iter_result_valid(result: NDArray[np.int32], n_iter: object) -> bool:
    expected = np.asarray(n_iter, dtype=np.int32)[:, 0]
    return bool(isinstance(result, np.ndarray) and result.dtype == np.int32 and np.array_equal(result, expected))


def _path_results_result_valid(result: tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]], path_results: object) -> bool:
    expected = tuple(zip(*tuple(path_results)))  # type: ignore[arg-type]
    return bool(isinstance(result, tuple) and len(result) == 3 and result == expected)


def _coef_with_intercept_result_valid(
    result: NDArray[np.float64],
    fold_coefs: object,
    multi_class: str,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> bool:
    width = int(n_features) + int(bool(fit_intercept))
    if multi_class == "multinomial":
        expected = np.asarray(fold_coefs[0][0])  # type: ignore[index]
    else:
        expected = np.asarray(fold_coefs).reshape(int(n_classes), width)
    return bool(
        isinstance(result, np.ndarray)
        and result.shape == (int(n_classes), width)
        and result.dtype == expected.dtype
        and np.array_equal(result, expected)
        and np.all(np.isfinite(result))
    )


def _final_coef_result_valid(result: NDArray[np.float64], coef_with_intercept: object, fit_intercept: bool) -> bool:
    values = np.asarray(coef_with_intercept)
    expected = values[:, :-1] if fit_intercept else values
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _final_intercept_result_valid(
    result: NDArray[np.float64],
    coef_with_intercept: object,
    n_classes: int,
    fit_intercept: bool,
) -> bool:
    values = np.asarray(coef_with_intercept)
    expected = values[:, -1] if fit_intercept else np.zeros(int(n_classes))
    return bool(isinstance(result, np.ndarray) and result.shape == (int(n_classes),) and result.dtype == expected.dtype and np.array_equal(result, expected))


@register_atom(witness_logistic_fit_path_results)
@icontract.require(lambda path_results: _path_results_valid(path_results), "path_results must be nonempty 3-tuples")
@icontract.ensure(lambda result, path_results: _path_results_result_valid(result, path_results), "path results must match zip(*path_results)")
def logistic_fit_path_results(path_results: object) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """Return unzipped LogisticRegression.fit path results."""
    return tuple(zip(*tuple(path_results)))  # type: ignore[arg-type]


@register_atom(witness_logistic_fit_n_iter_from_path_results)
@icontract.require(lambda n_iter: _n_iter_input_valid(n_iter), "n_iter must be a nonnegative two-dimensional path result")
@icontract.ensure(lambda result, n_iter: _n_iter_result_valid(result, n_iter), "n_iter_ must match sklearn post-path slicing")
def logistic_fit_n_iter_from_path_results(n_iter: object) -> NDArray[np.int32]:
    """Return LogisticRegression.fit n_iter_ from path result n_iter values."""
    return np.asarray(n_iter, dtype=np.int32)[:, 0]


@register_atom(witness_logistic_fit_coef_with_intercept)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(
    lambda fold_coefs, multi_class, n_classes, n_features, fit_intercept: _fold_coefs_valid(
        fold_coefs,
        multi_class,
        n_classes,
        n_features,
        fit_intercept,
    ),
    "fold coefficients must match sklearn post-path layout",
)
@icontract.ensure(
    lambda result, fold_coefs, multi_class, n_classes, n_features, fit_intercept: _coef_with_intercept_result_valid(
        result,
        fold_coefs,
        multi_class,
        n_classes,
        n_features,
        fit_intercept,
    ),
    "coef matrix must match LogisticRegression.fit post-path packaging",
)
def logistic_fit_coef_with_intercept(
    fold_coefs: object,
    *,
    multi_class: str,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> NDArray[np.floating]:
    """Return the post-path coefficient matrix before intercept splitting."""
    width = int(n_features) + int(bool(fit_intercept))
    if multi_class == "multinomial":
        return np.asarray(fold_coefs[0][0])  # type: ignore[index]
    return np.asarray(fold_coefs).reshape(int(n_classes), width)


@register_atom(witness_logistic_fit_final_coef)
@icontract.require(lambda coef_with_intercept: _finite_matrix(coef_with_intercept), "coef_with_intercept must be a finite 2D matrix")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda coef_with_intercept, fit_intercept: _coef_width_valid(coef_with_intercept, fit_intercept), "coefficient matrix must have enough columns")
@icontract.ensure(lambda result, coef_with_intercept, fit_intercept: _final_coef_result_valid(result, coef_with_intercept, fit_intercept), "final coef must match sklearn intercept split")
def logistic_fit_final_coef(coef_with_intercept: object, *, fit_intercept: bool) -> NDArray[np.floating]:
    """Return LogisticRegression.fit final coef_ from the packaged coefficient matrix."""
    values = np.asarray(coef_with_intercept)
    if fit_intercept:
        return values[:, :-1]
    return values


@register_atom(witness_logistic_fit_final_intercept)
@icontract.require(lambda coef_with_intercept: _finite_matrix(coef_with_intercept), "coef_with_intercept must be a finite 2D matrix")
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda coef_with_intercept, n_classes: _coef_classes_valid(coef_with_intercept, n_classes), "coefficient rows must match n_classes")
@icontract.require(lambda coef_with_intercept, fit_intercept: _coef_width_valid(coef_with_intercept, fit_intercept), "coefficient matrix must have enough columns")
@icontract.ensure(
    lambda result, coef_with_intercept, n_classes, fit_intercept: _final_intercept_result_valid(
        result,
        coef_with_intercept,
        n_classes,
        fit_intercept,
    ),
    "final intercept must match sklearn intercept split",
)
def logistic_fit_final_intercept(
    coef_with_intercept: object,
    *,
    n_classes: int,
    fit_intercept: bool,
) -> NDArray[np.floating]:
    """Return LogisticRegression.fit final intercept_ from the packaged coefficient matrix."""
    values = np.asarray(coef_with_intercept)
    if fit_intercept:
        return values[:, -1]
    return np.zeros(int(n_classes))
