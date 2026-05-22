"""Sklearn LogisticRegressionCV l1-axis packaging tail atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_cv_coefs_paths_dict_l1_axis,
    witness_logistic_cv_coefs_paths_l1_axis,
    witness_logistic_cv_l1_axis_enabled,
    witness_logistic_cv_n_iter_l1_axis,
    witness_logistic_cv_scores_dict_l1_axis,
    witness_logistic_cv_scores_l1_axis,
)


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _finite_numeric_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _nonnegative_integer_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.integer) and np.all(array >= 0))


def _mapping_value_valid(value: object) -> bool:
    return bool(isinstance(value, Mapping) and len(value) >= 1)


def _reshape_possible(value: object, shape: tuple[int, ...]) -> bool:
    try:
        np.asarray(value).reshape(shape)
    except (TypeError, ValueError):
        return False
    return True


def _coefs_l1_axis_input_valid(coefs_paths: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    return bool(
        _finite_numeric_array(coefs_paths)
        and np.asarray(coefs_paths).ndim >= 2
        and _reshape_possible(coefs_paths, (int(n_folds), int(n_l1_ratios), int(n_Cs), -1))
    )


def _scores_l1_axis_input_valid(scores: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    return bool(
        _finite_numeric_array(scores)
        and np.asarray(scores).ndim >= 1
        and _reshape_possible(scores, (int(n_folds), int(n_l1_ratios), int(n_Cs)))
    )


def _n_iter_l1_axis_input_valid(n_iter: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    return bool(
        _nonnegative_integer_array(n_iter)
        and np.asarray(n_iter).ndim >= 2
        and _reshape_possible(n_iter, (-1, int(n_folds), int(n_l1_ratios), int(n_Cs)))
    )


def _dict_values_valid(values_by_class: object, validator: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    if not _mapping_value_valid(values_by_class):
        return False
    return all(validator(value, n_folds, n_Cs, n_l1_ratios) for value in values_by_class.values())


def _coefs_l1_axis_result_valid(result: NDArray[np.floating], coefs_paths: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    expected = np.asarray(coefs_paths).reshape((int(n_folds), int(n_l1_ratios), int(n_Cs), -1))
    expected = np.transpose(expected, (0, 2, 1, 3))
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _scores_l1_axis_result_valid(result: NDArray[np.floating], scores: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    expected = np.asarray(scores).reshape((int(n_folds), int(n_l1_ratios), int(n_Cs)))
    expected = np.transpose(expected, (0, 2, 1))
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _n_iter_l1_axis_result_valid(result: NDArray[np.integer], n_iter: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    expected = np.asarray(n_iter).reshape((-1, int(n_folds), int(n_l1_ratios), int(n_Cs)))
    expected = np.transpose(expected, (0, 1, 3, 2))
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _dict_result_valid(result: dict[object, object], values_by_class: Mapping[object, object], transformer: object, n_folds: int, n_Cs: int, n_l1_ratios: int) -> bool:
    expected = {key: transformer(value, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios) for key, value in values_by_class.items()}
    return bool(
        isinstance(result, dict)
        and list(result.keys()) == list(expected.keys())
        and all(np.array_equal(result[key], expected[key]) and result[key].dtype == expected[key].dtype for key in expected)
    )


@register_atom(witness_logistic_cv_l1_axis_enabled)
@icontract.ensure(lambda result, public_l1_ratios_param: result is (public_l1_ratios_param is not None), "branch predicate must match self.l1_ratios is not None")
def logistic_cv_l1_axis_enabled(public_l1_ratios_param: object) -> bool:
    """Return whether LogisticRegressionCV adds the public l1-ratio axis."""
    return public_l1_ratios_param is not None


@register_atom(witness_logistic_cv_coefs_paths_l1_axis)
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(lambda coefs_paths, n_folds, n_Cs, n_l1_ratios: _coefs_l1_axis_input_valid(coefs_paths, n_folds, n_Cs, n_l1_ratios), "coefs_paths must reshape to fold-l1-C-coef layout")
@icontract.ensure(lambda result, coefs_paths, n_folds, n_Cs, n_l1_ratios: _coefs_l1_axis_result_valid(result, coefs_paths, n_folds, n_Cs, n_l1_ratios), "coefs_paths must match sklearn l1-axis transpose")
def logistic_cv_coefs_paths_l1_axis(
    coefs_paths: object,
    *,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> NDArray[np.floating]:
    """Return one class's coefs_paths_ with public C-by-l1-ratio axes."""
    values = np.asarray(coefs_paths).reshape((int(n_folds), int(n_l1_ratios), int(n_Cs), -1))
    return np.transpose(values, (0, 2, 1, 3))


@register_atom(witness_logistic_cv_coefs_paths_dict_l1_axis)
@icontract.require(lambda coefs_paths_by_class: _mapping_value_valid(coefs_paths_by_class), "coefs_paths_by_class must be a nonempty mapping")
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(lambda coefs_paths_by_class, n_folds, n_Cs, n_l1_ratios: _dict_values_valid(coefs_paths_by_class, _coefs_l1_axis_input_valid, n_folds, n_Cs, n_l1_ratios), "every coefs_paths value must reshape to fold-l1-C-coef layout")
@icontract.ensure(lambda result, coefs_paths_by_class, n_folds, n_Cs, n_l1_ratios: _dict_result_valid(result, coefs_paths_by_class, logistic_cv_coefs_paths_l1_axis, n_folds, n_Cs, n_l1_ratios), "coefs_paths dict must match per-class l1-axis reshaping")
def logistic_cv_coefs_paths_dict_l1_axis(
    coefs_paths_by_class: Mapping[object, object],
    *,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> dict[object, NDArray[np.floating]]:
    """Return class-keyed coefs_paths_ values with public C-by-l1-ratio axes."""
    return {
        key: logistic_cv_coefs_paths_l1_axis(value, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios)
        for key, value in coefs_paths_by_class.items()
    }


@register_atom(witness_logistic_cv_scores_l1_axis)
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(lambda scores, n_folds, n_Cs, n_l1_ratios: _scores_l1_axis_input_valid(scores, n_folds, n_Cs, n_l1_ratios), "scores must reshape to fold-l1-C layout")
@icontract.ensure(lambda result, scores, n_folds, n_Cs, n_l1_ratios: _scores_l1_axis_result_valid(result, scores, n_folds, n_Cs, n_l1_ratios), "scores must match sklearn l1-axis transpose")
def logistic_cv_scores_l1_axis(
    scores: object,
    *,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> NDArray[np.floating]:
    """Return one class's scores_ with public C-by-l1-ratio axes."""
    values = np.asarray(scores).reshape((int(n_folds), int(n_l1_ratios), int(n_Cs)))
    return np.transpose(values, (0, 2, 1))


@register_atom(witness_logistic_cv_scores_dict_l1_axis)
@icontract.require(lambda scores_by_class: _mapping_value_valid(scores_by_class), "scores_by_class must be a nonempty mapping")
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(lambda scores_by_class, n_folds, n_Cs, n_l1_ratios: _dict_values_valid(scores_by_class, _scores_l1_axis_input_valid, n_folds, n_Cs, n_l1_ratios), "every score value must reshape to fold-l1-C layout")
@icontract.ensure(lambda result, scores_by_class, n_folds, n_Cs, n_l1_ratios: _dict_result_valid(result, scores_by_class, logistic_cv_scores_l1_axis, n_folds, n_Cs, n_l1_ratios), "scores dict must match per-class l1-axis reshaping")
def logistic_cv_scores_dict_l1_axis(
    scores_by_class: Mapping[object, object],
    *,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> dict[object, NDArray[np.floating]]:
    """Return class-keyed scores_ values with public C-by-l1-ratio axes."""
    return {key: logistic_cv_scores_l1_axis(value, n_folds=n_folds, n_Cs=n_Cs, n_l1_ratios=n_l1_ratios) for key, value in scores_by_class.items()}


@register_atom(witness_logistic_cv_n_iter_l1_axis)
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(lambda n_iter, n_folds, n_Cs, n_l1_ratios: _n_iter_l1_axis_input_valid(n_iter, n_folds, n_Cs, n_l1_ratios), "n_iter must reshape to inferred-class-fold-l1-C layout")
@icontract.ensure(lambda result, n_iter, n_folds, n_Cs, n_l1_ratios: _n_iter_l1_axis_result_valid(result, n_iter, n_folds, n_Cs, n_l1_ratios), "n_iter must match sklearn l1-axis transpose")
def logistic_cv_n_iter_l1_axis(
    n_iter: object,
    *,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> NDArray[np.integer]:
    """Return LogisticRegressionCV n_iter_ with public C-by-l1-ratio axes."""
    values = np.asarray(n_iter).reshape((-1, int(n_folds), int(n_l1_ratios), int(n_Cs)))
    return np.transpose(values, (0, 1, 3, 2))
