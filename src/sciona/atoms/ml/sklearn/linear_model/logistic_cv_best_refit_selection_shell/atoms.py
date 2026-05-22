"""Sklearn LogisticRegressionCV best/refit selection atoms."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_cv_best_C_l1_selection,
    witness_logistic_cv_best_flat_index,
    witness_logistic_cv_loop_path_views,
    witness_logistic_cv_multinomial_final_components,
    witness_logistic_cv_nonrefit_average_C,
    witness_logistic_cv_nonrefit_average_l1_ratio,
    witness_logistic_cv_nonrefit_average_w,
    witness_logistic_cv_nonrefit_best_indices,
    witness_logistic_cv_ovr_final_row,
    witness_logistic_cv_refit_coef_init,
)

_MULTI_CLASS_VALUES = {"ovr", "multinomial"}


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _nonnegative_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0)


def _multi_class_valid(value: object) -> bool:
    return bool(value in _MULTI_CLASS_VALUES)


def _finite_numeric_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _finite_1d_array(value: object) -> bool:
    return bool(_finite_numeric_array(value) and np.asarray(value).ndim == 1)


def _nonempty_1d_array(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and array.ndim == 1)


def _integer_1d_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and array.ndim == 1 and np.issubdtype(array.dtype, np.integer) and np.all(array >= 0))


def _bool_value(value: object) -> bool:
    return bool(isinstance(value, bool))


def _mapping_has_key(values: object, key: object) -> bool:
    return bool(isinstance(values, Mapping) and key in values)


def _scores_valid(scores: object) -> bool:
    return bool(_finite_numeric_array(scores) and np.asarray(scores).ndim >= 1)


def _best_index_in_range(best_index: int, Cs: object, l1_ratios: object) -> bool:
    return bool(
        _nonnegative_integer(best_index)
        and _finite_1d_array(Cs)
        and _nonempty_1d_array(l1_ratios)
        and int(best_index) < np.asarray(Cs).size * np.asarray(l1_ratios).size
    )


def _refit_coef_input_valid(coefs_paths: object, multi_class: str, best_index: int) -> bool:
    if not (_multi_class_valid(multi_class) and _nonnegative_integer(best_index) and _finite_numeric_array(coefs_paths)):
        return False
    values = np.asarray(coefs_paths)
    if multi_class == "multinomial":
        return bool(values.ndim == 4 and int(best_index) < values.shape[2])
    return bool(values.ndim == 3 and int(best_index) < values.shape[1])


def _nonrefit_w_input_valid(coefs_paths: object, best_indices: object, multi_class: str) -> bool:
    if not (_multi_class_valid(multi_class) and _finite_numeric_array(coefs_paths) and _integer_1d_array(best_indices)):
        return False
    values = np.asarray(coefs_paths)
    indices = np.asarray(best_indices)
    if multi_class == "multinomial":
        return bool(values.ndim == 4 and len(indices) == values.shape[1] and np.all(indices < values.shape[2]))
    return bool(values.ndim == 3 and len(indices) == values.shape[0] and np.all(indices < values.shape[1]))


def _n_features_valid(w: object, n_features: int, fit_intercept: bool) -> bool:
    if not (_positive_integer(n_features) and _bool_value(fit_intercept) and _finite_numeric_array(w)):
        return False
    values = np.asarray(w)
    width = int(n_features) + (1 if fit_intercept else 0)
    return bool(values.shape[-1] >= width)


def _elasticnet_l1_grid_valid(l1_ratios: object, penalty: str) -> bool:
    if penalty != "elasticnet":
        return _nonempty_1d_array(l1_ratios)
    return _finite_1d_array(l1_ratios)


def _nonrefit_l1_indices_valid(best_indices: object, Cs: object, l1_ratios: object, penalty: str) -> bool:
    if not (_integer_1d_array(best_indices) and _finite_1d_array(Cs) and _elasticnet_l1_grid_valid(l1_ratios, penalty)):
        return False
    if penalty != "elasticnet":
        return True
    return bool(np.all(np.asarray(best_indices) < np.asarray(Cs).size * np.asarray(l1_ratios).size))


def _best_flat_index_result_valid(result: int, scores: object) -> bool:
    expected = int(np.asarray(scores).sum(axis=0).argmax())
    return bool(isinstance(result, int) and result == expected)


def _best_C_l1_result_valid(result: tuple[object, object], best_index: int, Cs: object, l1_ratios: object) -> bool:
    C_values = np.asarray(Cs)
    l1_values = np.asarray(l1_ratios)
    best_index_C = int(best_index) % C_values.size
    best_index_l1 = int(best_index) // C_values.size
    return bool(len(result) == 2 and result[0] == C_values[best_index_C] and result[1] == l1_values[best_index_l1])


def _refit_coef_result_valid(result: NDArray[np.floating], coefs_paths: object, multi_class: str, best_index: int) -> bool:
    values = np.asarray(coefs_paths)
    if multi_class == "multinomial":
        expected = np.mean(values[:, :, int(best_index), :], axis=1)
    else:
        expected = np.mean(values[:, int(best_index), :], axis=0)
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _nonrefit_best_indices_result_valid(result: NDArray[np.integer], scores: object) -> bool:
    expected = np.argmax(scores, axis=1)
    return bool(isinstance(result, np.ndarray) and np.array_equal(result, expected))


def _nonrefit_w_result_valid(result: NDArray[np.floating], coefs_paths: object, best_indices: object, multi_class: str) -> bool:
    values = np.asarray(coefs_paths)
    indices = np.asarray(best_indices)
    if multi_class == "multinomial":
        expected = np.mean([values[:, i, indices[i], :] for i in range(len(indices))], axis=0)
    else:
        expected = np.mean([values[i, indices[i], :] for i in range(len(indices))], axis=0)
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _nonrefit_average_C_valid(result: object, best_indices: object, Cs: object) -> bool:
    expected = np.mean(np.asarray(Cs)[np.asarray(best_indices) % np.asarray(Cs).size])
    return bool(result == expected)


def _nonrefit_average_l1_valid(result: object, best_indices: object, Cs: object, l1_ratios: object, penalty: str) -> bool:
    if penalty != "elasticnet":
        return result is None
    expected = np.mean(np.asarray(l1_ratios)[np.asarray(best_indices) // np.asarray(Cs).size])
    return bool(result == expected)


def _multinomial_final_components_valid(
    result: tuple[object, object, object, object],
    C_values: object,
    l1_ratio_values: object,
    w: object,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> bool:
    expected_C = np.tile(C_values, int(n_classes))
    expected_l1 = np.tile(l1_ratio_values, int(n_classes))
    weights = np.asarray(w)
    expected_coef = weights[:, : int(n_features)]
    expected_intercept = weights[:, -1] if fit_intercept else None
    return bool(
        isinstance(result, tuple)
        and len(result) == 4
        and np.array_equal(result[0], expected_C)
        and np.array_equal(result[1], expected_l1)
        and isinstance(result[2], np.ndarray)
        and result[2].dtype == expected_coef.dtype
        and np.array_equal(result[2], expected_coef)
        and (
            (result[3] is None and expected_intercept is None)
            or (
                isinstance(result[3], np.ndarray)
                and isinstance(expected_intercept, np.ndarray)
                and result[3].dtype == expected_intercept.dtype
                and np.array_equal(result[3], expected_intercept)
            )
        )
    )


def _ovr_final_row_valid(result: tuple[object, object], w: object, n_features: int, fit_intercept: bool) -> bool:
    weights = np.asarray(w)
    expected_coef = weights[: int(n_features)]
    expected_intercept = weights[-1] if fit_intercept else None
    return bool(
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], np.ndarray)
        and result[0].dtype == expected_coef.dtype
        and np.array_equal(result[0], expected_coef)
        and ((result[1] is None and expected_intercept is None) or result[1] == expected_intercept)
    )


@register_atom(witness_logistic_cv_loop_path_views)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(
    lambda multi_class, cls, scores_by_class, coefs_paths_by_class, multinomial_scores, multinomial_coefs_paths: (
        (_mapping_has_key(scores_by_class, cls) and _mapping_has_key(coefs_paths_by_class, cls))
        if multi_class == "ovr"
        else (_scores_valid(multinomial_scores) and np.asarray(multinomial_scores).ndim >= 2 and _finite_numeric_array(multinomial_coefs_paths))
    ),
    "path views must be available for the selected multi_class branch",
)
def logistic_cv_loop_path_views(
    multi_class: str,
    cls: object,
    scores_by_class: Mapping[object, object],
    coefs_paths_by_class: Mapping[object, object],
    multinomial_scores: object,
    multinomial_coefs_paths: object,
) -> tuple[object, object]:
    """Return the scores and coefficient paths visible inside one CV loop body."""
    if multi_class == "ovr":
        return scores_by_class[cls], coefs_paths_by_class[cls]
    return np.asarray(multinomial_scores)[0], multinomial_coefs_paths


@register_atom(witness_logistic_cv_best_flat_index)
@icontract.require(lambda scores: _scores_valid(scores), "scores must be finite numeric values")
@icontract.ensure(lambda result, scores: _best_flat_index_result_valid(result, scores), "best index must match scores.sum(axis=0).argmax()")
def logistic_cv_best_flat_index(scores: object) -> int:
    """Return the flattened best path index from summed CV scores."""
    return int(np.asarray(scores).sum(axis=0).argmax())


@register_atom(witness_logistic_cv_best_C_l1_selection)
@icontract.require(lambda best_index, Cs, l1_ratios: _best_index_in_range(best_index, Cs, l1_ratios), "best_index must fit the flattened C/l1 grid")
@icontract.ensure(lambda result, best_index, Cs, l1_ratios: _best_C_l1_result_valid(result, best_index, Cs, l1_ratios), "best C and l1-ratio must match sklearn flattened-index split")
def logistic_cv_best_C_l1_selection(best_index: int, Cs: object, l1_ratios: object) -> tuple[object, object]:
    """Return selected C and l1-ratio from a flattened best index."""
    C_values = np.asarray(Cs)
    l1_values = np.asarray(l1_ratios)
    best_index_C = int(best_index) % C_values.size
    best_index_l1 = int(best_index) // C_values.size
    return C_values[best_index_C], l1_values[best_index_l1]


@register_atom(witness_logistic_cv_refit_coef_init)
@icontract.require(lambda coefs_paths, multi_class, best_index: _refit_coef_input_valid(coefs_paths, multi_class, best_index), "coefs_paths must contain the selected best index")
@icontract.ensure(lambda result, coefs_paths, multi_class, best_index: _refit_coef_result_valid(result, coefs_paths, multi_class, best_index), "coef_init must match sklearn branch-specific fold mean")
def logistic_cv_refit_coef_init(coefs_paths: object, *, multi_class: str, best_index: int) -> NDArray[np.floating]:
    """Return the initial coefficients supplied to the refit solver boundary."""
    values = np.asarray(coefs_paths)
    if multi_class == "multinomial":
        return np.mean(values[:, :, int(best_index), :], axis=1)
    return np.mean(values[:, int(best_index), :], axis=0)


@register_atom(witness_logistic_cv_nonrefit_best_indices)
@icontract.require(lambda scores: _scores_valid(scores) and np.asarray(scores).ndim >= 2, "scores must be a finite fold-by-path matrix")
@icontract.ensure(lambda result, scores: _nonrefit_best_indices_result_valid(result, scores), "best indices must match np.argmax(scores, axis=1)")
def logistic_cv_nonrefit_best_indices(scores: object) -> NDArray[np.integer]:
    """Return one best path index per fold for non-refit CV."""
    return np.argmax(scores, axis=1)


@register_atom(witness_logistic_cv_nonrefit_average_w)
@icontract.require(lambda coefs_paths, best_indices, multi_class: _nonrefit_w_input_valid(coefs_paths, best_indices, multi_class), "coefs_paths and best_indices must align by fold")
@icontract.ensure(lambda result, coefs_paths, best_indices, multi_class: _nonrefit_w_result_valid(result, coefs_paths, best_indices, multi_class), "non-refit weights must match sklearn fold-winner average")
def logistic_cv_nonrefit_average_w(coefs_paths: object, best_indices: object, *, multi_class: str) -> NDArray[np.floating]:
    """Return averaged coefficients from fold-specific best path entries."""
    values = np.asarray(coefs_paths)
    indices = np.asarray(best_indices)
    if multi_class == "multinomial":
        return np.mean([values[:, i, indices[i], :] for i in range(len(indices))], axis=0)
    return np.mean([values[i, indices[i], :] for i in range(len(indices))], axis=0)


@register_atom(witness_logistic_cv_nonrefit_average_C)
@icontract.require(lambda best_indices: _integer_1d_array(best_indices), "best_indices must be nonnegative integer indices")
@icontract.require(lambda Cs: _finite_1d_array(Cs), "Cs must be a finite one-dimensional C grid")
@icontract.ensure(lambda result, best_indices, Cs: _nonrefit_average_C_valid(result, best_indices, Cs), "mean C must match sklearn non-refit selection")
def logistic_cv_nonrefit_average_C(best_indices: object, Cs: object) -> object:
    """Return the non-refit mean selected C value."""
    C_values = np.asarray(Cs)
    return np.mean(C_values[np.asarray(best_indices) % C_values.size])


@register_atom(witness_logistic_cv_nonrefit_average_l1_ratio)
@icontract.require(lambda best_indices: _integer_1d_array(best_indices), "best_indices must be nonnegative integer indices")
@icontract.require(lambda Cs: _finite_1d_array(Cs), "Cs must be a finite one-dimensional C grid")
@icontract.require(lambda best_indices, Cs, l1_ratios, penalty: _nonrefit_l1_indices_valid(best_indices, Cs, l1_ratios, penalty), "l1-ratio inputs must match the penalty branch")
@icontract.require(lambda penalty: isinstance(penalty, str), "penalty must be a string")
@icontract.ensure(lambda result, best_indices, Cs, l1_ratios, penalty: _nonrefit_average_l1_valid(result, best_indices, Cs, l1_ratios, penalty), "mean l1-ratio must match sklearn non-refit selection")
def logistic_cv_nonrefit_average_l1_ratio(best_indices: object, Cs: object, l1_ratios: object, penalty: str) -> object:
    """Return the non-refit mean selected l1-ratio for elastic-net, else None."""
    if penalty != "elasticnet":
        return None
    C_values = np.asarray(Cs)
    return np.mean(np.asarray(l1_ratios)[np.asarray(best_indices) // C_values.size])


@register_atom(witness_logistic_cv_multinomial_final_components)
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be a bool")
@icontract.require(lambda w, n_features, fit_intercept: _n_features_valid(w, n_features, fit_intercept) and np.asarray(w).ndim == 2, "w must be a finite class-by-weight matrix")
@icontract.ensure(lambda result, C_values, l1_ratio_values, w, n_classes, n_features, fit_intercept: _multinomial_final_components_valid(result, C_values, l1_ratio_values, w, n_classes, n_features, fit_intercept), "multinomial final components must match sklearn packaging")
def logistic_cv_multinomial_final_components(
    C_values: object,
    l1_ratio_values: object,
    w: object,
    *,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> tuple[NDArray[object], NDArray[object], NDArray[np.floating], NDArray[np.floating] | None]:
    """Return multinomial final C, l1-ratio, coef, and optional intercept components."""
    C_tiled = np.tile(C_values, int(n_classes))
    l1_tiled = np.tile(l1_ratio_values, int(n_classes))
    weights = np.asarray(w)
    coef = weights[:, : int(n_features)]
    intercept = weights[:, -1] if fit_intercept else None
    return C_tiled, l1_tiled, coef, intercept


@register_atom(witness_logistic_cv_ovr_final_row)
@icontract.require(lambda n_features: _positive_integer(n_features), "n_features must be positive")
@icontract.require(lambda fit_intercept: _bool_value(fit_intercept), "fit_intercept must be a bool")
@icontract.require(lambda w, n_features, fit_intercept: _n_features_valid(w, n_features, fit_intercept) and np.asarray(w).ndim == 1, "w must be a finite one-dimensional weight vector")
@icontract.ensure(lambda result, w, n_features, fit_intercept: _ovr_final_row_valid(result, w, n_features, fit_intercept), "OvR final row must match sklearn packaging")
def logistic_cv_ovr_final_row(w: object, *, n_features: int, fit_intercept: bool) -> tuple[NDArray[np.floating], object]:
    """Return one OvR coefficient row and optional intercept from a supplied weight vector."""
    weights = np.asarray(w)
    coef_row = weights[: int(n_features)]
    intercept = weights[-1] if fit_intercept else None
    return coef_row, intercept
