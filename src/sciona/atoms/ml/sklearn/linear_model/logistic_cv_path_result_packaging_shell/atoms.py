"""Sklearn LogisticRegressionCV path-result packaging atoms."""

from __future__ import annotations

from numbers import Integral

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_cv_coefs_paths_by_class,
    witness_logistic_cv_coefs_paths_layout,
    witness_logistic_cv_n_iter_layout,
    witness_logistic_cv_path_results,
    witness_logistic_cv_public_Cs,
    witness_logistic_cv_scores_by_class,
    witness_logistic_cv_scores_layout,
)

_MULTI_CLASS_VALUES = {"ovr", "multinomial"}


def _positive_integer(value: object) -> bool:
    return bool(isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0)


def _multi_class_valid(value: object) -> bool:
    return bool(value in _MULTI_CLASS_VALUES)


def _path_results_valid(path_results: object) -> bool:
    try:
        values = tuple(path_results)  # type: ignore[arg-type]
    except TypeError:
        return False
    return bool(len(values) >= 1 and all(isinstance(item, tuple) and len(item) == 4 for item in values))


def _finite_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _integer_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.integer) and np.all(array >= 0))


def _Cs_valid(Cs: object) -> bool:
    try:
        rows = tuple(Cs)  # type: ignore[arg-type]
    except TypeError:
        return False
    if len(rows) < 1:
        return False
    return _finite_array(rows[0]) and np.asarray(rows[0]).ndim == 1 and np.all(np.asarray(rows[0]) > 0)


def _reshape_possible(value: object, shape: tuple[int, ...]) -> bool:
    try:
        np.reshape(value, shape)
    except (TypeError, ValueError):
        return False
    return True


def _classes_align(classes: object, matrix: object) -> bool:
    try:
        class_values = tuple(classes)  # type: ignore[arg-type]
        values = np.asarray(matrix)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim >= 1 and len(class_values) == values.shape[0])


def _path_results_result_valid(result: tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...], tuple[object, ...]], path_results: object) -> bool:
    expected = tuple(zip(*tuple(path_results)))  # type: ignore[arg-type]
    return bool(isinstance(result, tuple) and len(result) == 4 and result == expected)


def _public_Cs_result_valid(result: NDArray[np.floating], Cs: object) -> bool:
    expected = np.asarray(tuple(Cs)[0])  # type: ignore[arg-type]
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _coefs_paths_layout_valid(
    coefs_paths: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> bool:
    product = int(n_Cs) * int(n_l1_ratios)
    if multi_class == "multinomial":
        shape = (int(n_folds), product, int(n_classes), -1)
    else:
        shape = (int(n_classes), int(n_folds), product, -1)
    return bool(_finite_array(coefs_paths) and _reshape_possible(coefs_paths, shape))


def _n_iter_layout_valid(
    n_iter: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> bool:
    product = int(n_Cs) * int(n_l1_ratios)
    shape = (1 if multi_class == "multinomial" else int(n_classes), int(n_folds), product)
    return bool(_integer_array(n_iter) and _reshape_possible(n_iter, shape))


def _scores_layout_valid(scores: object, multi_class: str, n_classes: int, n_folds: int) -> bool:
    try:
        values = np.asarray(scores)
    except (TypeError, ValueError):
        return False
    if not _finite_array(values):
        return False
    if multi_class == "multinomial":
        try:
            tiled = np.tile(values, (int(n_classes), 1, 1))
            np.reshape(tiled, (int(n_classes), int(n_folds), -1))
        except ValueError:
            return False
        return True
    return _reshape_possible(values, (int(n_classes), int(n_folds), -1))


def _coefs_paths_result_valid(
    result: NDArray[np.floating],
    coefs_paths: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> bool:
    product = int(n_Cs) * int(n_l1_ratios)
    if multi_class == "multinomial":
        expected = np.reshape(coefs_paths, (int(n_folds), product, int(n_classes), -1))
        expected = np.swapaxes(expected, 0, 1)
        expected = np.swapaxes(expected, 0, 2)
    else:
        expected = np.reshape(coefs_paths, (int(n_classes), int(n_folds), product, -1))
    return bool(
        isinstance(result, np.ndarray)
        and result.shape == expected.shape
        and result.dtype == expected.dtype
        and np.array_equal(result, expected)
    )


def _n_iter_result_valid(
    result: NDArray[np.integer],
    n_iter: object,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> bool:
    product = int(n_Cs) * int(n_l1_ratios)
    shape = (1 if multi_class == "multinomial" else int(n_classes), int(n_folds), product)
    expected = np.reshape(n_iter, shape)
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _scores_result_valid(result: NDArray[np.floating], scores: object, multi_class: str, n_classes: int, n_folds: int) -> bool:
    if multi_class == "multinomial":
        expected = np.tile(scores, (int(n_classes), 1, 1))
    else:
        expected = np.asarray(scores)
    expected = np.reshape(expected, (int(n_classes), int(n_folds), -1))
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


def _dict_result_valid(result: dict[object, object], classes: object, values: object) -> bool:
    class_values = tuple(classes)  # type: ignore[arg-type]
    matrix = np.asarray(values)
    expected = dict(zip(class_values, matrix))
    return bool(isinstance(result, dict) and result.keys() == expected.keys() and all(np.array_equal(result[key], expected[key]) for key in expected))


@register_atom(witness_logistic_cv_path_results)
@icontract.require(lambda path_results: _path_results_valid(path_results), "path_results must be nonempty 4-tuples")
@icontract.ensure(lambda result, path_results: _path_results_result_valid(result, path_results), "path results must match zip(*path_results)")
def logistic_cv_path_results(path_results: object) -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    """Return unzipped LogisticRegressionCV scoring-path results."""
    return tuple(zip(*tuple(path_results)))  # type: ignore[arg-type]


@register_atom(witness_logistic_cv_public_Cs)
@icontract.require(lambda Cs: _Cs_valid(Cs), "Cs must contain at least one positive finite C grid")
@icontract.ensure(lambda result, Cs: _public_Cs_result_valid(result, Cs), "public Cs_ must be the first returned C grid")
def logistic_cv_public_Cs(Cs: object) -> NDArray[np.floating]:
    """Return LogisticRegressionCV.fit public Cs_ from path-result C grids."""
    return np.asarray(tuple(Cs)[0])  # type: ignore[arg-type]


@register_atom(witness_logistic_cv_coefs_paths_layout)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(
    lambda coefs_paths, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios: _coefs_paths_layout_valid(
        coefs_paths,
        multi_class,
        n_classes,
        n_folds,
        n_Cs,
        n_l1_ratios,
    ),
    "coefs_paths must match LogisticRegressionCV path-result layout",
)
@icontract.ensure(
    lambda result, coefs_paths, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios: _coefs_paths_result_valid(
        result,
        coefs_paths,
        multi_class,
        n_classes,
        n_folds,
        n_Cs,
        n_l1_ratios,
    ),
    "coefs_paths layout must match sklearn CV packaging",
)
def logistic_cv_coefs_paths_layout(
    coefs_paths: object,
    *,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> NDArray[np.floating]:
    """Return LogisticRegressionCV coefficient paths in class-fold-path layout."""
    product = int(n_Cs) * int(n_l1_ratios)
    if multi_class == "multinomial":
        values = np.reshape(coefs_paths, (int(n_folds), product, int(n_classes), -1))
        values = np.swapaxes(values, 0, 1)
        return np.swapaxes(values, 0, 2)
    return np.reshape(coefs_paths, (int(n_classes), int(n_folds), product, -1))


@register_atom(witness_logistic_cv_n_iter_layout)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda n_Cs: _positive_integer(n_Cs), "n_Cs must be positive")
@icontract.require(lambda n_l1_ratios: _positive_integer(n_l1_ratios), "n_l1_ratios must be positive")
@icontract.require(
    lambda n_iter, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios: _n_iter_layout_valid(
        n_iter,
        multi_class,
        n_classes,
        n_folds,
        n_Cs,
        n_l1_ratios,
    ),
    "n_iter must match LogisticRegressionCV path-result layout",
)
@icontract.ensure(
    lambda result, n_iter, multi_class, n_classes, n_folds, n_Cs, n_l1_ratios: _n_iter_result_valid(
        result,
        n_iter,
        multi_class,
        n_classes,
        n_folds,
        n_Cs,
        n_l1_ratios,
    ),
    "n_iter layout must match sklearn CV packaging",
)
def logistic_cv_n_iter_layout(
    n_iter: object,
    *,
    multi_class: str,
    n_classes: int,
    n_folds: int,
    n_Cs: int,
    n_l1_ratios: int,
) -> NDArray[np.integer]:
    """Return LogisticRegressionCV n_iter_ in class/fold/path layout."""
    product = int(n_Cs) * int(n_l1_ratios)
    shape = (1 if multi_class == "multinomial" else int(n_classes), int(n_folds), product)
    return np.reshape(n_iter, shape)


@register_atom(witness_logistic_cv_scores_layout)
@icontract.require(lambda multi_class: _multi_class_valid(multi_class), "multi_class must be ovr or multinomial")
@icontract.require(lambda n_classes: _positive_integer(n_classes), "n_classes must be positive")
@icontract.require(lambda n_folds: _positive_integer(n_folds), "n_folds must be positive")
@icontract.require(lambda scores, multi_class, n_classes, n_folds: _scores_layout_valid(scores, multi_class, n_classes, n_folds), "scores must match LogisticRegressionCV path-result layout")
@icontract.ensure(lambda result, scores, multi_class, n_classes, n_folds: _scores_result_valid(result, scores, multi_class, n_classes, n_folds), "scores layout must match sklearn CV packaging")
def logistic_cv_scores_layout(scores: object, *, multi_class: str, n_classes: int, n_folds: int) -> NDArray[np.floating]:
    """Return LogisticRegressionCV scores in class-fold-path layout."""
    if multi_class == "multinomial":
        scores = np.tile(scores, (int(n_classes), 1, 1))
    return np.reshape(scores, (int(n_classes), int(n_folds), -1))


@register_atom(witness_logistic_cv_scores_by_class)
@icontract.require(lambda classes, scores: _classes_align(classes, scores), "classes must align with score rows")
@icontract.ensure(lambda result, classes, scores: _dict_result_valid(result, classes, scores), "scores dict must match dict(zip(classes, scores))")
def logistic_cv_scores_by_class(classes: object, scores: object) -> dict[object, NDArray[np.floating]]:
    """Return LogisticRegressionCV scores_ dictionary keyed by class."""
    return dict(zip(tuple(classes), np.asarray(scores)))  # type: ignore[arg-type]


@register_atom(witness_logistic_cv_coefs_paths_by_class)
@icontract.require(lambda classes, coefs_paths: _classes_align(classes, coefs_paths), "classes must align with coefficient-path rows")
@icontract.ensure(lambda result, classes, coefs_paths: _dict_result_valid(result, classes, coefs_paths), "coefs_paths dict must match dict(zip(classes, coefs_paths))")
def logistic_cv_coefs_paths_by_class(classes: object, coefs_paths: object) -> dict[object, NDArray[np.floating]]:
    """Return LogisticRegressionCV coefs_paths_ dictionary keyed by class."""
    return dict(zip(tuple(classes), np.asarray(coefs_paths)))  # type: ignore[arg-type]
