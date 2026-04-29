"""One-vs-one partial-fit preprocessing helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_one_partial_fit_binary_targets,
    witness_one_vs_one_partial_fit_estimator_count,
    witness_one_vs_one_partial_fit_pair_mask,
    witness_one_vs_one_partial_fit_subset_indices,
    witness_one_vs_one_partial_fit_unknown_classes,
)


def _class_vector_valid(values: object, *, min_classes: int = 2) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= min_classes
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _target_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_class_value(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _pair_classes_valid(class_i: object, class_j: object) -> bool:
    return bool(_finite_class_value(class_i) and _finite_class_value(class_j) and float(class_i) != float(class_j))


def _pair_mask_valid(result: object, y: object) -> bool:
    values = np.asarray(result)
    targets = np.asarray(y, dtype=np.float64)
    return bool(values.dtype == np.bool_ and values.shape == targets.shape)


def _subset_indices_valid(result: object, pair_mask: object) -> bool:
    values = np.asarray(result)
    mask = np.asarray(pair_mask, dtype=np.bool_)
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.array_equal(values, np.flatnonzero(mask).astype(np.int64))
    )


def _unknown_classes_valid(result: object, classes: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
        class_values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
        and np.intersect1d(values, class_values).size == 0
    )


def _binary_targets_valid(result: object, y: object, class_i: object, class_j: object) -> bool:
    values = np.asarray(result)
    targets = np.asarray(y, dtype=np.float64)
    cond = np.logical_or(targets == float(class_i), targets == float(class_j))
    return bool(
        values.ndim == 1
        and values.shape == (int(np.sum(cond)),)
        and np.issubdtype(values.dtype, np.integer)
        and np.all((values == 0) | (values == 1))
    )


@register_atom(witness_one_vs_one_partial_fit_estimator_count)
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector with at least two classes")
@icontract.ensure(lambda result: isinstance(result, int) and result >= 1, "estimator count must be a positive integer")
def one_vs_one_partial_fit_estimator_count(
    classes: NDArray[np.float64],
) -> int:
    """Count sklearn's one-vs-one binary estimators for a supplied class vector."""
    n_classes = int(np.asarray(classes, dtype=np.float64).shape[0])
    return int(n_classes * (n_classes - 1) // 2)


@register_atom(witness_one_vs_one_partial_fit_unknown_classes)
@icontract.require(lambda y: _target_vector_valid(y), "y must be a nonempty finite 1D target vector")
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector with at least two classes")
@icontract.ensure(lambda result, classes: _unknown_classes_valid(result, classes), "unknown classes must be finite, unique, and disjoint from classes")
def one_vs_one_partial_fit_unknown_classes(
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return sorted unique partial-fit labels absent from the known one-vs-one class vector."""
    targets = np.asarray(y, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(np.setdiff1d(np.unique(targets), class_values), dtype=np.float64)


@register_atom(witness_one_vs_one_partial_fit_pair_mask)
@icontract.require(lambda y: _target_vector_valid(y), "y must be a nonempty finite 1D target vector")
@icontract.require(lambda class_i, class_j: _pair_classes_valid(class_i, class_j), "class_i and class_j must be distinct finite class values")
@icontract.ensure(lambda result, y: _pair_mask_valid(result, y), "pair mask must be a boolean vector aligned to y")
def one_vs_one_partial_fit_pair_mask(
    y: NDArray[np.float64],
    class_i: float,
    class_j: float,
) -> NDArray[np.bool_]:
    """Select the samples that belong to one one-vs-one class pair."""
    targets = np.asarray(y, dtype=np.float64)
    return np.asarray(np.logical_or(targets == float(class_i), targets == float(class_j)), dtype=np.bool_)


@register_atom(witness_one_vs_one_partial_fit_subset_indices)
@icontract.require(lambda pair_mask: isinstance(np.asarray(pair_mask).dtype.type(), np.bool_), "pair_mask must be a boolean vector")
@icontract.require(lambda pair_mask: np.asarray(pair_mask).ndim == 1 and np.asarray(pair_mask).shape[0] >= 1, "pair_mask must be a nonempty 1D vector")
@icontract.ensure(lambda result, pair_mask: _subset_indices_valid(result, pair_mask), "subset indices must equal the mask's true positions")
def one_vs_one_partial_fit_subset_indices(
    pair_mask: NDArray[np.bool_],
) -> NDArray[np.int64]:
    """Return the selected sample indices for one one-vs-one partial-fit pair."""
    return np.asarray(np.flatnonzero(np.asarray(pair_mask, dtype=np.bool_)), dtype=np.int64)


@register_atom(witness_one_vs_one_partial_fit_binary_targets)
@icontract.require(lambda y: _target_vector_valid(y), "y must be a nonempty finite 1D target vector")
@icontract.require(lambda class_i, class_j: _pair_classes_valid(class_i, class_j), "class_i and class_j must be distinct finite class values")
@icontract.ensure(lambda result, y, class_i, class_j: _binary_targets_valid(result, y, class_i, class_j), "binary targets must align to the selected pair subset and contain only 0/1 values")
def one_vs_one_partial_fit_binary_targets(
    y: NDArray[np.float64],
    class_i: float,
    class_j: float,
) -> NDArray[np.int64]:
    """Encode one one-vs-one partial-fit class pair as 0/1 targets after pair filtering."""
    targets = np.asarray(y, dtype=np.float64)
    cond = np.logical_or(targets == float(class_i), targets == float(class_j))
    selected = targets[cond]
    binary_targets = np.zeros_like(selected, dtype=np.int64)
    binary_targets[selected == float(class_j)] = 1
    return np.asarray(binary_targets, dtype=np.int64)
