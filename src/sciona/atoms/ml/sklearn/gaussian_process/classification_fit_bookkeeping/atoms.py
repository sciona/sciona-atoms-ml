"""Gaussian-process classification fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_fit_class_count,
    witness_gpc_fit_classes,
    witness_gpc_fit_dtype_name,
    witness_gpc_fit_require_multiple_classes,
    witness_gpc_fit_require_not_compound_kernel,
    witness_gpc_fit_use_one_vs_one,
    witness_gpc_fit_use_one_vs_rest,
    witness_gpc_fit_validate_ensure_2d,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _nonempty_1d(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)


def _classes_valid(result: object, y: object) -> bool:
    try:
        observed = np.asarray(result)
        source = np.asarray(y)
    except (TypeError, ValueError):
        return False
    return bool(observed.ndim == 1 and observed.shape[0] >= 1 and np.array_equal(observed, np.unique(source)))


def _unique_nonempty_1d(values: object) -> bool:
    if not _nonempty_1d(values):
        return False
    array = np.asarray(values)
    return bool(np.array_equal(array, np.unique(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _one_vs_mode(value: object) -> bool:
    return bool(isinstance(value, str) and value in {"one_vs_rest", "one_vs_one"})


@register_atom(witness_gpc_fit_require_not_compound_kernel)
@icontract.require(lambda kernel_is_compound: _bool(kernel_is_compound), "kernel_is_compound must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is True, "result must be True when the kernel passes the CompoundKernel guard")
def gpc_fit_require_not_compound_kernel(
    kernel_is_compound: bool,
) -> bool:
    """Apply GaussianProcessClassifier.fit's CompoundKernel rejection guard."""
    if kernel_is_compound:
        raise ValueError("kernel cannot be a CompoundKernel")
    return True


@register_atom(witness_gpc_fit_dtype_name)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(
    lambda result: result in {None, "numeric"},
    "dtype mode must match sklearn's fit-time validation choices",
)
def gpc_fit_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Resolve sklearn's fit-time validate_data dtype mode for Gaussian-process classification."""
    if kernel_is_none_or_requires_vector_input:
        return "numeric"
    return None


@register_atom(witness_gpc_fit_validate_ensure_2d)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "ensure_2d mode must be boolean")
def gpc_fit_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Resolve sklearn's fit-time validate_data ensure_2d mode for Gaussian-process classification."""
    return bool(kernel_is_none_or_requires_vector_input)


@register_atom(witness_gpc_fit_classes)
@icontract.require(lambda y: _nonempty_1d(y), "y must be a nonempty 1D label vector")
@icontract.ensure(lambda result, y: _classes_valid(result, y), "classes must equal np.unique(y)")
def gpc_fit_classes(
    y: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Compute GaussianProcessClassifier.fit's sorted unique class vector."""
    return np.asarray(np.unique(np.asarray(y)), dtype=np.asarray(y).dtype)


@register_atom(witness_gpc_fit_class_count)
@icontract.require(lambda classes: _unique_nonempty_1d(classes), "classes must be a nonempty unique 1D class vector")
@icontract.ensure(lambda result, classes: _positive_int(result) and result == int(np.asarray(classes).shape[0]), "class count must equal classes.size")
def gpc_fit_class_count(
    classes: NDArray[np.object_],
) -> int:
    """Compute GaussianProcessClassifier.fit's fitted class count."""
    return int(np.asarray(classes).shape[0])


@register_atom(witness_gpc_fit_require_multiple_classes)
@icontract.require(lambda classes: _unique_nonempty_1d(classes), "classes must be a nonempty unique 1D class vector")
@icontract.ensure(lambda result, classes: _positive_int(result) and result == int(np.asarray(classes).shape[0]), "validated class count must preserve the number of observed classes")
def gpc_fit_require_multiple_classes(
    classes: NDArray[np.object_],
) -> int:
    """Apply GaussianProcessClassifier.fit's distinct-class requirement."""
    values = np.asarray(classes)
    n_classes = int(values.shape[0])
    if n_classes == 1:
        raise ValueError(
            "GaussianProcessClassifier requires 2 or more distinct classes; "
            "got %d class (only class %s is present)" % (n_classes, values[0])
        )
    return n_classes


@register_atom(witness_gpc_fit_use_one_vs_rest)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda multi_class: _one_vs_mode(multi_class), "multi_class must be 'one_vs_rest' or 'one_vs_one'")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def gpc_fit_use_one_vs_rest(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Decide whether GaussianProcessClassifier.fit wraps the binary estimator in OneVsRestClassifier."""
    return bool(int(n_classes) > 2 and multi_class == "one_vs_rest")


@register_atom(witness_gpc_fit_use_one_vs_one)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda multi_class: _one_vs_mode(multi_class), "multi_class must be 'one_vs_rest' or 'one_vs_one'")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def gpc_fit_use_one_vs_one(
    n_classes: int,
    multi_class: str,
) -> bool:
    """Decide whether GaussianProcessClassifier.fit wraps the binary estimator in OneVsOneClassifier."""
    return bool(int(n_classes) > 2 and multi_class == "one_vs_one")
