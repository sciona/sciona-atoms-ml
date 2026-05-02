"""Binary Gaussian-process classification fit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Kernel, RBF

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_fit_classes,
    witness_gpc_binary_fit_encoded_targets,
    witness_gpc_binary_fit_kernel,
    witness_gpc_binary_fit_require_binary_classes,
    witness_gpc_binary_fit_stored_train_inputs,
    witness_gpc_binary_fit_use_optimizer_branch,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _kernel_or_none(value: object) -> bool:
    return value is None or isinstance(value, Kernel)


def _kernel(value: object) -> bool:
    return isinstance(value, Kernel)


def _nonempty_1d(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)


def _same_shape_and_values(result: object, source: object) -> bool:
    try:
        left = np.asarray(result)
        right = np.asarray(source)
    except (TypeError, ValueError):
        return False
    return bool(left.shape == right.shape and np.array_equal(left, right))


def _classes_valid(result: object, y: object) -> bool:
    try:
        observed = np.asarray(result)
        source = np.asarray(y)
    except (TypeError, ValueError):
        return False
    return bool(observed.ndim == 1 and observed.shape[0] >= 1 and np.array_equal(observed, np.unique(source)))


def _binary_codes_valid(result: object, y: object) -> bool:
    try:
        encoded = np.asarray(result)
        source = np.asarray(y)
    except (TypeError, ValueError):
        return False
    if encoded.ndim != 1 or source.ndim != 1 or encoded.shape != source.shape:
        return False
    classes, inverse = np.unique(source, return_inverse=True)
    return bool(classes.shape[0] >= 1 and np.array_equal(encoded, inverse))


@register_atom(witness_gpc_binary_fit_kernel)
@icontract.require(lambda kernel: _kernel_or_none(kernel), "kernel must be a sklearn Kernel or None")
@icontract.ensure(lambda result: _kernel(result), "result must be a sklearn Kernel")
def gpc_binary_fit_kernel(
    kernel: Kernel | None,
) -> Kernel:
    """Resolve _BinaryGaussianProcessClassifierLaplace.fit's default-or-cloned kernel."""
    if kernel is None:
        return C(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
    return clone(kernel)


@register_atom(witness_gpc_binary_fit_stored_train_inputs)
@icontract.require(lambda X: _nonempty_1d(X) or (hasattr(np.asarray(X), "ndim") and np.asarray(X).ndim >= 1), "X must be a nonempty array-like")
@icontract.require(lambda copy_X_train: _bool(copy_X_train), "copy_X_train must be boolean")
@icontract.ensure(
    lambda result, X: _same_shape_and_values(result, X),
    "stored training inputs must preserve the input values and shape",
)
def gpc_binary_fit_stored_train_inputs(
    X: NDArray[np.object_],
    copy_X_train: bool,
) -> NDArray[np.object_]:
    """Store training inputs using sklearn's copy_X_train policy."""
    values = np.asarray(X)
    if copy_X_train:
        return np.copy(values)
    return values


@register_atom(witness_gpc_binary_fit_classes)
@icontract.require(lambda y: _nonempty_1d(y), "y must be a nonempty 1D label vector")
@icontract.ensure(lambda result, y: _classes_valid(result, y), "classes must equal np.unique(y)")
def gpc_binary_fit_classes(
    y: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Compute _BinaryGaussianProcessClassifierLaplace.fit's sorted class vector."""
    values = np.asarray(y)
    return np.asarray(np.unique(values), dtype=values.dtype)


@register_atom(witness_gpc_binary_fit_encoded_targets)
@icontract.require(lambda y: _nonempty_1d(y), "y must be a nonempty 1D label vector")
@icontract.ensure(lambda result, y: _binary_codes_valid(result, y), "encoded targets must match LabelEncoder.fit_transform(y)")
def gpc_binary_fit_encoded_targets(
    y: NDArray[np.object_],
) -> NDArray[np.int64]:
    """Compute _BinaryGaussianProcessClassifierLaplace.fit's LabelEncoder-style target codes."""
    _, inverse = np.unique(np.asarray(y), return_inverse=True)
    return np.asarray(inverse, dtype=np.int64)


@register_atom(witness_gpc_binary_fit_require_binary_classes)
@icontract.require(lambda classes: _nonempty_1d(classes), "classes must be a nonempty 1D class vector")
@icontract.require(lambda class_name: _nonempty_string(class_name), "class_name must be a nonempty string")
@icontract.ensure(
    lambda result, classes: _nonnegative_int(result) and result == int(np.asarray(classes).shape[0]),
    "validated class count must preserve the number of observed classes",
)
def gpc_binary_fit_require_binary_classes(
    classes: NDArray[np.object_],
    class_name: str,
) -> int:
    """Apply _BinaryGaussianProcessClassifierLaplace.fit's binary-class guard."""
    values = np.asarray(classes)
    n_classes = int(values.shape[0])
    if n_classes > 2:
        raise ValueError(
            "%s supports only binary classification. y contains classes %s"
            % (class_name, values)
        )
    if n_classes == 1:
        raise ValueError(
            "{0:s} requires 2 classes; got {1:d} class".format(
                class_name, n_classes
            )
        )
    return n_classes


@register_atom(witness_gpc_binary_fit_use_optimizer_branch)
@icontract.require(
    lambda optimizer_is_not_none: _bool(optimizer_is_not_none),
    "optimizer_is_not_none must be boolean",
)
@icontract.require(
    lambda kernel_n_dims: _nonnegative_int(kernel_n_dims),
    "kernel_n_dims must be a nonnegative integer",
)
@icontract.ensure(lambda result: _bool(result), "optimizer-branch predicate must be boolean")
def gpc_binary_fit_use_optimizer_branch(
    optimizer_is_not_none: bool,
    kernel_n_dims: int,
) -> bool:
    """Decide whether _BinaryGaussianProcessClassifierLaplace.fit enters optimizer selection."""
    return bool(optimizer_is_not_none and kernel_n_dims > 0)
