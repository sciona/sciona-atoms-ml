"""Ghost witnesses for sklearn one-vs-rest target encoding helpers."""

from __future__ import annotations

from scipy.sparse import csc_matrix

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_fit_classes(y: AbstractArray) -> AbstractArray:
    """Describe the class vector discovered by sklearn's sparse LabelBinarizer."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if int(y.shape[0]) < 1:
        raise ValueError("y must be nonempty")
    return AbstractArray(shape=(None,), dtype="float64")


def witness_one_vs_rest_fit_target_indicator_csc(y: AbstractArray) -> csc_matrix:
    """Describe the CSC target indicator matrix fitted from y."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    n_samples = int(y.shape[0])
    if n_samples < 1:
        raise ValueError("y must be nonempty")
    return csc_matrix((n_samples, 1), dtype=int)


def witness_one_vs_rest_partial_fit_unknown_classes(
    y: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe the sorted unique labels in y that are absent from classes."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty 1D vector")
    return AbstractArray(shape=(None,), dtype="float64")


def witness_one_vs_rest_partial_fit_target_indicator_csc(
    y: AbstractArray,
    classes: AbstractArray,
) -> csc_matrix:
    """Describe the CSC target indicator matrix from known partial-fit classes."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty 1D vector")
    n_samples = int(y.shape[0])
    if n_samples < 1:
        raise ValueError("y must be nonempty")
    return csc_matrix((n_samples, int(classes.shape[0])), dtype=int)


def witness_one_vs_rest_target_columns_dense(indicator: AbstractArray) -> AbstractArray:
    """Describe the output-by-sample dense target column stack used by sklearn's OvR worker loop."""
    if len(indicator.shape) != 2:
        raise ValueError("indicator must be 2D")
    n_samples = int(indicator.shape[0])
    n_outputs = int(indicator.shape[1])
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("indicator must be nonempty")
    return AbstractArray(shape=(n_outputs, n_samples), dtype="float64")
