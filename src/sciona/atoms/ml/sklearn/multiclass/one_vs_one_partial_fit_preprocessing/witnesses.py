"""Ghost witnesses for one-vs-one partial-fit preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_class_vector(classes: AbstractArray) -> int:
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    n_classes = int(classes.shape[0])
    if n_classes < 2:
        raise ValueError("classes must contain at least two labels")
    return n_classes


def _check_target_vector(y: AbstractArray) -> int:
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    n_samples = int(y.shape[0])
    if n_samples < 1:
        raise ValueError("y must be nonempty")
    return n_samples


def witness_one_vs_one_partial_fit_estimator_count(
    classes: AbstractArray,
) -> int:
    """Describe the number of one-vs-one estimators for a class vector."""
    _check_class_vector(classes)
    return 1


def witness_one_vs_one_partial_fit_unknown_classes(
    y: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe labels present in y but absent from the known class vector."""
    _check_target_vector(y)
    _check_class_vector(classes)
    return AbstractArray(shape=(int(y.shape[0]),), dtype="float64")


def witness_one_vs_one_partial_fit_pair_mask(
    y: AbstractArray,
    class_i: float,
    class_j: float,
) -> AbstractArray:
    """Describe the boolean mask selecting one one-vs-one class pair."""
    del class_i, class_j
    n_samples = _check_target_vector(y)
    return AbstractArray(shape=(n_samples,), dtype="bool")


def witness_one_vs_one_partial_fit_subset_indices(
    pair_mask: AbstractArray,
) -> AbstractArray:
    """Describe the selected sample indices for one one-vs-one class pair."""
    if len(pair_mask.shape) != 1:
        raise ValueError("pair_mask must be 1D")
    n_samples = int(pair_mask.shape[0])
    if n_samples < 1:
        raise ValueError("pair_mask must be nonempty")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0.0)


def witness_one_vs_one_partial_fit_binary_targets(
    y: AbstractArray,
    class_i: float,
    class_j: float,
) -> AbstractArray:
    """Describe filtered 0/1 targets for one one-vs-one class pair."""
    del class_i, class_j
    n_samples = _check_target_vector(y)
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0.0)
