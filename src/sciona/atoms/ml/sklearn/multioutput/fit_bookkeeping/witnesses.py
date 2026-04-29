"""Ghost witnesses for sklearn multioutput fit bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_multioutput_fit_require_base_fit_method(estimator_has_fit: bool) -> bool:
    """Describe sklearn's base-estimator fit-method requirement."""
    del estimator_has_fit
    return bool


def witness_multioutput_fit_require_2d_targets(y: AbstractArray) -> AbstractArray:
    """Describe sklearn's 2D target requirement for multioutput fit."""
    if len(y.shape) < 1:
        raise ValueError("y must have at least one dimension")
    n_samples = int(y.shape[0])
    if n_samples < 1:
        raise ValueError("y must be nonempty")
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    n_outputs = int(y.shape[1])
    if n_outputs < 1:
        raise ValueError("y must have at least one output column")
    return AbstractArray(shape=(n_samples, n_outputs), dtype="float64")


def witness_multioutput_fit_output_count(y: AbstractArray) -> int:
    """Describe the number of target columns used by the multioutput fit loop."""
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    n_samples = int(y.shape[0])
    n_outputs = int(y.shape[1])
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("y must be nonempty")
    return int


def witness_multioutput_fit_target_column(
    y: AbstractArray,
    output_idx: int,
) -> AbstractArray:
    """Describe one per-output target column passed to a base estimator fit call."""
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    n_samples = int(y.shape[0])
    n_outputs = int(y.shape[1])
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("y must be nonempty")
    if output_idx < 0 or output_idx >= n_outputs:
        raise ValueError("output_idx must select an existing output column")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_multioutput_fit_require_sample_weight_support(
    *,
    sample_weight_provided: bool,
    estimator_supports_sample_weight: bool,
) -> bool:
    """Describe sklearn's sample-weight support guard for multioutput fit."""
    del sample_weight_provided, estimator_supports_sample_weight
    return bool
