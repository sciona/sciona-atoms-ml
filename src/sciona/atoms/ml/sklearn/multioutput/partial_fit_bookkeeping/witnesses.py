"""Ghost witnesses for multioutput partial-fit bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_multioutput_partial_fit_first_call(has_estimators: bool) -> AbstractArray:
    """Describe the Boolean first-call flag for multioutput partial_fit state."""
    del has_estimators
    return AbstractArray(shape=(), dtype="bool")


def witness_multioutput_partial_fit_use_base_estimator(first_time: bool) -> AbstractArray:
    """Describe whether multioutput partial_fit should use the base estimator template."""
    del first_time
    return AbstractArray(shape=(), dtype="bool")


def witness_multioutput_partial_fit_class_vector(
    classes_by_output: tuple[AbstractArray, ...] | None,
    output_idx: int,
) -> AbstractArray:
    """Describe the per-output classes vector routed into one partial_fit worker."""
    del output_idx
    if classes_by_output is None:
        return AbstractArray(shape=(), dtype="object")
    if not isinstance(classes_by_output, tuple) or len(classes_by_output) < 1:
        raise ValueError("classes_by_output must be None or a nonempty tuple")
    target = classes_by_output[0]
    if len(target.shape) != 1 or int(target.shape[0]) < 1:
        raise ValueError("class vector must be nonempty and 1D")
    return AbstractArray(shape=(int(target.shape[0]),), dtype="float64")
