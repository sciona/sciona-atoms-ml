"""Ghost witnesses for sklearn output-code matrix preparation helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_output_code_fit_classes(y: AbstractArray) -> AbstractArray:
    """Describe sklearn's sorted unique class vector for a 1D target input."""
    if len(y.shape) != 1 or int(y.shape[0]) < 1:
        raise ValueError("y must be a nonempty 1D vector")
    return AbstractArray(shape=(None,), dtype="float64")


def witness_output_code_target_matrix(
    y: AbstractArray,
    classes: AbstractArray,
    code_book: AbstractArray,
) -> AbstractArray:
    """Describe the sample-by-estimator integer code matrix used for binary fits."""
    if len(y.shape) != 1 or int(y.shape[0]) < 1:
        raise ValueError("y must be a nonempty 1D vector")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty 1D vector")
    if len(code_book.shape) != 2 or int(code_book.shape[0]) < 1 or int(code_book.shape[1]) < 1:
        raise ValueError("code_book must be a nonempty 2D matrix")
    return AbstractArray(shape=(int(y.shape[0]), int(code_book.shape[1])), dtype="int64")


def witness_output_code_response_matrix(estimator_predictions: AbstractArray) -> AbstractArray:
    """Describe sklearn's sample-major decode matrix built from estimator-major responses."""
    if len(estimator_predictions.shape) != 2:
        raise ValueError("estimator_predictions must be 2D")
    n_estimators = int(estimator_predictions.shape[0])
    n_samples = int(estimator_predictions.shape[1])
    if n_estimators < 1 or n_samples < 1:
        raise ValueError("estimator_predictions must be nonempty")
    return AbstractArray(shape=(n_samples, n_estimators), dtype="float64")
