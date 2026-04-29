"""Ghost witnesses for one-vs-rest prediction-output helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_predict_maxima_init(n_samples: int) -> AbstractArray:
    """Describe the running maxima vector for multiclass OvR predict."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_one_vs_rest_predict_argmaxima_init(n_samples: int) -> AbstractArray:
    """Describe the running winning-class index vector for multiclass OvR predict."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="int64")


def witness_one_vs_rest_predict_multiclass_update(
    maxima: AbstractArray,
    argmaxima: AbstractArray,
    pred: AbstractArray,
    *,
    class_index: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe one running-max update step for multiclass OvR predict."""
    if len(maxima.shape) != 1 or len(argmaxima.shape) != 1 or len(pred.shape) != 1:
        raise ValueError("maxima, argmaxima, and pred must be 1D")
    n_samples = int(pred.shape[0])
    if n_samples < 1 or int(maxima.shape[0]) != n_samples or int(argmaxima.shape[0]) != n_samples:
        raise ValueError("maxima, argmaxima, and pred must have the same nonzero length")
    if class_index < 0:
        raise ValueError("class_index must be nonnegative")
    return (
        AbstractArray(shape=(n_samples,), dtype="float64"),
        AbstractArray(shape=(n_samples,), dtype="int64"),
    )


def witness_one_vs_rest_predict_labels_from_argmaxima(
    argmaxima: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe decoding multiclass OvR winning-class indices into labels."""
    if len(argmaxima.shape) != 1 or len(classes.shape) != 1:
        raise ValueError("argmaxima and classes must be 1D")
    n_samples = int(argmaxima.shape[0])
    n_classes = int(classes.shape[0])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("argmaxima and classes must be nonempty")
    return AbstractArray(shape=(n_samples,), dtype="float64")
