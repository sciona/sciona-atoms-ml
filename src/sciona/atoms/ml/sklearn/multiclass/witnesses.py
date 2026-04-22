"""Ghost witnesses for sklearn multiclass meta-estimator helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_class_vector(classes: AbstractArray) -> int:
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    n_classes = int(classes.shape[0])
    if n_classes < 1:
        raise ValueError("classes must be nonempty")
    return n_classes


def _check_response_matrix(responses: AbstractArray, classes: AbstractArray) -> tuple[int, int]:
    if len(responses.shape) != 2:
        raise ValueError("responses must be 2D")
    n_samples, n_classes = int(responses.shape[0]), int(responses.shape[1])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("responses must be nonempty")
    if n_classes != _check_class_vector(classes):
        raise ValueError("responses and classes must have matching class counts")
    return n_samples, n_classes


def witness_one_vs_rest_multiclass_labels(
    responses: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe one OvR multiclass label per sample."""
    n_samples, _n_classes = _check_response_matrix(responses, classes)
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_one_vs_rest_binary_indicator(
    responses: AbstractArray,
    *,
    threshold: float = 0.0,
) -> AbstractArray:
    """Describe thresholded OvR binary decisions for every sample and class."""
    del threshold
    if len(responses.shape) != 2:
        raise ValueError("responses must be 2D")
    n_samples, n_classes = int(responses.shape[0]), int(responses.shape[1])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("responses must be nonempty")
    return AbstractArray(shape=(n_samples, n_classes), dtype="bool")


def witness_one_vs_one_decision_scores(
    predictions: AbstractArray,
    confidences: AbstractArray,
    *,
    n_classes: int,
) -> AbstractArray:
    """Describe OvO vote scores with confidence-based tie breaking."""
    if len(predictions.shape) != 2 or len(confidences.shape) != 2:
        raise ValueError("predictions and confidences must be 2D")
    if predictions.shape != confidences.shape:
        raise ValueError("predictions and confidences must have the same shape")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    n_samples = int(predictions.shape[0])
    n_classifiers = int(predictions.shape[1])
    if n_samples < 1 or n_classifiers != n_classes * (n_classes - 1) // 2:
        raise ValueError("classifier count must match n_classes choose two")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_one_vs_one_class_pairs(n_classes: int) -> AbstractArray:
    """Describe the ordered class-index pairs used by OvO fitting."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    return AbstractArray(shape=(n_classes * (n_classes - 1) // 2, 2), dtype="int64")


def witness_output_code_book(
    n_classes: int,
    *,
    code_size: float = 1.5,
    random_state: int = 0,
    estimator_has_decision_function: bool = True,
) -> AbstractArray:
    """Describe a deterministic error-correcting output-code book."""
    del random_state, estimator_has_decision_function
    if n_classes < 1 or code_size <= 0.0 or int(n_classes * code_size) < 1:
        raise ValueError("code book dimensions must be positive")
    return AbstractArray(shape=(n_classes, int(n_classes * code_size)), dtype="float64")


def witness_output_code_decode(
    responses: AbstractArray,
    code_book: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe nearest-code-row decoding for each sample."""
    if len(responses.shape) != 2 or len(code_book.shape) != 2:
        raise ValueError("responses and code_book must be 2D")
    n_samples, n_estimators = int(responses.shape[0]), int(responses.shape[1])
    n_classes, code_width = int(code_book.shape[0]), int(code_book.shape[1])
    if n_samples < 1 or n_estimators < 1 or n_estimators != code_width:
        raise ValueError("response width must match code-book width")
    if _check_class_vector(classes) != n_classes:
        raise ValueError("code_book and classes must have matching class counts")
    return AbstractArray(shape=(n_samples,), dtype="float64")
