"""Ghost witnesses for sklearn one-vs-rest postprocessing helpers."""

from __future__ import annotations

from scipy.sparse import csc_matrix

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_binary_predict_threshold(
    *,
    estimator_has_decision_function: bool,
    estimator_is_classifier: bool = True,
) -> AbstractArray:
    """Describe sklearn's binary OvR prediction threshold scalar."""
    del estimator_has_decision_function, estimator_is_classifier
    return AbstractArray(shape=(), dtype="float64")


def witness_one_vs_rest_multilabel_indicator_csc(
    responses: AbstractArray,
    *,
    threshold: float = 0.0,
) -> csc_matrix:
    """Describe the CSC indicator matrix built from multilabel OvR responses."""
    del threshold
    if len(responses.shape) != 2:
        raise ValueError("responses must be 2D")
    n_samples = int(responses.shape[0])
    n_classes = int(responses.shape[1])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("responses must be nonempty")
    return csc_matrix((n_samples, n_classes), dtype=int)


def witness_one_vs_rest_positive_probability_matrix(
    positive_class_probabilities: AbstractArray,
) -> AbstractArray:
    """Describe sample-by-class positive probabilities from per-estimator outputs."""
    if len(positive_class_probabilities.shape) != 2:
        raise ValueError("positive_class_probabilities must be 2D")
    n_classes = int(positive_class_probabilities.shape[0])
    n_samples = int(positive_class_probabilities.shape[1])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("positive_class_probabilities must be nonempty")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_one_vs_rest_binary_probability_matrix(
    probabilities: AbstractArray,
) -> AbstractArray:
    """Describe the two-column binary probability matrix for a single OvR estimator."""
    if len(probabilities.shape) != 2:
        raise ValueError("probabilities must be 2D")
    n_samples = int(probabilities.shape[0])
    n_columns = int(probabilities.shape[1])
    if n_samples < 1 or n_columns != 1:
        raise ValueError("probabilities must have shape (n_samples, 1)")
    return AbstractArray(shape=(n_samples, 2), dtype="float64")


def witness_one_vs_rest_normalized_probability_matrix(
    probabilities: AbstractArray,
) -> AbstractArray:
    """Describe row-normalized multiclass OvR probabilities."""
    if len(probabilities.shape) != 2:
        raise ValueError("probabilities must be 2D")
    n_samples = int(probabilities.shape[0])
    n_classes = int(probabilities.shape[1])
    if n_samples < 1 or n_classes < 1:
        raise ValueError("probabilities must be nonempty")
    return AbstractArray(shape=(n_samples, n_classes), dtype="float64")


def witness_one_vs_rest_decision_output(
    decision_outputs: AbstractArray,
) -> AbstractArray:
    """Describe sklearn's OvR decision output shape rule."""
    if len(decision_outputs.shape) != 2:
        raise ValueError("decision_outputs must be 2D")
    n_estimators = int(decision_outputs.shape[0])
    n_samples = int(decision_outputs.shape[1])
    if n_estimators < 1 or n_samples < 1:
        raise ValueError("decision_outputs must be nonempty")
    if n_estimators == 1:
        return AbstractArray(shape=(n_samples,), dtype="float64")
    return AbstractArray(shape=(n_samples, n_estimators), dtype="float64")
