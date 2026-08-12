"""Ghost witnesses for generic supervised-tabular workflow atoms."""

from __future__ import annotations

from typing import Any


def witness_stratified_tabular_split(
    dataset: Any,
) -> tuple[Any, Any, Any, Any]:
    """Describe a deterministic train/test partition of a labeled table."""
    return dataset, dataset, dataset, dataset


def witness_fit_prior_probability(y_train: Any) -> float:
    """Describe a scalar empirical positive-class probability."""
    del y_train
    return 0.5


def witness_predict_prior_probabilities(
    class_probability: float,
    X_test: Any,
    y_test: Any,
) -> tuple[Any, Any]:
    """Describe aligned probability and target vectors."""
    del class_probability, X_test
    return y_test, y_test


def witness_fit_one_hot_logistic(X_train: Any, y_train: Any) -> Any:
    """Describe a fitted mixed-type logistic pipeline."""
    del y_train
    return X_train


def witness_fit_cross_validated_logistic(X_train: Any, y_train: Any) -> Any:
    """Describe a fitted cross-validated logistic pipeline."""
    del y_train
    return X_train


def witness_predict_binary_probabilities(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> tuple[Any, Any]:
    """Describe aligned positive-class probabilities and targets."""
    del model, X_test
    return y_test, y_test
