"""Ghost witnesses for sklearn SGDOneClassSVM fit shell atoms."""

from __future__ import annotations

import numpy as np


def witness_sgd_one_class_target(X: object) -> np.ndarray:
    """Describe the artificial all-one target vector used by SGDOneClassSVM."""
    array = np.asarray(X)
    return np.ones(array.shape[0], dtype=array.dtype, order="C")


def witness_sgd_one_class_fixed_solver_context(one_class: int = 1) -> tuple[int, int, int]:
    """Describe SGDOneClassSVM's fixed one-class and class-weight flags."""
    del one_class
    return (1, 1, 1)


def witness_sgd_one_class_validation_sample_mask(sample_weight: object) -> np.ndarray:
    """Describe the positive-weight validation sample mask."""
    return np.asarray(sample_weight) > 0


def witness_sgd_one_class_intercept_from_offset(offset: object) -> np.ndarray:
    """Describe the solver intercept initialized from a one-class offset."""
    return 1 - np.atleast_1d(offset)


def witness_sgd_one_class_offset_from_intercept(intercept: object) -> np.ndarray:
    """Describe the public one-class offset written from a solver intercept."""
    return 1 - np.atleast_1d(intercept)


def witness_sgd_one_class_time_step_after_fit(t_before: float, n_iter: int, n_samples: int) -> float:
    """Describe the t_ update after one fit pass."""
    return t_before + n_iter * n_samples


def witness_sgd_one_class_average_active(average: object, t_after: float) -> bool:
    """Describe whether averaged solver state should replace standard state."""
    return bool(average > 0 and average <= t_after - 1.0)


def witness_sgd_one_class_average_buffers(n_features: int, dtype: object) -> dict[str, np.ndarray]:
    """Describe the average coefficient and intercept buffers."""
    return {
        "average_coef": np.zeros(n_features, dtype=np.dtype(dtype), order="C"),
        "average_intercept": np.zeros(1, dtype=np.dtype(dtype), order="C"),
    }


def witness_sgd_one_class_parameter_allocation_payload(
    n_features: int,
    input_dtype: object,
    coef_init: object,
    offset_init: object,
) -> dict[str, object]:
    """Describe the one-class _allocate_parameter_mem payload."""
    return {
        "n_classes": 1,
        "n_features": n_features,
        "input_dtype": np.dtype(input_dtype),
        "coef_init": coef_init,
        "intercept_init": offset_init,
        "one_class": 1,
    }


def witness_sgd_one_class_fit_one_class_payload(
    X: object,
    alpha: float,
    C: float,
    learning_rate: str,
    sample_weight: object,
    max_iter: int,
) -> dict[str, object]:
    """Describe the delegated _fit_one_class payload."""
    return {
        "X": X,
        "alpha": alpha,
        "C": C,
        "learning_rate": learning_rate,
        "sample_weight": sample_weight,
        "max_iter": max_iter,
    }


def witness_sgd_one_class_partial_fit_result(estimator: object) -> object:
    """Describe the estimator returned by _partial_fit."""
    return estimator
