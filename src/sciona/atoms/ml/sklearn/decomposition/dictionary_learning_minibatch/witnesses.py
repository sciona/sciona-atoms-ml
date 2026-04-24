"""Ghost witnesses for MiniBatch dictionary-learning helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_dictionary_learning_minibatch_component_count(
    n_components: int | None,
    n_features: int,
) -> int:
    """Describe the resolved component count after sklearn defaulting."""
    del n_components
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return 1


def witness_dictionary_learning_minibatch_fit_algorithm(
    fit_algorithm: str,
    positive_code: bool,
) -> str:
    """Describe the prefixed sparse-coding algorithm label."""
    del positive_code
    if fit_algorithm not in {"lars", "cd"}:
        raise ValueError("fit_algorithm must be lars or cd")
    return "lasso_lars"


def witness_dictionary_learning_minibatch_batch_size(batch_size: int, n_samples: int) -> int:
    """Describe the minibatch size after clamping to the sample count."""
    if batch_size < 1 or n_samples < 1:
        raise ValueError("batch_size and n_samples must be positive")
    return 1


def witness_dictionary_learning_minibatch_stats_decay(batch_size: int, step: int) -> float:
    """Describe the inner-statistics decay factor for one minibatch step."""
    if batch_size < 1 or step < 0:
        raise ValueError("batch_size must be positive and step must be nonnegative")
    return 0.0


def witness_dictionary_learning_minibatch_inner_stats(
    A: AbstractArray,
    B: AbstractArray,
    X_batch: AbstractArray,
    code: AbstractArray,
    *,
    batch_size: int,
    step: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe the updated sufficient statistics after one minibatch."""
    n_components_a0, n_components_a1 = _check_matrix(A, "A")
    n_features_b0, n_components_b1 = _check_matrix(B, "B")
    rows_x, cols_x = _check_matrix(X_batch, "X_batch")
    rows_c, cols_c = _check_matrix(code, "code")
    if n_components_a0 != n_components_a1 or n_components_a0 != cols_c:
        raise ValueError("A must be square and match the code width")
    if n_features_b0 != cols_x or n_components_b1 != cols_c:
        raise ValueError("B must match the feature count and code width")
    if rows_x != rows_c:
        raise ValueError("X_batch and code must have the same sample count")
    if batch_size < 1 or step < 0:
        raise ValueError("batch_size must be positive and step must be nonnegative")
    return (
        AbstractArray(shape=(n_components_a0, n_components_a1), dtype="float64"),
        AbstractArray(shape=(n_features_b0, n_components_b1), dtype="float64"),
    )


def witness_dictionary_learning_minibatch_monitoring_started(
    step: int,
    n_samples: int,
    batch_size: int,
) -> bool:
    """Describe whether sklearn starts convergence monitoring at this step."""
    if step < 0 or n_samples < 1 or batch_size < 1:
        raise ValueError("step must be nonnegative and sizes must be positive")
    return False


def witness_dictionary_learning_minibatch_ewa_cost(
    previous_ewa_cost: float | None,
    batch_cost: float,
    batch_size: int,
    n_samples: int,
) -> float:
    """Describe the updated exponentially weighted average cost."""
    del previous_ewa_cost, batch_cost
    if batch_size < 1 or n_samples < 1:
        raise ValueError("batch_size and n_samples must be positive")
    return 0.0


def witness_dictionary_learning_minibatch_dictionary_change_converged(
    new_dict: AbstractArray,
    old_dict: AbstractArray,
    *,
    n_components: int,
    tol: float,
) -> bool:
    """Describe the dictionary-change stopping predicate."""
    if _check_matrix(new_dict, "new_dict") != _check_matrix(old_dict, "old_dict"):
        raise ValueError("new_dict and old_dict must match")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if tol < 0.0:
        raise ValueError("tol must be nonnegative")
    return False


def witness_dictionary_learning_minibatch_improvement_state(
    ewa_cost: float,
    ewa_cost_min: float | None,
    no_improvement: int,
    max_no_improvement: int | None,
) -> tuple[float, int, bool]:
    """Describe the smoothed-cost improvement tracker update."""
    del ewa_cost, ewa_cost_min, no_improvement, max_no_improvement
    return 0.0, 0, False
