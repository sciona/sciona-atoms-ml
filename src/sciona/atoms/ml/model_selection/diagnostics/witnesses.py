"""Ghost witnesses for dataset diagnostic atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_compute_condition_number(X: AbstractArray) -> float:
    """Condition number is a non-negative scalar >= 1."""
    return 1.0


def witness_compute_n_p_ratio(X: AbstractArray) -> float:
    """Sample-to-feature ratio is a positive scalar."""
    return 1.0


def witness_compute_mutual_incoherence(X: AbstractArray) -> float:
    """Mutual incoherence is a non-negative scalar."""
    return 0.0


def witness_check_lasso_sample_complexity(
    X: AbstractArray, sparsity_estimate: int
) -> bool:
    """Boolean check on sample complexity."""
    return True


def witness_compute_excess_kurtosis(y: AbstractArray) -> float:
    """Excess kurtosis is a scalar (can be negative)."""
    return 0.0


def witness_compute_residual_kurtosis(X: AbstractArray, y: AbstractArray) -> float:
    """Residual kurtosis is a scalar."""
    return 0.0


def witness_compute_dispersion_index(y: AbstractArray) -> float:
    """Dispersion index is a non-negative scalar."""
    return 1.0


def witness_estimate_tweedie_power(y: AbstractArray) -> float:
    """Tweedie power parameter is a scalar in [0, 3]."""
    return 1.0


def witness_estimate_noise_level(X: AbstractArray, y: AbstractArray) -> float:
    """Noise level is a non-negative scalar."""
    return 0.0


def witness_count_categorical_features(X: AbstractArray) -> int:
    """Count is a non-negative integer."""
    return 0


def witness_compute_skewness(x: AbstractArray) -> float:
    """Skewness is a scalar (can be negative)."""
    return 0.0


def witness_compute_vif(X: AbstractArray) -> AbstractArray:
    """VIF is a 1D array with one value per feature, all >= 1."""
    return AbstractArray(shape=(X.shape[1] if hasattr(X, "shape") else 1,), dtype="float64")


def witness_test_normality(x: AbstractArray) -> float:
    """Shapiro-Wilk p-value in [0, 1]."""
    return 0.5


def witness_is_sparse(X: object) -> bool:
    """Boolean sparsity check."""
    return False


def witness_compute_explained_variance_ratio(
    X: AbstractArray, n_components: int
) -> float:
    """Explained variance ratio in [0, 1]."""
    return 0.95


def witness_check_time_series_index(X: AbstractArray) -> bool:
    """Boolean monotonicity check."""
    return False
