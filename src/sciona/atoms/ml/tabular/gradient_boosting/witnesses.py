"""Ghost witnesses for tabular gradient-boosting helper atoms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sciona.ghost.abstract import AbstractArray, AbstractScalar


def _check_1d(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    return int(values.shape[0])


def _check_2d(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    return int(values.shape[0]), int(values.shape[1])


def _check_1d_or_2d(values: AbstractArray, name: str) -> tuple[int, ...]:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    return tuple(int(dim) for dim in values.shape)


def _check_same_length(expected: int, actual: AbstractArray, name: str) -> None:
    if _check_1d(actual, name) != expected:
        raise ValueError(f"{name} must match the first input length")


def _valid_group_agg(agg_fn: str) -> bool:
    return agg_fn in {"sum", "mean", "var", "std", "min", "max", "count", "first", "last"}


def _valid_rolling_aggs(agg_fns: Sequence[str]) -> bool:
    return bool(len(agg_fns) > 0 and all(agg_fn in {"mean", "std", "min", "max", "sum"} for agg_fn in agg_fns))


def witness_group_aggregate(values: AbstractArray, groups: AbstractArray, agg_fn: str) -> AbstractArray:
    """Describe group aggregation output as a finite vector with unknown unique-group count."""
    row_count = _check_1d(values, "values")
    _check_same_length(row_count, groups, "groups")
    if not _valid_group_agg(agg_fn):
        raise ValueError("unsupported group aggregation")
    return AbstractArray(shape=(row_count,), dtype="float64")


def witness_aggregate_child_table(
    child_values: AbstractArray,
    parent_keys: AbstractArray,
    child_keys: AbstractArray,
    agg_fns: Sequence[str],
    fill_value: float = 0.0,
) -> AbstractArray:
    """Describe parent-aligned child aggregate columns."""
    del fill_value
    child_count = _check_1d(child_values, "child_values")
    _check_same_length(child_count, child_keys, "child_keys")
    parent_count = _check_1d(parent_keys, "parent_keys")
    if not agg_fns or not all(_valid_group_agg(agg_fn) for agg_fn in agg_fns):
        raise ValueError("unsupported child aggregation")
    return AbstractArray(shape=(parent_count, len(agg_fns)), dtype="float64")


def witness_temporal_difference(values: AbstractArray, entity_ids: AbstractArray, sort_keys: AbstractArray) -> AbstractArray:
    """Describe within-entity temporal differencing."""
    row_count = _check_1d(values, "values")
    _check_same_length(row_count, entity_ids, "entity_ids")
    _check_same_length(row_count, sort_keys, "sort_keys")
    return AbstractArray(shape=(row_count,), dtype="float64")


def witness_rolling_statistics(values: AbstractArray, window_size: int, agg_fns: Sequence[str]) -> AbstractArray:
    """Describe trailing rolling-statistic feature columns."""
    row_count = _check_1d(values, "values")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not _valid_rolling_aggs(agg_fns):
        raise ValueError("unsupported rolling aggregation")
    return AbstractArray(shape=(row_count, len(agg_fns)), dtype="float64")


def witness_time_decay_aggregate(
    values: AbstractArray,
    timestamps: AbstractArray,
    groups: AbstractArray,
    decay_rate: float,
) -> AbstractArray:
    """Describe per-group exponentially decayed aggregation."""
    row_count = _check_1d(values, "values")
    _check_same_length(row_count, timestamps, "timestamps")
    _check_same_length(row_count, groups, "groups")
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive")
    return AbstractArray(shape=(row_count,), dtype="float64", min_val=0.0)


def witness_pairwise_products(features: AbstractArray) -> AbstractArray:
    """Describe unordered pairwise product feature expansion."""
    rows, columns = _check_2d(features, "features")
    return AbstractArray(shape=(rows, columns * (columns - 1) // 2), dtype="float64")


def witness_pairwise_ratios(features: AbstractArray, epsilon: float = 1e-12) -> AbstractArray:
    """Describe unordered pairwise ratio feature expansion."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return witness_pairwise_products(features)


def witness_missing_indicator_and_impute(
    values: AbstractArray,
    fill_value: float = 0.0,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe NaN imputation paired with binary missingness indicators."""
    del fill_value
    shape = _check_1d_or_2d(values, "values")
    return AbstractArray(shape=shape, dtype="float64"), AbstractArray(shape=shape, dtype="float64", min_val=0.0, max_val=1.0)


def witness_rank_transform(values: AbstractArray, method: str = "average") -> AbstractArray:
    """Describe rank transformation preserving vector shape."""
    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError("unsupported rank method")
    return AbstractArray(shape=(_check_1d(values, "values"),), dtype="float64", min_val=1.0)


def witness_frequency_encode_fit(categories: AbstractArray) -> AbstractScalar:
    """Describe learning a category-frequency mapping."""
    _check_1d(categories, "categories")
    return AbstractScalar(dtype="mapping", min_val=0.0)


def witness_frequency_encode(
    categories: AbstractArray,
    frequency_map: Mapping[object, float],
    unknown_value: float = 0.0,
) -> AbstractArray:
    """Describe category-to-frequency transformation."""
    del frequency_map
    if unknown_value < 0:
        raise ValueError("unknown_value must be nonnegative")
    return AbstractArray(shape=(_check_1d(categories, "categories"),), dtype="float64", min_val=0.0)


def witness_target_encode(
    categories: AbstractArray,
    targets: AbstractArray,
    smoothing: float = 1.0,
    prior: float | None = None,
) -> AbstractScalar:
    """Describe learning a smoothed target-mean mapping."""
    del prior
    row_count = _check_1d(categories, "categories")
    _check_same_length(row_count, targets, "targets")
    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative")
    return AbstractScalar(dtype="mapping")


def witness_tweedie_gradient(
    predictions: AbstractArray,
    targets: AbstractArray,
    power: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe Tweedie gradient and hessian vectors."""
    row_count = _check_1d(predictions, "predictions")
    _check_same_length(row_count, targets, "targets")
    if not 1.0 < power < 2.0:
        raise ValueError("power must be in (1, 2)")
    return AbstractArray(shape=(row_count,), dtype="float64"), AbstractArray(shape=(row_count,), dtype="float64", min_val=0.0)


def witness_log_cosh_gradient(
    targets: AbstractArray,
    predictions: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe log-cosh gradient and hessian vectors."""
    row_count = _check_1d(targets, "targets")
    _check_same_length(row_count, predictions, "predictions")
    return AbstractArray(shape=(row_count,), dtype="float64"), AbstractArray(shape=(row_count,), dtype="float64", min_val=0.0)


def witness_null_importance_p_values(
    actual_importances: AbstractArray,
    null_importances_matrix: AbstractArray,
) -> AbstractArray:
    """Describe per-feature null-importance p-values."""
    feature_count = _check_1d(actual_importances, "actual_importances")
    _, null_features = _check_2d(null_importances_matrix, "null_importances_matrix")
    if null_features != feature_count:
        raise ValueError("null matrix columns must match actual importances")
    return AbstractArray(shape=(feature_count,), dtype="float64", min_val=0.0, max_val=1.0)


def witness_extract_pseudo_labels(
    test_predictions: AbstractArray,
    upper_threshold: float,
    lower_threshold: float,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe high-confidence positive and negative pseudo-label index vectors."""
    row_count = _check_1d(test_predictions, "test_predictions")
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")
    return AbstractArray(shape=(row_count,), dtype="int64", min_val=0.0), AbstractArray(shape=(row_count,), dtype="int64", min_val=0.0)

