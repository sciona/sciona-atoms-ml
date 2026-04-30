"""Pure NumPy tabular helpers used around gradient-boosting models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_aggregate_child_table,
    witness_extract_pseudo_labels,
    witness_frequency_encode,
    witness_frequency_encode_fit,
    witness_group_aggregate,
    witness_log_cosh_gradient,
    witness_missing_indicator_and_impute,
    witness_null_importance_p_values,
    witness_pairwise_products,
    witness_pairwise_ratios,
    witness_rank_transform,
    witness_rolling_statistics,
    witness_target_encode,
    witness_temporal_difference,
    witness_time_decay_aggregate,
    witness_tweedie_gradient,
)

GROUP_AGGREGATIONS = frozenset({"sum", "mean", "var", "std", "min", "max", "count", "first", "last"})
ROLLING_AGGREGATIONS = frozenset({"mean", "std", "min", "max", "sum"})
RANK_METHODS = frozenset({"average", "min", "max", "dense", "ordinal"})


def _is_1d(values: NDArray[np.float64] | NDArray[np.object_]) -> bool:
    return bool(np.asarray(values).ndim == 1)


def _is_2d(values: NDArray[np.float64]) -> bool:
    return bool(np.asarray(values).ndim == 2)


def _is_1d_or_2d(values: NDArray[np.float64]) -> bool:
    return bool(np.asarray(values).ndim in {1, 2})


def _same_length(*arrays: NDArray[np.float64] | NDArray[np.object_]) -> bool:
    lengths = [np.asarray(array).shape[0] for array in arrays]
    return bool(len(set(lengths)) == 1)


def _finite_1d(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and np.all(np.isfinite(array)))


def _finite_1d_nonnegative(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _finite_1d_or_2d(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and np.all(np.isfinite(array)))


def _finite_2d(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and np.all(np.isfinite(array)))


def _valid_group_agg(agg_fn: str) -> bool:
    return agg_fn in GROUP_AGGREGATIONS


def _valid_rolling_aggs(agg_fns: Sequence[str]) -> bool:
    return bool(len(agg_fns) > 0 and all(agg_fn in ROLLING_AGGREGATIONS for agg_fn in agg_fns))


def _valid_group_aggs(agg_fns: Sequence[str]) -> bool:
    return bool(len(agg_fns) > 0 and all(agg_fn in GROUP_AGGREGATIONS for agg_fn in agg_fns))


def _finite_mapping_values(values: Mapping[object, float]) -> bool:
    try:
        return bool(all(np.isfinite(float(value)) and float(value) >= 0.0 for value in values.values()))
    except (TypeError, ValueError):
        return False


def _finite_encoding_values(values: Mapping[object, float]) -> bool:
    try:
        return bool(len(values) > 0 and all(np.isfinite(float(value)) for value in values.values()))
    except (TypeError, ValueError):
        return False


def _mapping_has_all_categories(categories: NDArray[np.object_], values: Mapping[object, float]) -> bool:
    return bool(set(np.asarray(categories, dtype=object).tolist()).issubset(set(values.keys())))


def _valid_thresholds(lower_threshold: float, upper_threshold: float) -> bool:
    return bool(0.0 <= float(lower_threshold) < float(upper_threshold) <= 1.0)


def _finite_scalar(value: float) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _output_shape_matches(result: NDArray[np.float64], rows: int, columns: int) -> bool:
    return bool(np.asarray(result).shape == (rows, columns))


@register_atom(witness_group_aggregate)
@icontract.require(lambda values: _finite_1d(values), "values must be a finite 1D numeric array")
@icontract.require(lambda groups: _is_1d(groups), "groups must be a 1D array")
@icontract.require(lambda values, groups: _same_length(values, groups), "values and groups must have equal length")
@icontract.require(lambda agg_fn: _valid_group_agg(agg_fn), "agg_fn must be a supported group aggregation")
@icontract.ensure(lambda result, groups: result.shape == (np.unique(groups).shape[0],), "one aggregate must be returned per unique group")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "group aggregates must be finite")
def group_aggregate(
    values: NDArray[np.float64],
    groups: NDArray[np.object_],
    agg_fn: str,
) -> NDArray[np.float64]:
    """Aggregate a numeric vector by group keys using deterministic NumPy reductions."""
    numeric_values = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups)
    unique_groups, inverse = np.unique(group_values, return_inverse=True)
    group_count = int(unique_groups.shape[0])

    if agg_fn == "count":
        return np.bincount(inverse, minlength=group_count).astype(np.float64)

    if agg_fn in {"sum", "mean", "var", "std"}:
        sums = np.bincount(inverse, weights=numeric_values, minlength=group_count)
        if agg_fn == "sum":
            return sums.astype(np.float64)
        counts = np.bincount(inverse, minlength=group_count).astype(np.float64)
        means = sums / counts
        if agg_fn == "mean":
            return means.astype(np.float64)
        centered = numeric_values - means[inverse]
        variances = np.bincount(inverse, weights=centered * centered, minlength=group_count) / counts
        if agg_fn == "var":
            return variances.astype(np.float64)
        return np.sqrt(variances).astype(np.float64)

    order = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[order]
    sorted_values = numeric_values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]

    if agg_fn == "min":
        return np.minimum.reduceat(sorted_values, starts).astype(np.float64)
    if agg_fn == "max":
        return np.maximum.reduceat(sorted_values, starts).astype(np.float64)
    if agg_fn == "first":
        return sorted_values[starts].astype(np.float64)

    ends = np.r_[starts[1:], sorted_values.shape[0]] - 1
    return sorted_values[ends].astype(np.float64)


@register_atom(witness_aggregate_child_table)
@icontract.require(lambda child_values: _finite_1d(child_values), "child_values must be a finite 1D numeric array")
@icontract.require(lambda parent_keys: _is_1d(parent_keys), "parent_keys must be a 1D array")
@icontract.require(lambda child_keys: _is_1d(child_keys), "child_keys must be a 1D array")
@icontract.require(lambda child_values, child_keys: _same_length(child_values, child_keys), "child_values and child_keys must align")
@icontract.require(lambda parent_keys: np.unique(parent_keys).shape[0] == np.asarray(parent_keys).shape[0], "parent_keys must be unique")
@icontract.require(lambda agg_fns: _valid_group_aggs(agg_fns), "agg_fns must be a non-empty supported aggregation list")
@icontract.require(lambda fill_value: np.isnan(fill_value) or _finite_scalar(fill_value), "fill_value must be finite or NaN")
@icontract.ensure(lambda result, parent_keys, agg_fns: _output_shape_matches(result, np.asarray(parent_keys).shape[0], len(agg_fns)), "output must align parent rows and aggregation columns")
def aggregate_child_table(
    child_values: NDArray[np.float64],
    parent_keys: NDArray[np.object_],
    child_keys: NDArray[np.object_],
    agg_fns: Sequence[str],
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    """Aggregate child records and align the resulting columns to parent key order."""
    parent_array = np.asarray(parent_keys)
    child_key_array = np.asarray(child_keys)
    unique_child_keys = np.unique(child_key_array)
    result = np.full((parent_array.shape[0], len(agg_fns)), float(fill_value), dtype=np.float64)

    insertion_points = np.searchsorted(unique_child_keys, parent_array)
    in_bounds = insertion_points < unique_child_keys.shape[0]
    matched = np.zeros(parent_array.shape, dtype=bool)
    matched[in_bounds] = unique_child_keys[insertion_points[in_bounds]] == parent_array[in_bounds]

    for column, agg_fn in enumerate(agg_fns):
        aggregated = group_aggregate(child_values, child_key_array, agg_fn)
        result[matched, column] = aggregated[insertion_points[matched]]
        if agg_fn == "count":
            result[~matched, column] = 0.0
    return result


@register_atom(witness_temporal_difference)
@icontract.require(lambda values: _finite_1d(values), "values must be a finite 1D numeric array")
@icontract.require(lambda entity_ids: _is_1d(entity_ids), "entity_ids must be a 1D array")
@icontract.require(lambda sort_keys: _is_1d(sort_keys), "sort_keys must be a 1D array")
@icontract.require(lambda values, entity_ids, sort_keys: _same_length(values, entity_ids, sort_keys), "values, entity_ids, and sort_keys must align")
@icontract.ensure(lambda result, values: result.shape == np.asarray(values).shape, "differences must preserve row count")
def temporal_difference(
    values: NDArray[np.float64],
    entity_ids: NDArray[np.object_],
    sort_keys: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute within-entity first differences after sorting rows by entity and time."""
    value_array = np.asarray(values, dtype=np.float64)
    entity_array = np.asarray(entity_ids)
    sort_array = np.asarray(sort_keys)

    order = np.lexsort((sort_array, entity_array))
    sorted_values = value_array[order]
    sorted_entities = entity_array[order]

    sorted_result = np.full(sorted_values.shape, np.nan, dtype=np.float64)
    same_entity = sorted_entities[1:] == sorted_entities[:-1]
    sorted_result[1:][same_entity] = np.diff(sorted_values)[same_entity]

    result = np.empty_like(sorted_result)
    result[order] = sorted_result
    return result


@register_atom(witness_rolling_statistics)
@icontract.require(lambda values: _finite_1d(values), "values must be a finite 1D numeric array")
@icontract.require(lambda window_size: window_size > 0, "window_size must be positive")
@icontract.require(lambda agg_fns: _valid_rolling_aggs(agg_fns), "agg_fns must be a non-empty supported rolling aggregation list")
@icontract.ensure(lambda result, values, agg_fns: result.shape == (np.asarray(values).shape[0], len(agg_fns)), "rolling output must preserve row count")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "rolling statistics must be finite")
def rolling_statistics(
    values: NDArray[np.float64],
    window_size: int,
    agg_fns: Sequence[str],
) -> NDArray[np.float64]:
    """Compute trailing rolling statistics using partial windows at the start."""
    value_array = np.asarray(values, dtype=np.float64)
    width = int(window_size)
    output = np.empty((value_array.shape[0], len(agg_fns)), dtype=np.float64)
    for row in range(value_array.shape[0]):
        window = value_array[max(0, row - width + 1) : row + 1]
        for column, agg_fn in enumerate(agg_fns):
            if agg_fn == "mean":
                output[row, column] = float(np.mean(window))
            elif agg_fn == "std":
                output[row, column] = float(np.std(window))
            elif agg_fn == "min":
                output[row, column] = float(np.min(window))
            elif agg_fn == "max":
                output[row, column] = float(np.max(window))
            else:
                output[row, column] = float(np.sum(window))
    return output


@register_atom(witness_time_decay_aggregate)
@icontract.require(lambda values: _finite_1d(values), "values must be a finite 1D numeric array")
@icontract.require(lambda timestamps: _finite_1d(timestamps), "timestamps must be a finite 1D numeric array")
@icontract.require(lambda groups: _is_1d(groups), "groups must be a 1D array")
@icontract.require(lambda values, timestamps, groups: _same_length(values, timestamps, groups), "values, timestamps, and groups must align")
@icontract.require(lambda decay_rate: _finite_scalar(decay_rate) and decay_rate > 0.0, "decay_rate must be positive and finite")
@icontract.ensure(lambda result, groups: result.shape == (np.unique(groups).shape[0],), "one decayed aggregate must be returned per unique group")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "decayed aggregates must be finite")
def time_decay_aggregate(
    values: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    groups: NDArray[np.object_],
    decay_rate: float,
) -> NDArray[np.float64]:
    """Compute per-group exponentially decayed sums relative to each group's latest timestamp."""
    value_array = np.asarray(values, dtype=np.float64)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    group_array = np.asarray(groups)
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    max_timestamps = np.full(unique_groups.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(max_timestamps, inverse, timestamp_array)
    weights = np.exp(-float(decay_rate) * (max_timestamps[inverse] - timestamp_array))
    return np.bincount(inverse, weights=value_array * weights, minlength=unique_groups.shape[0]).astype(np.float64)


@register_atom(witness_pairwise_products)
@icontract.require(lambda features: _finite_2d(features), "features must be a finite 2D numeric matrix")
@icontract.ensure(lambda result, features: result.shape == (np.asarray(features).shape[0], np.asarray(features).shape[1] * (np.asarray(features).shape[1] - 1) // 2), "one column must be returned per unordered feature pair")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "pairwise products must be finite")
def pairwise_products(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create unordered pairwise feature-product columns for a tabular design matrix."""
    feature_array = np.asarray(features, dtype=np.float64)
    pairs = [(left, right) for left in range(feature_array.shape[1]) for right in range(left + 1, feature_array.shape[1])]
    if not pairs:
        return np.empty((feature_array.shape[0], 0), dtype=np.float64)
    columns = [feature_array[:, left] * feature_array[:, right] for left, right in pairs]
    return np.column_stack(columns).astype(np.float64)


@register_atom(witness_pairwise_ratios)
@icontract.require(lambda features: _finite_2d(features), "features must be a finite 2D numeric matrix")
@icontract.require(lambda epsilon: _finite_scalar(epsilon) and epsilon > 0.0, "epsilon must be positive and finite")
@icontract.ensure(lambda result, features: result.shape == (np.asarray(features).shape[0], np.asarray(features).shape[1] * (np.asarray(features).shape[1] - 1) // 2), "one column must be returned per unordered feature pair")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "pairwise ratios must be finite")
def pairwise_ratios(
    features: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Create unordered pairwise ratio columns using a positive denominator guard."""
    feature_array = np.asarray(features, dtype=np.float64)
    pairs = [(left, right) for left in range(feature_array.shape[1]) for right in range(left + 1, feature_array.shape[1])]
    if not pairs:
        return np.empty((feature_array.shape[0], 0), dtype=np.float64)
    columns = [
        feature_array[:, left] / np.where(np.abs(feature_array[:, right]) < float(epsilon), float(epsilon), feature_array[:, right])
        for left, right in pairs
    ]
    return np.column_stack(columns).astype(np.float64)


@register_atom(witness_missing_indicator_and_impute)
@icontract.require(lambda values: _is_1d_or_2d(values), "values must be a 1D or 2D numeric array")
@icontract.require(lambda fill_value: _finite_scalar(fill_value), "fill_value must be finite")
@icontract.ensure(lambda result, values: result[0].shape == np.asarray(values).shape and result[1].shape == np.asarray(values).shape, "imputed values and indicators must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])), "imputed values must be finite")
def missing_indicator_and_impute(
    values: NDArray[np.float64],
    fill_value: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return mean-free missing indicators plus an array with NaNs replaced by a fill value."""
    array = np.asarray(values, dtype=np.float64)
    missing = np.isnan(array)
    imputed = np.where(missing, float(fill_value), array).astype(np.float64)
    return imputed, missing.astype(np.float64)


@register_atom(witness_rank_transform)
@icontract.require(lambda values: _finite_1d(values), "values must be a finite 1D numeric array")
@icontract.require(lambda method: method in RANK_METHODS, "method must be supported by scipy.stats.rankdata")
@icontract.ensure(lambda result, values: result.shape == np.asarray(values).shape, "rank output must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "ranks must be finite")
def rank_transform(
    values: NDArray[np.float64],
    method: str = "average",
) -> NDArray[np.float64]:
    """Replace numeric values with one-based ranks using SciPy's tie handling semantics."""
    return np.asarray(stats.rankdata(np.asarray(values, dtype=np.float64), method=method), dtype=np.float64)


@register_atom(witness_frequency_encode_fit)
@icontract.require(lambda categories: _is_1d(categories), "categories must be a 1D array")
@icontract.ensure(lambda result: _finite_mapping_values(result), "frequency map values must be finite nonnegative fractions")
def frequency_encode_fit(
    categories: NDArray[np.object_],
) -> dict[object, float]:
    """Fit a deterministic category-to-relative-frequency mapping."""
    category_array = np.asarray(categories, dtype=object)
    unique_values, counts = np.unique(category_array, return_counts=True)
    total = float(category_array.shape[0])
    return {value.item() if hasattr(value, "item") else value: float(count) / total for value, count in zip(unique_values, counts, strict=True)}


@register_atom(witness_frequency_encode)
@icontract.require(lambda categories: _is_1d(categories), "categories must be a 1D array")
@icontract.require(lambda frequency_map: _finite_mapping_values(frequency_map), "frequency_map values must be finite nonnegative fractions")
@icontract.require(lambda unknown_value: _finite_scalar(unknown_value) and unknown_value >= 0.0, "unknown_value must be finite and nonnegative")
@icontract.ensure(lambda result, categories: result.shape == np.asarray(categories).shape, "encoded frequencies must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)) and np.all(result >= 0.0), "encoded frequencies must be finite and nonnegative")
def frequency_encode(
    categories: NDArray[np.object_],
    frequency_map: Mapping[object, float],
    unknown_value: float = 0.0,
) -> NDArray[np.float64]:
    """Map categories to fitted relative frequencies, using a fixed value for unknowns."""
    category_array = np.asarray(categories, dtype=object)
    encoded = [float(frequency_map.get(value, unknown_value)) for value in category_array.tolist()]
    return np.asarray(encoded, dtype=np.float64)


@register_atom(witness_target_encode)
@icontract.require(lambda categories: _is_1d(categories), "categories must be a 1D array")
@icontract.require(lambda targets: _finite_1d(targets), "targets must be a finite 1D numeric array")
@icontract.require(lambda categories, targets: _same_length(categories, targets), "categories and targets must align")
@icontract.require(lambda smoothing: _finite_scalar(smoothing) and smoothing >= 0.0, "smoothing must be finite and nonnegative")
@icontract.require(lambda prior: prior is None or _finite_scalar(prior), "prior must be finite when supplied")
@icontract.ensure(lambda result, categories: _mapping_has_all_categories(categories, result), "all observed categories must be encoded")
@icontract.ensure(lambda result: _finite_encoding_values(result), "target encodings must be finite")
def target_encode(
    categories: NDArray[np.object_],
    targets: NDArray[np.float64],
    smoothing: float = 1.0,
    prior: float | None = None,
) -> dict[object, float]:
    """Compute smoothed target means for each category without storing estimator state."""
    category_array = np.asarray(categories, dtype=object)
    target_array = np.asarray(targets, dtype=np.float64)
    global_mean = float(np.mean(target_array) if prior is None else prior)
    unique_values, inverse = np.unique(category_array, return_inverse=True)
    sums = np.bincount(inverse, weights=target_array, minlength=unique_values.shape[0])
    counts = np.bincount(inverse, minlength=unique_values.shape[0]).astype(np.float64)
    smooth = float(smoothing)
    encodings = (sums + smooth * global_mean) / (counts + smooth)
    return {
        value.item() if hasattr(value, "item") else value: float(encoded)
        for value, encoded in zip(unique_values, encodings, strict=True)
    }


@register_atom(witness_tweedie_gradient)
@icontract.require(lambda predictions: _finite_1d(predictions), "predictions must be finite raw-score values")
@icontract.require(lambda targets: _finite_1d_nonnegative(targets), "targets must be finite nonnegative values")
@icontract.require(lambda predictions, targets: _same_length(predictions, targets), "predictions and targets must align")
@icontract.require(lambda power: _finite_scalar(power) and 1.0 < power < 2.0, "Tweedie power must be in (1, 2)")
@icontract.ensure(lambda result, targets: result[0].shape == np.asarray(targets).shape and result[1].shape == np.asarray(targets).shape, "gradient and hessian must align with targets")
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])) and np.all(np.isfinite(result[1])) and np.all(result[1] > 0.0), "Tweedie derivatives must be finite with positive hessians")
def tweedie_gradient(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    power: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute gradient and hessian for a Tweedie loss with log-link raw predictions."""
    raw = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    rho = float(power)
    exp_one = np.exp((1.0 - rho) * raw)
    exp_two = np.exp((2.0 - rho) * raw)
    gradient = -target * exp_one + exp_two
    hessian = -target * (1.0 - rho) * exp_one + (2.0 - rho) * exp_two
    return gradient.astype(np.float64), hessian.astype(np.float64)


@register_atom(witness_log_cosh_gradient)
@icontract.require(lambda targets: _finite_1d(targets), "targets must be finite")
@icontract.require(lambda predictions: _finite_1d(predictions), "predictions must be finite")
@icontract.require(lambda targets, predictions: _same_length(targets, predictions), "targets and predictions must align")
@icontract.ensure(lambda result, targets: result[0].shape == np.asarray(targets).shape and result[1].shape == np.asarray(targets).shape, "gradient and hessian must align with targets")
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])) and np.all(np.isfinite(result[1])) and np.all(result[1] >= 0.0), "log-cosh derivatives must be finite with nonnegative hessians")
def log_cosh_gradient(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute gradient and hessian for the smooth log-cosh regression loss."""
    residual = np.clip(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64), -20.0, 20.0)
    gradient = np.tanh(residual)
    hessian = 1.0 / np.cosh(residual) ** 2
    return gradient.astype(np.float64), hessian.astype(np.float64)


@register_atom(witness_null_importance_p_values)
@icontract.require(lambda actual_importances: _finite_1d(actual_importances), "actual_importances must be a finite 1D vector")
@icontract.require(lambda null_importances_matrix: _finite_2d(null_importances_matrix), "null_importances_matrix must be a finite 2D matrix")
@icontract.require(lambda actual_importances, null_importances_matrix: np.asarray(null_importances_matrix).shape[1] == np.asarray(actual_importances).shape[0], "null matrix columns must match actual importances")
@icontract.ensure(lambda result, actual_importances: result.shape == np.asarray(actual_importances).shape, "p-values must align with feature importances")
@icontract.ensure(lambda result: np.all((0.0 <= result) & (result <= 1.0)), "p-values must lie in [0, 1]")
def null_importance_p_values(
    actual_importances: NDArray[np.float64],
    null_importances_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Estimate feature-importance p-values from a null-importance sample matrix."""
    actual = np.asarray(actual_importances, dtype=np.float64)
    nulls = np.asarray(null_importances_matrix, dtype=np.float64)
    return np.mean(nulls >= actual[np.newaxis, :], axis=0).astype(np.float64)


@register_atom(witness_extract_pseudo_labels)
@icontract.require(lambda test_predictions: _finite_1d(test_predictions), "test_predictions must be finite")
@icontract.require(lambda test_predictions: np.all((0.0 <= np.asarray(test_predictions)) & (np.asarray(test_predictions) <= 1.0)), "test_predictions must be probabilities")
@icontract.require(lambda lower_threshold, upper_threshold: _valid_thresholds(lower_threshold, upper_threshold), "thresholds must satisfy 0 <= lower < upper <= 1")
@icontract.ensure(lambda result: np.intersect1d(result[0], result[1]).shape[0] == 0, "positive and negative pseudo-label indices must be disjoint")
def extract_pseudo_labels(
    test_predictions: NDArray[np.float64],
    upper_threshold: float,
    lower_threshold: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return high-confidence positive and negative indices from probability predictions."""
    predictions = np.asarray(test_predictions, dtype=np.float64)
    positive = np.flatnonzero(predictions >= float(upper_threshold)).astype(np.int64)
    negative = np.flatnonzero(predictions <= float(lower_threshold)).astype(np.int64)
    return positive, negative
