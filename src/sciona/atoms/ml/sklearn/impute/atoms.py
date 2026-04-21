"""Dense numeric imputation atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import MissingIndicatorState, SimpleImputerState
from .witnesses import (
    witness_missing_indicator_fit,
    witness_missing_indicator_transform,
    witness_simple_imputer_fit,
    witness_simple_imputer_transform,
)

SimpleStrategy = Literal["mean", "median", "most_frequent", "constant"]
IndicatorFeatures = Literal["missing-only", "all"]


def _numeric_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2)


def _valid_strategy(strategy: str) -> bool:
    return strategy in {"mean", "median", "most_frequent", "constant"}


def _valid_features_mode(features: str) -> bool:
    return features in {"missing-only", "all"}


def _feature_count(X: NDArray[np.float64]) -> int:
    return int(np.asarray(X).shape[1])


def _row_count(X: NDArray[np.float64]) -> int:
    return int(np.asarray(X).shape[0])


def _state_valid(state: SimpleImputerState) -> bool:
    return bool(
        state.statistics.shape == (state.n_features_in,)
        and state.valid_features.ndim == 1
        and np.all((0 <= state.valid_features) & (state.valid_features < state.n_features_in))
    )


def _indicator_state_valid(state: MissingIndicatorState) -> bool:
    return bool(
        state.features.ndim == 1
        and np.all((0 <= state.features) & (state.features < state.n_features_in))
    )


def _simple_fit_result_valid(result: SimpleImputerState, X: NDArray[np.float64]) -> bool:
    return bool(_state_valid(result) and result.n_features_in == _feature_count(X))


def _simple_transform_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: SimpleImputerState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_row_count(X), state.valid_features.shape[0]) and np.all(np.isfinite(values)))


def _indicator_fit_result_valid(result: MissingIndicatorState, X: NDArray[np.float64]) -> bool:
    return bool(_indicator_state_valid(result) and result.n_features_in == _feature_count(X))


def _indicator_transform_result_valid(result: NDArray[np.bool_], X: NDArray[np.float64], state: MissingIndicatorState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (_row_count(X), state.features.shape[0]) and values.dtype == np.bool_)


def _most_frequent_observed(values: NDArray[np.float64]) -> float:
    observed = values[~np.isnan(values)]
    if observed.size == 0:
        return np.nan
    uniques, counts = np.unique(observed, return_counts=True)
    return float(np.min(uniques[counts == np.max(counts)]))


@register_atom(witness_simple_imputer_fit)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda strategy: _valid_strategy(strategy), "strategy must be one of the supported simple imputation modes")
@icontract.require(lambda fill_value: np.isfinite(float(fill_value)), "fill_value must be finite")
@icontract.ensure(lambda result, X: _simple_fit_result_valid(result, X), "state must contain one statistic per input feature")
def simple_imputer_fit(
    X: NDArray[np.float64],
    *,
    strategy: SimpleStrategy = "mean",
    fill_value: float = 0.0,
    keep_empty_features: bool = False,
) -> SimpleImputerState:
    """Learn dense numeric fill statistics for simple missing-value imputation."""
    values = np.asarray(X, dtype=np.float64)
    missing_mask = np.isnan(values)
    statistics = np.empty(values.shape[1], dtype=np.float64)

    for col in range(values.shape[1]):
        observed = values[~missing_mask[:, col], col]
        if strategy == "constant":
            statistic = float(fill_value)
        elif observed.size == 0:
            statistic = 0.0 if keep_empty_features else np.nan
        elif strategy == "mean":
            statistic = float(np.mean(observed))
        elif strategy == "median":
            statistic = float(np.median(observed))
        else:
            statistic = _most_frequent_observed(values[:, col])

        if strategy == "constant" and observed.size == 0 and not keep_empty_features:
            statistic = np.nan
        statistics[col] = statistic

    valid_features = np.arange(values.shape[1], dtype=np.int64)
    if not keep_empty_features:
        valid_features = np.flatnonzero(~np.isnan(statistics)).astype(np.int64)

    return SimpleImputerState(
        statistics=statistics,
        valid_features=valid_features,
        n_features_in=int(values.shape[1]),
        keep_empty_features=bool(keep_empty_features),
    )


@register_atom(witness_simple_imputer_transform)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain valid fitted simple-imputer statistics")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _simple_transform_result_valid(result, X, state), "transformed matrix must match fitted valid features")
def simple_imputer_transform(X: NDArray[np.float64], state: SimpleImputerState) -> NDArray[np.float64]:
    """Replace missing entries with fitted simple-imputer statistics."""
    values = np.asarray(X, dtype=np.float64)
    selected = values[:, state.valid_features].copy()
    statistics = state.statistics[state.valid_features]
    missing_mask = np.isnan(selected)
    if np.any(missing_mask):
        n_missing = np.sum(missing_mask, axis=0)
        fill_values = np.repeat(statistics, n_missing)
        coordinates = np.where(missing_mask.T)[::-1]
        selected[coordinates] = fill_values
    return selected


@register_atom(witness_missing_indicator_fit)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda features: _valid_features_mode(features), "features must be 'missing-only' or 'all'")
@icontract.ensure(lambda result, X: _indicator_fit_result_valid(result, X), "indicator state must reference input features")
def missing_indicator_fit(
    X: NDArray[np.float64],
    *,
    features: IndicatorFeatures = "missing-only",
) -> MissingIndicatorState:
    """Learn which dense numeric feature columns should emit missing indicators."""
    values = np.asarray(X, dtype=np.float64)
    if features == "all":
        feature_indices = np.arange(values.shape[1], dtype=np.int64)
    else:
        feature_indices = np.flatnonzero(np.isnan(values).sum(axis=0)).astype(np.int64)
    return MissingIndicatorState(
        features=feature_indices,
        n_features_in=int(values.shape[1]),
        missing_only=features == "missing-only",
    )


@register_atom(witness_missing_indicator_transform)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda state: _indicator_state_valid(state), "state must contain valid indicator feature indices")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _indicator_transform_result_valid(result, X, state), "indicator mask shape must match fitted features")
def missing_indicator_transform(
    X: NDArray[np.float64],
    state: MissingIndicatorState,
    *,
    error_on_new: bool = True,
) -> NDArray[np.bool_]:
    """Return a fitted boolean missing-value indicator mask for dense input."""
    values = np.asarray(X, dtype=np.float64)
    missing_mask = np.isnan(values)
    if state.missing_only and error_on_new:
        current_missing = np.flatnonzero(missing_mask.sum(axis=0))
        new_missing = np.setdiff1d(current_missing, state.features)
        if new_missing.size > 0:
            raise ValueError(f"features {new_missing} have new missing values")
    return np.asarray(missing_mask[:, state.features], dtype=np.bool_)
