"""Dense numeric imputation atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import KNNImputerState, MissingIndicatorState, SimpleImputerState
from .witnesses import (
    witness_knn_imputer_calc_impute,
    witness_knn_imputer_fit,
    witness_knn_imputer_transform,
    witness_missing_indicator_fit,
    witness_missing_indicator_transform,
    witness_nan_euclidean_distances,
    witness_simple_imputer_fit,
    witness_simple_imputer_transform,
)

SimpleStrategy = Literal["mean", "median", "most_frequent", "constant"]
IndicatorFeatures = Literal["missing-only", "all"]
KNNWeights = Literal["uniform", "distance"]


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


def _valid_knn_weights(weights: str) -> bool:
    return weights in {"uniform", "distance"}


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


def _distance_inputs_valid(X: NDArray[np.float64], Y: NDArray[np.float64]) -> bool:
    return bool(_numeric_matrix(X) and _numeric_matrix(Y) and _feature_count(X) == _feature_count(Y))


def _distance_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], Y: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_row_count(X), _row_count(Y)) and np.all(np.isfinite(values) | np.isnan(values)))


def _knn_state_valid(state: KNNImputerState) -> bool:
    return bool(
        state.fit_X.ndim == 2
        and state.mask_fit_X.shape == state.fit_X.shape
        and state.mask_fit_X.dtype == np.bool_
        and state.valid_mask.shape == (state.fit_X.shape[1],)
        and state.valid_mask.dtype == np.bool_
        and state.n_neighbors >= 1
        and _valid_knn_weights(state.weights)
    )


def _knn_fit_result_valid(result: KNNImputerState, X: NDArray[np.float64]) -> bool:
    return bool(_knn_state_valid(result) and result.fit_X.shape == np.asarray(X).shape)


def _knn_transform_result_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: KNNImputerState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    output_features = state.fit_X.shape[1] if state.keep_empty_features else int(np.sum(state.valid_mask))
    return bool(values.shape == (_row_count(X), output_features) and np.all(np.isfinite(values)))


def _calc_impute_inputs_valid(
    dist_pot_donors: NDArray[np.float64],
    fit_X_col: NDArray[np.float64],
    mask_fit_X_col: NDArray[np.bool_],
    n_neighbors: int,
) -> bool:
    distances = np.asarray(dist_pot_donors, dtype=np.float64)
    values = np.asarray(fit_X_col, dtype=np.float64)
    mask = np.asarray(mask_fit_X_col, dtype=np.bool_)
    return bool(
        distances.ndim == 2
        and values.ndim == 1
        and mask.ndim == 1
        and values.shape == mask.shape
        and distances.shape[1] == values.shape[0]
        and distances.shape[0] >= 1
        and 1 <= n_neighbors <= max(1, distances.shape[1])
        and np.all(np.isfinite(distances) | np.isnan(distances))
    )


def _calc_impute_result_valid(result: NDArray[np.float64], dist_pot_donors: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(dist_pot_donors).shape[0],) and np.all(np.isfinite(values)))


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


@register_atom(witness_nan_euclidean_distances)
@icontract.require(lambda X, Y: _distance_inputs_valid(X, Y), "X and Y must be dense numeric matrices with matching feature counts")
@icontract.ensure(lambda result, X, Y: _distance_result_valid(result, X, Y), "distance matrix must match sample counts")
def nan_euclidean_distances(X: NDArray[np.float64], Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return nan-aware Euclidean distances scaled by observed feature count."""
    left = np.asarray(X, dtype=np.float64)
    right = np.asarray(Y, dtype=np.float64)
    left_mask = np.isnan(left)
    right_mask = np.isnan(right)
    valid = (~left_mask[:, np.newaxis, :]) & (~right_mask[np.newaxis, :, :])
    differences = np.where(valid, left[:, np.newaxis, :] - right[np.newaxis, :, :], 0.0)
    present = np.sum(valid, axis=2)
    squared = np.sum(differences**2, axis=2)
    distances = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    distances.fill(np.nan)
    valid_pairs = present > 0
    distances[valid_pairs] = np.sqrt(squared[valid_pairs] * left.shape[1] / present[valid_pairs])
    return distances


@register_atom(witness_knn_imputer_calc_impute)
@icontract.require(lambda dist_pot_donors, fit_X_col, mask_fit_X_col, n_neighbors: _calc_impute_inputs_valid(dist_pot_donors, fit_X_col, mask_fit_X_col, n_neighbors), "donor distances, values, masks, and neighbor count must align")
@icontract.require(lambda weights: _valid_knn_weights(weights), "weights must be 'uniform' or 'distance'")
@icontract.ensure(lambda result, dist_pot_donors: _calc_impute_result_valid(result, dist_pot_donors), "imputed values must match receiver count")
def knn_imputer_calc_impute(
    dist_pot_donors: NDArray[np.float64],
    fit_X_col: NDArray[np.float64],
    mask_fit_X_col: NDArray[np.bool_],
    *,
    n_neighbors: int,
    weights: KNNWeights = "uniform",
) -> NDArray[np.float64]:
    """Return donor-weighted imputed values for one feature column."""
    distances = np.asarray(dist_pot_donors, dtype=np.float64)
    donor_count = distances.shape[1]
    neighbor_count = min(int(n_neighbors), donor_count)
    donor_indices = np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
    donor_distances = distances[np.arange(donor_indices.shape[0])[:, np.newaxis], donor_indices]

    if weights == "distance":
        with np.errstate(divide="ignore"):
            weight_matrix = 1.0 / donor_distances
        infinite_mask = np.isinf(weight_matrix)
        infinite_rows = np.any(infinite_mask, axis=1)
        weight_matrix[infinite_rows] = infinite_mask[infinite_rows]
        weight_matrix[np.isnan(weight_matrix)] = 0.0
    else:
        weight_matrix = np.ones_like(donor_distances)
        weight_matrix[np.isnan(donor_distances)] = 0.0

    donor_values = np.asarray(fit_X_col, dtype=np.float64).take(donor_indices)
    donor_missing = np.asarray(mask_fit_X_col, dtype=np.bool_).take(donor_indices)
    usable_weights = np.where(donor_missing, 0.0, weight_matrix)
    weight_sums = np.sum(usable_weights, axis=1)
    weighted_values = np.sum(np.where(donor_missing, 0.0, donor_values) * usable_weights, axis=1)
    return weighted_values / weight_sums


@register_atom(witness_knn_imputer_fit)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda n_neighbors: n_neighbors >= 1, "n_neighbors must be positive")
@icontract.require(lambda weights: _valid_knn_weights(weights), "weights must be 'uniform' or 'distance'")
@icontract.ensure(lambda result, X: _knn_fit_result_valid(result, X), "KNN imputer state must store training data and masks")
def knn_imputer_fit(
    X: NDArray[np.float64],
    *,
    n_neighbors: int = 5,
    weights: KNNWeights = "uniform",
    keep_empty_features: bool = False,
) -> KNNImputerState:
    """Store dense numeric training data and masks for KNN imputation."""
    values = np.asarray(X, dtype=np.float64)
    mask = np.isnan(values)
    return KNNImputerState(
        fit_X=values.copy(),
        mask_fit_X=mask,
        valid_mask=~np.all(mask, axis=0),
        n_neighbors=int(n_neighbors),
        weights=weights,
        keep_empty_features=bool(keep_empty_features),
    )


@register_atom(witness_knn_imputer_transform)
@icontract.require(lambda X: _numeric_matrix(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda state: _knn_state_valid(state), "state must contain valid KNN imputer training arrays")
@icontract.require(lambda X, state: _feature_count(X) == state.fit_X.shape[1], "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _knn_transform_result_valid(result, X, state), "KNN imputed output must match valid feature count")
def knn_imputer_transform(X: NDArray[np.float64], state: KNNImputerState) -> NDArray[np.float64]:
    """Impute dense numeric missing values from nearest training donors."""
    values = np.asarray(X, dtype=np.float64).copy()
    mask = np.isnan(values)
    valid_mask = state.valid_mask

    if np.any(mask[:, valid_mask]):
        row_missing_idx = np.flatnonzero(mask[:, valid_mask].any(axis=1))
        non_missing_fit_X = ~state.mask_fit_X
        distances = nan_euclidean_distances(values[row_missing_idx, :], state.fit_X)

        for col in range(values.shape[1]):
            if not valid_mask[col]:
                continue
            col_mask = mask[row_missing_idx, col]
            if not np.any(col_mask):
                continue
            potential_donors_idx = np.flatnonzero(non_missing_fit_X[:, col])
            receivers_idx = row_missing_idx[np.flatnonzero(col_mask)]
            dist_subset = distances[np.flatnonzero(col_mask)][:, potential_donors_idx]
            all_nan_dist_mask = np.isnan(dist_subset).all(axis=1)
            all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]

            if all_nan_receivers_idx.size:
                observed = state.fit_X[~state.mask_fit_X[:, col], col]
                values[all_nan_receivers_idx, col] = float(np.mean(observed))
                if len(all_nan_receivers_idx) == len(receivers_idx):
                    continue
                receivers_idx = receivers_idx[~all_nan_dist_mask]
                dist_subset = dist_subset[~all_nan_dist_mask]

            imputed = knn_imputer_calc_impute(
                dist_subset,
                state.fit_X[potential_donors_idx, col],
                state.mask_fit_X[potential_donors_idx, col],
                n_neighbors=min(state.n_neighbors, len(potential_donors_idx)),
                weights=state.weights,
            )
            values[receivers_idx, col] = imputed

    if state.keep_empty_features:
        values[:, ~valid_mask] = 0.0
        return values
    return values[:, valid_mask]
