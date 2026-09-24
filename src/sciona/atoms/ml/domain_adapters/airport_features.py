"""Draft domain boundaries around corrected public airport feature operations.

The reusable numerical core remains separate. Data-dependent table joins have
not been qualified for whole-graph Ghost propagation; their witnesses reject
that unsupported claim instead of inventing a surviving row count.
"""
import numpy as np
import pandas as pd

from sciona.ghost.registry import register_atom
from sciona.nasa_feature_adapters import (
    _check, estimated_departure_features, fit_airline_vocabulary,
    airline_features, arrival_history_features,
)
from sciona.nasa_workflow_inputs import _join, ETD_FEATURES, HISTORY_FEATURES, attach_predictions
from sciona.tabular_contracts import stable_feature_order, ordered_feature_matrix
from sciona.ghost.dimensions import DimensionalSignature
from sciona.residual_metadata_preflight import KEYS, propagate_metadata, validate_materialized_inputs


def witness_unmaterialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Domain table witness requires materialized population/schema qualification; whole-graph symbolic propagation is not yet supported')


def _vocabulary(vocabulary):
    if not isinstance(vocabulary, tuple) or any(not isinstance(value, str) or len(value) != 3
            for value in vocabulary) or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Unique immutable three-character vocabulary required')


def _schema(names):
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name
            or any(character in name for character in '[]<') for name in names) or len(set(names)) != len(names):
        raise ValueError('Unique ordered backend-compatible feature names required')


def _identities(frame):
    _check(frame, ['gufi', 'timestamp', 'airport'], ['timestamp'])
    if frame.empty or frame.duplicated(['gufi', 'timestamp']).any() or frame.airport.nunique() != 1:
        raise ValueError('Nonempty unique identities in one airport required')


def _unpack(records, airport):
    if not isinstance(records, dict) or set(records) != {'queries', 'estimates', 'stands', 'arrivals'}:
        raise ValueError('Explicit query and event runtime tables required')
    queries = records['queries']
    _check(queries, ['gufi', 'timestamp', 'airport'], ['timestamp'])
    if not isinstance(airport, str) or len(airport) != 4 or not (queries.airport == airport).all():
        raise ValueError('One explicit airport slot required')
    if queries.empty or queries.duplicated(['gufi', 'timestamp']).any():
        raise ValueError('Nonempty unique query identities required')
    if any(not isinstance(records[key], pd.DataFrame) for key in ['estimates', 'stands', 'arrivals']):
        raise ValueError('Event DataFrames required')
    return queries, records['estimates'], records['stands'], records['arrivals']


@register_atom(witness=witness_unmaterialized)
def unpack_training(training_records: dict, airport: str) -> tuple:
    values = _unpack(training_records, airport)
    if 'minutes_until_pushback' not in values[0]:
        raise ValueError('Training target required')
    return values


@register_atom(witness=witness_unmaterialized)
def unpack_prediction(prediction_records: dict, airport: str) -> tuple:
    return _unpack(prediction_records, airport)


@register_atom(witness=witness_unmaterialized)
def fit_categories(queries: pd.DataFrame) -> tuple:
    return fit_airline_vocabulary(queries)


@register_atom(witness=witness_unmaterialized)
def departure_features(queries: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    return estimated_departure_features(queries, estimates)


@register_atom(witness=witness_unmaterialized)
def category_features(queries: pd.DataFrame, vocabulary: tuple) -> pd.DataFrame:
    return airline_features(queries, vocabulary)


@register_atom(witness=witness_unmaterialized)
def history_features(queries: pd.DataFrame, stands: pd.DataFrame, arrivals: pd.DataFrame, airport: str) -> pd.DataFrame:
    _identities(queries)
    if not isinstance(airport, str) or len(airport) != 4 or not (queries.airport == airport).all():
        raise ValueError('Explicit airport slot required')
    return arrival_history_features(queries.timestamp, stands, arrivals, airport[-3:])


@register_atom(witness=witness_unmaterialized)
def join_training(queries: pd.DataFrame, departures: pd.DataFrame, categories: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return _join(queries, departures, categories, history, True)


@register_atom(witness=witness_unmaterialized)
def join_prediction(queries: pd.DataFrame, departures: pd.DataFrame, categories: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    return _join(queries, departures, categories, history, False)


@register_atom(witness=witness_unmaterialized)
def training_arrays(frame: pd.DataFrame, vocabulary: tuple) -> tuple:
    if frame.empty:
        raise ValueError('No complete training feature rows')
    _identities(frame)
    _vocabulary(vocabulary)
    names = stable_feature_order(['unix_time']+ETD_FEATURES+['Other']+list(vocabulary)+HISTORY_FEATURES)
    _schema(names)
    features = ordered_feature_matrix(frame, names)
    if 'minutes_until_pushback' not in frame or not pd.api.types.is_numeric_dtype(frame.minutes_until_pushback.dtype) or pd.api.types.is_complex_dtype(frame.minutes_until_pushback.dtype):
        raise ValueError('Numeric training targets required')
    targets = frame.minutes_until_pushback.to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError('Finite training targets required')
    _, groups = np.unique(frame.gufi.to_numpy(), return_inverse=True)
    return features, targets, groups.astype(np.int64), names


@register_atom(witness=witness_unmaterialized)
def prediction_arrays(frame: pd.DataFrame, queries: pd.DataFrame, feature_names: list) -> tuple:
    _identities(queries)
    _schema(feature_names)
    _check(frame, ['gufi', 'timestamp', 'airport'], ['timestamp'])
    if not frame.columns.is_unique or not set(feature_names) <= set(frame):
        raise ValueError('Prediction row or trained feature coverage differs')
    if any(not pd.api.types.is_numeric_dtype(frame[name].dtype) or pd.api.types.is_complex_dtype(frame[name].dtype)
           for name in feature_names):
        raise ValueError('Real numeric prediction features required')
    identities = queries[['gufi','timestamp','airport']].reset_index(drop=True)
    ordered = identities.merge(frame, on=['gufi','timestamp','airport'], how='left', validate='one_to_one')
    if len(ordered) != len(queries) or not ordered.columns.is_unique or not set(feature_names) <= set(ordered):
        raise ValueError('Prediction row or trained feature coverage differs')
    values = ordered[feature_names].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError('Infinite prediction features rejected')
    return values, identities


@register_atom(witness=witness_unmaterialized)
def attach_output(identities: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    _identities(identities)
    return attach_predictions(identities, predictions)


def witness_numerical_handoff(features: object, targets: object, groups: object, feature_names: list,
        prediction_features: object, train_fraction: object, seed: object, maximum_error: object,
        threshold: object, prediction_feature_name: str) -> tuple:
    values = (features,targets,groups,feature_names,prediction_features,train_fraction,seed,
              maximum_error,threshold,prediction_feature_name)
    if targets.dim != DimensionalSignature(T=1) or maximum_error.dim != targets.dim:
        raise ValueError('Common time dimension required for airport target and residual threshold')
    propagate_metadata(dict(zip(KEYS,values)))
    return values


@register_atom(witness=witness_numerical_handoff)
def numerical_handoff(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, feature_names: list,
        prediction_features: np.ndarray, train_fraction: float, seed: int, maximum_error: float,
        threshold: float, prediction_feature_name: str) -> tuple:
    """Check observed array metadata through every numerical node before fitting.

    Airport target and residual threshold share minutes. This checkpoint neither
    predicts join cardinality nor proves future model-dependent populations.
    """
    values = (features,targets,groups,feature_names,prediction_features,train_fraction,seed,
              maximum_error,threshold,prediction_feature_name)
    validate_materialized_inputs(dict(zip(KEYS,values)), DimensionalSignature(T=1))
    return values
