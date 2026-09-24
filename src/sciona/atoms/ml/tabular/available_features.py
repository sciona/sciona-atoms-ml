"""Materialized, domain-neutral feature operations with explicit availability.

These wrappers expose qualified numerical/table contracts to execution graphs.
Runtime data and policy values are caller-owned. No static table-shape proof or
automatic unit inference is claimed; the witness rejects those unsupported uses.
"""
import numpy as np
import pandas as pd

from sciona.ghost.registry import register_atom
from sciona.asof_indices import latest_available_indices as _asof
from sciona.configuration_state_features import configuration_state_features as _configuration
from sciona.estimate_history_features import estimate_history_features as _estimates
from sciona.forecast_history_features import forecast_history_features as _forecasts
from sciona.event_count_windows import event_count_windows as _counts
from sciona.event_delay_windows import event_delay_windows as _delays
from sciona.calendar_time_features import calendar_time_features as _calendar
from sciona.unique_entity_features import unique_entity_features as _entities
from sciona.structured_identity_features import structured_identity_features as _identifiers
from sciona.feature_table_assembly import assemble_feature_tables as _assemble
from sciona.model_feature_policy import prepare_model_features as _prepare
from sciona.asof_estimate_fallback import asof_estimate_fallback as _fallback
from sciona.prediction_postprocessing import clip_truncate_prediction as _clip


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Explicit materialized tables, availability and unit policies required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def available_indices(observed_at: np.ndarray, query_times: np.ndarray) -> tuple:
    """Latest available original row indices and missing mask on one int64 clock."""
    return _asof(observed_at,query_times)


@register_atom(witness=witness_materialized)
def configuration_features(observed_at: np.ndarray, states: np.ndarray, query_times: np.ndarray,
        vocabulary: list, suffixes: list, change_windows: np.ndarray, grid_step: int) -> dict:
    """Available categorical states and full-window change counts on an explicit grid."""
    return _configuration(observed_at,states,query_times,vocabulary,suffixes,change_windows,grid_step)


@register_atom(witness=witness_materialized)
def estimate_features(entity_ids: np.ndarray, observed_at: np.ndarray, estimates: np.ndarray,
        estimate_valid: np.ndarray, query_ids: np.ndarray, query_times: np.ndarray,
        lookbacks: np.ndarray, rounding_step: int, ticks_per_output_unit: int) -> dict:
    """Per-entity estimate changes, remaining times and query-independent windows."""
    return _estimates(entity_ids,observed_at,estimates,estimate_valid,query_ids,query_times,lookbacks,rounding_step,ticks_per_output_unit)


@register_atom(witness=witness_materialized)
def forecast_features(issued_at: np.ndarray, valid_at: np.ndarray, values: np.ndarray,
        query_times: np.ndarray, history_window: int, lead_bands: list,
        contrast_leads: np.ndarray, revision_lags: np.ndarray) -> dict:
    """Issued-as-of forecast history, lead bands, paired contrasts and revisions."""
    return _forecasts(issued_at,valid_at,values,query_times,history_window=history_window,
        lead_bands=lead_bands,contrast_leads=contrast_leads,revision_lags=revision_lags)


@register_atom(witness=witness_materialized)
def event_counts(observed_at: np.ndarray, actual_at: np.ndarray, countable: np.ndarray,
        query_times: np.ndarray, lookbacks: np.ndarray) -> np.ndarray:
    """Per-channel event counts requiring both observation and actual availability."""
    return _counts(observed_at,actual_at,countable,query_times,lookbacks)


@register_atom(witness=witness_materialized)
def event_delays(event_ids: np.ndarray, observed_at: np.ndarray, actual_at: np.ndarray,
        estimate_ids: np.ndarray, estimate_observed_at: np.ndarray, estimate_times: np.ndarray,
        estimate_valid: np.ndarray, query_times: np.ndarray, lookbacks: np.ndarray,
        rounding_step: int, ticks_per_output_unit: int) -> tuple:
    """Fixed-window event counts and available first-estimate delay statistics."""
    return _delays(event_ids,observed_at,actual_at,estimate_ids,estimate_observed_at,estimate_times,
        estimate_valid,query_times,lookbacks,rounding_step,ticks_per_output_unit)


@register_atom(witness=witness_materialized)
def calendar_features(query_times: pd.DatetimeIndex, holiday_midnights: pd.DatetimeIndex) -> dict:
    """Calendar features with an explicit holiday schedule and coverage checks."""
    return _calendar(query_times,holiday_midnights)


@register_atom(witness=witness_materialized)
def entity_attributes(query_ids: np.ndarray, record_ids: np.ndarray, attributes: pd.DataFrame,
        vocabularies: dict, fallback: str) -> pd.DataFrame:
    """Unique authoritative entity attributes with explicit vocabulary fallbacks."""
    return _entities(query_ids,record_ids,attributes,vocabularies,fallback)


@register_atom(witness=witness_materialized)
def identifier_features(query_ids: np.ndarray, query_times: pd.DatetimeIndex, record_ids: np.ndarray,
        delimiter: str, minimum_parts: int, primary_part: int, category_part: int,
        reference_parts: list, reference_format: str, vocabularies: dict, fallback: str,
        seconds_per_unit: float) -> pd.DataFrame:
    """Declared structured-identifier categories and signed reference-time features."""
    return _identifiers(query_ids,query_times,record_ids,delimiter=delimiter,minimum_parts=minimum_parts,
        primary_part=primary_part,category_part=category_part,reference_parts=reference_parts,
        reference_format=reference_format,vocabularies=vocabularies,fallback=fallback,seconds_per_unit=seconds_per_unit)


@register_atom(witness=witness_materialized)
def assemble_tables(queries: pd.DataFrame, feature_tables: list) -> pd.DataFrame:
    """Explicit many-to-one feature joins preserving all query rows and order."""
    return _assemble(queries,feature_tables)


@register_atom(witness=witness_materialized)
def prepare_features(frame: pd.DataFrame, feature_columns: list, numeric_fills: dict,
        categorical_fills: dict, integer_dtype: str) -> pd.DataFrame:
    """Fixed ordered numeric/categorical model inputs without batch-derived fills."""
    return _prepare(frame,feature_columns,numeric_fills,categorical_fills,integer_dtype=integer_dtype)


@register_atom(witness=witness_materialized)
def estimate_fallback(entity_ids: np.ndarray, observed_at: np.ndarray, estimates: np.ndarray,
        estimate_valid: np.ndarray, query_ids: np.ndarray, query_times: np.ndarray,
        ticks_per_unit: float, offset: float, lower: float, upper: float, missing: float) -> np.ndarray:
    """Latest observed nonmissing estimate fallback with explicit bounds and units."""
    return _fallback(entity_ids,observed_at,estimates,estimate_valid,query_ids,query_times,
        ticks_per_unit=ticks_per_unit,offset=offset,lower=lower,upper=upper,missing=missing)


@register_atom(witness=witness_materialized)
def clip_truncate(values: np.ndarray, offset: np.ndarray, lower: int, upper: int) -> np.ndarray:
    """Add an aligned offset, clip, then truncate predictions toward zero."""
    return _clip(values,offset,lower,upper)
