"""Public software field conversion for the observation-safe baseline."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.nasa_first_feature_adapter import _times, MINUTE
from sciona.atoms.ml.model_selection.graph_fallbacks import witness_materialized


@register_atom(witness=witness_materialized)
def baseline_inputs(queries: pd.DataFrame, raw_options: dict) -> tuple:
    """Expose baseline history, availability and source constants as graph operands."""
    history = raw_options['etd']
    return (history['gufi'].to_numpy(), _times(history['timestamp']),
        _times(history['departure_runway_estimated_time'], missing=True),
        history['departure_runway_estimated_time'].notna().to_numpy(), queries['gufi'].to_numpy(),
        _times(queries['timestamp']), float(MINUTE), 30., 1., 299., 15.)


@register_atom(witness=witness_materialized)
def prediction_values(predictions: pd.DataFrame) -> np.ndarray:
    """Extract an owned positional prediction vector for fallback validation."""
    return predictions['minutes_until_pushback'].to_numpy(copy=True)


@register_atom(witness=witness_materialized)
def execution_inputs(queries: pd.DataFrame, models: dict, population: str, raw_options: dict,
        feature_columns: list, numeric_fills: dict, categorical_fills: dict) -> tuple:
    """Expose fixed branch definitions and caller inputs for generic lazy routing."""
    from sciona.nasa_first_guarded_graph import branch_descriptors
    primary, baseline = branch_descriptors()
    inputs = dict(queries=queries,models=models,population=population,raw_options=raw_options,
        feature_columns=feature_columns,numeric_fills=numeric_fills,categorical_fills=categorical_fills,integer_dtype='int16')
    return primary,baseline,inputs,len(queries),30.
