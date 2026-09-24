"""Source model-slot, feature-schema and output naming adapters."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.atoms.ml.domain_adapters.first_place_features import witness_materialized


@register_atom(witness=witness_materialized)
def prediction_inputs(frame: pd.DataFrame, models: dict, population: str) -> tuple:
    """Select ordered model inputs and expose explicit source clipping constants."""
    selected = []
    for slot in (0, 1, 2):
        model = models[slot]
        selected.extend([model, frame[model.feature_names_].copy()])
    global_frame = frame.copy()
    global_frame['feat_cat_airport'] = population.lower()
    model = models['global_model']
    selected.extend([model, global_frame[model.feature_names_].copy()])
    return (*selected, frame['etd_time_till_est_dep'].to_numpy(dtype=np.float64),
            np.zeros(len(frame)), 1, 299, 4, 260)


@register_atom(witness=witness_materialized)
def ensemble_inputs(local0: np.ndarray, local1: np.ndarray, local2: np.ndarray,
        global_values: np.ndarray, population: str) -> list:
    """Apply the three public-software population overrides before generic averaging."""
    return [local1] if population.upper() in ['KPHX', 'KMIA', 'KJFK'] else [local0, local1, local2, global_values]


@register_atom(witness=witness_materialized)
def format_predictions(queries: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Attach positional prediction values to an owned query table, preserving duplicate indices."""
    if values.shape != (len(queries),) or not np.isfinite(values).all():
        raise ValueError('Finite aligned output predictions required')
    result = queries.copy(deep=True)
    result['minutes_until_pushback'] = values
    return result
