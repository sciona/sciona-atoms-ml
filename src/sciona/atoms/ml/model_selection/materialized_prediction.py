"""Materialized native prediction and cross-domain ensemble operations."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.materialized_prediction import predict_named_frame, average_prediction_vectors


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Materialized model and named feature frame required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def predict(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Finite scalar predictions in row order with exact model schema and one native thread."""
    return predict_named_frame(model, frame)


@register_atom(witness=witness_materialized)
def average(vectors: list) -> np.ndarray:
    """Equal-weight ordered mean without broadcasting or nonfinite values."""
    return average_prediction_vectors(vectors)
