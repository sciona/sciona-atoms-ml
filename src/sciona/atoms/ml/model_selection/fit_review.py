"""Materialized regression score acceptance and feature-importance providers."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.regression_fit_review import mean_absolute_error, accept_model_below, native_feature_importance, normalize_ranked_importance
from sciona.atoms.ml.model_selection.named_regression import witness_materialized


@register_atom(witness=witness_materialized)
def absolute_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error for nonempty finite aligned target-unit vectors."""
    return mean_absolute_error(observed,predicted)


@register_atom(witness=witness_materialized)
def accept(model: object, score: float, ceiling: float) -> object:
    """Retain native model identity only for finite error strictly below an explicit ceiling."""
    return accept_model_below(model,score,ceiling)


@register_atom(witness=witness_materialized)
def importance(model: object) -> tuple:
    """Native importance and aligned names using one thread."""
    return native_feature_importance(model)


@register_atom(witness=witness_materialized)
def normalize(names: list, values: np.ndarray) -> pd.DataFrame:
    """Normalize nonnegative importance to unit sum and sort descending."""
    return normalize_ranked_importance(names,values)
