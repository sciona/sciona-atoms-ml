"""Materialized cross-domain temporal regression split and bounded native fit."""
import numpy as np
import pandas as pd
from sciona.ghost.registry import register_atom
from sciona.named_regression_training import temporal_regression_split, fit_catboost_regression


def witness_materialized(*args: object, **kwargs: object) -> object:
    raise ValueError('Materialized named training tables required; static propagation is unsupported')


@register_atom(witness=witness_materialized)
def temporal_split(frame: pd.DataFrame, targets: np.ndarray, times: pd.DatetimeIndex,
        eligible: np.ndarray, cutoff: object, offsets: np.ndarray, categorical_columns: list) -> tuple:
    """Explicit temporal split with row-aligned target offsets and categorical schema."""
    return temporal_regression_split(frame,targets,times,eligible,cutoff,offsets,categorical_columns)


@register_atom(witness=witness_materialized)
def fit(x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame, y_valid: pd.Series,
        categorical_indices: list, parameters: dict, early_stopping_rounds: int) -> object:
    """One CPU CatBoost regressor with fixed single-thread execution and no backend files."""
    return fit_catboost_regression(x_train,y_train,x_valid,y_valid,categorical_indices,parameters,early_stopping_rounds)
