from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_cartesian_grid,
    witness_fit_cv_candidate,
)

@register_atom(witness_generate_cartesian_grid, name="generate_cartesian_grid")
@icontract.require(lambda param_grid: param_grid is not None, "Precondition failed: param_grid is not None")
@icontract.ensure(lambda result, param_grid: len(configurations) > 0, "Postcondition failed: len(configurations) > 0")
def generate_cartesian_grid(param_grid: str) -> str:
    """Compile hyperparameter dictionary combinations into list of explicit configurations.

    Args:
        param_grid: dict[str, list[Any]]

    Returns:
        configurations: list[dict[str, Any]]
    """
    import sklearn.model_selection
    return sklearn.model_selection.ParameterGrid(param_grid=param_grid) # type: ignore

@register_atom(witness_fit_cv_candidate, name="fit_cv_candidate")
@icontract.require(lambda estimator, features, targets, params, train_indices, val_indices: estimator is not None, "Precondition failed: estimator is not None")
@icontract.ensure(lambda result, estimator, features, targets, params, train_indices, val_indices: result is not None, "Postcondition failed: result is not None")
def fit_cv_candidate(estimator: Any, features: NDArray[np.float64], targets: NDArray[Any], params: str, train_indices: NDArray[np.int64], val_indices: NDArray[np.int64]) -> float:
    """Fit a single parameter combination on a specific training fold and score on validation.

    Args:
        estimator: Any
        features: NDArray[np.float64]
        targets: NDArray[Any]
        params: dict[str, Any]
        train_indices: NDArray[np.int64]
        val_indices: NDArray[np.int64]

    Returns:
        validation_score: float
    """
    import sklearn.model_selection._validation
    return sklearn.model_selection._validation._fit_and_score(estimator=estimator, features=features, targets=targets, params=params, train_indices=train_indices, val_indices=val_indices) # type: ignore

