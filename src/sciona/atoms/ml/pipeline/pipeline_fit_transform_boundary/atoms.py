from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fit_transformer_stage,
    witness_transform_only_stage,
)

@register_atom(witness_fit_transformer_stage, name="fit_transformer_stage")
@icontract.require(lambda data, transformer_template, targets: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.ensure(lambda result, data, transformer_template, targets: transformed_data.shape[0] == data.shape[0], "Postcondition failed: transformed_data.shape[0] == data.shape[0]")
def fit_transformer_stage(data: NDArray[np.float64], transformer_template: Any, targets: Optional[NDArray[Any]] = None) -> Any:
    """Fit a single transformer state on training data and return the updated state parameters.

    Args:
        data: NDArray[np.float64]
        transformer_template: Any
        targets: Optional[NDArray[Any]]

    Returns:
        fitted_state: Any
    """
    import sklearn.base.TransformerMixin
    return sklearn.base.TransformerMixin.fit(data=data, transformer_template=transformer_template, targets=targets) # type: ignore

@register_atom(witness_transform_only_stage, name="transform_only_stage")
@icontract.require(lambda data, fitted_state: data.ndim == 2, "Precondition failed: data.ndim == 2")
@icontract.ensure(lambda result, data, fitted_state: transformed_data.shape[0] == data.shape[0], "Postcondition failed: transformed_data.shape[0] == data.shape[0]")
def transform_only_stage(data: NDArray[np.float64], fitted_state: Any) -> NDArray[np.float64]:
    """Apply a pre-fitted transformer state to validation or test data.

    Args:
        data: NDArray[np.float64]
        fitted_state: Any

    Returns:
        transformed_data: NDArray[np.float64]
    """
    import sklearn.base.TransformerMixin
    return sklearn.base.TransformerMixin.transform(data=data, fitted_state=fitted_state) # type: ignore

