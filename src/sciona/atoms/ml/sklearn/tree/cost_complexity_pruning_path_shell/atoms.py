"""Sklearn tree pruning-path atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_tree_pruning_path_estimator, witness_tree_pruning_path_result

def _supports_ccp_alpha(value: object) -> bool:
    return bool(hasattr(value, "get_params") and hasattr(value, "set_params"))

def _float_vector(values: object) -> bool:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] >= 1 and np.all(np.isfinite(vector)))

def _pruning_path_mapping(values: object) -> bool:
    if not isinstance(values, Mapping):
        return False
    if set(values.keys()) != {"ccp_alphas", "impurities"}:
        return False
    alphas = values["ccp_alphas"]
    impurities = values["impurities"]
    return bool(
        _float_vector(alphas)
        and _float_vector(impurities)
        and np.asarray(alphas, dtype=np.float64).shape == np.asarray(impurities, dtype=np.float64).shape
    )

@register_atom(witness_tree_pruning_path_estimator)
@icontract.require(
    lambda estimator: _supports_ccp_alpha(estimator),
    "estimator must support get_params and set_params",
)
@icontract.ensure(
    lambda result, estimator: _supports_ccp_alpha(result)
    and result is not estimator
    and result.get_params()["ccp_alpha"] == 0.0
    and estimator.get_params() == result.get_params() | {"ccp_alpha": estimator.get_params().get("ccp_alpha")},
    "pruning-path estimator must be a cloned estimator with ccp_alpha reset to 0.0",
)
def tree_pruning_path_estimator(estimator: object) -> object:
    from sklearn.base import clone
    """Return the cloned zero-alpha estimator used by cost_complexity_pruning_path."""
    return clone(estimator).set_params(ccp_alpha=0.0)

@register_atom(witness_tree_pruning_path_result)
@icontract.require(
    lambda pruning_path: _pruning_path_mapping(pruning_path),
    "pruning_path must be a mapping with aligned ccp_alphas and impurities vectors",
)
@icontract.ensure(
    lambda result, pruning_path: isinstance(result, Bunch)
    and _float_vector(result.ccp_alphas)
    and _float_vector(result.impurities)
    and np.allclose(result.ccp_alphas, np.asarray(pruning_path["ccp_alphas"], dtype=np.float64))
    and np.allclose(result.impurities, np.asarray(pruning_path["impurities"], dtype=np.float64)),
    "pruning-path result must package ccp_alphas and impurities into a Bunch",
)
def tree_pruning_path_result(
    pruning_path: Mapping[str, NDArray[np.float64]],
) -> Bunch:
    from sklearn.utils import Bunch
    """Return the Bunch packaging used by cost_complexity_pruning_path."""
    return Bunch(**pruning_path)
