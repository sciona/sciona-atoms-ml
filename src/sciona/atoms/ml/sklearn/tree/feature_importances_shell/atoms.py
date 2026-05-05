"""Sklearn tree feature-importance atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_tree_feature_importances_result


def _feature_importance_vector(values: object) -> bool:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        vector.ndim == 1
        and vector.shape[0] >= 1
        and np.all(np.isfinite(vector))
        and np.all(vector >= 0.0)
        and float(vector.sum()) <= 1.0 + 1e-12
    )


@register_atom(witness_tree_feature_importances_result)
@icontract.require(
    lambda importances: _feature_importance_vector(importances),
    "importances must be a nonempty finite nonnegative feature-importance vector",
)
@icontract.ensure(
    lambda result, importances: _feature_importance_vector(result)
    and np.allclose(np.asarray(result, dtype=np.float64), np.asarray(importances, dtype=np.float64)),
    "feature importances must preserve the supplied native result vector",
)
def tree_feature_importances_result(
    importances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return BaseDecisionTree.feature_importances_' final importance vector."""
    return np.asarray(importances, dtype=np.float64)
