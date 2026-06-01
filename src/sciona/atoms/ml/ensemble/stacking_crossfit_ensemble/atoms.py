from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compile_oof_meta_features,
    witness_fit_meta_model,
)

@register_atom(witness_compile_oof_meta_features, name="compile_oof_meta_features")
@icontract.require(lambda base_estimators, features, targets, cv_folds: len(base_estimators) >= 1, "Precondition failed: len(base_estimators) >= 1")
@icontract.ensure(lambda result, base_estimators, features, targets, cv_folds: meta_features.shape[0] == features.shape[0], "Postcondition failed: meta_features.shape[0] == features.shape[0]")
@icontract.ensure(lambda result, base_estimators, features, targets, cv_folds: meta_features.shape[1] == len(base_estimators), "Postcondition failed: meta_features.shape[1] == len(base_estimators)")
def compile_oof_meta_features(base_estimators: Any, features: NDArray[np.float64], targets: NDArray[Any], cv_folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]]) -> NDArray[np.float64]:
    """Generate out-of-fold prediction matrix from base estimators using cross-validation.

    Args:
        base_estimators: list[Any]
        features: NDArray[np.float64]
        targets: NDArray[Any]
        cv_folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]]

    Returns:
        meta_features: NDArray[np.float64]
    """
    import sklearn.model_selection
    return sklearn.model_selection.cross_val_predict(base_estimators=base_estimators, features=features, targets=targets, cv_folds=cv_folds) # type: ignore

@register_atom(witness_fit_meta_model, name="fit_meta_model")
@icontract.require(lambda meta_features, targets, meta_estimator_template: meta_features is not None, "Precondition failed: meta_features is not None")
@icontract.ensure(lambda result, meta_features, targets, meta_estimator_template: result is not None, "Postcondition failed: result is not None")
def fit_meta_model(meta_features: NDArray[np.float64], targets: NDArray[Any], meta_estimator_template: Any) -> Any:
    """Train the meta-estimator to combine the out-of-fold predictions of the base estimators.

    Args:
        meta_features: NDArray[np.float64]
        targets: NDArray[Any]
        meta_estimator_template: Any

    Returns:
        fitted_meta_model: Any
    """
    import sklearn.base.BaseEstimator
    return sklearn.base.BaseEstimator.fit(meta_features=meta_features, targets=targets, meta_estimator_template=meta_estimator_template) # type: ignore

