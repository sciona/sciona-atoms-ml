from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compile_oof_meta_features(base_estimators: AbstractScalar | Any, features: AbstractArray, targets: AbstractArray, cv_folds: AbstractArray) -> AbstractArray:
    """Ghost witness for compile_oof_meta_features."""
    _ = (base_estimators, features, targets, cv_folds)
    return AbstractArray(shape=features.shape, dtype=features.dtype)

def witness_fit_meta_model(meta_features: AbstractArray, targets: AbstractArray, meta_estimator_template: AbstractScalar | Any) -> AbstractScalar:
    """Ghost witness for fit_meta_model."""
    _ = (meta_features, targets, meta_estimator_template)
    return AbstractScalar(dtype="float64")

