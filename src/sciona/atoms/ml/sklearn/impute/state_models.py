"""State containers for sklearn imputation atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimpleImputerState:
    """Learned fill statistics for dense numeric simple imputation."""

    statistics: NDArray[np.float64]
    valid_features: NDArray[np.int64]
    n_features_in: int
    keep_empty_features: bool


@dataclass(frozen=True)
class MissingIndicatorState:
    """Learned feature selection for missing-value indicator masks."""

    features: NDArray[np.int64]
    n_features_in: int
    missing_only: bool


@dataclass(frozen=True)
class KNNImputerState:
    """Training data and masks for dense numeric k-nearest-neighbor imputation."""

    fit_X: NDArray[np.float64]
    mask_fit_X: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    n_neighbors: int
    weights: str
    keep_empty_features: bool
