"""State containers for sklearn preprocessing atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KernelCentererState:
    """Learned means for centering kernel matrices in feature space."""

    k_fit_rows: NDArray[np.float64]
    k_fit_all: float
    n_features_in: int


@dataclass(frozen=True)
class MaxAbsScalerState:
    """Learned maximum-absolute-value scale factors for each feature."""

    scale: NDArray[np.float64]
    max_abs: NDArray[np.float64]
    n_features_in: int
    n_samples_seen: int


@dataclass(frozen=True)
class MinMaxScalerState:
    """Learned min/max scale factors for mapping features into a range."""

    min_: NDArray[np.float64]
    scale: NDArray[np.float64]
    data_min: NDArray[np.float64]
    data_max: NDArray[np.float64]
    data_range: NDArray[np.float64]
    feature_range: tuple[float, float]
    n_features_in: int
    n_samples_seen: int
