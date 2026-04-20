"""State containers for sklearn kernel approximation atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RBFSamplerState:
    """Fitted random Fourier weights for an RBF kernel approximation."""

    random_weights: NDArray[np.float64]
    random_offset: NDArray[np.float64]
    gamma: float
    n_components: int
    n_features_in: int
    random_state: int | None


@dataclass(frozen=True)
class SkewedChi2SamplerState:
    """Fitted random Fourier weights for a skewed chi-square approximation."""

    random_weights: NDArray[np.float64]
    random_offset: NDArray[np.float64]
    skewedness: float
    n_components: int
    n_features_in: int
    random_state: int | None
