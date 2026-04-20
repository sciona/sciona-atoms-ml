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


@dataclass(frozen=True)
class PolynomialCountSketchState:
    """Fitted hash tables for a polynomial Tensor Sketch approximation."""

    index_hash: NDArray[np.int64]
    bit_hash: NDArray[np.int64]
    gamma: float
    degree: int
    coef0: float
    n_components: int
    n_features_in: int
    random_state: int | None


@dataclass(frozen=True)
class NystroemState:
    """Fitted basis and normalization matrix for a Nystroem kernel map."""

    components: NDArray[np.float64]
    component_indices: NDArray[np.int64]
    normalization: NDArray[np.float64]
    kernel: str
    kernel_params: dict[str, float]
    n_components: int
    n_features_in: int
    random_state: int | None
    n_jobs: int | None
