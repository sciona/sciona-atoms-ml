"""State containers for sklearn dummy estimator atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DummyRegressorState:
    """Fitted constant prediction state for a dummy regressor."""

    constant: NDArray[np.float64]
    n_outputs: int
    strategy: str
    quantile: float | None


@dataclass(frozen=True)
class DummyClassifierState:
    """Fitted class-prior state for a dummy classifier."""

    classes: tuple[NDArray[np.float64], ...]
    class_prior: tuple[NDArray[np.float64], ...]
    n_classes: tuple[int, ...]
    n_outputs: int
    strategy: str
    constant: tuple[float, ...] | None
