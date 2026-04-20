"""State containers for sklearn kernel ridge atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KernelRidgeState:
    """Fitted dual coefficients and training data for kernel ridge regression."""

    dual_coef: NDArray[np.float64]
    X_fit: NDArray[np.float64]
    alpha: NDArray[np.float64]
    kernel: str
    gamma: float | None
    degree: float
    coef0: float
    n_features_in: int
