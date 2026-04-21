"""State containers for sklearn naive Bayes atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianNBState:
    """Learned dense Gaussian naive Bayes class statistics."""

    classes: NDArray[np.int64]
    class_count: NDArray[np.float64]
    class_prior: NDArray[np.float64]
    theta: NDArray[np.float64]
    var: NDArray[np.float64]
    epsilon: float
    n_features_in: int
