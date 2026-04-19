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
