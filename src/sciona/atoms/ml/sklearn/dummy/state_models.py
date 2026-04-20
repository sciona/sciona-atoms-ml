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
