"""State containers for sklearn isotonic regression atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

OutOfBoundsMode = Literal["nan", "clip", "raise"]


@dataclass(frozen=True)
class IsotonicRegressionState:
    """Threshold representation of a fitted isotonic regression model."""

    x_thresholds: NDArray[np.float64]
    y_thresholds: NDArray[np.float64]
    x_min: float
    x_max: float
    increasing: bool
    out_of_bounds: OutOfBoundsMode
