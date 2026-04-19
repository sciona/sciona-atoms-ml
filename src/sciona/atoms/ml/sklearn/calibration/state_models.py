"""State containers for sklearn calibration atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SigmoidCalibrationState:
    """Fitted Platt sigmoid calibration parameters."""

    a: float
    b: float


@dataclass(frozen=True)
class TemperatureScalingState:
    """Fitted inverse temperature scaling parameter."""

    beta: float


@dataclass(frozen=True)
class CalibratedClassifierCVState:
    """Fitted calibrated classifier CV estimator and metadata."""

    estimator: object
    classes: NDArray[np.object_]
    method: str
    ensemble: bool | str
    n_features_in: int | None
