"""State containers for sklearn MLP optimizer helper atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


LrScheduleName = Literal["constant", "adaptive", "invscaling"]
TensorTuple = tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class SgdOptimizerState:
    """Persistent state for sklearn's SGDOptimizer update kernel."""

    learning_rate_init: float
    learning_rate: float
    lr_schedule: LrScheduleName
    momentum: float
    nesterov: bool
    power_t: float
    velocities: TensorTuple


@dataclass(frozen=True)
class AdamOptimizerState:
    """Persistent state for sklearn's AdamOptimizer update kernel."""

    learning_rate_init: float
    learning_rate: float
    beta_1: float
    beta_2: float
    epsilon: float
    t: int
    ms: TensorTuple
    vs: TensorTuple
