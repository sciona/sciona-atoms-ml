"""State containers for sklearn neighbors atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NeighborsGraphTransformerState:
    """Fitted dense neighbor-graph transformer state."""

    training_data: NDArray[np.float64]
    mode: str
    n_neighbors: int | None
    radius: float | None
    metric: str
    p: float
    transformer_kind: str
    n_features_in: int
