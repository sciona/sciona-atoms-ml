"""State containers for sklearn ensemble atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BaggingClassifierTargetState:
    """Learned class ordering and class count for BaggingClassifier."""

    classes: NDArray[np.object_]
    n_classes: int
