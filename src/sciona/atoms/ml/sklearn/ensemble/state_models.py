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


@dataclass(frozen=True)
class ForestClassifierTargetState:
    """Learned per-output class ordering and class counts for ForestClassifier."""

    classes: tuple[NDArray[np.object_], ...]
    n_classes: tuple[int, ...]
