"""State containers for sklearn manifold atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassicalMDSState:
    """Fitted classical MDS coordinates and source dissimilarities."""

    embedding: NDArray[np.float64]
    dissimilarity_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    n_components: int
    metric: str
    n_features_in: int
