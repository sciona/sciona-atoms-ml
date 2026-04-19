"""State containers for sklearn random projection atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

ProjectionKind = Literal["gaussian", "sparse"]
MatrixLike = NDArray[np.float64] | sp.spmatrix


@dataclass(frozen=True)
class RandomProjectionState:
    """Learned random projection matrix and transform options."""

    components: MatrixLike
    n_components: int
    n_features_in: int
    projection_kind: ProjectionKind
    compute_inverse_components: bool
    inverse_components: NDArray[np.float64] | None = None
    density: float | None = None
    dense_output: bool = False
