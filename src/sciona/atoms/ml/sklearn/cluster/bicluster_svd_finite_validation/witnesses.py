"""Ghost witnesses for deterministic biclustering SVD finite-output checks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_bicluster_svd_checked_u(
    u: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Witness for validating the finite left singular-vector matrix."""


def witness_bicluster_svd_checked_vt(
    vt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Witness for validating the finite right singular-vector matrix."""
