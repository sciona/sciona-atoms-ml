"""Ghost witnesses for binary Gaussian-process classification optimizer-bookkeeping atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gpc_restart_bounds(
    bounds: NDArray[np.float64],
    *,
    n_restarts_optimizer: int = 0,
) -> NDArray[np.float64]:
    """Describe validated restart bounds for binary GPC optimization."""
    del n_restarts_optimizer
    return np.asarray(bounds, dtype=np.float64)


def witness_gpc_restart_thetas(
    bounds: NDArray[np.float64],
    *,
    n_restarts_optimizer: int,
    random_state: int | np.random.RandomState | None = 0,
) -> NDArray[np.float64]:
    """Describe binary GPC restart theta draws."""
    del bounds
    del random_state
    return np.zeros((int(n_restarts_optimizer), 1), dtype=np.float64)


def witness_gpc_select_best_optimum(
    candidate_thetas: NDArray[np.float64],
    objective_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Describe the selected best optimizer result."""
    del objective_values
    thetas = np.asarray(candidate_thetas, dtype=np.float64)
    return thetas[0], 0.0
