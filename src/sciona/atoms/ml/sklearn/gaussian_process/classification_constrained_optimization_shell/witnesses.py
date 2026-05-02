"""Ghost witnesses for Gaussian-process classification constrained-optimization shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gpc_constrained_optimization_use_lbfgsb(
    optimizer_name: str,
) -> bool:
    """Describe whether _BinaryGaussianProcessClassifierLaplace uses the internal L-BFGS-B branch."""
    del optimizer_name
    return False


def witness_gpc_constrained_optimization_use_callable(
    optimizer_is_callable: bool,
    optimizer_name: str,
) -> bool:
    """Describe whether _BinaryGaussianProcessClassifierLaplace uses the user-callable optimizer branch."""
    del optimizer_is_callable
    del optimizer_name
    return False


def witness_gpc_constrained_optimization_unknown_optimizer_message(
    optimizer_name: str,
) -> str:
    """Describe the unsupported-optimizer error message."""
    del optimizer_name
    return ""


def witness_gpc_constrained_optimization_result(
    theta_opt: NDArray[np.float64],
    func_min: float,
) -> tuple[NDArray[np.float64], float]:
    """Describe the final constrained-optimization return tuple."""
    del theta_opt
    del func_min
    return np.zeros(1, dtype=np.float64), 0.0
