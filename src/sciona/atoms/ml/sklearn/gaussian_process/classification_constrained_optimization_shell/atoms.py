"""Gaussian-process classification constrained-optimization shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_constrained_optimization_result,
    witness_gpc_constrained_optimization_unknown_optimizer_message,
    witness_gpc_constrained_optimization_use_callable,
    witness_gpc_constrained_optimization_use_lbfgsb,
)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_scalar(value: object) -> bool:
    return bool(np.isscalar(value) and not isinstance(value, bool) and np.isfinite(float(value)))


@register_atom(witness_gpc_constrained_optimization_use_lbfgsb)
@icontract.require(lambda optimizer_name: _nonempty_string(optimizer_name), "optimizer_name must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def gpc_constrained_optimization_use_lbfgsb(
    optimizer_name: str,
) -> bool:
    """Decide whether _BinaryGaussianProcessClassifierLaplace._constrained_optimization uses sklearn's L-BFGS-B branch."""
    return optimizer_name == "fmin_l_bfgs_b"


@register_atom(witness_gpc_constrained_optimization_use_callable)
@icontract.require(lambda optimizer_is_callable: _bool(optimizer_is_callable), "optimizer_is_callable must be boolean")
@icontract.require(lambda optimizer_name: _nonempty_string(optimizer_name), "optimizer_name must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def gpc_constrained_optimization_use_callable(
    optimizer_is_callable: bool,
    optimizer_name: str,
) -> bool:
    """Decide whether _BinaryGaussianProcessClassifierLaplace._constrained_optimization uses the user-callable optimizer branch."""
    return bool(not gpc_constrained_optimization_use_lbfgsb(optimizer_name) and optimizer_is_callable)


@register_atom(witness_gpc_constrained_optimization_unknown_optimizer_message)
@icontract.require(lambda optimizer_name: _nonempty_string(optimizer_name), "optimizer_name must be a nonempty string")
@icontract.ensure(
    lambda result, optimizer_name: isinstance(result, str) and result == f"Unknown optimizer {optimizer_name}.",
    "result must match sklearn's unsupported-optimizer error message",
)
def gpc_constrained_optimization_unknown_optimizer_message(
    optimizer_name: str,
) -> str:
    """Format sklearn's unsupported-optimizer error message."""
    return f"Unknown optimizer {optimizer_name}."


@register_atom(witness_gpc_constrained_optimization_result)
@icontract.require(lambda theta_opt: _finite_vector(theta_opt), "theta_opt must be a finite nonempty vector")
@icontract.require(lambda func_min: _finite_scalar(func_min), "func_min must be a finite scalar")
@icontract.ensure(
    lambda result, theta_opt, func_min: isinstance(result, tuple)
    and len(result) == 2
    and np.array_equal(np.asarray(result[0], dtype=np.float64), np.asarray(theta_opt, dtype=np.float64))
    and float(result[1]) == float(func_min),
    "result must preserve theta_opt and func_min in sklearn's return order",
)
def gpc_constrained_optimization_result(
    theta_opt: NDArray[np.float64],
    func_min: float,
) -> tuple[NDArray[np.float64], float]:
    """Package _BinaryGaussianProcessClassifierLaplace._constrained_optimization's final return tuple."""
    return np.asarray(theta_opt, dtype=np.float64), float(func_min)
