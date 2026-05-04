"""Ghost witnesses for graphical_lasso wrapper atoms."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


def witness_graphical_lasso_constructor_kwargs(
    alpha: float,
    mode: str = "cd",
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    max_iter: int = 100,
    verbose: bool | int = False,
    eps: float = np.finfo(np.float64).eps,
) -> AbstractArray:
    """Describe the fixed GraphicalLasso constructor kwargs used by graphical_lasso."""
    del alpha, mode, tol, enet_tol, max_iter, verbose, eps
    return AbstractArray(shape=(), dtype="object")


def witness_graphical_lasso_return_values(
    covariance: AbstractArray,
    precision: AbstractArray,
    return_costs: bool = False,
    costs: AbstractArray | None = None,
    return_n_iter: bool = False,
    n_iter: int | None = None,
) -> AbstractArray:
    """Describe the public graphical_lasso return tuple from fitted estimator outputs."""
    del covariance, precision, return_costs, costs, return_n_iter, n_iter
    return AbstractArray(shape=(), dtype="object")
