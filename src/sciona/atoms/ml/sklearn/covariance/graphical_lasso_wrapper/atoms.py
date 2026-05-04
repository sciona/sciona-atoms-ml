"""graphical_lasso wrapper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Any

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_constructor_kwargs,
    witness_graphical_lasso_return_values,
)


Matrix = NDArray[np.float64]


def _positive_finite_float(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_bool_or_int(value: object) -> bool:
    return bool(
        isinstance(value, (bool, int, np.integer))
        and not isinstance(value, np.bool_)
        and int(value) >= 0
    )


def _positive_int(value: object) -> bool:
    return bool(
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and int(value) >= 1
    )


def _mode_valid(value: object) -> bool:
    return value in {"cd", "lars"}


def _finite_square_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _costs_valid(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        if not all(_positive_finite_float(x) or (isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))) for x in item):
            return False
    return True


def _constructor_kwargs_valid(
    result: object,
    alpha: float,
    mode: str,
    tol: float,
    enet_tol: float,
    max_iter: int,
    verbose: bool | int,
    eps: float,
) -> bool:
    return bool(
        isinstance(result, dict)
        and result
        == {
            "alpha": float(alpha),
            "mode": mode,
            "covariance": "precomputed",
            "tol": float(tol),
            "enet_tol": float(enet_tol),
            "max_iter": int(max_iter),
            "verbose": verbose,
            "eps": float(eps),
            "assume_centered": True,
        }
    )


def _return_inputs_valid(
    covariance: object,
    precision: object,
    return_costs: object,
    costs: object,
    return_n_iter: object,
    n_iter: object,
) -> bool:
    if not (_finite_square_matrix(covariance) and _finite_square_matrix(precision)):
        return False
    cov = np.asarray(covariance, dtype=np.float64)
    prec = np.asarray(precision, dtype=np.float64)
    if cov.shape != prec.shape:
        return False
    if not isinstance(return_costs, bool) or not isinstance(return_n_iter, bool):
        return False
    if return_costs and not _costs_valid(costs):
        return False
    if not return_costs and costs is not None:
        return False
    if return_n_iter and not _positive_int(n_iter):
        return False
    if not return_n_iter and n_iter is not None:
        return False
    return True


def _return_values_valid(
    result: object,
    covariance: Matrix,
    precision: Matrix,
    return_costs: bool,
    costs: tuple[tuple[float, float], ...] | None,
    return_n_iter: bool,
    n_iter: int | None,
) -> bool:
    if not _return_inputs_valid(covariance, precision, return_costs, costs, return_n_iter, n_iter):
        return False
    if not isinstance(result, tuple):
        return False
    expected: list[object] = [
        np.asarray(covariance, dtype=np.float64),
        np.asarray(precision, dtype=np.float64),
    ]
    if return_costs:
        assert costs is not None
        expected.append(costs)
    if return_n_iter:
        assert n_iter is not None
        expected.append(int(n_iter))
    if len(result) != len(expected):
        return False
    if not np.array_equal(np.asarray(result[0], dtype=np.float64), expected[0]):
        return False
    if not np.array_equal(np.asarray(result[1], dtype=np.float64), expected[1]):
        return False
    if return_costs and result[2] != expected[2]:
        return False
    if return_n_iter:
        index = 3 if return_costs else 2
        if int(result[index]) != expected[index]:
            return False
    return True


@register_atom(witness_graphical_lasso_constructor_kwargs)
@icontract.require(lambda alpha: _positive_finite_float(alpha), "alpha must be a positive finite scalar")
@icontract.require(lambda mode: isinstance(mode, str) and _mode_valid(mode), "mode must be 'cd' or 'lars'")
@icontract.require(lambda tol: _positive_finite_float(tol), "tol must be a positive finite scalar")
@icontract.require(lambda enet_tol: _positive_finite_float(enet_tol), "enet_tol must be a positive finite scalar")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda verbose: _nonnegative_bool_or_int(verbose), "verbose must be bool or a nonnegative integer")
@icontract.require(lambda eps: _positive_finite_float(eps), "eps must be a positive finite scalar")
@icontract.ensure(
    lambda result, alpha, mode, tol, enet_tol, max_iter, verbose, eps: _constructor_kwargs_valid(
        result, alpha, mode, tol, enet_tol, max_iter, verbose, eps
    ),
    "result must match the GraphicalLasso constructor kwargs used by graphical_lasso",
)
def graphical_lasso_constructor_kwargs(
    alpha: float,
    mode: str = "cd",
    tol: float = 1e-4,
    enet_tol: float = 1e-4,
    max_iter: int = 100,
    verbose: bool | int = False,
    eps: float = np.finfo(np.float64).eps,
) -> dict[str, object]:
    """Resolve the fixed GraphicalLasso constructor kwargs used by graphical_lasso."""
    return {
        "alpha": float(alpha),
        "mode": mode,
        "covariance": "precomputed",
        "tol": float(tol),
        "enet_tol": float(enet_tol),
        "max_iter": int(max_iter),
        "verbose": verbose,
        "eps": float(eps),
        "assume_centered": True,
    }


@register_atom(witness_graphical_lasso_return_values)
@icontract.require(
    lambda covariance, precision, return_costs=False, costs=None, return_n_iter=False, n_iter=None: _return_inputs_valid(
        covariance, precision, return_costs, costs, return_n_iter, n_iter
    ),
    "wrapper return inputs must be finite, shape-compatible, and match the selected output flags",
)
@icontract.ensure(
    lambda result, covariance, precision, return_costs=False, costs=None, return_n_iter=False, n_iter=None: _return_values_valid(
        result, covariance, precision, return_costs, costs, return_n_iter, n_iter
    ),
    "result must match graphical_lasso's public tuple packaging",
)
def graphical_lasso_return_values(
    covariance: Matrix,
    precision: Matrix,
    return_costs: bool = False,
    costs: tuple[tuple[float, float], ...] | None = None,
    return_n_iter: bool = False,
    n_iter: int | None = None,
) -> tuple[object, ...]:
    """Package graphical_lasso's public return tuple from fitted estimator outputs."""
    output: list[Any] = [
        np.asarray(covariance, dtype=np.float64),
        np.asarray(precision, dtype=np.float64),
    ]
    if return_costs:
        assert costs is not None
        output.append(costs)
    if return_n_iter:
        assert n_iter is not None
        output.append(int(n_iter))
    return tuple(output)
