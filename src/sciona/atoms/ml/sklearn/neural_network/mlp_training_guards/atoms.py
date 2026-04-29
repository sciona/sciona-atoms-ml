"""MLP training-guard helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_fit_require_finite_weights,
    witness_mlp_partial_fit_require_stochastic_solver,
)

WeightArray = NDArray[np.float64]
_STOCHASTIC_SOLVERS = frozenset({"sgd", "adam"})


def _solver_name_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _weight_array_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.size >= 1)


def _weight_sequence_valid(values: object) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    arrays = list(values)
    return bool(arrays and all(_weight_array_valid(array) for array in arrays))


def _all_finite(values: Sequence[WeightArray]) -> bool:
    return all(np.isfinite(np.asarray(array, dtype=np.float64)).all() for array in values)


@register_atom(witness_mlp_partial_fit_require_stochastic_solver)
@icontract.require(lambda solver: _solver_name_valid(solver), "solver must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "guard result must be boolean")
def mlp_partial_fit_require_stochastic_solver(
    *,
    solver: str,
) -> bool:
    """Enforce sklearn's partial_fit restriction to stochastic MLP solvers."""
    if solver not in _STOCHASTIC_SOLVERS:
        raise AttributeError(
            "partial_fit is only available for stochastic optimizers. "
            f"{solver} is not stochastic."
        )
    return True


@register_atom(witness_mlp_fit_require_finite_weights)
@icontract.require(lambda coefs: _weight_sequence_valid(coefs), "coefs must be a nonempty sequence of numeric weight arrays")
@icontract.require(lambda intercepts: _weight_sequence_valid(intercepts), "intercepts must be a nonempty sequence of numeric bias arrays")
@icontract.require(lambda coefs, intercepts: len(coefs) == len(intercepts), "coefs and intercepts must have matching layer counts")
@icontract.ensure(lambda result: isinstance(result, bool), "guard result must be boolean")
def mlp_fit_require_finite_weights(
    coefs: Sequence[WeightArray],
    intercepts: Sequence[WeightArray],
) -> bool:
    """Enforce sklearn's post-fit finite-parameter guard for MLP training."""
    if not _all_finite(coefs) or not _all_finite(intercepts):
        raise ValueError(
            "Solver produced non-finite parameter weights. "
            "The input data may contain large values and need to be preprocessed."
        )
    return True
