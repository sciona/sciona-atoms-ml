"""Explicit divide-before-sum contributions and final coverage validation."""
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .prediction_accumulation import (
    masked_prediction_contribution,witness_masked_prediction_contribution,
    witness_finalize_prediction_mean,_state,_expected,
)


def _divisor(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not np.isfinite(value) or value<=0:
        raise ValueError('Positive finite dimensionless divisor required')


def witness_divided_prediction_contribution(predictions: AbstractArray,selection: AbstractArray,divisor: float) -> tuple:
    _divisor(divisor)
    return witness_masked_prediction_contribution(predictions,selection)


@register_atom(witness_divided_prediction_contribution)
def divided_prediction_contribution(predictions: NDArray[np.float64],selection: NDArray[np.bool_],divisor: float) -> tuple[NDArray[np.float64],NDArray[np.int64]]:
    """Divide each selected float64 contribution before any ordered summation.

    Counts remain unscaled. Divisor is explicit, not inferred from coverage;
    graph configuration must establish intended weights and normalization.
    """
    _divisor(divisor)
    if np.asarray(predictions).dtype!=np.float64:raise ValueError('Float64 predictions required')
    total,counts=masked_prediction_contribution(predictions,selection)
    with np.errstate(over='raise',invalid='raise',divide='raise'):total=total/divisor
    return total,counts


@register_atom(witness_finalize_prediction_mean)
def finalize_prediction_sum(total: NDArray[np.float64],counts: NDArray[np.int64],expected_coverage: int) -> NDArray[np.float64]:
    """Return a copy of already weighted sums after exact coverage validation.

    No final division: this preserves divide-before-sum rounding. Coverage
    proves counts only, not model uniqueness or training independence.
    """
    values,coverage=_state(total,counts);_expected(expected_coverage)
    if not np.all(coverage==expected_coverage):raise ValueError('Every row must have exactly the expected coverage')
    return values.copy()
