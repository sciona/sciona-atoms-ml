"""Ordered, row-aligned prediction sums with explicit contribution counts."""
import numbers
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .masked_operations import _mask, _mask_metadata


def _vector_metadata(value):
    if len(value.shape)!=1 or value.shape[0]<1 or value.dtype!='float64':
        raise ValueError('Nonempty float64 vector metadata required')


def _count_metadata(total,counts):
    _vector_metadata(total)
    if counts.shape!=total.shape or counts.dtype!='int64':
        raise ValueError('Aligned int64 contribution counts required')
    if counts.dim is not None and not counts.dim.is_dimensionless:
        raise ValueError('Contribution counts are dimensionless')


def _state(total,counts):
    total,counts=np.asarray(total),np.asarray(counts)
    if total.dtype!=np.float64 or total.ndim!=1 or not len(total) or not np.isfinite(total).all():
        raise ValueError('Nonempty finite float64 sum required')
    if counts.dtype!=np.int64 or counts.shape!=total.shape or np.any(counts<0):
        raise ValueError('Aligned nonnegative int64 contribution counts required')
    if np.any(total[counts==0]!=0):
        raise ValueError('Uncovered rows must have zero sum')
    return total,counts


def witness_masked_prediction_contribution(predictions: AbstractArray,selection: AbstractArray) -> tuple:
    if len(predictions.shape)!=1 or predictions.shape[0]<1 or predictions.dtype not in ('float32','float64'):
        raise ValueError('Nonempty float32 or float64 prediction metadata required')
    _mask_metadata(selection,predictions.shape[0])
    return (AbstractArray(shape=predictions.shape,dtype='float64',dim=predictions.dim),
            AbstractArray(shape=predictions.shape,dtype='int64'))


@register_atom(witness_masked_prediction_contribution)
def masked_prediction_contribution(predictions: NDArray[np.float32] | NDArray[np.float64],selection: NDArray[np.bool_]) -> tuple[NDArray[np.float64],NDArray[np.int64]]:
    """Select row-aligned predictions and emit zero/one contribution counts.

    Only selected values must be finite. Unselected values are discarded, not
    multiplied by zero, so NaNs cannot contaminate held-out accumulation.
    Empty selections are valid; final coverage validation rejects missing rows.
    No training, grouping or independence claim is inferred from the mask.
    Native float32 predictions are promoted exactly to float64 before summing.
    """
    values=np.asarray(predictions)
    if values.dtype not in (np.dtype('float32'),np.dtype('float64')) or values.ndim!=1 or not len(values):
        raise ValueError('Nonempty float32 or float64 predictions required')
    mask=_mask(selection,len(values),False)
    if not np.isfinite(values[mask]).all():
        raise ValueError('Selected predictions must be finite')
    return np.where(mask,values.astype(np.float64),0.),mask.astype(np.int64)


def witness_merge_prediction_sums(left: AbstractArray,left_counts: AbstractArray,
        right: AbstractArray,right_counts: AbstractArray) -> tuple:
    _count_metadata(left,left_counts);_count_metadata(right,right_counts)
    if left.shape!=right.shape:
        raise ValueError('Prediction sums must align')
    if left.dim is not None and right.dim is not None and not left.dim.is_compatible(right.dim):
        raise ValueError('Prediction sums must have compatible units')
    return (AbstractArray(shape=left.shape,dtype='float64',dim=left.dim or right.dim),
            AbstractArray(shape=left.shape,dtype='int64'))


@register_atom(witness_merge_prediction_sums)
def merge_prediction_sums(left: NDArray[np.float64],left_counts: NDArray[np.int64],
        right: NDArray[np.float64],right_counts: NDArray[np.int64]) -> tuple[NDArray[np.float64],NDArray[np.int64]]:
    """Add aligned sums and counts without mutating inputs or broadcasting.

    A sequential left fold retains caller accumulation order. Rebalancing a
    graph may change floating-point rounding. Counts record contributions,
    not unique model identities; graphs must establish training isolation.
    """
    a,ac=_state(left,left_counts);b,bc=_state(right,right_counts)
    if a.shape!=b.shape:
        raise ValueError('Prediction sums must align')
    if np.any(ac>np.iinfo(np.int64).max-bc):
        raise ValueError('Contribution count overflow')
    with np.errstate(over='ignore',invalid='ignore'):
        result=a+b
    if not np.isfinite(result).all():
        raise ValueError('Prediction sum overflow')
    return result,ac+bc


def _expected(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or not 1<=value<=np.iinfo(np.int64).max:
        raise ValueError('Positive int64 expected coverage required')


def witness_finalize_prediction_mean(total: AbstractArray,counts: AbstractArray,expected_coverage: int) -> AbstractArray:
    _count_metadata(total,counts);_expected(expected_coverage)
    return AbstractArray(shape=total.shape,dtype='float64',dim=total.dim)


@register_atom(witness_finalize_prediction_mean)
def finalize_prediction_mean(total: NDArray[np.float64],counts: NDArray[np.int64],expected_coverage: int) -> NDArray[np.float64]:
    """Divide sums only after every row has the explicit expected coverage.

    This rejects gaps and excess contributions; it does not certify unique
    folds, disjoint training populations or predictive quality. Units persist.
    """
    values,coverage=_state(total,counts);_expected(expected_coverage)
    if not np.all(coverage==expected_coverage):
        raise ValueError('Every row must have exactly the expected coverage')
    return values/coverage
