"""Explicit six-input nonlinear score blend from the source method."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom

ProbabilityVector = NDArray[np.float32] | NDArray[np.float64]


def witness_six_model_power_blend(first: AbstractArray,second: AbstractArray,third: AbstractArray,
        fourth: AbstractArray,fifth: AbstractArray,sixth: AbstractArray) -> AbstractArray:
    values=(first,second,third,fourth,fifth,sixth)
    if len(first.shape)!=1 or first.shape[0]<1 or any(v.shape!=first.shape or v.dtype not in ('float32','float64') for v in values):
        raise ValueError('Aligned floating-point probability vectors required')
    if any(v.dim is not None and not v.dim.is_dimensionless for v in values):
        raise ValueError('Probabilities must be dimensionless')
    return AbstractArray(shape=first.shape,dtype='float64',dim=DimensionalSignature())


@register_atom(witness_six_model_power_blend)
def six_model_power_blend(first: ProbabilityVector,second: ProbabilityVector,third: ProbabilityVector,
        fourth: ProbabilityVector,fifth: ProbabilityVector,sixth: ProbabilityVector) -> NDArray[np.float64]:
    """Blend six aligned probability vectors using the fixed source powers.

    Inputs are role-ordered; model family and task domain are not inferred.
    Converts native float32 probabilities to float64 before arithmetic. The
    output is an unnormalized ranking score, not a calibrated probability.
    No tuning, clipping, training, sorting or population reordering occurs.
    """
    values=[np.asarray(v) for v in (first,second,third,fourth,fifth,sixth)]
    witness_six_model_power_blend(*(AbstractArray(shape=v.shape,dtype=str(v.dtype)) for v in values))
    if any(not np.isfinite(v).all() or np.any((v<0)|(v>1)) for v in values):
        raise ValueError('Finite input probabilities in [0,1] required')
    a,b,c,d,e,n=[v.astype(np.float64,copy=False) for v in values]
    low=np.sqrt(a)*np.square(b)/2
    high=(c*d**.85*e**.01+.85*d*n**900+2*e**1000)/3.85
    return .5*high**3.9+.2*low**.6+.0001*low**.01*high**.2
