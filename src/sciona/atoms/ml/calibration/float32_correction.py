"""Explicit float32 application boundary for source estimator predictions."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_apply_float32_offsets(predictions, probabilities, offsets, threshold):
    if len(predictions.shape)!=1 or predictions.dtype!='float32' or probabilities.shape!=predictions.shape or probabilities.dtype!='float32' or offsets.shape!=(2,) or offsets.dtype!='float64':
        raise ValueError('Aligned float32 predictions/probabilities and two float64 offsets required')
    for x in (probabilities,threshold):
        if x.dim is not None and not x.dim.is_dimensionless:
            raise ValueError('Probability controls must be dimensionless')
    if predictions.dim is not None and offsets.dim is not None and not predictions.dim.is_compatible(offsets.dim):
        raise ValueError('Prediction and offset units differ')
    return AbstractArray(shape=predictions.shape,dtype='float32',dim=predictions.dim or offsets.dim)


@register_atom(witness_apply_float32_offsets)
def apply_float32_offsets(predictions: NDArray[np.float32], probabilities: NDArray[np.float32], offsets: NDArray[np.float64], threshold: float) -> NDArray[np.float32]:
    """Apply signed conditional offsets with explicit float32 scalar arithmetic.

    Above threshold add offsets[0]; below subtract offsets[1]; equality unchanged.
    Unlike float64 correction, offsets round to float32 before arithmetic. This
    matches source pandas operations on estimator float32 predictions. Reject
    nonfinite inputs, cast overflow and result overflow; never clip or impute.
    """
    p,q,o=map(np.asarray,(predictions,probabilities,offsets))
    if p.ndim!=1 or p.dtype!=np.float32 or q.shape!=p.shape or q.dtype!=np.float32 or o.shape!=(2,) or o.dtype!=np.float64:
        raise ValueError('Prediction, probability and offset contracts differ')
    if not all(np.isfinite(x).all() for x in (p,q,o)) or ((q<0)|(q>1)).any():
        raise ValueError('Finite predictions/offsets and probabilities in [0,1] required')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Explicit finite threshold in [0,1] required')
    with np.errstate(over='ignore',invalid='ignore'):
        narrowed=o.astype(np.float32)
        result=np.where(q>threshold,p+narrowed[0],p)
        result=np.where(q<threshold,result-narrowed[1],result)
    if not np.isfinite(narrowed).all() or not np.isfinite(result).all():
        raise ValueError('Float32 correction overflow')
    return result
