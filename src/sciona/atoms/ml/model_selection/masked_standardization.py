"""Selected-row standardization using existing scaler atoms and numeric state."""
import numbers
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom
from .masked_operations import _mask,_mask_metadata
from ..xgboost.model_io import _names
from ..sklearn.preprocessing.atoms import standard_scaler_fit,standard_scaler_transform
from ..sklearn.preprocessing.state_models import StandardScalerState


def _metadata(features,names):
    if len(features.shape)!=2 or min(features.shape)<1 or features.dtype!='float64':
        raise ValueError('Nonempty float64 matrix metadata required')
    _names(names,features.shape[1])


def witness_fit_masked_standardization(features: AbstractArray,feature_names: list,selection: AbstractArray) -> dict:
    _metadata(features,feature_names);_mask_metadata(selection,features.shape[0])
    return dict(format='sciona.standardization.abstract.v1',feature_names=list(feature_names),input_dim=features.dim)


@register_atom(witness_fit_masked_standardization)
def fit_masked_standardization(features: NDArray[np.float64],feature_names: list,selection: NDArray[np.bool_]) -> dict:
    """Fit mean/variance on selected ordered rows; return portable numeric state.

    Reuses the registered standard-scaler fit with centering and population
    variance scaling. Constant features retain backend scale-one semantics.
    Selected inputs must be finite; unselected values cannot affect fitting.
    Caller owns selection identity, column meaning and consistent feature units.
    """
    x=np.asarray(features)
    if x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1:
        raise ValueError('Nonempty float64 feature matrix required')
    _names(feature_names,x.shape[1]);mask=_mask(selection,len(x))
    selected=x[mask]
    if not np.isfinite(selected).all():raise ValueError('Selected features must be finite')
    with threadpool_limits(limits=1):state=standard_scaler_fit(selected)
    if any(not np.isfinite(v).all() for v in (state.mean,state.var,state.scale)):
        raise ValueError('Nonfinite fitted standardization state')
    return dict(format='sciona.standardization.v1',feature_names=list(feature_names),
                mean=state.mean.tolist(),variance=state.var.tolist(),scale=state.scale.tolist(),
                samples=int(mask.sum()))


def witness_apply_standardization(features: AbstractArray,feature_names: list,state: dict) -> AbstractArray:
    _metadata(features,feature_names)
    if not isinstance(state,dict) or state.get('format')!='sciona.standardization.abstract.v1' or state.get('feature_names')!=feature_names:
        raise ValueError('Matching abstract standardization state required')
    dim=state.get('input_dim')
    if features.dim is not None and dim is not None and not features.dim.is_compatible(dim):
        raise ValueError('Features must use fitted units')
    return AbstractArray(shape=features.shape,dtype='float64',dim=DimensionalSignature())


@register_atom(witness_apply_standardization)
def apply_standardization(features: NDArray[np.float64],feature_names: list,state: dict) -> NDArray[np.float64]:
    """Apply the fitted scaler without refitting or mutating input features.

    Requires finite float64 inputs and exact ordered feature names. Numeric
    JSON state reconstructs the existing scaler atom's state model; no pickle
    or executable deserialization. Output features are dimensionless.
    """
    x=np.asarray(features)
    if x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Nonempty finite float64 features required')
    _names(feature_names,x.shape[1])
    if (not isinstance(state,dict) or set(state)!={'format','feature_names','mean','variance','scale','samples'}
            or state['format']!='sciona.standardization.v1' or state['feature_names']!=feature_names):
        raise ValueError('Matching numeric standardization state required')
    samples=state['samples']
    if isinstance(samples,(bool,np.bool_)) or not isinstance(samples,numbers.Integral) or samples<1:
        raise ValueError('Positive fitted sample count required')
    mean,var,scale=[np.asarray(state[k],dtype=np.float64) for k in ('mean','variance','scale')]
    if any(v.shape!=(x.shape[1],) or not np.isfinite(v).all() for v in (mean,var,scale)) or np.any(var<0) or np.any(scale<=0):
        raise ValueError('Aligned finite scaler vectors with nonnegative variance and positive scale required')
    restored=StandardScalerState(mean=mean,var=var,scale=scale,n_samples_seen=np.full(x.shape[1],samples,dtype=np.float64),
        with_mean=True,with_std=True,n_features_in=x.shape[1])
    with threadpool_limits(limits=1),np.errstate(over='raise',invalid='raise',divide='raise'):
        result=standard_scaler_transform(x,restored)
    if not np.isfinite(result).all():raise ValueError('Nonfinite standardized features')
    return result
