"""Explicit source neural schedule with preflight of all training partitions."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .xgboost.model_io import _names
from .model_selection.configured_crossfit import shuffled_kfold_masks
from .model_selection.nested_stratified_masks import stratified_inner_masks


def _settings():
    return (5,555,0,1,2,3,4,200,0,10.,50.,10,50,
        *[dict(hidden_units=8,epochs=3000,batch_size=128,seed=seed,learning_rate=.01,epsilon=1e-6) for seed in range(10)])


def witness_neural_ensemble_inputs(features: AbstractArray,targets: AbstractArray,query_features: AbstractArray,feature_names: list) -> tuple:
    if (len(features.shape)!=2 or len(query_features.shape)!=2 or min(features.shape)<1
            or min(query_features.shape)<1 or features.shape[1]!=query_features.shape[1]
            or features.dtype!='float64' or query_features.dtype!='float64'
            or targets.shape!=(features.shape[0],) or targets.dtype!='int64'):
        raise ValueError('Aligned float64 feature matrices and int64 targets required')
    _names(feature_names,features.shape[1])
    if targets.dim is not None and not targets.dim.is_dimensionless:raise ValueError('Dimensionless labels required')
    if features.dim is not None and query_features.dim is not None and not features.dim.is_compatible(query_features.dim):
        raise ValueError('Training and query units must agree')
    return (features,targets,query_features,list(feature_names),AbstractArray(shape=(query_features.shape[0],),dtype='bool'),*_settings())


@register_atom(witness_neural_ensemble_inputs)
def neural_ensemble_inputs(features: NDArray[np.float64],targets: NDArray[np.int64],query_features: NDArray[np.float64],feature_names: list) -> tuple:
    """Validate all five outer/inner partitions before any source model is fit.

    Exposes ordered inputs, ten initialization configurations and fixed fold
    controls. Every outer fit requires at least 200 rows of each binary class.
    No feature preparation, scaling, fitting or population identity inference.
    """
    x,y,q=map(np.asarray,(features,targets,query_features))
    witness_neural_ensemble_inputs(AbstractArray(shape=x.shape,dtype=str(x.dtype)),
        AbstractArray(shape=y.shape,dtype=str(y.dtype)),AbstractArray(shape=q.shape,dtype=str(q.dtype)),feature_names)
    if not np.isfinite(x).all() or not np.isfinite(q).all() or set(y)!={0,1}:
        raise ValueError('Finite features and both binary classes required')
    for fold in range(5):
        outer,_=shuffled_kfold_masks(x,5,555,fold)
        stratified_inner_masks(y,outer,200,0)
    return (x,y,q,list(feature_names),np.ones(len(q),dtype=np.bool_),*_settings())
