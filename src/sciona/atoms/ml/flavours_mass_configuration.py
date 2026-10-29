"""Explicit source configuration; numerical fitting remains in generic atoms."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _constants():
    return (5,1,2,3,4,5,6,555,0,1,2,3,4,
            dict(eta=.05,max_depth=6,min_child_weight=10,subsample=.8,
                 colsample_bytree=.9,seed=1,tree_method='exact',base_score=.5),2800,7,35)


def witness_mass_regression_inputs(features: AbstractArray,query_features: AbstractArray) -> tuple:
    if (features.dtype!='float64' or query_features.dtype!='float64'
            or len(features.shape)!=2 or len(query_features.shape)!=2
            or features.shape[0]<5 or features.shape[1]<1 or query_features.shape[0]<1
            or features.shape[1]!=query_features.shape[1]):
        raise ValueError('Aligned float64 matrices with at least five training rows required')
    if features.dim is not None and query_features.dim is not None and not features.dim.is_compatible(query_features.dim):
        raise ValueError('Training and query feature units must agree')
    return (features,query_features,AbstractArray(shape=(query_features.shape[0],),dtype='bool'),*_constants())


@register_atom(witness_mass_regression_inputs)
def mass_regression_inputs(features: NDArray[np.float64],query_features: NDArray[np.float64]) -> tuple:
    """Expose source 7x5 fold schedule, 2800 rounds and all-query selection.

    Caller supplies ordered numerical features, with matching units and column
    meaning. No feature engineering, population inference, fitting or targets
    are hidden here. Modern exact-tree squared-error configuration matches the
    retained reconstruction; historical engine equivalence is not asserted.
    """
    train,query=np.asarray(features),np.asarray(query_features)
    witness_mass_regression_inputs(AbstractArray(shape=train.shape,dtype=str(train.dtype)),
                                  AbstractArray(shape=query.shape,dtype=str(query.dtype)))
    if np.isinf(train).any() or np.isinf(query).any():
        raise ValueError('Infinite features are unsupported; NaN missing values are allowed')
    return (train,query,np.ones(len(query),dtype=np.bool_),*_constants())
