"""Explicit source input validation and feature routing, without hidden fitting."""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .flavours_feature_views import witness_positive_longitudinal_kinematics
from .model_selection.configured_crossfit import shuffled_kfold_masks
from .model_selection.nested_stratified_masks import stratified_inner_masks


def _physical_metadata(values):
    if not isinstance(values,(list,tuple)) or len(values)!=6:raise ValueError('Six ordered physical arrays required')
    base,p,pt,parent,distance,time=values
    if len(base.shape)!=2 or base.shape[0]<1 or base.shape[1]!=46 or base.dtype!='float64':
        raise ValueError('Finite float64 base matrix with 46 caller-ordered columns required')
    if p.shape!=(base.shape[0],3):raise ValueError('Source requires three aligned daughters')
    witness_positive_longitudinal_kinematics(p,pt,parent,distance,time)


def witness_training_prediction_inputs(training_inputs: list,labels: AbstractArray,mass: AbstractArray,
        query_inputs: list,excluded_column: int) -> tuple:
    _physical_metadata(training_inputs);_physical_metadata(query_inputs)
    rows=training_inputs[0].shape[0]
    if labels.shape!=(rows,) or labels.dtype!='int64' or mass.shape!=(rows,) or mass.dtype!='float64':
        raise ValueError('Aligned int64 labels and float64 training mass required')
    if isinstance(excluded_column,(bool,np.bool_)) or not isinstance(excluded_column,(int,np.integer)) or not 0<=excluded_column<46:
        raise ValueError('Explicit excluded base-column position required')
    return (*training_inputs,*query_inputs,labels,mass,int(excluded_column))


@register_atom(witness_training_prediction_inputs)
def training_prediction_inputs(training_inputs: list,labels: np.ndarray,mass: np.ndarray,
        query_inputs: list,excluded_column: int) -> tuple:
    """Validate source input shapes and every neural partition before training.

    No evaluation controls, feature-name inference or fitting. Caller owns
    physical units, ordered base-column semantics and population disjointness.
    Only training labels and mass are accepted. Outputs preserve row order.
    """
    if any(not isinstance(v,(list,tuple)) or len(v)!=6 for v in (training_inputs,query_inputs)):
        raise ValueError('Six ordered physical arrays per population required')
    training=list(map(np.asarray,training_inputs));query=list(map(np.asarray,query_inputs))
    y,m=np.asarray(labels),np.asarray(mass)
    abstract=lambda v:AbstractArray(shape=v.shape,dtype=str(v.dtype))
    witness_training_prediction_inputs(list(map(abstract,training)),abstract(y),abstract(m),list(map(abstract,query)),excluded_column)
    if any(not np.isfinite(v).all() for v in (*training,*query,m)) or set(y)!={0,1}:
        raise ValueError('Finite inputs and both binary classes required')
    for population in (training,query):
        _,p,pt,parent,distance,time=population
        if np.any(p<0) or np.any(pt<0) or np.any(pt>p) or np.any(parent<0) or np.any(distance<=0) or np.any(time<=0):
            raise ValueError('Invalid physical magnitudes')
    for fold in range(5):
        outer,_=shuffled_kfold_masks(training[0],5,555,fold)
        stratified_inner_masks(y,outer,200,0)
    return (*training,*query,y,m,int(excluded_column))


def _view_metadata(views,key,width):
    if not isinstance(views,dict) or key not in views:raise ValueError('Required prepared view is missing')
    value=views[key]
    if len(value.shape)!=2 or value.shape[0]<1 or value.shape[1]!=width or value.dtype!='float64':
        raise ValueError('Invalid prepared feature view')
    return value


def witness_regression_view_pair(training: dict,query: dict) -> tuple:
    return (_view_metadata(training,'regression',50),_view_metadata(query,'regression',50),[f'column_{i}' for i in range(50)])


@register_atom(witness_regression_view_pair)
def regression_view_pair(training: dict,query: dict) -> tuple:
    """Route the two 50-column regression matrices with opaque positional names.

    Names identify column positions only, not dataset semantics. No reordering,
    feature selection or transformation; upstream assembly owns validation.
    """
    return witness_regression_view_pair(training,query)


def witness_classifier_neural_view_pair(training: dict,query: dict) -> tuple:
    names={}
    for key,width in [('base',46),('restricted',45),('proxy',47),('corrected',50)]:
        _view_metadata(training,key,width);_view_metadata(query,key,width)
        names[key]=[f'column_{i}' for i in range(width)]
    return training,query,names,training['corrected'],query['corrected'],names['corrected']


@register_atom(witness_classifier_neural_view_pair)
def classifier_neural_view_pair(training: dict,query: dict) -> tuple:
    """Route classifier views and corrected neural matrices without fitting.

    Positional names preserve exact constructed column order and do not infer
    semantic feature names. Corrected mass remains non-nested downstream.
    """
    return witness_classifier_neural_view_pair(training,query)
