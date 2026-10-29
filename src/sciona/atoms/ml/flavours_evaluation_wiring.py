"""Explicit query/control routing and source diagnostic contracts; no fitting."""
import numbers
import numpy as np
from sciona.ghost.abstract import AbstractArray,AbstractScalar
from sciona.ghost.registry import register_atom
from .flavours_pipeline_wiring import _physical_metadata

KEYS={'agreement_a','agreement_b','correlation','weights_a','weights_b','correlation_mass','quality'}


def _vector(value,probability=False):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Nonempty finite vector required')
    if probability and np.any((a<0)|(a>1)):raise ValueError('Scores must lie in [0,1]')
    return a


def witness_evaluation_inputs(query_inputs: list,controls: dict,labels: AbstractArray) -> tuple:
    if not isinstance(controls,dict) or set(controls)!=KEYS:raise ValueError('Exactly the required evaluation controls must be supplied')
    populations=[query_inputs,controls['agreement_a'],controls['agreement_b'],controls['correlation']]
    for values in populations:_physical_metadata(values)
    sizes=[values[0].shape[0] for values in populations]
    if labels.dtype!='int64' or len(labels.shape)!=1 or labels.shape[0]<1:raise ValueError('Int64 label vector required')
    for key,rows in [('weights_a',sizes[1]),('weights_b',sizes[2]),('correlation_mass',sizes[3]),('quality',labels.shape[0])]:
        if controls[key].shape!=(rows,):raise ValueError('Evaluation metadata must align with its population')
    if sizes[3]<200:raise ValueError('At least 200 correlation rows required')
    merged=[AbstractArray(shape=(sum(sizes),*v.shape[1:]),dtype='float64',dim=v.dim) for v in query_inputs]
    return merged,sizes,controls['weights_a'],controls['weights_b'],controls['correlation_mass'],controls['quality'],200,50


@register_atom(witness_evaluation_inputs)
def evaluation_inputs(query_inputs: list,controls: dict,labels: np.ndarray) -> tuple:
    """Concatenate query then three controls, retaining evaluation-only arrays.

    Weights, quality and control masses are never appended to physical feature
    matrices. Validate their boundary contract before training; do not compute
    placeholder metrics or infer population identity. Caller owns disjointness.
    """
    if not isinstance(controls,dict) or set(controls)!=KEYS:raise ValueError('Exactly the required evaluation controls must be supplied')
    raw=[query_inputs,controls['agreement_a'],controls['agreement_b'],controls['correlation']]
    if any(not isinstance(v,(list,tuple)) or len(v)!=6 for v in raw):raise ValueError('Six physical arrays per population required')
    populations=[list(map(np.asarray,v)) for v in raw]
    abstract=lambda v:AbstractArray(shape=v.shape,dtype=str(v.dtype))
    y=np.asarray(labels);metadata={k:_vector(controls[k]) for k in ('weights_a','weights_b','correlation_mass','quality')}
    abstract_controls={k:list(map(abstract,populations[i])) for i,k in enumerate(('agreement_a','agreement_b','correlation'),1)}
    abstract_controls.update({k:abstract(v) for k,v in metadata.items()})
    witness_evaluation_inputs(list(map(abstract,populations[0])),abstract_controls,abstract(y))
    if not set(y).issubset({0,1}) or set(y[metadata['quality']>.4])!={0,1}:
        raise ValueError('Both binary classes required after strict quality selection')
    for key in ('weights_a','weights_b'):
        w=metadata[key]
        if np.any(w<0) or not np.isfinite(w.sum()) or w.sum()<=0:raise ValueError('Nonnegative weights with finite positive sum required')
    for values in populations:
        if any(not np.isfinite(v).all() for v in values):raise ValueError('Finite physical populations required')
        _,p,pt,parent,distance,time=values
        if np.any(p<0) or np.any(pt<0) or np.any(pt>p) or np.any(parent<0) or np.any(distance<=0) or np.any(time<=0):raise ValueError('Invalid physical magnitudes')
    sizes=[len(v[0]) for v in populations]
    merged=[np.concatenate([v[i] for v in populations],axis=0) for i in range(6)]
    return merged,sizes,metadata['weights_a'],metadata['weights_b'],metadata['correlation_mass'],metadata['quality'],200,50


def _sizes(sizes):
    if not isinstance(sizes,(list,tuple)) or len(sizes)!=4 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Integral) or v<1 for v in sizes):
        raise ValueError('Four positive population sizes required')


def witness_split_population_scores(scores: AbstractArray,sizes: list) -> tuple:
    _sizes(sizes)
    if scores.shape!=(sum(sizes),) or scores.dtype!='float64':raise ValueError('Aligned float64 population scores required')
    return tuple(AbstractArray(shape=(n,),dtype='float64') for n in sizes)


@register_atom(witness_split_population_scores)
def split_population_scores(scores: np.ndarray,sizes: list) -> tuple:
    """Split ordered predictions without sorting or changing values."""
    _sizes(sizes);values=_vector(scores,True)
    if len(values)!=sum(sizes):raise ValueError('Population sizes must account for every score')
    return tuple(np.split(values,np.cumsum(sizes)[:-1]))


def witness_discrimination_inputs(labels: AbstractArray,scores: AbstractArray,quality: AbstractArray) -> tuple:
    if len(labels.shape)!=1 or labels.shape[0]<1 or scores.shape!=labels.shape or quality.shape!=labels.shape:
        raise ValueError('Aligned label, score and quality vectors required')
    raise ValueError('Quality selection has data-dependent row count; materialized arrays required')


@register_atom(witness_discrimination_inputs)
def discrimination_inputs(labels: np.ndarray,scores: np.ndarray,quality: np.ndarray) -> tuple:
    """Select quality strictly above .4 for diagnostic AUC; never feed fitting."""
    y=_vector(labels);prediction=_vector(scores,True);q=_vector(quality)
    if y.shape!=prediction.shape or q.shape!=y.shape or not set(y).issubset({0,1}):raise ValueError('Aligned binary labels, quality and scores required')
    selected=q>.4
    if set(y[selected])!={0,1}:raise ValueError('Both selected classes required')
    return y[selected],prediction[selected]


def witness_diagnostic_report(query_scores: AbstractArray,ks: AbstractScalar,cvm: AbstractScalar,auc: AbstractScalar,sizes: list) -> dict:
    _sizes(sizes)
    if query_scores.shape!=(sizes[0],):raise ValueError('Query scores must match primary population')
    return dict(query_scores=query_scores,population_sizes=tuple(sizes),evaluation=dict(
        ks_agreement=ks,cvm_mass=cvm,truncated_weighted_auc=auc,
        agreement_passed=AbstractScalar(dtype='bool'),correlation_passed=AbstractScalar(dtype='bool'),
        oof_scope='Diagnostic only; upstream mass features are not nested within classifier folds'))


@register_atom(witness_diagnostic_report)
def diagnostic_report(query_scores: np.ndarray,ks: float,cvm: float,auc: float,sizes: list) -> dict:
    """Retain metric values and strict threshold failures without remediation.

    No model selection, prediction modification or calibrated-probability claim.
    Execution traces retain fit histories separately from this compact output.
    """
    _sizes(sizes);scores=_vector(query_scores,True)
    if len(scores)!=sizes[0] or any(not np.isfinite(v) or not 0<=v<=1 for v in (ks,cvm,auc)):
        raise ValueError('Aligned query scores and finite unit-interval metrics required')
    return dict(query_scores=scores.copy(),population_sizes=tuple(sizes),evaluation=dict(
        ks_agreement=float(ks),cvm_mass=float(cvm),truncated_weighted_auc=float(auc),
        agreement_passed=bool(ks<.09),correlation_passed=bool(cvm<.002),
        oof_scope='Diagnostic only; upstream mass features are not nested within classifier folds'))
