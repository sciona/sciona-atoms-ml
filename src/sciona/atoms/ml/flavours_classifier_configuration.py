"""Source classifier settings and explicit numerical feature-view boundary."""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom

VIEWS=('base','restricted','proxy','corrected')
SETTINGS=((5,.39,100,.1,.4,4),(35,.2,15,15,.9,5),
          (700,.05,6,10,1.,1),(3,.5,155,.1,.4,4),(1500,.05,6,10,1.,1))


def _settings():
    result=[5,555,0,1,2,3,4,1]
    for rounds,eta,depth,child,columns,seed in SETTINGS:
        result.extend([dict(eta=eta,max_depth=depth,min_child_weight=child,
            subsample=1.,colsample_bytree=columns,seed=seed,tree_method='exact',base_score=.5),rounds])
    return tuple(result)


def witness_classifier_view_inputs(training_views: dict,query_views: dict,feature_names: dict) -> tuple:
    if any(not isinstance(v,dict) or set(v)!=set(VIEWS) for v in (training_views,query_views,feature_names)):
        raise ValueError('Exactly four explicit named views and feature-name lists required')
    outputs=[];rows=None;query_rows=None
    for name in VIEWS:
        a,b=training_views[name],query_views[name];names=feature_names[name]
        if (not isinstance(a,AbstractArray) or not isinstance(b,AbstractArray)
                or a.dtype!='float64' or b.dtype!='float64' or len(a.shape)!=2 or len(b.shape)!=2
                or a.shape[0]<5 or a.shape[1]<1 or b.shape[0]<1 or a.shape[1]!=b.shape[1]):
            raise ValueError('Aligned float64 feature matrices required')
        if (rows is not None and a.shape[0]!=rows) or (query_rows is not None and b.shape[0]!=query_rows):
            raise ValueError('All feature views must share row alignment')
        rows,query_rows=a.shape[0],b.shape[0]
        if not isinstance(names,list) or len(names)!=a.shape[1] or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise ValueError('Unique ordered feature names must match each view width')
        if a.dim is not None and b.dim is not None and not a.dim.is_compatible(b.dim):
            raise ValueError('Training and query view units must agree')
        outputs.extend([a,b,names])
    return (*outputs,*_settings())


@register_atom(witness_classifier_view_inputs)
def classifier_view_inputs(training_views: dict,query_views: dict,feature_names: dict) -> tuple:
    """Expose four caller-prepared views and the five source configurations.

    Performs no feature engineering or fitting. Caller establishes consistent
    row identities, column meanings and units. NaN features retain native
    missing-value semantics; infinities are rejected. Labels enter fitting
    nodes separately, and query predictions use full-data refits.
    """
    if any(not isinstance(v,dict) or set(v)!=set(VIEWS) for v in (training_views,query_views,feature_names)):
        raise ValueError('Exactly four explicit named views and feature-name lists required')
    train={k:np.asarray(training_views[k]) for k in VIEWS}
    query={k:np.asarray(query_views[k]) for k in VIEWS}
    witness_classifier_view_inputs(
        {k:AbstractArray(shape=v.shape,dtype=str(v.dtype)) for k,v in train.items()},
        {k:AbstractArray(shape=v.shape,dtype=str(v.dtype)) for k,v in query.items()},feature_names)
    if any(np.isinf(v).any() for v in (*train.values(),*query.values())):
        raise ValueError('Infinite features are unsupported')
    outputs=[]
    for name in VIEWS:outputs.extend([train[name],query[name],list(feature_names[name])])
    return (*outputs,*_settings())
