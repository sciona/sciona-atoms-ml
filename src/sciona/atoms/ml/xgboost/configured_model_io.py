"""Configured exact-tree CPU fitting with the existing native model-state ABI.

Individual fits are reusable numerical operations. Fold construction, physical
feature adapters, aggregation and source configuration belong in caller graphs.
"""
import base64
import hashlib
import math
import numbers

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom
from .model_io import _fit_metadata, _matrix


def _configuration(configuration, boosting_rounds, task):
    required={'eta','max_depth','min_child_weight','subsample','colsample_bytree','seed','tree_method','base_score'}
    if not isinstance(configuration,dict) or set(configuration)!=required:
        raise ValueError('Complete exact-tree configuration required; implicit or extra controls are unsupported')
    if isinstance(boosting_rounds,(bool,np.bool_)) or not isinstance(boosting_rounds,numbers.Integral) or boosting_rounds<1:
        raise ValueError('Positive integer boosting rounds required')
    result=dict(configuration)
    for key in ['max_depth','seed']:
        value=result[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or value<0:
            raise ValueError('Nonnegative integer '+key+' required')
        result[key]=int(value)
    if result['max_depth']==0:
        raise ValueError('Positive maximum depth required for exact trees')
    for key in ['eta','min_child_weight','subsample','colsample_bytree','base_score']:
        value=result[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not math.isfinite(value):
            raise ValueError('Finite numeric '+key+' required')
        result[key]=float(value)
    if any(not 0<result[key]<=1 for key in ['eta','subsample','colsample_bytree']) or result['min_child_weight']<0:
        raise ValueError('Invalid learning rate, sample fractions or child weight')
    if result['tree_method']!='exact':
        raise ValueError('This fitting boundary requires CPU exact trees')
    if task=='binary' and not 0<result['base_score']<1:
        raise ValueError('Binary base score must lie strictly between zero and one')
    result.update(objective='reg:squarederror' if task=='regression' else 'binary:logistic',nthread=1,device='cpu')
    return result,int(boosting_rounds)


def witness_fit_configured_regression_model(features: AbstractArray, targets: AbstractArray, feature_names: list,
        configuration: dict, boosting_rounds: int) -> tuple:
    _,rounds=_configuration(configuration,boosting_rounds,'regression')
    return _fit_metadata(features,targets,feature_names,'regression'),rounds


def witness_fit_configured_binary_model(features: AbstractArray, targets: AbstractArray, feature_names: list,
        configuration: dict, boosting_rounds: int) -> tuple:
    _,rounds=_configuration(configuration,boosting_rounds,'binary')
    return _fit_metadata(features,targets,feature_names,'binary'),rounds


def _fit(features,targets,feature_names,configuration,boosting_rounds,task):
    import xgboost as xgb
    parameters,rounds=_configuration(configuration,boosting_rounds,task)
    frame=_matrix(features,feature_names)
    target=np.asarray(targets)
    dtype=np.float64 if task=='regression' else np.int64
    if target.dtype!=dtype or target.shape!=(len(frame),) or not np.isfinite(target).all():
        raise ValueError('Aligned finite targets of the declared dtype required')
    if task=='binary' and not np.array_equal(np.unique(target),[0,1]):
        raise ValueError('Both binary classes 0 and 1 required')
    # Avoid changing XGBoost's global default: some backend versions do not
    # restore nthread=0 after config_context exits. All fitting/data operations
    # also receive their own explicit single-thread controls.
    with threadpool_limits(limits=1):
        model=xgb.train(parameters,xgb.DMatrix(frame,label=target,missing=np.nan,nthread=1),num_boost_round=rounds)
        actual=model.num_boosted_rounds()
        if actual!=rounds:
            raise RuntimeError('Backend did not perform the complete requested fitting budget')
        payload=bytes(model.save_raw(raw_format='ubj'))
    state=dict(format='sciona.xgboost.model.v1',task=task,backend_version=xgb.__version__,
        feature_names=list(feature_names),payload=base64.b64encode(payload).decode('ascii'),
        payload_sha256=hashlib.sha256(payload).hexdigest())
    return state,actual


@register_atom(witness_fit_configured_regression_model)
def fit_configured_regression_model(features: NDArray[np.float64], targets: NDArray[np.float64],
        feature_names: list, configuration: dict, boosting_rounds: int) -> tuple[dict,int]:
    """Fit one exact-tree squared-error model using explicit numerical controls.

    Nonempty float64 matrices, NaN missing features, finite aligned targets and
    unique ordered feature names follow the existing model-state contract.
    Configuration contains exactly eta, max_depth, min_child_weight, subsample,
    colsample_bytree, seed, tree_method='exact', and base_score. CPU execution is
    limited to one thread, including the native scoped runtime. No validation
    stopping, resampling, unit conversion or implicit fold aggregation occurs.
    Return compatible native UBJ state and the actual complete boosting rounds.
    State is private runtime material tied to the exact backend version.
    """
    return _fit(features,targets,feature_names,configuration,boosting_rounds,'regression')


@register_atom(witness_fit_configured_binary_model)
def fit_configured_binary_model(features: NDArray[np.float64], targets: NDArray[np.int64],
        feature_names: list, configuration: dict, boosting_rounds: int) -> tuple[dict,int]:
    """Fit one exact-tree logistic model with both integer classes zero and one.

    Uses the configured regression boundary's explicit controls, complete-round
    and native-state contract. Base score must lie strictly between zero and
    one. Graphs own folds, feature semantics, full refits and probability blends.
    The state is consumed by the existing binary prediction operation.
    """
    return _fit(features,targets,feature_names,configuration,boosting_rounds,'binary')
