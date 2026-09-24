"""Reusable CPU XGBoost fitting and native model-state prediction boundaries.

Unspecified estimator controls use the installed backend defaults. This is not
historical-engine reproduction. Models remain runtime values, never fixtures.
"""
import base64
import hashlib
import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _names(names, width):
    if not isinstance(names, list) or len(names) != width or any(
            not isinstance(n, str) or not n or any(c in n for c in '[]<') for n in names) or len(set(names)) != width:
        raise ValueError('Unique ordered backend-compatible feature names required')
    return list(names)


def _matrix(features, names):
    matrix = np.asarray(features)
    if matrix.dtype != np.float64 or matrix.ndim != 2 or min(matrix.shape) < 1 or np.isinf(matrix).any():
        raise ValueError('Nonempty float64 feature matrix without infinities required')
    return pd.DataFrame(matrix, columns=_names(names, matrix.shape[1]))


def _fit_metadata(features, targets, names, task):
    if len(features.shape) != 2 or min(features.shape) < 1 or features.dtype != 'float64':
        raise ValueError('Nonempty float64 matrix metadata required')
    if targets.shape != (features.shape[0],) or targets.dtype != ('float64' if task == 'regression' else 'int64'):
        raise ValueError('Aligned target metadata required')
    _names(names, features.shape[1])
    if task == 'binary' and targets.dim is not None and not targets.dim.is_dimensionless:
        raise ValueError('Binary labels must be dimensionless')
    return dict(format='sciona.xgboost.abstract.v1', task=task,
                feature_names=list(names), target_dim=targets.dim)


def witness_fit_regression_model(features: AbstractArray, targets: AbstractArray, feature_names: list) -> dict:
    return _fit_metadata(features, targets, feature_names, 'regression')


def witness_fit_binary_model(features: AbstractArray, targets: AbstractArray, feature_names: list) -> dict:
    return _fit_metadata(features, targets, feature_names, 'binary')


def _prediction_metadata(features, names, state, task):
    if len(features.shape) != 2 or min(features.shape) < 1 or features.dtype != 'float64':
        raise ValueError('Nonempty float64 matrix metadata required')
    _names(names, features.shape[1])
    if not isinstance(state, dict) or state.get('format') != 'sciona.xgboost.abstract.v1' or state.get('task') != task or state.get('feature_names') != names:
        raise ValueError('Matching model task and ordered feature metadata required')
    return AbstractArray(shape=(features.shape[0],), dtype='float32',
                         dim=state.get('target_dim') if task == 'regression' else None)


def witness_predict_regression_model(prediction_features: AbstractArray, feature_names: list, state: dict) -> AbstractArray:
    return _prediction_metadata(prediction_features, feature_names, state, 'regression')


def witness_predict_binary_model(prediction_features: AbstractArray, feature_names: list, state: dict) -> AbstractArray:
    return _prediction_metadata(prediction_features, feature_names, state, 'binary')


def _fit(features, targets, names, task):
    import xgboost as xgb
    frame = _matrix(features, names)
    labels = np.asarray(targets)
    if labels.shape != (len(frame),) or labels.dtype != (np.float64 if task == 'regression' else np.int64) or not np.isfinite(labels).all():
        raise ValueError('Aligned finite targets of the declared dtype required')
    if task == 'binary' and not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError('Both binary classes 0 and 1 required')
    estimator = (xgb.XGBRegressor if task == 'regression' else xgb.XGBClassifier)(n_jobs=1, device='cpu')
    estimator.fit(frame, labels)
    payload = bytes(estimator.get_booster().save_raw(raw_format='ubj'))
    return dict(format='sciona.xgboost.model.v1', task=task, backend_version=xgb.__version__,
        feature_names=list(names), payload=base64.b64encode(payload).decode('ascii'),
        payload_sha256=hashlib.sha256(payload).hexdigest())


@register_atom(witness_fit_regression_model)
def fit_regression_model(features: NDArray[np.float64], targets: NDArray[np.float64], feature_names: list) -> dict:
    """Fit default squared-error regression on explicit ordered feature columns.

    One CPU thread; NaN features use backend missing-value behavior, infinities
    and nonfinite targets are rejected. No splitting, scaling, quantization or
    target-unit conversion. Native UBJ model plus exact names/version/digest is
    returned as a JSON-compatible runtime value. No pickle or filesystem IO.
    """
    return _fit(features, targets, feature_names, 'regression')


@register_atom(witness_fit_binary_model)
def fit_binary_model(features: NDArray[np.float64], targets: NDArray[np.int64], feature_names: list) -> dict:
    """Fit default binary logistic classification with both integer classes.

    Shares the regression atom's ordered-column, missing-value and one-thread
    contract. Returned state is portable within the exact installed version;
    cross-version compatibility is not inferred.
    """
    return _fit(features, targets, feature_names, 'binary')


def _predict(features, names, state, task):
    import xgboost as xgb
    frame = _matrix(features, names)
    if not isinstance(state, dict) or set(state) != {'format', 'task', 'backend_version', 'feature_names', 'payload', 'payload_sha256'}:
        raise ValueError('Complete native model state required')
    if state['format'] != 'sciona.xgboost.model.v1' or state['task'] != task or state['backend_version'] != xgb.__version__ or state['feature_names'] != names:
        raise ValueError('Model task, backend version and feature order must match')
    try:
        payload = base64.b64decode(state['payload'], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid native model encoding') from exc
    if hashlib.sha256(payload).hexdigest() != state['payload_sha256']:
        raise ValueError('Native model payload integrity mismatch')
    booster = xgb.Booster(params={'nthread': 1, 'device': 'cpu'})
    booster.load_model(bytearray(payload))
    booster.set_param({'nthread': 1, 'device': 'cpu'})
    expected = 'reg:squarederror' if task == 'regression' else 'binary:logistic'
    if booster.feature_names != names or booster.num_features() != len(names) or json.loads(booster.save_config())['learner']['objective']['name'] != expected:
        raise ValueError('Native model contents differ from state contract')
    output = booster.predict(xgb.DMatrix(frame, nthread=1), validate_features=True)
    if output.shape != (len(frame),) or output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError('Invalid model predictions')
    if task == 'binary' and ((output < 0).any() or (output > 1).any()):
        raise ValueError('Invalid positive-class probabilities')
    return output


@register_atom(witness_predict_regression_model)
def predict_regression_model(prediction_features: NDArray[np.float64], feature_names: list, state: dict) -> NDArray[np.float32]:
    """Predict float32 values in learned target units, without rounding or correction."""
    return _predict(prediction_features, feature_names, state, 'regression')


@register_atom(witness_predict_binary_model)
def predict_binary_model(prediction_features: NDArray[np.float64], feature_names: list, state: dict) -> NDArray[np.float32]:
    """Predict float32 probabilities for class 1, without thresholding labels."""
    return _predict(prediction_features, feature_names, state, 'binary')
