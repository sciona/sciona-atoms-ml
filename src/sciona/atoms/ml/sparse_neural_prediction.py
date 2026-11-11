"""Registered sparse neural fit/predict operations with isolated CPU execution.

Registration is not catalog approval. Numerical jobs run synchronously in one
fresh single-core process so TensorFlow import order is controlled. Callers
must include this worker in their shared concurrency budget.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def _identities(columns, feature_coordinate, response_coordinate, width):
    if (type(columns) is not list or len(columns) != width or width < 1
            or any(type(v) is not str or not v.strip() for v in columns) or len(set(columns)) != width):
        raise ValueError('Unique ordered feature identifiers required')
    if any(type(v) is not str or not v.strip() for v in (feature_coordinate, response_coordinate)):
        raise ValueError('Explicit normalized feature and response coordinates required')


def _controls(mode, controls):
    keys = {'n_hidden', 'learning_rate', 'lr_decay', 'n_epoch', 'batch_size', 'bs_decay',
            'decay_epochs', 'reg_l2', 'reg_l1', 'use_target_scaling', 'n_bins', 'seed', 'binary_X',
            'quantile_precision', 'classification_loss_multiplier', 'activation'}
    if mode not in ('mse', 'quantile') or type(controls) is not dict or set(controls) != keys:
        raise ValueError('Explicit complete sparse neural configuration required')
    if controls['activation'] not in ('relu', 'prelu'):
        raise ValueError('Known relu/prelu activation required')
    hidden = controls['n_hidden']
    if type(hidden) is not list or len(hidden) not in (2, 3) or any(type(n) is not int or n < 1 for n in hidden):
        raise ValueError('Two or three positive hidden widths required')
    for key in ('n_epoch', 'batch_size', 'bs_decay', 'n_bins'):
        if type(controls[key]) is not int or controls[key] < 1:
            raise ValueError('Positive integer control required: '+key)
    if type(controls['decay_epochs']) is not int or controls['decay_epochs'] < 0:
        raise ValueError('Nonnegative decay epoch count required')
    if type(controls['seed']) is not int or not 0 <= controls['seed'] < 2**32:
        raise ValueError('Unsigned 32-bit seed required')
    for key in ('learning_rate', 'lr_decay', 'reg_l1', 'reg_l2', 'quantile_precision', 'classification_loss_multiplier'):
        value = controls[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative control required: '+key)
        if key not in ('reg_l1', 'reg_l2') and value == 0:
            raise ValueError('Positive control required: '+key)
    if any(type(controls[k]) is not bool for k in ('use_target_scaling', 'binary_X')):
        raise ValueError('Explicit boolean feature and target policies required')
    if mode == 'quantile' and not controls['use_target_scaling']:
        raise ValueError('Quantile encoding must be enabled')


def _abstract(X, columns, fc, rc):
    if len(X.shape) != 2 or X.dtype != 'float32':
        raise ValueError('Float32 feature matrix metadata required')
    _identities(columns, fc, rc, X.shape[1])
    if X.dim is not None and not X.dim.is_dimensionless:
        raise ValueError('Normalize features to the declared coordinate upstream')


def _matrix(X, columns, fc, rc):
    if not isinstance(X, csr_matrix) or X.dtype != np.float32 or not np.isfinite(X.data).all():
        raise ValueError('Finite float32 CSR matrix required')
    if not X.has_canonical_format:
        raise ValueError('Canonical CSR storage required; normalize upstream')
    _identities(columns, fc, rc, X.shape[1])


def witness_fit_sparse_neural(features: AbstractArray, targets: AbstractArray, feature_columns: list,
        feature_coordinate: str, response_coordinate: str, mode: str, controls: dict) -> dict:
    _abstract(features, feature_columns, feature_coordinate, response_coordinate)
    _controls(mode, controls)
    if features.shape[0] < 1 or targets.shape != (features.shape[0],) or targets.dtype not in ('float32', 'float64'):
        raise ValueError('Nonempty aligned floating targets required')
    if targets.dim is not None and not targets.dim.is_dimensionless:
        raise ValueError('Normalize targets to the declared response coordinate upstream')
    return dict(format='sciona.sparse_neural.abstract.v1', mode=mode, feature_columns=list(feature_columns),
                feature_coordinate=feature_coordinate, response_coordinate=response_coordinate)


@register_atom(witness_fit_sparse_neural)
def fit_sparse_neural(features: csr_matrix, targets: np.ndarray, feature_columns: list,
        feature_coordinate: str, response_coordinate: str, mode: str, controls: dict) -> dict:
    """Train an MSE or quantile sparse neural predictor and return numeric JSON state.

    All training controls and normalized coordinate identities are explicit.
    Caller owns training-only fitting and feature semantics. Binary mode treats
    every stored entry, including explicit zeros, as one. Regularization applies
    to the first-layer weight matrix. No log transform, clamp or application
    inverse transform is implied. The prediction state excludes optimizer state.
    One single-core worker runs synchronously and releases its session on exit.
    """
    from sciona._runtime.ml.sparse_neural_process import run_sparse_neural
    _matrix(features, feature_columns, feature_coordinate, response_coordinate)
    _controls(mode, controls)
    y = np.asarray(targets)
    if (not features.shape[0] or y.shape != (features.shape[0],) or y.dtype not in (np.float32, np.float64)
            or not np.isfinite(y).all()):
        raise ValueError('Finite nonempty aligned float32/float64 target vector required')
    return run_sparse_neural(dict(operation='fit', feature_columns=feature_columns,
        feature_coordinate=feature_coordinate, response_coordinate=response_coordinate,
        mode=mode, controls=controls), features, y)


def witness_predict_sparse_neural(features: AbstractArray, feature_columns: list,
        feature_coordinate: str, response_coordinate: str, state: dict) -> AbstractArray:
    _abstract(features, feature_columns, feature_coordinate, response_coordinate)
    if (type(state) is not dict or state.get('mode') not in ('mse', 'quantile')
            or state != dict(format='sciona.sparse_neural.abstract.v1', mode=state['mode'],
                feature_columns=feature_columns, feature_coordinate=feature_coordinate,
                response_coordinate=response_coordinate)):
        raise ValueError('Matching abstract model state required')
    return AbstractArray(shape=(features.shape[0],), dtype='float64' if state['mode'] == 'quantile' else 'float32',
                         dim=DimensionalSignature())


@register_atom(witness_predict_sparse_neural)
def predict_sparse_neural(features: csr_matrix, feature_columns: list,
        feature_coordinate: str, response_coordinate: str, state: dict) -> np.ndarray:
    """Restore numeric prediction state in a fresh CPU worker; never train.

    Requires exact ordered feature identities and coordinate declarations.
    Returns finite float32 MSE predictions or float64 quantile predictions in
    the declared response coordinate. Empty query rows are supported. Known
    graph construction replaces executable checkpoint loading; sessions close
    after each call. Learned weights and fitted decoder state remain private
    runtime inputs, not public benchmark evidence.
    """
    from sciona._runtime.ml.sparse_neural_process import run_sparse_neural
    _matrix(features, feature_columns, feature_coordinate, response_coordinate)
    return run_sparse_neural(dict(operation='predict', feature_columns=feature_columns,
        feature_coordinate=feature_coordinate, response_coordinate=response_coordinate, state=state), features)
