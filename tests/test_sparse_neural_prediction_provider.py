import json
# Exercise the ordinary pipeline import order; TensorFlow stays in the worker.
import sklearn
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sciona.atoms.ml.sparse_neural_prediction import (
    fit_sparse_neural, predict_sparse_neural,
    witness_fit_sparse_neural, witness_predict_sparse_neural,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import REGISTRY


CONTROLS = dict(n_hidden=[3, 2], learning_rate=.004, lr_decay=.75, n_epoch=1,
    batch_size=4, bs_decay=2, decay_epochs=2, reg_l1=0., reg_l2=1e-5,
    use_target_scaling=True, n_bins=2, seed=0, binary_X=False,
    quantile_precision=160., classification_loss_multiplier=6., activation='prelu')


@pytest.mark.parametrize('mode', ['mse', 'quantile'])
def test_isolated_fit_and_portable_prediction(mode):
    X = csr_matrix(np.array([[0., 1.], [1., 0.], [1., 1.], [.5, .5]], dtype=np.float32))
    state = fit_sparse_neural(X, np.array([1., 2., 3., 2.]), ['a', 'b'], 'f', 'r', mode, CONTROLS)
    state = json.loads(json.dumps(state, allow_nan=False))
    result = predict_sparse_neural(X, ['a', 'b'], 'f', 'r', state)
    replay = predict_sparse_neural(X, ['a', 'b'], 'f', 'r', state)
    np.testing.assert_array_equal(result, replay)
    assert result.shape == (4,) and np.isfinite(result).all()
    assert result.dtype == (np.float32 if mode == 'mse' else np.float64)
    empty = predict_sparse_neural(csr_matrix((0, 2), dtype=np.float32), ['a', 'b'], 'f', 'r', state)
    assert empty.shape == (0,) and empty.dtype == result.dtype
    with pytest.raises(ValueError):
        predict_sparse_neural(X, ['b', 'a'], 'f', 'r', state)
    for name in ('fit_sparse_neural', 'predict_sparse_neural'):
        assert 'sciona.atoms.ml.sparse_neural_prediction.'+name in REGISTRY


def test_abstract_controls_and_state_identity():
    X = AbstractArray(shape=(4, 2), dtype='float32')
    y = AbstractArray(shape=(4,), dtype='float64')
    state = witness_fit_sparse_neural(X, y, ['a', 'b'], 'f', 'r', 'quantile', CONTROLS)
    result = witness_predict_sparse_neural(X, ['a', 'b'], 'f', 'r', state)
    assert result.shape == (4,) and result.dtype == 'float64'
    with pytest.raises(ValueError):
        witness_predict_sparse_neural(X, ['a', 'b'], 'f', 'other', state)
    with pytest.raises(ValueError):
        witness_fit_sparse_neural(X, y, ['a', 'b'], 'f', 'r', 'quantile', {})


def test_invalid_inputs_rejected_before_worker_launch(monkeypatch):
    from sciona._runtime.ml import sparse_neural_process
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid inputs reached the worker')
    monkeypatch.setattr(sparse_neural_process, 'run_sparse_neural', forbidden)
    X = csr_matrix(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError):
        fit_sparse_neural(X, np.array([np.nan, 1.]), ['a', 'b'], 'f', 'r', 'mse', CONTROLS)
    with pytest.raises(ValueError):
        fit_sparse_neural(X, np.array([1., 2.]), ['a', 'b'], 'f', 'r', 'mse', dict(CONTROLS, batch_size=True))
    with pytest.raises(ValueError):
        fit_sparse_neural(X.astype(np.float64), np.array([1., 2.]), ['a', 'b'], 'f', 'r', 'mse', CONTROLS)
