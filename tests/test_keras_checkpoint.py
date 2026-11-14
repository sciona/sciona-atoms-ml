"""Synthetic saved-state selection checks across two model architectures."""
import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from sciona.atoms.ml.keras_checkpoint import load_trusted_keras_checkpoint

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


@pytest.mark.parametrize('hidden', [False, True])
def test_selected_checkpoint_preserves_earlier_weights_and_optimizer(tmp_path, hidden):
    keras.utils.set_random_seed(73)
    layers = [keras.layers.Input(shape=(3,))]
    if hidden:
        layers.append(keras.layers.Dense(4, activation='tanh'))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.legacy.Adam(.01), loss='mse')
    x = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    y = np.array([[1], [-1], [2], [-2]], dtype=np.float32)
    model.train_on_batch(x, y)
    expected = model(x, training=False).numpy()
    weights = model.get_weights()
    optimizer = [v.numpy().copy() for v in model.optimizer.variables()]
    checkpoint = tmp_path/'selected.h5'
    model.save(checkpoint)
    original_bytes = checkpoint.read_bytes()
    model.train_on_batch(x, y)
    assert any(not np.array_equal(a, b) for a, b in zip(weights, model.get_weights()))
    restored = load_trusted_keras_checkpoint(checkpoint, {})
    np.testing.assert_array_equal(restored(x, training=False), expected)
    assert len(restored.get_weights()) == len(weights)
    for actual, prior in zip(restored.get_weights(), weights):
        np.testing.assert_array_equal(actual, prior)
    assert len(restored.optimizer.variables()) == len(optimizer)
    for actual, prior in zip(restored.optimizer.variables(), optimizer):
        np.testing.assert_array_equal(actual.numpy(), prior)
    assert int(restored.optimizer.iterations) == 1
    assert int(model.optimizer.iterations) == 2
    assert checkpoint.read_bytes() == original_bytes


def test_invalid_contracts_do_not_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid contract reached deserializer')
    monkeypatch.setattr(keras.models, 'load_model', forbidden)
    for path, objects in [(None, {}), (tmp_path, {}), (tmp_path/'absent', {}),
                          (tmp_path, None), (tmp_path, {'': object()})]:
        with pytest.raises(ValueError):
            load_trusted_keras_checkpoint(path, objects)
