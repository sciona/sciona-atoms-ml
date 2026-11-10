"""Synthetic value and derivative checks independent of competition source."""
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[key] = '1'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import pytest

from sciona.atoms.ml.tensorflow_distribution_losses import (
    mean_pinball, clipped_laplace_score, witness_distribution_loss,
)


@pytest.mark.parametrize('unit_scale', [1., 1000.], ids=['duration-seconds', 'energy-joules'])
def test_pinball_value_gradient_and_translation(unit_scale):
    targets = np.array([[2.], [7.]], np.float32)*unit_scale
    predictions = np.array([[1., 3., 4.], [5., 8., 6.]], np.float32)*unit_scale
    quantiles = np.array([.1, .5, .9], np.float32)
    residual = targets.astype(np.float64)-predictions
    expected = np.mean(np.where(residual > 0, quantiles*residual,
                                (quantiles-1.)*residual))
    expected_gradient = np.where(residual > 0, -quantiles, 1.-quantiles)/6
    variable = tf.Variable(predictions)
    with tf.GradientTape() as tape:
        actual = mean_pinball(targets, variable, quantiles)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-7)
    np.testing.assert_allclose(tape.gradient(actual, variable).numpy(), expected_gradient, atol=2e-8)
    translated = mean_pinball(targets+10*unit_scale, predictions+10*unit_scale, quantiles)
    np.testing.assert_array_equal(actual.numpy(), translated.numpy())
    compiled = tf.function(mean_pinball)
    np.testing.assert_array_equal(compiled(targets, predictions, quantiles).numpy(), actual.numpy())


@pytest.mark.parametrize('unit_scale', [1., 1000.], ids=['duration-seconds', 'energy-joules'])
def test_clipped_score_scalar_reference_and_derivative(unit_scale):
    targets = np.array([[2.], [7.], [10.]], np.float32)*unit_scale
    predictions = np.array([[0., 1., 4.], [8., 6., 3.], [0., 1., 6.]], np.float32)*unit_scale
    floor, cap = 2.*unit_scale, 5.*unit_scale
    total = 0.
    gradient = np.zeros_like(predictions, dtype=np.float64)
    for i, (target, row) in enumerate(zip(targets[:, 0], predictions)):
        low, point, high = map(float, row)
        error = abs(float(target)-point)
        spread = high-low
        scale = max(spread, floor)
        total += min(error, cap)/scale*np.sqrt(2)+np.log(scale*np.sqrt(2))
        if error < cap:
            gradient[i, 1] = np.sign(point-target)*np.sqrt(2)/scale/3
        if spread > floor:
            derivative = (1/scale-min(error, cap)*np.sqrt(2)/scale**2)/3
            gradient[i, 0], gradient[i, 2] = -derivative, derivative
    variable = tf.Variable(predictions)
    with tf.GradientTape() as tape:
        actual = clipped_laplace_score(targets, variable, floor, cap)
    np.testing.assert_allclose(actual.numpy(), total/3, rtol=2e-7)
    # The spread derivative subtracts two terms: allow one float32 epsilon
    # in inverse target units near cancellation against the float64 oracle.
    np.testing.assert_allclose(tape.gradient(actual, variable).numpy(), gradient,
                               rtol=3e-7, atol=np.finfo(np.float32).eps/unit_scale)
    compiled = tf.function(lambda y, p: clipped_laplace_score(y, p, floor, cap))
    np.testing.assert_allclose(compiled(targets, predictions).numpy(), actual.numpy(), rtol=2e-7)


def test_abstract_execution_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='Concrete float32'):
        witness_distribution_loss(None, None)
