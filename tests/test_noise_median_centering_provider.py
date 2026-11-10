import numpy as np
import pytest
from sciona.atoms.ml.noise_median_centering import add_noise_median_center, witness_add_noise_median_center
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import REGISTRY


def test_perturbed_median_order_and_independent_storage():
    x = np.array([[0, 0, 8], [10, 20, 30]], dtype=np.int16)
    noise = np.array([[2., -1., 0.], [0., 8., -8.]])
    before_x, before_noise = x.copy(), noise.copy()
    actual = add_noise_median_center(x, noise, 'synthetic-amplitude.v1')
    np.testing.assert_array_equal(actual, [[0., -3., 6.], [-12., 6., 0.]])
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(noise, before_noise)
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, x) and not np.shares_memory(actual, noise)
    assert 'sciona.atoms.ml.noise_median_centering.add_noise_median_center' in REGISTRY


def test_row_translation_unit_scaling_and_batch_partition():
    rng = np.random.default_rng(457)
    x = rng.integers(-50, 50, (5, 20)).astype(float)
    noise = rng.integers(-4, 4, x.shape).astype(float) / 4
    baseline = add_noise_median_center(x, noise, 'coordinate-a')
    translated = add_noise_median_center(x + np.arange(5)[:, None]*100, noise, 'coordinate-a')
    np.testing.assert_array_equal(baseline, translated)
    scaled = add_noise_median_center(x*8, noise*8, 'coordinate-b')
    np.testing.assert_array_equal(scaled, baseline*8)
    partition = np.vstack([add_noise_median_center(x[i:i+1], noise[i:i+1], 'coordinate-a') for i in range(5)])
    np.testing.assert_array_equal(baseline, partition)
    np.testing.assert_array_equal(np.median(baseline, axis=1), np.zeros(5))


def test_noncontiguous_even_windows_and_singleton():
    x = np.array([[1., 3., 9., 11.]])[:, ::-1]
    np.testing.assert_array_equal(add_noise_median_center(x, np.zeros_like(x), 'u'), [[5., 3., -3., -5.]])
    np.testing.assert_array_equal(add_noise_median_center(np.ones((2, 1)), np.ones((2, 1)), 'u'), [[0.], [0.]])


@pytest.mark.parametrize('x,n', [
    (np.ones(3), np.ones(3)), (np.ones((0, 3)), np.ones((0, 3))),
    (np.ones((2, 3)), np.ones((1, 3))), (np.ones((2, 3), complex), np.ones((2, 3))),
    (np.array([[np.nan]]), np.ones((1, 1))), (np.ones((1, 1)), np.array([[np.inf]])),
    (np.full((1, 2), 1e308), np.full((1, 2), 1e308)),
])
def test_invalid_or_ambiguous_arrays_rejected(x, n):
    with pytest.raises(ValueError):
        add_noise_median_center(x, n, 'u')


def test_explicit_coordinate_required():
    for coordinate in ('', ' ', None):
        with pytest.raises(ValueError):
            add_noise_median_center(np.ones((1, 2)), np.ones((1, 2)), coordinate)


def test_witness_dimensions_shape_and_no_false_bounds():
    signal = AbstractArray(shape=(3, 4), dtype='int16', dim=DimensionalSignature(L=1))
    noise = AbstractArray(shape=(3, 4), dtype='float64', dim=DimensionalSignature(L=1))
    out = witness_add_noise_median_center(signal, noise, 'metres')
    assert out.shape == signal.shape and out.dim == signal.dim and out.dtype == 'float64'
    assert out.min_val is None and out.max_val is None
    for changed in (noise.model_copy(update={'dim': None}),
                    noise.model_copy(update={'dim': DimensionalSignature(T=1)}),
                    noise.model_copy(update={'shape': (1, 4)})):
        with pytest.raises(ValueError):
            witness_add_noise_median_center(signal, changed, 'metres')
