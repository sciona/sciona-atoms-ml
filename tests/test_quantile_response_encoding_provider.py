import json
import numpy as np
import pytest
from sciona.atoms.ml.quantile_response_encoding import (
    fit_quantile_response, encode_quantile_response, decode_quantile_response,
    witness_fit_quantile_response, witness_encode_quantile_response, witness_decode_quantile_response,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import REGISTRY


def test_json_roundtrip_and_independent_soft_oracle():
    state = json.loads(json.dumps(fit_quantile_response(np.array([0., .5, 1.]), 'u', 2), allow_nan=False))
    x = np.array([0., .25, .5, .75, 1.])
    actual = encode_quantile_response(x, 'u', state, True, 8.)
    right = 1 / (1 + np.exp(-4 * (x - .5)))
    np.testing.assert_allclose(actual, np.column_stack([1-right, right]), rtol=1e-14)
    np.testing.assert_allclose(decode_quantile_response(actual, 'u', state), .25*(1-right) + .75*right)
    for name in ('fit_quantile_response', 'encode_quantile_response', 'decode_quantile_response'):
        assert 'sciona.atoms.ml.quantile_response_encoding.' + name in REGISTRY


def test_hard_edges_and_invalid_zero_probability_row():
    state = fit_quantile_response(np.array([0., 1.]), 'u', 2)
    actual = encode_quantile_response(np.array([-1., 0., .5, 1., 2.]), 'u', state, False, 1.)
    np.testing.assert_array_equal(actual, [[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]])
    assert actual.dtype == np.float32
    with pytest.raises(ValueError):
        decode_quantile_response(actual, 'u', state)


def test_affine_coordinate_transfer_requires_precision_conversion():
    x = np.array([.1, .4, .8])
    a = fit_quantile_response(np.array([0., .5, 1.]), 'u', 2)
    b = fit_quantile_response(np.array([-20., -15., -10.]), 'v', 2)
    p = encode_quantile_response(x, 'u', a, True, 8.)
    q = encode_quantile_response(10*x-20, 'v', b, True, .08)
    np.testing.assert_allclose(p, q, atol=1e-14)
    np.testing.assert_allclose(decode_quantile_response(q, 'v', b), 10*decode_quantile_response(p, 'u', a)-20)
    with pytest.raises(ValueError, match='Matching'):
        encode_quantile_response(x, 'v', a, True, 8.)


@pytest.mark.parametrize('edges', [[1., 0.], [float('nan'), 1.], [0.], [False, 1.], ['0', 1.]])
def test_restored_state_corruption(edges):
    state = dict(format='sciona.quantile_response.v1', response_coordinate='u', boundaries=edges)
    with pytest.raises(ValueError):
        encode_quantile_response(np.array([.5]), 'u', state, True, 8.)


def test_repeated_edges_underflow_and_empty_population():
    state = fit_quantile_response(np.full(3, .25), 'u', 3)
    np.testing.assert_allclose(encode_quantile_response(np.array([.25]), 'u', state, True, 160.), [[1/3]*3])
    for x in (np.array([1e6]), np.array([])):
        with pytest.raises(ValueError):
            encode_quantile_response(x, 'u', state, True, 160.)


def test_abstract_shapes_dtypes_and_identity():
    meta = AbstractArray(shape=(5,), dtype='float64')
    state = witness_fit_quantile_response(meta, 'u', 3)
    soft = witness_encode_quantile_response(meta, 'u', state, True, 160.)
    hard = witness_encode_quantile_response(meta, 'u', state, False, 160.)
    assert soft.shape == (5, 3) and hard.dtype == 'float32'
    assert witness_decode_quantile_response(soft, 'u', state).shape == (5,)
    with pytest.raises(ValueError):
        witness_decode_quantile_response(soft, 'v', state)
