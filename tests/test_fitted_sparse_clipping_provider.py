import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sciona.atoms.ml.fitted_sparse_clipping import (
    fit_sparse_clip, apply_sparse_clip, witness_fit_sparse_clip, witness_apply_sparse_clip,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import REGISTRY


def test_portable_state_frozen_bound_and_nonmutation():
    train = csr_matrix([[2., 0.], [1., 5.]])
    state = json.loads(json.dumps(fit_sparse_clip(train, ['a', 'b'], 'u'), allow_nan=False))
    query = csr_matrix([[np.inf, -np.inf], [np.nan, 100.]])
    before = query.copy()
    result = apply_sparse_clip(query, ['a', 'b'], 'u', state)
    np.testing.assert_array_equal(result.toarray(), [[5., 0.], [0., 5.]])
    np.testing.assert_array_equal(query.data, before.data)
    assert not np.shares_memory(result.data, query.data)
    assert result.dtype == np.float32 and state['upper'] == 5
    assert apply_sparse_clip(csr_matrix((0, 2)), ['a', 'b'], 'u', state).shape == (0, 2)
    for name in ('fit_sparse_clip', 'apply_sparse_clip'):
        assert 'sciona.atoms.ml.fitted_sparse_clipping.' + name in REGISTRY


@pytest.mark.parametrize('columns,coordinate', [(['b', 'a'], 'u'), (['a', 'b'], 'v')])
def test_equal_width_semantic_mismatch_rejected(columns, coordinate):
    x = csr_matrix([[1., 2.]])
    state = fit_sparse_clip(x, ['a', 'b'], 'u')
    with pytest.raises(ValueError, match='Matching'):
        apply_sparse_clip(x, columns, coordinate, state)


@pytest.mark.parametrize('bound', [-1., float('inf'), float('nan'), True, '2', 1e50])
def test_restored_numeric_state_corruption(bound):
    x = csr_matrix([[1., 2.]])
    state = fit_sparse_clip(x, ['a', 'b'], 'u')
    state['upper'] = bound
    with pytest.raises(ValueError):
        apply_sparse_clip(x, ['a', 'b'], 'u', state)


def test_explicit_zero_vs_implicit_zero_training():
    x = csr_matrix(([0.], ([0], [1])), shape=(1, 2))
    assert fit_sparse_clip(x, ['a', 'b'], 'u')['upper'] == 0
    with pytest.raises(ValueError):
        fit_sparse_clip(csr_matrix((1, 2)), ['a', 'b'], 'u')
    duplicate = csr_matrix(([1., 2.], [0, 0], [0, 2]), shape=(1, 2))
    with pytest.raises(ValueError, match='Canonical'):
        fit_sparse_clip(duplicate, ['a', 'b'], 'u')


def test_abstract_identity_and_cross_coordinate_scaling():
    meta = AbstractArray(shape=(3, 2), dtype='float64')
    state = witness_fit_sparse_clip(meta, ['a', 'b'], 'u')
    assert witness_apply_sparse_clip(meta, ['a', 'b'], 'u', state).dtype == 'float32'
    with pytest.raises(ValueError):
        witness_apply_sparse_clip(meta, ['b', 'a'], 'u', state)
    train, query = csr_matrix([[1., 2.]]), csr_matrix([[3., -1.]])
    baseline = apply_sparse_clip(query, ['a', 'b'], 'u', fit_sparse_clip(train, ['a', 'b'], 'u'))
    scaled = apply_sparse_clip(query * 8, ['a', 'b'], 'scaled',
                              fit_sparse_clip(train * 8, ['a', 'b'], 'scaled'))
    np.testing.assert_array_equal(scaled.toarray(), baseline.toarray() * 8)
