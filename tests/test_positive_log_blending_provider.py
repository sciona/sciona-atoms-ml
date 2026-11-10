import json

import numpy as np
import pytest

from sciona.atoms.ml.positive_log_blending import (
    fit_positive_log_blend, apply_positive_log_blend,
    witness_fit_positive_log_blend, witness_apply_positive_log_blend,
)
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import REGISTRY

CONTROLS = dict(alpha=.0001, max_iter=1000, tolerance=.0001, seed=9999)


def test_registry_and_json_roundtrip():
    x = np.array([[1., 2.], [3., 2.], [4., 6.], [7., 8.]])
    y = np.array([2., 3., 5., 7.])
    state = fit_positive_log_blend(x, y, ['a', 'b'], 'normalized-response', [0., 10.], CONTROLS)
    restored = json.loads(json.dumps(state, allow_nan=False))
    predicted = apply_positive_log_blend(x, ['a', 'b'], 'normalized-response', restored)
    oracle = np.expm1(np.log1p(x) @ np.asarray(state['coefficients']) + state['intercept'])
    np.testing.assert_array_equal(predicted, oracle)
    assert 'sciona.atoms.ml.positive_log_blending.fit_positive_log_blend' in REGISTRY
    assert 'sciona.atoms.ml.positive_log_blending.apply_positive_log_blend' in REGISTRY


@pytest.mark.parametrize('columns,coordinate', [(['b', 'a'], 'u'), (['a', 'b'], 'v')])
def test_identity_mismatch_rejected(columns, coordinate):
    x = np.array([[1., 2.], [3., 2.], [4., 6.], [7., 8.]])
    state = fit_positive_log_blend(x, np.array([2., 3., 5., 7.]), ['a', 'b'], 'u', [0., 10.], CONTROLS)
    with pytest.raises(ValueError, match='Matching'):
        apply_positive_log_blend(x, columns, coordinate, state)


def test_abstract_flow_and_wrong_columns():
    x = AbstractArray(shape=(10, 2), dtype='float64')
    y = AbstractArray(shape=(10,), dtype='float64')
    state = witness_fit_positive_log_blend(x, y, ['a', 'b'], 'u', [0., 10.], CONTROLS)
    result = witness_apply_positive_log_blend(x, ['a', 'b'], 'u', state)
    assert result.shape == (10,) and result.dtype == 'float64'
    with pytest.raises(ValueError):
        witness_apply_positive_log_blend(x, ['b', 'a'], 'u', state)


def test_runtime_state_corruption_and_dtype():
    x = np.array([[1., 2.], [3., 2.], [4., 6.], [7., 8.]])
    state = fit_positive_log_blend(x, np.array([2., 3., 5., 7.]), ['a', 'b'], 'u', [0., 10.], CONTROLS)
    state['coefficients'][0] = -1.
    with pytest.raises(ValueError):
        apply_positive_log_blend(x, ['a', 'b'], 'u', state)
    with pytest.raises(ValueError):
        fit_positive_log_blend(x.astype(np.float32), np.array([2., 3., 5., 7.]), ['a', 'b'], 'u', [0., 10.], CONTROLS)


def test_nonconverged_fit_is_rejected():
    from sklearn.exceptions import ConvergenceWarning
    with pytest.raises(ConvergenceWarning):
        fit_positive_log_blend(np.array([[1., 2.], [3., 4.]]), np.array([2., 4.]),
                               ['a', 'b'], 'u', [0., 10.], CONTROLS)
