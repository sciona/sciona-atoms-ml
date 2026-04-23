from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._bicluster import (
    _bistochastic_normalize,
    _log_normalize,
    _scale_normalize,
)


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.5],
            [3.0, 4.0, 1.5],
            [0.5, 1.0, 2.0],
            [2.0, 0.5, 3.0],
        ],
        dtype=np.float64,
    )


def test_bicluster_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import (
        bicluster_bistochastic_normalize,
        bicluster_log_normalize,
        bicluster_scale_normalize,
    )

    assert callable(bicluster_scale_normalize)
    assert callable(bicluster_bistochastic_normalize)
    assert callable(bicluster_log_normalize)


def test_bicluster_scale_normalize_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import bicluster_scale_normalize

    X = _data()
    result = bicluster_scale_normalize(X)
    expected = _scale_normalize(X)

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.allclose(result[2], expected[2])


def test_bicluster_scale_normalize_matches_sklearn_after_nonnegative_shift() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import bicluster_scale_normalize

    X = np.array(
        [
            [-0.5, 1.0, 2.0],
            [3.0, 0.0, 1.5],
            [2.0, 4.0, -0.25],
        ],
        dtype=np.float64,
    )
    result = bicluster_scale_normalize(X)
    expected = _scale_normalize(X)

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.allclose(result[2], expected[2])


def test_bicluster_bistochastic_normalize_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import bicluster_bistochastic_normalize

    X = _data()
    result = bicluster_bistochastic_normalize(X, max_iter=40, tol=1e-8)
    expected = _bistochastic_normalize(X, max_iter=40, tol=1e-8)

    assert np.allclose(result, expected)
    assert np.all(result >= 0.0)


def test_bicluster_log_normalize_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import bicluster_log_normalize

    X = np.array(
        [
            [-1.0, 0.0, 3.0],
            [2.0, 5.0, 1.0],
            [4.0, 1.5, 2.5],
        ],
        dtype=np.float64,
    )
    result = bicluster_log_normalize(X)
    expected = _log_normalize(X)

    assert np.allclose(result, expected)
    assert np.allclose(result.mean(axis=0), np.zeros(result.shape[1]))
    assert np.allclose(result.mean(axis=1), np.zeros(result.shape[0]))


def test_bicluster_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import (
        bicluster_bistochastic_normalize,
        bicluster_log_normalize,
        bicluster_scale_normalize,
    )

    with pytest.raises(ViolationError):
        bicluster_scale_normalize(np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        bicluster_scale_normalize(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(ViolationError):
        bicluster_bistochastic_normalize(_data(), max_iter=0)

    with pytest.raises(ViolationError):
        bicluster_bistochastic_normalize(_data(), tol=0.0)

    with pytest.raises(ViolationError):
        bicluster_log_normalize(np.array([1.0, 2.0], dtype=np.float64))
