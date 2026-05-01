from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from scipy.linalg import norm
from sklearn.cluster._bicluster import _bistochastic_normalize, _scale_normalize
from sklearn.utils.extmath import make_nonnegative


def _sparse_data() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.5, 3.0, 0.0],
                [4.0, 1.5, 2.5],
            ],
            dtype=np.float64,
        )
    )


def test_bicluster_sparse_preprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_bistochastic_distance,
        bicluster_sparse_bistochastic_normalize,
        bicluster_sparse_scale_normalize,
    )

    assert callable(bicluster_sparse_scale_normalize)
    assert callable(bicluster_sparse_bistochastic_distance)
    assert callable(bicluster_sparse_bistochastic_normalize)


def test_bicluster_sparse_scale_normalize_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_scale_normalize,
    )

    X = _sparse_data()
    result = bicluster_sparse_scale_normalize(X)
    expected = _scale_normalize(X)

    assert sp.issparse(result[0])
    assert np.allclose(result[0].toarray(), expected[0].toarray())
    assert np.allclose(result[1], expected[1])
    assert np.allclose(result[2], expected[2])


def test_bicluster_sparse_bistochastic_distance_matches_sklearn_sparse_branch() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_bistochastic_distance,
        bicluster_sparse_scale_normalize,
    )

    X = _sparse_data()
    shifted = make_nonnegative(X.copy())
    scaled, _, _ = bicluster_sparse_scale_normalize(shifted)
    result = bicluster_sparse_bistochastic_distance(scaled, shifted)
    expected = float(norm(scaled.data - shifted.data))

    assert result == pytest.approx(expected)
    assert result >= 0.0


def test_bicluster_sparse_bistochastic_normalize_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_bistochastic_normalize,
    )

    X = _sparse_data()
    result = bicluster_sparse_bistochastic_normalize(X, max_iter=40, tol=1e-8)
    expected = _bistochastic_normalize(X, max_iter=40, tol=1e-8)

    assert sp.issparse(result)
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.all(result.data >= 0.0)


def test_bicluster_sparse_preprocessing_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_bistochastic_distance,
        bicluster_sparse_bistochastic_normalize,
        bicluster_sparse_scale_normalize,
    )

    with pytest.raises(ViolationError):
        bicluster_sparse_scale_normalize(np.array([[1.0, 2.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        bicluster_sparse_bistochastic_normalize(_sparse_data(), max_iter=0)

    with pytest.raises(ViolationError):
        bicluster_sparse_bistochastic_normalize(_sparse_data(), tol=0.0)

    with pytest.raises(ViolationError):
        bicluster_sparse_bistochastic_distance(
            sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)),
            sp.csr_matrix(np.array([[1.0, 0.0, 1.0]], dtype=np.float64)),
        )
