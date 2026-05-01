from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def test_bicluster_fit_normalization_dispatch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_normalization_dispatch import (
        bicluster_dense_normalized_data,
        bicluster_sparse_normalized_data,
    )

    assert callable(bicluster_dense_normalized_data)
    assert callable(bicluster_sparse_normalized_data)


def test_bicluster_fit_normalization_dispatch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_normalization_dispatch import (
        bicluster_dense_normalized_data,
        bicluster_sparse_normalized_data,
    )
    from sciona.atoms.ml.sklearn.cluster.bicluster import (
        bicluster_bistochastic_normalize,
        bicluster_log_normalize,
        bicluster_scale_normalize,
    )
    from sciona.atoms.ml.sklearn.cluster.bicluster_sparse_preprocessing import (
        bicluster_sparse_bistochastic_normalize,
        bicluster_sparse_scale_normalize,
    )

    dense = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    sparse = sp.csr_matrix(dense)

    dense_scale, _, _ = bicluster_scale_normalize(dense)
    sparse_scale, _, _ = bicluster_sparse_scale_normalize(sparse)

    assert np.allclose(bicluster_dense_normalized_data(dense, "scale"), dense_scale)
    assert np.allclose(bicluster_dense_normalized_data(dense, "bistochastic"), bicluster_bistochastic_normalize(dense))
    assert np.allclose(bicluster_dense_normalized_data(dense, "log"), bicluster_log_normalize(dense))

    assert np.allclose(bicluster_sparse_normalized_data(sparse, "scale").toarray(), sparse_scale.toarray())
    assert np.allclose(
        bicluster_sparse_normalized_data(sparse, "bistochastic").toarray(),
        bicluster_sparse_bistochastic_normalize(sparse).toarray(),
    )


def test_bicluster_fit_normalization_dispatch_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_normalization_dispatch import (
        bicluster_dense_normalized_data,
        bicluster_sparse_normalized_data,
    )

    with pytest.raises(Exception):
        bicluster_dense_normalized_data(np.array([[1.0]], dtype=np.float64), "invalid")

    with pytest.raises(Exception):
        bicluster_sparse_normalized_data(sp.csr_matrix([[1.0]]), "log")
