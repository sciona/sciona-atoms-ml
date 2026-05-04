from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def test_bicluster_svd_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_postprocessing import (
        bicluster_svd_left_vectors,
        bicluster_svd_right_vectors,
    )

    assert callable(bicluster_svd_left_vectors)
    assert callable(bicluster_svd_right_vectors)


def test_bicluster_svd_postprocessing_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_postprocessing import (
        bicluster_svd_left_vectors,
        bicluster_svd_right_vectors,
    )

    u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    vt = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float64)

    assert np.array_equal(
        bicluster_svd_left_vectors(u, 1),
        np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float64),
    )
    assert np.array_equal(
        bicluster_svd_right_vectors(vt, 1),
        np.array([[20.0, 30.0], [21.0, 31.0]], dtype=np.float64),
    )


def test_bicluster_svd_postprocessing_matches_base_spectral_svd() -> None:
    from sklearn.cluster import SpectralBiclustering

    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_postprocessing import (
        bicluster_svd_left_vectors,
        bicluster_svd_right_vectors,
    )

    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    u = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    vt = np.array(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]],
        dtype=np.float64,
    )
    model = SpectralBiclustering(n_clusters=2, n_components=2, method="log", svd_method="randomized", random_state=0)

    with patch("sklearn.cluster._bicluster.randomized_svd", autospec=True, return_value=(u, np.ones(3, dtype=np.float64), vt)):
        got_u, got_v = model._svd(array, n_components=3, n_discard=1)

    assert np.array_equal(got_u, bicluster_svd_left_vectors(u, 1))
    assert np.array_equal(got_v, bicluster_svd_right_vectors(vt, 1))


def test_bicluster_svd_postprocessing_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_postprocessing import (
        bicluster_svd_left_vectors,
        bicluster_svd_right_vectors,
    )

    with pytest.raises(Exception):
        bicluster_svd_left_vectors(np.array([[1.0]], dtype=np.float64), 1)

    with pytest.raises(Exception):
        bicluster_svd_right_vectors(np.array([[1.0]], dtype=np.float64), 1)
