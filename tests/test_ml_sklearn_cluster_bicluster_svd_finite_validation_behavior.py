from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering


def test_bicluster_svd_finite_validation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_finite_validation import (
        bicluster_svd_checked_u,
        bicluster_svd_checked_vt,
    )

    assert callable(bicluster_svd_checked_u)
    assert callable(bicluster_svd_checked_vt)


def test_bicluster_svd_finite_validation_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_finite_validation import (
        bicluster_svd_checked_u,
        bicluster_svd_checked_vt,
    )

    u = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    vt = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    assert np.array_equal(bicluster_svd_checked_u(u), u)
    assert np.array_equal(bicluster_svd_checked_vt(vt), vt)


def test_bicluster_svd_finite_validation_matches_arpack_success_path() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_finite_validation import (
        bicluster_svd_checked_u,
        bicluster_svd_checked_vt,
    )

    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    model = SpectralBiclustering(
        n_clusters=2,
        n_components=2,
        method="log",
        svd_method="arpack",
        random_state=13,
        n_svd_vecs=7,
    )

    u = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    vt = np.array([[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]], dtype=np.float64)

    with patch(
        "sklearn.cluster._bicluster.svds",
        autospec=True,
        return_value=(u, np.ones(2, dtype=np.float64), vt),
    ):
        got_u, got_v = model._svd(array, n_components=2, n_discard=0)

    assert np.array_equal(bicluster_svd_checked_u(u), got_u)
    assert np.array_equal(bicluster_svd_checked_vt(vt).T, got_v)


def test_bicluster_svd_finite_validation_matches_source_failures() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_finite_validation import (
        bicluster_svd_checked_u,
        bicluster_svd_checked_vt,
    )

    model = SpectralBiclustering(
        n_clusters=2,
        n_components=2,
        method="log",
        svd_method="randomized",
        random_state=13,
    )
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    bad_u = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    good_vt = np.array([[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        bicluster_svd_checked_u(bad_u)

    with patch(
        "sklearn.cluster._bicluster.randomized_svd",
        autospec=True,
        return_value=(bad_u, np.ones(2, dtype=np.float64), good_vt),
    ):
        with pytest.raises(ValueError):
            model._svd(array, n_components=2, n_discard=0)

    bad_vt = np.array([[6.0, 7.0, np.inf, 9.0], [10.0, 11.0, 12.0, 13.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        bicluster_svd_checked_vt(bad_vt)

    with patch(
        "sklearn.cluster._bicluster.randomized_svd",
        autospec=True,
        return_value=(np.abs(bad_u), np.ones(2, dtype=np.float64), bad_vt),
    ):
        with pytest.raises(ValueError):
            model._svd(array, n_components=2, n_discard=0)


def test_bicluster_svd_finite_validation_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_finite_validation import (
        bicluster_svd_checked_u,
        bicluster_svd_checked_vt,
    )

    with pytest.raises(Exception):
        bicluster_svd_checked_u(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(Exception):
        bicluster_svd_checked_vt(np.array([1.0, 2.0], dtype=np.float64))
