from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering


def test_bicluster_svd_nan_recovery_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_nan_recovery_shell import (
        bicluster_svd_arpack_init_vector,
        bicluster_svd_eigsh_kwargs,
        bicluster_svd_left_gram_matrix,
        bicluster_svd_right_gram_matrix,
        bicluster_svd_u_nan_recovery_required,
        bicluster_svd_vt_nan_recovery_required,
    )

    assert callable(bicluster_svd_vt_nan_recovery_required)
    assert callable(bicluster_svd_u_nan_recovery_required)
    assert callable(bicluster_svd_right_gram_matrix)
    assert callable(bicluster_svd_left_gram_matrix)
    assert callable(bicluster_svd_arpack_init_vector)
    assert callable(bicluster_svd_eigsh_kwargs)


def test_bicluster_svd_nan_recovery_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_nan_recovery_shell import (
        bicluster_svd_arpack_init_vector,
        bicluster_svd_eigsh_kwargs,
        bicluster_svd_left_gram_matrix,
        bicluster_svd_right_gram_matrix,
        bicluster_svd_u_nan_recovery_required,
        bicluster_svd_vt_nan_recovery_required,
    )

    vt = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float64)
    u = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float64)
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    assert bicluster_svd_vt_nan_recovery_required(vt) is True
    assert bicluster_svd_u_nan_recovery_required(u) is True
    assert np.array_equal(
        bicluster_svd_right_gram_matrix(array),
        array.T @ array,
    )
    assert np.array_equal(
        bicluster_svd_left_gram_matrix(array),
        array @ array.T,
    )
    v0 = bicluster_svd_arpack_init_vector(7, 3)
    assert v0.shape == (3,)
    assert np.all(v0 >= -1.0)
    assert np.all(v0 <= 1.0)
    assert bicluster_svd_eigsh_kwargs(9, v0) == {"ncv": 9, "v0": v0}


def test_bicluster_svd_nan_recovery_shell_matches_arpack_fallback_calls() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_nan_recovery_shell import (
        bicluster_svd_arpack_init_vector,
        bicluster_svd_eigsh_kwargs,
        bicluster_svd_left_gram_matrix,
        bicluster_svd_right_gram_matrix,
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
    vt = np.array([[np.nan, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    recovered_v = np.array(
        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
        dtype=np.float64,
    )

    with (
        patch("sklearn.cluster._bicluster.svds", autospec=True, return_value=(u, np.ones(2, dtype=np.float64), vt)),
        patch("sklearn.cluster._bicluster.eigsh", autospec=True, return_value=(np.ones(2, dtype=np.float64), recovered_v)) as eigsh_mock,
    ):
        got_u, got_v = model._svd(array, n_components=2, n_discard=0)

    expected_A = bicluster_svd_right_gram_matrix(array)
    expected_v0 = bicluster_svd_arpack_init_vector(model.random_state, expected_A.shape[0])  # type: ignore[arg-type]
    assert np.array_equal(eigsh_mock.call_args.args[0], expected_A)
    expected_kwargs = bicluster_svd_eigsh_kwargs(model.n_svd_vecs, expected_v0)
    assert eigsh_mock.call_args.kwargs["ncv"] == expected_kwargs["ncv"]
    assert np.array_equal(eigsh_mock.call_args.kwargs["v0"], expected_kwargs["v0"])
    assert np.array_equal(got_u, u)
    assert np.array_equal(got_v, recovered_v)

    u_nan = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    vt_ok = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    recovered_u = np.array(
        [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]],
        dtype=np.float64,
    )

    with (
        patch("sklearn.cluster._bicluster.svds", autospec=True, return_value=(u_nan, np.ones(2, dtype=np.float64), vt_ok)),
        patch("sklearn.cluster._bicluster.eigsh", autospec=True, return_value=(np.ones(2, dtype=np.float64), recovered_u)) as eigsh_mock,
    ):
        got_u, got_v = model._svd(array, n_components=2, n_discard=0)

    expected_A = bicluster_svd_left_gram_matrix(array)
    expected_v0 = bicluster_svd_arpack_init_vector(model.random_state, expected_A.shape[0])  # type: ignore[arg-type]
    assert np.array_equal(eigsh_mock.call_args.args[0], expected_A)
    expected_kwargs = bicluster_svd_eigsh_kwargs(model.n_svd_vecs, expected_v0)
    assert eigsh_mock.call_args.kwargs["ncv"] == expected_kwargs["ncv"]
    assert np.array_equal(eigsh_mock.call_args.kwargs["v0"], expected_kwargs["v0"])
    assert np.array_equal(got_u, recovered_u)
    assert np.array_equal(got_v, vt_ok.T)


def test_bicluster_svd_nan_recovery_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_nan_recovery_shell import (
        bicluster_svd_arpack_init_vector,
        bicluster_svd_eigsh_kwargs,
        bicluster_svd_left_gram_matrix,
        bicluster_svd_right_gram_matrix,
        bicluster_svd_u_nan_recovery_required,
        bicluster_svd_vt_nan_recovery_required,
    )

    with pytest.raises(Exception):
        bicluster_svd_vt_nan_recovery_required(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(Exception):
        bicluster_svd_u_nan_recovery_required(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(Exception):
        bicluster_svd_right_gram_matrix(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        bicluster_svd_left_gram_matrix(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        bicluster_svd_arpack_init_vector("seed", 2)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        bicluster_svd_eigsh_kwargs(0, np.array([0.1, 0.2], dtype=np.float64))
