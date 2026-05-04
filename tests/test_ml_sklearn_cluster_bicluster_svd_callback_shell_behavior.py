from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def test_bicluster_svd_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_callback_shell import (
        bicluster_svd_randomized_kwargs,
        bicluster_svd_svds_kwargs,
        bicluster_svd_use_arpack,
        bicluster_svd_use_randomized,
    )

    assert callable(bicluster_svd_use_randomized)
    assert callable(bicluster_svd_use_arpack)
    assert callable(bicluster_svd_randomized_kwargs)
    assert callable(bicluster_svd_svds_kwargs)


def test_bicluster_svd_callback_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_callback_shell import (
        bicluster_svd_randomized_kwargs,
        bicluster_svd_svds_kwargs,
        bicluster_svd_use_arpack,
        bicluster_svd_use_randomized,
    )

    rng = np.random.RandomState(7)
    assert bicluster_svd_use_randomized("randomized") is True
    assert bicluster_svd_use_randomized("arpack") is False
    assert bicluster_svd_use_arpack("arpack") is True
    assert bicluster_svd_use_arpack("randomized") is False
    assert bicluster_svd_randomized_kwargs(rng, None) == {"random_state": rng}
    assert bicluster_svd_randomized_kwargs(rng, 9) == {
        "random_state": rng,
        "n_oversamples": 9,
    }
    assert bicluster_svd_svds_kwargs(5, None) == {"k": 5, "ncv": None}
    assert bicluster_svd_svds_kwargs(5, 11) == {"k": 5, "ncv": 11}


def test_bicluster_svd_callback_shell_matches_base_spectral_calls() -> None:
    from sklearn.cluster import SpectralBiclustering

    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_callback_shell import (
        bicluster_svd_randomized_kwargs,
        bicluster_svd_svds_kwargs,
    )

    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    model_randomized = SpectralBiclustering(
        n_clusters=2,
        n_components=2,
        method="log",
        svd_method="randomized",
        random_state=13,
        n_svd_vecs=7,
    )
    with patch(
        "sklearn.cluster._bicluster.randomized_svd",
        autospec=True,
        return_value=(
            np.ones((3, 3), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.ones((3, 4), dtype=np.float64),
        ),
    ) as randomized_mock:
        model_randomized._svd(array, n_components=3, n_discard=1)
    assert randomized_mock.call_args.args[:2] == (array, 3)
    assert randomized_mock.call_args.kwargs == bicluster_svd_randomized_kwargs(
        model_randomized.random_state,
        model_randomized.n_svd_vecs,
    )

    model_arpack = SpectralBiclustering(
        n_clusters=2,
        n_components=2,
        method="log",
        svd_method="arpack",
        n_svd_vecs=9,
    )
    with patch(
        "sklearn.cluster._bicluster.svds",
        autospec=True,
        return_value=(
            np.ones((3, 3), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.ones((3, 4), dtype=np.float64),
        ),
    ) as svds_mock:
        model_arpack._svd(array, n_components=3, n_discard=1)
    assert svds_mock.call_args.args[0] is array
    assert svds_mock.call_args.kwargs == bicluster_svd_svds_kwargs(
        3,
        model_arpack.n_svd_vecs,
    )


def test_bicluster_svd_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_svd_callback_shell import (
        bicluster_svd_randomized_kwargs,
        bicluster_svd_svds_kwargs,
        bicluster_svd_use_arpack,
        bicluster_svd_use_randomized,
    )

    with pytest.raises(Exception):
        bicluster_svd_use_randomized("")

    with pytest.raises(Exception):
        bicluster_svd_use_arpack("")

    with pytest.raises(Exception):
        bicluster_svd_randomized_kwargs("seed", 3)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        bicluster_svd_svds_kwargs(0, None)
