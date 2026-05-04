from __future__ import annotations

import pytest


def test_spectral_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_bookkeeping import (
        spectral_fit_n_components,
        spectral_fit_return_self,
        spectral_fit_use_cluster_qr,
        spectral_fit_use_kmeans,
        spectral_fit_verbose_message,
    )

    assert callable(spectral_fit_n_components)
    assert callable(spectral_fit_verbose_message)
    assert callable(spectral_fit_use_kmeans)
    assert callable(spectral_fit_use_cluster_qr)
    assert callable(spectral_fit_return_self)


def test_spectral_fit_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_bookkeeping import (
        spectral_fit_n_components,
        spectral_fit_return_self,
        spectral_fit_use_cluster_qr,
        spectral_fit_use_kmeans,
        spectral_fit_verbose_message,
    )

    assert spectral_fit_n_components(8, None) == 8
    assert spectral_fit_n_components(8, 5) == 5
    assert spectral_fit_verbose_message("cluster_qr") == "Computing label assignment using cluster_qr"
    assert spectral_fit_use_kmeans("kmeans") is True
    assert spectral_fit_use_kmeans("cluster_qr") is False
    assert spectral_fit_use_cluster_qr("cluster_qr") is True
    assert spectral_fit_use_cluster_qr("discretize") is False
    assert spectral_fit_return_self("SpectralClustering") == "SpectralClustering"


def test_spectral_fit_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.cluster.spectral_fit_bookkeeping import (
        spectral_fit_n_components,
        spectral_fit_return_self,
        spectral_fit_use_cluster_qr,
        spectral_fit_use_kmeans,
        spectral_fit_verbose_message,
    )

    with pytest.raises(Exception):
        spectral_fit_n_components(0, None)

    with pytest.raises(Exception):
        spectral_fit_n_components(2, 0)

    with pytest.raises(Exception):
        spectral_fit_verbose_message("")

    with pytest.raises(Exception):
        spectral_fit_use_kmeans("")

    with pytest.raises(Exception):
        spectral_fit_use_cluster_qr("")

    with pytest.raises(Exception):
        spectral_fit_return_self("")
