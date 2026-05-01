from __future__ import annotations

import pytest


def test_bicluster_fit_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_preflight import (
        bicluster_checked_cluster_counts,
        bicluster_checked_method,
        bicluster_checked_n_best,
    )

    assert callable(bicluster_checked_cluster_counts)
    assert callable(bicluster_checked_method)
    assert callable(bicluster_checked_n_best)


def test_bicluster_fit_preflight_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_preflight import (
        bicluster_checked_cluster_counts,
        bicluster_checked_method,
        bicluster_checked_n_best,
    )

    assert bicluster_checked_cluster_counts(3, 7) == (3, 3)
    assert bicluster_checked_cluster_counts((2, 5), 7) == (2, 5)
    assert bicluster_checked_n_best(3, 6) == 3
    assert bicluster_checked_method("bistochastic", True) == "bistochastic"
    assert bicluster_checked_method("log", False) == "log"


def test_bicluster_fit_preflight_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_fit_preflight import (
        bicluster_checked_cluster_counts,
        bicluster_checked_method,
        bicluster_checked_n_best,
    )

    with pytest.raises(ValueError, match="n_clusters should be <= n_samples=4. Got 5 instead."):
        bicluster_checked_cluster_counts(5, 4)

    with pytest.raises(ValueError, match="Incorrect parameter n_clusters has value:"):
        bicluster_checked_cluster_counts((0, 2), 4)

    with pytest.raises(ValueError, match="n_best=4 must be <= n_components=3."):
        bicluster_checked_n_best(4, 3)

    with pytest.raises(ValueError, match="Cannot compute log of a sparse matrix,"):
        bicluster_checked_method("log", True)
