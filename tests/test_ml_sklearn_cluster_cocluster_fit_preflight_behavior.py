from __future__ import annotations

import pytest


def test_cocluster_fit_preflight_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.cocluster_fit_preflight import cocluster_checked_n_clusters

    assert callable(cocluster_checked_n_clusters)


def test_cocluster_fit_preflight_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.cocluster_fit_preflight import cocluster_checked_n_clusters

    assert cocluster_checked_n_clusters(3, 7) == 3


def test_cocluster_fit_preflight_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.cocluster_fit_preflight import cocluster_checked_n_clusters

    with pytest.raises(ValueError, match="n_clusters should be <= n_samples=4. Got 5 instead."):
        cocluster_checked_n_clusters(5, 4)
