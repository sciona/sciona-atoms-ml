from __future__ import annotations

import pytest


def test_hdbscan_tags_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tags import (
        hdbscan_allow_nan_tag,
        hdbscan_sparse_input_tag,
    )

    assert callable(hdbscan_sparse_input_tag)
    assert callable(hdbscan_allow_nan_tag)


def test_hdbscan_tags_match_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tags import (
        hdbscan_allow_nan_tag,
        hdbscan_sparse_input_tag,
    )

    assert hdbscan_sparse_input_tag("euclidean") is True
    assert hdbscan_sparse_input_tag("precomputed") is True
    assert hdbscan_allow_nan_tag("euclidean") is True
    assert hdbscan_allow_nan_tag("precomputed") is False


def test_hdbscan_tags_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tags import hdbscan_allow_nan_tag

    with pytest.raises(Exception):
        hdbscan_allow_nan_tag(None)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        hdbscan_sparse_input_tag(None)  # type: ignore[arg-type]
