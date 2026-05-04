from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering


def _sample_data() -> np.ndarray:
    return np.array(
        [
            [3.0, 3.1, 0.1, 0.2],
            [2.9, 3.2, 0.0, 0.1],
            [0.2, 0.1, 4.0, 4.1],
            [0.1, 0.2, 4.2, 3.9],
        ],
        dtype=np.float64,
    )


def test_bicluster_postfit_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_postfit_state import (
        bicluster_fit_column_labels,
        bicluster_fit_columns,
        bicluster_fit_return_self,
        bicluster_fit_row_labels,
        bicluster_fit_rows,
    )

    assert callable(bicluster_fit_row_labels)
    assert callable(bicluster_fit_column_labels)
    assert callable(bicluster_fit_rows)
    assert callable(bicluster_fit_columns)
    assert callable(bicluster_fit_return_self)


def test_bicluster_postfit_state_matches_fitted_models() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_postfit_state import (
        bicluster_fit_column_labels,
        bicluster_fit_columns,
        bicluster_fit_return_self,
        bicluster_fit_row_labels,
        bicluster_fit_rows,
    )

    X = _sample_data()
    bicluster = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
    cocluster = SpectralCoclustering(n_clusters=2, random_state=0).fit(X)

    for model, token in [
        (bicluster, "SpectralBiclustering"),
        (cocluster, "SpectralCoclustering"),
    ]:
        assert np.array_equal(bicluster_fit_row_labels(model.row_labels_), model.row_labels_)
        assert np.array_equal(bicluster_fit_column_labels(model.column_labels_), model.column_labels_)
        assert np.array_equal(bicluster_fit_rows(model.rows_), model.rows_)
        assert np.array_equal(bicluster_fit_columns(model.columns_), model.columns_)
        assert bicluster_fit_return_self(token) == token


def test_bicluster_postfit_state_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_postfit_state import (
        bicluster_fit_column_labels,
        bicluster_fit_columns,
        bicluster_fit_return_self,
        bicluster_fit_row_labels,
        bicluster_fit_rows,
    )

    with pytest.raises(ViolationError):
        bicluster_fit_row_labels(np.array([0, -1], dtype=np.int64))

    with pytest.raises(ViolationError):
        bicluster_fit_column_labels(np.array([-1, 0], dtype=np.int64))

    with pytest.raises(ViolationError):
        bicluster_fit_rows(np.array([[1, 0]], dtype=np.int64))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        bicluster_fit_columns(np.array([[True], [np.nan]], dtype=object))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        bicluster_fit_return_self("")
