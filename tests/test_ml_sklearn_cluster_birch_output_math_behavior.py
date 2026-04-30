from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import Birch as SklearnBirch
from sklearn.metrics import pairwise_distances_argmin


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [4.9, 5.0],
            [5.0, 4.9],
            [5.1, 5.0],
        ],
        dtype=np.float64,
    )


def test_birch_output_math_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_output_math import (
        birch_predict_argmin,
        birch_predict_labels,
        birch_subcluster_norms,
        birch_transform_distances,
    )

    assert callable(birch_subcluster_norms)
    assert callable(birch_predict_argmin)
    assert callable(birch_predict_labels)
    assert callable(birch_transform_distances)


def test_birch_output_math_matches_sklearn_no_global_outputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_output_math import (
        birch_predict_argmin,
        birch_predict_labels,
        birch_subcluster_norms,
        birch_transform_distances,
    )

    X = _data()
    model = SklearnBirch(n_clusters=None, threshold=0.3, branching_factor=10).fit(X)

    norms = birch_subcluster_norms(model.subcluster_centers_)
    argmin = birch_predict_argmin(X, model.subcluster_centers_, norms)
    labels = birch_predict_labels(argmin, model.subcluster_labels_)
    distances = birch_transform_distances(X, model.subcluster_centers_)

    expected_argmin = pairwise_distances_argmin(
        X,
        model.subcluster_centers_,
        metric_kwargs={"Y_norm_squared": model._subcluster_norms},
    )

    assert np.allclose(norms, model._subcluster_norms)
    assert np.array_equal(argmin, expected_argmin.astype(np.int64))
    assert np.array_equal(labels, model.predict(X).astype(np.int64))
    assert np.allclose(distances, model.transform(X))


def test_birch_output_math_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_output_math import (
        birch_predict_argmin,
        birch_predict_labels,
        birch_subcluster_norms,
        birch_transform_distances,
    )

    X = _data()

    with pytest.raises(ViolationError):
        birch_subcluster_norms(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_predict_argmin(X, np.array([[0.0], [1.0]], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_predict_labels(np.array([0, 2], dtype=np.int64), np.array([0, 1], dtype=np.int64))

    with pytest.raises(ViolationError):
        birch_transform_distances(X, np.array([[0.0], [1.0]], dtype=np.float64))
