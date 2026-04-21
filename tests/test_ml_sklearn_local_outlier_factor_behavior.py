from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor as SklearnLocalOutlierFactor


def _lof_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-1.1, 0.0],
            [0.2, 0.1],
            [0.3, -0.1],
            [0.0, 0.2],
            [4.0, 4.0],
            [0.1, -0.2],
            [-0.2, 0.1],
        ],
        dtype=np.float64,
    )
    query = np.array([[0.1, 0.0], [3.5, 3.8], [-0.7, 0.1]], dtype=np.float64)
    return X, query


def test_local_outlier_factor_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        LocalOutlierFactorState,
        local_outlier_factor_decision_function,
        local_outlier_factor_fit,
        local_outlier_factor_fit_predict,
        local_outlier_factor_predict,
        local_outlier_factor_score_samples,
    )

    assert LocalOutlierFactorState is not None
    assert callable(local_outlier_factor_fit)
    assert callable(local_outlier_factor_fit_predict)
    assert callable(local_outlier_factor_score_samples)
    assert callable(local_outlier_factor_decision_function)
    assert callable(local_outlier_factor_predict)


def test_local_outlier_factor_fit_predict_matches_sklearn_training_labels() -> None:
    from sciona.atoms.ml.sklearn.neighbors import local_outlier_factor_fit, local_outlier_factor_fit_predict

    X, _ = _lof_data()
    expected = SklearnLocalOutlierFactor(n_neighbors=2, contamination=0.2)
    labels = expected.fit_predict(X)
    state = local_outlier_factor_fit(X, n_neighbors=2, contamination=0.2)
    assert np.array_equal(local_outlier_factor_fit_predict(X, n_neighbors=2, contamination=0.2), labels)
    assert np.allclose(state.negative_outlier_factor, expected.negative_outlier_factor_)
    assert np.allclose(state.offset, expected.offset_)


def test_local_outlier_factor_novelty_scores_and_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.neighbors import (
        local_outlier_factor_decision_function,
        local_outlier_factor_fit,
        local_outlier_factor_predict,
        local_outlier_factor_score_samples,
    )

    X, query = _lof_data()
    state = local_outlier_factor_fit(X, n_neighbors=2, contamination=0.2, novelty=True)
    expected = SklearnLocalOutlierFactor(n_neighbors=2, contamination=0.2, novelty=True).fit(X)
    assert np.allclose(local_outlier_factor_score_samples(query, state), expected.score_samples(query))
    assert np.allclose(local_outlier_factor_decision_function(query, state), expected.decision_function(query))
    assert np.array_equal(local_outlier_factor_predict(query, state), expected.predict(query))


def test_local_outlier_factor_matches_sklearn_manhattan_and_auto_contamination() -> None:
    from sciona.atoms.ml.sklearn.neighbors import local_outlier_factor_fit, local_outlier_factor_score_samples

    X, query = _lof_data()
    state = local_outlier_factor_fit(X, n_neighbors=3, p=1.0, novelty=True)
    expected = SklearnLocalOutlierFactor(n_neighbors=3, p=1.0, novelty=True).fit(X)
    assert np.allclose(state.negative_outlier_factor, expected.negative_outlier_factor_)
    assert np.allclose(local_outlier_factor_score_samples(query, state), expected.score_samples(query))


def test_local_outlier_factor_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors import local_outlier_factor_fit, local_outlier_factor_predict

    X, query = _lof_data()
    with pytest.raises(Exception):
        local_outlier_factor_fit(X[:1], n_neighbors=1)
    with pytest.raises(Exception):
        local_outlier_factor_fit(X, n_neighbors=0)
    with pytest.raises(Exception):
        local_outlier_factor_fit(X, metric="cosine")
    with pytest.raises(Exception):
        local_outlier_factor_fit(X, contamination=0.75)
    state = local_outlier_factor_fit(X, n_neighbors=2, novelty=False)
    with pytest.raises(Exception):
        local_outlier_factor_predict(query, state)
