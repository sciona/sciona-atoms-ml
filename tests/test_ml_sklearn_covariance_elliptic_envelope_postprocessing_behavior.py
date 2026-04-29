from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import EllipticEnvelope

from sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postprocessing import (
    elliptic_envelope_decision_function,
    elliptic_envelope_labels,
    elliptic_envelope_offset,
    elliptic_envelope_score_samples,
)


def _training_data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.1, 0.0],
            [0.0, 0.2],
            [-0.1, 0.1],
            [4.0, 4.0],
            [4.5, 3.8],
        ],
        dtype=np.float64,
    )


def _query_data() -> np.ndarray:
    return np.array(
        [
            [0.1, 0.1],
            [4.2, 4.1],
            [0.3, -0.2],
        ],
        dtype=np.float64,
    )


def test_elliptic_envelope_postprocessing_atoms_import() -> None:
    assert callable(elliptic_envelope_offset)
    assert callable(elliptic_envelope_score_samples)
    assert callable(elliptic_envelope_decision_function)
    assert callable(elliptic_envelope_labels)


def test_elliptic_envelope_offset_matches_fitted_offset() -> None:
    X = _training_data()
    model = EllipticEnvelope(contamination=0.25, random_state=0).fit(X)

    training_scores = elliptic_envelope_score_samples(np.asarray(model.dist_, dtype=np.float64))
    observed = elliptic_envelope_offset(training_scores, float(model.contamination))

    assert np.isclose(observed, float(model.offset_))


def test_elliptic_envelope_score_and_decision_match_sklearn() -> None:
    X = _training_data()
    query = _query_data()
    model = EllipticEnvelope(contamination=0.25, random_state=0).fit(X)

    mahalanobis = np.asarray(model.mahalanobis(query), dtype=np.float64)
    observed_scores = elliptic_envelope_score_samples(mahalanobis)
    observed_decision = elliptic_envelope_decision_function(observed_scores, float(model.offset_))

    assert np.allclose(observed_scores, model.score_samples(query))
    assert np.allclose(observed_decision, model.decision_function(query))


def test_elliptic_envelope_labels_match_predict() -> None:
    X = _training_data()
    query = _query_data()
    model = EllipticEnvelope(contamination=0.25, random_state=0).fit(X)

    decision = np.asarray(model.decision_function(query), dtype=np.float64)
    observed = elliptic_envelope_labels(decision)

    assert np.array_equal(observed, model.predict(query))


def test_elliptic_envelope_postprocessing_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        elliptic_envelope_offset(np.array([np.nan], dtype=np.float64), 0.1)

    with pytest.raises((ViolationError, ValueError)):
        elliptic_envelope_offset(np.array([-1.0], dtype=np.float64), 0.0)

    with pytest.raises((ViolationError, ValueError)):
        elliptic_envelope_score_samples(np.array([-1.0], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        elliptic_envelope_decision_function(np.array([0.0], dtype=np.float64), float("nan"))

    with pytest.raises((ViolationError, ValueError)):
        elliptic_envelope_labels(np.array([np.nan], dtype=np.float64))
