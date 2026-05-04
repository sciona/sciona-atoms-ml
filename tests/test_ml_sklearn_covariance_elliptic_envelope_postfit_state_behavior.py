from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import EllipticEnvelope


def _fit_model() -> EllipticEnvelope:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.1],
            [2.0, 1.1],
            [3.0, 2.2],
            [4.1, 2.9],
            [5.1, 3.8],
        ],
        dtype=np.float64,
    )
    model = EllipticEnvelope(contamination=0.2, random_state=0)
    model.fit(X)
    return model


def test_elliptic_envelope_postfit_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postfit_state import (
        elliptic_envelope_fit_covariance,
        elliptic_envelope_fit_distances,
        elliptic_envelope_fit_location,
        elliptic_envelope_fit_offset,
        elliptic_envelope_fit_precision,
        elliptic_envelope_fit_raw_covariance,
        elliptic_envelope_fit_raw_location,
        elliptic_envelope_fit_raw_support,
        elliptic_envelope_fit_return_self,
        elliptic_envelope_fit_support,
    )

    assert callable(elliptic_envelope_fit_raw_location)
    assert callable(elliptic_envelope_fit_raw_covariance)
    assert callable(elliptic_envelope_fit_raw_support)
    assert callable(elliptic_envelope_fit_location)
    assert callable(elliptic_envelope_fit_covariance)
    assert callable(elliptic_envelope_fit_precision)
    assert callable(elliptic_envelope_fit_support)
    assert callable(elliptic_envelope_fit_distances)
    assert callable(elliptic_envelope_fit_offset)
    assert callable(elliptic_envelope_fit_return_self)


def test_elliptic_envelope_postfit_state_matches_fitted_model() -> None:
    from sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postfit_state import (
        elliptic_envelope_fit_covariance,
        elliptic_envelope_fit_distances,
        elliptic_envelope_fit_location,
        elliptic_envelope_fit_offset,
        elliptic_envelope_fit_precision,
        elliptic_envelope_fit_raw_covariance,
        elliptic_envelope_fit_raw_location,
        elliptic_envelope_fit_raw_support,
        elliptic_envelope_fit_return_self,
        elliptic_envelope_fit_support,
    )

    model = _fit_model()

    assert np.allclose(elliptic_envelope_fit_raw_location(model.raw_location_), model.raw_location_)
    assert np.allclose(elliptic_envelope_fit_raw_covariance(model.raw_covariance_), model.raw_covariance_)
    assert np.array_equal(elliptic_envelope_fit_raw_support(model.raw_support_), model.raw_support_)
    assert np.allclose(elliptic_envelope_fit_location(model.location_), model.location_)
    assert np.allclose(elliptic_envelope_fit_covariance(model.covariance_), model.covariance_)
    assert np.allclose(elliptic_envelope_fit_precision(model.precision_), model.precision_)
    assert np.array_equal(elliptic_envelope_fit_support(model.support_), model.support_)
    assert np.allclose(elliptic_envelope_fit_distances(model.dist_), model.dist_)
    assert np.isclose(elliptic_envelope_fit_offset(model.offset_), model.offset_)
    assert elliptic_envelope_fit_return_self("EllipticEnvelope") == "EllipticEnvelope"


def test_elliptic_envelope_postfit_state_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.elliptic_envelope_postfit_state import (
        elliptic_envelope_fit_covariance,
        elliptic_envelope_fit_distances,
        elliptic_envelope_fit_location,
        elliptic_envelope_fit_offset,
        elliptic_envelope_fit_precision,
        elliptic_envelope_fit_raw_covariance,
        elliptic_envelope_fit_raw_location,
        elliptic_envelope_fit_raw_support,
        elliptic_envelope_fit_return_self,
        elliptic_envelope_fit_support,
    )

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_raw_location(np.array([0.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_raw_covariance(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_raw_support(np.array([1, 0], dtype=np.int64))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_location(np.array([np.inf], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_covariance(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_precision(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_support(np.array([True, 0], dtype=object))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_distances(np.array([0.0, -1.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_offset(float("nan"))

    with pytest.raises(ViolationError):
        elliptic_envelope_fit_return_self("")
