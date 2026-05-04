from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import MinCovDet


def _fit_model() -> MinCovDet:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.2],
            [2.0, 1.3],
            [3.1, 2.0],
            [4.2, 2.7],
            [5.0, 3.6],
        ],
        dtype=np.float64,
    )
    model = MinCovDet(random_state=0)
    model.fit(X)
    return model


def test_mincovdet_postfit_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_postfit_state import (
        mincovdet_fit_covariance,
        mincovdet_fit_distances,
        mincovdet_fit_location,
        mincovdet_fit_raw_covariance,
        mincovdet_fit_raw_location,
        mincovdet_fit_raw_support,
        mincovdet_fit_return_self,
        mincovdet_fit_support,
    )

    assert callable(mincovdet_fit_raw_location)
    assert callable(mincovdet_fit_raw_covariance)
    assert callable(mincovdet_fit_raw_support)
    assert callable(mincovdet_fit_location)
    assert callable(mincovdet_fit_covariance)
    assert callable(mincovdet_fit_support)
    assert callable(mincovdet_fit_distances)
    assert callable(mincovdet_fit_return_self)


def test_mincovdet_postfit_state_matches_fitted_model() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_postfit_state import (
        mincovdet_fit_covariance,
        mincovdet_fit_distances,
        mincovdet_fit_location,
        mincovdet_fit_raw_covariance,
        mincovdet_fit_raw_location,
        mincovdet_fit_raw_support,
        mincovdet_fit_return_self,
        mincovdet_fit_support,
    )

    model = _fit_model()

    assert np.allclose(mincovdet_fit_raw_location(model.raw_location_), model.raw_location_)
    assert np.allclose(mincovdet_fit_raw_covariance(model.raw_covariance_), model.raw_covariance_)
    assert np.array_equal(mincovdet_fit_raw_support(model.raw_support_), model.raw_support_)
    assert np.allclose(mincovdet_fit_location(model.location_), model.location_)
    assert np.allclose(mincovdet_fit_covariance(model.covariance_), model.covariance_)
    assert np.array_equal(mincovdet_fit_support(model.support_), model.support_)
    assert np.allclose(mincovdet_fit_distances(model.dist_), model.dist_)
    assert mincovdet_fit_return_self("MinCovDet") == "MinCovDet"


def test_mincovdet_postfit_state_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_postfit_state import (
        mincovdet_fit_covariance,
        mincovdet_fit_distances,
        mincovdet_fit_location,
        mincovdet_fit_raw_covariance,
        mincovdet_fit_raw_location,
        mincovdet_fit_raw_support,
        mincovdet_fit_return_self,
        mincovdet_fit_support,
    )

    with pytest.raises(ViolationError):
        mincovdet_fit_raw_location(np.array([0.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        mincovdet_fit_raw_covariance(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        mincovdet_fit_raw_support(np.array([1, 0], dtype=np.int64))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        mincovdet_fit_location(np.array([np.inf], dtype=np.float64))

    with pytest.raises(ViolationError):
        mincovdet_fit_covariance(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        mincovdet_fit_support(np.array([True, 0], dtype=object))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        mincovdet_fit_distances(np.array([0.0, -1.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        mincovdet_fit_return_self("")
