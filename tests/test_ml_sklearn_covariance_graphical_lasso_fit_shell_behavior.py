from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import GraphicalLasso


def _fit_sample_model() -> GraphicalLasso:
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.5],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    model = GraphicalLasso(alpha=0.1)
    model.fit(X)
    return model


def _fit_precomputed_model() -> GraphicalLasso:
    emp_cov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    model = GraphicalLasso(covariance="precomputed")
    model.fit(emp_cov)
    return model


def test_graphical_lasso_fit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell import (
        graphical_lasso_fit_costs,
        graphical_lasso_fit_covariance,
        graphical_lasso_fit_empirical_covariance,
        graphical_lasso_fit_location,
        graphical_lasso_fit_n_iter,
        graphical_lasso_fit_precision,
        graphical_lasso_fit_return_self,
        graphical_lasso_fit_use_precomputed_covariance,
    )

    assert callable(graphical_lasso_fit_use_precomputed_covariance)
    assert callable(graphical_lasso_fit_empirical_covariance)
    assert callable(graphical_lasso_fit_location)
    assert callable(graphical_lasso_fit_covariance)
    assert callable(graphical_lasso_fit_precision)
    assert callable(graphical_lasso_fit_costs)
    assert callable(graphical_lasso_fit_n_iter)
    assert callable(graphical_lasso_fit_return_self)


def test_graphical_lasso_fit_shell_matches_sklearn_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance import empirical_covariance
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell import (
        graphical_lasso_fit_costs,
        graphical_lasso_fit_covariance,
        graphical_lasso_fit_empirical_covariance,
        graphical_lasso_fit_location,
        graphical_lasso_fit_n_iter,
        graphical_lasso_fit_precision,
        graphical_lasso_fit_return_self,
        graphical_lasso_fit_use_precomputed_covariance,
    )

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.5],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )
    sample_model = _fit_sample_model()
    precomputed_model = _fit_precomputed_model()
    precomputed = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)

    assert graphical_lasso_fit_use_precomputed_covariance("precomputed") is True
    assert graphical_lasso_fit_use_precomputed_covariance("raw") is False

    assert np.allclose(
        graphical_lasso_fit_empirical_covariance(X, "raw", assume_centered=False),
        empirical_covariance(X, assume_centered=False),
    )
    assert np.allclose(
        graphical_lasso_fit_empirical_covariance(precomputed, "precomputed", assume_centered=False),
        precomputed,
    )

    assert np.allclose(graphical_lasso_fit_location(X, "raw", assume_centered=False), sample_model.location_)
    assert np.allclose(graphical_lasso_fit_location(X, "raw", assume_centered=True), np.zeros(X.shape[1], dtype=np.float64))
    assert np.allclose(
        graphical_lasso_fit_location(precomputed, "precomputed", assume_centered=False),
        precomputed_model.location_,
    )

    assert np.allclose(graphical_lasso_fit_covariance(sample_model.covariance_), sample_model.covariance_)
    assert np.allclose(graphical_lasso_fit_precision(sample_model.precision_), sample_model.precision_)
    assert graphical_lasso_fit_costs(sample_model.costs_) == sample_model.costs_
    assert graphical_lasso_fit_n_iter(sample_model.n_iter_) == sample_model.n_iter_
    assert graphical_lasso_fit_return_self("GraphicalLasso") == "GraphicalLasso"


def test_graphical_lasso_fit_shell_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_fit_shell import (
        graphical_lasso_fit_costs,
        graphical_lasso_fit_covariance,
        graphical_lasso_fit_empirical_covariance,
        graphical_lasso_fit_location,
        graphical_lasso_fit_n_iter,
        graphical_lasso_fit_precision,
        graphical_lasso_fit_return_self,
        graphical_lasso_fit_use_precomputed_covariance,
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ViolationError):
        graphical_lasso_fit_use_precomputed_covariance("")

    with pytest.raises(ViolationError):
        graphical_lasso_fit_empirical_covariance(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64), "raw", assume_centered=False)

    with pytest.raises(ViolationError):
        graphical_lasso_fit_location(X, "", assume_centered=False)

    with pytest.raises(ViolationError):
        graphical_lasso_fit_covariance(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        graphical_lasso_fit_precision(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        graphical_lasso_fit_costs([(1.0, float("nan"))])

    with pytest.raises(ViolationError):
        graphical_lasso_fit_n_iter(0)

    with pytest.raises(ViolationError):
        graphical_lasso_fit_return_self("")
