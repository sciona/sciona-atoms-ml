from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import GraphicalLassoCV


def _fit_model() -> GraphicalLassoCV:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    model = GraphicalLassoCV(alphas=3, cv=2)
    model.fit(X)
    return model


def test_graphical_lasso_cv_postfit_state_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_postfit_state import (
        graphical_lasso_cv_fit_alpha,
        graphical_lasso_cv_fit_costs,
        graphical_lasso_cv_fit_covariance,
        graphical_lasso_cv_fit_n_iter,
        graphical_lasso_cv_fit_precision,
        graphical_lasso_cv_fit_return_self,
    )

    assert callable(graphical_lasso_cv_fit_alpha)
    assert callable(graphical_lasso_cv_fit_covariance)
    assert callable(graphical_lasso_cv_fit_precision)
    assert callable(graphical_lasso_cv_fit_costs)
    assert callable(graphical_lasso_cv_fit_n_iter)
    assert callable(graphical_lasso_cv_fit_return_self)


def test_graphical_lasso_cv_postfit_state_matches_fitted_model() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_postfit_state import (
        graphical_lasso_cv_fit_alpha,
        graphical_lasso_cv_fit_costs,
        graphical_lasso_cv_fit_covariance,
        graphical_lasso_cv_fit_n_iter,
        graphical_lasso_cv_fit_precision,
        graphical_lasso_cv_fit_return_self,
    )

    model = _fit_model()

    assert np.isclose(graphical_lasso_cv_fit_alpha(model.alpha_), model.alpha_)
    assert np.allclose(graphical_lasso_cv_fit_covariance(model.covariance_), model.covariance_)
    assert np.allclose(graphical_lasso_cv_fit_precision(model.precision_), model.precision_)
    assert graphical_lasso_cv_fit_costs(model.costs_) == model.costs_
    assert graphical_lasso_cv_fit_n_iter(model.n_iter_) == model.n_iter_
    assert graphical_lasso_cv_fit_return_self("GraphicalLassoCV") == "GraphicalLassoCV"


def test_graphical_lasso_cv_postfit_state_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_postfit_state import (
        graphical_lasso_cv_fit_alpha,
        graphical_lasso_cv_fit_costs,
        graphical_lasso_cv_fit_covariance,
        graphical_lasso_cv_fit_n_iter,
        graphical_lasso_cv_fit_precision,
        graphical_lasso_cv_fit_return_self,
    )

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_alpha(-0.1)

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_covariance(np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_precision(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_costs([(1.0, float("nan"))])

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_n_iter(0)

    with pytest.raises(ViolationError):
        graphical_lasso_cv_fit_return_self("")
