from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process import GaussianProcessRegressor


def _fit_single_output_model() -> GaussianProcessRegressor:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    model = GaussianProcessRegressor(alpha=1e-5, optimizer=None)
    model.fit(X, y)
    return model


def _fit_multi_output_model() -> GaussianProcessRegressor:
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.column_stack(
        [
            np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float64),
        ]
    )
    model = GaussianProcessRegressor(alpha=1e-5, optimizer=None, normalize_y=True)
    model.fit(X, y)
    return model


def test_gp_regression_postfit_attributes_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_postfit_attributes import (
        gp_regression_fit_L,
        gp_regression_fit_alpha,
        gp_regression_fit_log_marginal_likelihood_value,
        gp_regression_fit_return_self,
    )

    assert callable(gp_regression_fit_L)
    assert callable(gp_regression_fit_alpha)
    assert callable(gp_regression_fit_log_marginal_likelihood_value)
    assert callable(gp_regression_fit_return_self)


def test_gp_regression_postfit_attributes_match_fitted_models() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_postfit_attributes import (
        gp_regression_fit_L,
        gp_regression_fit_alpha,
        gp_regression_fit_log_marginal_likelihood_value,
        gp_regression_fit_return_self,
    )

    single = _fit_single_output_model()
    multi = _fit_multi_output_model()

    assert np.allclose(gp_regression_fit_L(single.L_), single.L_)
    assert np.allclose(gp_regression_fit_alpha(single.alpha_), single.alpha_)
    assert np.isclose(
        gp_regression_fit_log_marginal_likelihood_value(single.log_marginal_likelihood_value_),
        single.log_marginal_likelihood_value_,
    )

    assert np.allclose(gp_regression_fit_L(multi.L_), multi.L_)
    assert np.allclose(gp_regression_fit_alpha(multi.alpha_), multi.alpha_)
    assert np.isclose(
        gp_regression_fit_log_marginal_likelihood_value(multi.log_marginal_likelihood_value_),
        multi.log_marginal_likelihood_value_,
    )

    assert gp_regression_fit_return_self("GaussianProcessRegressor") == "GaussianProcessRegressor"


def test_gp_regression_postfit_attributes_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_postfit_attributes import (
        gp_regression_fit_L,
        gp_regression_fit_alpha,
        gp_regression_fit_log_marginal_likelihood_value,
        gp_regression_fit_return_self,
    )

    with pytest.raises(ViolationError):
        gp_regression_fit_L(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_regression_fit_alpha(np.array([0.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        gp_regression_fit_log_marginal_likelihood_value(float("nan"))

    with pytest.raises(ViolationError):
        gp_regression_fit_return_self("")
