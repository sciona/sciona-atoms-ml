from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


def test_classification_fit_state_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_state_shell import (
        gpc_fit_binary_base_estimator,
        gpc_fit_binary_log_marginal_likelihood_value,
        gpc_fit_multiclass_log_marginal_likelihood_value,
        gpc_fit_one_vs_one_estimator,
        gpc_fit_one_vs_rest_estimator,
        gpc_fit_return_self,
    )

    assert callable(gpc_fit_binary_base_estimator)
    assert callable(gpc_fit_one_vs_rest_estimator)
    assert callable(gpc_fit_one_vs_one_estimator)
    assert callable(gpc_fit_binary_log_marginal_likelihood_value)
    assert callable(gpc_fit_multiclass_log_marginal_likelihood_value)
    assert callable(gpc_fit_return_self)


def test_gpc_fit_binary_base_estimator_preserves_constructor_arguments() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_state_shell import (
        gpc_fit_binary_base_estimator,
    )

    kernel = ConstantKernel(1.3, constant_value_bounds="fixed") * RBF(0.8, length_scale_bounds="fixed")
    estimator = gpc_fit_binary_base_estimator(
        kernel,
        optimizer=None,
        n_restarts_optimizer=2,
        max_iter_predict=17,
        warm_start=True,
        copy_X_train=False,
        random_state=7,
    )

    assert estimator.kernel is not kernel
    assert np.array_equal(estimator.kernel.theta, kernel.theta)
    assert estimator.optimizer is None
    assert estimator.n_restarts_optimizer == 2
    assert estimator.max_iter_predict == 17
    assert estimator.warm_start is True
    assert estimator.copy_X_train is False
    assert estimator.random_state == 7


def test_gpc_fit_multiclass_wrapper_construction_matches_sklearn_types() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_state_shell import (
        gpc_fit_binary_base_estimator,
        gpc_fit_one_vs_one_estimator,
        gpc_fit_one_vs_rest_estimator,
    )

    base = gpc_fit_binary_base_estimator(
        None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=9,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
    )

    ovr = gpc_fit_one_vs_rest_estimator(base, n_jobs=3)
    ovo = gpc_fit_one_vs_one_estimator(base, n_jobs=-1)

    assert isinstance(ovr, OneVsRestClassifier)
    assert ovr.estimator is base
    assert ovr.n_jobs == 3

    assert isinstance(ovo, OneVsOneClassifier)
    assert ovo.estimator is base
    assert ovo.n_jobs == -1


def test_gpc_fit_log_marginal_likelihood_values_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_state_shell import (
        gpc_fit_binary_log_marginal_likelihood_value,
        gpc_fit_multiclass_log_marginal_likelihood_value,
        gpc_fit_return_self,
    )

    values = np.array([-4.0, -2.5, -3.5], dtype=np.float64)

    assert gpc_fit_binary_log_marginal_likelihood_value(-1.75) == -1.75
    assert np.isclose(gpc_fit_multiclass_log_marginal_likelihood_value(values), float(np.mean(values)))
    assert gpc_fit_return_self("GaussianProcessClassifier") == "GaussianProcessClassifier"


def test_gpc_fit_state_shell_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_state_shell import (
        gpc_fit_binary_base_estimator,
        gpc_fit_multiclass_log_marginal_likelihood_value,
        gpc_fit_return_self,
    )

    with pytest.raises(ViolationError):
        gpc_fit_binary_base_estimator(
            None,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=-1,
            max_iter_predict=5,
            warm_start=False,
            copy_X_train=True,
            random_state=None,
        )

    with pytest.raises(ViolationError):
        gpc_fit_multiclass_log_marginal_likelihood_value(np.array([1.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        gpc_fit_return_self("")
