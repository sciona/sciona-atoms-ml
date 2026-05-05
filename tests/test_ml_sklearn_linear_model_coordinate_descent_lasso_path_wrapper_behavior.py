from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_lasso_path_wrapper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_path_wrapper import (
        cd_lasso_path_call_kwargs,
        cd_lasso_path_result,
    )

    assert callable(cd_lasso_path_call_kwargs)
    assert callable(cd_lasso_path_result)


def test_coordinate_descent_lasso_path_wrapper_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_path_wrapper import (
        cd_lasso_path_call_kwargs,
        cd_lasso_path_result,
    )

    kwargs = cd_lasso_path_call_kwargs(
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        precompute="auto",
        Xy=None,
        copy_X=True,
        coef_init=None,
        verbose=False,
        positive=False,
        return_n_iter=True,
        params={"tol": 1e-4},
    )
    assert kwargs["l1_ratio"] == 1.0
    assert kwargs["tol"] == 1e-4
    assert kwargs["return_n_iter"] is True

    result = (("alphas",), ("coefs",), ("dual_gaps",), [1, 2])
    assert cd_lasso_path_result(result, True) == result


def test_coordinate_descent_lasso_path_wrapper_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_lasso_path_wrapper import (
        cd_lasso_path_call_kwargs,
        cd_lasso_path_result,
    )

    with pytest.raises(ViolationError):
        cd_lasso_path_call_kwargs(
            eps=0.0,
            n_alphas=100,
            alphas=None,
            precompute="auto",
            Xy=None,
            copy_X=True,
            coef_init=None,
            verbose=False,
            positive=False,
            return_n_iter=False,
            params={},
        )

    with pytest.raises(ViolationError):
        cd_lasso_path_result((1, 2), False)
