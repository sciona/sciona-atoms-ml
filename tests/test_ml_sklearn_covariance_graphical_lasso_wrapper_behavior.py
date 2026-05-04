from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.covariance import graphical_lasso


def test_graphical_lasso_wrapper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_wrapper import (
        graphical_lasso_constructor_kwargs,
        graphical_lasso_return_values,
    )

    assert callable(graphical_lasso_constructor_kwargs)
    assert callable(graphical_lasso_return_values)


def test_graphical_lasso_wrapper_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_wrapper import (
        graphical_lasso_constructor_kwargs,
        graphical_lasso_return_values,
    )

    covariance = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    precision = np.array([[1.0, -0.5], [-0.5, 2.0]], dtype=np.float64)
    costs = ((1.5, 0.2), (1.3, 0.0))

    assert graphical_lasso_constructor_kwargs(0.1, mode="lars", tol=1e-3, enet_tol=2e-3, max_iter=7, verbose=2, eps=1e-8) == {
        "alpha": 0.1,
        "mode": "lars",
        "covariance": "precomputed",
        "tol": 1e-3,
        "enet_tol": 2e-3,
        "max_iter": 7,
        "verbose": 2,
        "eps": 1e-8,
        "assume_centered": True,
    }
    assert graphical_lasso_return_values(covariance, precision) == (covariance, precision)
    assert graphical_lasso_return_values(covariance, precision, return_costs=True, costs=costs) == (
        covariance,
        precision,
        costs,
    )
    assert graphical_lasso_return_values(covariance, precision, return_n_iter=True, n_iter=5) == (
        covariance,
        precision,
        5,
    )
    assert graphical_lasso_return_values(
        covariance,
        precision,
        return_costs=True,
        costs=costs,
        return_n_iter=True,
        n_iter=5,
    ) == (covariance, precision, costs, 5)


def test_graphical_lasso_wrapper_matches_wrapper_call_and_output() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_wrapper import (
        graphical_lasso_constructor_kwargs,
        graphical_lasso_return_values,
    )

    emp_cov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    covariance = np.array([[2.1, 0.4], [0.4, 1.2]], dtype=np.float64)
    precision = np.array([[0.7, -0.2], [-0.2, 1.0]], dtype=np.float64)
    costs = ((1.5, 0.1), (1.3, 0.0))
    model = MagicMock()
    model.fit.return_value = model
    model.covariance_ = covariance
    model.precision_ = precision
    model.costs_ = costs
    model.n_iter_ = 4

    with patch("sklearn.covariance._graph_lasso.GraphicalLasso", autospec=True, return_value=model) as ctor:
        result = graphical_lasso(
            emp_cov,
            0.2,
            mode="lars",
            tol=1e-3,
            enet_tol=2e-3,
            max_iter=9,
            verbose=1,
            return_costs=True,
            return_n_iter=True,
            eps=1e-8,
        )

    expected_kwargs = graphical_lasso_constructor_kwargs(
        0.2, mode="lars", tol=1e-3, enet_tol=2e-3, max_iter=9, verbose=1, eps=1e-8
    )
    assert ctor.call_args.kwargs == expected_kwargs
    model.fit.assert_called_once()
    assert np.array_equal(model.fit.call_args.args[0], emp_cov)
    expected_result = graphical_lasso_return_values(
        covariance,
        precision,
        return_costs=True,
        costs=costs,
        return_n_iter=True,
        n_iter=4,
    )
    assert result == expected_result


def test_graphical_lasso_wrapper_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_wrapper import (
        graphical_lasso_constructor_kwargs,
        graphical_lasso_return_values,
    )

    covariance = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    precision = np.array([[1.0, -0.5], [-0.5, 2.0]], dtype=np.float64)

    with pytest.raises(Exception):
        graphical_lasso_constructor_kwargs(0.0)

    with pytest.raises(Exception):
        graphical_lasso_constructor_kwargs(0.1, mode="bad")

    with pytest.raises(Exception):
        graphical_lasso_return_values(covariance, precision, return_costs=True, costs=None)

    with pytest.raises(Exception):
        graphical_lasso_return_values(covariance, precision, return_n_iter=True, n_iter=None)

    with pytest.raises(Exception):
        graphical_lasso_return_values(np.array([1.0, 2.0]), precision)  # type: ignore[arg-type]
