from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence as sklearn_partial_dependence
from sklearn.utils import Bunch


def _fitted_regressor() -> tuple[RandomForestRegressor, np.ndarray]:
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    est = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    est.fit(X, y)
    return est, X


def test_partial_dependence_result_packaging_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging import (
        partial_dependence_grid_shaped_averages,
        partial_dependence_grid_shaped_individual,
        partial_dependence_grid_value_lengths,
        partial_dependence_result_bunch,
    )

    assert callable(partial_dependence_grid_value_lengths)
    assert callable(partial_dependence_grid_shaped_averages)
    assert callable(partial_dependence_grid_shaped_individual)
    assert callable(partial_dependence_result_bunch)


def test_partial_dependence_result_packaging_matches_sklearn_outputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging import (
        partial_dependence_grid_shaped_averages,
        partial_dependence_grid_shaped_individual,
        partial_dependence_grid_value_lengths,
        partial_dependence_result_bunch,
    )

    est, X = _fitted_regressor()
    expected = sklearn_partial_dependence(
        est,
        X,
        features=[0, 1],
        method="brute",
        kind="both",
        grid_resolution=6,
    )

    lengths = partial_dependence_grid_value_lengths(tuple(expected.grid_values))
    average_flat = np.asarray(expected.average, dtype=np.float64).reshape(expected.average.shape[0], -1)
    individual_flat = np.asarray(expected.individual, dtype=np.float64).reshape(expected.individual.shape[0], expected.individual.shape[1], -1)

    average = partial_dependence_grid_shaped_averages(average_flat, lengths)
    individual = partial_dependence_grid_shaped_individual(individual_flat, lengths)
    bunch = partial_dependence_result_bunch(
        "both",
        tuple(expected.grid_values),
        average=average,
        individual=individual,
    )

    assert lengths == tuple(len(values) for values in expected.grid_values)
    assert np.allclose(average, expected.average)
    assert np.allclose(individual, expected.individual)
    assert isinstance(bunch, Bunch)
    assert np.allclose(bunch["average"], expected.average)
    assert np.allclose(bunch["individual"], expected.individual)
    assert bunch["grid_values"] == tuple(expected.grid_values)


def test_partial_dependence_result_packaging_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_result_packaging import (
        partial_dependence_grid_shaped_averages,
        partial_dependence_grid_shaped_individual,
        partial_dependence_grid_value_lengths,
        partial_dependence_result_bunch,
    )

    with pytest.raises(ViolationError):
        partial_dependence_grid_value_lengths(tuple())

    with pytest.raises(ViolationError):
        partial_dependence_grid_shaped_averages(np.ones((1, 5), dtype=np.float64), (2, 2))

    with pytest.raises(ViolationError):
        partial_dependence_grid_shaped_individual(np.ones((1, 3, 5), dtype=np.float64), (2, 2))

    with pytest.raises(ViolationError):
        partial_dependence_result_bunch("average", (np.array([0.0, 1.0]),), individual=np.ones((1, 2), dtype=np.float64))
