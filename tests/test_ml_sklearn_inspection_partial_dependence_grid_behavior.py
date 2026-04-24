from __future__ import annotations

import numpy as np
import pytest
from sklearn.inspection._partial_dependence import _grid_from_X


def test_partial_dependence_grid_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import (
        partial_dependence_feature_axis,
        partial_dependence_grid,
        partial_dependence_grid_parameters,
    )

    assert callable(partial_dependence_grid_parameters)
    assert callable(partial_dependence_feature_axis)
    assert callable(partial_dependence_grid)


def test_partial_dependence_grid_parameters_match_source_validation_contract() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import partial_dependence_grid_parameters

    assert partial_dependence_grid_parameters((0.1, 0.9), grid_resolution=5) == (0.1, 0.9)

    with pytest.raises(Exception):
        partial_dependence_grid_parameters((0.9, 0.1), grid_resolution=5)
    with pytest.raises(Exception):
        partial_dependence_grid_parameters((0.1, 0.9), grid_resolution=1)


def test_partial_dependence_feature_axis_matches_unique_and_percentile_branches() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import partial_dependence_feature_axis

    low_resolution = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    assert np.array_equal(
        partial_dependence_feature_axis(low_resolution, percentiles=(0.05, 0.95), is_categorical=False, grid_resolution=10),
        np.unique(low_resolution),
    )

    categorical_values = np.array([3.0, 1.0, 2.0, 1.0], dtype=np.float64)
    assert np.array_equal(
        partial_dependence_feature_axis(categorical_values, percentiles=(0.05, 0.95), is_categorical=True, grid_resolution=3),
        np.unique(categorical_values),
    )

    numeric_values = np.array([0.0, 1.0, 3.0, 7.0, 9.0], dtype=np.float64)
    expected_grid, expected_values = _grid_from_X(
        numeric_values.reshape(-1, 1),
        percentiles=(0.1, 0.9),
        is_categorical=[False],
        grid_resolution=4,
    )
    del expected_grid
    result = partial_dependence_feature_axis(
        numeric_values,
        percentiles=(0.1, 0.9),
        is_categorical=False,
        grid_resolution=4,
    )
    assert np.allclose(result, expected_values[0])


def test_partial_dependence_grid_matches_private_helper_for_numeric_input() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import partial_dependence_grid

    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 12.0],
            [2.0, 14.0],
            [3.0, 16.0],
            [4.0, 18.0],
        ],
        dtype=np.float64,
    )

    expected_grid, expected_values = _grid_from_X(
        X,
        percentiles=(0.1, 0.9),
        is_categorical=[False, False],
        grid_resolution=3,
    )
    actual_grid, actual_values = partial_dependence_grid(
        X,
        percentiles=(0.1, 0.9),
        is_categorical=(False, False),
        grid_resolution=3,
    )

    assert np.allclose(actual_grid, expected_grid)
    assert len(actual_values) == len(expected_values)
    for actual, expected in zip(actual_values, expected_values):
        assert np.allclose(actual, expected)


def test_partial_dependence_grid_raises_when_percentiles_too_close() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import partial_dependence_feature_axis

    with pytest.raises(ValueError, match="percentiles are too close"):
        partial_dependence_feature_axis(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            percentiles=(0.49, 0.51),
            is_categorical=False,
            grid_resolution=2,
        )


def test_partial_dependence_grid_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_grid import (
        partial_dependence_feature_axis,
        partial_dependence_grid,
    )

    with pytest.raises(Exception):
        partial_dependence_feature_axis(np.array([1.0, np.nan], dtype=np.float64), percentiles=(0.1, 0.9), is_categorical=False, grid_resolution=3)
    with pytest.raises(Exception):
        partial_dependence_grid(
            np.ones((2, 2), dtype=np.float64),
            percentiles=(0.1, 0.9),
            is_categorical=(False,),
            grid_resolution=3,
        )
