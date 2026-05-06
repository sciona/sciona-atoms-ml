from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_mono_output_normalization_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_mono_output_normalization import (
        cd_path_residuals_mono_output_coefs,
        cd_path_residuals_mono_output_y_offset,
        cd_path_residuals_mono_output_y_test,
        cd_path_residuals_use_mono_output_normalization,
    )

    assert callable(cd_path_residuals_use_mono_output_normalization)
    assert callable(cd_path_residuals_mono_output_coefs)
    assert callable(cd_path_residuals_mono_output_y_offset)
    assert callable(cd_path_residuals_mono_output_y_test)


def test_coordinate_descent_path_residuals_mono_output_normalization_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_mono_output_normalization import (
        cd_path_residuals_mono_output_coefs,
        cd_path_residuals_mono_output_y_offset,
        cd_path_residuals_mono_output_y_test,
        cd_path_residuals_use_mono_output_normalization,
    )

    coefs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y_test = np.array([5.0, 6.0, 7.0], dtype=np.float64)

    assert cd_path_residuals_use_mono_output_normalization(1) is True
    assert cd_path_residuals_use_mono_output_normalization(2) is False
    assert np.array_equal(cd_path_residuals_mono_output_coefs(coefs), coefs[np.newaxis, :, :])
    assert np.array_equal(cd_path_residuals_mono_output_y_offset(2.5), np.array([2.5]))
    assert np.array_equal(cd_path_residuals_mono_output_y_test(y_test), y_test[:, np.newaxis])


def test_coordinate_descent_path_residuals_mono_output_normalization_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_mono_output_normalization import (
        cd_path_residuals_mono_output_coefs,
        cd_path_residuals_mono_output_y_test,
        cd_path_residuals_use_mono_output_normalization,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_use_mono_output_normalization(0)

    with pytest.raises(ViolationError):
        cd_path_residuals_mono_output_coefs(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        cd_path_residuals_mono_output_y_test(np.array([[1.0], [2.0]], dtype=np.float64))
