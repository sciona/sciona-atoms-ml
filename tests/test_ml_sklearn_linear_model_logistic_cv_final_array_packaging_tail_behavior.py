from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_cv_final_array_packaging_tail_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_final_array_packaging_tail import (
        logistic_cv_C_array,
        logistic_cv_l1_ratio_array,
        logistic_cv_public_l1_ratios_array,
    )

    assert callable(logistic_cv_C_array)
    assert callable(logistic_cv_l1_ratio_array)
    assert callable(logistic_cv_public_l1_ratios_array)


def test_logistic_cv_final_array_packaging_matches_source_np_asarray() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_final_array_packaging_tail import (
        logistic_cv_C_array,
        logistic_cv_l1_ratio_array,
        logistic_cv_public_l1_ratios_array,
    )

    C_values = [np.float32(0.1), np.float32(1.0)]
    l1_values = [np.float64(0.25), np.float64(0.75)]
    public_l1_grid = [np.float64(0.25), np.float64(0.75)]

    C_result = logistic_cv_C_array(C_values)
    l1_result = logistic_cv_l1_ratio_array(l1_values)
    public_l1_result = logistic_cv_public_l1_ratios_array(public_l1_grid)

    np.testing.assert_array_equal(C_result, np.asarray(C_values))
    np.testing.assert_array_equal(l1_result, np.asarray(l1_values))
    np.testing.assert_array_equal(public_l1_result, np.asarray(public_l1_grid))
    assert C_result.dtype == np.asarray(C_values).dtype
    assert l1_result.dtype == np.asarray(l1_values).dtype
    assert public_l1_result.dtype == np.asarray(public_l1_grid).dtype


def test_logistic_cv_final_array_packaging_preserves_none_l1_values() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_final_array_packaging_tail import (
        logistic_cv_l1_ratio_array,
        logistic_cv_public_l1_ratios_array,
    )

    l1_result = logistic_cv_l1_ratio_array([None, None])
    public_l1_result = logistic_cv_public_l1_ratios_array([None])

    np.testing.assert_array_equal(l1_result, np.asarray([None, None]))
    np.testing.assert_array_equal(public_l1_result, np.asarray([None]))
    assert l1_result.dtype == object
    assert public_l1_result.dtype == object


def test_logistic_cv_final_array_packaging_accepts_pre_tiled_multinomial_arrays() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_final_array_packaging_tail import (
        logistic_cv_C_array,
        logistic_cv_l1_ratio_array,
    )

    C_values = np.tile([0.5], 3)
    l1_values = np.tile([None], 3)

    np.testing.assert_array_equal(logistic_cv_C_array(C_values), np.asarray(C_values))
    np.testing.assert_array_equal(logistic_cv_l1_ratio_array(l1_values), np.asarray(l1_values))


def test_logistic_cv_final_array_packaging_tail_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_final_array_packaging_tail import (
        logistic_cv_C_array,
        logistic_cv_l1_ratio_array,
        logistic_cv_public_l1_ratios_array,
    )

    with pytest.raises(ViolationError):
        logistic_cv_C_array([])

    with pytest.raises(ViolationError):
        logistic_cv_C_array([np.nan])

    with pytest.raises(ViolationError):
        logistic_cv_l1_ratio_array([])

    with pytest.raises(ViolationError):
        logistic_cv_public_l1_ratios_array([])
