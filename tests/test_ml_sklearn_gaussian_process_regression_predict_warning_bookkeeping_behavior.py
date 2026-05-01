from __future__ import annotations

import numpy as np


def test_gp_regression_predict_warning_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_warning_bookkeeping import (
        gp_predict_negative_variance_mask,
        gp_predict_negative_variance_warning_required,
        gp_predict_nonnegative_variance,
    )

    assert callable(gp_predict_negative_variance_mask)
    assert callable(gp_predict_negative_variance_warning_required)
    assert callable(gp_predict_nonnegative_variance)


def test_gp_regression_predict_warning_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_warning_bookkeeping import (
        gp_predict_negative_variance_mask,
        gp_predict_negative_variance_warning_required,
        gp_predict_nonnegative_variance,
    )

    y_var = np.array([0.2, -1e-12, 0.0, -0.5], dtype=np.float64)
    mask = gp_predict_negative_variance_mask(y_var)
    clipped = gp_predict_nonnegative_variance(y_var, mask)

    assert np.array_equal(mask, np.array([False, True, False, True], dtype=np.bool_))
    assert gp_predict_negative_variance_warning_required(mask) is True
    assert np.allclose(clipped, np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float64))


def test_gp_regression_predict_warning_bookkeeping_no_warning_case() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_warning_bookkeeping import (
        gp_predict_negative_variance_mask,
        gp_predict_negative_variance_warning_required,
    )

    y_var = np.array([0.2, 0.0, 1.5], dtype=np.float64)
    mask = gp_predict_negative_variance_mask(y_var)
    assert np.array_equal(mask, np.array([False, False, False], dtype=np.bool_))
    assert gp_predict_negative_variance_warning_required(mask) is False
