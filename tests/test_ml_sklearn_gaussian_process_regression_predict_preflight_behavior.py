from __future__ import annotations

import pytest


def test_gp_regression_predict_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_preflight import (
        gp_predict_dtype_name,
        gp_predict_require_single_uncertainty_mode,
        gp_predict_use_prior_branch,
        gp_predict_validate_ensure_2d,
    )

    assert callable(gp_predict_require_single_uncertainty_mode)
    assert callable(gp_predict_dtype_name)
    assert callable(gp_predict_validate_ensure_2d)
    assert callable(gp_predict_use_prior_branch)


def test_gp_predict_require_single_uncertainty_mode_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_preflight import (
        gp_predict_require_single_uncertainty_mode,
    )

    gp_predict_require_single_uncertainty_mode(False, False)
    gp_predict_require_single_uncertainty_mode(True, False)
    gp_predict_require_single_uncertainty_mode(False, True)

    with pytest.raises(RuntimeError, match="At most one of return_std or return_cov can be requested."):
        gp_predict_require_single_uncertainty_mode(True, True)


def test_gp_predict_validation_mode_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_preflight import (
        gp_predict_dtype_name,
        gp_predict_validate_ensure_2d,
    )

    assert gp_predict_dtype_name(True, False) == "numeric"
    assert gp_predict_validate_ensure_2d(True, False) is True

    assert gp_predict_dtype_name(False, True) == "numeric"
    assert gp_predict_validate_ensure_2d(False, True) is True

    assert gp_predict_dtype_name(False, False) is None
    assert gp_predict_validate_ensure_2d(False, False) is False


def test_gp_predict_use_prior_branch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.regression_predict_preflight import (
        gp_predict_use_prior_branch,
    )

    assert gp_predict_use_prior_branch(False) is True
    assert gp_predict_use_prior_branch(True) is False
