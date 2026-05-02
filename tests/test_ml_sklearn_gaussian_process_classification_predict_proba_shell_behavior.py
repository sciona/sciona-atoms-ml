from __future__ import annotations

import pytest


def test_gpc_predict_proba_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_predict_proba_shell import (
        gpc_predict_proba_dtype_name,
        gpc_predict_proba_require_supported_multiclass_mode,
        gpc_predict_proba_validate_ensure_2d,
    )

    assert callable(gpc_predict_proba_require_supported_multiclass_mode)
    assert callable(gpc_predict_proba_dtype_name)
    assert callable(gpc_predict_proba_validate_ensure_2d)


def test_gpc_predict_proba_one_vs_one_probability_guard_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_predict_proba_shell import (
        gpc_predict_proba_require_supported_multiclass_mode,
    )

    assert gpc_predict_proba_require_supported_multiclass_mode(2, "one_vs_one") is True
    assert gpc_predict_proba_require_supported_multiclass_mode(3, "one_vs_rest") is True

    with pytest.raises(
        ValueError,
        match="one_vs_one multi-class mode does not support predicting probability estimates. Use one_vs_rest mode instead.",
    ):
        gpc_predict_proba_require_supported_multiclass_mode(3, "one_vs_one")


def test_gpc_predict_proba_validation_mode_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_predict_proba_shell import (
        gpc_predict_proba_dtype_name,
        gpc_predict_proba_validate_ensure_2d,
    )

    assert gpc_predict_proba_dtype_name(True) == "numeric"
    assert gpc_predict_proba_validate_ensure_2d(True) is True

    assert gpc_predict_proba_dtype_name(False) is None
    assert gpc_predict_proba_validate_ensure_2d(False) is False
