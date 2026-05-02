from __future__ import annotations


def test_gpc_predict_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_predict_shell import (
        gpc_predict_dtype_name,
        gpc_predict_validate_ensure_2d,
    )

    assert callable(gpc_predict_dtype_name)
    assert callable(gpc_predict_validate_ensure_2d)


def test_gpc_predict_validation_mode_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_predict_shell import (
        gpc_predict_dtype_name,
        gpc_predict_validate_ensure_2d,
    )

    assert gpc_predict_dtype_name(True) == "numeric"
    assert gpc_predict_validate_ensure_2d(True) is True

    assert gpc_predict_dtype_name(False) is None
    assert gpc_predict_validate_ensure_2d(False) is False
