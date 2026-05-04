from __future__ import annotations

import pytest
from icontract import ViolationError


def test_classification_fit_multiclass_mode_guard_atom_import() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_multiclass_mode_guard import (
        gpc_fit_require_supported_multiclass_mode,
    )

    assert callable(gpc_fit_require_supported_multiclass_mode)


def test_gpc_fit_multiclass_mode_guard_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_multiclass_mode_guard import (
        gpc_fit_require_supported_multiclass_mode,
    )

    assert gpc_fit_require_supported_multiclass_mode(2, "weird_mode") == "weird_mode"
    assert gpc_fit_require_supported_multiclass_mode(4, "one_vs_rest") == "one_vs_rest"
    assert gpc_fit_require_supported_multiclass_mode(4, "one_vs_one") == "one_vs_one"


def test_gpc_fit_multiclass_mode_guard_rejects_unknown_mode_only_for_multiclass() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_multiclass_mode_guard import (
        gpc_fit_require_supported_multiclass_mode,
    )

    with pytest.raises(ValueError, match=r"Unknown multi-class mode strange_mode"):
        gpc_fit_require_supported_multiclass_mode(3, "strange_mode")


def test_gpc_fit_multiclass_mode_guard_contracts() -> None:
    from sciona.atoms.ml.sklearn.gaussian_process.classification_fit_multiclass_mode_guard import (
        gpc_fit_require_supported_multiclass_mode,
    )

    with pytest.raises(ViolationError):
        gpc_fit_require_supported_multiclass_mode(0, "one_vs_rest")

    with pytest.raises(ViolationError):
        gpc_fit_require_supported_multiclass_mode(3, "")
