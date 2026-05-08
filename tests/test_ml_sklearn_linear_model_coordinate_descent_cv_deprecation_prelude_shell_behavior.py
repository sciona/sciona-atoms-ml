from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_deprecation_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_deprecation_prelude_shell import (
        cd_cv_alphas_none_deprecation_message,
        cd_cv_alphas_none_deprecation_warning_required,
        cd_cv_alphas_warn_sentinel,
        cd_cv_n_alphas_deprecation_message,
        cd_cv_n_alphas_deprecation_warning_required,
        cd_cv_resolved_alphas,
    )

    assert callable(cd_cv_n_alphas_deprecation_warning_required)
    assert callable(cd_cv_n_alphas_deprecation_message)
    assert callable(cd_cv_alphas_warn_sentinel)
    assert callable(cd_cv_alphas_none_deprecation_warning_required)
    assert callable(cd_cv_alphas_none_deprecation_message)
    assert callable(cd_cv_resolved_alphas)


def test_coordinate_descent_cv_deprecation_prelude_shell_matches_sklearn_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_deprecation_prelude_shell import (
        cd_cv_alphas_none_deprecation_message,
        cd_cv_alphas_none_deprecation_warning_required,
        cd_cv_alphas_warn_sentinel,
        cd_cv_n_alphas_deprecation_message,
        cd_cv_n_alphas_deprecation_warning_required,
        cd_cv_resolved_alphas,
    )

    explicit_grid = np.array([0.2, 0.1])

    assert cd_cv_n_alphas_deprecation_warning_required("deprecated") is False
    assert cd_cv_n_alphas_deprecation_warning_required(7) is True
    assert "n_alphas" in cd_cv_n_alphas_deprecation_message(True)
    assert "removed in 1.9" in cd_cv_n_alphas_deprecation_message(True)

    assert cd_cv_alphas_warn_sentinel("warn") is True
    assert cd_cv_alphas_warn_sentinel(None) is False
    assert cd_cv_alphas_none_deprecation_warning_required(None) is True
    assert cd_cv_alphas_none_deprecation_warning_required("warn") is False
    assert "alphas=None" in cd_cv_alphas_none_deprecation_message(True)

    assert cd_cv_resolved_alphas("deprecated", "warn") == 100
    assert cd_cv_resolved_alphas("deprecated", None) == 100
    assert cd_cv_resolved_alphas(12, "warn") == 12
    assert cd_cv_resolved_alphas(12, None) == 12
    assert cd_cv_resolved_alphas("deprecated", explicit_grid) is explicit_grid
    assert cd_cv_resolved_alphas(12, explicit_grid) is explicit_grid


def test_coordinate_descent_cv_deprecation_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_deprecation_prelude_shell import (
        cd_cv_alphas_none_deprecation_message,
        cd_cv_n_alphas_deprecation_message,
        cd_cv_n_alphas_deprecation_warning_required,
        cd_cv_resolved_alphas,
    )

    with pytest.raises(ViolationError):
        cd_cv_n_alphas_deprecation_warning_required(0)

    with pytest.raises(ViolationError):
        cd_cv_n_alphas_deprecation_message(False)

    with pytest.raises(ViolationError):
        cd_cv_alphas_none_deprecation_message(False)

    with pytest.raises(ViolationError):
        cd_cv_resolved_alphas(True, "warn")
