from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from icontract import ViolationError
from sklearn.utils import check_array


def test_permutation_preflight_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preflight_shell import (
        permutation_importance_checked_array,
        permutation_importance_max_samples_guard_required,
        permutation_importance_use_dataframe_passthrough,
    )

    assert callable(permutation_importance_use_dataframe_passthrough)
    assert callable(permutation_importance_checked_array)
    assert callable(permutation_importance_max_samples_guard_required)


def test_permutation_preflight_shell_matches_sklearn_branches() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preflight_shell import (
        permutation_importance_checked_array,
        permutation_importance_max_samples_guard_required,
        permutation_importance_use_dataframe_passthrough,
    )

    X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    expected = np.asarray(check_array(X, ensure_all_finite="allow-nan", dtype=None), dtype=np.float64)

    assert permutation_importance_use_dataframe_passthrough(hasattr(pd.DataFrame(X), "iloc")) is True
    assert permutation_importance_use_dataframe_passthrough(hasattr(X, "iloc")) is False
    assert np.array_equal(permutation_importance_checked_array(X), expected, equal_nan=True)
    assert permutation_importance_max_samples_guard_required(4, 3) is True
    assert permutation_importance_max_samples_guard_required(3, 3) is False
    assert permutation_importance_max_samples_guard_required(0.5, 3) is False


def test_permutation_preflight_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_preflight_shell import (
        permutation_importance_checked_array,
        permutation_importance_max_samples_guard_required,
        permutation_importance_use_dataframe_passthrough,
    )

    with pytest.raises(ViolationError):
        permutation_importance_use_dataframe_passthrough(1)

    with pytest.raises(ViolationError):
        permutation_importance_checked_array(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        permutation_importance_max_samples_guard_required(0, 3)
