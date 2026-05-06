from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.model_selection import KFold


def test_coordinate_descent_cv_splitter_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_splitter_callback_shell import (
        cd_cv_checked_cv,
        cd_cv_split_iterator,
        cd_cv_split_kwargs,
    )

    assert callable(cd_cv_checked_cv)
    assert callable(cd_cv_split_kwargs)
    assert callable(cd_cv_split_iterator)


def test_coordinate_descent_cv_splitter_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_splitter_callback_shell import (
        cd_cv_checked_cv,
        cd_cv_split_iterator,
        cd_cv_split_kwargs,
    )

    cv = KFold(n_splits=2)
    assert cd_cv_checked_cv(cv) is cv

    split_params: dict[str, object] = {}
    assert cd_cv_split_kwargs(split_params) == {}

    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([0, 1, 0], dtype=np.int64)
    split_iterator = cv.split(X, y, **split_params)
    assert cd_cv_split_iterator(split_iterator) is split_iterator


def test_coordinate_descent_cv_splitter_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_splitter_callback_shell import (
        cd_cv_split_iterator,
        cd_cv_split_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_cv_split_kwargs([("sample_weight", [1.0, 2.0])])

    with pytest.raises(ViolationError):
        cd_cv_split_iterator(3)
