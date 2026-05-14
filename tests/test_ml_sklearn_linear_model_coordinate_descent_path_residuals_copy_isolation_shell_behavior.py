from __future__ import annotations

from collections.abc import Mapping

import pytest
from icontract import ViolationError


class MinimalMapping(Mapping[object, object]):
    def __init__(self, data: dict[object, object]) -> None:
        self._data = data

    def __getitem__(self, key: object) -> object:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_coordinate_descent_path_residuals_copy_isolation_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_copy_isolation_shell import (
        cd_path_residuals_path_params_copy,
    )

    assert callable(cd_path_residuals_path_params_copy)


def test_cd_path_residuals_path_params_copy_returns_distinct_shallow_dict() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_copy_isolation_shell import (
        cd_path_residuals_path_params_copy,
    )

    nested = {"inner": 1}
    path_params = {"precompute": "auto", "nested": nested, "l1_ratio": 0.5}

    copied = cd_path_residuals_path_params_copy(path_params)

    assert copied == path_params
    assert copied is not path_params
    assert copied["nested"] is nested


def test_cd_path_residuals_path_params_copy_isolates_top_level_mutation() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_copy_isolation_shell import (
        cd_path_residuals_path_params_copy,
    )

    path_params = {"precompute": "auto", "l1_ratio": 0.5}

    copied = cd_path_residuals_path_params_copy(path_params)
    copied["Xy"] = object()
    copied["copy_X"] = False
    copied["l1_ratio"] = 0.25

    assert "Xy" not in path_params
    assert "copy_X" not in path_params
    assert path_params["l1_ratio"] == 0.5


def test_cd_path_residuals_path_params_copy_accepts_mapping() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_copy_isolation_shell import (
        cd_path_residuals_path_params_copy,
    )

    value = object()
    mapping = MinimalMapping({"precompute": value})

    copied = cd_path_residuals_path_params_copy(mapping)

    assert copied == {"precompute": value}
    assert copied is not mapping
    assert copied["precompute"] is value


def test_cd_path_residuals_path_params_copy_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_copy_isolation_shell import (
        cd_path_residuals_path_params_copy,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_path_params_copy([("precompute", "auto")])
