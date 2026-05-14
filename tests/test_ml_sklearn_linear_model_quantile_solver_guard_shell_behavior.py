from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from packaging.version import parse as parse_version
from scipy import sparse
from scipy import __version__ as scipy_version
from sklearn.linear_model import QuantileRegressor


def test_quantile_solver_guard_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_solver_guard_shell import (
        quantile_interior_point_removed_guard,
        quantile_interior_point_removed_message,
        quantile_solver_options_payload,
        quantile_sparse_solver_guard,
        quantile_sparse_solver_message,
    )

    assert callable(quantile_interior_point_removed_guard)
    assert callable(quantile_interior_point_removed_message)
    assert callable(quantile_sparse_solver_guard)
    assert callable(quantile_sparse_solver_message)
    assert callable(quantile_solver_options_payload)


def test_quantile_interior_point_removed_guard_matches_sklearn_error() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_solver_guard_shell import (
        quantile_interior_point_removed_guard,
        quantile_interior_point_removed_message,
    )

    scipy_at_least_1_11 = parse_version(scipy_version) >= parse_version("1.11.0")

    assert quantile_interior_point_removed_guard("interior-point", scipy_at_least_1_11) is True
    assert quantile_interior_point_removed_guard("highs", scipy_at_least_1_11) is False
    with pytest.raises(ValueError, match="not anymore available"):
        QuantileRegressor(solver="interior-point").fit(np.ones((3, 1)), np.arange(3.0))

    expected = "Solver interior-point is not anymore available in SciPy >= 1.11.0."
    assert quantile_interior_point_removed_message("interior-point") == expected


def test_quantile_sparse_solver_guard_matches_sklearn_error() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_solver_guard_shell import (
        quantile_sparse_solver_guard,
        quantile_sparse_solver_message,
    )

    X = sparse.csr_matrix([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert quantile_sparse_solver_guard(True, "revised simplex") is True
    assert quantile_sparse_solver_guard(True, "highs") is False
    assert quantile_sparse_solver_guard(False, "revised simplex") is False
    with pytest.raises(ValueError, match="does not support sparse X"):
        QuantileRegressor(solver="revised simplex").fit(X, y)

    expected = "Solver revised simplex does not support sparse X. Use solver 'highs' for example."
    assert quantile_sparse_solver_message("revised simplex") == expected


def test_quantile_solver_options_payload_matches_source_branching() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_solver_guard_shell import (
        quantile_solver_options_payload,
    )

    explicit = {"presolve": False}

    assert quantile_solver_options_payload(None, "interior-point") == {"lstsq": True}
    assert quantile_solver_options_payload(None, "highs") is None
    assert quantile_solver_options_payload(explicit, "interior-point") is explicit
    assert quantile_solver_options_payload(explicit, "highs") is explicit


def test_quantile_solver_guard_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.quantile_solver_guard_shell import (
        quantile_interior_point_removed_guard,
        quantile_interior_point_removed_message,
        quantile_solver_options_payload,
        quantile_sparse_solver_guard,
        quantile_sparse_solver_message,
    )

    with pytest.raises(ViolationError):
        quantile_interior_point_removed_guard("", True)

    with pytest.raises(ViolationError):
        quantile_interior_point_removed_guard("interior-point", 1)

    with pytest.raises(ViolationError):
        quantile_interior_point_removed_message("highs")

    with pytest.raises(ViolationError):
        quantile_sparse_solver_guard(True, "")

    with pytest.raises(ViolationError):
        quantile_sparse_solver_guard(1, "highs")

    with pytest.raises(ViolationError):
        quantile_sparse_solver_message("highs")

    with pytest.raises(ViolationError):
        quantile_solver_options_payload(["lstsq"], "interior-point")
