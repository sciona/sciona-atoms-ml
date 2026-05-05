from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_column_slice_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_slice_shell import (
        partial_dependence_slice_column_indices,
        partial_dependence_slice_stop_exclusive,
        partial_dependence_slice_uses_default_stop,
    )

    assert callable(partial_dependence_slice_uses_default_stop)
    assert callable(partial_dependence_slice_stop_exclusive)
    assert callable(partial_dependence_slice_column_indices)


def test_partial_dependence_column_slice_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_slice_shell import (
        partial_dependence_slice_column_indices,
        partial_dependence_slice_stop_exclusive,
        partial_dependence_slice_uses_default_stop,
    )

    assert partial_dependence_slice_uses_default_stop(None) is True
    assert partial_dependence_slice_uses_default_stop(3) is False

    assert partial_dependence_slice_stop_exclusive(None, 5) == 6
    assert partial_dependence_slice_stop_exclusive(3, 5) == 4

    assert partial_dependence_slice_column_indices(5, None, 3) == (0, 1, 2)
    assert partial_dependence_slice_column_indices(5, 1, 4) == (1, 2, 3)
    assert partial_dependence_slice_column_indices(5, 4, 6) == (4,)


def test_partial_dependence_column_slice_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_slice_shell import (
        partial_dependence_slice_column_indices,
        partial_dependence_slice_stop_exclusive,
        partial_dependence_slice_uses_default_stop,
    )

    with pytest.raises(ViolationError):
        partial_dependence_slice_uses_default_stop(-1)

    with pytest.raises(ViolationError):
        partial_dependence_slice_stop_exclusive(-1, 5)

    with pytest.raises(ViolationError):
        partial_dependence_slice_column_indices(5, 6, 3)
