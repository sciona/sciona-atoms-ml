from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_partial_dependence_column_lookup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_lookup_shell import (
        partial_dependence_column_indices_appended,
        partial_dependence_missing_column_message,
        partial_dependence_nonunique_column_guard_required,
        partial_dependence_nonunique_column_message,
    )

    assert callable(partial_dependence_nonunique_column_guard_required)
    assert callable(partial_dependence_nonunique_column_message)
    assert callable(partial_dependence_column_indices_appended)
    assert callable(partial_dependence_missing_column_message)


def test_partial_dependence_column_lookup_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_lookup_shell import (
        partial_dependence_column_indices_appended,
        partial_dependence_missing_column_message,
        partial_dependence_nonunique_column_guard_required,
        partial_dependence_nonunique_column_message,
    )

    assert partial_dependence_nonunique_column_guard_required(3) is False
    assert partial_dependence_nonunique_column_guard_required(np.int64(4)) is False
    assert partial_dependence_nonunique_column_guard_required(slice(0, 2)) is True
    assert partial_dependence_nonunique_column_guard_required(np.array([1, 2], dtype=np.int64)) is True

    assert (
        partial_dependence_nonunique_column_message(("age", "height"))
        == "Selected columns, ['age', 'height'], are not unique in dataframe"
    )

    assert partial_dependence_column_indices_appended(tuple(), 2) == (2,)
    assert partial_dependence_column_indices_appended((1, 4), 7) == (1, 4, 7)

    assert partial_dependence_missing_column_message("unknown") == "A given column is not a column of the dataframe"


def test_partial_dependence_column_lookup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_lookup_shell import (
        partial_dependence_column_indices_appended,
        partial_dependence_nonunique_column_message,
    )

    with pytest.raises(ViolationError):
        partial_dependence_nonunique_column_message(tuple())

    with pytest.raises(ViolationError):
        partial_dependence_column_indices_appended((1, True), 3)

    with pytest.raises(ViolationError):
        partial_dependence_column_indices_appended((1, 2), -1)
