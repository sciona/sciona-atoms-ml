from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_partial_dependence_feature_index_output_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_output_shell import (
        partial_dependence_feature_indices_array,
        partial_dependence_feature_indices_vector,
        partial_dependence_selected_feature_count,
    )

    assert callable(partial_dependence_feature_indices_array)
    assert callable(partial_dependence_feature_indices_vector)
    assert callable(partial_dependence_selected_feature_count)


def test_partial_dependence_feature_index_output_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_output_shell import (
        partial_dependence_feature_indices_array,
        partial_dependence_feature_indices_vector,
        partial_dependence_selected_feature_count,
    )

    column_indices = (3, 0, 2)
    actual_array = partial_dependence_feature_indices_array(column_indices)
    expected_array = np.asarray(column_indices, dtype=np.intp, order="C")
    assert np.array_equal(actual_array, expected_array)
    assert actual_array.dtype == np.dtype(np.intp)
    assert actual_array.flags["C_CONTIGUOUS"]

    actual_vector = partial_dependence_feature_indices_vector(actual_array)
    expected_vector = expected_array.ravel()
    assert np.array_equal(actual_vector, expected_vector)
    assert actual_vector.ndim == 1
    assert partial_dependence_selected_feature_count(actual_vector) == 3


def test_partial_dependence_feature_index_output_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_output_shell import (
        partial_dependence_feature_indices_array,
        partial_dependence_feature_indices_vector,
        partial_dependence_selected_feature_count,
    )

    with pytest.raises(ViolationError):
        partial_dependence_feature_indices_array(())

    with pytest.raises(ViolationError):
        partial_dependence_feature_indices_vector(np.array([], dtype=np.intp))

    with pytest.raises(ViolationError):
        partial_dependence_selected_feature_count(np.array([], dtype=np.intp))
