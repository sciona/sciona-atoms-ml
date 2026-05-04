from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_custom_values_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_custom_values_shell import (
        partial_dependence_custom_values_mapping,
        partial_dependence_custom_values_subset_mapping,
        partial_dependence_feature_sequence,
    )

    assert callable(partial_dependence_custom_values_mapping)
    assert callable(partial_dependence_feature_sequence)
    assert callable(partial_dependence_custom_values_subset_mapping)


def test_partial_dependence_custom_values_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_custom_values_shell import (
        partial_dependence_custom_values_mapping,
        partial_dependence_custom_values_subset_mapping,
        partial_dependence_feature_sequence,
    )

    assert partial_dependence_custom_values_mapping(None) == {}
    assert partial_dependence_custom_values_mapping({0: [1.0, 2.0]}) == {0: [1.0, 2.0]}

    assert partial_dependence_feature_sequence("age") == ("age",)
    assert partial_dependence_feature_sequence(3) == (3,)
    assert partial_dependence_feature_sequence([("age", "income"), "state"]) == (("age", "income"), "state")

    custom_values = {
        "age": [18.0, 30.0, 45.0],
        ("age", "income"): [(18.0, 40000.0)],
    }
    features = partial_dependence_feature_sequence([("age", "income"), "state", "age"])
    assert partial_dependence_custom_values_subset_mapping(features, custom_values) == {
        0: [(18.0, 40000.0)],
        2: [18.0, 30.0, 45.0],
    }


def test_partial_dependence_custom_values_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_custom_values_shell import (
        partial_dependence_custom_values_mapping,
        partial_dependence_custom_values_subset_mapping,
        partial_dependence_feature_sequence,
    )

    with pytest.raises(ViolationError):
        partial_dependence_custom_values_mapping([])

    with pytest.raises(ViolationError):
        partial_dependence_feature_sequence([])

    with pytest.raises(ViolationError):
        partial_dependence_custom_values_subset_mapping(("age",), [])
