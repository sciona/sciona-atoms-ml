from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_partial_dependence_feature_index_guard_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_guard import (
        partial_dependence_integer_feature_key_type,
        partial_dependence_negative_feature_guard_required,
        partial_dependence_negative_feature_message,
    )

    assert callable(partial_dependence_integer_feature_key_type)
    assert callable(partial_dependence_negative_feature_guard_required)
    assert callable(partial_dependence_negative_feature_message)


def test_partial_dependence_feature_index_guard_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_guard import (
        partial_dependence_integer_feature_key_type,
        partial_dependence_negative_feature_guard_required,
        partial_dependence_negative_feature_message,
    )

    assert partial_dependence_integer_feature_key_type("int") is True
    assert partial_dependence_integer_feature_key_type("str") is False

    assert partial_dependence_negative_feature_guard_required(np.asarray([0, 2, 5], dtype=np.int64)) is False
    assert partial_dependence_negative_feature_guard_required(np.asarray([0, -1, 5], dtype=np.int64)) is True

    assert partial_dependence_negative_feature_message(6) == "all features must be in [0, 5]"


def test_partial_dependence_feature_index_guard_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_index_guard import (
        partial_dependence_integer_feature_key_type,
        partial_dependence_negative_feature_guard_required,
        partial_dependence_negative_feature_message,
    )

    with pytest.raises(ViolationError):
        partial_dependence_integer_feature_key_type("")

    with pytest.raises(ViolationError):
        partial_dependence_negative_feature_guard_required(np.asarray([], dtype=np.int64))

    with pytest.raises(ViolationError):
        partial_dependence_negative_feature_message(0)
