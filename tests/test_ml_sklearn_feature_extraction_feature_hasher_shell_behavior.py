from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.feature_extraction.feature_hasher_shell import (
    feature_hasher_csr_matrix,
    feature_hasher_dict_items,
    feature_hasher_pair_items,
    feature_hasher_require_nonempty_samples,
    feature_hasher_sample_count,
    feature_hasher_string_items,
)


def test_feature_hasher_dict_items_preserves_sample_structure() -> None:
    observed = feature_hasher_dict_items(({"dog": 2, "cat": -1.5}, {"run": 3.0}))
    assert observed == ((("dog", 2.0), ("cat", -1.5)), (("run", 3.0),))


def test_feature_hasher_pair_items_normalizes_numeric_values() -> None:
    observed = feature_hasher_pair_items(((("dog", 2.0), ("cat", -1)),),)
    assert observed == ((("dog", 2.0), ("cat", -1.0)),)


def test_feature_hasher_string_items_assigns_unit_weights() -> None:
    observed = feature_hasher_string_items((("dog", "cat"), tuple()))
    assert observed == ((("dog", 1.0), ("cat", 1.0)), tuple())


def test_feature_hasher_sample_count_and_nonempty_guard() -> None:
    indptr = np.array([0, 2, 5], dtype=np.int64)
    assert feature_hasher_sample_count(indptr) == 2
    assert feature_hasher_require_nonempty_samples(2) == 2
    with pytest.raises((ViolationError, ValueError)):
        feature_hasher_require_nonempty_samples(0)


def test_feature_hasher_csr_matrix_sums_duplicate_entries() -> None:
    matrix = feature_hasher_csr_matrix(
        np.array([1, 0, 1, 1], dtype=np.int64),
        np.array([0, 2, 4], dtype=np.int64),
        np.array([2.0, 1.0, 3.0, -1.0], dtype=np.float64),
        n_features=3,
    )
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float64
    assert np.array_equal(matrix.toarray(), np.array([[1.0, 2.0, 0.0], [0.0, 2.0, 0.0]]))


def test_feature_hasher_shell_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        feature_hasher_dict_items(({"dog": True},))

    with pytest.raises((ViolationError, ValueError)):
        feature_hasher_string_items(("dog",))

    with pytest.raises((ViolationError, ValueError)):
        feature_hasher_sample_count(np.array([1, 2], dtype=np.int64))

    with pytest.raises((ViolationError, ValueError)):
        feature_hasher_csr_matrix(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            n_features=1,
        )
