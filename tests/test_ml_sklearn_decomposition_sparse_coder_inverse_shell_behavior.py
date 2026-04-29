from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError

from sciona.atoms.ml.sklearn.decomposition.sparse_coder_inverse_shell import (
    sparse_coder_fit_require_matching_features,
    sparse_coder_inverse_transform,
    sparse_coding_expected_code_width,
    sparse_coding_merge_split_sign,
)


def test_sparse_coder_fit_require_matching_features_accepts_equal_counts() -> None:
    assert sparse_coder_fit_require_matching_features(4, 4) == 4


def test_sparse_coding_expected_code_width_respects_split_sign() -> None:
    dictionary = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]], dtype=np.float64)
    assert sparse_coding_expected_code_width(dictionary) == 3
    assert sparse_coding_expected_code_width(dictionary, split_sign=True) == 6


def test_sparse_coding_merge_split_sign_reconstructs_signed_code() -> None:
    split_code = np.array([[2.0, 0.0, 0.0, 1.5], [0.0, 3.0, 4.0, 0.0]], dtype=np.float64)
    merged = sparse_coding_merge_split_sign(split_code)
    assert np.array_equal(merged, np.array([[2.0, -1.5], [-4.0, 3.0]], dtype=np.float64))


def test_sparse_coder_inverse_transform_reconstructs_original_space() -> None:
    dictionary = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float64)
    code = np.array([[2.0, -1.0], [0.5, 0.5]], dtype=np.float64)
    observed = sparse_coder_inverse_transform(code, dictionary)
    assert np.allclose(observed, code @ dictionary)


def test_sparse_coder_inverse_transform_reconstructs_from_split_sign_code() -> None:
    dictionary = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    split_code = np.array([[3.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    observed = sparse_coder_inverse_transform(split_code, dictionary, split_sign=True)
    expected = np.array([[3.0, -2.0]], dtype=np.float64)
    assert np.allclose(observed, expected)


def test_sparse_coder_inverse_shell_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        sparse_coder_fit_require_matching_features(3, 4)

    with pytest.raises((ViolationError, ValueError)):
        sparse_coding_merge_split_sign(np.ones((2, 3), dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        sparse_coder_inverse_transform(
            np.ones((2, 3), dtype=np.float64),
            np.ones((2, 4), dtype=np.float64),
            split_sign=False,
        )
