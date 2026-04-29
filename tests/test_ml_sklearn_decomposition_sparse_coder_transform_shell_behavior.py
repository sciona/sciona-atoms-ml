from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import SparseCoder

from sciona.atoms.ml.sklearn.decomposition.sparse_coder_transform_shell import (
    sparse_coder_n_components,
    sparse_coder_n_features_in,
    sparse_coding_split_sign,
    sparse_coding_transform_alpha,
)


def _dictionary() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_sparse_coder_transform_shell_atoms_import() -> None:
    assert callable(sparse_coding_transform_alpha)
    assert callable(sparse_coding_split_sign)
    assert callable(sparse_coder_n_components)
    assert callable(sparse_coder_n_features_in)


def test_sparse_coding_transform_alpha_matches_base_sparse_coding_resolution() -> None:
    assert sparse_coding_transform_alpha(None, fit_alpha=0.25) == pytest.approx(0.25)
    assert sparse_coding_transform_alpha(0.5, fit_alpha=0.25) == pytest.approx(0.5)
    assert sparse_coding_transform_alpha(None, fit_alpha=None) is None


def test_sparse_coding_split_sign_matches_sparse_coder_threshold_transform() -> None:
    dictionary = _dictionary()
    X = _data()

    coder_plain = SparseCoder(
        dictionary=dictionary,
        transform_algorithm="threshold",
        transform_alpha=0.3,
        split_sign=False,
    )
    coder_split = SparseCoder(
        dictionary=dictionary,
        transform_algorithm="threshold",
        transform_alpha=0.3,
        split_sign=True,
    )

    base_code = np.asarray(coder_plain.transform(X), dtype=np.float64)
    observed = sparse_coding_split_sign(base_code)
    expected = np.asarray(coder_split.transform(X), dtype=np.float64)

    assert np.allclose(observed, expected)


def test_sparse_coder_properties_match_sklearn() -> None:
    dictionary = _dictionary()
    coder = SparseCoder(dictionary=dictionary)

    assert sparse_coder_n_components(dictionary) == coder.n_components_
    assert sparse_coder_n_features_in(dictionary) == coder.n_features_in_


def test_sparse_coder_transform_shell_rejects_invalid_inputs() -> None:
    with pytest.raises((ViolationError, ValueError)):
        sparse_coding_transform_alpha(-0.1, fit_alpha=None)

    with pytest.raises((ViolationError, ValueError)):
        sparse_coding_split_sign(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        sparse_coder_n_components(np.array([], dtype=np.float64))
