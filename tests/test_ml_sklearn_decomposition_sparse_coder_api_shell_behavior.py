from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import SparseCoder


def _dictionary() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_sparse_coder_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell import (
        sparse_coder_fit_return_self,
        sparse_coder_n_features_out,
        sparse_coder_preserves_dtype_tags,
        sparse_coder_requires_fit_tag,
        sparse_coder_transform_dictionary,
    )

    assert callable(sparse_coder_fit_return_self)
    assert callable(sparse_coder_transform_dictionary)
    assert callable(sparse_coder_requires_fit_tag)
    assert callable(sparse_coder_preserves_dtype_tags)
    assert callable(sparse_coder_n_features_out)


def test_sparse_coder_api_shell_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell import (
        sparse_coder_fit_return_self,
        sparse_coder_n_features_out,
        sparse_coder_preserves_dtype_tags,
        sparse_coder_requires_fit_tag,
        sparse_coder_transform_dictionary,
    )

    dictionary = _dictionary()
    coder = SparseCoder(dictionary=dictionary)

    assert sparse_coder_fit_return_self("SparseCoder") == "SparseCoder"
    assert np.array_equal(sparse_coder_transform_dictionary(dictionary), dictionary)
    assert sparse_coder_requires_fit_tag(True) is False
    assert sparse_coder_preserves_dtype_tags(("float64",)) == ("float64", "float32")
    assert sparse_coder_n_features_out(dictionary) == coder._n_features_out


def test_sparse_coder_api_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_coder_api_shell import (
        sparse_coder_fit_return_self,
        sparse_coder_n_features_out,
        sparse_coder_preserves_dtype_tags,
        sparse_coder_transform_dictionary,
    )

    with pytest.raises(Exception):
        sparse_coder_fit_return_self("")

    with pytest.raises(Exception):
        sparse_coder_transform_dictionary(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        sparse_coder_preserves_dtype_tags(())

    with pytest.raises(Exception):
        sparse_coder_n_features_out(np.array([], dtype=np.float64))
