from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.extmath import row_norms


def test_sparse_encode_precomputed_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precomputed_bookkeeping import (
        sparse_encode_lasso_alpha,
        sparse_encode_omp_norms_squared,
        sparse_encode_precomputed_output,
        sparse_encode_writable_init,
    )

    assert callable(sparse_encode_lasso_alpha)
    assert callable(sparse_encode_writable_init)
    assert callable(sparse_encode_omp_norms_squared)
    assert callable(sparse_encode_precomputed_output)


def test_sparse_encode_precomputed_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precomputed_bookkeeping import (
        sparse_encode_lasso_alpha,
        sparse_encode_omp_norms_squared,
        sparse_encode_precomputed_output,
        sparse_encode_writable_init,
    )

    assert sparse_encode_lasso_alpha(0.75, 6) == pytest.approx(0.125)

    writeable = np.array([[1.0, 2.0]], dtype=np.float64)
    same = sparse_encode_writable_init(writeable)
    assert same is writeable

    readonly = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    readonly.setflags(write=False)
    copied = sparse_encode_writable_init(readonly)
    assert np.array_equal(copied, readonly)
    assert copied.flags["WRITEABLE"]
    assert copied is not readonly

    X = np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float64)
    assert np.allclose(sparse_encode_omp_norms_squared(X), row_norms(X, squared=True))

    new_code = np.arange(6.0, dtype=np.float64)
    reshaped = sparse_encode_precomputed_output(new_code, 2, 3)
    assert np.array_equal(reshaped, new_code.reshape(2, 3))


def test_sparse_encode_precomputed_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precomputed_bookkeeping import (
        sparse_encode_lasso_alpha,
        sparse_encode_omp_norms_squared,
        sparse_encode_precomputed_output,
        sparse_encode_writable_init,
    )

    with pytest.raises(Exception):
        sparse_encode_lasso_alpha(-1.0, 4)

    with pytest.raises(Exception):
        sparse_encode_writable_init(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(Exception):
        sparse_encode_omp_norms_squared(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(Exception):
        sparse_encode_precomputed_output(np.arange(5.0, dtype=np.float64), 2, 3)
