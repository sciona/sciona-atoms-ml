from __future__ import annotations

import numpy as np


def test_sparse_encode_scheduling_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_scheduling import (
        sparse_encode_code_from_views,
        sparse_encode_parallel_required,
        sparse_encode_sample_bounds,
    )

    assert callable(sparse_encode_parallel_required)
    assert callable(sparse_encode_sample_bounds)
    assert callable(sparse_encode_code_from_views)


def test_sparse_encode_scheduling_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_scheduling import (
        sparse_encode_code_from_views,
        sparse_encode_parallel_required,
        sparse_encode_sample_bounds,
    )

    assert sparse_encode_parallel_required(1, "lasso_cd") is False
    assert sparse_encode_parallel_required(3, "threshold") is False
    assert sparse_encode_parallel_required(3, "lasso_cd") is True

    bounds = sparse_encode_sample_bounds(5, 3)
    expected_bounds = np.array([[0, 2], [2, 4], [4, 5]], dtype=np.int64)
    assert np.array_equal(bounds, expected_bounds)

    code_views = (
        np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        np.array([[3.0, 1.0], [4.0, 1.0]], dtype=np.float64),
        np.array([[5.0, 2.0]], dtype=np.float64),
    )
    assembled = sparse_encode_code_from_views(code_views, bounds, 5, 2)
    expected = np.vstack(code_views)
    assert np.array_equal(assembled, expected)

