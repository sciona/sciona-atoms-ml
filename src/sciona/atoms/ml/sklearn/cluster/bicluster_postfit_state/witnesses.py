"""Ghost witnesses for spectral biclustering and coclustering post-fit state atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_bicluster_fit_row_labels(
    row_labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Describe the fitted row-label vector."""
    return np.asarray(row_labels, dtype=np.int64)


def witness_bicluster_fit_column_labels(
    column_labels: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Describe the fitted column-label vector."""
    return np.asarray(column_labels, dtype=np.int64)


def witness_bicluster_fit_rows(
    rows: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Describe the fitted row-indicator matrix."""
    return np.asarray(rows, dtype=np.bool_)


def witness_bicluster_fit_columns(
    columns: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Describe the fitted column-indicator matrix."""
    return np.asarray(columns, dtype=np.bool_)


def witness_bicluster_fit_return_self(estimator_token: str) -> str:
    """Describe the final self-return after biclustering state assignment."""
    return estimator_token
