"""Ghost witnesses for HDBSCAN non-finite remapping helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_hdbscan_missing_indices(
    reduced_row_sums: NDArray[float],
) -> NDArray[int]:
    """Describe HDBSCAN's raw indices for rows flagged as missing by NaN row sums."""
    del reduced_row_sums
    raise NotImplementedError


def witness_hdbscan_infinite_indices(
    reduced_row_sums: NDArray[float],
) -> NDArray[int]:
    """Describe HDBSCAN's raw indices for rows flagged as infinite by Inf row sums."""
    del reduced_row_sums
    raise NotImplementedError


def witness_hdbscan_finite_row_indices(
    X: object,
) -> NDArray[int]:
    """Describe HDBSCAN's purely finite row indices for dense or sparse input."""
    del X
    raise NotImplementedError


def witness_hdbscan_internal_to_raw_map(
    finite_index: NDArray[int],
) -> dict[int, int]:
    """Describe HDBSCAN's mapping from internal finite-sample indices back to raw indices."""
    del finite_index
    raise NotImplementedError


def witness_hdbscan_nonfinite_raw_indices(
    infinite_index: NDArray[int],
    missing_index: NDArray[int],
) -> set[int]:
    """Describe HDBSCAN's non-finite raw-index set used for linkage-tree remapping."""
    del infinite_index
    del missing_index
    return set()


def witness_hdbscan_remapped_labels(
    raw_sample_count: int,
    finite_index: NDArray[int],
    finite_labels: NDArray[int],
    infinite_index: NDArray[int],
    missing_index: NDArray[int],
) -> NDArray[int]:
    """Describe HDBSCAN's remapped label vector after restoring non-finite rows."""
    del raw_sample_count
    del finite_index
    del finite_labels
    del infinite_index
    del missing_index
    raise NotImplementedError


def witness_hdbscan_remapped_probabilities(
    raw_sample_count: int,
    finite_index: NDArray[int],
    finite_probabilities: NDArray[float],
    infinite_index: NDArray[int],
    missing_index: NDArray[int],
) -> NDArray[float]:
    """Describe HDBSCAN's remapped probability vector after restoring non-finite rows."""
    del raw_sample_count
    del finite_index
    del finite_probabilities
    del infinite_index
    del missing_index
    raise NotImplementedError
