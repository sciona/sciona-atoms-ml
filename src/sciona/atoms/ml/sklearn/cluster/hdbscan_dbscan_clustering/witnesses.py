"""Ghost witnesses for HDBSCAN DBSCAN-clustering helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_hdbscan_dbscan_infinite_mask(
    fitted_labels: NDArray[int],
) -> NDArray[bool]:
    """Describe HDBSCAN's inferred infinite-row mask from fitted labels."""
    del fitted_labels
    raise NotImplementedError


def witness_hdbscan_dbscan_missing_mask(
    fitted_labels: NDArray[int],
) -> NDArray[bool]:
    """Describe HDBSCAN's inferred missing-row mask from fitted labels."""
    del fitted_labels
    raise NotImplementedError


def witness_hdbscan_dbscan_labels(
    labels_at_cut: NDArray[int],
    infinite_mask: NDArray[bool],
    missing_mask: NDArray[bool],
) -> NDArray[int]:
    """Describe HDBSCAN's final DBSCAN-style labels after outlier overrides."""
    del labels_at_cut
    del infinite_mask
    del missing_mask
    raise NotImplementedError
