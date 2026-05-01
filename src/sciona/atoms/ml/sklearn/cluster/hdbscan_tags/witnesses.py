"""Ghost witnesses for HDBSCAN estimator-tag helper atoms."""

from __future__ import annotations


def witness_hdbscan_sparse_input_tag(metric: str) -> bool:
    """Describe HDBSCAN's sparse-input estimator tag."""
    del metric
    return True


def witness_hdbscan_allow_nan_tag(metric: str) -> bool:
    """Describe HDBSCAN's allow-nan estimator tag from the metric choice."""
    del metric
    return False
