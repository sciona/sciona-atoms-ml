"""Spectral coclustering preflight atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_cocluster_checked_n_clusters


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


@register_atom(witness_cocluster_checked_n_clusters)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(lambda result, n_clusters: result == int(n_clusters), "result must return the validated n_clusters value")
def cocluster_checked_n_clusters(n_clusters: int, n_samples: int) -> int:
    """Validate that spectral coclustering does not request more clusters than samples."""
    if int(n_clusters) > int(n_samples):
        raise ValueError(
            f"n_clusters should be <= n_samples={int(n_samples)}. Got"
            f" {int(n_clusters)} instead."
        )
    return int(n_clusters)
