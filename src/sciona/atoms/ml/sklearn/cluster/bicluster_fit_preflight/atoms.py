"""Spectral biclustering preflight atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_checked_cluster_counts,
    witness_bicluster_checked_method,
    witness_bicluster_checked_n_best,
)

ClusterCounts = tuple[int, int]


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _cluster_counts_result_valid(result: object, n_samples: int) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) == 2
        and all(_positive_int(value) and int(value) <= int(n_samples) for value in result)
    )


def _method_valid(method: object) -> bool:
    return method in {"bistochastic", "scale", "log"}


@register_atom(witness_bicluster_checked_cluster_counts)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.ensure(
    lambda result, n_samples: _cluster_counts_result_valid(result, n_samples),
    "result must be a pair of positive cluster counts within n_samples",
)
def bicluster_checked_cluster_counts(
    n_clusters: int | tuple[int, int],
    n_samples: int,
) -> ClusterCounts:
    """Validate spectral biclustering cluster counts against the sample count."""
    checked_n_samples = int(n_samples)
    if isinstance(n_clusters, int) and not isinstance(n_clusters, bool):
        if n_clusters > checked_n_samples:
            raise ValueError(
                f"n_clusters should be <= n_samples={checked_n_samples}. Got"
                f" {n_clusters} instead."
            )
        return int(n_clusters), int(n_clusters)

    try:
        n_row_clusters, n_column_clusters = n_clusters
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Incorrect parameter n_clusters has value:"
            f" {n_clusters}. It should either be a single integer"
            " or an iterable with two integers:"
            " (n_row_clusters, n_column_clusters)"
            " And the values are should be in the"
            " range: (1, n_samples)"
        ) from e

    try:
        if not _positive_int(n_row_clusters) or int(n_row_clusters) > checked_n_samples:
            raise ValueError("n_row_clusters out of range")
        if not _positive_int(n_column_clusters) or int(n_column_clusters) > checked_n_samples:
            raise ValueError("n_column_clusters out of range")
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Incorrect parameter n_clusters has value:"
            f" {n_clusters}. It should either be a single integer"
            " or an iterable with two integers:"
            " (n_row_clusters, n_column_clusters)"
            " And the values are should be in the"
            " range: (1, n_samples)"
        ) from e
    return int(n_row_clusters), int(n_column_clusters)


@register_atom(witness_bicluster_checked_n_best)
@icontract.require(lambda n_best: _positive_int(n_best), "n_best must be a positive integer")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result, n_best: result == int(n_best), "result must return the validated n_best value")
def bicluster_checked_n_best(n_best: int, n_components: int) -> int:
    """Validate that spectral biclustering keeps at most n_components best vectors."""
    if int(n_best) > int(n_components):
        raise ValueError(
            f"n_best={int(n_best)} must be <= n_components={int(n_components)}."
        )
    return int(n_best)


@register_atom(witness_bicluster_checked_method)
@icontract.require(lambda method: isinstance(method, str) and _method_valid(method), "method must be one of 'bistochastic', 'scale', or 'log'")
@icontract.require(lambda is_sparse: isinstance(is_sparse, bool), "is_sparse must be boolean")
@icontract.ensure(lambda result, method: result == method, "result must return the validated normalization method")
def bicluster_checked_method(method: str, is_sparse: bool) -> str:
    """Validate spectral biclustering's normalization method against sparse input."""
    if bool(is_sparse) and method == "log":
        raise ValueError(
            "Cannot compute log of a sparse matrix,"
            " because log(x) diverges to -infinity as x"
            " goes to 0."
        )
    return method
