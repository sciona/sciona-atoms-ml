"""Helpers for deterministic biclustering KMeans callback setup adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_kmeans_kwargs,
    witness_bicluster_minibatch_kmeans_kwargs,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _valid_init(value: object) -> bool:
    return bool(
        (isinstance(value, str) and value != "")
        or callable(value)
        or (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.shape[0] >= 1
            and value.shape[1] >= 1
            and np.all(np.isfinite(value))
        )
    )


def _valid_random_state_like(value: object) -> bool:
    return bool(
        value is None
        or isinstance(value, (int, np.integer, np.random.RandomState))
    )


def _kmeans_kwargs_valid(
    result: object,
    n_clusters: int,
    init: object,
    n_init: int,
    random_state: object,
) -> bool:
    del n_clusters
    return bool(
        isinstance(result, dict)
        and result
        == {
            "init": init,
            "n_init": int(n_init),
            "random_state": random_state,
        }
    )


@register_atom(witness_bicluster_kmeans_kwargs)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda init: _valid_init(init), "init must be a nonempty string, callable, or finite 2D ndarray")
@icontract.require(lambda n_init: _positive_int(n_init), "n_init must be a positive integer")
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.ensure(
    lambda result, n_clusters, init, n_init, random_state: _kmeans_kwargs_valid(
        result,
        n_clusters,
        init,
        n_init,
        random_state,
    ),
    "result must match the KMeans kwargs used by BaseSpectral._k_means",
)
def bicluster_kmeans_kwargs(
    n_clusters: int,
    init: object,
    n_init: int,
    random_state: object,
) -> dict[str, object]:
    """Resolve the KMeans kwargs used by BaseSpectral._k_means."""
    del n_clusters
    return {
        "init": init,
        "n_init": int(n_init),
        "random_state": random_state,
    }


@register_atom(witness_bicluster_minibatch_kmeans_kwargs)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda init: _valid_init(init), "init must be a nonempty string, callable, or finite 2D ndarray")
@icontract.require(lambda n_init: _positive_int(n_init), "n_init must be a positive integer")
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.ensure(
    lambda result, n_clusters, init, n_init, random_state: _kmeans_kwargs_valid(
        result,
        n_clusters,
        init,
        n_init,
        random_state,
    ),
    "result must match the MiniBatchKMeans kwargs used by BaseSpectral._k_means",
)
def bicluster_minibatch_kmeans_kwargs(
    n_clusters: int,
    init: object,
    n_init: int,
    random_state: object,
) -> dict[str, object]:
    """Resolve the MiniBatchKMeans kwargs used by BaseSpectral._k_means."""
    del n_clusters
    return {
        "init": init,
        "n_init": int(n_init),
        "random_state": random_state,
    }
