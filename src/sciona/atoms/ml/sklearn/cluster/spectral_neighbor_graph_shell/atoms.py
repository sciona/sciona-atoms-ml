"""Spectral clustering neighbor-graph atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_kneighbors_graph_kwargs,
    witness_spectral_fit_precomputed_kneighbors_graph_mode,
    witness_spectral_fit_precomputed_neighbor_estimator_kwargs,
    witness_spectral_fit_precomputed_neighbor_metric,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_int(value: object) -> bool:
    return bool(value is None or (isinstance(value, int) and not isinstance(value, bool)))


def _optional_string(value: object) -> bool:
    return bool(value is None or isinstance(value, str))


def _kneighbors_kwargs_valid(result: object, n_neighbors: int, n_jobs: int | None) -> bool:
    return bool(
        isinstance(result, dict)
        and result == {
            "n_neighbors": int(n_neighbors),
            "include_self": True,
            "n_jobs": n_jobs,
        }
    )


def _precomputed_neighbor_estimator_kwargs_valid(result: object, n_neighbors: int, n_jobs: int | None) -> bool:
    return bool(
        isinstance(result, dict)
        and result == {
            "n_neighbors": int(n_neighbors),
            "n_jobs": n_jobs,
            "metric": "precomputed",
        }
    )


@register_atom(witness_spectral_fit_kneighbors_graph_kwargs)
@icontract.require(lambda n_neighbors: _positive_int(n_neighbors), "n_neighbors must be a positive integer")
@icontract.require(lambda n_jobs=None: _optional_int(n_jobs), "n_jobs must be None or an integer")
@icontract.ensure(
    lambda result, n_neighbors, n_jobs=None: _kneighbors_kwargs_valid(result, n_neighbors, n_jobs),
    "result must match SpectralClustering's kneighbors_graph kwargs",
)
def spectral_fit_kneighbors_graph_kwargs(
    n_neighbors: int,
    n_jobs: int | None = None,
) -> dict[str, object]:
    """Resolve the kneighbors_graph kwargs used by SpectralClustering.fit for nearest-neighbor affinity."""
    return {
        "n_neighbors": int(n_neighbors),
        "include_self": True,
        "n_jobs": n_jobs,
    }


@register_atom(witness_spectral_fit_precomputed_neighbor_metric)
@icontract.require(lambda parent_metric=None: _optional_string(parent_metric), "parent_metric must be None or a string")
@icontract.ensure(lambda result: isinstance(result, str) and result == "precomputed", "metric must be 'precomputed'")
def spectral_fit_precomputed_neighbor_metric(parent_metric: str | None = None) -> str:
    """Expose the fixed metric used by SpectralClustering.fit for precomputed-neighbor affinity."""
    del parent_metric
    return "precomputed"


@register_atom(witness_spectral_fit_precomputed_neighbor_estimator_kwargs)
@icontract.require(lambda n_neighbors: _positive_int(n_neighbors), "n_neighbors must be a positive integer")
@icontract.require(lambda n_jobs=None: _optional_int(n_jobs), "n_jobs must be None or an integer")
@icontract.ensure(
    lambda result, n_neighbors, n_jobs=None: _precomputed_neighbor_estimator_kwargs_valid(result, n_neighbors, n_jobs),
    "result must match SpectralClustering's NearestNeighbors kwargs for precomputed-neighbor affinity",
)
def spectral_fit_precomputed_neighbor_estimator_kwargs(
    n_neighbors: int,
    n_jobs: int | None = None,
) -> dict[str, object]:
    """Resolve the NearestNeighbors kwargs used by SpectralClustering.fit for precomputed-neighbor affinity."""
    return {
        "n_neighbors": int(n_neighbors),
        "n_jobs": n_jobs,
        "metric": spectral_fit_precomputed_neighbor_metric(),
    }


@register_atom(witness_spectral_fit_precomputed_kneighbors_graph_mode)
@icontract.require(lambda parent_mode=None: _optional_string(parent_mode), "parent_mode must be None or a string")
@icontract.ensure(lambda result: isinstance(result, str) and result == "connectivity", "mode must be 'connectivity'")
def spectral_fit_precomputed_kneighbors_graph_mode(parent_mode: str | None = None) -> str:
    """Expose the fixed estimator.kneighbors_graph mode used by SpectralClustering.fit for precomputed-neighbor affinity."""
    del parent_mode
    return "connectivity"
