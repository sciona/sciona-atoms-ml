"""spectral_clustering wrapper-constructor atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from ..spectral_clustering_wrapper import spectral_clustering_precomputed_affinity
from .witnesses import witness_spectral_clustering_constructor_kwargs


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_positive_int(value: object) -> bool:
    return bool(value is None or _positive_int(value))


def _optional_nonempty_string(value: object) -> bool:
    return bool(value is None or (isinstance(value, str) and value != ""))


def _valid_random_state_like(value: object) -> bool:
    return bool(
        value is None
        or isinstance(value, (int, np.integer, np.random.RandomState))
    )


def _valid_eigen_tol(value: object) -> bool:
    return bool(
        value == "auto"
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
        )
    )


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _bool_or_int(value: object) -> bool:
    return bool(isinstance(value, (bool, int)) and not isinstance(value, np.bool_))


def _constructor_kwargs_valid(
    result: object,
    n_clusters: int,
    n_components: int | None,
    eigen_solver: str | None,
    random_state: object,
    n_init: int,
    eigen_tol: float | str,
    assign_labels: str,
    verbose: bool | int,
) -> bool:
    return bool(
        isinstance(result, dict)
        and result
        == {
            "n_clusters": int(n_clusters),
            "n_components": n_components,
            "eigen_solver": eigen_solver,
            "random_state": random_state,
            "n_init": int(n_init),
            "affinity": spectral_clustering_precomputed_affinity(None),
            "eigen_tol": eigen_tol,
            "assign_labels": assign_labels,
            "verbose": verbose,
        }
    )


@register_atom(witness_spectral_clustering_constructor_kwargs)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda n_components: _optional_positive_int(n_components), "n_components must be None or a positive integer")
@icontract.require(lambda eigen_solver: _optional_nonempty_string(eigen_solver), "eigen_solver must be None or a nonempty string")
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.require(lambda n_init: _positive_int(n_init), "n_init must be a positive integer")
@icontract.require(lambda eigen_tol: _valid_eigen_tol(eigen_tol), "eigen_tol must be 'auto' or a finite real scalar")
@icontract.require(lambda assign_labels: _nonempty_string(assign_labels), "assign_labels must be a nonempty string")
@icontract.require(lambda verbose: _bool_or_int(verbose), "verbose must be bool or int")
@icontract.ensure(
    lambda result, n_clusters, n_components, eigen_solver, random_state, n_init, eigen_tol, assign_labels, verbose: _constructor_kwargs_valid(
        result,
        n_clusters,
        n_components,
        eigen_solver,
        random_state,
        n_init,
        eigen_tol,
        assign_labels,
        verbose,
    ),
    "result must match the SpectralClustering constructor kwargs used by spectral_clustering",
)
def spectral_clustering_constructor_kwargs(
    n_clusters: int,
    n_components: int | None,
    eigen_solver: str | None,
    random_state: object = None,
    n_init: int = 10,
    eigen_tol: float | str = "auto",
    assign_labels: str = "kmeans",
    verbose: bool | int = False,
) -> dict[str, object]:
    """Resolve the SpectralClustering constructor kwargs used by spectral_clustering."""
    return {
        "n_clusters": int(n_clusters),
        "n_components": n_components,
        "eigen_solver": eigen_solver,
        "random_state": random_state,
        "n_init": int(n_init),
        "affinity": spectral_clustering_precomputed_affinity(None),
        "eigen_tol": eigen_tol,
        "assign_labels": assign_labels,
        "verbose": verbose,
    }
