"""Spectral clustering label-callback atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_discretize_kwargs,
    witness_spectral_fit_kmeans_kwargs,
    witness_spectral_fit_kmeans_output_labels,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _bool_or_int(value: object) -> bool:
    return bool(isinstance(value, (bool, int)) and not isinstance(value, np.bool_))


def _random_state_object(value: object) -> bool:
    return isinstance(value, np.random.RandomState)


def _integer_vector(value: object) -> bool:
    array = np.asarray(value)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _kmeans_kwargs_valid(
    result: object,
    n_clusters: int,
    n_init: int,
    verbose: bool | int,
    random_state: np.random.RandomState,
) -> bool:
    return bool(
        isinstance(result, dict)
        and set(result) == {"random_state", "n_init", "verbose"}
        and result["random_state"] is random_state
        and result["n_init"] == int(n_init)
        and result["verbose"] == verbose
        and isinstance(n_clusters, int)
    )


def _discretize_kwargs_valid(result: object, random_state: np.random.RandomState) -> bool:
    return bool(
        isinstance(result, dict)
        and set(result) == {"random_state"}
        and result["random_state"] is random_state
    )


@register_atom(witness_spectral_fit_kmeans_kwargs)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda n_init: _positive_int(n_init), "n_init must be a positive integer")
@icontract.require(lambda verbose: _bool_or_int(verbose), "verbose must be bool or int")
@icontract.require(lambda random_state: _random_state_object(random_state), "random_state must be a numpy RandomState")
@icontract.ensure(
    lambda result, n_clusters, n_init, verbose, random_state: _kmeans_kwargs_valid(
        result, n_clusters, n_init, verbose, random_state
    ),
    "result must match SpectralClustering's k_means kwargs",
)
def spectral_fit_kmeans_kwargs(
    n_clusters: int,
    n_init: int,
    verbose: bool | int,
    random_state: np.random.RandomState,
) -> dict[str, object]:
    """Resolve the k_means kwargs used by SpectralClustering.fit."""
    del n_clusters
    return {
        "random_state": random_state,
        "n_init": int(n_init),
        "verbose": verbose,
    }


@register_atom(witness_spectral_fit_kmeans_output_labels)
@icontract.require(lambda labels: _integer_vector(labels), "labels must be a nonempty one-dimensional integer vector")
@icontract.ensure(
    lambda result, labels: _integer_vector(result) and np.asarray(result).shape == np.asarray(labels).shape and np.array_equal(np.asarray(result), np.asarray(labels)),
    "result must preserve the k_means label vector",
)
def spectral_fit_kmeans_output_labels(labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Expose the label vector unpacked from the deferred k_means return tuple."""
    return np.asarray(labels).copy()


@register_atom(witness_spectral_fit_discretize_kwargs)
@icontract.require(lambda random_state: _random_state_object(random_state), "random_state must be a numpy RandomState")
@icontract.ensure(
    lambda result, random_state: _discretize_kwargs_valid(result, random_state),
    "result must match SpectralClustering's discretize kwargs",
)
def spectral_fit_discretize_kwargs(
    random_state: np.random.RandomState,
) -> dict[str, object]:
    """Resolve the discretize kwargs used by SpectralClustering.fit."""
    return {"random_state": random_state}
