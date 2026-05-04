"""Helpers for deterministic biclustering SVD callback setup adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_svd_randomized_kwargs,
    witness_bicluster_svd_svds_kwargs,
    witness_bicluster_svd_use_arpack,
    witness_bicluster_svd_use_randomized,
)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_positive_int(value: object) -> bool:
    return bool(value is None or _positive_int(value))


def _valid_random_state_like(value: object) -> bool:
    return bool(
        value is None
        or isinstance(value, (int, np.integer, np.random.RandomState))
    )


def _randomized_kwargs_valid(result: object, random_state: object, n_svd_vecs: int | None) -> bool:
    expected = {"random_state": random_state}
    if n_svd_vecs is not None:
        expected["n_oversamples"] = int(n_svd_vecs)
    return bool(isinstance(result, dict) and result == expected)


def _svds_kwargs_valid(result: object, n_components: int, n_svd_vecs: int | None) -> bool:
    expected = {"k": int(n_components), "ncv": n_svd_vecs}
    return bool(isinstance(result, dict) and result == expected)


@register_atom(witness_bicluster_svd_use_randomized)
@icontract.require(lambda svd_method: _nonempty_string(svd_method), "svd_method must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def bicluster_svd_use_randomized(svd_method: str) -> bool:
    """Return whether BaseSpectral._svd takes the randomized_svd branch."""
    return svd_method == "randomized"


@register_atom(witness_bicluster_svd_use_arpack)
@icontract.require(lambda svd_method: _nonempty_string(svd_method), "svd_method must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def bicluster_svd_use_arpack(svd_method: str) -> bool:
    """Return whether BaseSpectral._svd takes the svds/arpack branch."""
    return svd_method == "arpack"


@register_atom(witness_bicluster_svd_randomized_kwargs)
@icontract.require(lambda random_state: _valid_random_state_like(random_state), "random_state must be None, an integer seed, or a numpy RandomState")
@icontract.require(lambda n_svd_vecs: _optional_positive_int(n_svd_vecs), "n_svd_vecs must be None or a positive integer")
@icontract.ensure(
    lambda result, random_state, n_svd_vecs: _randomized_kwargs_valid(result, random_state, n_svd_vecs),
    "result must match the randomized_svd kwargs used by BaseSpectral._svd",
)
def bicluster_svd_randomized_kwargs(
    random_state: object,
    n_svd_vecs: int | None,
) -> dict[str, object]:
    """Resolve the randomized_svd kwargs used by BaseSpectral._svd."""
    kwargs: dict[str, object] = {"random_state": random_state}
    if n_svd_vecs is not None:
        kwargs["n_oversamples"] = int(n_svd_vecs)
    return kwargs


@register_atom(witness_bicluster_svd_svds_kwargs)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda n_svd_vecs: _optional_positive_int(n_svd_vecs), "n_svd_vecs must be None or a positive integer")
@icontract.ensure(
    lambda result, n_components, n_svd_vecs: _svds_kwargs_valid(result, n_components, n_svd_vecs),
    "result must match the svds kwargs used by BaseSpectral._svd",
)
def bicluster_svd_svds_kwargs(
    n_components: int,
    n_svd_vecs: int | None,
) -> dict[str, object]:
    """Resolve the svds kwargs used by BaseSpectral._svd."""
    return {
        "k": int(n_components),
        "ncv": n_svd_vecs,
    }
