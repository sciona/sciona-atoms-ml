"""Spectral clustering fit API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_accept_sparse_formats,
    witness_spectral_fit_affinity_allows_square_input,
    witness_spectral_fit_dtype_name,
    witness_spectral_fit_square_input_warning_required,
    witness_spectral_pairwise_input_tag,
)

_SQUARE_AFFINITIES = ("precomputed", "precomputed_nearest_neighbors")
_ACCEPT_SPARSE_FORMATS = ("csr", "csc", "coo")


def _boolean(value: object) -> bool:
    return isinstance(value, bool)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _shape_2d(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in value)
    )


def _string_tuple(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 1
        and all(isinstance(item, str) and item != "" for item in value)
    )


@register_atom(witness_spectral_fit_accept_sparse_formats)
@icontract.require(
    lambda parent_accept_sparse: parent_accept_sparse is None or _string_tuple(parent_accept_sparse),
    "parent_accept_sparse must be None or a tuple of nonempty strings",
)
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == _ACCEPT_SPARSE_FORMATS,
    "accept_sparse formats must be ('csr', 'csc', 'coo')",
)
def spectral_fit_accept_sparse_formats(
    parent_accept_sparse: tuple[str, ...] | None = None,
) -> tuple[str, str, str]:
    """Expose the sparse formats accepted by SpectralClustering.fit validation."""
    del parent_accept_sparse
    return _ACCEPT_SPARSE_FORMATS


@register_atom(witness_spectral_fit_dtype_name)
@icontract.require(
    lambda parent_dtype_name: parent_dtype_name is None or isinstance(parent_dtype_name, str),
    "parent_dtype_name must be None or a string",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "float64",
    "fit dtype name must be 'float64'",
)
def spectral_fit_dtype_name(parent_dtype_name: str | None = None) -> str:
    """Expose the dtype name used by SpectralClustering.fit validation."""
    del parent_dtype_name
    return "float64"


@register_atom(witness_spectral_fit_affinity_allows_square_input)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.ensure(lambda result: _boolean(result), "result must be boolean")
def spectral_fit_affinity_allows_square_input(affinity: str) -> bool:
    """Return whether SpectralClustering.fit accepts square inputs without warning for this affinity."""
    return affinity in _SQUARE_AFFINITIES


@register_atom(witness_spectral_fit_square_input_warning_required)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.require(lambda shape: _shape_2d(shape), "shape must be a 2D positive integer shape tuple")
@icontract.ensure(lambda result: _boolean(result), "result must be boolean")
def spectral_fit_square_input_warning_required(
    affinity: str,
    shape: tuple[int, int],
) -> bool:
    """Decide whether SpectralClustering.fit would warn about square input being treated as data."""
    return int(shape[0]) == int(shape[1]) and not spectral_fit_affinity_allows_square_input(affinity)


@register_atom(witness_spectral_pairwise_input_tag)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.require(lambda parent_pairwise: _boolean(parent_pairwise), "parent_pairwise must be boolean")
@icontract.ensure(lambda result: _boolean(result), "result must be boolean")
def spectral_pairwise_input_tag(affinity: str, parent_pairwise: bool) -> bool:
    """Override SpectralClustering's pairwise-input tag from the configured affinity."""
    del parent_pairwise
    return affinity in _SQUARE_AFFINITIES
