"""Spectral biclustering fit API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_fit_accept_sparse_format,
    witness_bicluster_fit_dtype_name,
    witness_bicluster_sparse_input_tag,
)


def _boolean(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_bicluster_fit_accept_sparse_format)
@icontract.require(
    lambda parent_accept_sparse: parent_accept_sparse is None
    or isinstance(parent_accept_sparse, str)
    or (
        isinstance(parent_accept_sparse, tuple)
        and all(isinstance(item, str) and item != "" for item in parent_accept_sparse)
    ),
    "parent_accept_sparse must be None, a string, or a tuple of nonempty strings",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "csr",
    "accept_sparse format must be 'csr'",
)
def bicluster_fit_accept_sparse_format(
    parent_accept_sparse: str | tuple[str, ...] | None = None,
) -> str:
    """Expose the accept_sparse mode used by spectral biclustering fit validation."""
    del parent_accept_sparse
    return "csr"


@register_atom(witness_bicluster_fit_dtype_name)
@icontract.require(
    lambda parent_dtype_name: parent_dtype_name is None or isinstance(parent_dtype_name, str),
    "parent_dtype_name must be None or a string",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "float64",
    "fit dtype name must be 'float64'",
)
def bicluster_fit_dtype_name(parent_dtype_name: str | None = None) -> str:
    """Expose the dtype name used by spectral biclustering fit validation."""
    del parent_dtype_name
    return "float64"


@register_atom(witness_bicluster_sparse_input_tag)
@icontract.require(lambda parent_sparse: _boolean(parent_sparse), "parent_sparse must be boolean")
@icontract.ensure(
    lambda result: _boolean(result) and result is True,
    "spectral biclustering sparse input tag must be True",
)
def bicluster_sparse_input_tag(parent_sparse: bool) -> bool:
    """Override the sparse-input tag shared by spectral biclustering estimators."""
    del parent_sparse
    return True
