"""Spectral clustering fit-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_n_components,
    witness_spectral_fit_return_self,
    witness_spectral_fit_use_cluster_qr,
    witness_spectral_fit_use_kmeans,
    witness_spectral_fit_verbose_message,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _optional_positive_int(value: object) -> bool:
    return bool(value is None or _positive_int(value))


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


@register_atom(witness_spectral_fit_n_components)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.require(lambda n_components=None: _optional_positive_int(n_components), "n_components must be None or a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "resolved n_components must be a positive integer")
def spectral_fit_n_components(
    n_clusters: int,
    n_components: int | None = None,
) -> int:
    """Resolve SpectralClustering.fit's effective embedding width."""
    return int(n_clusters) if n_components is None else int(n_components)


@register_atom(witness_spectral_fit_verbose_message)
@icontract.require(lambda assign_labels: _nonempty_string(assign_labels), "assign_labels must be a nonempty string")
@icontract.ensure(
    lambda result, assign_labels: isinstance(result, str) and result == f"Computing label assignment using {assign_labels}",
    "verbose message must match SpectralClustering.fit",
)
def spectral_fit_verbose_message(assign_labels: str) -> str:
    """Format SpectralClustering.fit's verbose label-assignment message."""
    return f"Computing label assignment using {assign_labels}"


@register_atom(witness_spectral_fit_use_kmeans)
@icontract.require(lambda assign_labels: _nonempty_string(assign_labels), "assign_labels must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def spectral_fit_use_kmeans(assign_labels: str) -> bool:
    """Return whether SpectralClustering.fit takes the k-means label-assignment branch."""
    return assign_labels == "kmeans"


@register_atom(witness_spectral_fit_use_cluster_qr)
@icontract.require(lambda assign_labels: _nonempty_string(assign_labels), "assign_labels must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def spectral_fit_use_cluster_qr(assign_labels: str) -> bool:
    """Return whether SpectralClustering.fit takes the cluster_qr label-assignment branch."""
    return assign_labels == "cluster_qr"


@register_atom(witness_spectral_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(
    lambda result, estimator_token: isinstance(result, str) and result == estimator_token,
    "result must return the estimator token unchanged",
)
def spectral_fit_return_self(estimator_token: str) -> str:
    """Model SpectralClustering.fit returning the fitted estimator itself."""
    return estimator_token
