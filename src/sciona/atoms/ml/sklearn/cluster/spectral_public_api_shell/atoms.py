"""Public spectral clustering API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_clustering_function_payload,
    witness_spectral_public_fit_payload,
    witness_spectral_public_fit_return_self,
    witness_spectral_public_fitted_state_summary,
    witness_spectral_public_solver_boundary,
    witness_spectral_public_surface_catalog,
    witness_spectral_public_surface_family,
)

_SURFACES = ("SpectralClustering", "spectral_clustering", "SpectralBiclustering", "SpectralCoclustering")
_FAMILIES = {"spectral_clustering", "spectral_biclustering"}
_BOUNDARIES = {"spectral_embedding_eigensolver", "label_assignment_callback", "bicluster_svd_or_kmeans_callback"}
_ASSIGN_LABELS = {"kmeans", "cluster_qr", "discretize"}
_EIGEN_SOLVERS = {None, "arpack", "lobpcg", "amg"}
_SVD_METHODS = {"randomized", "arpack"}


def _known_surface(value: object) -> bool:
    return value in _SURFACES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _public_spectral_estimator(estimator: object) -> bool:
    from sklearn.cluster import SpectralBiclustering, SpectralClustering, SpectralCoclustering

    return isinstance(estimator, (SpectralClustering, SpectralBiclustering, SpectralCoclustering))


def _public_spectral_clustering(estimator: object) -> bool:
    from sklearn.cluster import SpectralClustering

    return isinstance(estimator, SpectralClustering)


def _public_bicluster(estimator: object) -> bool:
    from sklearn.cluster import SpectralBiclustering, SpectralCoclustering

    return isinstance(estimator, (SpectralBiclustering, SpectralCoclustering))


def _fitted_public_spectral(estimator: object) -> bool:
    if not _public_spectral_estimator(estimator):
        return False
    if _public_spectral_clustering(estimator):
        return bool(hasattr(estimator, "labels_") and hasattr(estimator, "affinity_matrix_") and hasattr(estimator, "n_features_in_"))
    return bool(
        hasattr(estimator, "rows_")
        and hasattr(estimator, "columns_")
        and hasattr(estimator, "row_labels_")
        and hasattr(estimator, "column_labels_")
        and hasattr(estimator, "n_features_in_")
    )


def _solver_boundary_valid(surface_name: str, solver_name: object) -> bool:
    if surface_name in {"SpectralClustering", "spectral_clustering"}:
        return solver_name in _ASSIGN_LABELS or solver_name in _EIGEN_SOLVERS or solver_name in {"eigen_solver", "spectral_embedding"}
    return solver_name in _SVD_METHODS or solver_name in {"kmeans", "minibatch_kmeans", "svd"}


def _fit_payload_valid(result: object, estimator: object, X: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    return bool(
        result.get("surface") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == 1
        and args[0] is X
        and result.get("kwargs") == {}
    )


def _function_payload_valid(result: object, affinity: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    kwargs = result.get("kwargs")
    return bool(result.get("surface") == "spectral_clustering" and isinstance(args, tuple) and len(args) == 1 and args[0] is affinity and isinstance(kwargs, dict))


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


@register_atom(witness_spectral_public_surface_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_surfaces", "catalog_scope must be 'public_surfaces'")
@icontract.ensure(lambda result: result == _SURFACES, "catalog must expose covered public spectral surfaces")
def spectral_public_surface_catalog(catalog_scope: str = "public_surfaces") -> tuple[str, ...]:
    """Expose public spectral clustering surface names for framework selection."""
    del catalog_scope
    return _SURFACES


@register_atom(witness_spectral_public_surface_family)
@icontract.require(lambda surface_name: _known_surface(surface_name), "surface_name must name a covered public spectral surface")
@icontract.ensure(lambda result: result in _FAMILIES, "family must be a covered spectral family")
def spectral_public_surface_family(surface_name: str) -> str:
    """Return the public spectral clustering family for a surface."""
    if surface_name in {"SpectralClustering", "spectral_clustering"}:
        return "spectral_clustering"
    return "spectral_biclustering"


@register_atom(witness_spectral_public_solver_boundary)
@icontract.require(lambda surface_name: _known_surface(surface_name), "surface_name must name a covered public spectral surface")
@icontract.require(lambda surface_name, solver_name: _solver_boundary_valid(surface_name, solver_name), "solver_name must match a public spectral solver or assignment option")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered spectral callback family")
def spectral_public_solver_boundary(surface_name: str, solver_name: object) -> str:
    """Return the eigensolver, SVD, or label-assignment boundary for a spectral surface."""
    if surface_name in {"SpectralClustering", "spectral_clustering"}:
        if solver_name in _ASSIGN_LABELS:
            return "label_assignment_callback"
        return "spectral_embedding_eigensolver"
    return "bicluster_svd_or_kmeans_callback"


@register_atom(witness_spectral_public_fit_payload)
@icontract.require(lambda estimator: _public_spectral_estimator(estimator), "estimator must be a public spectral estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X: _fit_payload_valid(result, estimator, X), "fit payload must preserve estimator and positional X input")
def spectral_public_fit_payload(estimator: object, X: object) -> dict[str, object]:
    """Package a public spectral estimator fit call without executing solver callbacks."""
    return {"surface": estimator, "method_name": "fit", "args": (X,), "kwargs": {}}


@register_atom(witness_spectral_clustering_function_payload)
@icontract.require(lambda affinity: _finite_dense_matrix(affinity), "affinity must be a finite dense 2D matrix")
@icontract.require(lambda affinity: np.asarray(affinity).shape[0] == np.asarray(affinity).shape[1], "affinity must be square")
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be positive")
@icontract.ensure(lambda result, affinity: _function_payload_valid(result, affinity), "function payload must preserve positional affinity input")
def spectral_clustering_function_payload(affinity: object, *, n_clusters: int = 8) -> dict[str, object]:
    """Package a public spectral_clustering function call without fitting the estimator wrapper."""
    return {"surface": "spectral_clustering", "args": (affinity,), "kwargs": {"n_clusters": int(n_clusters)}}


@register_atom(witness_spectral_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_spectral(estimator), "estimator must be a fitted public spectral estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_spectral(result), "fit shell must return fitted self")
def spectral_public_fit_return_self(estimator: object) -> object:
    """Return the fitted spectral estimator from the public fit shell."""
    return estimator


@register_atom(witness_spectral_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_spectral(estimator), "estimator must be a fitted public spectral estimator")
@icontract.ensure(lambda result: isinstance(result, dict) and result["family"] in _FAMILIES, "summary must expose spectral family metadata")
@icontract.ensure(lambda result: result["feature_count"] >= 1, "summary must expose fitted feature count")
def spectral_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted spectral state after delegated solver callbacks."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "surface_name": name,
        "family": spectral_public_surface_family(name),
        "feature_count": int(getattr(estimator, "n_features_in_")),
    }
    if _public_spectral_clustering(estimator):
        state.update(
            {
                "eigensolver_boundary": spectral_public_solver_boundary(name, getattr(estimator, "eigen_solver")),
                "assignment_boundary": spectral_public_solver_boundary(name, getattr(estimator, "assign_labels")),
                "affinity": str(getattr(estimator, "affinity")),
                "labels_shape": _shape_tuple(getattr(estimator, "labels_")),
                "affinity_shape": _shape_tuple(getattr(estimator, "affinity_matrix_")),
                "cluster_count": int(np.unique(getattr(estimator, "labels_")).shape[0]),
            }
        )
        return state
    state.update(
        {
            "svd_boundary": spectral_public_solver_boundary(name, getattr(estimator, "svd_method")),
            "assignment_boundary": "bicluster_svd_or_kmeans_callback",
            "rows_shape": _shape_tuple(getattr(estimator, "rows_")),
            "columns_shape": _shape_tuple(getattr(estimator, "columns_")),
            "row_labels_shape": _shape_tuple(getattr(estimator, "row_labels_")),
            "column_labels_shape": _shape_tuple(getattr(estimator, "column_labels_")),
            "row_cluster_count": int(np.unique(getattr(estimator, "row_labels_")).shape[0]),
            "column_cluster_count": int(np.unique(getattr(estimator, "column_labels_")).shape[0]),
            "mini_batch": bool(getattr(estimator, "mini_batch")),
        }
    )
    return state
