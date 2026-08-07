"""Ghost witnesses for public spectral clustering API-shell atoms."""

from __future__ import annotations


def witness_spectral_public_surface_catalog(catalog_scope: str = "public_surfaces") -> tuple[str, ...]:
    """Describe public spectral clustering surfaces covered by this shell."""
    del catalog_scope
    return ("SpectralClustering", "spectral_clustering", "SpectralBiclustering", "SpectralCoclustering")


def witness_spectral_public_surface_family(surface_name: str) -> str:
    """Describe the public spectral surface family."""
    if surface_name in {"SpectralClustering", "spectral_clustering"}:
        return "spectral_clustering"
    return "spectral_biclustering"


def witness_spectral_public_solver_boundary(surface_name: str, solver_name: str) -> str:
    """Describe the delegated eigensolver, SVD, or assignment boundary."""
    del surface_name
    if solver_name in {"kmeans", "cluster_qr", "discretize"}:
        return "label_assignment_callback"
    if solver_name in {"arpack", "lobpcg", "amg", "eigen_solver", "spectral_embedding"}:
        return "spectral_embedding_eigensolver"
    return "bicluster_svd_or_kmeans_callback"


def witness_spectral_public_fit_payload(estimator: object, X: object) -> dict[str, object]:
    """Describe a public spectral estimator fit callback payload."""
    return {"surface": estimator, "method_name": "fit", "args": (X,), "kwargs": {}}


def witness_spectral_clustering_function_payload(affinity: object, *, n_clusters: int = 8) -> dict[str, object]:
    """Describe a public spectral_clustering function callback payload."""
    return {"surface": "spectral_clustering", "args": (affinity,), "kwargs": {"n_clusters": n_clusters}}


def witness_spectral_public_fit_return_self(estimator: object) -> object:
    """Describe public spectral fit returning the fitted estimator."""
    return estimator


def witness_spectral_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted spectral state after delegated callbacks."""
    return {"estimator": estimator}
