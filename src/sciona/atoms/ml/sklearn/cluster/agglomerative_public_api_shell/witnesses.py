"""Ghost witnesses for public agglomerative hierarchy API-shell atoms."""

from __future__ import annotations


def witness_agglomerative_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public agglomerative estimator names covered by this shell."""
    del catalog_scope
    return ("AgglomerativeClustering", "FeatureAgglomeration")


def witness_agglomerative_public_estimator_axis(estimator_name: str) -> str:
    """Describe whether the public estimator clusters samples or features."""
    return "features" if estimator_name == "FeatureAgglomeration" else "samples"


def witness_agglomerative_public_linkage_boundary(estimator_name: str, linkage: str) -> str:
    """Describe the tree-construction boundary behind a linkage setting."""
    del estimator_name
    if linkage == "ward":
        return "ward_tree_or_scipy_hierarchy"
    return "linkage_tree_builder"


def witness_agglomerative_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for agglomerative routing."""
    return (estimator_name,)


def witness_agglomerative_public_fit_method_payload(estimator: object, X: object) -> dict[str, object]:
    """Describe a public agglomerative fit callback payload."""
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": {}}


def witness_feature_agglomeration_transform_payload(estimator: object, X: object) -> dict[str, object]:
    """Describe a public FeatureAgglomeration transform callback payload."""
    return {"estimator": estimator, "method_name": "transform", "args": (X,), "kwargs": {}}


def witness_agglomerative_public_fit_return_self(estimator: object) -> object:
    """Describe public agglomerative fit returning the fitted estimator."""
    return estimator


def witness_agglomerative_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted agglomerative hierarchy state after delegated tree construction."""
    return {"estimator": estimator}
