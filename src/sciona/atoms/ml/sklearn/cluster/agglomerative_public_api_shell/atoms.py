"""Public agglomerative hierarchy API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_public_estimator_axis,
    witness_agglomerative_public_estimator_catalog,
    witness_agglomerative_public_estimator_methods,
    witness_agglomerative_public_fit_method_payload,
    witness_agglomerative_public_fit_return_self,
    witness_agglomerative_public_fitted_state_summary,
    witness_agglomerative_public_linkage_boundary,
    witness_feature_agglomeration_transform_payload,
)

_ESTIMATOR_NAMES = ("AgglomerativeClustering", "FeatureAgglomeration")
_METHODS = {
    "AgglomerativeClustering": ("fit", "fit_predict"),
    "FeatureAgglomeration": ("fit", "fit_transform", "transform", "inverse_transform"),
}
_LINKAGES = {"ward", "complete", "average", "single"}
_BOUNDARIES = {"ward_tree_or_scipy_hierarchy", "linkage_tree_builder"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _known_linkage(value: object) -> bool:
    return value in _LINKAGES


def _finite_dense_matrix(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _public_agglomerative_estimator(estimator: object) -> bool:
    from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration

    return isinstance(estimator, (AgglomerativeClustering, FeatureAgglomeration))


def _public_feature_agglomeration(estimator: object) -> bool:
    from sklearn.cluster import FeatureAgglomeration

    return isinstance(estimator, FeatureAgglomeration)


def _fitted_public_agglomerative(estimator: object) -> bool:
    return bool(
        _public_agglomerative_estimator(estimator)
        and hasattr(estimator, "children_")
        and hasattr(estimator, "labels_")
        and hasattr(estimator, "n_clusters_")
        and hasattr(estimator, "n_leaves_")
        and hasattr(estimator, "n_connected_components_")
        and hasattr(estimator, "n_features_in_")
    )


def _children_valid(estimator: object) -> bool:
    if not _fitted_public_agglomerative(estimator):
        return False
    children = np.asarray(getattr(estimator, "children_"))
    labels = np.asarray(getattr(estimator, "labels_"))
    return bool(
        children.ndim == 2
        and children.shape[1] == 2
        and children.shape[0] >= 1
        and labels.ndim == 1
        and labels.shape[0] >= 1
    )


def _fit_payload_valid(result: object, estimator: object, X: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "fit"
        and isinstance(args, tuple)
        and len(args) == 1
        and args[0] is X
        and result.get("kwargs") == {}
    )


def _transform_payload_valid(result: object, estimator: object, X: object) -> bool:
    if not isinstance(result, dict):
        return False
    args = result.get("args")
    return bool(
        result.get("estimator") is estimator
        and result.get("method_name") == "transform"
        and isinstance(args, tuple)
        and len(args) == 1
        and args[0] is X
        and result.get("kwargs") == {}
    )


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(value).shape)


@register_atom(witness_agglomerative_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose covered public agglomerative estimators")
def agglomerative_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose public agglomerative estimator names for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_agglomerative_public_estimator_axis)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public agglomerative estimator")
@icontract.ensure(lambda result: result in {"samples", "features"}, "axis must identify samples or features")
def agglomerative_public_estimator_axis(estimator_name: str) -> str:
    """Return whether a public agglomerative estimator clusters samples or features."""
    return "features" if estimator_name == "FeatureAgglomeration" else "samples"


@register_atom(witness_agglomerative_public_linkage_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public agglomerative estimator")
@icontract.require(lambda linkage: _known_linkage(linkage), "linkage must be a public agglomerative linkage")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name a covered tree-construction family")
def agglomerative_public_linkage_boundary(estimator_name: str, linkage: str) -> str:
    """Return the tree-construction boundary selected by a public agglomerative estimator."""
    del estimator_name
    if linkage == "ward":
        return "ward_tree_or_scipy_hierarchy"
    return "linkage_tree_builder"


@register_atom(witness_agglomerative_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must name a covered public agglomerative estimator")
@icontract.ensure(lambda result: isinstance(result, tuple) and "fit" in result, "methods must include fit")
def agglomerative_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level agglomerative routing."""
    return _METHODS[estimator_name]


@register_atom(witness_agglomerative_public_fit_method_payload)
@icontract.require(lambda estimator: _public_agglomerative_estimator(estimator), "estimator must be a public agglomerative estimator")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X: _fit_payload_valid(result, estimator, X), "fit payload must preserve estimator and positional X input")
def agglomerative_public_fit_method_payload(estimator: object, X: object) -> dict[str, object]:
    """Package a public agglomerative fit call without constructing the hierarchy."""
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": {}}


@register_atom(witness_feature_agglomeration_transform_payload)
@icontract.require(lambda estimator: _fitted_public_agglomerative(estimator) and _public_feature_agglomeration(estimator), "estimator must be a fitted public FeatureAgglomeration")
@icontract.require(lambda X: _finite_dense_matrix(X), "X must be a finite dense 2D array")
@icontract.require(lambda estimator, X: np.asarray(X).shape[1] == getattr(estimator, "n_features_in_"), "X feature count must match fitted state")
@icontract.ensure(lambda result, estimator, X: _transform_payload_valid(result, estimator, X), "transform payload must preserve estimator and positional X input")
def feature_agglomeration_transform_payload(estimator: object, X: object) -> dict[str, object]:
    """Expose a public FeatureAgglomeration transform payload without pooling features."""
    return {"estimator": estimator, "method_name": "transform", "args": (X,), "kwargs": {}}


@register_atom(witness_agglomerative_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_agglomerative(estimator), "estimator must be a fitted public agglomerative estimator")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_agglomerative(result), "fit shell must return fitted self")
def agglomerative_public_fit_return_self(estimator: object) -> object:
    """Return the fitted agglomerative estimator from the public fit shell."""
    return estimator


@register_atom(witness_agglomerative_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_agglomerative(estimator), "estimator must be a fitted public agglomerative estimator")
@icontract.require(lambda estimator: _children_valid(estimator), "fitted hierarchy must expose children and labels")
@icontract.ensure(lambda result: isinstance(result, dict) and result["axis"] in {"samples", "features"}, "summary must expose clustered axis metadata")
@icontract.ensure(lambda result: result["n_clusters"] >= 1 and result["n_leaves"] >= 1, "summary must expose cluster and leaf counts")
def agglomerative_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted agglomerative hierarchy state after delegated tree construction."""
    name = estimator.__class__.__name__
    state: dict[str, object] = {
        "estimator_name": name,
        "axis": agglomerative_public_estimator_axis(name),
        "linkage": str(getattr(estimator, "linkage")),
        "linkage_boundary": agglomerative_public_linkage_boundary(name, str(getattr(estimator, "linkage"))),
        "metric": str(getattr(estimator, "metric")),
        "n_clusters": int(getattr(estimator, "n_clusters_")),
        "n_leaves": int(getattr(estimator, "n_leaves_")),
        "n_connected_components": int(getattr(estimator, "n_connected_components_")),
        "feature_count": int(getattr(estimator, "n_features_in_")),
        "children_shape": _shape_tuple(getattr(estimator, "children_")),
        "labels_shape": _shape_tuple(getattr(estimator, "labels_")),
        "label_count": int(np.asarray(getattr(estimator, "labels_")).shape[0]),
    }
    if hasattr(estimator, "distances_"):
        state["distances_shape"] = _shape_tuple(getattr(estimator, "distances_"))
    if name == "FeatureAgglomeration":
        state["pooling_boundary"] = "feature_pooling_callback"
    return state
