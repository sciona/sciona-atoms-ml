"""Public IterativeImputer API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterative_public_callback_boundary,
    witness_iterative_public_estimator_catalog,
    witness_iterative_public_estimator_methods,
    witness_iterative_public_fit_method_payload,
    witness_iterative_public_fit_return_self,
    witness_iterative_public_fitted_state_summary,
    witness_iterative_public_transform_method_payload,
)

_ESTIMATOR_NAMES = ("IterativeImputer",)
_METHODS = ("fit", "fit_transform", "transform", "get_feature_names_out")
_BOUNDARIES = {"per_feature_estimator_fit_predict_callbacks"}


def _known_estimator_name(value: object) -> bool:
    return value in _ESTIMATOR_NAMES


def _finite_dense_matrix_with_missing(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array) | np.isnan(array))
    )


def _fit_params_valid(fit_params: object) -> bool:
    if fit_params is None:
        return True
    return bool(isinstance(fit_params, dict) and all(isinstance(key, str) and key != "" for key in fit_params))


def _public_iterative_imputer(estimator: object) -> bool:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    return isinstance(estimator, IterativeImputer)


def _fitted_public_iterative_imputer(estimator: object) -> bool:
    return bool(
        _public_iterative_imputer(estimator)
        and hasattr(estimator, "initial_imputer_")
        and hasattr(estimator, "imputation_sequence_")
        and hasattr(estimator, "n_iter_")
        and hasattr(estimator, "n_features_in_")
        and hasattr(estimator, "n_features_with_missing_")
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
        and isinstance(result.get("kwargs"), dict)
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


def _sequence_triplets_valid(estimator: object) -> bool:
    if not _fitted_public_iterative_imputer(estimator):
        return False
    sequence = getattr(estimator, "imputation_sequence_")
    if not isinstance(sequence, list):
        return False
    for triplet in sequence:
        if not hasattr(triplet, "feat_idx") or not hasattr(triplet, "neighbor_feat_idx") or not hasattr(triplet, "estimator"):
            return False
    return True


@register_atom(witness_iterative_public_estimator_catalog)
@icontract.require(lambda catalog_scope: catalog_scope == "public_estimators", "catalog_scope must be 'public_estimators'")
@icontract.ensure(lambda result: result == _ESTIMATOR_NAMES, "catalog must expose IterativeImputer")
def iterative_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Expose the public experimental IterativeImputer estimator name for framework selection."""
    del catalog_scope
    return _ESTIMATOR_NAMES


@register_atom(witness_iterative_public_callback_boundary)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must be IterativeImputer")
@icontract.ensure(lambda result: result in _BOUNDARIES, "boundary must name the per-feature estimator callback family")
def iterative_public_callback_boundary(estimator_name: str) -> str:
    """Return the estimator callback boundary represented by public IterativeImputer execution."""
    del estimator_name
    return "per_feature_estimator_fit_predict_callbacks"


@register_atom(witness_iterative_public_estimator_methods)
@icontract.require(lambda estimator_name: _known_estimator_name(estimator_name), "estimator_name must be IterativeImputer")
@icontract.ensure(lambda result: result == _METHODS, "methods must expose public IterativeImputer methods")
def iterative_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Expose public methods useful for high-level IterativeImputer routing."""
    del estimator_name
    return _METHODS


@register_atom(witness_iterative_public_fit_method_payload)
@icontract.require(lambda estimator: _public_iterative_imputer(estimator), "estimator must be a public IterativeImputer")
@icontract.require(lambda X: _finite_dense_matrix_with_missing(X), "X must be a finite dense 2D array apart from missing entries")
@icontract.require(lambda fit_params: _fit_params_valid(fit_params), "fit_params must be a string-keyed mapping when provided")
@icontract.ensure(lambda result: isinstance(result, dict) and result["method_name"] == "fit", "payload must target public fit")
@icontract.ensure(lambda result, estimator, X: _fit_payload_valid(result, estimator, X), "fit payload must preserve estimator and positional X input")
def iterative_public_fit_method_payload(
    estimator: object,
    X: object,
    *,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Package a public IterativeImputer fit call without executing estimator callbacks."""
    kwargs: dict[str, object] = {}
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": kwargs}


@register_atom(witness_iterative_public_transform_method_payload)
@icontract.require(lambda estimator: _fitted_public_iterative_imputer(estimator), "estimator must be a fitted public IterativeImputer")
@icontract.require(lambda X: _finite_dense_matrix_with_missing(X), "X must be a finite dense 2D array apart from missing entries")
@icontract.require(lambda estimator, X: np.asarray(X).shape[1] == getattr(estimator, "n_features_in_"), "X feature count must match fitted state")
@icontract.ensure(lambda result, estimator, X: _transform_payload_valid(result, estimator, X), "transform payload must preserve estimator and positional X input")
def iterative_public_transform_method_payload(estimator: object, X: object) -> dict[str, object]:
    """Expose a public IterativeImputer transform payload without replaying imputations."""
    return {"estimator": estimator, "method_name": "transform", "args": (X,), "kwargs": {}}


@register_atom(witness_iterative_public_fit_return_self)
@icontract.require(lambda estimator: _fitted_public_iterative_imputer(estimator), "estimator must be a fitted public IterativeImputer")
@icontract.ensure(lambda result, estimator: result is estimator and _fitted_public_iterative_imputer(result), "fit shell must return fitted self")
def iterative_public_fit_return_self(estimator: object) -> object:
    """Return the fitted IterativeImputer from the public fit shell."""
    return estimator


@register_atom(witness_iterative_public_fitted_state_summary)
@icontract.require(lambda estimator: _fitted_public_iterative_imputer(estimator), "estimator must be a fitted public IterativeImputer")
@icontract.require(lambda estimator: _sequence_triplets_valid(estimator), "imputation_sequence_ must expose fitted triplets")
@icontract.ensure(lambda result: isinstance(result, dict) and result["estimator_name"] == "IterativeImputer", "summary must identify IterativeImputer")
@icontract.ensure(lambda result: result["feature_count"] >= 1 and result["n_iter"] >= 0, "summary must expose fitted feature and iteration counts")
def iterative_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Expose compact fitted IterativeImputer state after delegated estimator callbacks."""
    sequence = getattr(estimator, "imputation_sequence_")
    estimator_types = tuple(sorted({triplet.estimator.__class__.__name__ for triplet in sequence}))
    feature_indices = tuple(int(triplet.feat_idx) for triplet in sequence)
    neighbor_counts = tuple(int(np.asarray(triplet.neighbor_feat_idx).shape[0]) for triplet in sequence)
    n_iter = int(getattr(estimator, "n_iter_"))
    sequence_length = len(sequence)
    state: dict[str, object] = {
        "estimator_name": "IterativeImputer",
        "callback_boundary": iterative_public_callback_boundary("IterativeImputer"),
        "feature_count": int(getattr(estimator, "n_features_in_")),
        "features_with_missing": int(getattr(estimator, "n_features_with_missing_")),
        "n_iter": n_iter,
        "sequence_length": sequence_length,
        "imputations_per_round": 0 if n_iter == 0 else int(sequence_length // n_iter),
        "feature_indices": feature_indices,
        "neighbor_counts": neighbor_counts,
        "estimator_types": estimator_types,
        "sample_posterior": bool(getattr(estimator, "sample_posterior")),
        "imputation_order": str(getattr(estimator, "imputation_order")),
        "skip_complete": bool(getattr(estimator, "skip_complete")),
        "initial_strategy": str(getattr(estimator, "initial_strategy")),
        "initial_imputer_type": getattr(estimator, "initial_imputer_").__class__.__name__,
        "empty_feature_count": int(np.sum(getattr(estimator, "_is_empty_feature"))),
    }
    if hasattr(estimator, "indicator_"):
        state["indicator_present"] = getattr(estimator, "indicator_") is not None
    if hasattr(estimator, "_min_value"):
        state["min_value_shape"] = tuple(int(dim) for dim in np.asarray(getattr(estimator, "_min_value")).shape)
    if hasattr(estimator, "_max_value"):
        state["max_value_shape"] = tuple(int(dim) for dim in np.asarray(getattr(estimator, "_max_value")).shape)
    return state
