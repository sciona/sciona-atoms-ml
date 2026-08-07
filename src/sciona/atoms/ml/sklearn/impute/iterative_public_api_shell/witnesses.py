"""Ghost witnesses for public IterativeImputer API-shell atoms."""

from __future__ import annotations


def witness_iterative_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public iterative-imputer estimator names covered by this shell."""
    del catalog_scope
    return ("IterativeImputer",)


def witness_iterative_public_callback_boundary(estimator_name: str) -> str:
    """Describe the callback boundary behind public IterativeImputer execution."""
    del estimator_name
    return "per_feature_estimator_fit_predict_callbacks"


def witness_iterative_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for iterative-imputer routing."""
    return (estimator_name,)


def witness_iterative_public_fit_method_payload(
    estimator: object,
    X: object,
    *,
    fit_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public IterativeImputer fit callback payload."""
    kwargs: dict[str, object] = {}
    if fit_params is not None:
        kwargs.update(fit_params)
    return {"estimator": estimator, "method_name": "fit", "args": (X,), "kwargs": kwargs}


def witness_iterative_public_transform_method_payload(estimator: object, X: object) -> dict[str, object]:
    """Describe a public IterativeImputer transform callback payload."""
    return {"estimator": estimator, "method_name": "transform", "args": (X,), "kwargs": {}}


def witness_iterative_public_fit_return_self(estimator: object) -> object:
    """Describe public IterativeImputer fit returning the fitted estimator."""
    return estimator


def witness_iterative_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted IterativeImputer state after delegated callbacks."""
    return {"estimator": estimator}
