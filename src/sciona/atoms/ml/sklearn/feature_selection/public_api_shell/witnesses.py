"""Ghost witnesses for public feature-selection selector API-shell atoms."""

from __future__ import annotations


def witness_feature_selector_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public feature-selection selector names covered by this shell."""
    del catalog_scope
    return ("RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector")


def witness_feature_selector_family(estimator_name: str) -> str:
    """Describe the public feature-selector family."""
    if estimator_name == "RFECV":
        return "recursive_feature_elimination_cv"
    if estimator_name == "RFE":
        return "recursive_feature_elimination"
    if estimator_name == "SelectFromModel":
        return "select_from_model"
    return "sequential_feature_selector"


def witness_feature_selector_boundary(estimator_name: str) -> str:
    """Describe the callback boundary behind a public feature selector."""
    if estimator_name == "RFECV":
        return "estimator_importance_cv_scorer_callbacks"
    if estimator_name == "SequentialFeatureSelector":
        return "estimator_cv_scorer_callbacks"
    return "estimator_importance_callbacks"


def witness_feature_selector_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for feature-selector routing."""
    return (estimator_name,)


def witness_feature_selector_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public feature-selector fit callback payload."""
    args = (X,) if y is None else (X, y)
    kwargs = {} if params is None else dict(params)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


def witness_feature_selector_transform_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public feature-selector transform-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_feature_selector_fit_return_self(estimator: object) -> object:
    """Describe public feature-selector fit returning the fitted selector."""
    return estimator


def witness_feature_selector_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted feature-selector state after delegated callbacks."""
    return {"estimator": estimator}
