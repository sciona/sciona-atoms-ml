"""Ghost witnesses for public sklearn orchestration API-shell atoms."""

from __future__ import annotations


def witness_orchestration_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public sklearn orchestration estimator names covered by this shell."""
    del catalog_scope
    return ("Pipeline", "ColumnTransformer", "FeatureUnion", "GridSearchCV", "RandomizedSearchCV")


def witness_orchestration_estimator_family(estimator_name: str) -> str:
    """Describe the public orchestration estimator family."""
    if estimator_name == "Pipeline":
        return "pipeline"
    if estimator_name == "ColumnTransformer":
        return "column_transformer"
    if estimator_name == "FeatureUnion":
        return "feature_union"
    if estimator_name == "GridSearchCV":
        return "grid_search_cv"
    return "randomized_search_cv"


def witness_orchestration_boundary(estimator_name: str) -> str:
    """Describe the callback boundary behind a public orchestration estimator."""
    if estimator_name == "ColumnTransformer":
        return "column_transformer_callbacks"
    if estimator_name == "FeatureUnion":
        return "parallel_transformer_callbacks"
    if estimator_name in {"GridSearchCV", "RandomizedSearchCV"}:
        return "cv_search_fit_score_callbacks"
    return "pipeline_step_callbacks"


def witness_orchestration_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level orchestration routing."""
    return (estimator_name,)


def witness_orchestration_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe a public orchestration fit callback payload."""
    args = (X,) if y is None else (X, y)
    kwargs = {} if params is None else dict(params)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


def witness_orchestration_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public orchestration prediction or transform callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_orchestration_fit_return_self(estimator: object) -> object:
    """Describe public orchestration fit returning the fitted estimator object."""
    return estimator


def witness_orchestration_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted orchestration state exposed after delegated callbacks."""
    return {"estimator": estimator}
