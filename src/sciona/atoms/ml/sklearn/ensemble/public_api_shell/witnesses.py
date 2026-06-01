"""Ghost witnesses for public sklearn ensemble API-shell atoms."""

from __future__ import annotations


def witness_ensemble_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Describe the public tree-ensemble estimators covered by this shell."""
    del catalog_scope
    return (
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "IsolationForest",
    )


def witness_ensemble_estimator_family(estimator_name: str) -> str:
    """Describe the public ensemble family for an estimator name."""
    return estimator_name


def witness_ensemble_estimator_task(estimator_name: str) -> str:
    """Describe the high-level learning task for a tree-ensemble estimator."""
    return estimator_name


def witness_ensemble_estimator_backend(estimator_name: str) -> str:
    """Describe the native/backend boundary for a tree-ensemble estimator."""
    return estimator_name


def witness_ensemble_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level ensemble routing."""
    return (estimator_name,)


def witness_ensemble_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
) -> dict[str, object]:
    """Describe a public ensemble prediction-like method call payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_ensemble_fit_return_self(estimator: object) -> object:
    """Describe public ensemble fit returning the fitted estimator object."""
    return estimator


def witness_ensemble_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted public ensemble state after native/tree-backed fitting."""
    return {"estimator": estimator}
