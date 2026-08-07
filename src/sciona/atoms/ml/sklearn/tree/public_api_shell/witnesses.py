"""Ghost witnesses for public sklearn decision-tree API-shell atoms."""

from __future__ import annotations


def witness_tree_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe the public sklearn.tree estimator names covered by this shell."""
    del catalog_scope
    return ("DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreeClassifier", "ExtraTreeRegressor")


def witness_tree_estimator_family(estimator_name: str) -> str:
    """Describe whether the estimator is deterministic or extra-randomized."""
    return "extra_tree" if estimator_name.startswith("Extra") else "decision_tree"


def witness_tree_estimator_task(estimator_name: str) -> str:
    """Describe the high-level task exposed by a public tree estimator."""
    return "classification" if estimator_name.endswith("Classifier") else "regression"


def witness_tree_estimator_backend(estimator_name: str) -> str:
    """Describe the native backend family behind a public tree estimator."""
    del estimator_name
    return "cython_tree_builder"


def witness_tree_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level tree routing."""
    return (estimator_name,)


def witness_tree_fit_method_payload(
    estimator: object,
    X: object,
    y: object,
    *,
    sample_weight: object = None,
    check_input: bool = True,
) -> dict[str, object]:
    """Describe a public tree fit-method callback payload."""
    kwargs: dict[str, object] = {"check_input": check_input}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    return {"estimator": estimator, "method_name": "fit", "args": (X, y), "kwargs": kwargs}


def witness_tree_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe a public tree prediction-like method callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_tree_fit_return_self(estimator: object) -> object:
    """Describe public tree fit returning the fitted estimator object."""
    return estimator


def witness_tree_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted tree state exposed after native tree building."""
    return {"estimator": estimator}

