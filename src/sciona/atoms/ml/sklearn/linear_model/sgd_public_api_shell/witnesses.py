"""Ghost witnesses for public SGD-family estimator API-shell atoms."""

from __future__ import annotations


def witness_sgd_public_estimator_catalog(catalog_scope: str = "public_estimators") -> tuple[str, ...]:
    """Describe public SGD-family estimator names covered by this shell."""
    del catalog_scope
    return (
        "SGDClassifier",
        "SGDRegressor",
        "SGDOneClassSVM",
        "PassiveAggressiveClassifier",
        "PassiveAggressiveRegressor",
        "Perceptron",
    )


def witness_sgd_public_estimator_family(estimator_name: str) -> str:
    """Describe the public SGD-family estimator family."""
    if estimator_name == "SGDClassifier":
        return "sgd_classifier"
    if estimator_name == "SGDRegressor":
        return "sgd_regressor"
    if estimator_name == "SGDOneClassSVM":
        return "sgd_one_class_svm"
    if estimator_name == "PassiveAggressiveClassifier":
        return "passive_aggressive_classifier"
    if estimator_name == "PassiveAggressiveRegressor":
        return "passive_aggressive_regressor"
    return "perceptron"


def witness_sgd_public_estimator_task(estimator_name: str) -> str:
    """Describe the learning task for a public SGD-family estimator."""
    if estimator_name in {"SGDRegressor", "PassiveAggressiveRegressor"}:
        return "regression"
    if estimator_name == "SGDOneClassSVM":
        return "anomaly_detection"
    return "classification"


def witness_sgd_public_training_boundary(estimator_name: str) -> str:
    """Describe the compiled training boundary behind a public SGD-family estimator."""
    if estimator_name == "SGDOneClassSVM":
        return "compiled_one_class_plain_sgd"
    return "compiled_plain_sgd"


def witness_sgd_public_estimator_methods(estimator_name: str) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level SGD-family routing."""
    return (estimator_name,)


def witness_sgd_public_fit_method_payload(
    estimator: object,
    X: object,
    y: object = None,
    *,
    sample_weight: object = None,
    coef_init: object = None,
    intercept_init: object = None,
    offset_init: object = None,
) -> dict[str, object]:
    """Describe a public SGD-family fit callback payload."""
    kwargs: dict[str, object] = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if coef_init is not None:
        kwargs["coef_init"] = coef_init
    if intercept_init is not None:
        kwargs["intercept_init"] = intercept_init
    if offset_init is not None:
        kwargs["offset_init"] = offset_init
    args = (X,) if y is None else (X, y)
    return {"estimator": estimator, "method_name": "fit", "args": args, "kwargs": kwargs}


def witness_sgd_public_prediction_method_payload(estimator: object, method_name: str, X: object) -> dict[str, object]:
    """Describe an SGD-family prediction-like callback payload."""
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_sgd_public_fit_return_self(estimator: object) -> object:
    """Describe public SGD-family fit returning the fitted estimator object."""
    return estimator


def witness_sgd_public_fitted_state_summary(estimator: object) -> dict[str, object]:
    """Describe fitted SGD-family state exposed after compiled training."""
    return {"estimator": estimator}
