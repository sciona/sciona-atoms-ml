"""Ghost witnesses for public sklearn SVM API-shell atoms."""

from __future__ import annotations


def witness_svm_estimator_catalog(
    catalog_scope: str = "public_estimators",
) -> tuple[str, ...]:
    """Describe the public sklearn.svm estimator names covered by this shell."""
    del catalog_scope
    return ("SVC", "NuSVC", "SVR", "NuSVR", "LinearSVC", "LinearSVR", "OneClassSVM")


def witness_svm_estimator_backend(estimator_name: str) -> str:
    """Describe whether a public SVM estimator delegates to libsvm or liblinear."""
    return "liblinear" if estimator_name in {"LinearSVC", "LinearSVR"} else "libsvm"


def witness_svm_estimator_task(estimator_name: str) -> str:
    """Describe the high-level learning task exposed by a public SVM estimator."""
    if estimator_name in {"SVC", "NuSVC", "LinearSVC"}:
        return "classification"
    if estimator_name in {"SVR", "NuSVR", "LinearSVR"}:
        return "regression"
    return "outlier_detection"


def witness_svm_estimator_methods(
    estimator_name: str,
    *,
    probability_enabled: bool = False,
) -> tuple[str, ...]:
    """Describe public methods exposed for framework-level SVM selection."""
    del probability_enabled
    return (estimator_name,)


def witness_svm_linear_fit_liblinear_payload(
    estimator_name: str,
    C: float,
    fit_intercept: bool,
    intercept_scaling: float,
    class_weight: object,
    penalty: str,
    dual: bool,
    verbose: int,
    max_iter: int,
    tol: float,
    random_state: object,
    multi_class: str,
    loss: str,
    *,
    epsilon: float = 0.0,
    sample_weight: object = None,
) -> dict[str, object]:
    """Describe the _fit_liblinear call payload assembled by linear SVM fits."""
    return {
        "estimator_name": estimator_name,
        "C": C,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
        "class_weight": class_weight,
        "penalty": penalty,
        "dual": dual,
        "verbose": verbose,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": random_state,
        "multi_class": multi_class,
        "loss": loss,
        "epsilon": epsilon,
        "sample_weight": sample_weight,
    }


def witness_svm_prediction_method_payload(
    estimator: object,
    method_name: str,
    X: object,
    *,
    check_probability: bool = True,
) -> dict[str, object]:
    """Describe a public SVM prediction-like method call payload."""
    del check_probability
    return {"estimator": estimator, "method_name": method_name, "args": (X,), "kwargs": {}}


def witness_svm_fit_return_self(estimator: object) -> object:
    """Describe public SVM fit returning the fitted estimator object."""
    return estimator


def witness_svm_libsvm_fitted_support_state(estimator: object) -> dict[str, object]:
    """Describe fitted support-vector state exposed after libsvm fitting."""
    return {"estimator": estimator}


def witness_svm_liblinear_fitted_state(estimator: object) -> dict[str, object]:
    """Describe fitted linear coefficient state exposed after liblinear fitting."""
    return {"estimator": estimator}
