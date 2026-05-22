"""Ghost witnesses for sklearn LogisticRegression fit post-path atoms."""

from __future__ import annotations


def witness_logistic_fit_path_results(path_results: object) -> object:
    """Describe unzipping _logistic_regression_path results."""
    return path_results


def witness_logistic_fit_n_iter_from_path_results(n_iter: object) -> object:
    """Describe n_iter_ packaging after _logistic_regression_path returns."""
    return n_iter


def witness_logistic_fit_coef_with_intercept(
    fold_coefs: object,
    multi_class: str,
    n_classes: int,
    n_features: int,
    fit_intercept: bool,
) -> object:
    """Describe coefficient matrix packaging before the intercept split."""
    return (fold_coefs, multi_class, n_classes, n_features, fit_intercept)


def witness_logistic_fit_final_coef(coef_with_intercept: object, fit_intercept: bool) -> object:
    """Describe final coef_ extraction from a packaged coefficient matrix."""
    return (coef_with_intercept, fit_intercept)


def witness_logistic_fit_final_intercept(
    coef_with_intercept: object,
    n_classes: int,
    fit_intercept: bool,
) -> object:
    """Describe final intercept_ extraction from a packaged coefficient matrix."""
    return (coef_with_intercept, n_classes, fit_intercept)
