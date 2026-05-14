"""Ghost witnesses for sklearn SGD classifier fit callback atoms."""

from __future__ import annotations


def witness_sgd_classifier_fit_more_validate_params_result(validation_result: object) -> object:
    """Describe the unused result of _more_validate_params()."""
    return validation_result


def witness_sgd_classifier_fit_c_value(alpha: float) -> float:
    """Describe the fixed C value passed by BaseSGDClassifier.fit."""
    return 1.0


def witness_sgd_classifier_fit_callback_payload(
    X: object,
    y: object,
    alpha: float,
    loss: str,
    learning_rate: str,
    coef_init: object,
    intercept_init: object,
    sample_weight: object,
) -> dict[str, object]:
    """Describe the keyword payload passed from fit to _fit."""
    return {
        "X": X,
        "y": y,
        "alpha": alpha,
        "C": 1.0,
        "loss": loss,
        "learning_rate": learning_rate,
        "coef_init": coef_init,
        "intercept_init": intercept_init,
        "sample_weight": sample_weight,
    }


def witness_sgd_classifier_fit_result(fit_result: object) -> object:
    """Describe the object returned by the delegated _fit callback."""
    return fit_result
