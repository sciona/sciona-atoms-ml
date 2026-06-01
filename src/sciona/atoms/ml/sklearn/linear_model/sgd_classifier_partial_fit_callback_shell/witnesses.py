"""Ghost witnesses for sklearn SGD classifier partial-fit callback atoms."""

from __future__ import annotations


def witness_sgd_classifier_partial_fit_first_call(has_classes: bool) -> bool:
    """Describe the first partial_fit call predicate."""
    return not has_classes


def witness_sgd_classifier_partial_fit_validate_params_kwargs(first_call: bool) -> dict[str, object]:
    """Describe the _more_validate_params kwargs for partial_fit."""
    if not first_call:
        return {}
    return {"for_partial_fit": True}


def witness_sgd_classifier_partial_fit_balanced_class_weight_error(class_weight: object) -> str:
    """Describe the balanced class-weight partial_fit error message."""
    return (
        "class_weight '{0}' is not supported for "
        "partial_fit. In order to use 'balanced' weights,"
        " use compute_class_weight('{0}', "
        "classes=classes, y=y). "
        "In place of y you can use a large enough sample "
        "of the full training set target to properly "
        "estimate the class frequency distributions. "
        "Pass the resulting weights as the class_weight "
        "parameter.".format(class_weight)
    )


def witness_sgd_classifier_partial_fit_callback_payload(
    X: object,
    y: object,
    alpha: float,
    loss: str,
    learning_rate: str,
    classes: object,
    sample_weight: object,
) -> dict[str, object]:
    """Describe the keyword payload passed from partial_fit to _partial_fit."""
    return {
        "X": X,
        "y": y,
        "alpha": alpha,
        "C": 1.0,
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter": 1,
        "classes": classes,
        "sample_weight": sample_weight,
        "coef_init": None,
        "intercept_init": None,
    }


def witness_sgd_classifier_partial_fit_result(partial_fit_result: object) -> object:
    """Describe the object returned by the delegated _partial_fit callback."""
    return partial_fit_result
