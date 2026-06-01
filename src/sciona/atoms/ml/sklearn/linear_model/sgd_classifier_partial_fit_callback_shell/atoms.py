"""Sklearn SGD classifier partial-fit callback atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sgd_classifier_partial_fit_balanced_class_weight_error,
    witness_sgd_classifier_partial_fit_callback_payload,
    witness_sgd_classifier_partial_fit_first_call,
    witness_sgd_classifier_partial_fit_result,
    witness_sgd_classifier_partial_fit_validate_params_kwargs,
)

_PARTIAL_FIT_PAYLOAD_KEYS = {
    "X",
    "y",
    "alpha",
    "C",
    "loss",
    "learning_rate",
    "max_iter",
    "classes",
    "sample_weight",
    "coef_init",
    "intercept_init",
}


def _bool_value(value: object) -> bool:
    return isinstance(value, bool)


def _finite_nonnegative(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _nonempty_string(value: str) -> bool:
    return bool(isinstance(value, str) and value)


def _payload_valid(
    result: dict[str, object],
    X: object,
    y: object,
    alpha: float,
    loss: str,
    learning_rate: str,
    classes: object,
    sample_weight: object,
) -> bool:
    return bool(
        set(result) == _PARTIAL_FIT_PAYLOAD_KEYS
        and result["X"] is X
        and result["y"] is y
        and result["alpha"] == float(alpha)
        and result["C"] == 1.0
        and result["loss"] == loss
        and result["learning_rate"] == learning_rate
        and result["max_iter"] == 1
        and result["classes"] is classes
        and result["sample_weight"] is sample_weight
        and result["coef_init"] is None
        and result["intercept_init"] is None
    )


@register_atom(witness_sgd_classifier_partial_fit_first_call)
@icontract.require(lambda has_classes: _bool_value(has_classes), "has_classes must be boolean")
@icontract.ensure(lambda result, has_classes: result is (not has_classes), "first-call predicate must match absence of classes_")
def sgd_classifier_partial_fit_first_call(has_classes: bool) -> bool:
    """Return whether partial_fit is running before classes_ exists."""
    return not has_classes


@register_atom(witness_sgd_classifier_partial_fit_validate_params_kwargs)
@icontract.require(lambda first_call: _bool_value(first_call), "first_call must be boolean")
@icontract.ensure(
    lambda result, first_call: result == ({"for_partial_fit": True} if first_call else {}),
    "partial_fit validation kwargs must request partial-fit mode only on the first call",
)
def sgd_classifier_partial_fit_validate_params_kwargs(first_call: bool) -> dict[str, object]:
    """Return kwargs for BaseSGDClassifier._more_validate_params in partial_fit."""
    if not first_call:
        return {}
    return {"for_partial_fit": True}


@register_atom(witness_sgd_classifier_partial_fit_balanced_class_weight_error)
@icontract.require(lambda class_weight: class_weight == "balanced", "class_weight must be balanced")
@icontract.ensure(
    lambda result, class_weight: result == witness_sgd_classifier_partial_fit_balanced_class_weight_error(class_weight),
    "balanced class-weight error message must match sklearn source",
)
def sgd_classifier_partial_fit_balanced_class_weight_error(class_weight: str) -> str:
    """Return the balanced class_weight error message used by partial_fit."""
    return witness_sgd_classifier_partial_fit_balanced_class_weight_error(class_weight)


@register_atom(witness_sgd_classifier_partial_fit_callback_payload)
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda y: y is not None, "y must be provided")
@icontract.require(lambda alpha: _finite_nonnegative(alpha), "alpha must be finite and nonnegative")
@icontract.require(lambda loss: _nonempty_string(loss), "loss must be a nonempty string")
@icontract.require(lambda learning_rate: _nonempty_string(learning_rate), "learning_rate must be a nonempty string")
@icontract.ensure(
    lambda result, X, y, alpha, loss, learning_rate, classes, sample_weight: _payload_valid(
        result,
        X,
        y,
        alpha,
        loss,
        learning_rate,
        classes,
        sample_weight,
    ),
    "partial_fit payload must match the BaseSGDClassifier._partial_fit callback arguments",
)
def sgd_classifier_partial_fit_callback_payload(
    X: object,
    y: object,
    *,
    alpha: float,
    loss: str,
    learning_rate: str,
    classes: object,
    sample_weight: object = None,
) -> dict[str, object]:
    """Return the _partial_fit callback payload assembled by partial_fit."""
    return {
        "X": X,
        "y": y,
        "alpha": float(alpha),
        "C": 1.0,
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter": 1,
        "classes": classes,
        "sample_weight": sample_weight,
        "coef_init": None,
        "intercept_init": None,
    }


@register_atom(witness_sgd_classifier_partial_fit_result)
@icontract.require(lambda partial_fit_result: partial_fit_result is not None, "partial_fit_result must be provided")
@icontract.ensure(
    lambda result, partial_fit_result: result is partial_fit_result,
    "BaseSGDClassifier.partial_fit must return the delegated _partial_fit result",
)
def sgd_classifier_partial_fit_result(partial_fit_result: object) -> object:
    """Return the object produced by the delegated BaseSGDClassifier._partial_fit call."""
    return partial_fit_result
