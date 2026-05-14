"""Sklearn SGD classifier fit callback atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sgd_classifier_fit_c_value,
    witness_sgd_classifier_fit_callback_payload,
    witness_sgd_classifier_fit_result,
    witness_sgd_classifier_fit_more_validate_params_result,
)

_FIT_PAYLOAD_KEYS = {
    "X",
    "y",
    "alpha",
    "C",
    "loss",
    "learning_rate",
    "coef_init",
    "intercept_init",
    "sample_weight",
}


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
    coef_init: object,
    intercept_init: object,
    sample_weight: object,
) -> bool:
    return bool(
        set(result) == _FIT_PAYLOAD_KEYS
        and result["X"] is X
        and result["y"] is y
        and result["alpha"] == float(alpha)
        and result["C"] == 1.0
        and result["loss"] == loss
        and result["learning_rate"] == learning_rate
        and result["coef_init"] is coef_init
        and result["intercept_init"] is intercept_init
        and result["sample_weight"] is sample_weight
    )


@register_atom(witness_sgd_classifier_fit_more_validate_params_result)
@icontract.ensure(
    lambda result, validation_result: result is validation_result,
    "_more_validate_params callback result must be preserved",
)
def sgd_classifier_fit_more_validate_params_result(validation_result: object) -> object:
    """Return the unused result from BaseSGDClassifier._more_validate_params."""
    return validation_result


@register_atom(witness_sgd_classifier_fit_c_value)
@icontract.require(lambda alpha: _finite_nonnegative(alpha), "alpha must be finite and nonnegative")
@icontract.ensure(lambda result: result == 1.0, "BaseSGDClassifier.fit must pass fixed C=1.0")
def sgd_classifier_fit_c_value(alpha: float) -> float:
    """Return the fixed C value passed from BaseSGDClassifier.fit to _fit."""
    return 1.0


@register_atom(witness_sgd_classifier_fit_callback_payload)
@icontract.require(lambda X: X is not None, "X must be provided")
@icontract.require(lambda y: y is not None, "y must be provided")
@icontract.require(lambda alpha: _finite_nonnegative(alpha), "alpha must be finite and nonnegative")
@icontract.require(lambda loss: _nonempty_string(loss), "loss must be a nonempty string")
@icontract.require(lambda learning_rate: _nonempty_string(learning_rate), "learning_rate must be a nonempty string")
@icontract.ensure(
    lambda result, X, y, alpha, loss, learning_rate, coef_init, intercept_init, sample_weight: _payload_valid(
        result,
        X,
        y,
        alpha,
        loss,
        learning_rate,
        coef_init,
        intercept_init,
        sample_weight,
    ),
    "fit payload must match the BaseSGDClassifier.fit _fit callback arguments",
)
def sgd_classifier_fit_callback_payload(
    X: object,
    y: object,
    *,
    alpha: float,
    loss: str,
    learning_rate: str,
    coef_init: object = None,
    intercept_init: object = None,
    sample_weight: object = None,
) -> dict[str, object]:
    """Return the _fit callback payload assembled by BaseSGDClassifier.fit."""
    return {
        "X": X,
        "y": y,
        "alpha": float(alpha),
        "C": sgd_classifier_fit_c_value(alpha),
        "loss": loss,
        "learning_rate": learning_rate,
        "coef_init": coef_init,
        "intercept_init": intercept_init,
        "sample_weight": sample_weight,
    }


@register_atom(witness_sgd_classifier_fit_result)
@icontract.require(lambda fit_result: fit_result is not None, "fit_result must be provided")
@icontract.ensure(
    lambda result, fit_result: result is fit_result,
    "BaseSGDClassifier.fit must return the delegated _fit result",
)
def sgd_classifier_fit_result(fit_result: object) -> object:
    """Return the object produced by the delegated BaseSGDClassifier._fit call."""
    return fit_result
