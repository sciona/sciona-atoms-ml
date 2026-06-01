"""Sklearn RANSAC public predict/score callback atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_ransac_public_nonrouting_params,
    witness_ransac_public_predict_callback_payload,
    witness_ransac_public_score_callback_payload,
    witness_ransac_public_validation_kwargs,
)

_METHOD_NAMES = {"predict", "score"}
_VALIDATION_KWARGS = {
    "ensure_all_finite": False,
    "accept_sparse": True,
    "reset": False,
}
_PREDICT_PAYLOAD_KEYS = {"estimator", "X", "predict_params"}
_SCORE_PAYLOAD_KEYS = {"estimator", "X", "y", "score_params"}


def _method_valid(method_name: object) -> bool:
    return bool(method_name in _METHOD_NAMES)


def _mapping_valid(value: object) -> bool:
    return bool(isinstance(value, Mapping))


def _validation_kwargs_valid(result: dict[str, object]) -> bool:
    return bool(result == _VALIDATION_KWARGS)


def _nonrouting_params_valid(result: dict[str, object]) -> bool:
    return bool(isinstance(result, dict) and result == {})


def _predict_payload_valid(result: dict[str, object], estimator: object, X: object, predict_params: Mapping[str, object]) -> bool:
    return bool(
        set(result) == _PREDICT_PAYLOAD_KEYS
        and result["estimator"] is estimator
        and result["X"] is X
        and result["predict_params"] is predict_params
    )


def _score_payload_valid(result: dict[str, object], estimator: object, X: object, y: object, score_params: Mapping[str, object]) -> bool:
    return bool(
        set(result) == _SCORE_PAYLOAD_KEYS
        and result["estimator"] is estimator
        and result["X"] is X
        and result["y"] is y
        and result["score_params"] is score_params
    )


@register_atom(witness_ransac_public_validation_kwargs)
@icontract.require(lambda method_name: _method_valid(method_name), "method_name must be predict or score")
@icontract.ensure(lambda result: _validation_kwargs_valid(result), "validation kwargs must match RANSAC public API source")
def ransac_public_validation_kwargs(method_name: str) -> dict[str, object]:
    """Return fixed validate_data kwargs for RANSACRegressor.predict/score."""
    del method_name
    return dict(_VALIDATION_KWARGS)


@register_atom(witness_ransac_public_nonrouting_params)
@icontract.require(lambda method_name: _method_valid(method_name), "method_name must be predict or score")
@icontract.ensure(lambda result: _nonrouting_params_valid(result), "non-routing params must be an empty mapping")
def ransac_public_nonrouting_params(method_name: str) -> dict[str, object]:
    """Return the non-routing predict/score parameter fallback."""
    del method_name
    return {}


@register_atom(witness_ransac_public_predict_callback_payload)
@icontract.require(lambda predict_params: _mapping_valid(predict_params), "predict_params must be a mapping")
@icontract.ensure(
    lambda result, estimator, X, predict_params: _predict_payload_valid(result, estimator, X, predict_params),
    "predict callback payload must preserve estimator, X, and routed params",
)
def ransac_public_predict_callback_payload(
    estimator: object,
    X: object,
    predict_params: Mapping[str, object],
) -> dict[str, object]:
    """Return payload for RANSACRegressor.estimator_.predict."""
    return {"estimator": estimator, "X": X, "predict_params": predict_params}


@register_atom(witness_ransac_public_score_callback_payload)
@icontract.require(lambda score_params: _mapping_valid(score_params), "score_params must be a mapping")
@icontract.ensure(
    lambda result, estimator, X, y, score_params: _score_payload_valid(result, estimator, X, y, score_params),
    "score callback payload must preserve estimator, X, y, and routed params",
)
def ransac_public_score_callback_payload(
    estimator: object,
    X: object,
    y: object,
    score_params: Mapping[str, object],
) -> dict[str, object]:
    """Return payload for RANSACRegressor.estimator_.score."""
    return {"estimator": estimator, "X": X, "y": y, "score_params": score_params}
