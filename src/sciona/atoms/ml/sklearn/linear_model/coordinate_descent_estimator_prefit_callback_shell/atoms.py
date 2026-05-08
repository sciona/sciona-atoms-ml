"""Sklearn coordinate-descent estimator pre-fit callback atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_prefit_result_unpack,
    witness_cd_estimator_prefit_xy_payload,
    witness_cd_estimator_set_order_result_unpack,
)


def _sequence_of_length(value: object, length: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == length
    )


@register_atom(witness_cd_estimator_prefit_result_unpack)
@icontract.require(
    lambda prefit_result: _sequence_of_length(prefit_result, 7),
    "prefit_result must be the seven-item _pre_fit return",
)
@icontract.ensure(
    lambda result, prefit_result: isinstance(result, dict)
    and set(result)
    == {"X", "y", "X_offset", "y_offset", "X_scale", "precompute", "Xy"}
    and result["X"] is prefit_result[0]
    and result["y"] is prefit_result[1]
    and result["X_offset"] is prefit_result[2]
    and result["y_offset"] is prefit_result[3]
    and result["X_scale"] is prefit_result[4]
    and result["precompute"] is prefit_result[5]
    and result["Xy"] is prefit_result[6],
    "ElasticNet.fit must retain all assigned fields from _pre_fit",
)
def cd_estimator_prefit_result_unpack(prefit_result: Sequence[object]) -> dict[str, object]:
    """Return the _pre_fit output fields assigned by ElasticNet.fit."""
    return {
        "X": prefit_result[0],
        "y": prefit_result[1],
        "X_offset": prefit_result[2],
        "y_offset": prefit_result[3],
        "X_scale": prefit_result[4],
        "precompute": prefit_result[5],
        "Xy": prefit_result[6],
    }


@register_atom(witness_cd_estimator_set_order_result_unpack)
@icontract.require(
    lambda order_result: _sequence_of_length(order_result, 2),
    "order_result must be the two-item _set_order return",
)
@icontract.ensure(
    lambda result, order_result: isinstance(result, dict)
    and set(result) == {"X", "y"}
    and result["X"] is order_result[0]
    and result["y"] is order_result[1],
    "ElasticNet.fit must retain X and y from _set_order",
)
def cd_estimator_set_order_result_unpack(order_result: Sequence[object]) -> dict[str, object]:
    """Return the _set_order output fields assigned by ElasticNet.fit."""
    return {"X": order_result[0], "y": order_result[1]}


@register_atom(witness_cd_estimator_prefit_xy_payload)
@icontract.ensure(
    lambda result, X, y, X_offset, y_offset, X_scale, precompute, Xy: isinstance(result, dict)
    and set(result)
    == {"X", "y", "X_offset", "y_offset", "X_scale", "precompute", "Xy"}
    and result["X"] is X
    and result["y"] is y
    and result["X_offset"] is X_offset
    and result["y_offset"] is y_offset
    and result["X_scale"] is X_scale
    and result["precompute"] is precompute
    and result["Xy"] is Xy,
    "post-prefit payload must preserve the objects consumed by later fit stages",
)
def cd_estimator_prefit_xy_payload(
    X: object,
    y: object,
    X_offset: object,
    y_offset: object,
    X_scale: object,
    precompute: object,
    Xy: object,
) -> dict[str, object]:
    """Return the named post-_pre_fit payload consumed after ordering normalization."""
    return {
        "X": X,
        "y": y,
        "X_offset": X_offset,
        "y_offset": y_offset,
        "X_scale": X_scale,
        "precompute": precompute,
        "Xy": Xy,
    }
