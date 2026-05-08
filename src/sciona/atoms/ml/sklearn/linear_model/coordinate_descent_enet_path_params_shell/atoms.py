"""Sklearn coordinate-descent enet_path params atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import witness_cd_enet_path_popped_params


_PARAM_DEFAULTS = {
    "X_offset": None,
    "X_scale": None,
    "sample_weight": None,
    "tol": 1e-4,
    "max_iter": 1000,
    "random_state": None,
    "selection": "cyclic",
}


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _popped_params_valid(result: object, params: Mapping[str, object]) -> bool:
    if not isinstance(result, dict):
        return False
    expected_keys = {
        "X_offset_param",
        "X_scale_param",
        "sample_weight",
        "tol",
        "max_iter",
        "random_state",
        "selection",
        "remaining_params",
    }
    if set(result) != expected_keys or not isinstance(result["remaining_params"], dict):
        return False
    remaining = {key: value for key, value in params.items() if key not in _PARAM_DEFAULTS}
    return bool(
        result["X_offset_param"] is params.get("X_offset", None)
        and result["X_scale_param"] is params.get("X_scale", None)
        and result["sample_weight"] is params.get("sample_weight", None)
        and result["tol"] == params.get("tol", 1e-4)
        and result["max_iter"] == params.get("max_iter", 1000)
        and result["random_state"] is params.get("random_state", None)
        and result["selection"] == params.get("selection", "cyclic")
        and result["remaining_params"] == remaining
    )


@register_atom(witness_cd_enet_path_popped_params)
@icontract.require(lambda params: isinstance(params, Mapping), "params must be a mapping")
@icontract.require(
    lambda params: "max_iter" not in params or _positive_int(params["max_iter"]),
    "max_iter must be positive when supplied",
)
@icontract.require(
    lambda params: "selection" not in params or isinstance(params["selection"], str),
    "selection must be a string when supplied",
)
@icontract.ensure(
    lambda result, params: _popped_params_valid(result, params),
    "popped params must match enet_path defaults and leftover-param mapping",
)
def cd_enet_path_popped_params(params: Mapping[str, object]) -> dict[str, object]:
    """Return enet_path solver-only params after sklearn's pop/default shell."""
    remaining_params = dict(params)
    X_offset_param = remaining_params.pop("X_offset", None)
    X_scale_param = remaining_params.pop("X_scale", None)
    sample_weight = remaining_params.pop("sample_weight", None)
    tol = remaining_params.pop("tol", 1e-4)
    max_iter = remaining_params.pop("max_iter", 1000)
    random_state = remaining_params.pop("random_state", None)
    selection = remaining_params.pop("selection", "cyclic")
    return {
        "X_offset_param": X_offset_param,
        "X_scale_param": X_scale_param,
        "sample_weight": sample_weight,
        "tol": tol,
        "max_iter": max_iter,
        "random_state": random_state,
        "selection": selection,
        "remaining_params": remaining_params,
    }
