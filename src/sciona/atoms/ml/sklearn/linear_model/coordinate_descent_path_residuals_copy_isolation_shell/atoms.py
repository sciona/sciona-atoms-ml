"""Sklearn coordinate-descent path-residual copy-isolation atoms."""

from __future__ import annotations

from collections.abc import Mapping

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import witness_cd_path_residuals_path_params_copy


def _mapping_valid(value: object) -> bool:
    return isinstance(value, Mapping)


def _shallow_copy_valid(result: dict[object, object], path_params: Mapping[object, object]) -> bool:
    return bool(
        isinstance(result, dict)
        and result is not path_params
        and set(result.keys()) == set(path_params.keys())
        and all(result[key] is path_params[key] for key in result)
    )


@register_atom(witness_cd_path_residuals_path_params_copy)
@icontract.require(lambda path_params: _mapping_valid(path_params), "path_params must be a mapping")
@icontract.ensure(
    lambda result, path_params: _shallow_copy_valid(result, path_params),
    "path_params copy must be a distinct shallow dict with identical value objects",
)
def cd_path_residuals_path_params_copy(path_params: Mapping[object, object]) -> dict[object, object]:
    """Return the shallow copy made before _path_residuals mutates path_params."""
    return dict(path_params)
