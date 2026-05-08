"""Sklearn coordinate-descent enet_path pre-fit/grid callback atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_alpha_grid_18_result,
    witness_cd_enet_path_prefit_18_result_unpack,
)


def _prefit_result(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 7


@register_atom(witness_cd_enet_path_prefit_18_result_unpack)
@icontract.require(
    lambda prefit_result: _prefit_result(prefit_result),
    "prefit_result must be the seven-item _pre_fit return",
)
@icontract.ensure(
    lambda result, prefit_result: isinstance(result, dict)
    and set(result) == {"X", "y", "precompute", "Xy"}
    and result["X"] is prefit_result[0]
    and result["y"] is prefit_result[1]
    and result["precompute"] is prefit_result[5]
    and result["Xy"] is prefit_result[6],
    "enet_path must retain X, y, precompute, and Xy from _pre_fit",
)
def cd_enet_path_prefit_18_result_unpack(prefit_result: Sequence[object]) -> dict[str, object]:
    """Return the _pre_fit output fields assigned by sklearn 1.8 enet_path."""
    return {
        "X": prefit_result[0],
        "y": prefit_result[1],
        "precompute": prefit_result[5],
        "Xy": prefit_result[6],
    }


@register_atom(witness_cd_enet_path_alpha_grid_18_result)
@icontract.require(lambda generated_alphas: generated_alphas is not None, "generated_alphas must be present")
@icontract.ensure(
    lambda result, generated_alphas: result is generated_alphas,
    "enet_path must use the generated alpha grid unchanged before later bookkeeping",
)
def cd_enet_path_alpha_grid_18_result(generated_alphas: object) -> object:
    """Return the automatic alpha grid assigned by sklearn 1.8 enet_path."""
    return generated_alphas
