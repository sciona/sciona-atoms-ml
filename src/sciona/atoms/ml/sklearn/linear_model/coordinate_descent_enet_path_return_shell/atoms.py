"""Sklearn coordinate-descent enet_path return-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_result_tuple,
    witness_cd_enet_path_return_arity,
)


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


@register_atom(witness_cd_enet_path_return_arity)
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.ensure(
    lambda result, return_n_iter: result == (4 if return_n_iter else 3),
    "return arity must match sklearn's return_n_iter branch",
)
def cd_enet_path_return_arity(return_n_iter: bool) -> int:
    """Return the final enet_path tuple arity selected by return_n_iter."""
    return 4 if return_n_iter else 3


@register_atom(witness_cd_enet_path_result_tuple)
@icontract.require(lambda n_iters: _sequence(n_iters), "n_iters must be a non-string sequence")
@icontract.require(lambda return_n_iter: isinstance(return_n_iter, bool), "return_n_iter must be boolean")
@icontract.ensure(
    lambda result, alphas, coefs, dual_gaps, n_iters, return_n_iter: isinstance(result, tuple)
    and len(result) == cd_enet_path_return_arity(return_n_iter)
    and result[0] is alphas
    and result[1] is coefs
    and result[2] is dual_gaps
    and ((not return_n_iter) or result[3] is n_iters),
    "return tuple must match sklearn's enet_path final return branch",
)
def cd_enet_path_result_tuple(
    alphas: object,
    coefs: object,
    dual_gaps: object,
    n_iters: Sequence[object],
    return_n_iter: bool,
) -> tuple[object, ...]:
    """Return the final public enet_path result tuple."""
    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps
