"""Sklearn coordinate-descent enet_path screening-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_dense_screening_args,
    witness_cd_enet_path_do_screening_param,
    witness_cd_enet_path_gram_screening_args,
    witness_cd_enet_path_multitask_screening_args,
    witness_cd_enet_path_sparse_screening_kwarg,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


@register_atom(witness_cd_enet_path_do_screening_param)
@icontract.require(lambda params: isinstance(params, Mapping), "params must be a mapping")
@icontract.require(
    lambda params: "do_screening" not in params or _bool(params["do_screening"]),
    "do_screening must be boolean when supplied",
)
@icontract.ensure(
    lambda result, params: isinstance(result, dict)
    and set(result) == {"do_screening", "remaining_params"}
    and isinstance(result["do_screening"], bool)
    and result["do_screening"] is params.get("do_screening", True)
    and result["remaining_params"]
    == {key: value for key, value in params.items() if key != "do_screening"},
    "do_screening pop/default result must match enet_path",
)
def cd_enet_path_do_screening_param(params: Mapping[str, object]) -> dict[str, object]:
    """Return do_screening and params remaining after sklearn's pop shell."""
    remaining_params = dict(params)
    do_screening = remaining_params.pop("do_screening", True)
    return {"do_screening": do_screening, "remaining_params": remaining_params}


@register_atom(witness_cd_enet_path_sparse_screening_kwarg)
@icontract.require(lambda do_screening: _bool(do_screening), "do_screening must be boolean")
@icontract.ensure(
    lambda result, do_screening: result == {"do_screening": do_screening},
    "sparse screening payload must be the do_screening keyword",
)
def cd_enet_path_sparse_screening_kwarg(do_screening: bool) -> dict[str, bool]:
    """Return the do_screening keyword payload for the sparse solver."""
    return {"do_screening": do_screening}


@register_atom(witness_cd_enet_path_multitask_screening_args)
@icontract.require(lambda base_args: _sequence(base_args), "base_args must be a non-string sequence")
@icontract.require(lambda do_screening: _bool(do_screening), "do_screening must be boolean")
@icontract.ensure(
    lambda result, base_args, do_screening: isinstance(result, tuple)
    and result == tuple(base_args) + (do_screening,),
    "multitask solver args must append do_screening",
)
def cd_enet_path_multitask_screening_args(
    base_args: Sequence[object], do_screening: bool
) -> tuple[object, ...]:
    """Append do_screening to the multitask solver positional payload."""
    return tuple(base_args) + (do_screening,)


@register_atom(witness_cd_enet_path_gram_screening_args)
@icontract.require(lambda base_args: _sequence(base_args), "base_args must be a non-string sequence")
@icontract.require(lambda do_screening: _bool(do_screening), "do_screening must be boolean")
@icontract.ensure(
    lambda result, base_args, do_screening: isinstance(result, tuple)
    and result == tuple(base_args) + (do_screening,),
    "Gram solver args must append do_screening",
)
def cd_enet_path_gram_screening_args(
    base_args: Sequence[object], do_screening: bool
) -> tuple[object, ...]:
    """Append do_screening to the Gram solver positional payload."""
    return tuple(base_args) + (do_screening,)


@register_atom(witness_cd_enet_path_dense_screening_args)
@icontract.require(lambda base_args: _sequence(base_args), "base_args must be a non-string sequence")
@icontract.require(lambda do_screening: _bool(do_screening), "do_screening must be boolean")
@icontract.ensure(
    lambda result, base_args, do_screening: isinstance(result, tuple)
    and result == tuple(base_args) + (do_screening,),
    "dense solver args must append do_screening",
)
def cd_enet_path_dense_screening_args(
    base_args: Sequence[object], do_screening: bool
) -> tuple[object, ...]:
    """Append do_screening to the dense solver positional payload."""
    return tuple(base_args) + (do_screening,)
