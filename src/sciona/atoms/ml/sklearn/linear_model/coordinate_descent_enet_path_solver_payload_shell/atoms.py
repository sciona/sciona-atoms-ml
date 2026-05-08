"""Sklearn coordinate-descent enet_path solver payload atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_dense_solver_args,
    witness_cd_enet_path_gram_solver_args,
    witness_cd_enet_path_multitask_solver_args,
    witness_cd_enet_path_sparse_solver_kwargs,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_enet_path_sparse_solver_kwargs)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda random: _bool(random), "random must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.ensure(
    lambda result, coef, l1_reg, l2_reg, X_data, X_indices, X_indptr, y, sample_weight, X_sparse_scaling, max_iter, tol, rng, random, positive: isinstance(result, dict)
    and set(result)
    == {
        "w",
        "alpha",
        "beta",
        "X_data",
        "X_indices",
        "X_indptr",
        "y",
        "sample_weight",
        "X_mean",
        "max_iter",
        "tol",
        "rng",
        "random",
        "positive",
    }
    and result["w"] is coef
    and result["alpha"] is l1_reg
    and result["beta"] is l2_reg
    and result["X_data"] is X_data
    and result["X_indices"] is X_indices
    and result["X_indptr"] is X_indptr
    and result["y"] is y
    and result["sample_weight"] is sample_weight
    and result["X_mean"] is X_sparse_scaling
    and result["max_iter"] is max_iter
    and result["tol"] is tol
    and result["rng"] is rng
    and result["random"] is random
    and result["positive"] is positive,
    "sparse solver kwargs must match enet_path call names and values",
)
def cd_enet_path_sparse_solver_kwargs(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X_data: object,
    X_indices: object,
    X_indptr: object,
    y: object,
    sample_weight: object,
    X_sparse_scaling: object,
    max_iter: int,
    tol: object,
    rng: object,
    random: bool,
    positive: bool,
) -> dict[str, object]:
    """Return keyword payload for sparse_enet_coordinate_descent."""
    return {
        "w": coef,
        "alpha": l1_reg,
        "beta": l2_reg,
        "X_data": X_data,
        "X_indices": X_indices,
        "X_indptr": X_indptr,
        "y": y,
        "sample_weight": sample_weight,
        "X_mean": X_sparse_scaling,
        "max_iter": max_iter,
        "tol": tol,
        "rng": rng,
        "random": random,
        "positive": positive,
    }


@register_atom(witness_cd_enet_path_multitask_solver_args)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda random: _bool(random), "random must be boolean")
@icontract.ensure(
    lambda result, coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random: isinstance(result, tuple)
    and len(result) == 9
    and result[0] is coef
    and result[1] is l1_reg
    and result[2] is l2_reg
    and result[3] is X
    and result[4] is y
    and result[5] is max_iter
    and result[6] is tol
    and result[7] is rng
    and result[8] is random,
    "multitask solver args must match enet_path call order",
)
def cd_enet_path_multitask_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: int,
    tol: object,
    rng: object,
    random: bool,
) -> tuple[object, object, object, object, object, int, object, object, bool]:
    """Return positional payload for enet_coordinate_descent_multi_task."""
    return (coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random)


@register_atom(witness_cd_enet_path_gram_solver_args)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda random: _bool(random), "random must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.ensure(
    lambda result, coef, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, rng, random, positive: isinstance(result, tuple)
    and len(result) == 11
    and result[0] is coef
    and result[1] is l1_reg
    and result[2] is l2_reg
    and result[3] is precompute
    and result[4] is Xy
    and result[5] is y
    and result[6] is max_iter
    and result[7] is tol
    and result[8] is rng
    and result[9] is random
    and result[10] is positive,
    "Gram solver args must match enet_path call order",
)
def cd_enet_path_gram_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    precompute: object,
    Xy: object,
    y: object,
    max_iter: int,
    tol: object,
    rng: object,
    random: bool,
    positive: bool,
) -> tuple[object, object, object, object, object, object, int, object, object, bool, bool]:
    """Return positional payload for enet_coordinate_descent_gram."""
    return (coef, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, rng, random, positive)


@register_atom(witness_cd_enet_path_dense_solver_args)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda random: _bool(random), "random must be boolean")
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.ensure(
    lambda result, coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive: isinstance(result, tuple)
    and len(result) == 10
    and result[0] is coef
    and result[1] is l1_reg
    and result[2] is l2_reg
    and result[3] is X
    and result[4] is y
    and result[5] is max_iter
    and result[6] is tol
    and result[7] is rng
    and result[8] is random
    and result[9] is positive,
    "dense solver args must match enet_path call order",
)
def cd_enet_path_dense_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: int,
    tol: object,
    rng: object,
    random: bool,
    positive: bool,
) -> tuple[object, object, object, object, object, int, object, object, bool, bool]:
    """Return positional payload for enet_coordinate_descent."""
    return (coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive)
