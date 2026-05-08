"""Ghost witnesses for sklearn enet_path solver payload atoms."""

from __future__ import annotations


def witness_cd_enet_path_sparse_solver_kwargs(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X_data: object,
    X_indices: object,
    X_indptr: object,
    y: object,
    sample_weight: object,
    X_sparse_scaling: object,
    max_iter: object,
    tol: object,
    rng: object,
    random: object,
    positive: object,
) -> object:
    """Describe sparse solver keyword payload."""
    return (
        coef,
        l1_reg,
        l2_reg,
        X_data,
        X_indices,
        X_indptr,
        y,
        sample_weight,
        X_sparse_scaling,
        max_iter,
        tol,
        rng,
        random,
        positive,
    )


def witness_cd_enet_path_multitask_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: object,
    tol: object,
    rng: object,
    random: object,
) -> object:
    """Describe multitask solver positional payload."""
    return coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random


def witness_cd_enet_path_gram_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    precompute: object,
    Xy: object,
    y: object,
    max_iter: object,
    tol: object,
    rng: object,
    random: object,
    positive: object,
) -> object:
    """Describe Gram solver positional payload."""
    return coef, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, rng, random, positive


def witness_cd_enet_path_dense_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: object,
    tol: object,
    rng: object,
    random: object,
    positive: object,
) -> object:
    """Describe dense solver positional payload."""
    return coef, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
