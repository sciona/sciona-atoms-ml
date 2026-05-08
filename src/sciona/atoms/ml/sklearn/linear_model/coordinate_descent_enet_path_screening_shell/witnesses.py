"""Ghost witnesses for sklearn coordinate-descent enet_path screening atoms."""

from __future__ import annotations


def witness_cd_enet_path_do_screening_param(params: object) -> object:
    """Describe the do_screening pop/default shell in enet_path."""
    return params


def witness_cd_enet_path_sparse_screening_kwarg(do_screening: object) -> object:
    """Describe the sparse solver do_screening keyword handoff."""
    return do_screening


def witness_cd_enet_path_multitask_screening_args(base_args: object, do_screening: object) -> object:
    """Describe the multitask solver do_screening positional tail."""
    return base_args, do_screening


def witness_cd_enet_path_gram_screening_args(base_args: object, do_screening: object) -> object:
    """Describe the Gram solver do_screening positional tail."""
    return base_args, do_screening


def witness_cd_enet_path_dense_screening_args(base_args: object, do_screening: object) -> object:
    """Describe the dense solver do_screening positional tail."""
    return base_args, do_screening
