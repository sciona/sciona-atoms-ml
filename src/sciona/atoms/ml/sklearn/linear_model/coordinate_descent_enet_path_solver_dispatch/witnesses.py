"""Ghost witnesses for sklearn coordinate-descent enet_path solver-dispatch atoms."""

from __future__ import annotations


def witness_cd_enet_path_gram_validation_required(
    check_input: object, precompute_is_array: object
) -> object:
    """Describe the Gram check_array gate in enet_path."""
    return check_input, precompute_is_array


def witness_cd_enet_path_use_sparse_solver(
    multi_output: object, x_is_sparse: object
) -> object:
    """Describe the sparse solver dispatch predicate in enet_path."""
    return multi_output, x_is_sparse


def witness_cd_enet_path_use_multi_task_solver(
    multi_output: object, use_sparse_solver: object
) -> object:
    """Describe the multitask solver dispatch predicate in enet_path."""
    return multi_output, use_sparse_solver


def witness_cd_enet_path_use_gram_solver(
    use_sparse_solver: object, multi_output: object, precompute_is_array: object
) -> object:
    """Describe the Gram solver dispatch predicate in enet_path."""
    return use_sparse_solver, multi_output, precompute_is_array


def witness_cd_enet_path_use_dense_solver(
    use_sparse_solver: object, use_multi_task_solver: object, precompute_is_false: object
) -> object:
    """Describe the dense solver dispatch predicate in enet_path."""
    return use_sparse_solver, use_multi_task_solver, precompute_is_false


def witness_cd_enet_path_invalid_precompute_message(precompute: object) -> object:
    """Describe the invalid precompute ValueError message in enet_path."""
    return precompute
