"""Ghost witnesses for sklearn coordinate-descent enet_path params atoms."""

from __future__ import annotations


def witness_cd_enet_path_popped_params(params: object) -> object:
    """Describe the solver-only params popped at the start of enet_path."""
    return params
