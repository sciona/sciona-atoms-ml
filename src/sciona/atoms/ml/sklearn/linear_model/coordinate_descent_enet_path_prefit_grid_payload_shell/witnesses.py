"""Ghost witnesses for sklearn coordinate-descent enet_path pre-fit/grid payload atoms."""

from __future__ import annotations


def witness_cd_enet_path_prefit_18_kwargs(path_helper: object) -> object:
    """Describe the fixed _pre_fit kwargs in sklearn 1.8 enet_path."""
    return path_helper


def witness_cd_enet_path_alpha_grid_18_kwargs(
    path_helper: object,
    Xy: object,
    l1_ratio: object,
    eps: object,
    n_alphas: object,
) -> object:
    """Describe the _alpha_grid keyword payload assembled by enet_path."""
    return path_helper, Xy, l1_ratio, eps, n_alphas
