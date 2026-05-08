"""Ghost witnesses for sklearn coordinate-descent enet_path pre-fit/grid callbacks."""

from __future__ import annotations


def witness_cd_enet_path_prefit_18_result_unpack(prefit_result: object) -> object:
    """Describe the _pre_fit result fields retained by enet_path."""
    return prefit_result


def witness_cd_enet_path_alpha_grid_18_result(generated_alphas: object) -> object:
    """Describe the automatic alpha grid returned by _alpha_grid."""
    return generated_alphas
