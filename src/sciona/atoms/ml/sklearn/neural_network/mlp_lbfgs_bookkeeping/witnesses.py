"""Ghost witnesses for MLP LBFGS bookkeeping helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray

CoefSlice = tuple[int, int, tuple[int, int]]
InterceptSlice = tuple[int, int]


def witness_mlp_lbfgs_coef_indptr(
    layer_units: tuple[int, ...],
) -> tuple[CoefSlice, ...]:
    """Describe the coefficient slice layout used before MLP LBFGS optimization."""
    del layer_units
    return ((0, 1, (1, 1)),)


def witness_mlp_lbfgs_intercept_indptr(
    layer_units: tuple[int, ...],
    coef_indptr: tuple[CoefSlice, ...],
) -> tuple[InterceptSlice, ...]:
    """Describe the intercept slice layout used before MLP LBFGS optimization."""
    del layer_units
    del coef_indptr
    return ((1, 2),)


def witness_mlp_lbfgs_pack_parameters(
    coefs: tuple[NDArray[float], ...],
    intercepts: tuple[NDArray[float], ...],
) -> NDArray[float]:
    """Describe the packed MLP LBFGS parameter vector."""
    del coefs
    del intercepts
    raise NotImplementedError


def witness_mlp_lbfgs_iprint(
    verbose: bool | int,
) -> int:
    """Describe the LBFGS iprint option derived from verbose."""
    del verbose
    return 0
