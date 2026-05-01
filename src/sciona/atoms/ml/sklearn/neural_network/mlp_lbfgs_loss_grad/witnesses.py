"""Ghost witnesses for MLP LBFGS loss/gradient helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray

CoefSlice = tuple[int, int, tuple[int, int]]
InterceptSlice = tuple[int, int]


def witness_mlp_lbfgs_unpack_parameters(
    packed_parameters: NDArray[float],
    coef_indptr: tuple[CoefSlice, ...],
    intercept_indptr: tuple[InterceptSlice, ...],
) -> tuple[tuple[NDArray[float], ...], tuple[NDArray[float], ...]]:
    """Describe the unpacked coefficient and intercept blocks derived from a flat LBFGS vector."""
    del packed_parameters
    del coef_indptr
    del intercept_indptr
    raise NotImplementedError


def witness_mlp_lbfgs_loss_grad(
    packed_parameters: NDArray[float],
    coef_indptr: tuple[CoefSlice, ...],
    intercept_indptr: tuple[InterceptSlice, ...],
    X: NDArray[float],
    y: NDArray[float],
    hidden_activation: str,
    output_activation: str,
    loss_name: str,
    alpha: float = 0.0,
) -> tuple[float, NDArray[float]]:
    """Describe the sklearn-style LBFGS loss scalar and packed gradient vector."""
    del packed_parameters
    del coef_indptr
    del intercept_indptr
    del X
    del y
    del hidden_activation
    del output_activation
    del loss_name
    del alpha
    raise NotImplementedError
