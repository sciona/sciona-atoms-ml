"""Ghost witnesses for MLP fit buffer-setup helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_mlp_fit_targets_2d(
    y: NDArray[float],
) -> NDArray[float]:
    """Describe the 2D target matrix consumed by sklearn's MLP fit loop."""
    del y
    raise NotImplementedError


def witness_mlp_fit_layer_units(
    n_features: int,
    hidden_layer_sizes: tuple[int, ...],
    n_outputs: int,
) -> tuple[int, ...]:
    """Describe the layer-width sequence used to allocate MLP fit buffers."""
    del n_features
    del hidden_layer_sizes
    del n_outputs
    return (1, 1)


def witness_mlp_fit_coef_gradient_buffers(
    layer_units: tuple[int, ...],
    dtype_name: str,
) -> tuple[NDArray[float], ...]:
    """Describe the coefficient-gradient buffers allocated before MLP solver execution."""
    del layer_units
    del dtype_name
    raise NotImplementedError


def witness_mlp_fit_intercept_gradient_buffers(
    layer_units: tuple[int, ...],
    dtype_name: str,
) -> tuple[NDArray[float], ...]:
    """Describe the intercept-gradient buffers allocated before MLP solver execution."""
    del layer_units
    del dtype_name
    raise NotImplementedError
