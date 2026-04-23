"""Ghost witnesses for sklearn MLP initialization helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_mlp_output_activation_name(
    *,
    is_classifier: bool,
    loss_name: str = "squared_error",
    label_binarizer_type: str | None = None,
) -> str:
    """Describe the output activation name chosen for an MLP head."""
    del is_classifier, loss_name, label_binarizer_type
    return "identity"


def witness_mlp_glorot_init_bound(
    fan_in: int,
    fan_out: int,
    *,
    activation: str,
) -> float:
    """Describe one scalar initialization bound for a layer."""
    del activation
    if fan_in < 1 or fan_out < 1:
        raise ValueError("fan_in and fan_out must be positive")
    return 0.0


def witness_mlp_init_layer_parameters(
    fan_in: int,
    fan_out: int,
    *,
    activation: str,
    random_state: int | None = None,
    dtype_name: str = "float64",
) -> tuple[AbstractArray, AbstractArray]:
    """Describe one initialized coefficient matrix and intercept vector."""
    del activation, random_state, dtype_name
    if fan_in < 1 or fan_out < 1:
        raise ValueError("fan_in and fan_out must be positive")
    return (
        AbstractArray(shape=(fan_in, fan_out), dtype="float64"),
        AbstractArray(shape=(fan_out,), dtype="float64"),
    )


def witness_mlp_initialize_parameters(
    layer_units: tuple[int, ...],
    *,
    activation: str,
    random_state: int | None = None,
    dtype_name: str = "float64",
) -> tuple[
    tuple[AbstractArray, ...],
    tuple[AbstractArray, ...],
    tuple[AbstractArray, ...],
    tuple[AbstractArray, ...],
]:
    """Describe initialized parameter tuples and their best-copy mirrors."""
    del activation, random_state, dtype_name
    if len(layer_units) < 2:
        raise ValueError("layer_units must include input and output widths")
    if any(unit < 1 for unit in layer_units):
        raise ValueError("layer_units must all be positive")
    coefs = tuple(
        AbstractArray(shape=(layer_units[index], layer_units[index + 1]), dtype="float64")
        for index in range(len(layer_units) - 1)
    )
    intercepts = tuple(
        AbstractArray(shape=(layer_units[index + 1],), dtype="float64")
        for index in range(len(layer_units) - 1)
    )
    return coefs, intercepts, coefs, intercepts
