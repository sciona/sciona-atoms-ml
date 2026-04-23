"""Ghost witnesses for sklearn MLP helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_mlp_activation(values: AbstractArray, *, activation: str) -> AbstractArray:
    """Describe one dense activation output matrix."""
    del activation
    if len(values.shape) != 2:
        raise ValueError("values must be a 2D matrix")
    return AbstractArray(shape=values.shape, dtype="float64")


def witness_mlp_activation_derivative(
    activated_values: AbstractArray,
    delta: AbstractArray,
    *,
    activation: str,
) -> AbstractArray:
    """Describe one dense delta matrix after a hidden-activation derivative."""
    del activation
    if len(activated_values.shape) != 2 or len(delta.shape) != 2:
        raise ValueError("activated_values and delta must be 2D matrices")
    if activated_values.shape != delta.shape:
        raise ValueError("activated_values and delta must have matching shapes")
    return AbstractArray(shape=delta.shape, dtype="float64")


def witness_mlp_loss(
    y_true: AbstractArray,
    y_pred: AbstractArray,
    *,
    loss_name: str,
    output_activation: str,
    sample_weight: AbstractArray | None = None,
) -> float:
    """Describe one scalar MLP output-layer loss value."""
    del loss_name, output_activation
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        raise ValueError("y_true and y_pred must be 2D matrices")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    if sample_weight is not None:
        if len(sample_weight.shape) != 1 or int(sample_weight.shape[0]) != int(y_true.shape[0]):
            raise ValueError("sample_weight must be one-dimensional over samples")
    return 0.0


def witness_mlp_forward_pass(
    X: AbstractArray,
    coefs: tuple[AbstractArray, ...],
    intercepts: tuple[AbstractArray, ...],
    *,
    hidden_activation: str,
    output_activation: str,
) -> tuple[AbstractArray, ...]:
    """Describe one dense activation matrix for each MLP layer, including the input layer."""
    del hidden_activation, output_activation
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D matrix")
    if not coefs or len(coefs) != len(intercepts):
        raise ValueError("coefs and intercepts must be nonempty tuples of matching length")

    n_samples = int(X.shape[0])
    previous_width = int(X.shape[1])
    activations: list[AbstractArray] = [AbstractArray(shape=X.shape, dtype="float64")]
    for coef, intercept in zip(coefs, intercepts):
        if len(coef.shape) != 2 or len(intercept.shape) != 1:
            raise ValueError("each coefficient must be 2D and each intercept must be 1D")
        if int(coef.shape[0]) != previous_width or int(intercept.shape[0]) != int(coef.shape[1]):
            raise ValueError("coefs and intercepts must form a consistent dense network chain")
        previous_width = int(coef.shape[1])
        activations.append(AbstractArray(shape=(n_samples, previous_width), dtype="float64"))
    return tuple(activations)


def witness_mlp_layer_gradients(
    layer_activation: AbstractArray,
    delta: AbstractArray,
    coefs: AbstractArray,
    *,
    alpha: float = 0.0,
    sample_weight_sum: float | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe one dense coefficient-gradient matrix and one intercept-gradient vector."""
    del alpha, sample_weight_sum
    if len(layer_activation.shape) != 2 or len(delta.shape) != 2 or len(coefs.shape) != 2:
        raise ValueError("layer_activation, delta, and coefs must be dense matrices")
    if int(layer_activation.shape[0]) != int(delta.shape[0]):
        raise ValueError("layer_activation and delta must align over samples")
    if int(layer_activation.shape[1]) != int(coefs.shape[0]) or int(delta.shape[1]) != int(coefs.shape[1]):
        raise ValueError("coefs must align with layer_activation inputs and delta outputs")
    return (
        AbstractArray(shape=coefs.shape, dtype="float64"),
        AbstractArray(shape=(int(delta.shape[1]),), dtype="float64"),
    )


def witness_mlp_backprop(
    X: AbstractArray,
    y: AbstractArray,
    coefs: tuple[AbstractArray, ...],
    intercepts: tuple[AbstractArray, ...],
    *,
    hidden_activation: str,
    output_activation: str,
    loss_name: str,
    alpha: float = 0.0,
    sample_weight: AbstractArray | None = None,
) -> tuple[float, tuple[AbstractArray, ...], tuple[AbstractArray, ...]]:
    """Describe dense MLP loss and parameter gradients for fixed network parameters."""
    del hidden_activation, output_activation, loss_name, alpha
    activations = witness_mlp_forward_pass(
        X,
        coefs,
        intercepts,
        hidden_activation="identity",
        output_activation="identity",
    )
    output = activations[-1]
    if len(y.shape) != 2 or y.shape != output.shape:
        raise ValueError("y must be a 2D matrix matching the network output shape")
    if sample_weight is not None:
        if len(sample_weight.shape) != 1 or int(sample_weight.shape[0]) != int(X.shape[0]):
            raise ValueError("sample_weight must be one-dimensional over samples")
    coef_grads = tuple(AbstractArray(shape=coef.shape, dtype="float64") for coef in coefs)
    intercept_grads = tuple(AbstractArray(shape=intercept.shape, dtype="float64") for intercept in intercepts)
    return 0.0, coef_grads, intercept_grads
