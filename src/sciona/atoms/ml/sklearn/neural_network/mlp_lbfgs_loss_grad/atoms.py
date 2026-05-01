"""MLP LBFGS loss/gradient helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from ..mlp_lbfgs_bookkeeping.atoms import CoefSlice, InterceptSlice
from ..mlp_lbfgs_bookkeeping.atoms import (
    mlp_lbfgs_pack_parameters,
)
from ..mlp_primitives.atoms import (
    ActivationName,
    HiddenActivationName,
    LossName,
    mlp_backprop,
)
from .witnesses import (
    witness_mlp_lbfgs_loss_grad,
    witness_mlp_lbfgs_unpack_parameters,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _coef_indptr_valid(result: tuple[CoefSlice, ...]) -> bool:
    if not isinstance(result, tuple) or len(result) < 1:
        return False
    start = 0
    for entry in result:
        if not (isinstance(entry, tuple) and len(entry) == 3):
            return False
        slice_start, slice_end, shape = entry
        if (
            not isinstance(slice_start, int)
            or not isinstance(slice_end, int)
            or slice_start != start
            or not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(_positive_int(dim) for dim in shape)
            or slice_end - slice_start != int(shape[0]) * int(shape[1])
        ):
            return False
        start = slice_end
    return True


def _intercept_indptr_valid(
    result: tuple[InterceptSlice, ...],
    coef_indptr: tuple[CoefSlice, ...],
) -> bool:
    if not isinstance(result, tuple) or len(result) != len(coef_indptr):
        return False
    start = int(coef_indptr[-1][1])
    for index, entry in enumerate(result):
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return False
        slice_start, slice_end = entry
        expected_width = int(coef_indptr[index][2][1])
        if (
            not isinstance(slice_start, int)
            or not isinstance(slice_end, int)
            or slice_start != start
            or slice_end - slice_start != expected_width
        ):
            return False
        start = slice_end
    return True


def _packed_length_valid(
    packed_parameters: NDArray[np.float64],
    intercept_indptr: tuple[InterceptSlice, ...],
) -> bool:
    return bool(_finite_vector(packed_parameters) and int(np.asarray(packed_parameters).shape[0]) == int(intercept_indptr[-1][1]))


def _unpacked_parameter_blocks_valid(
    result: tuple[tuple[NDArray[np.float64], ...], tuple[NDArray[np.float64], ...]],
    coef_indptr: tuple[CoefSlice, ...],
    intercept_indptr: tuple[InterceptSlice, ...],
) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    coefs, intercepts = result
    if not (isinstance(coefs, tuple) and isinstance(intercepts, tuple)):
        return False
    if len(coefs) != len(coef_indptr) or len(intercepts) != len(intercept_indptr):
        return False
    for coef, (slice_start, slice_end, shape) in zip(coefs, coef_indptr):
        values = np.asarray(coef, dtype=np.float64)
        if values.shape != shape or values.size != slice_end - slice_start or not np.all(np.isfinite(values)):
            return False
    for intercept, (slice_start, slice_end) in zip(intercepts, intercept_indptr):
        values = np.asarray(intercept, dtype=np.float64)
        if values.shape != (slice_end - slice_start,) or not np.all(np.isfinite(values)):
            return False
    return True


def _targets_valid(y: object, X: object, coef_indptr: tuple[CoefSlice, ...]) -> bool:
    try:
        targets = np.asarray(y, dtype=np.float64)
        samples = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        targets.ndim == 2
        and samples.ndim == 2
        and targets.shape[0] == samples.shape[0]
        and targets.shape[1] == int(coef_indptr[-1][2][1])
        and np.all(np.isfinite(targets))
    )


def _hidden_activation_name_valid(value: object) -> bool:
    return isinstance(value, str) and value in {"identity", "logistic", "tanh", "relu"}


def _activation_name_valid(value: object) -> bool:
    return isinstance(value, str) and value in {"identity", "logistic", "tanh", "relu", "softmax"}


def _loss_name_valid(value: object) -> bool:
    return isinstance(value, str) and value in {"log_loss", "squared_error"}


def _canonical_loss_combo_valid(loss_name: str, output_activation: str) -> bool:
    return bool(
        _loss_name_valid(loss_name)
        and _activation_name_valid(output_activation)
        and (
            (loss_name == "squared_error" and output_activation == "identity")
            or (loss_name == "log_loss" and output_activation in {"logistic", "softmax"})
        )
    )


def _nonnegative_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0


def _lbfgs_loss_grad_result_valid(
    result: tuple[float, NDArray[np.float64]],
    packed_parameters: NDArray[np.float64],
) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    loss, grad = result
    gradient = np.asarray(grad, dtype=np.float64)
    return bool(
        isinstance(loss, float)
        and np.isfinite(loss)
        and gradient.ndim == 1
        and gradient.shape == np.asarray(packed_parameters, dtype=np.float64).shape
        and np.all(np.isfinite(gradient))
    )


@register_atom(witness_mlp_lbfgs_unpack_parameters)
@icontract.require(lambda coef_indptr: _coef_indptr_valid(coef_indptr), "coef_indptr must define consecutive coefficient slices")
@icontract.require(lambda coef_indptr, intercept_indptr: _intercept_indptr_valid(intercept_indptr, coef_indptr), "intercept_indptr must continue the coefficient layout")
@icontract.require(lambda packed_parameters, intercept_indptr: _packed_length_valid(packed_parameters, intercept_indptr), "packed_parameters must be a finite vector matching the full LBFGS parameter width")
@icontract.ensure(lambda result, coef_indptr, intercept_indptr: _unpacked_parameter_blocks_valid(result, coef_indptr, intercept_indptr), "unpacked coefficient and intercept blocks must match the supplied LBFGS slice layout")
def mlp_lbfgs_unpack_parameters(
    packed_parameters: NDArray[np.float64],
    coef_indptr: tuple[CoefSlice, ...],
    intercept_indptr: tuple[InterceptSlice, ...],
) -> tuple[tuple[NDArray[np.float64], ...], tuple[NDArray[np.float64], ...]]:
    """Unpack sklearn's flat MLP LBFGS parameter vector into coefficient and intercept blocks."""
    packed = np.asarray(packed_parameters, dtype=np.float64)
    coefs = tuple(
        np.asarray(packed[start:end].reshape(shape), dtype=np.float64)
        for start, end, shape in coef_indptr
    )
    intercepts = tuple(
        np.asarray(packed[start:end], dtype=np.float64)
        for start, end in intercept_indptr
    )
    return coefs, intercepts


@register_atom(witness_mlp_lbfgs_loss_grad)
@icontract.require(lambda coef_indptr: _coef_indptr_valid(coef_indptr), "coef_indptr must define consecutive coefficient slices")
@icontract.require(lambda coef_indptr, intercept_indptr: _intercept_indptr_valid(intercept_indptr, coef_indptr), "intercept_indptr must continue the coefficient layout")
@icontract.require(lambda packed_parameters, intercept_indptr: _packed_length_valid(packed_parameters, intercept_indptr), "packed_parameters must match the full LBFGS parameter width")
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite dense sample matrix")
@icontract.require(lambda y, X, coef_indptr: _targets_valid(y, X, coef_indptr), "y must be a finite dense target matrix aligned with X and the final layer width")
@icontract.require(lambda hidden_activation: _hidden_activation_name_valid(hidden_activation), "hidden_activation must be one of sklearn's hidden activations")
@icontract.require(lambda loss_name, output_activation: _canonical_loss_combo_valid(loss_name, output_activation), "loss_name and output_activation must form a sklearn MLP canonical output-loss pair")
@icontract.require(lambda alpha: _nonnegative_scalar(alpha), "alpha must be finite and nonnegative")
@icontract.ensure(lambda result, packed_parameters: _lbfgs_loss_grad_result_valid(result, packed_parameters), "LBFGS loss/gradient must return a finite loss and a finite packed gradient vector")
def mlp_lbfgs_loss_grad(
    packed_parameters: NDArray[np.float64],
    coef_indptr: tuple[CoefSlice, ...],
    intercept_indptr: tuple[InterceptSlice, ...],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    hidden_activation: HiddenActivationName,
    output_activation: ActivationName,
    loss_name: LossName,
    alpha: float = 0.0,
) -> tuple[float, NDArray[np.float64]]:
    """Compute sklearn's LBFGS loss scalar and packed gradient from a flat parameter vector."""
    coefs, intercepts = mlp_lbfgs_unpack_parameters(packed_parameters, coef_indptr, intercept_indptr)
    loss, coef_grads, intercept_grads = mlp_backprop(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        coefs,
        intercepts,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        loss_name=loss_name,
        alpha=float(alpha),
    )
    gradient = mlp_lbfgs_pack_parameters(coef_grads, intercept_grads)
    return float(loss), np.asarray(gradient, dtype=np.float64)
