"""Deterministic NMF helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.utils.extmath import squared_norm

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_nmf_beta_divergence,
    witness_nmf_beta_loss_to_float,
    witness_nmf_trace_dot,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix
BetaLoss = float | str
EPSILON = np.finfo(np.float64).eps


def _finite_matrix(values: MatrixLike) -> bool:
    if sp.issparse(values):
        return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values.data)))
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_matrix(values: MatrixLike) -> bool:
    if not _finite_matrix(values):
        return False
    if sp.issparse(values):
        return bool(np.all(values.data >= 0.0))
    return bool(np.all(np.asarray(values, dtype=np.float64) >= 0.0))


def _dense_nonnegative_matrix(values: MatrixLike) -> bool:
    return bool(not sp.issparse(values) and _nonnegative_matrix(values))


def _same_shape(X: MatrixLike, Y: MatrixLike) -> bool:
    return bool(_finite_matrix(X) and _finite_matrix(Y) and X.shape == Y.shape)


def _factor_shapes_valid(X: MatrixLike, W: MatrixLike, H: MatrixLike) -> bool:
    return bool(
        _nonnegative_matrix(X)
        and _dense_nonnegative_matrix(W)
        and _dense_nonnegative_matrix(H)
        and X.shape[0] == W.shape[0]
        and X.shape[1] == H.shape[1]
        and W.shape[1] == H.shape[0]
    )


def _beta_loss_valid(beta_loss: BetaLoss) -> bool:
    if isinstance(beta_loss, str):
        return beta_loss in {"frobenius", "kullback-leibler", "itakura-saito"}
    if isinstance(beta_loss, bool):
        return False
    try:
        value = float(beta_loss)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value))


def _beta_divergence_input_valid(X: MatrixLike, W: MatrixLike, H: MatrixLike, beta: BetaLoss) -> bool:
    if not (_factor_shapes_valid(X, W, H) and _beta_loss_valid(beta)):
        return False
    beta_value = _beta_loss_to_float_unchecked(beta)
    if beta_value <= 0:
        if sp.issparse(X):
            return bool(X.nnz == np.prod(X.shape))
        return bool(np.all(np.asarray(X, dtype=np.float64) > 0.0))
    return True


def _finite_scalar(value: float) -> bool:
    return bool(np.isscalar(value) and np.isfinite(float(value)))


def _nonnegative_finite_scalar(value: float) -> bool:
    return bool(_finite_scalar(value) and float(value) >= 0.0)


def _beta_loss_to_float_unchecked(beta_loss: BetaLoss) -> float:
    beta_loss_map = {"frobenius": 2.0, "kullback-leibler": 1.0, "itakura-saito": 0.0}
    if isinstance(beta_loss, str):
        return float(beta_loss_map[beta_loss])
    return float(beta_loss)


def _special_sparse_dot(W: NDArray[np.float64], H: NDArray[np.float64], X: sp.spmatrix) -> sp.csr_matrix:
    ii, jj = X.nonzero()
    n_vals = ii.shape[0]
    dot_vals = np.empty(n_vals, dtype=np.float64)
    n_components = W.shape[1]
    batch_size = max(n_components, n_vals // n_components)
    for start in range(0, n_vals, batch_size):
        batch = slice(start, start + batch_size)
        dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(axis=1)
    return sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape).tocsr()


@register_atom(witness_nmf_beta_loss_to_float)
@icontract.require(lambda beta_loss: _beta_loss_valid(beta_loss), "beta_loss must be finite or one of sklearn's named beta losses")
@icontract.ensure(lambda result: _finite_scalar(result), "numeric beta loss must be finite")
def nmf_beta_loss_to_float(beta_loss: BetaLoss) -> float:
    """Convert sklearn's named beta-loss options to numeric values."""
    return _beta_loss_to_float_unchecked(beta_loss)


@register_atom(witness_nmf_trace_dot)
@icontract.require(lambda X, Y: _same_shape(X, Y), "X and Y must be finite matrices with the same shape")
@icontract.ensure(lambda result: _finite_scalar(result), "trace product must be finite")
def nmf_trace_dot(X: MatrixLike, Y: MatrixLike) -> float:
    """Compute sklearn's flattened trace product for two equal-shaped matrices."""
    if sp.issparse(X):
        x_values = X.toarray()
    else:
        x_values = np.asarray(X, dtype=np.float64)
    if sp.issparse(Y):
        y_values = Y.toarray()
    else:
        y_values = np.asarray(Y, dtype=np.float64)
    return float(np.dot(x_values.ravel(), y_values.ravel()))


@register_atom(witness_nmf_beta_divergence)
@icontract.require(lambda X, W, H, beta: _beta_divergence_input_valid(X, W, H, beta), "X, W, and H must be compatible nonnegative matrices")
@icontract.ensure(lambda result: _nonnegative_finite_scalar(result), "beta divergence must be finite and nonnegative")
def nmf_beta_divergence(
    X: MatrixLike,
    W: NDArray[np.float64],
    H: NDArray[np.float64],
    beta: BetaLoss,
    *,
    square_root: bool = False,
) -> float:
    """Compute sklearn's beta-divergence between X and the product W H."""
    beta_value = nmf_beta_loss_to_float(beta)
    w_values = np.asarray(W, dtype=np.float64)
    h_values = np.asarray(H, dtype=np.float64)

    if not sp.issparse(X):
        x_values: MatrixLike = np.atleast_2d(np.asarray(X, dtype=np.float64))
    else:
        x_values = X

    if beta_value == 2.0:
        if sp.issparse(x_values):
            norm_X = np.dot(x_values.data, x_values.data)
            norm_WH = nmf_trace_dot(np.linalg.multi_dot([w_values.T, w_values, h_values]), h_values)
            cross_prod = nmf_trace_dot((x_values @ h_values.T), w_values)
            result = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            result = squared_norm(np.asarray(x_values, dtype=np.float64) - np.dot(w_values, h_values)) / 2.0
        return float(np.sqrt(result * 2.0) if square_root else result)

    if sp.issparse(x_values):
        wh_data = _special_sparse_dot(w_values, h_values, x_values).data
        x_data = x_values.data
    else:
        wh = np.dot(w_values, h_values)
        wh_data = wh.ravel()
        x_data = np.asarray(x_values, dtype=np.float64).ravel()

    indices = x_data > EPSILON
    wh_data = wh_data[indices]
    x_data = x_data[indices]
    wh_data[wh_data < EPSILON] = EPSILON

    if beta_value == 1.0:
        sum_WH = np.dot(np.sum(w_values, axis=0), np.sum(h_values, axis=1))
        div = x_data / wh_data
        result = np.dot(x_data, np.log(div))
        result += sum_WH - x_data.sum()
    elif beta_value == 0.0:
        div = x_data / wh_data
        result = np.sum(div) - np.prod(x_values.shape) - np.sum(np.log(div))
    else:
        if sp.issparse(x_values):
            sum_WH_beta = 0.0
            for i in range(x_values.shape[1]):
                sum_WH_beta += np.sum(np.dot(w_values, h_values[:, i]) ** beta_value)
        else:
            sum_WH_beta = np.sum(wh**beta_value)
        sum_X_WH = np.dot(x_data, wh_data ** (beta_value - 1.0))
        result = (x_data**beta_value).sum() - beta_value * sum_X_WH
        result += sum_WH_beta * (beta_value - 1.0)
        result /= beta_value * (beta_value - 1.0)

    if square_root:
        result = max(float(result), 0.0)
        return float(np.sqrt(2.0 * result))
    return float(result)
