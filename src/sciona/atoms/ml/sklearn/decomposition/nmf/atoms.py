"""Deterministic NMF helper atoms adapted from scikit-learn."""

from __future__ import annotations

from typing import Literal

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_nmf_beta_divergence,
    witness_nmf_beta_loss_to_float,
    witness_nmf_check_init_matrix,
    witness_nmf_nndsvd_from_svd,
    witness_nmf_random_initialize,
    witness_nmf_trace_dot,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix
BetaLoss = float | str
RandomStateLike = int | np.random.RandomState | None
NndsvdInit = Literal["nndsvd", "nndsvda", "nndsvdar"]
ShapeDim = int | Literal["auto"]
ShapeSpec = tuple[ShapeDim, ShapeDim]
NmfFactors = tuple[NDArray[np.float64], NDArray[np.float64]]
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

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _nndsvd_init_valid(init: str) -> bool:
    return init in {"nndsvd", "nndsvda", "nndsvdar"}

def _dense_matrix(values: object) -> bool:
    if sp.issparse(values):
        return False
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _dense_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _nonnegative_vector(values: object) -> bool:
    return bool(_dense_vector(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0))

def _shape_spec_valid(shape: ShapeSpec) -> bool:
    if not isinstance(shape, tuple) or len(shape) != 2:
        return False
    return all(value == "auto" or _positive_int(value) for value in shape)

def _matching_or_auto(actual: int, expected: ShapeDim) -> bool:
    return bool(expected == "auto" or actual == expected)

def _checked_matrix_matches_shape(result: NDArray[np.float64], shape: ShapeSpec) -> bool:
    return bool(
        result.ndim == 2
        and result.shape[0] >= 1
        and result.shape[1] >= 1
        and np.all(np.isfinite(result))
        and np.all(result >= 0.0)
        and _matching_or_auto(result.shape[0], shape[0])
        and _matching_or_auto(result.shape[1], shape[1])
    )

def _factor_pair_valid(result: NmfFactors, n_samples: int, n_features: int, n_components: int) -> bool:
    W, H = result
    return bool(
        W.shape == (n_samples, n_components)
        and H.shape == (n_components, n_features)
        and np.all(np.isfinite(W))
        and np.all(np.isfinite(H))
        and np.all(W >= 0.0)
        and np.all(H >= 0.0)
    )

def _svd_triplet_valid(U: NDArray[np.float64], S: NDArray[np.float64], V: NDArray[np.float64]) -> bool:
    return bool(
        _dense_matrix(U)
        and _nonnegative_vector(S)
        and _dense_matrix(V)
        and U.shape[1] == S.shape[0] == V.shape[0]
    )

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

def _nndsvd_checked_inputs(
    U: NDArray[np.float64],
    S: NDArray[np.float64],
    V: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    from sklearn.utils.validation import check_array, check_non_negative, check_random_state
    left = np.asarray(check_array(U, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    singular_values = np.asarray(check_array(S, dtype=np.float64, ensure_2d=False), dtype=np.float64)
    right = np.asarray(check_array(V, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    if singular_values.ndim != 1:
        raise ValueError("S must be one-dimensional")
    if left.shape[1] != singular_values.shape[0] or right.shape[0] != singular_values.shape[0]:
        raise ValueError("U, S, and V must share the same component count")
    if np.any(singular_values < 0.0):
        raise ValueError("S must be nonnegative")
    return left, singular_values, right

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
    from sklearn.utils.extmath import squared_norm
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

@register_atom(witness_nmf_random_initialize)
@icontract.require(lambda X: _dense_nonnegative_matrix(X), "X must be a dense finite nonnegative matrix")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(
    lambda X, n_components, result: _factor_pair_valid(result, X.shape[0], X.shape[1], n_components),
    "random initialization must return finite nonnegative factors with sklearn-compatible shapes",
)
def nmf_random_initialize(
    X: NDArray[np.float64],
    n_components: int,
    *,
    random_state: RandomStateLike = None,
) -> NmfFactors:
    from sklearn.utils.validation import check_array, check_non_negative, check_random_state
    """Initialize dense nonnegative NMF factors with sklearn's random scaling rule."""
    checked_x = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    check_non_negative(checked_x, "NMF initialization")

    avg = float(np.sqrt(checked_x.mean() / n_components))
    rng = check_random_state(random_state)
    H = avg * rng.standard_normal(size=(n_components, checked_x.shape[1])).astype(np.float64, copy=False)
    W = avg * rng.standard_normal(size=(checked_x.shape[0], n_components)).astype(np.float64, copy=False)
    np.abs(H, out=H)
    np.abs(W, out=W)
    return W, H

@register_atom(witness_nmf_nndsvd_from_svd)
@icontract.require(lambda U, S, V: _svd_triplet_valid(U, S, V), "U, S, and V must form a finite compatible SVD triplet")
@icontract.require(lambda init: _nndsvd_init_valid(init), "init must be one of sklearn's NNDSVD fill modes")
@icontract.require(lambda data_mean: _nonnegative_finite_scalar(data_mean), "data_mean must be finite and nonnegative")
@icontract.require(lambda eps: _nonnegative_finite_scalar(eps), "eps must be finite and nonnegative")
@icontract.ensure(
    lambda U, V, result: _factor_pair_valid(result, U.shape[0], V.shape[1], U.shape[1]),
    "NNDSVD initialization must return finite nonnegative factors with shapes derived from the SVD triplet",
)
def nmf_nndsvd_from_svd(
    U: NDArray[np.float64],
    S: NDArray[np.float64],
    V: NDArray[np.float64],
    init: NndsvdInit,
    data_mean: float,
    *,
    eps: float = 1e-6,
    random_state: RandomStateLike = None,
) -> NmfFactors:
    from sklearn.utils.validation import check_array, check_non_negative, check_random_state
    """Build sklearn's NNDSVD-style initial factors from a supplied dense SVD triplet."""
    left, singular_values, right = _nndsvd_checked_inputs(U, S, V)
    W = np.zeros_like(left)
    H = np.zeros_like(right)

    W[:, 0] = np.sqrt(singular_values[0]) * np.abs(left[:, 0])
    H[0, :] = np.sqrt(singular_values[0]) * np.abs(right[0, :])

    for j in range(1, singular_values.shape[0]):
        x = left[:, j]
        y = right[j, :]

        x_p = np.maximum(x, 0.0)
        y_p = np.maximum(y, 0.0)
        x_n = np.abs(np.minimum(x, 0.0))
        y_n = np.abs(np.minimum(y, 0.0))

        x_p_nrm = float(np.linalg.norm(x_p))
        y_p_nrm = float(np.linalg.norm(y_p))
        x_n_nrm = float(np.linalg.norm(x_n))
        y_n_nrm = float(np.linalg.norm(y_n))

        m_p = x_p_nrm * y_p_nrm
        m_n = x_n_nrm * y_n_nrm

        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = float(np.sqrt(singular_values[j] * sigma))
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0.0
    H[H < eps] = 0.0

    if init == "nndsvda":
        W[W == 0.0] = data_mean
        H[H == 0.0] = data_mean
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        W[W == 0.0] = np.abs(data_mean * rng.standard_normal(size=int(np.count_nonzero(W == 0.0))) / 100.0)
        H[H == 0.0] = np.abs(data_mean * rng.standard_normal(size=int(np.count_nonzero(H == 0.0))) / 100.0)

    return W, H

@register_atom(witness_nmf_check_init_matrix)
@icontract.require(lambda A: _dense_nonnegative_matrix(A), "A must be a dense finite nonnegative matrix")
@icontract.require(lambda shape: _shape_spec_valid(shape), "shape must be a pair of positive integers or 'auto'")
@icontract.require(lambda whom: isinstance(whom, str) and whom.strip() != "", "whom must be a nonempty label")
@icontract.ensure(
    lambda shape, result: _checked_matrix_matches_shape(result, shape),
    "validated initialization matrix must remain finite, nonnegative, nonempty, and match requested dimensions",
)
def nmf_check_init_matrix(
    A: NDArray[np.float64],
    shape: ShapeSpec,
    whom: str,
) -> NDArray[np.float64]:
    from sklearn.utils.validation import check_array, check_non_negative, check_random_state
    """Validate and return a dense nonnegative matrix used to initialize nonnegative matrix factorization."""
    checked = np.asarray(check_array(A, dtype=np.float64, ensure_2d=True), dtype=np.float64)

    if shape[0] != "auto" and checked.shape[0] != shape[0]:
        raise ValueError(
            f"Array with wrong first dimension passed to {whom}. Expected {shape[0]}, but got {checked.shape[0]}."
        )
    if shape[1] != "auto" and checked.shape[1] != shape[1]:
        raise ValueError(
            f"Array with wrong second dimension passed to {whom}. Expected {shape[1]}, but got {checked.shape[1]}."
        )
    check_non_negative(checked, whom)
    if float(np.max(checked)) == 0.0:
        raise ValueError(f"Array passed to {whom} is full of zeros.")
    return checked
