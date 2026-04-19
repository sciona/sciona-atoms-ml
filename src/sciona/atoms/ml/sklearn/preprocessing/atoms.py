"""Selected preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from numpy.typing import NDArray
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis, min_max_axis
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2
from sklearn.utils.validation import FLOAT_DTYPES

from sciona.ghost.registry import register_atom

from .state_models import KernelCentererState, MaxAbsScalerState, MinMaxScalerState
from .witnesses import (
    witness_add_dummy_feature,
    witness_binarize,
    witness_binarizer_transform,
    witness_kernel_centerer_fit,
    witness_kernel_centerer_transform,
    witness_maxabs_scale,
    witness_maxabs_scaler_fit,
    witness_maxabs_scaler_inverse_transform,
    witness_maxabs_scaler_partial_fit,
    witness_maxabs_scaler_transform,
    witness_minmax_scale,
    witness_minmax_scaler_fit,
    witness_minmax_scaler_inverse_transform,
    witness_minmax_scaler_partial_fit,
    witness_minmax_scaler_transform,
    witness_normalize,
    witness_normalizer_transform,
    witness_robust_scale,
    witness_scale,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix
NormalizeResult = MatrixLike | tuple[MatrixLike, NDArray[np.float64]]


def _is_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) == 2)


def _is_1d_or_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) in {1, 2})


def _row_count(X: MatrixLike) -> int:
    return int(X.shape[0])


def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1])


def _valid_norm(norm: str) -> bool:
    return norm in {"l1", "l2", "max"}


def _valid_axis(axis: int) -> bool:
    return axis in {0, 1}


def _is_binary_matrix(X: MatrixLike) -> bool:
    values = X.data if sp.issparse(X) else np.asarray(X)
    return bool(np.all((values == 0) | (values == 1)))


def _leading_column_matches(X: MatrixLike, value: float) -> bool:
    column = X.getcol(0).toarray().ravel() if sp.issparse(X) else X[:, 0]
    return bool(np.all(column == value))


def _normalize_shape_matches(result: NormalizeResult, X: MatrixLike, axis: int, return_norm: bool) -> bool:
    if return_norm:
        if not isinstance(result, tuple) or len(result) != 2:
            return False
        normalized, norms = result
        norm_count = _feature_count(X) if axis == 0 else _row_count(X)
        return normalized.shape == X.shape and norms.shape == (norm_count,)
    return not isinstance(result, tuple) and result.shape == X.shape


def _kernel_state_valid(state: KernelCentererState) -> bool:
    return bool(state.k_fit_rows.ndim == 1 and state.k_fit_rows.shape[0] == state.n_features_in)


def _maxabs_state_valid(state: MaxAbsScalerState) -> bool:
    return bool(
        state.scale.ndim == 1
        and state.max_abs.ndim == 1
        and state.scale.shape == state.max_abs.shape
        and state.scale.shape[0] == state.n_features_in
        and state.n_samples_seen >= 1
    )


def _valid_feature_range(feature_range: tuple[float, float]) -> bool:
    return bool(feature_range[0] < feature_range[1])


def _minmax_state_valid(state: MinMaxScalerState) -> bool:
    shape = state.scale.shape
    return bool(
        state.min_.ndim == 1
        and state.scale.ndim == 1
        and state.data_min.ndim == 1
        and state.data_max.ndim == 1
        and state.data_range.ndim == 1
        and state.min_.shape == shape
        and state.data_min.shape == shape
        and state.data_max.shape == shape
        and state.data_range.shape == shape
        and shape[0] == state.n_features_in
        and state.n_samples_seen >= 1
        and _valid_feature_range(state.feature_range)
    )


@register_atom(witness_add_dummy_feature)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.ensure(lambda result, X: result.shape == (_row_count(X), _feature_count(X) + 1), "output must add one feature column")
@icontract.ensure(lambda result, value: _leading_column_matches(result, value), "first column must contain the dummy value")
def add_dummy_feature(
    X: MatrixLike,
    value: float = 1.0,
) -> MatrixLike:
    """Add a leading constant feature column to a dense or sparse matrix."""
    checked_x = check_array(X, accept_sparse=["csc", "csr", "coo"], dtype=FLOAT_DTYPES)
    n_samples, n_features = checked_x.shape
    shape = (n_samples, n_features + 1)
    if sp.issparse(checked_x):
        if checked_x.format == "coo":
            col = np.concatenate((np.zeros(n_samples), checked_x.col + 1))
            row = np.concatenate((np.arange(n_samples), checked_x.row))
            data = np.concatenate((np.full(n_samples, value), checked_x.data))
            return sp.coo_matrix((data, (row, col)), shape)
        if checked_x.format == "csc":
            indptr = np.concatenate((np.array([0]), checked_x.indptr + n_samples))
            indices = np.concatenate((np.arange(n_samples), checked_x.indices))
            data = np.concatenate((np.full(n_samples, value), checked_x.data))
            return sp.csc_matrix((data, indices, indptr), shape)
        klass = checked_x.__class__
        return klass(add_dummy_feature(checked_x.tocoo(), value))
    return np.hstack((np.full((n_samples, 1), value), checked_x))


@register_atom(witness_binarize)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.ensure(lambda result, X: result.shape == X.shape, "binarized output must preserve shape")
@icontract.ensure(lambda result: _is_binary_matrix(result), "binarized output must contain only 0 and 1 values")
def binarize(
    X: MatrixLike,
    *,
    threshold: float = 0.0,
    copy: bool = True,
) -> MatrixLike:
    """Threshold each matrix entry to 0 or 1."""
    checked_x = check_array(X, accept_sparse=["csr", "csc"], force_writeable=True, copy=copy)
    if sp.issparse(checked_x):
        if threshold < 0:
            raise ValueError("Cannot binarize a sparse matrix with threshold < 0")
        cond = checked_x.data > threshold
        checked_x.data[cond] = 1
        checked_x.data[np.logical_not(cond)] = 0
        checked_x.eliminate_zeros()
    else:
        cond = checked_x.astype(np.result_type(checked_x.dtype, float, type(threshold)), copy=False) > threshold
        checked_x[cond] = 1
        checked_x[np.logical_not(cond)] = 0
    return checked_x


@register_atom(witness_binarizer_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.ensure(lambda result, X: result.shape == X.shape, "binarizer output must preserve shape")
@icontract.ensure(lambda result: _is_binary_matrix(result), "binarizer output must contain only 0 and 1 values")
def binarizer_transform(
    X: MatrixLike,
    *,
    threshold: float = 0.0,
    copy: bool = True,
) -> MatrixLike:
    """Apply stateless Binarizer.transform semantics to a matrix."""
    checked_x = check_array(X, accept_sparse=["csr", "csc"], force_writeable=True, copy=copy)
    return binarize(checked_x, threshold=threshold, copy=False)


@register_atom(witness_normalize)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda norm: _valid_norm(norm), "norm must be 'l1', 'l2', or 'max'")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.ensure(lambda result, X, axis, return_norm: _normalize_shape_matches(result, X, axis, return_norm), "normalized output must preserve matrix shape and norm vector shape")
def normalize(
    X: MatrixLike,
    norm: str = "l2",
    *,
    axis: int = 1,
    copy: bool = True,
    return_norm: bool = False,
) -> NormalizeResult:
    """Scale rows or columns of a matrix to unit l1, l2, or max norm."""
    sparse_format = "csc" if axis == 0 else "csr"
    checked_x = check_array(
        X,
        accept_sparse=sparse_format,
        copy=copy,
        estimator="the normalize function",
        dtype=FLOAT_DTYPES,
        force_writeable=True,
    )
    if axis == 0:
        checked_x = checked_x.T

    if sp.issparse(checked_x):
        if return_norm and norm in ("l1", "l2"):
            raise ValueError(
                "return_norm=True is not implemented for sparse matrices with norm 'l1' or norm 'l2'"
            )
        if norm == "l1":
            inplace_csr_row_normalize_l1(checked_x)
        elif norm == "l2":
            inplace_csr_row_normalize_l2(checked_x)
        elif norm == "max":
            mins, maxes = min_max_axis(checked_x, 1)
            norms = np.maximum(abs(mins), maxes)
            norms_elementwise = norms.repeat(np.diff(checked_x.indptr))
            mask = norms_elementwise != 0
            checked_x.data[mask] /= norms_elementwise[mask]
    else:
        if norm == "l1":
            norms = np.sum(np.abs(checked_x), axis=1)
        elif norm == "l2":
            norms = row_norms(checked_x)
        elif norm == "max":
            norms = np.max(np.abs(checked_x), axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        checked_x /= norms[:, None]

    if axis == 0:
        checked_x = checked_x.T

    if return_norm:
        return checked_x, np.asarray(norms)
    return checked_x


@register_atom(witness_normalizer_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda norm: _valid_norm(norm), "norm must be 'l1', 'l2', or 'max'")
@icontract.ensure(lambda result, X: result.shape == X.shape, "normalizer output must preserve shape")
def normalizer_transform(
    X: MatrixLike,
    norm: str = "l2",
    *,
    copy: bool = True,
) -> MatrixLike:
    """Apply stateless Normalizer.transform semantics row-wise."""
    checked_x = check_array(X, accept_sparse="csr", force_writeable=True, copy=copy)
    result = normalize(checked_x, norm=norm, axis=1, copy=False, return_norm=False)
    if isinstance(result, tuple):
        return result[0]
    return result


@register_atom(witness_kernel_centerer_fit)
@icontract.require(lambda K: _is_2d(K), "K must be a 2D kernel matrix")
@icontract.require(lambda K: _row_count(K) == _feature_count(K), "K must be square")
@icontract.ensure(lambda result, K: result.k_fit_rows.shape == (_row_count(K),), "one fitted mean per training sample")
@icontract.ensure(lambda result, K: result.n_features_in == _feature_count(K), "state feature count must match kernel columns")
@icontract.ensure(lambda result: _kernel_state_valid(result), "state means must match fitted feature count")
def kernel_centerer_fit(
    K: NDArray[np.float64],
) -> KernelCentererState:
    """Learn column and global means from a square training kernel matrix."""
    checked_k = check_array(K, dtype=FLOAT_DTYPES)
    if checked_k.shape[0] != checked_k.shape[1]:
        raise ValueError(
            "Kernel matrix must be a square matrix. Input is a {}x{} matrix.".format(
                checked_k.shape[0],
                checked_k.shape[1],
            )
        )
    n_samples = checked_k.shape[0]
    k_fit_rows = np.sum(checked_k, axis=0) / n_samples
    k_fit_all = float(np.sum(k_fit_rows) / n_samples)
    return KernelCentererState(
        k_fit_rows=np.asarray(k_fit_rows, dtype=np.float64),
        k_fit_all=k_fit_all,
        n_features_in=int(checked_k.shape[1]),
    )


@register_atom(witness_kernel_centerer_transform)
@icontract.require(lambda K: _is_2d(K), "K must be a 2D kernel matrix")
@icontract.require(lambda state: _kernel_state_valid(state), "state means must match fitted feature count")
@icontract.require(lambda K, state: _feature_count(K) == state.n_features_in, "K columns must match fitted training samples")
@icontract.ensure(lambda result, K: result.shape == K.shape, "centered kernel output must preserve shape")
def kernel_centerer_transform(
    K: NDArray[np.float64],
    state: KernelCentererState,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Center a kernel matrix block using fitted training-kernel means."""
    checked_k = check_array(K, copy=copy, force_writeable=True, dtype=FLOAT_DTYPES)
    if checked_k.shape[1] != state.n_features_in:
        raise ValueError("K columns must match fitted training samples")
    k_pred_cols = (np.sum(checked_k, axis=1) / state.k_fit_rows.shape[0])[:, None]
    checked_k -= state.k_fit_rows
    checked_k -= k_pred_cols
    checked_k += state.k_fit_all
    return checked_k


@register_atom(witness_scale)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D array")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.ensure(lambda result, X: result.shape == X.shape, "scaled output must preserve shape")
def scale(
    X: MatrixLike,
    *,
    axis: int = 0,
    with_mean: bool = True,
    with_std: bool = True,
    copy: bool = True,
) -> MatrixLike:
    """Center and scale a dense or sparse dataset along an axis."""
    checked_x = check_array(
        X,
        accept_sparse="csc",
        copy=copy,
        ensure_2d=False,
        estimator="the scale function",
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
        input_name="X",
    )
    if sp.issparse(checked_x):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead See docstring for motivation and alternatives."
            )
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0,  got axis=%d" % axis)
        if with_std:
            _mean, var = mean_variance_axis(checked_x, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(checked_x, 1 / np.sqrt(var))
    else:
        checked_x = np.asarray(checked_x)
        if with_mean:
            mean_ = np.nanmean(checked_x, axis)
        if with_std:
            scale_ = np.nanstd(checked_x, axis)
        Xr = np.rollaxis(checked_x, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = np.nanmean(Xr, axis=0)
            if not np.allclose(mean_1, 0):
                warnings.warn(
                    "Numerical issues were encountered when centering the data and might not be solved. "
                    "Dataset may contain too large values. You may need to prescale your features.",
                    UserWarning,
                    stacklevel=2,
                )
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = np.nanmean(Xr, axis=0)
                if not np.allclose(mean_2, 0):
                    warnings.warn(
                        "Numerical issues were encountered when scaling the data and might not be solved. "
                        "The standard deviation of the data is probably very close to 0.",
                        UserWarning,
                        stacklevel=2,
                    )
                    Xr -= mean_2
    return checked_x


def _maxabs_scale_axis0(X: MatrixLike) -> MatrixLike:
    if sp.issparse(X):
        mins, maxes = min_max_axis(X, axis=0, ignore_nan=True)
        max_abs = np.maximum(np.abs(mins), np.abs(maxes))
        scale_ = _handle_zeros_in_scale(max_abs, copy=True)
        inplace_column_scale(X, 1.0 / scale_)
    else:
        scale_ = _handle_zeros_in_scale(np.nanmax(np.abs(X), axis=0), copy=True)
        X /= scale_
    return X


@register_atom(witness_maxabs_scaler_partial_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda X, state: state is None or _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result: _maxabs_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X, state: result.n_samples_seen == _row_count(X) + (0 if state is None else state.n_samples_seen), "sample count must accumulate")
def maxabs_scaler_partial_fit(
    X: MatrixLike,
    state: MaxAbsScalerState | None = None,
) -> MaxAbsScalerState:
    """Update MaxAbsScaler state from one batch of training samples."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    if sp.issparse(checked_x):
        mins, maxes = min_max_axis(checked_x, axis=0, ignore_nan=True)
        max_abs = np.maximum(np.abs(mins), np.abs(maxes))
    else:
        max_abs = np.nanmax(np.abs(checked_x), axis=0)

    if state is not None:
        if checked_x.shape[1] != state.n_features_in:
            raise ValueError("X feature count does not match fitted state")
        max_abs = np.maximum(state.max_abs, max_abs)
        n_samples_seen = state.n_samples_seen + int(checked_x.shape[0])
    else:
        n_samples_seen = int(checked_x.shape[0])

    max_abs_array = np.asarray(max_abs, dtype=np.float64)
    scale = _handle_zeros_in_scale(max_abs_array, copy=True)
    return MaxAbsScalerState(
        scale=np.asarray(scale, dtype=np.float64),
        max_abs=max_abs_array,
        n_features_in=int(checked_x.shape[1]),
        n_samples_seen=n_samples_seen,
    )


@register_atom(witness_maxabs_scaler_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.ensure(lambda result: _maxabs_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_samples_seen == _row_count(X), "fresh fit sample count must match input rows")
def maxabs_scaler_fit(
    X: MatrixLike,
) -> MaxAbsScalerState:
    """Fit MaxAbsScaler state from a complete training matrix."""
    return maxabs_scaler_partial_fit(X, state=None)


@register_atom(witness_maxabs_scaler_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _maxabs_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "transformed output must preserve shape")
def maxabs_scaler_transform(
    X: MatrixLike,
    state: MaxAbsScalerState,
    copy: bool = True,
    clip: bool = False,
) -> MatrixLike:
    """Scale features by fitted maximum absolute values."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    if sp.issparse(checked_x):
        inplace_column_scale(checked_x, 1.0 / state.scale)
        if clip:
            np.clip(checked_x.data, -1.0, 1.0, out=checked_x.data)
    else:
        checked_x /= state.scale
        if clip:
            np.clip(checked_x, -1.0, 1.0, out=checked_x)
    return checked_x


@register_atom(witness_maxabs_scaler_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _maxabs_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "inverse transformed output must preserve shape")
def maxabs_scaler_inverse_transform(
    X: MatrixLike,
    state: MaxAbsScalerState,
    copy: bool = True,
) -> MatrixLike:
    """Undo scaling by fitted maximum absolute values."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    if sp.issparse(checked_x):
        inplace_column_scale(checked_x, state.scale)
    else:
        checked_x *= state.scale
    return checked_x


@register_atom(witness_maxabs_scale)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D array")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.ensure(lambda result, X: result.shape == X.shape, "maxabs scaled output must preserve shape")
def maxabs_scale(
    X: MatrixLike,
    *,
    axis: int = 0,
    copy: bool = True,
) -> MatrixLike:
    """Scale rows or columns by their maximum absolute values."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    original_ndim = checked_x.ndim
    if original_ndim == 1:
        checked_x = checked_x.reshape(checked_x.shape[0], 1)
    if copy:
        checked_x = checked_x.copy()

    if axis == 0:
        scaled = _maxabs_scale_axis0(checked_x)
    else:
        scaled = _maxabs_scale_axis0(checked_x.T).T

    if original_ndim == 1:
        scaled = scaled.ravel()
    return scaled


@register_atom(witness_minmax_scaler_partial_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda feature_range: _valid_feature_range(feature_range), "feature_range minimum must be smaller than maximum")
@icontract.require(lambda X, state: state is None or _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result: _minmax_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X, state: result.n_samples_seen == _row_count(X) + (0 if state is None else state.n_samples_seen), "sample count must accumulate")
def minmax_scaler_partial_fit(
    X: NDArray[np.float64],
    feature_range: tuple[float, float] = (0, 1),
    state: MinMaxScalerState | None = None,
) -> MinMaxScalerState:
    """Update MinMaxScaler state from one dense training batch."""
    if sp.issparse(X):
        raise TypeError("MinMaxScaler does not support sparse input. Consider using MaxAbsScaler instead.")
    checked_x = check_array(X, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan")
    data_min = np.nanmin(checked_x, axis=0)
    data_max = np.nanmax(checked_x, axis=0)

    if state is not None:
        if checked_x.shape[1] != state.n_features_in:
            raise ValueError("X feature count does not match fitted state")
        data_min = np.minimum(state.data_min, data_min)
        data_max = np.maximum(state.data_max, data_max)
        n_samples_seen = state.n_samples_seen + int(checked_x.shape[0])
        feature_range = state.feature_range
    else:
        n_samples_seen = int(checked_x.shape[0])

    data_range = data_max - data_min
    scale = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range, copy=True)
    min_ = feature_range[0] - data_min * scale
    return MinMaxScalerState(
        min_=np.asarray(min_, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        data_min=np.asarray(data_min, dtype=np.float64),
        data_max=np.asarray(data_max, dtype=np.float64),
        data_range=np.asarray(data_range, dtype=np.float64),
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        n_features_in=int(checked_x.shape[1]),
        n_samples_seen=n_samples_seen,
    )


@register_atom(witness_minmax_scaler_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda feature_range: _valid_feature_range(feature_range), "feature_range minimum must be smaller than maximum")
@icontract.ensure(lambda result: _minmax_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_samples_seen == _row_count(X), "fresh fit sample count must match input rows")
def minmax_scaler_fit(
    X: NDArray[np.float64],
    feature_range: tuple[float, float] = (0, 1),
) -> MinMaxScalerState:
    """Fit MinMaxScaler state from a complete dense training matrix."""
    return minmax_scaler_partial_fit(X, feature_range=feature_range, state=None)


@register_atom(witness_minmax_scaler_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _minmax_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "transformed output must preserve shape")
def minmax_scaler_transform(
    X: NDArray[np.float64],
    state: MinMaxScalerState,
    copy: bool = True,
    clip: bool = False,
) -> NDArray[np.float64]:
    """Scale dense features into the fitted MinMaxScaler feature range."""
    checked_x = check_array(
        X,
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    checked_x *= state.scale
    checked_x += state.min_
    if clip:
        np.clip(checked_x, state.feature_range[0], state.feature_range[1], out=checked_x)
    return checked_x


@register_atom(witness_minmax_scaler_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _minmax_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "inverse transformed output must preserve shape")
def minmax_scaler_inverse_transform(
    X: NDArray[np.float64],
    state: MinMaxScalerState,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Undo fitted MinMaxScaler feature-range scaling."""
    checked_x = check_array(
        X,
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    checked_x -= state.min_
    checked_x /= state.scale
    return checked_x


@register_atom(witness_minmax_scale)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D array")
@icontract.require(lambda feature_range: feature_range[0] < feature_range[1], "feature_range minimum must be smaller than maximum")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.ensure(lambda result, X: result.shape == X.shape, "minmax scaled output must preserve shape")
def minmax_scale(
    X: NDArray[np.float64],
    feature_range: tuple[float, float] = (0, 1),
    *,
    axis: int = 0,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Scale rows or columns into a fixed feature range."""
    checked_x = check_array(
        X,
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    original_ndim = checked_x.ndim
    if original_ndim == 1:
        checked_x = checked_x.reshape(checked_x.shape[0], 1)
    if copy:
        checked_x = checked_x.copy()

    work = checked_x if axis == 0 else checked_x.T
    data_min = np.nanmin(work, axis=0)
    data_max = np.nanmax(work, axis=0)
    data_range = data_max - data_min
    scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range, copy=True)
    work *= scale_
    work += feature_range[0] - data_min * scale_

    scaled = work if axis == 0 else work.T
    if original_ndim == 1:
        scaled = scaled.ravel()
    return scaled


def _valid_quantile_range(quantile_range: tuple[float, float]) -> bool:
    q_min, q_max = quantile_range
    return bool(0 <= q_min <= q_max <= 100)


def _robust_scale_axis0(
    X: MatrixLike,
    *,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: tuple[float, float],
    unit_variance: bool,
) -> MatrixLike:
    if sp.issparse(X):
        if with_centering:
            raise ValueError(
                "Cannot center sparse matrices: use `with_centering=False` instead. See docstring for motivation and alternatives."
            )
        if with_scaling:
            quantiles = []
            csc_x = X.tocsc()
            for feature_idx in range(csc_x.shape[1]):
                column_nnz_data = csc_x.data[csc_x.indptr[feature_idx] : csc_x.indptr[feature_idx + 1]]
                column_data = np.zeros(shape=csc_x.shape[0], dtype=csc_x.dtype)
                column_data[: len(column_nnz_data)] = column_nnz_data
                quantiles.append(np.nanpercentile(column_data, quantile_range))
            quantiles = np.transpose(quantiles)
            scale_ = _handle_zeros_in_scale(quantiles[1] - quantiles[0], copy=False)
            if unit_variance:
                q_min, q_max = quantile_range
                adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
                scale_ = scale_ / adjust
            inplace_column_scale(X, 1.0 / scale_)
    else:
        if with_centering:
            X -= np.nanmedian(X, axis=0)
        if with_scaling:
            quantiles = np.nanpercentile(X, quantile_range, axis=0)
            scale_ = _handle_zeros_in_scale(quantiles[1] - quantiles[0], copy=False)
            if unit_variance:
                q_min, q_max = quantile_range
                adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
                scale_ = scale_ / adjust
            X /= scale_
    return X


@register_atom(witness_robust_scale)
@icontract.require(lambda X: _is_1d_or_2d(X), "X must be a 1D or 2D array")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.require(lambda quantile_range: _valid_quantile_range(quantile_range), "quantile_range must satisfy 0 <= q_min <= q_max <= 100")
@icontract.ensure(lambda result, X: result.shape == X.shape, "robust scaled output must preserve shape")
def robust_scale(
    X: MatrixLike,
    *,
    axis: int = 0,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    copy: bool = True,
    unit_variance: bool = False,
) -> MatrixLike:
    """Center by median and scale by a quantile range along rows or columns."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    original_ndim = checked_x.ndim
    if original_ndim == 1:
        checked_x = checked_x.reshape(checked_x.shape[0], 1)
    if copy:
        checked_x = checked_x.copy()

    if axis == 0:
        scaled = _robust_scale_axis0(
            checked_x,
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        )
    else:
        scaled = _robust_scale_axis0(
            checked_x.T,
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        ).T

    if original_ndim == 1:
        scaled = scaled.ravel()
    return scaled
