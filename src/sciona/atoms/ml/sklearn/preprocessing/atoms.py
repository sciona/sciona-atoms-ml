"""Selected preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
import scipy.special as special
import scipy.stats as stats
from numpy.typing import NDArray
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale, _is_constant_feature
from sklearn.utils import check_array, check_random_state, resample
from sklearn.utils.extmath import _incremental_mean_and_var, row_norms
from sklearn.utils.sparsefuncs import incr_mean_variance_axis, inplace_column_scale, mean_variance_axis, min_max_axis
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2
from sklearn.utils.validation import FLOAT_DTYPES, _check_sample_weight

from sciona.ghost.registry import register_atom

from .state_models import (
    KernelCentererState,
    MaxAbsScalerState,
    MinMaxScalerState,
    RobustScalerState,
    StandardScalerState,
)
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
    witness_robust_scaler_fit,
    witness_robust_scaler_inverse_transform,
    witness_robust_scaler_transform,
    witness_scale,
    witness_standard_scaler_fit,
    witness_standard_scaler_inverse_transform,
    witness_standard_scaler_partial_fit,
    witness_standard_scaler_transform,
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


def _robust_state_valid(state: RobustScalerState) -> bool:
    center_ok = state.center is None or (state.center.ndim == 1 and state.center.shape[0] == state.n_features_in)
    scale_ok = state.scale is None or (state.scale.ndim == 1 and state.scale.shape[0] == state.n_features_in)
    return bool(
        center_ok
        and scale_ok
        and (state.center is not None) == state.with_centering
        and (state.scale is not None) == state.with_scaling
        and _valid_quantile_range(state.quantile_range)
    )


def _standard_state_valid(state: StandardScalerState) -> bool:
    shape = (state.n_features_in,)
    mean_ok = state.mean is None or (state.mean.ndim == 1 and state.mean.shape == shape)
    var_ok = state.var is None or (state.var.ndim == 1 and state.var.shape == shape)
    scale_ok = state.scale is None or (state.scale.ndim == 1 and state.scale.shape == shape)
    return bool(
        mean_ok
        and var_ok
        and scale_ok
        and state.n_samples_seen.ndim == 1
        and state.n_samples_seen.shape == shape
        and np.all(state.n_samples_seen >= 0)
        and (state.scale is not None) == state.with_std
        and (state.var is not None) == state.with_std
        and (state.mean is not None) == (state.with_mean or state.with_std)
    )


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, X: MatrixLike) -> bool:
    return sample_weight is None or tuple(sample_weight.shape) == (_row_count(X),)


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


@register_atom(witness_standard_scaler_partial_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda X, state: state is None or _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.require(lambda X, sample_weight: _sample_weight_valid(sample_weight, X), "sample_weight must have one value per sample")
@icontract.ensure(lambda result: _standard_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_features_in == _feature_count(X), "state feature count must match input columns")
def standard_scaler_partial_fit(
    X: MatrixLike,
    state: StandardScalerState | None = None,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    sample_weight: NDArray[np.float64] | None = None,
) -> StandardScalerState:
    """Update StandardScaler mean and variance state from one batch."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    n_features = int(checked_x.shape[1])
    checked_weight = None
    if sample_weight is not None:
        checked_weight = _check_sample_weight(sample_weight, checked_x, dtype=checked_x.dtype)

    if state is not None:
        if checked_x.shape[1] != state.n_features_in:
            raise ValueError("X feature count does not match fitted state")
        with_mean = state.with_mean
        with_std = state.with_std
        n_samples_seen = state.n_samples_seen.astype(np.float64, copy=True)
        mean = None if state.mean is None else state.mean.astype(np.float64, copy=True)
        var = None if state.var is None else state.var.astype(np.float64, copy=True)
    else:
        n_samples_seen = np.zeros(n_features, dtype=np.float64)
        mean = None
        var = None

    if sp.issparse(checked_x):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives."
            )
        sparse_constructor = sp.csr_matrix if checked_x.format == "csr" else sp.csc_matrix
        if with_std:
            if state is None:
                mean, var, n_samples_seen = mean_variance_axis(
                    checked_x,
                    axis=0,
                    weights=checked_weight,
                    return_sum_weights=True,
                )
            else:
                if mean is None or var is None:
                    raise ValueError("Existing StandardScaler state is missing mean or variance")
                mean, var, n_samples_seen = incr_mean_variance_axis(
                    checked_x,
                    axis=0,
                    last_mean=mean,
                    last_var=var,
                    last_n=n_samples_seen,
                    weights=checked_weight,
                )
            mean = mean.astype(np.float64, copy=False)
            var = var.astype(np.float64, copy=False)
        else:
            mean = None
            var = None
            weights = _check_sample_weight(sample_weight, checked_x)
            sum_weights_nan = weights @ sparse_constructor(
                (np.isnan(checked_x.data), checked_x.indices, checked_x.indptr),
                shape=checked_x.shape,
            )
            n_samples_seen += (np.sum(weights) - sum_weights_nan).astype(np.float64)
    else:
        dense_x = np.asarray(checked_x)
        if not with_mean and not with_std:
            mean = None
            var = None
            n_samples_seen += dense_x.shape[0] - np.isnan(dense_x).sum(axis=0)
        else:
            last_mean = np.zeros(n_features, dtype=np.float64) if mean is None else mean
            last_var = np.zeros(n_features, dtype=np.float64) if with_std and var is None else var
            mean, var, n_samples_seen = _incremental_mean_and_var(
                dense_x,
                last_mean,
                last_var,
                n_samples_seen,
                sample_weight=checked_weight,
            )

    n_samples_seen_array = np.asarray(n_samples_seen, dtype=np.float64)
    if with_std:
        if mean is None or var is None:
            raise ValueError("StandardScaler variance state is missing")
        constant_mask = _is_constant_feature(var, mean, n_samples_seen_array)
        scale = _handle_zeros_in_scale(np.sqrt(var), copy=False, constant_mask=constant_mask)
        scale_array = np.asarray(scale, dtype=np.float64)
        var_array = np.asarray(var, dtype=np.float64)
    else:
        scale_array = None
        var_array = None

    mean_array = None if mean is None else np.asarray(mean, dtype=np.float64)
    return StandardScalerState(
        mean=mean_array,
        var=var_array,
        scale=scale_array,
        n_samples_seen=n_samples_seen_array,
        with_mean=bool(with_mean),
        with_std=bool(with_std),
        n_features_in=n_features,
    )


@register_atom(witness_standard_scaler_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda X, sample_weight: _sample_weight_valid(sample_weight, X), "sample_weight must have one value per sample")
@icontract.ensure(lambda result: _standard_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_features_in == _feature_count(X), "state feature count must match input columns")
def standard_scaler_fit(
    X: MatrixLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    sample_weight: NDArray[np.float64] | None = None,
) -> StandardScalerState:
    """Fit StandardScaler mean and variance state from a complete batch."""
    return standard_scaler_partial_fit(
        X,
        state=None,
        with_mean=with_mean,
        with_std=with_std,
        sample_weight=sample_weight,
    )


@register_atom(witness_standard_scaler_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _standard_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "transformed output must preserve shape")
def standard_scaler_transform(
    X: MatrixLike,
    state: StandardScalerState,
    copy: bool = True,
) -> MatrixLike:
    """Center and scale data with fitted standard mean and variance statistics."""
    checked_x = check_array(
        X,
        accept_sparse="csr",
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    if sp.issparse(checked_x):
        if state.with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives."
            )
        if state.scale is not None:
            inplace_column_scale(checked_x, 1.0 / state.scale)
    else:
        if state.with_mean and state.mean is not None:
            checked_x -= state.mean.astype(checked_x.dtype, copy=False)
        if state.with_std and state.scale is not None:
            checked_x /= state.scale.astype(checked_x.dtype, copy=False)
    return checked_x


@register_atom(witness_standard_scaler_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _standard_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "inverse transformed output must preserve shape")
def standard_scaler_inverse_transform(
    X: MatrixLike,
    state: StandardScalerState,
    copy: bool = True,
) -> MatrixLike:
    """Undo fitted standard centering and variance scaling."""
    checked_x = check_array(
        X,
        accept_sparse="csr",
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    if sp.issparse(checked_x):
        if state.with_mean:
            raise ValueError(
                "Cannot uncenter sparse matrices: pass `with_mean=False` instead See docstring for motivation and alternatives."
            )
        if state.scale is not None:
            inplace_column_scale(checked_x, state.scale)
    else:
        if state.with_std and state.scale is not None:
            checked_x *= state.scale.astype(checked_x.dtype, copy=False)
        if state.with_mean and state.mean is not None:
            checked_x += state.mean.astype(checked_x.dtype, copy=False)
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


@register_atom(witness_robust_scaler_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda quantile_range: _valid_quantile_range(quantile_range), "quantile_range must satisfy 0 <= q_min <= q_max <= 100")
@icontract.ensure(lambda result: _robust_state_valid(result), "state arrays must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_features_in == _feature_count(X), "state feature count must match input columns")
def robust_scaler_fit(
    X: MatrixLike,
    *,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    unit_variance: bool = False,
) -> RobustScalerState:
    """Fit robust per-feature medians and quantile-range scales."""
    checked_x = check_array(
        X,
        accept_sparse="csc",
        dtype=FLOAT_DTYPES,
        ensure_all_finite="allow-nan",
    )
    q_min, q_max = quantile_range
    if not 0 <= q_min <= q_max <= 100:
        raise ValueError("Invalid quantile range: %s" % str(quantile_range))

    if with_centering:
        if sp.issparse(checked_x):
            raise ValueError(
                "Cannot center sparse matrices: use `with_centering=False` instead. See docstring for motivation and alternatives."
            )
        center = np.nanmedian(checked_x, axis=0)
    else:
        center = None

    if with_scaling:
        quantiles = []
        for feature_idx in range(checked_x.shape[1]):
            if sp.issparse(checked_x):
                column_nnz_data = checked_x.data[checked_x.indptr[feature_idx] : checked_x.indptr[feature_idx + 1]]
                column_data = np.zeros(shape=checked_x.shape[0], dtype=checked_x.dtype)
                column_data[: len(column_nnz_data)] = column_nnz_data
            else:
                column_data = checked_x[:, feature_idx]
            quantiles.append(np.nanpercentile(column_data, quantile_range))
        quantile_array = np.transpose(quantiles)
        scale = _handle_zeros_in_scale(quantile_array[1] - quantile_array[0], copy=False)
        if unit_variance:
            adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
            scale = scale / adjust
        scale_array = np.asarray(scale, dtype=np.float64)
    else:
        scale_array = None

    center_array = None if center is None else np.asarray(center, dtype=np.float64)
    return RobustScalerState(
        center=center_array,
        scale=scale_array,
        with_centering=bool(with_centering),
        with_scaling=bool(with_scaling),
        quantile_range=(float(q_min), float(q_max)),
        unit_variance=bool(unit_variance),
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_robust_scaler_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _robust_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "transformed output must preserve shape")
def robust_scaler_transform(
    X: MatrixLike,
    state: RobustScalerState,
    copy: bool = True,
) -> MatrixLike:
    """Center and scale data with fitted robust per-feature statistics."""
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
        if state.with_scaling and state.scale is not None:
            inplace_column_scale(checked_x, 1.0 / state.scale)
    else:
        if state.with_centering and state.center is not None:
            checked_x -= state.center
        if state.with_scaling and state.scale is not None:
            checked_x /= state.scale
    return checked_x


@register_atom(witness_robust_scaler_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _robust_state_valid(state), "state arrays must match fitted feature count")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: result.shape == X.shape, "inverse transformed output must preserve shape")
def robust_scaler_inverse_transform(
    X: MatrixLike,
    state: RobustScalerState,
    copy: bool = True,
) -> MatrixLike:
    """Undo fitted robust per-feature centering and scaling."""
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
        if state.with_scaling and state.scale is not None:
            inplace_column_scale(checked_x, state.scale)
    else:
        if state.with_scaling and state.scale is not None:
            checked_x *= state.scale
        if state.with_centering and state.center is not None:
            checked_x += state.center
    return checked_x


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


from .state_models import (
    LabelBinarizerState,
    LabelEncoderState,
    MultiLabelBinarizerState,
    PolynomialFeaturesState,
    PowerTransformerState,
    QuantileTransformerState,
)
from .witnesses import (
    witness_label_encoder_fit,
    witness_label_encoder_fit_transform,
    witness_label_encoder_inverse_transform,
    witness_label_encoder_transform,
)

LabelInput = NDArray[np.object_] | list[object] | tuple[object, ...]
EncodedLabelInput = NDArray[np.int_] | list[int] | tuple[int, ...]
LabelEncoderFitTransformResult = tuple[LabelEncoderState, NDArray[np.int_]]


def _label_input_is_vector_like(y: LabelInput | EncodedLabelInput) -> bool:
    array = np.asarray(y, dtype=object)
    return bool(array.ndim == 1 or (array.ndim == 2 and 1 in array.shape))


def _label_sample_count(y: LabelInput | EncodedLabelInput) -> int:
    array = np.asarray(y, dtype=object)
    return int(array.shape[0]) if array.ndim > 0 else 0


def _label_encoder_state_valid(state: LabelEncoderState) -> bool:
    return bool(state.classes.ndim == 1)


def _encoded_labels_valid(result: NDArray[np.int_], state: LabelEncoderState) -> bool:
    if result.ndim != 1:
        return False
    if result.size == 0:
        return True
    return bool(np.all(result >= 0) and np.all(result < state.classes.shape[0]))


def _label_encoder_fit_transform_valid(result: LabelEncoderFitTransformResult, y: LabelInput) -> bool:
    state, encoded = result
    return bool(
        _label_encoder_state_valid(state)
        and encoded.ndim == 1
        and encoded.shape == (_label_sample_count(y),)
        and _encoded_labels_valid(encoded, state)
    )


@register_atom(witness_label_encoder_fit)
@icontract.require(lambda y: _label_input_is_vector_like(y), "y must be 1D or a column vector")
@icontract.ensure(lambda result: _label_encoder_state_valid(result), "classes must be a one-dimensional array")
def label_encoder_fit(y: LabelInput) -> LabelEncoderState:
    """Learn sorted unique classes for one-dimensional target labels."""
    from sklearn.utils import column_or_1d
    from sklearn.utils._encode import _unique

    y_checked = column_or_1d(y, warn=True)
    classes = _unique(y_checked)
    return LabelEncoderState(classes=np.asarray(classes))


@register_atom(witness_label_encoder_fit_transform)
@icontract.require(lambda y: _label_input_is_vector_like(y), "y must be 1D or a column vector")
@icontract.ensure(lambda result, y: _label_encoder_fit_transform_valid(result, y), "encoded labels must match learned classes")
def label_encoder_fit_transform(y: LabelInput) -> LabelEncoderFitTransformResult:
    """Learn sorted unique classes and encode target labels in one pass."""
    from sklearn.utils import column_or_1d
    from sklearn.utils._encode import _unique

    y_checked = column_or_1d(y, warn=True)
    classes, encoded = _unique(y_checked, return_inverse=True)
    state = LabelEncoderState(classes=np.asarray(classes))
    return state, np.asarray(encoded, dtype=np.int_)


@register_atom(witness_label_encoder_transform)
@icontract.require(lambda y: _label_input_is_vector_like(y), "y must be 1D or a column vector")
@icontract.require(lambda state: _label_encoder_state_valid(state), "classes must be a one-dimensional array")
@icontract.ensure(lambda result, y: result.shape == (_label_sample_count(y),), "encoded labels must preserve sample count")
@icontract.ensure(lambda result, state: _encoded_labels_valid(result, state), "encoded labels must be valid class positions")
def label_encoder_transform(y: LabelInput, state: LabelEncoderState) -> NDArray[np.int_]:
    """Map target labels to integer positions in fitted class order."""
    from sklearn.utils import column_or_1d
    from sklearn.utils._encode import _encode
    from sklearn.utils.validation import _num_samples

    y_checked = column_or_1d(y, dtype=state.classes.dtype, warn=True)
    if _num_samples(y_checked) == 0:
        return np.asarray([], dtype=np.int_)
    return np.asarray(_encode(y_checked, uniques=state.classes), dtype=np.int_)


@register_atom(witness_label_encoder_inverse_transform)
@icontract.require(lambda y: _label_input_is_vector_like(y), "y must be 1D or a column vector")
@icontract.require(lambda state: _label_encoder_state_valid(state), "classes must be a one-dimensional array")
@icontract.ensure(lambda result, y: result.shape == (_label_sample_count(y),), "decoded labels must preserve sample count")
def label_encoder_inverse_transform(y: EncodedLabelInput, state: LabelEncoderState) -> NDArray[np.object_]:
    """Map integer class positions back to the original target labels."""
    from sklearn.utils import column_or_1d
    from sklearn.utils._array_api import device, get_namespace, xpx
    from sklearn.utils.validation import _num_samples

    xp, _ = get_namespace(y)
    y_checked = column_or_1d(y, warn=True)
    if _num_samples(y_checked) == 0:
        return np.asarray([])
    diff = xpx.setdiff1d(
        y_checked,
        xp.arange(state.classes.shape[0], device=device(y_checked)),
        xp=xp,
    )
    if diff.shape[0]:
        raise ValueError("y contains previously unseen labels: %s" % str(diff))
    decoded = xp.take(state.classes, xp.asarray(y_checked), axis=0)
    return np.asarray(decoded)


from .witnesses import witness_label_binarize

LabelBinarizeInput = LabelInput | MatrixLike
LabelBinarizeResult = NDArray[np.int_] | sp.csr_matrix


def _label_binarize_input_valid(y: LabelBinarizeInput) -> bool:
    if sp.issparse(y):
        return bool(y.ndim == 2)
    array = np.asarray(y, dtype=object)
    return bool(array.ndim in {1, 2})


def _classes_input_valid(classes: LabelInput) -> bool:
    return bool(np.asarray(classes, dtype=object).ndim == 1)


def _label_binarize_sample_count(y: LabelBinarizeInput) -> int:
    return int(y.shape[0]) if hasattr(y, "shape") else len(y)


def _label_binarize_shape_matches(result: LabelBinarizeResult, y: LabelBinarizeInput, classes: LabelInput) -> bool:
    n_classes = int(np.asarray(classes, dtype=object).shape[0])
    n_outputs = 1 if n_classes == 2 else n_classes
    return bool(result.shape == (_label_binarize_sample_count(y), n_outputs))


@register_atom(witness_label_binarize)
@icontract.require(lambda y: _label_binarize_input_valid(y), "y must be a 1D label vector or 2D multilabel indicator")
@icontract.require(lambda classes: _classes_input_valid(classes), "classes must be a 1D label vector")
@icontract.ensure(lambda result, y, classes: _label_binarize_shape_matches(result, y, classes), "binarized labels must have sklearn-compatible shape")
def label_binarize(
    y: LabelBinarizeInput,
    *,
    classes: LabelInput,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> LabelBinarizeResult:
    """Encode labels as one-vs-all indicator columns for fixed classes."""
    from sklearn.utils import check_array, column_or_1d
    from sklearn.utils.multiclass import type_of_target, unique_labels

    if not isinstance(y, list):
        y_checked = check_array(
            y,
            input_name="y",
            accept_sparse="csr",
            ensure_2d=False,
            dtype=None,
        )
    else:
        if len(y) == 0:
            raise ValueError("y has 0 samples: %r" % y)
        y_checked = y

    if neg_label >= pos_label:
        raise ValueError(f"neg_label={neg_label} must be strictly less than pos_label={pos_label}.")

    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError(
            "Sparse binarization is only supported with non zero pos_label and zero neg_label, got "
            f"pos_label={pos_label} and neg_label={neg_label}"
        )

    pos_switch = pos_label == 0
    effective_pos_label = -neg_label if pos_switch else pos_label

    y_type = type_of_target(y_checked)
    if "multioutput" in y_type:
        raise ValueError("Multioutput target data is not supported with label binarization")
    if y_type == "unknown":
        raise ValueError("The type of target data is not known")

    class_array = np.asarray(classes)
    n_samples = y_checked.shape[0] if hasattr(y_checked, "shape") else len(y_checked)
    n_classes = int(class_array.shape[0])
    int_dtype = y_checked.dtype if hasattr(y_checked, "dtype") and np.issubdtype(y_checked.dtype, np.integer) else int

    if y_type == "binary":
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            Y_single = np.zeros((n_samples, 1), dtype=int_dtype)
            Y_single += neg_label
            return Y_single
        if n_classes >= 3:
            y_type = "multiclass"

    sorted_class = np.sort(class_array)
    if y_type == "multilabel-indicator":
        y_n_classes = y_checked.shape[1] if hasattr(y_checked, "shape") else len(y_checked[0])
        if n_classes != y_n_classes:
            raise ValueError(
                "classes {0} mismatch with the labels {1} found in the data".format(
                    class_array,
                    unique_labels(y_checked),
                )
            )

    if y_type in ("binary", "multiclass"):
        y_vector = column_or_1d(y_checked)
        y_in_classes = np.isin(y_vector, class_array)
        y_seen = y_vector[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.concatenate(([0], np.cumsum(y_in_classes.astype(int), axis=0)))
        data = np.full_like(indices, effective_pos_label)
        Y: LabelBinarizeResult = sp.csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
        if not sparse_output:
            Y = np.asarray(Y.toarray())
    elif y_type == "multilabel-indicator":
        if sparse_output:
            Y = sp.csr_matrix(y_checked)
            if effective_pos_label != 1:
                Y.data = np.full_like(Y.data, effective_pos_label)
        else:
            dense_y = y_checked.toarray() if sp.issparse(y_checked) else y_checked
            Y = np.asarray(dense_y, copy=True)
            if effective_pos_label != 1:
                Y[Y != 0] = effective_pos_label
    else:
        raise ValueError("%s target data is not supported with label binarization" % y_type)

    if not sparse_output:
        dense_y = np.asarray(Y)
        if neg_label != 0:
            dense_y[dense_y == 0] = neg_label
        if pos_switch:
            dense_y[dense_y == effective_pos_label] = 0
        Y = dense_y.astype(int_dtype, copy=False)
    else:
        Y.data = Y.data.astype(int, copy=False)

    if np.any(class_array != sorted_class):
        indices = np.searchsorted(sorted_class, class_array)
        Y = Y[:, indices]

    if y_type == "binary":
        if sparse_output:
            Y = Y[:, [-1]]
        else:
            Y = np.reshape(np.asarray(Y)[:, -1], (-1, 1))

    return Y


from .witnesses import (
    witness_label_binarizer_fit,
    witness_label_binarizer_fit_transform,
    witness_label_binarizer_inverse_transform,
    witness_label_binarizer_transform,
)

LabelBinarizerFitTransformResult = tuple[LabelBinarizerState, LabelBinarizeResult]
LabelBinarizerInverseResult = NDArray[np.object_] | sp.csr_matrix


def _label_binarizer_y_type_valid(y_type: str) -> bool:
    return y_type in {"binary", "multiclass", "multilabel-indicator"}


def _label_binarizer_state_valid(state: LabelBinarizerState) -> bool:
    return bool(
        state.classes.ndim == 1
        and _label_binarizer_y_type_valid(state.y_type)
        and state.neg_label < state.pos_label
        and (not state.sparse_output or (state.pos_label != 0 and state.neg_label == 0))
    )


def _label_binarizer_fit_transform_valid(result: LabelBinarizerFitTransformResult, y: LabelBinarizeInput) -> bool:
    state, transformed = result
    return bool(
        _label_binarizer_state_valid(state)
        and _label_binarize_shape_matches(transformed, y, state.classes)
    )


def _label_binarizer_inverse_sample_count(result: LabelBinarizerInverseResult, Y: MatrixLike) -> bool:
    return bool(result.shape[0] == Y.shape[0])


@register_atom(witness_label_binarizer_fit)
@icontract.require(lambda y: _label_binarize_input_valid(y), "y must be a 1D label vector or 2D multilabel indicator")
@icontract.ensure(lambda result: _label_binarizer_state_valid(result), "state must contain fitted classes and valid labels")
def label_binarizer_fit(
    y: LabelBinarizeInput,
    *,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> LabelBinarizerState:
    """Learn classes, target type, and sparse-input mode for label binarization."""
    from sklearn.utils.multiclass import type_of_target, unique_labels
    from sklearn.utils.validation import _num_samples

    if neg_label >= pos_label:
        raise ValueError(f"neg_label={neg_label} must be strictly less than pos_label={pos_label}.")
    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError(
            "Sparse binarization is only supported with non zero pos_label and zero neg_label, got "
            f"pos_label={pos_label} and neg_label={neg_label}"
        )

    y_type = type_of_target(y, input_name="y")
    if "multioutput" in y_type:
        raise ValueError("Multioutput target data is not supported with label binarization")
    if _num_samples(y) == 0:
        raise ValueError("y has 0 samples: %r" % y)
    return LabelBinarizerState(
        classes=np.asarray(unique_labels(y)),
        y_type=y_type,
        sparse_input=bool(sp.issparse(y)),
        neg_label=int(neg_label),
        pos_label=int(pos_label),
        sparse_output=bool(sparse_output),
    )


@register_atom(witness_label_binarizer_fit_transform)
@icontract.require(lambda y: _label_binarize_input_valid(y), "y must be a 1D label vector or 2D multilabel indicator")
@icontract.ensure(lambda result, y: _label_binarizer_fit_transform_valid(result, y), "fit-transform output must match fitted classes")
def label_binarizer_fit_transform(
    y: LabelBinarizeInput,
    *,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> LabelBinarizerFitTransformResult:
    """Learn label-binarizer state and transform labels in one pass."""
    state = label_binarizer_fit(
        y,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    transformed = label_binarizer_transform(y, state)
    return state, transformed


@register_atom(witness_label_binarizer_transform)
@icontract.require(lambda y: _label_binarize_input_valid(y), "y must be a 1D label vector or 2D multilabel indicator")
@icontract.require(lambda state: _label_binarizer_state_valid(state), "state must contain fitted classes and valid labels")
@icontract.ensure(lambda result, y, state: _label_binarize_shape_matches(result, y, state.classes), "transformed labels must have fitted class shape")
def label_binarizer_transform(y: LabelBinarizeInput, state: LabelBinarizerState) -> LabelBinarizeResult:
    """Transform labels to one-vs-all columns with fitted binarizer state."""
    from sklearn.utils.multiclass import type_of_target

    y_is_multilabel = type_of_target(y).startswith("multilabel")
    if y_is_multilabel and not state.y_type.startswith("multilabel"):
        raise ValueError("The object was not fitted with multilabel input.")
    return label_binarize(
        y,
        classes=state.classes,
        pos_label=state.pos_label,
        neg_label=state.neg_label,
        sparse_output=state.sparse_output,
    )


@register_atom(witness_label_binarizer_inverse_transform)
@icontract.require(lambda Y: _is_2d(Y), "Y must be a 2D matrix")
@icontract.require(lambda state: _label_binarizer_state_valid(state), "state must contain fitted classes and valid labels")
@icontract.ensure(lambda result, Y: _label_binarizer_inverse_sample_count(result, Y), "inverse output must preserve sample count")
def label_binarizer_inverse_transform(
    Y: MatrixLike,
    state: LabelBinarizerState,
    threshold: float | None = None,
) -> LabelBinarizerInverseResult:
    """Map binary indicator scores back to labels using fitted binarizer state."""
    from sklearn.preprocessing._label import _inverse_binarize_multiclass, _inverse_binarize_thresholding

    effective_threshold = (state.pos_label + state.neg_label) / 2.0 if threshold is None else threshold
    if state.y_type == "multiclass":
        y_inv = _inverse_binarize_multiclass(Y, state.classes)
    else:
        y_inv = _inverse_binarize_thresholding(
            Y,
            state.y_type,
            state.classes,
            effective_threshold,
        )

    if state.sparse_input:
        return sp.csr_matrix(y_inv)
    if sp.issparse(y_inv):
        return y_inv.toarray()
    return np.asarray(y_inv)


from collections.abc import Iterable
import array
import itertools

from .witnesses import (
    witness_multi_label_binarizer_fit,
    witness_multi_label_binarizer_fit_transform,
    witness_multi_label_binarizer_inverse_transform,
    witness_multi_label_binarizer_transform,
)

LabelSet = Iterable[object]
MultiLabelInput = Iterable[LabelSet]
MultiLabelClassesInput = LabelInput | None
MultiLabelBinarizeResult = NDArray[np.int_] | sp.csr_matrix
MultiLabelBinarizerFitTransformResult = tuple[MultiLabelBinarizerState, MultiLabelBinarizeResult]
MultiLabelInverseResult = list[tuple[object, ...]]


def _multi_label_classes_valid(classes: MultiLabelClassesInput) -> bool:
    return classes is None or np.asarray(classes, dtype=object).ndim == 1


def _multi_label_state_valid(state: MultiLabelBinarizerState) -> bool:
    return bool(state.classes.ndim == 1)


def _multi_label_result_width_matches(result: MultiLabelBinarizeResult, state: MultiLabelBinarizerState) -> bool:
    return bool(result.ndim == 2 and result.shape[1] == state.classes.shape[0])


def _multi_label_fit_transform_valid(result: MultiLabelBinarizerFitTransformResult) -> bool:
    state, transformed = result
    return bool(_multi_label_state_valid(state) and _multi_label_result_width_matches(transformed, state))


def _multi_label_indicator_valid(yt: MatrixLike, state: MultiLabelBinarizerState) -> bool:
    return bool(getattr(yt, "ndim", 0) == 2 and yt.shape[1] == state.classes.shape[0])


def _materialize_label_sets(y: MultiLabelInput) -> list[LabelSet]:
    return list(y)


def _multi_label_classes_array(classes: Iterable[object]) -> NDArray[np.object_]:
    class_list = list(classes)
    dtype = int if all(isinstance(c, int) for c in class_list) else object
    result = np.empty(len(class_list), dtype=dtype)
    result[:] = class_list
    return np.asarray(result)


def _multi_label_transform_csr(y: MultiLabelInput, class_mapping: dict[object, int]) -> sp.csr_matrix:
    indices = array.array("i")
    indptr = array.array("i", [0])
    unknown: set[object] = set()
    for labels in y:
        index: set[int] = set()
        for label in labels:
            try:
                index.add(class_mapping[label])
            except KeyError:
                unknown.add(label)
        indices.extend(index)
        indptr.append(len(indices))
    if unknown:
        warnings.warn(
            "unknown class(es) {0} will be ignored".format(sorted(unknown, key=str)),
            stacklevel=2,
        )
    data = np.ones(len(indices), dtype=int)
    return sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(class_mapping)))


@register_atom(witness_multi_label_binarizer_fit)
@icontract.require(lambda classes: _multi_label_classes_valid(classes), "classes must be a 1D label vector when provided")
@icontract.ensure(lambda result: _multi_label_state_valid(result), "state classes must be one-dimensional")
def multi_label_binarizer_fit(
    y: MultiLabelInput,
    *,
    classes: MultiLabelClassesInput = None,
    sparse_output: bool = False,
) -> MultiLabelBinarizerState:
    """Learn a class ordering for iterable multilabel target sets."""
    if classes is None:
        class_values = sorted(set(itertools.chain.from_iterable(y)))
    else:
        class_values = list(classes)
        if len(set(class_values)) < len(class_values):
            raise ValueError(
                "The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer."
            )
    return MultiLabelBinarizerState(
        classes=_multi_label_classes_array(class_values),
        sparse_output=bool(sparse_output),
    )


@register_atom(witness_multi_label_binarizer_fit_transform)
@icontract.require(lambda classes: _multi_label_classes_valid(classes), "classes must be a 1D label vector when provided")
@icontract.ensure(lambda result: _multi_label_fit_transform_valid(result), "fit-transform output must match fitted classes")
def multi_label_binarizer_fit_transform(
    y: MultiLabelInput,
    *,
    classes: MultiLabelClassesInput = None,
    sparse_output: bool = False,
) -> MultiLabelBinarizerFitTransformResult:
    """Learn multilabel classes and transform label sets to indicators."""
    y_list = _materialize_label_sets(y)
    state = multi_label_binarizer_fit(y_list, classes=classes, sparse_output=sparse_output)
    transformed = multi_label_binarizer_transform(y_list, state)
    return state, transformed


@register_atom(witness_multi_label_binarizer_transform)
@icontract.require(lambda state: _multi_label_state_valid(state), "state classes must be one-dimensional")
@icontract.ensure(lambda result, state: _multi_label_result_width_matches(result, state), "indicator matrix width must match fitted classes")
def multi_label_binarizer_transform(
    y: MultiLabelInput,
    state: MultiLabelBinarizerState,
) -> MultiLabelBinarizeResult:
    """Transform iterable label sets to a binary indicator matrix."""
    class_mapping = dict(zip(state.classes, range(len(state.classes))))
    indicator = _multi_label_transform_csr(y, class_mapping)
    if state.sparse_output:
        return indicator
    return indicator.toarray()


@register_atom(witness_multi_label_binarizer_inverse_transform)
@icontract.require(lambda yt, state: _multi_label_indicator_valid(yt, state), "indicator width must match fitted classes")
@icontract.ensure(lambda result, yt: len(result) == yt.shape[0], "inverse output must preserve sample count")
def multi_label_binarizer_inverse_transform(
    yt: MatrixLike,
    state: MultiLabelBinarizerState,
) -> MultiLabelInverseResult:
    """Convert a multilabel indicator matrix back to tuples of class labels."""
    if yt.shape[1] != len(state.classes):
        raise ValueError("Expected indicator for {0} classes, but got {1}".format(len(state.classes), yt.shape[1]))

    if sp.issparse(yt):
        yt_csr = yt.tocsr()
        if len(yt_csr.data) != 0 and len(np.setdiff1d(yt_csr.data, [0, 1])) > 0:
            raise ValueError("Expected only 0s and 1s in label indicator.")
        return [
            tuple(state.classes.take(yt_csr.indices[start:end]))
            for start, end in zip(yt_csr.indptr[:-1], yt_csr.indptr[1:])
        ]

    dense_yt = np.asarray(yt)
    unexpected = np.setdiff1d(dense_yt, [0, 1])
    if len(unexpected) > 0:
        raise ValueError("Expected only 0s and 1s in label indicator. Also got {0}".format(unexpected))
    return [tuple(state.classes.compress(indicators)) for indicators in dense_yt]


import math

from .witnesses import (
    witness_polynomial_features_fit,
    witness_polynomial_features_fit_transform,
    witness_polynomial_features_transform,
)

PolynomialDegree = int | tuple[int, int]
PolynomialFeaturesResult = MatrixLike
PolynomialFeaturesFitTransformResult = tuple[PolynomialFeaturesState, PolynomialFeaturesResult]


def _polynomial_order_valid(order: str) -> bool:
    return order in {"C", "F"}


def _polynomial_degree_bounds(degree: PolynomialDegree, include_bias: bool) -> tuple[int, int]:
    if isinstance(degree, int):
        if degree < 0:
            raise ValueError("degree must be a non-negative int or tuple (min_degree, max_degree), got {0}.".format(degree))
        if degree == 0 and not include_bias:
            raise ValueError("Setting degree to zero and include_bias to False would result in an empty output array.")
        return 0, int(degree)
    if len(degree) != 2:
        raise ValueError("degree must be a non-negative int or tuple (min_degree, max_degree), got {0}.".format(degree))
    min_degree, max_degree = degree
    if not (
        isinstance(min_degree, int)
        and isinstance(max_degree, int)
        and min_degree >= 0
        and min_degree <= max_degree
    ):
        raise ValueError(
            "degree=(min_degree, max_degree) must be non-negative integers that fulfil min_degree <= max_degree, got "
            f"{degree}."
        )
    if max_degree == 0 and not include_bias:
        raise ValueError("Setting both min_degree and max_degree to zero and include_bias to False would result in an empty output array.")
    return int(min_degree), int(max_degree)


def _polynomial_output_count(
    n_features: int,
    min_degree: int,
    max_degree: int,
    interaction_only: bool,
    include_bias: bool,
) -> int:
    if interaction_only:
        total = sum(
            math.comb(n_features, degree)
            for degree in range(max(1, min_degree), min(max_degree, n_features) + 1)
        )
    else:
        total = math.comb(n_features + max_degree, max_degree) - 1
        if min_degree > 0:
            previous_degree = min_degree - 1
            total -= math.comb(n_features + previous_degree, previous_degree) - 1
    if include_bias:
        total += 1
    return int(total)


def _polynomial_combinations(
    n_features: int,
    min_degree: int,
    max_degree: int,
    interaction_only: bool,
    include_bias: bool,
) -> list[tuple[int, ...]]:
    comb = itertools.combinations if interaction_only else itertools.combinations_with_replacement
    combinations = [
        item
        for degree in range(max(1, min_degree), max_degree + 1)
        for item in comb(range(n_features), degree)
    ]
    if include_bias:
        combinations.insert(0, ())
    return combinations


def _polynomial_powers(
    n_features: int,
    min_degree: int,
    max_degree: int,
    interaction_only: bool,
    include_bias: bool,
) -> NDArray[np.int_]:
    combinations = _polynomial_combinations(
        n_features=n_features,
        min_degree=min_degree,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    if not combinations:
        return np.zeros((0, n_features), dtype=np.int_)
    return np.asarray([np.bincount(item, minlength=n_features) for item in combinations], dtype=np.int_)


def _polynomial_state_valid(state: PolynomialFeaturesState) -> bool:
    return bool(
        state.powers.ndim == 2
        and state.powers.shape == (state.n_output_features, state.n_features_in)
        and state.min_degree >= 0
        and state.min_degree <= state.max_degree
        and state.order in {"C", "F"}
    )


def _polynomial_transform_shape_matches(result: PolynomialFeaturesResult, X: MatrixLike, state: PolynomialFeaturesState) -> bool:
    return bool(result.shape == (X.shape[0], state.n_output_features))


def _polynomial_fit_transform_valid(result: PolynomialFeaturesFitTransformResult, X: MatrixLike) -> bool:
    state, transformed = result
    return bool(_polynomial_state_valid(state) and _polynomial_transform_shape_matches(transformed, X, state))


@register_atom(witness_polynomial_features_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda order: _polynomial_order_valid(order), "order must be 'C' or 'F'")
@icontract.ensure(lambda result: _polynomial_state_valid(result), "polynomial state must contain one power row per output feature")
@icontract.ensure(lambda result, X: result.n_features_in == X.shape[1], "state feature count must match input columns")
def polynomial_features_fit(
    X: MatrixLike,
    degree: PolynomialDegree = 2,
    *,
    interaction_only: bool = False,
    include_bias: bool = True,
    order: str = "C",
) -> PolynomialFeaturesState:
    """Learn polynomial expansion powers for a fitted feature count."""
    checked_x = check_array(X, accept_sparse=True, dtype=FLOAT_DTYPES)
    n_features = int(checked_x.shape[1])
    min_degree, max_degree = _polynomial_degree_bounds(degree, include_bias)
    n_output_features = _polynomial_output_count(
        n_features=n_features,
        min_degree=min_degree,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    if n_output_features > np.iinfo(np.intp).max:
        raise ValueError("The output that would result from the current configuration would have too many features.")
    powers = _polynomial_powers(
        n_features=n_features,
        min_degree=min_degree,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    return PolynomialFeaturesState(
        powers=powers,
        n_features_in=n_features,
        n_output_features=n_output_features,
        min_degree=min_degree,
        max_degree=max_degree,
        interaction_only=bool(interaction_only),
        include_bias=bool(include_bias),
        order=order,
    )


@register_atom(witness_polynomial_features_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _polynomial_state_valid(state), "polynomial state must contain one power row per output feature")
@icontract.require(lambda X, state: X.shape[1] == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _polynomial_transform_shape_matches(result, X, state), "polynomial output shape must match fitted expansion")
def polynomial_features_transform(
    X: MatrixLike,
    state: PolynomialFeaturesState,
) -> PolynomialFeaturesResult:
    """Expand rows to polynomial and interaction feature columns."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=FLOAT_DTYPES)
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")

    if sp.issparse(checked_x):
        columns = []
        sparse_format = "csr" if checked_x.format == "csr" else "csc"
        for power in state.powers:
            if int(np.sum(power)) == 0:
                columns.append(sp.csr_matrix(np.ones((checked_x.shape[0], 1), dtype=checked_x.dtype)))
                continue
            out_col = None
            for feature_idx, exponent in enumerate(power):
                if exponent == 0:
                    continue
                feature_col = checked_x[:, [feature_idx]]
                powered_col = feature_col.copy()
                for _ in range(int(exponent) - 1):
                    powered_col = powered_col.multiply(feature_col)
                out_col = powered_col if out_col is None else out_col.multiply(powered_col)
            if out_col is None:
                out_col = sp.csr_matrix((checked_x.shape[0], 1), dtype=checked_x.dtype)
            columns.append(out_col)
        if not columns:
            return sp.csr_matrix((checked_x.shape[0], 0), dtype=checked_x.dtype)
        return sp.hstack(columns, dtype=checked_x.dtype, format=sparse_format)

    dense_x = np.asarray(checked_x)
    columns_dense = []
    for power in state.powers:
        if int(np.sum(power)) == 0:
            columns_dense.append(np.ones(dense_x.shape[0], dtype=dense_x.dtype))
            continue
        column = np.ones(dense_x.shape[0], dtype=dense_x.dtype)
        for feature_idx, exponent in enumerate(power):
            if exponent:
                column *= dense_x[:, feature_idx] ** int(exponent)
        columns_dense.append(column)
    if not columns_dense:
        result = np.empty((dense_x.shape[0], 0), dtype=dense_x.dtype)
    else:
        result = np.vstack(columns_dense).T
    return np.asfortranarray(result) if state.order == "F" else np.ascontiguousarray(result)


@register_atom(witness_polynomial_features_fit_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda order: _polynomial_order_valid(order), "order must be 'C' or 'F'")
@icontract.ensure(lambda result, X: _polynomial_fit_transform_valid(result, X), "fit-transform output must match learned polynomial state")
def polynomial_features_fit_transform(
    X: MatrixLike,
    degree: PolynomialDegree = 2,
    *,
    interaction_only: bool = False,
    include_bias: bool = True,
    order: str = "C",
) -> PolynomialFeaturesFitTransformResult:
    """Learn polynomial powers and expand rows in one pass."""
    state = polynomial_features_fit(
        X,
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
        order=order,
    )
    return state, polynomial_features_transform(X, state)


from .witnesses import (
    witness_power_transform,
    witness_power_transformer_fit,
    witness_power_transformer_fit_transform,
    witness_power_transformer_inverse_transform,
    witness_power_transformer_transform,
)

PowerTransformerFitTransformResult = tuple[PowerTransformerState, NDArray[np.float64]]


def _power_method_valid(method: str) -> bool:
    return method in {"yeo-johnson", "box-cox"}


def _power_state_valid(state: PowerTransformerState) -> bool:
    shape = (state.n_features_in,)
    return bool(
        _power_method_valid(state.method)
        and state.lambdas.ndim == 1
        and state.lambdas.shape == shape
        and (state.mean is None or (state.mean.ndim == 1 and state.mean.shape == shape))
        and (state.scale is None or (state.scale.ndim == 1 and state.scale.shape == shape))
        and (state.mean is not None) == state.standardize
        and (state.scale is not None) == state.standardize
    )


def _power_transform_shape_matches(result: NDArray[np.float64], X: MatrixLike) -> bool:
    return bool(result.shape == X.shape)


def _power_fit_transform_valid(result: PowerTransformerFitTransformResult, X: MatrixLike) -> bool:
    state, transformed = result
    return bool(_power_state_valid(state) and _power_transform_shape_matches(transformed, X))


def _yeo_johnson_transform_array(x: NDArray[np.float64], lmbda: float) -> NDArray[np.float64]:
    out = np.zeros_like(x)
    pos = x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:
        out[~pos] = -np.log1p(-x[~pos])
    return out


def _yeo_johnson_inverse_array(x: NDArray[np.float64], lmbda: float) -> NDArray[np.float64]:
    out = np.zeros_like(x)
    pos = x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.exp(x[pos]) - 1
    else:
        out[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
    else:
        out[~pos] = 1 - np.exp(-x[~pos])
    return out


def _power_transform_columns(X: NDArray[np.float64], state: PowerTransformerState) -> NDArray[np.float64]:
    transformed = X.copy()
    for i, lmbda in enumerate(state.lambdas):
        with np.errstate(invalid="ignore"):
            if state.method == "box-cox":
                transformed[:, i] = stats.boxcox(transformed[:, i], lmbda=float(lmbda))
            else:
                transformed[:, i] = _yeo_johnson_transform_array(transformed[:, i], float(lmbda))
    return transformed


def _power_inverse_columns(X: NDArray[np.float64], state: PowerTransformerState) -> NDArray[np.float64]:
    restored = X.copy()
    for i, lmbda in enumerate(state.lambdas):
        with warnings.catch_warnings(record=True) as captured_warnings:
            with np.errstate(invalid="warn"):
                if state.method == "box-cox":
                    restored[:, i] = special.inv_boxcox(restored[:, i], float(lmbda))
                else:
                    restored[:, i] = _yeo_johnson_inverse_array(restored[:, i], float(lmbda))
        if any("invalid value encountered in power" in str(w.message) for w in captured_warnings):
            warnings.warn(
                f"Some values in column {i} of the inverse-transformed data are NaN. This may be caused by numerical issues in the transformation process, e.g. extremely skewed data. Consider inspecting the input data or preprocessing it before applying the transformation.",
                UserWarning,
                stacklevel=2,
            )
    return restored


@register_atom(witness_power_transformer_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda method: _power_method_valid(method), "method must be 'yeo-johnson' or 'box-cox'")
@icontract.ensure(lambda result: _power_state_valid(result), "power-transform state must match fitted feature count")
@icontract.ensure(lambda result, X: result.n_features_in == X.shape[1], "state feature count must match input columns")
def power_transformer_fit(
    X: MatrixLike,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
) -> PowerTransformerState:
    """Estimate per-feature power-transform lambdas and optional scaling state."""
    checked_x = check_array(
        X,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        ensure_all_finite="allow-nan",
    )
    if method == "box-cox":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            if np.nanmin(checked_x) <= 0:
                raise ValueError("The Box-Cox transformation can only be applied to strictly positive data")

    n_samples = checked_x.shape[0]
    mean = np.mean(checked_x, axis=0, dtype=np.float64)
    var = np.var(checked_x, axis=0, dtype=np.float64)
    lambdas = np.empty(checked_x.shape[1], dtype=np.float64)
    transformed = checked_x.astype(np.float64, copy=True)

    with np.errstate(invalid="ignore"):
        for i, col in enumerate(checked_x.T):
            if method == "yeo-johnson" and _is_constant_feature(var[i], mean[i], n_samples):
                lambdas[i] = 1.0
            elif method == "box-cox":
                mask = np.isnan(col)
                if np.all(mask):
                    raise ValueError("Column must not be all nan.")
                _, lambdas[i] = stats.boxcox(col[~mask], lmbda=None)
            else:
                clean_col = col[~np.isnan(col)]
                _, lambdas[i] = stats.yeojohnson(clean_col, lmbda=None)
            if standardize:
                if method == "box-cox":
                    transformed[:, i] = stats.boxcox(transformed[:, i], lmbda=float(lambdas[i]))
                else:
                    transformed[:, i] = _yeo_johnson_transform_array(transformed[:, i], float(lambdas[i]))

    if standardize:
        scaler_state = standard_scaler_fit(transformed, with_mean=True, with_std=True)
        scale_mean = scaler_state.mean
        scale_scale = scaler_state.scale
    else:
        scale_mean = None
        scale_scale = None
    return PowerTransformerState(
        lambdas=lambdas,
        method=method,
        standardize=bool(standardize),
        mean=scale_mean,
        scale=scale_scale,
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_power_transformer_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _power_state_valid(state), "power-transform state must match fitted feature count")
@icontract.require(lambda X, state: X.shape[1] == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _power_transform_shape_matches(result, X), "transformed output must preserve shape")
def power_transformer_transform(
    X: MatrixLike,
    state: PowerTransformerState,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Apply fitted power-transform lambdas and optional standardization."""
    checked_x = check_array(
        X,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        copy=copy,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("Input data has a different number of features than fitting data.")
    if state.method == "box-cox":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            if np.nanmin(checked_x) <= 0:
                raise ValueError("The Box-Cox transformation can only be applied to strictly positive data")
    transformed = _power_transform_columns(checked_x.astype(np.float64, copy=True), state)
    if state.standardize:
        if state.mean is None or state.scale is None:
            raise ValueError("PowerTransformer standardization state is missing")
        transformed -= state.mean
        transformed /= state.scale
    return transformed


@register_atom(witness_power_transformer_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _power_state_valid(state), "power-transform state must match fitted feature count")
@icontract.require(lambda X, state: X.shape[1] == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _power_transform_shape_matches(result, X), "inverse transformed output must preserve shape")
def power_transformer_inverse_transform(
    X: MatrixLike,
    state: PowerTransformerState,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Undo fitted power transformation and optional standardization."""
    checked_x = check_array(
        X,
        dtype=FLOAT_DTYPES,
        force_writeable=True,
        copy=copy,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("Input data has a different number of features than fitting data.")
    restored = checked_x.astype(np.float64, copy=True)
    if state.standardize:
        if state.mean is None or state.scale is None:
            raise ValueError("PowerTransformer standardization state is missing")
        restored *= state.scale
        restored += state.mean
    return _power_inverse_columns(restored, state)


@register_atom(witness_power_transformer_fit_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda method: _power_method_valid(method), "method must be 'yeo-johnson' or 'box-cox'")
@icontract.ensure(lambda result, X: _power_fit_transform_valid(result, X), "fit-transform output must match fitted power-transform state")
def power_transformer_fit_transform(
    X: MatrixLike,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
    copy: bool = True,
) -> PowerTransformerFitTransformResult:
    """Estimate power-transform lambdas and transform the same data."""
    del copy
    state = power_transformer_fit(X, method=method, standardize=standardize)
    return state, power_transformer_transform(X, state, copy=True)


@register_atom(witness_power_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda method: _power_method_valid(method), "method must be 'yeo-johnson' or 'box-cox'")
@icontract.ensure(lambda result, X: _power_transform_shape_matches(result, X), "power transformed output must preserve shape")
def power_transform(
    X: MatrixLike,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Fit per-feature power-transform lambdas and transform the same data."""
    _, transformed = power_transformer_fit_transform(
        X,
        method=method,
        standardize=standardize,
        copy=copy,
    )
    return transformed


from .witnesses import (
    witness_quantile_transform,
    witness_quantile_transformer_fit,
    witness_quantile_transformer_fit_transform,
    witness_quantile_transformer_inverse_transform,
    witness_quantile_transformer_transform,
)

RandomStateLike = int | np.random.RandomState | None
QuantileTransformerFitTransformResult = tuple[QuantileTransformerState, MatrixLike]


def _quantile_distribution_valid(output_distribution: str) -> bool:
    return output_distribution in {"uniform", "normal"}


def _subsample_valid(subsample: int | None) -> bool:
    return subsample is None or subsample >= 1


def _quantile_state_valid(state: QuantileTransformerState) -> bool:
    expected_shape = (state.n_quantiles, state.n_features_in)
    return bool(
        state.n_quantiles >= 1
        and _quantile_distribution_valid(state.output_distribution)
        and state.quantiles.ndim == 2
        and state.quantiles.shape == expected_shape
        and state.references.ndim == 1
        and state.references.shape == (state.n_quantiles,)
    )


def _quantile_transform_shape_matches(result: MatrixLike, X: MatrixLike) -> bool:
    return bool(result.shape == X.shape)


def _quantile_fit_transform_valid(result: QuantileTransformerFitTransformResult, X: MatrixLike) -> bool:
    state, transformed = result
    return bool(_quantile_state_valid(state) and _quantile_transform_shape_matches(transformed, X))


def _quantile_check_inputs(
    X: MatrixLike,
    *,
    in_fit: bool,
    ignore_implicit_zeros: bool,
    accept_sparse_negative: bool = False,
    copy: bool = False,
) -> MatrixLike:
    checked_x = check_array(
        X,
        accept_sparse="csc",
        copy=copy,
        dtype=FLOAT_DTYPES,
        force_writeable=True if not in_fit else None,
        ensure_all_finite="allow-nan",
    )
    with np.errstate(invalid="ignore"):
        if (
            not accept_sparse_negative
            and not ignore_implicit_zeros
            and sp.issparse(checked_x)
            and np.any(checked_x.data < 0)
        ):
            raise ValueError("QuantileTransformer only accepts non-negative sparse matrices.")
    return checked_x


def _quantile_dense_fit(
    X: NDArray[np.float64],
    references: NDArray[np.float64],
    *,
    ignore_implicit_zeros: bool,
    subsample: int | None,
    random_state: np.random.RandomState,
) -> NDArray[np.float64]:
    if ignore_implicit_zeros:
        warnings.warn(
            "'ignore_implicit_zeros' takes effect only with sparse matrix. This parameter has no effect.",
            UserWarning,
            stacklevel=2,
        )
    fitting_x = X
    if subsample is not None and subsample < X.shape[0]:
        fitting_x = resample(X, replace=False, n_samples=subsample, random_state=random_state)
    return np.asarray(np.nanpercentile(fitting_x, references * 100, axis=0), dtype=np.float64)


def _quantile_sparse_fit(
    X: sp.spmatrix,
    references: NDArray[np.float64],
    *,
    ignore_implicit_zeros: bool,
    subsample: int | None,
    random_state: np.random.RandomState,
) -> NDArray[np.float64]:
    csc_x = X.tocsc()
    n_samples, n_features = csc_x.shape
    quantiles: list[NDArray[np.float64] | list[float]] = []
    for feature_idx in range(n_features):
        column_nnz_data = csc_x.data[csc_x.indptr[feature_idx] : csc_x.indptr[feature_idx + 1]]
        if subsample is not None and len(column_nnz_data) > subsample:
            column_subsample = subsample * len(column_nnz_data) // n_samples
            if ignore_implicit_zeros:
                column_data = np.zeros(shape=column_subsample, dtype=csc_x.dtype)
            else:
                column_data = np.zeros(shape=subsample, dtype=csc_x.dtype)
            column_data[:column_subsample] = random_state.choice(
                column_nnz_data,
                size=column_subsample,
                replace=False,
            )
        else:
            if ignore_implicit_zeros:
                column_data = np.zeros(shape=len(column_nnz_data), dtype=csc_x.dtype)
            else:
                column_data = np.zeros(shape=n_samples, dtype=csc_x.dtype)
            column_data[: len(column_nnz_data)] = column_nnz_data

        if not column_data.size:
            quantiles.append([0.0] * len(references))
        else:
            quantiles.append(np.nanpercentile(column_data, references * 100))
    return np.asarray(quantiles, dtype=np.float64).T


def _quantile_transform_col(
    X_col: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    references: NDArray[np.float64],
    output_distribution: str,
    *,
    inverse: bool,
) -> NDArray[np.float64]:
    if not inverse:
        lower_bound_x = quantiles[0]
        upper_bound_x = quantiles[-1]
        lower_bound_y = 0.0
        upper_bound_y = 1.0
    else:
        lower_bound_x = 0.0
        upper_bound_x = 1.0
        lower_bound_y = quantiles[0]
        upper_bound_y = quantiles[-1]
        with np.errstate(invalid="ignore"):
            if output_distribution == "normal":
                X_col = stats.norm.cdf(X_col)

    with np.errstate(invalid="ignore"):
        if output_distribution == "normal":
            lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
            upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
        else:
            lower_bounds_idx = X_col == lower_bound_x
            upper_bounds_idx = X_col == upper_bound_x

    isfinite_mask = ~np.isnan(X_col)
    x_col_finite = X_col[isfinite_mask]
    if not inverse:
        X_col[isfinite_mask] = 0.5 * (
            np.interp(x_col_finite, quantiles, references)
            - np.interp(-x_col_finite, -quantiles[::-1], -references[::-1])
        )
    else:
        X_col[isfinite_mask] = np.interp(x_col_finite, references, quantiles)

    X_col[upper_bounds_idx] = upper_bound_y
    X_col[lower_bounds_idx] = lower_bound_y
    if not inverse:
        with np.errstate(invalid="ignore"):
            if output_distribution == "normal":
                X_col = stats.norm.ppf(X_col)
                clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                X_col = np.clip(X_col, clip_min, clip_max)
    return X_col


def _quantile_transform_matrix(
    X: MatrixLike,
    state: QuantileTransformerState,
    *,
    inverse: bool,
) -> MatrixLike:
    if sp.issparse(X):
        transformed = X.tocsc(copy=True)
        for feature_idx in range(transformed.shape[1]):
            column_slice = slice(transformed.indptr[feature_idx], transformed.indptr[feature_idx + 1])
            transformed.data[column_slice] = _quantile_transform_col(
                np.asarray(transformed.data[column_slice], dtype=np.float64),
                state.quantiles[:, feature_idx],
                state.references,
                state.output_distribution,
                inverse=inverse,
            )
        return transformed

    dense = np.asarray(X, dtype=np.float64).copy()
    for feature_idx in range(dense.shape[1]):
        dense[:, feature_idx] = _quantile_transform_col(
            dense[:, feature_idx],
            state.quantiles[:, feature_idx],
            state.references,
            state.output_distribution,
            inverse=inverse,
        )
    return dense


@register_atom(witness_quantile_transformer_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda n_quantiles: n_quantiles >= 1, "n_quantiles must be at least one")
@icontract.require(lambda output_distribution: _quantile_distribution_valid(output_distribution), "output_distribution must be 'uniform' or 'normal'")
@icontract.require(lambda subsample: _subsample_valid(subsample), "subsample must be at least one or None")
@icontract.ensure(lambda result: _quantile_state_valid(result), "quantile state must contain fitted feature quantiles")
@icontract.ensure(lambda result, X: result.n_features_in == X.shape[1], "state feature count must match input columns")
def quantile_transformer_fit(
    X: MatrixLike,
    *,
    n_quantiles: int = 1000,
    output_distribution: str = "uniform",
    ignore_implicit_zeros: bool = False,
    subsample: int | None = 10_000,
    random_state: RandomStateLike = None,
) -> QuantileTransformerState:
    """Estimate empirical per-feature quantiles for distribution mapping."""
    if subsample is not None and n_quantiles > subsample:
        raise ValueError(
            "The number of quantiles cannot be greater than the number of samples used. "
            f"Got {n_quantiles} quantiles and {subsample} samples."
        )
    checked_x = _quantile_check_inputs(
        X,
        in_fit=True,
        ignore_implicit_zeros=ignore_implicit_zeros,
        copy=False,
    )
    n_samples = int(checked_x.shape[0])
    if n_quantiles > n_samples:
        warnings.warn(
            "n_quantiles ({0}) is greater than the total number of samples ({1}). n_quantiles is set to n_samples.".format(
                n_quantiles,
                n_samples,
            ),
            UserWarning,
            stacklevel=2,
        )
    n_quantiles_fit = max(1, min(n_quantiles, n_samples))
    references = np.linspace(0, 1, n_quantiles_fit, endpoint=True, dtype=np.float64)
    rng = check_random_state(random_state)
    if sp.issparse(checked_x):
        quantiles = _quantile_sparse_fit(
            checked_x,
            references,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=rng,
        )
    else:
        quantiles = _quantile_dense_fit(
            np.asarray(checked_x, dtype=np.float64),
            references,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=rng,
        )
    return QuantileTransformerState(
        quantiles=quantiles,
        references=references,
        n_quantiles=int(n_quantiles_fit),
        output_distribution=output_distribution,
        ignore_implicit_zeros=bool(ignore_implicit_zeros),
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_quantile_transformer_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _quantile_state_valid(state), "quantile state must contain fitted feature quantiles")
@icontract.require(lambda X, state: X.shape[1] == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _quantile_transform_shape_matches(result, X), "quantile transform output must preserve shape")
def quantile_transformer_transform(
    X: MatrixLike,
    state: QuantileTransformerState,
    copy: bool = True,
) -> MatrixLike:
    """Map features through fitted empirical quantiles to uniform or normal marginals."""
    checked_x = _quantile_check_inputs(
        X,
        in_fit=False,
        ignore_implicit_zeros=state.ignore_implicit_zeros,
        copy=copy,
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("Input data has a different number of features than fitting data.")
    return _quantile_transform_matrix(checked_x, state, inverse=False)


@register_atom(witness_quantile_transformer_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: _quantile_state_valid(state), "quantile state must contain fitted feature quantiles")
@icontract.require(lambda X, state: X.shape[1] == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _quantile_transform_shape_matches(result, X), "inverse quantile transform output must preserve shape")
def quantile_transformer_inverse_transform(
    X: MatrixLike,
    state: QuantileTransformerState,
    copy: bool = True,
) -> MatrixLike:
    """Map fitted uniform or normal marginal values back to original feature space."""
    checked_x = _quantile_check_inputs(
        X,
        in_fit=False,
        ignore_implicit_zeros=state.ignore_implicit_zeros,
        accept_sparse_negative=True,
        copy=copy,
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("Input data has a different number of features than fitting data.")
    return _quantile_transform_matrix(checked_x, state, inverse=True)


@register_atom(witness_quantile_transformer_fit_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda n_quantiles: n_quantiles >= 1, "n_quantiles must be at least one")
@icontract.require(lambda output_distribution: _quantile_distribution_valid(output_distribution), "output_distribution must be 'uniform' or 'normal'")
@icontract.require(lambda subsample: _subsample_valid(subsample), "subsample must be at least one or None")
@icontract.ensure(lambda result, X: _quantile_fit_transform_valid(result, X), "fit-transform output must match learned quantile state")
def quantile_transformer_fit_transform(
    X: MatrixLike,
    *,
    n_quantiles: int = 1000,
    output_distribution: str = "uniform",
    ignore_implicit_zeros: bool = False,
    subsample: int | None = 10_000,
    random_state: RandomStateLike = None,
    copy: bool = True,
) -> QuantileTransformerFitTransformResult:
    """Estimate empirical quantiles and transform the same feature matrix."""
    state = quantile_transformer_fit(
        X,
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=random_state,
    )
    return state, quantile_transformer_transform(X, state, copy=copy)


@register_atom(witness_quantile_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda axis: _valid_axis(axis), "axis must be 0 or 1")
@icontract.require(lambda n_quantiles: n_quantiles >= 1, "n_quantiles must be at least one")
@icontract.require(lambda output_distribution: _quantile_distribution_valid(output_distribution), "output_distribution must be 'uniform' or 'normal'")
@icontract.require(lambda subsample: _subsample_valid(subsample), "subsample must be at least one or None")
@icontract.ensure(lambda result, X: _quantile_transform_shape_matches(result, X), "quantile transformed output must preserve shape")
def quantile_transform(
    X: MatrixLike,
    *,
    axis: int = 0,
    n_quantiles: int = 1000,
    output_distribution: str = "uniform",
    ignore_implicit_zeros: bool = False,
    subsample: int | None = 100_000,
    random_state: RandomStateLike = None,
    copy: bool = True,
) -> MatrixLike:
    """Fit empirical quantiles on one axis and map values to uniform or normal marginals."""
    if axis == 0:
        _, transformed = quantile_transformer_fit_transform(
            X,
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )
        return transformed

    transposed = X.T if sp.issparse(X) else np.asarray(X).T
    _, transformed_t = quantile_transformer_fit_transform(
        transposed,
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        ignore_implicit_zeros=ignore_implicit_zeros,
        subsample=subsample,
        random_state=random_state,
        copy=copy,
    )
    return transformed_t.T
