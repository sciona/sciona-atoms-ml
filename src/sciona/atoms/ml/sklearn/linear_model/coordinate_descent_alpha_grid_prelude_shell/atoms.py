"""Sklearn coordinate-descent alpha-grid prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_alpha_grid_dense_Xyw,
    witness_cd_alpha_grid_l1_ratio_zero_error_message,
    witness_cd_alpha_grid_l1_ratio_zero_guard_required,
    witness_cd_alpha_grid_precomputed_Xy,
    witness_cd_alpha_grid_preprocess_kwargs,
    witness_cd_alpha_grid_sparse_mono_output_centered_Xyw,
    witness_cd_alpha_grid_yw,
)


def _numeric_scalar(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_array(value: object, ndim: int | None = None) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)) and (ndim is None or array.ndim == ndim))


def _valid_y_and_weight(y: object, sample_weight: object) -> bool:
    if not _finite_array(y):
        return False
    y_array = np.asarray(y)
    if y_array.ndim not in {1, 2}:
        return False
    if sample_weight is None:
        return True
    if not _finite_array(sample_weight, ndim=1):
        return False
    return bool(np.asarray(sample_weight).shape[0] == y_array.shape[0])


def _valid_dense_X_and_yw(X: object, yw: object) -> bool:
    if not (_finite_array(X, ndim=2) and _finite_array(yw)):
        return False
    X_array = np.asarray(X)
    yw_array = np.asarray(yw)
    return bool(yw_array.ndim in {1, 2} and yw_array.shape[0] == X_array.shape[0])


@register_atom(witness_cd_alpha_grid_l1_ratio_zero_guard_required)
@icontract.require(lambda l1_ratio: _numeric_scalar(l1_ratio), "l1_ratio must be a finite numeric scalar")
@icontract.ensure(
    lambda result, l1_ratio: isinstance(result, bool) and result == (float(l1_ratio) == 0.0),
    "zero-l1_ratio guard must match sklearn's l1_ratio == 0 branch",
)
def cd_alpha_grid_l1_ratio_zero_guard_required(l1_ratio: float) -> bool:
    """Return whether _alpha_grid should reject automatic grids for l1_ratio=0."""
    return float(l1_ratio) == 0.0


@register_atom(witness_cd_alpha_grid_l1_ratio_zero_error_message)
@icontract.require(lambda l1_ratio: _numeric_scalar(l1_ratio), "l1_ratio must be a finite numeric scalar")
@icontract.ensure(
    lambda result: isinstance(result, str)
    and result
    == (
        "Automatic alpha grid generation is not supported for"
        " l1_ratio=0. Please supply a grid by providing "
        "your estimator with the appropriate `alphas=` "
        "argument."
    ),
    "zero-l1_ratio error message must match sklearn",
)
def cd_alpha_grid_l1_ratio_zero_error_message(l1_ratio: float) -> str:
    """Return the ValueError text emitted by _alpha_grid for l1_ratio=0."""
    del l1_ratio
    return (
        "Automatic alpha grid generation is not supported for"
        " l1_ratio=0. Please supply a grid by providing "
        "your estimator with the appropriate `alphas=` "
        "argument."
    )


@register_atom(witness_cd_alpha_grid_precomputed_Xy)
@icontract.require(lambda Xy: Xy is not None, "Xy must be provided")
@icontract.ensure(
    lambda result, Xy: result is Xy,
    "precomputed Xy branch must preserve the supplied Xy object",
)
def cd_alpha_grid_precomputed_Xy(Xy: object) -> object:
    """Return the precomputed Xy object selected by _alpha_grid when present."""
    return Xy


@register_atom(witness_cd_alpha_grid_preprocess_kwargs)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.ensure(
    lambda result, fit_intercept, copy_X, sample_weight: result
    == {
        "fit_intercept": fit_intercept,
        "copy": copy_X,
        "sample_weight": sample_weight,
        "check_input": False,
    },
    "_preprocess_data kwargs must match _alpha_grid",
)
def cd_alpha_grid_preprocess_kwargs(
    fit_intercept: bool,
    copy_X: bool,
    sample_weight: object,
) -> dict[str, object]:
    """Return the fixed _preprocess_data kwargs assembled by _alpha_grid."""
    return {
        "fit_intercept": fit_intercept,
        "copy": copy_X,
        "sample_weight": sample_weight,
        "check_input": False,
    }


@register_atom(witness_cd_alpha_grid_yw)
@icontract.require(lambda y, sample_weight: _valid_y_and_weight(y, sample_weight), "y/sample_weight shapes must align")
@icontract.ensure(
    lambda result, y, sample_weight: (result is y if sample_weight is None else _finite_array(result)),
    "yw must preserve y without sample weights and be finite otherwise",
)
def cd_alpha_grid_yw(y: NDArray[np.floating], sample_weight: object) -> object:
    """Return the weighted target array used by _alpha_grid."""
    if sample_weight is None:
        return y
    y_array = np.asarray(y)
    weight_array = np.asarray(sample_weight)
    if y_array.ndim > 1:
        return y_array * weight_array.reshape(-1, 1)
    return y_array * weight_array


@register_atom(witness_cd_alpha_grid_dense_Xyw)
@icontract.require(lambda X, yw: _valid_dense_X_and_yw(X, yw), "dense X and yw must be finite with matching rows")
@icontract.ensure(lambda result: _finite_array(result), "dense Xyw must be finite")
def cd_alpha_grid_dense_Xyw(
    X: NDArray[np.floating],
    yw: NDArray[np.floating],
) -> NDArray[np.generic]:
    """Return the dense np.dot(X.T, yw) Xyw value used by _alpha_grid."""
    return np.asarray(np.dot(np.asarray(X).T, np.asarray(yw)))


@register_atom(witness_cd_alpha_grid_sparse_mono_output_centered_Xyw)
@icontract.require(lambda X: sparse.issparse(X), "X must be sparse")
@icontract.require(lambda yw: _finite_array(yw, ndim=1), "yw must be a finite mono-output vector")
@icontract.require(lambda X, yw: np.asarray(yw).shape[0] == X.shape[0], "yw must have one value per sparse row")
@icontract.require(lambda X_offset: _finite_array(X_offset, ndim=1), "X_offset must be a finite vector")
@icontract.require(lambda X, X_offset: np.asarray(X_offset).shape[0] == X.shape[1], "X_offset must match features")
@icontract.ensure(
    lambda result, X: _finite_array(result, ndim=1) and np.asarray(result).shape[0] == X.shape[1],
    "sparse mono-output centered Xyw must have one value per feature",
)
def cd_alpha_grid_sparse_mono_output_centered_Xyw(
    X: sparse.spmatrix,
    yw: NDArray[np.floating],
    X_offset: NDArray[np.floating],
) -> NDArray[np.generic]:
    """Return sparse mono-output Xyw after sklearn's centering correction."""
    weighted_dot = safe_sparse_dot(X.T, np.asarray(yw), dense_output=True)
    return np.asarray(weighted_dot) - np.sum(yw) * np.asarray(X_offset)
