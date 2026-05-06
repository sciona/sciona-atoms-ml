"""Sklearn coordinate-descent path-residual error-aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_intercepts,
    witness_cd_path_residuals_mean_mse,
    witness_cd_path_residuals_mse,
    witness_cd_path_residuals_residues,
    witness_cd_path_residuals_use_weighted_mse,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_array(value: object, ndim: int | None = None) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(array)) and (ndim is None or array.ndim == ndim))


def _intercepts_valid(result: object, y_offset: object, coefs: object) -> bool:
    if not (_finite_array(result, ndim=2) and _finite_array(y_offset, ndim=1) and _finite_array(coefs, ndim=3)):
        return False
    result_values = np.asarray(result, dtype=np.float64)
    y_offset_values = np.asarray(y_offset, dtype=np.float64)
    coef_values = np.asarray(coefs, dtype=np.float64)
    return bool(result_values.shape == (y_offset_values.shape[0], coef_values.shape[2]))


def _residues_valid(result: object, X_test_coefs: object, y_test: object) -> bool:
    if not (_finite_array(result, ndim=3) and _finite_array(X_test_coefs, ndim=3) and _finite_array(y_test, ndim=2)):
        return False
    result_values = np.asarray(result, dtype=np.float64)
    test_coef_values = np.asarray(X_test_coefs, dtype=np.float64)
    y_test_values = np.asarray(y_test, dtype=np.float64)
    return bool(
        result_values.shape == test_coef_values.shape
        and test_coef_values.shape[0] == y_test_values.shape[0]
        and test_coef_values.shape[1] == y_test_values.shape[1]
    )


def _mse_inputs_valid(residues: object, sw_test: object, use_weighted_mse: bool) -> bool:
    if not _finite_array(residues, ndim=3):
        return False
    residue_values = np.asarray(residues, dtype=np.float64)
    if not use_weighted_mse:
        return True
    if not _finite_array(sw_test, ndim=1):
        return False
    weight_values = np.asarray(sw_test, dtype=np.float64)
    return bool(weight_values.shape[0] == residue_values.shape[0])


def _mse_valid(result: object, residues: object) -> bool:
    if not (_finite_array(result, ndim=2) and _finite_array(residues, ndim=3)):
        return False
    result_values = np.asarray(result, dtype=np.float64)
    residue_values = np.asarray(residues, dtype=np.float64)
    return bool(result_values.shape == residue_values.shape[1:])


def _mean_mse_valid(result: object, this_mse: object) -> bool:
    if not (_finite_array(result, ndim=1) and _finite_array(this_mse, ndim=2)):
        return False
    result_values = np.asarray(result, dtype=np.float64)
    mse_values = np.asarray(this_mse, dtype=np.float64)
    return bool(result_values.shape == (mse_values.shape[1],))


@register_atom(witness_cd_path_residuals_intercepts)
@icontract.require(lambda y_offset: _finite_array(y_offset, ndim=1), "y_offset must be a finite rank-1 float array")
@icontract.require(lambda X_offset: _finite_array(X_offset, ndim=1), "X_offset must be a finite rank-1 float array")
@icontract.require(lambda coefs: _finite_array(coefs, ndim=3), "coefs must be a finite rank-3 float array")
@icontract.require(
    lambda X_offset, coefs: np.asarray(X_offset, dtype=np.float64).shape[0] == np.asarray(coefs, dtype=np.float64).shape[1],
    "X_offset feature count must match coefficient feature count",
)
@icontract.ensure(
    lambda result, y_offset, coefs: _intercepts_valid(result, y_offset, coefs),
    "intercepts must have one row per output and one column per alpha",
)
def cd_path_residuals_intercepts(
    y_offset: NDArray[np.floating],
    X_offset: NDArray[np.floating],
    coefs: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return sklearn's intercept tensor for the _path_residuals error shell."""
    return np.asarray(y_offset, dtype=np.float64)[:, np.newaxis] - np.dot(
        np.asarray(X_offset, dtype=np.float64), np.asarray(coefs, dtype=np.float64)
    )


@register_atom(witness_cd_path_residuals_residues)
@icontract.require(lambda X_test_coefs: _finite_array(X_test_coefs, ndim=3), "X_test_coefs must be a finite rank-3 float array")
@icontract.require(lambda y_test: _finite_array(y_test, ndim=2), "y_test must be a finite rank-2 float array")
@icontract.require(lambda intercepts: _finite_array(intercepts, ndim=2), "intercepts must be a finite rank-2 float array")
@icontract.require(
    lambda X_test_coefs, y_test: np.asarray(X_test_coefs, dtype=np.float64).shape[:2] == (
        np.asarray(y_test, dtype=np.float64).shape[0],
        np.asarray(y_test, dtype=np.float64).shape[1],
    ),
    "X_test_coefs and y_test must align on sample and output axes",
)
@icontract.require(
    lambda X_test_coefs, intercepts: np.asarray(X_test_coefs, dtype=np.float64).shape[1:] == (
        np.asarray(intercepts, dtype=np.float64).shape[0],
        np.asarray(intercepts, dtype=np.float64).shape[1],
    ),
    "intercepts must align with X_test_coefs output and alpha axes",
)
@icontract.ensure(
    lambda result, X_test_coefs, y_test: _residues_valid(result, X_test_coefs, y_test),
    "residues must preserve the X_test_coefs tensor shape",
)
def cd_path_residuals_residues(
    X_test_coefs: NDArray[np.floating],
    y_test: NDArray[np.floating],
    intercepts: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return sklearn's residual tensor after subtracting y_test and adding intercepts."""
    return (
        np.asarray(X_test_coefs, dtype=np.float64)
        - np.asarray(y_test, dtype=np.float64)[:, :, np.newaxis]
        + np.asarray(intercepts, dtype=np.float64)
    )


@register_atom(witness_cd_path_residuals_use_weighted_mse)
@icontract.ensure(
    lambda result, sample_weight: _bool(result) and result == (sample_weight is not None),
    "weighted-MSE branch must match sample_weight is not None",
)
def cd_path_residuals_use_weighted_mse(sample_weight: object) -> bool:
    """Return whether _path_residuals reduces squared residues with sample weights."""
    return sample_weight is not None


@register_atom(witness_cd_path_residuals_mse)
@icontract.require(lambda use_weighted_mse: _bool(use_weighted_mse), "use_weighted_mse must be boolean")
@icontract.require(
    lambda residues, sw_test, use_weighted_mse: _mse_inputs_valid(residues, sw_test, use_weighted_mse),
    "residues and optional sw_test must satisfy sklearn's MSE reduction shape rules",
)
@icontract.ensure(
    lambda result, residues: _mse_valid(result, residues),
    "this_mse must have one row per output and one column per alpha",
)
def cd_path_residuals_mse(
    residues: NDArray[np.floating], sw_test: object, use_weighted_mse: bool
) -> NDArray[np.float64]:
    """Return sklearn's weighted or unweighted squared-residue reduction over test samples."""
    squared = np.asarray(residues, dtype=np.float64) ** 2
    if use_weighted_mse:
        return np.asarray(np.average(squared, weights=np.asarray(sw_test, dtype=np.float64), axis=0), dtype=np.float64)
    return np.asarray(squared.mean(axis=0), dtype=np.float64)


@register_atom(witness_cd_path_residuals_mean_mse)
@icontract.require(lambda this_mse: _finite_array(this_mse, ndim=2), "this_mse must be a finite rank-2 float array")
@icontract.ensure(
    lambda result, this_mse: _mean_mse_valid(result, this_mse),
    "final MSE must have one value per alpha",
)
def cd_path_residuals_mean_mse(this_mse: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return sklearn's final mean(axis=0) reduction over outputs."""
    return np.asarray(np.asarray(this_mse, dtype=np.float64).mean(axis=0), dtype=np.float64)
