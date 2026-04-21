"""Gaussian process kernel atoms adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_constant_kernel,
    witness_constant_kernel_diag,
    witness_dot_product_kernel,
    witness_dot_product_kernel_diag,
    witness_exp_sine_squared_kernel,
    witness_exp_sine_squared_kernel_diag,
    witness_matern_kernel_diag,
    witness_matern_kernel_matrix,
    witness_rational_quadratic_kernel,
    witness_rational_quadratic_kernel_diag,
    witness_rbf_kernel_diag,
    witness_rbf_kernel_matrix,
    witness_white_kernel,
    witness_white_kernel_diag,
)


LengthScale = float | tuple[float, ...] | NDArray[np.float64]


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2)


def _optional_matrix_2d(Y: NDArray[np.float64] | None) -> bool:
    return bool(Y is None or np.asarray(Y).ndim == 2)


def _finite_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def _finite_optional_matrix(Y: NDArray[np.float64] | None) -> bool:
    return bool(Y is None or _finite_matrix(Y))


def _same_feature_count(X: NDArray[np.float64], Y: NDArray[np.float64] | None) -> bool:
    return bool(Y is None or (_matrix_2d(X) and _matrix_2d(Y) and np.asarray(X).shape[1] == np.asarray(Y).shape[1]))


def _positive_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _nonnegative_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _positive_or_inf_scalar(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and (np.isposinf(float(value)) or (np.isfinite(float(value)) and float(value) > 0.0)))


def _scalar_length_scale_valid(length_scale: float) -> bool:
    return _positive_scalar(length_scale)


def _length_scale_valid(length_scale: LengthScale, X: NDArray[np.float64]) -> bool:
    try:
        values = np.atleast_1d(np.asarray(length_scale, dtype=np.float64))
    except (TypeError, ValueError):
        return False
    return bool(
        _matrix_2d(X)
        and values.ndim == 1
        and values.shape[0] in {1, np.asarray(X).shape[1]}
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
    )


def _kernel_matrix_valid(result: NDArray[np.float64], X: NDArray[np.float64], Y: NDArray[np.float64] | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_rows = np.asarray(X).shape[0]
    n_cols = n_rows if Y is None else np.asarray(Y).shape[0]
    return bool(values.shape == (n_rows, n_cols) and np.all(np.isfinite(values)))


def _diag_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0],) and np.all(np.isfinite(values)))


def _normalized_kernel_matrix_valid(result: NDArray[np.float64], X: NDArray[np.float64], Y: NDArray[np.float64] | None) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(_kernel_matrix_valid(result, X, Y) and np.all(values >= 0.0) and np.all(values <= 1.0 + 1e-12))


def _length_scale_array(length_scale: LengthScale, n_features: int) -> NDArray[np.float64]:
    values = np.atleast_1d(np.asarray(length_scale, dtype=np.float64))
    if values.shape[0] == 1:
        return values
    if values.shape[0] != n_features:
        raise ValueError("length_scale must be scalar or match feature count")
    return values


@register_atom(witness_constant_kernel)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda constant_value: _positive_scalar(constant_value), "constant_value must be positive and finite")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def constant_kernel(X: NDArray[np.float64], Y: NDArray[np.float64] | None = None, *, constant_value: float = 1.0) -> NDArray[np.float64]:
    """Return a matrix filled with one positive covariance value."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    checked_y = checked_x if Y is None else np.atleast_2d(np.asarray(Y, dtype=np.float64))
    return np.full((checked_x.shape[0], checked_y.shape[0]), float(constant_value), dtype=np.float64)


@register_atom(witness_constant_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda constant_value: _positive_scalar(constant_value), "constant_value must be positive and finite")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def constant_kernel_diag(X: NDArray[np.float64], *, constant_value: float = 1.0) -> NDArray[np.float64]:
    """Return the diagonal values for the constant covariance kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.full(checked_x.shape[0], float(constant_value), dtype=np.float64)


@register_atom(witness_white_kernel)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda noise_level: _positive_scalar(noise_level), "noise_level must be positive and finite")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def white_kernel(X: NDArray[np.float64], Y: NDArray[np.float64] | None = None, *, noise_level: float = 1.0) -> NDArray[np.float64]:
    """Return diagonal white-noise covariance for matching samples."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if Y is None:
        return float(noise_level) * np.eye(checked_x.shape[0], dtype=np.float64)
    checked_y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    return np.zeros((checked_x.shape[0], checked_y.shape[0]), dtype=np.float64)


@register_atom(witness_white_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda noise_level: _positive_scalar(noise_level), "noise_level must be positive and finite")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def white_kernel_diag(X: NDArray[np.float64], *, noise_level: float = 1.0) -> NDArray[np.float64]:
    """Return the diagonal values for the white-noise covariance kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.full(checked_x.shape[0], float(noise_level), dtype=np.float64)


@register_atom(witness_dot_product_kernel)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda sigma_0: _nonnegative_scalar(sigma_0), "sigma_0 must be non-negative and finite")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def dot_product_kernel(X: NDArray[np.float64], Y: NDArray[np.float64] | None = None, *, sigma_0: float = 1.0) -> NDArray[np.float64]:
    """Return dot products plus a squared offset."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    checked_y = checked_x if Y is None else np.atleast_2d(np.asarray(Y, dtype=np.float64))
    return np.inner(checked_x, checked_y) + float(sigma_0) ** 2


@register_atom(witness_dot_product_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda sigma_0: _nonnegative_scalar(sigma_0), "sigma_0 must be non-negative and finite")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def dot_product_kernel_diag(X: NDArray[np.float64], *, sigma_0: float = 1.0) -> NDArray[np.float64]:
    """Return self dot products plus a squared offset."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.einsum("ij,ij->i", checked_x, checked_x) + float(sigma_0) ** 2


@register_atom(witness_rbf_kernel_matrix)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda length_scale, X: _length_scale_valid(length_scale, X), "length_scale must be positive and scalar or feature-matched")
@icontract.ensure(lambda result, X, Y: _kernel_matrix_valid(result, X, Y), "kernel matrix must match sample counts")
def rbf_kernel_matrix(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    length_scale: LengthScale = 1.0,
) -> NDArray[np.float64]:
    """Return squared-exponential covariance from scaled distances."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    scale = _length_scale_array(length_scale, checked_x.shape[1])
    scaled_x = checked_x / scale
    if Y is None:
        distances = pdist(scaled_x, metric="sqeuclidean")
        kernel = squareform(np.exp(-0.5 * distances))
        np.fill_diagonal(kernel, 1.0)
        return np.asarray(kernel, dtype=np.float64)
    checked_y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    distances = cdist(scaled_x, checked_y / scale, metric="sqeuclidean")
    return np.asarray(np.exp(-0.5 * distances), dtype=np.float64)


@register_atom(witness_rbf_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def rbf_kernel_diag(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit diagonal for the squared-exponential kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.ones(checked_x.shape[0], dtype=np.float64)


@register_atom(witness_rational_quadratic_kernel)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda length_scale: _scalar_length_scale_valid(length_scale), "length_scale must be positive and finite")
@icontract.require(lambda alpha: _positive_scalar(alpha), "alpha must be positive and finite")
@icontract.ensure(lambda result, X, Y: _normalized_kernel_matrix_valid(result, X, Y), "kernel matrix must contain normalized finite covariances")
def rational_quadratic_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    length_scale: float = 1.0,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Return rational quadratic covariance from squared distances."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if Y is None:
        distances = squareform(pdist(checked_x, metric="sqeuclidean"))
        kernel = (1.0 + distances / (2.0 * float(alpha) * float(length_scale) ** 2)) ** -float(alpha)
        np.fill_diagonal(kernel, 1.0)
        return np.asarray(kernel, dtype=np.float64)
    checked_y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    distances = cdist(checked_x, checked_y, metric="sqeuclidean")
    return np.asarray((1.0 + distances / (2.0 * float(alpha) * float(length_scale) ** 2)) ** -float(alpha), dtype=np.float64)


@register_atom(witness_rational_quadratic_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def rational_quadratic_kernel_diag(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit diagonal for the rational quadratic kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.ones(checked_x.shape[0], dtype=np.float64)


@register_atom(witness_matern_kernel_matrix)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda length_scale, X: _length_scale_valid(length_scale, X), "length_scale must be positive and scalar or feature-matched")
@icontract.require(lambda nu: _positive_or_inf_scalar(nu), "nu must be positive or infinity")
@icontract.ensure(lambda result, X, Y: _normalized_kernel_matrix_valid(result, X, Y), "kernel matrix must contain normalized finite covariances")
def matern_kernel_matrix(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    length_scale: LengthScale = 1.0,
    nu: float = 1.5,
) -> NDArray[np.float64]:
    """Return Matern covariance from scaled Euclidean distances."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    scale = _length_scale_array(length_scale, checked_x.shape[1])
    scaled_x = checked_x / scale
    if Y is None:
        distances = pdist(scaled_x, metric="euclidean")
    else:
        checked_y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        distances = cdist(scaled_x, checked_y / scale, metric="euclidean")

    if nu == 0.5:
        kernel = np.exp(-distances)
    elif nu == 1.5:
        scaled_distances = distances * math.sqrt(3.0)
        kernel = (1.0 + scaled_distances) * np.exp(-scaled_distances)
    elif nu == 2.5:
        scaled_distances = distances * math.sqrt(5.0)
        kernel = (1.0 + scaled_distances + scaled_distances**2 / 3.0) * np.exp(-scaled_distances)
    elif np.isposinf(float(nu)):
        kernel = np.exp(-(distances**2) / 2.0)
    else:
        adjusted = np.asarray(distances, dtype=np.float64)
        adjusted = adjusted.copy()
        adjusted[adjusted == 0.0] += np.finfo(float).eps
        tmp = math.sqrt(2.0 * float(nu)) * adjusted
        kernel = np.full_like(adjusted, (2.0 ** (1.0 - float(nu))) / gamma(float(nu)), dtype=np.float64)
        kernel *= tmp**float(nu)
        kernel *= kv(float(nu), tmp)

    if Y is None:
        matrix = squareform(kernel)
        np.fill_diagonal(matrix, 1.0)
        return np.asarray(matrix, dtype=np.float64)
    return np.asarray(kernel, dtype=np.float64)


@register_atom(witness_matern_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def matern_kernel_diag(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit diagonal for the Matern kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.ones(checked_x.shape[0], dtype=np.float64)


@register_atom(witness_exp_sine_squared_kernel)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda Y: _optional_matrix_2d(Y), "Y must be None or a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.require(lambda Y: _finite_optional_matrix(Y), "Y must contain only finite values")
@icontract.require(lambda X, Y: _same_feature_count(X, Y), "X and Y must have matching feature counts")
@icontract.require(lambda length_scale: _positive_scalar(length_scale), "length_scale must be positive and finite")
@icontract.require(lambda periodicity: _positive_scalar(periodicity), "periodicity must be positive and finite")
@icontract.ensure(lambda result, X, Y: _normalized_kernel_matrix_valid(result, X, Y), "kernel matrix must contain normalized finite covariances")
def exp_sine_squared_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    length_scale: float = 1.0,
    periodicity: float = 1.0,
) -> NDArray[np.float64]:
    """Return periodic covariance from Euclidean distances."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if Y is None:
        distances = squareform(pdist(checked_x, metric="euclidean"))
    else:
        checked_y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        distances = cdist(checked_x, checked_y, metric="euclidean")
    sine_values = np.sin(np.pi / float(periodicity) * distances)
    return np.asarray(np.exp(-2.0 * (sine_values / float(length_scale)) ** 2), dtype=np.float64)


@register_atom(witness_exp_sine_squared_kernel_diag)
@icontract.require(lambda X: _matrix_2d(X), "X must be a two-dimensional matrix")
@icontract.require(lambda X: _finite_matrix(X), "X must contain only finite values")
@icontract.ensure(lambda result, X: _diag_valid(result, X), "diagonal must match sample count")
def exp_sine_squared_kernel_diag(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit diagonal for the periodic kernel."""
    checked_x = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.ones(checked_x.shape[0], dtype=np.float64)
