"""Selected cross-decomposition atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinv, svd
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip

from sciona.ghost.registry import register_atom

from .state_models import PLSState, PLSSVDState
from .witnesses import witness_cca_fit, witness_pls_canonical_fit, witness_pls_regression_fit, witness_plssvd_fit


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).shape[0] == np.asarray(y).shape[0])


def _target_count(y: NDArray[np.float64]) -> int:
    values = np.asarray(y)
    return 1 if values.ndim == 1 else int(values.shape[1])


def _plssvd_components_valid(n_components: int, X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    if not isinstance(n_components, int) or n_components < 1:
        return False
    n_samples, n_features = np.asarray(X).shape
    return bool(n_components <= min(n_samples, n_features, _target_count(y)))


def _center_scale_xy(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    scale: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    centered_x = np.asarray(X, dtype=np.float64).copy()
    centered_y = np.asarray(y, dtype=np.float64).copy()
    x_mean = centered_x.mean(axis=0)
    y_mean = centered_y.mean(axis=0)
    centered_x -= x_mean
    centered_y -= y_mean
    if scale:
        x_std = centered_x.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        y_std = centered_y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        centered_x /= x_std
        centered_y /= y_std
    else:
        x_std = np.ones(centered_x.shape[1], dtype=np.float64)
        y_std = np.ones(centered_y.shape[1], dtype=np.float64)
    return centered_x, centered_y, x_mean, y_mean, x_std, y_std


def _plssvd_state_valid(state: PLSSVDState) -> bool:
    return bool(
        state.x_weights.shape == (state.n_features_in, state.n_components)
        and state.y_weights.shape == (state.n_targets, state.n_components)
        and state.singular_values.shape == (state.n_components,)
        and state.x_mean.shape == (state.n_features_in,)
        and state.y_mean.shape == (state.n_targets,)
        and state.x_std.shape == (state.n_features_in,)
        and state.y_std.shape == (state.n_targets,)
        and np.all(np.isfinite(state.x_weights))
        and np.all(np.isfinite(state.y_weights))
        and np.all(np.isfinite(state.singular_values))
        and np.all(np.isfinite(state.x_mean))
        and np.all(np.isfinite(state.y_mean))
        and np.all(state.x_std > 0.0)
        and np.all(state.y_std > 0.0)
    )


@register_atom(witness_plssvd_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda n_components, X, y: _plssvd_components_valid(n_components, X, y), "n_components exceeds the cross-covariance rank bound")
@icontract.ensure(lambda result: _plssvd_state_valid(result), "PLS-SVD state must contain fitted weights and metadata")
def plssvd_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_components: int = 2,
    scale: bool = True,
    copy: bool = True,
) -> PLSSVDState:
    """Fit PLS-SVD weights from the cross-covariance of X and y."""
    del copy
    check_consistent_length(X, y)
    checked_x = check_array(X, dtype=np.float64, ensure_min_samples=2)
    checked_y = check_array(y, input_name="y", dtype=np.float64, ensure_2d=False)
    if checked_y.ndim == 1:
        checked_y = checked_y.reshape(-1, 1)
    rank_upper_bound = min(checked_x.shape[0], checked_x.shape[1], checked_y.shape[1])
    if n_components > rank_upper_bound:
        raise ValueError(
            f"`n_components` upper bound is {rank_upper_bound}. "
            f"Got {n_components} instead. Reduce `n_components`."
        )

    scaled_x, scaled_y, x_mean, y_mean, x_std, y_std = _center_scale_xy(checked_x, checked_y, scale)
    cross_covariance = np.dot(scaled_x.T, scaled_y)
    u, singular_values, vt = svd(cross_covariance, full_matrices=False)
    u = u[:, :n_components]
    vt = vt[:n_components]
    u, vt = svd_flip(u, vt)
    return PLSSVDState(
        x_weights=np.asarray(u, dtype=np.float64).copy(),
        y_weights=np.asarray(vt.T, dtype=np.float64).copy(),
        singular_values=np.asarray(singular_values[:n_components], dtype=np.float64).copy(),
        x_mean=np.asarray(x_mean, dtype=np.float64).copy(),
        y_mean=np.asarray(y_mean, dtype=np.float64).copy(),
        x_std=np.asarray(x_std, dtype=np.float64).copy(),
        y_std=np.asarray(y_std, dtype=np.float64).copy(),
        n_components=int(n_components),
        scale=bool(scale),
        n_features_in=int(checked_x.shape[1]),
        n_targets=int(checked_y.shape[1]),
    )


def _positive_int(value: int) -> bool:
    return isinstance(value, int) and value >= 1


def _positive_float(value: float) -> bool:
    return bool(float(value) > 0.0)


def _pls_algorithm_valid(algorithm: str) -> bool:
    return algorithm in {"nipals", "svd"}


def _pls_components_valid(
    n_components: int,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    deflation_mode: str,
) -> bool:
    if not _positive_int(n_components):
        return False
    n_samples, n_features = np.asarray(X).shape
    n_targets = _target_count(y)
    upper_bound = min(n_samples, n_features) if deflation_mode == "regression" else min(n_samples, n_features, n_targets)
    return bool(n_components <= upper_bound)


def _pinv2_old(a: NDArray[np.float64]) -> NDArray[np.float64]:
    u, singular_values, vh = svd(a, full_matrices=False, check_finite=False)
    dtype_char = u.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}
    condition = np.max(singular_values) * factor[dtype_char] * np.finfo(dtype_char).eps
    rank = np.sum(singular_values > condition)
    u = u[:, :rank]
    u /= singular_values[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))


def _first_singular_vectors_power_method(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    mode: str,
    max_iter: int,
    tol: float,
    norm_y_weights: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    eps = np.finfo(X.dtype).eps
    nonconstant_columns = (col for col in y.T if np.any(np.abs(col) > eps))
    y_score = next(nonconstant_columns)
    x_weights_old: NDArray[np.float64] | float = 100.0
    if mode == "B":
        x_pinv = _pinv2_old(X)
        y_pinv = _pinv2_old(y)

    for iteration in range(max_iter):
        if mode == "B":
            x_weights = np.dot(x_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)
        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)

        if mode == "B":
            y_weights = np.dot(y_pinv, x_score)
        else:
            y_weights = np.dot(y.T, x_score) / np.dot(x_score.T, x_score)
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps

        y_score = np.dot(y, y_weights) / (np.dot(y_weights, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or y.shape[1] == 1:
            break
        x_weights_old = x_weights
    return x_weights, y_weights, iteration + 1


def _first_singular_vectors_svd(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    u, _, vt = svd(np.dot(X.T, y), full_matrices=False)
    return u[:, 0], vt[0, :]


def _svd_flip_1d(u: NDArray[np.float64], v: NDArray[np.float64]) -> None:
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


def _pls_state_valid(state: PLSState) -> bool:
    return bool(
        state.x_weights.shape == (state.n_features_in, state.n_components)
        and state.y_weights.shape == (state.n_targets, state.n_components)
        and state.x_scores.shape[1] == state.n_components
        and state.y_scores.shape == state.x_scores.shape
        and state.x_loadings.shape == state.x_weights.shape
        and state.y_loadings.shape == state.y_weights.shape
        and state.x_rotations.shape == state.x_weights.shape
        and state.y_rotations.shape == state.y_weights.shape
        and state.coef.shape == (state.n_targets, state.n_features_in)
        and state.intercept.shape == (state.n_targets,)
        and state.x_mean.shape == (state.n_features_in,)
        and state.y_mean.shape == (state.n_targets,)
        and state.x_std.shape == (state.n_features_in,)
        and state.y_std.shape == (state.n_targets,)
        and state.deflation_mode in {"regression", "canonical"}
        and state.mode in {"A", "B"}
        and state.algorithm in {"nipals", "svd"}
        and np.all(np.isfinite(state.x_weights))
        and np.all(np.isfinite(state.y_weights))
        and np.all(np.isfinite(state.x_loadings))
        and np.all(np.isfinite(state.y_loadings))
        and np.all(np.isfinite(state.x_rotations))
        and np.all(np.isfinite(state.y_rotations))
        and np.all(np.isfinite(state.coef))
        and np.all(np.isfinite(state.intercept))
        and np.all(state.x_std > 0.0)
        and np.all(state.y_std > 0.0)
    )


def _pls_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_components: int,
    scale: bool,
    max_iter: int,
    tol: float,
    deflation_mode: str,
    mode: str,
    algorithm: str,
) -> PLSState:
    check_consistent_length(X, y)
    checked_x = check_array(X, dtype=np.float64, ensure_min_samples=2)
    checked_y = check_array(y, input_name="y", dtype=np.float64, ensure_2d=False)
    predict_1d = checked_y.ndim == 1
    if predict_1d:
        checked_y = checked_y.reshape(-1, 1)

    n_samples, n_features = checked_x.shape
    n_targets = checked_y.shape[1]
    rank_upper_bound = min(n_samples, n_features) if deflation_mode == "regression" else min(n_samples, n_features, n_targets)
    if n_components > rank_upper_bound:
        raise ValueError(
            f"`n_components` upper bound is {rank_upper_bound}. "
            f"Got {n_components} instead. Reduce `n_components`."
        )

    norm_y_weights = deflation_mode == "canonical"
    xk, yk, x_mean, y_mean, x_std, y_std = _center_scale_xy(checked_x, checked_y, scale)
    x_weights_matrix = np.zeros((n_features, n_components), dtype=np.float64)
    y_weights_matrix = np.zeros((n_targets, n_components), dtype=np.float64)
    x_scores_matrix = np.zeros((n_samples, n_components), dtype=np.float64)
    y_scores_matrix = np.zeros((n_samples, n_components), dtype=np.float64)
    x_loadings_matrix = np.zeros((n_features, n_components), dtype=np.float64)
    y_loadings_matrix = np.zeros((n_targets, n_components), dtype=np.float64)
    n_iter: list[int] = []

    y_eps = np.finfo(yk.dtype).eps
    for component_index in range(n_components):
        if algorithm == "nipals":
            yk_mask = np.all(np.abs(yk) < 10 * y_eps, axis=0)
            yk[:, yk_mask] = 0.0
            try:
                x_weights, y_weights, component_iter = _first_singular_vectors_power_method(
                    xk,
                    yk,
                    mode=mode,
                    max_iter=max_iter,
                    tol=tol,
                    norm_y_weights=norm_y_weights,
                )
            except StopIteration:
                break
            n_iter.append(component_iter)
        else:
            x_weights, y_weights = _first_singular_vectors_svd(xk, yk)

        _svd_flip_1d(x_weights, y_weights)
        x_scores = np.dot(xk, x_weights)
        y_ss = 1.0 if norm_y_weights else np.dot(y_weights, y_weights)
        y_scores = np.dot(yk, y_weights) / y_ss

        x_loadings = np.dot(x_scores, xk) / np.dot(x_scores, x_scores)
        xk -= np.outer(x_scores, x_loadings)
        if deflation_mode == "canonical":
            y_loadings = np.dot(y_scores, yk) / np.dot(y_scores, y_scores)
            yk -= np.outer(y_scores, y_loadings)
        else:
            y_loadings = np.dot(x_scores, yk) / np.dot(x_scores, x_scores)
            yk -= np.outer(x_scores, y_loadings)

        x_weights_matrix[:, component_index] = x_weights
        y_weights_matrix[:, component_index] = y_weights
        x_scores_matrix[:, component_index] = x_scores
        y_scores_matrix[:, component_index] = y_scores
        x_loadings_matrix[:, component_index] = x_loadings
        y_loadings_matrix[:, component_index] = y_loadings

    x_rotations = np.dot(x_weights_matrix, pinv(np.dot(x_loadings_matrix.T, x_weights_matrix), check_finite=False))
    y_rotations = np.dot(y_weights_matrix, pinv(np.dot(y_loadings_matrix.T, y_weights_matrix), check_finite=False))
    coef = np.dot(x_rotations, y_loadings_matrix.T)
    coef = (coef * y_std).T / x_std
    return PLSState(
        x_weights=x_weights_matrix,
        y_weights=y_weights_matrix,
        x_scores=x_scores_matrix,
        y_scores=y_scores_matrix,
        x_loadings=x_loadings_matrix,
        y_loadings=y_loadings_matrix,
        x_rotations=x_rotations,
        y_rotations=y_rotations,
        coef=np.asarray(coef, dtype=np.float64),
        intercept=np.asarray(y_mean, dtype=np.float64).copy(),
        x_mean=np.asarray(x_mean, dtype=np.float64).copy(),
        y_mean=np.asarray(y_mean, dtype=np.float64).copy(),
        x_std=np.asarray(x_std, dtype=np.float64).copy(),
        y_std=np.asarray(y_std, dtype=np.float64).copy(),
        n_iter=tuple(n_iter),
        n_components=int(n_components),
        scale=bool(scale),
        deflation_mode=deflation_mode,
        mode=mode,
        algorithm=algorithm,
        n_features_in=int(n_features),
        n_targets=int(n_targets),
        predict_1d=bool(predict_1d),
    )


@register_atom(witness_pls_regression_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda n_components, X, y: _pls_components_valid(n_components, X, y, deflation_mode="regression"), "n_components exceeds the PLS rank bound")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda tol: _positive_float(tol), "tol must be positive")
@icontract.ensure(lambda result: _pls_state_valid(result), "PLS state must contain fitted weights, rotations, and coefficients")
def pls_regression_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_components: int = 2,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> PLSState:
    """Fit PLS regression and return immutable latent-variable state."""
    del copy
    return _pls_fit(
        X,
        y,
        n_components=n_components,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
        deflation_mode="regression",
        mode="A",
        algorithm="nipals",
    )


@register_atom(witness_pls_canonical_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda n_components, X, y: _pls_components_valid(n_components, X, y, deflation_mode="canonical"), "n_components exceeds the PLS rank bound")
@icontract.require(lambda algorithm: _pls_algorithm_valid(algorithm), "algorithm must be 'nipals' or 'svd'")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda tol: _positive_float(tol), "tol must be positive")
@icontract.ensure(lambda result: _pls_state_valid(result), "PLS state must contain fitted weights, rotations, and coefficients")
def pls_canonical_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_components: int = 2,
    scale: bool = True,
    algorithm: str = "nipals",
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> PLSState:
    """Fit PLS canonical analysis and return immutable latent-variable state."""
    del copy
    return _pls_fit(
        X,
        y,
        n_components=n_components,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
        deflation_mode="canonical",
        mode="A",
        algorithm=algorithm,
    )


@register_atom(witness_cca_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda n_components, X, y: _pls_components_valid(n_components, X, y, deflation_mode="canonical"), "n_components exceeds the CCA rank bound")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be at least one")
@icontract.require(lambda tol: _positive_float(tol), "tol must be positive")
@icontract.ensure(lambda result: _pls_state_valid(result), "CCA state must contain fitted weights, rotations, and coefficients")
def cca_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_components: int = 2,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    copy: bool = True,
) -> PLSState:
    """Fit canonical correlation analysis and return immutable PLS state."""
    del copy
    return _pls_fit(
        X,
        y,
        n_components=n_components,
        scale=scale,
        max_iter=max_iter,
        tol=tol,
        deflation_mode="canonical",
        mode="B",
        algorithm="nipals",
    )
