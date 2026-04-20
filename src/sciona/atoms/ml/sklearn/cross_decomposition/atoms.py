"""Selected cross-decomposition atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip

from sciona.ghost.registry import register_atom

from .state_models import PLSSVDState
from .witnesses import witness_plssvd_fit


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
