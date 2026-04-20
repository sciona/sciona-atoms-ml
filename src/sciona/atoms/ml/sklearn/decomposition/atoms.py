"""Selected decomposition atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip

from sciona.ghost.registry import register_atom

from .state_models import PCAState
from .witnesses import witness_pca_fit


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _has_enough_samples(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[0] >= 2)


def _has_positive_centered_variance(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return False
    centered = values - values.mean(axis=0)
    return bool(np.isfinite(centered).all() and np.sum(centered**2) > 0.0)


def _pca_components_valid(n_components: int | float | None, X: NDArray[np.float64]) -> bool:
    n_samples, n_features = np.asarray(X).shape
    rank_bound = min(n_samples, n_features)
    if n_components is None:
        return True
    if isinstance(n_components, bool):
        return False
    if isinstance(n_components, int):
        return 0 <= n_components <= rank_bound
    if isinstance(n_components, float):
        return 0.0 < n_components < 1.0
    return False


def _resolve_n_components(
    n_components: int | float | None,
    explained_variance_ratio: NDArray[np.float64],
    *,
    n_samples: int,
    n_features: int,
) -> int:
    if n_components is None:
        return min(n_samples, n_features)
    if isinstance(n_components, int):
        return int(n_components)
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    return int(np.searchsorted(ratio_cumsum, float(n_components), side="right") + 1)


def _pca_state_valid(state: PCAState) -> bool:
    return bool(
        state.components.shape == (state.n_components, state.n_features_in)
        and state.explained_variance.shape == (state.n_components,)
        and state.explained_variance_ratio.shape == (state.n_components,)
        and state.singular_values.shape == (state.n_components,)
        and state.mean.shape == (state.n_features_in,)
        and state.n_samples >= 2
        and state.n_components >= 0
        and state.svd_solver == "full"
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.explained_variance))
        and np.all(np.isfinite(state.explained_variance_ratio))
        and np.all(np.isfinite(state.singular_values))
        and np.all(np.isfinite(state.mean))
        and np.isfinite(state.noise_variance)
        and 0.0 <= float(np.sum(state.explained_variance_ratio)) <= 1.0 + 1e-12
    )


@register_atom(witness_pca_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _has_enough_samples(X), "X must contain at least two samples")
@icontract.require(lambda X: _has_positive_centered_variance(X), "X must have positive centered variance")
@icontract.require(lambda n_components, X: _pca_components_valid(n_components, X), "n_components must fit the full-SVD PCA rank bound")
@icontract.require(lambda svd_solver: svd_solver == "full", "only the full-SVD PCA solver is exposed")
@icontract.ensure(lambda result: _pca_state_valid(result), "PCA state must contain finite fitted components and variance metadata")
def pca_fit(
    X: NDArray[np.float64],
    n_components: int | float | None = None,
    *,
    whiten: bool = False,
    copy: bool = True,
    svd_solver: str = "full",
) -> PCAState:
    """Fit dense full-SVD PCA state from a sample-by-feature matrix."""
    del copy
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2)
    n_samples, n_features = checked_x.shape
    mean = np.mean(checked_x, axis=0)
    centered_x = np.asarray(checked_x, dtype=np.float64).copy()
    centered_x -= mean

    _, singular_values, vt = linalg.svd(centered_x, full_matrices=False)
    _, vt = svd_flip(None, vt, u_based_decision=False)
    explained_variance = (singular_values**2) / (n_samples - 1)
    total_var = float(np.sum(explained_variance))
    explained_variance_ratio = explained_variance / total_var

    resolved_components = _resolve_n_components(
        n_components,
        np.asarray(explained_variance_ratio, dtype=np.float64),
        n_samples=n_samples,
        n_features=n_features,
    )
    if resolved_components < min(n_features, n_samples):
        noise_variance = float(np.mean(explained_variance[resolved_components:]))
    else:
        noise_variance = 0.0

    return PCAState(
        components=np.asarray(vt[:resolved_components, :], dtype=np.float64).copy(),
        explained_variance=np.asarray(explained_variance[:resolved_components], dtype=np.float64).copy(),
        explained_variance_ratio=np.asarray(explained_variance_ratio[:resolved_components], dtype=np.float64).copy(),
        singular_values=np.asarray(singular_values[:resolved_components], dtype=np.float64).copy(),
        mean=np.asarray(mean, dtype=np.float64).copy(),
        noise_variance=noise_variance,
        n_samples=int(n_samples),
        n_components=int(resolved_components),
        n_features_in=int(n_features),
        whiten=bool(whiten),
        svd_solver=svd_solver,
    )
