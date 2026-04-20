"""Selected decomposition atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import check_array
from sklearn.utils.extmath import _incremental_mean_and_var, _randomized_svd, fast_logdet, squared_norm, svd_flip
from sklearn.utils.validation import _check_psd_eigenvalues

from sciona.ghost.registry import register_atom

from .state_models import FactorAnalysisState, IncrementalPCAState, KernelPCAState, PCAState, TruncatedSVDState
from .witnesses import (
    witness_factor_analysis_covariance,
    witness_factor_analysis_fit,
    witness_factor_analysis_precision,
    witness_factor_analysis_score,
    witness_factor_analysis_score_samples,
    witness_factor_analysis_transform,
    witness_incremental_pca_inverse_transform,
    witness_incremental_pca_partial_fit,
    witness_incremental_pca_transform,
    witness_kernel_pca_fit,
    witness_kernel_pca_transform,
    witness_pca_fit,
    witness_truncated_svd_fit,
    witness_truncated_svd_inverse_transform,
    witness_truncated_svd_transform,
)


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


def _has_positive_feature_variance(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return False
    return bool(np.isfinite(values).all() and float(np.sum(np.var(values, axis=0))) > 0.0)


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


def _incremental_components_valid(
    n_components: int | None,
    X: NDArray[np.float64],
    state: IncrementalPCAState | None,
) -> bool:
    values = np.asarray(X)
    if values.ndim != 2:
        return False
    n_samples, n_features = values.shape
    if state is None:
        if n_samples < 2:
            return False
        if n_components is None:
            return True
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            return False
        return bool(1 <= n_components <= min(n_samples, n_features))
    if values.shape[1] != state.n_features_in:
        return False
    if n_components is None:
        return True
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and n_components == state.n_components)


def _incremental_batch_size_valid(batch_size: int | None) -> bool:
    return bool(batch_size is None or (isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size >= 1))


def _incremental_state_valid(state: IncrementalPCAState) -> bool:
    return bool(
        state.components.shape == (state.n_components, state.n_features_in)
        and state.explained_variance.shape == (state.n_components,)
        and state.explained_variance_ratio.shape == (state.n_components,)
        and state.singular_values.shape == (state.n_components,)
        and state.mean.shape == (state.n_features_in,)
        and state.var.shape == (state.n_features_in,)
        and state.n_samples_seen >= 2
        and 1 <= state.n_components <= state.n_features_in
        and _incremental_batch_size_valid(state.batch_size)
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.explained_variance))
        and np.all(np.isfinite(state.explained_variance_ratio))
        and np.all(np.isfinite(state.singular_values))
        and np.all(np.isfinite(state.mean))
        and np.all(np.isfinite(state.var))
        and np.isfinite(state.noise_variance)
        and np.all(state.explained_variance >= 0.0)
        and np.all(state.explained_variance_ratio >= 0.0)
        and np.all(state.singular_values >= 0.0)
        and np.all(state.var >= 0.0)
        and 0.0 <= float(np.sum(state.explained_variance_ratio)) <= 1.0 + 1e-12
    )


def _truncated_svd_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        return False
    values = np.asarray(X)
    if values.ndim != 2:
        return False
    return bool(1 <= n_components <= min(values.shape))


def _truncated_svd_options_valid(
    algorithm: str,
    n_iter: int,
    n_oversamples: int,
    power_iteration_normalizer: str,
    random_state: int | None,
    tol: float,
) -> bool:
    return bool(
        algorithm == "randomized"
        and isinstance(n_iter, int)
        and not isinstance(n_iter, bool)
        and n_iter >= 0
        and isinstance(n_oversamples, int)
        and not isinstance(n_oversamples, bool)
        and n_oversamples >= 1
        and power_iteration_normalizer in {"auto", "OR", "LU", "none"}
        and (random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))
        and tol >= 0.0
    )


def _truncated_svd_state_valid(state: TruncatedSVDState) -> bool:
    return bool(
        state.components.shape == (state.n_components, state.n_features_in)
        and state.explained_variance.shape == (state.n_components,)
        and state.explained_variance_ratio.shape == (state.n_components,)
        and state.singular_values.shape == (state.n_components,)
        and state.n_components >= 1
        and state.n_features_in >= 2
        and state.algorithm == "randomized"
        and state.n_iter >= 0
        and state.n_oversamples >= 1
        and state.power_iteration_normalizer in {"auto", "OR", "LU", "none"}
        and (state.random_state is None or isinstance(state.random_state, int))
        and state.tol >= 0.0
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.explained_variance))
        and np.all(np.isfinite(state.explained_variance_ratio))
        and np.all(np.isfinite(state.singular_values))
        and np.all(state.explained_variance >= 0.0)
        and np.all(state.explained_variance_ratio >= 0.0)
        and np.all(state.singular_values >= 0.0)
        and 0.0 <= float(np.sum(state.explained_variance_ratio)) <= 1.0 + 1e-12
    )


def _kernel_pca_components_valid(n_components: int, X: NDArray[np.float64]) -> bool:
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        return False
    return bool(n_components >= 1 and np.asarray(X).ndim == 2)


def _kernel_pca_options_valid(
    kernel: str,
    gamma: float | None,
    degree: float,
    coef0: float,
    alpha: float,
    fit_inverse_transform: bool,
    eigen_solver: str,
    tol: float,
    max_iter: int | None,
    iterated_power: int | str,
    remove_zero_eig: bool,
    random_state: int | None,
    copy_X: bool,
    n_jobs: int | None,
) -> bool:
    return bool(
        kernel == "linear"
        and (gamma is None or gamma > 0.0)
        and degree >= 0.0
        and np.isfinite(coef0)
        and alpha >= 0.0
        and fit_inverse_transform is False
        and eigen_solver == "dense"
        and tol >= 0.0
        and (max_iter is None or (isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter >= 1))
        and (
            iterated_power == "auto"
            or (isinstance(iterated_power, int) and not isinstance(iterated_power, bool) and iterated_power >= 0)
        )
        and remove_zero_eig is False
        and (random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))
        and isinstance(copy_X, bool)
        and (n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool)))
    )


def _kernel_pca_state_valid(state: KernelPCAState) -> bool:
    return bool(
        state.eigenvalues.shape == (state.n_components,)
        and state.eigenvectors.shape == (state.X_fit.shape[0], state.n_components)
        and state.X_fit.ndim == 2
        and state.kernel_centerer_rows.shape == (state.X_fit.shape[0],)
        and state.n_features_in == state.X_fit.shape[1]
        and 1 <= state.n_components <= state.X_fit.shape[0]
        and state.gamma > 0.0
        and state.kernel == "linear"
        and state.eigen_solver == "dense"
        and state.remove_zero_eig is False
        and state.fit_inverse_transform is False
        and np.all(np.isfinite(state.eigenvalues))
        and np.all(state.eigenvalues >= 0.0)
        and np.all(np.isfinite(state.eigenvectors))
        and np.all(np.isfinite(state.X_fit))
        and np.all(np.isfinite(state.kernel_centerer_rows))
        and np.isfinite(state.kernel_centerer_all)
    )


def _factor_components_valid(n_components: int | None, X: NDArray[np.float64]) -> bool:
    if n_components is None:
        return True
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        return False
    values = np.asarray(X)
    if values.ndim != 2:
        return False
    return bool(0 <= n_components <= values.shape[1])


def _factor_noise_init_valid(noise_variance_init: tuple[float, ...] | None, X: NDArray[np.float64]) -> bool:
    if noise_variance_init is None:
        return True
    values = np.asarray(noise_variance_init, dtype=np.float64)
    n_features = np.asarray(X).shape[1]
    return bool(values.ndim == 1 and values.shape[0] == n_features and np.all(np.isfinite(values)) and np.all(values > 0.0))


def _factor_options_valid(
    tol: float,
    max_iter: int,
    svd_method: str,
    iterated_power: int,
    rotation: None,
    random_state: int | None,
) -> bool:
    return bool(
        tol >= 0.0
        and isinstance(max_iter, int)
        and not isinstance(max_iter, bool)
        and max_iter >= 1
        and svd_method == "lapack"
        and isinstance(iterated_power, int)
        and not isinstance(iterated_power, bool)
        and iterated_power >= 0
        and rotation is None
        and (random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))
    )


def _factor_state_valid(state: FactorAnalysisState) -> bool:
    return bool(
        state.components.shape == (state.n_components, state.n_features_in)
        and state.noise_variance.shape == (state.n_features_in,)
        and state.mean.shape == (state.n_features_in,)
        and state.loglike.ndim == 1
        and state.loglike.shape[0] == state.n_iter
        and 0 <= state.n_components <= state.n_features_in
        and 1 <= state.n_iter <= state.max_iter
        and state.tol >= 0.0
        and state.max_iter >= 1
        and state.svd_method == "lapack"
        and state.rotation is None
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.noise_variance))
        and np.all(state.noise_variance > 0.0)
        and np.all(np.isfinite(state.mean))
        and np.all(np.isfinite(state.loglike))
    )


def _feature_count_matches(X: NDArray[np.float64], state: TruncatedSVDState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _component_count_matches(X: NDArray[np.float64], state: TruncatedSVDState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_components)


def _transformed_valid(result: NDArray[np.float64], state: TruncatedSVDState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_components and np.all(np.isfinite(values)))


def _inverse_transformed_valid(result: NDArray[np.float64], state: TruncatedSVDState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in and np.all(np.isfinite(values)))


def _incremental_feature_count_matches(X: NDArray[np.float64], state: IncrementalPCAState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _incremental_component_count_matches(X: NDArray[np.float64], state: IncrementalPCAState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_components)


def _incremental_transformed_valid(result: NDArray[np.float64], state: IncrementalPCAState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_components and np.all(np.isfinite(values)))


def _incremental_inverse_valid(result: NDArray[np.float64], state: IncrementalPCAState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in and np.all(np.isfinite(values)))


def _kernel_pca_feature_count_matches(X: NDArray[np.float64], state: KernelPCAState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _kernel_pca_transform_valid(result: NDArray[np.float64], state: KernelPCAState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_components and np.all(np.isfinite(values)))


def _factor_feature_count_matches(X: NDArray[np.float64], state: FactorAnalysisState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _factor_transform_valid(result: NDArray[np.float64], state: FactorAnalysisState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape[1] == state.n_components and np.all(np.isfinite(values)))


def _factor_square_matrix_valid(result: NDArray[np.float64], state: FactorAnalysisState) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (state.n_features_in, state.n_features_in)
        and np.all(np.isfinite(values))
        and np.allclose(values, values.T)
    )


def _factor_score_samples_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 1 and values.shape[0] == np.asarray(X).shape[0] and np.all(np.isfinite(values)))


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


@register_atom(witness_incremental_pca_partial_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _has_positive_feature_variance(X), "X must have positive feature variance")
@icontract.require(lambda n_components, X, state: _incremental_components_valid(n_components, X, state), "n_components must fit the current incremental PCA state")
@icontract.require(lambda batch_size: _incremental_batch_size_valid(batch_size), "batch_size must be positive when provided")
@icontract.ensure(lambda result: _incremental_state_valid(result), "incremental PCA state must contain finite running components")
def incremental_pca_partial_fit(
    X: NDArray[np.float64],
    n_components: int | None = None,
    *,
    state: IncrementalPCAState | None = None,
    whiten: bool = False,
    copy: bool = True,
    batch_size: int | None = None,
) -> IncrementalPCAState:
    """Update dense incremental PCA state from one sample batch."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, copy=copy)
    n_samples, n_features = checked_x.shape
    if state is None:
        resolved_components = min(n_samples, n_features) if n_components is None else int(n_components)
        prior_samples_seen = 0
        prior_mean: float | NDArray[np.float64] = 0.0
        prior_var: float | NDArray[np.float64] = 0.0
    else:
        resolved_components = state.n_components if n_components is None else int(n_components)
        prior_samples_seen = state.n_samples_seen
        prior_mean = state.mean
        prior_var = state.var
        whiten = state.whiten
        batch_size = state.batch_size

    col_mean, col_var, n_total_samples_by_feature = _incremental_mean_and_var(
        checked_x,
        last_mean=prior_mean,
        last_variance=prior_var,
        last_sample_count=np.repeat(prior_samples_seen, n_features),
    )
    n_total_samples = int(n_total_samples_by_feature[0])

    if prior_samples_seen == 0:
        svd_input = np.asarray(checked_x - col_mean, dtype=np.float64)
    else:
        col_batch_mean = np.mean(checked_x, axis=0)
        centered_batch = checked_x - col_batch_mean
        mean_correction = np.sqrt((prior_samples_seen / n_total_samples) * n_samples) * (state.mean - col_batch_mean)
        svd_input = np.vstack(
            (
                state.singular_values.reshape((-1, 1)) * state.components,
                centered_batch,
                mean_correction,
            )
        )

    _, singular_values, vt = linalg.svd(svd_input, full_matrices=False, check_finite=False)
    _, vt = svd_flip(None, vt, u_based_decision=False)
    explained_variance = singular_values**2 / (n_total_samples - 1)
    explained_variance_ratio = singular_values**2 / np.sum(col_var * n_total_samples)

    if resolved_components not in (n_samples, n_features):
        noise_variance = float(np.mean(explained_variance[resolved_components:]))
    else:
        noise_variance = 0.0

    return IncrementalPCAState(
        components=np.asarray(vt[:resolved_components], dtype=np.float64).copy(),
        explained_variance=np.asarray(explained_variance[:resolved_components], dtype=np.float64).copy(),
        explained_variance_ratio=np.asarray(explained_variance_ratio[:resolved_components], dtype=np.float64).copy(),
        singular_values=np.asarray(singular_values[:resolved_components], dtype=np.float64).copy(),
        mean=np.asarray(col_mean, dtype=np.float64).copy(),
        var=np.asarray(col_var, dtype=np.float64).copy(),
        noise_variance=float(noise_variance),
        n_samples_seen=int(n_total_samples),
        n_components=int(resolved_components),
        n_features_in=int(n_features),
        whiten=bool(whiten),
        batch_size=batch_size,
    )


@register_atom(witness_incremental_pca_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _incremental_feature_count_matches(X, state), "X feature count must match fitted incremental PCA state")
@icontract.require(lambda state: _incremental_state_valid(state), "state must be a fitted incremental PCA state")
@icontract.ensure(lambda result, state: _incremental_transformed_valid(result, state), "projection must be a finite component matrix")
def incremental_pca_transform(
    X: NDArray[np.float64],
    state: IncrementalPCAState,
) -> NDArray[np.float64]:
    """Project samples onto fitted incremental PCA components."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    transformed = np.dot(checked_x, state.components.T)
    transformed -= np.reshape(state.mean, (1, -1)) @ state.components.T
    if state.whiten:
        scale = np.sqrt(state.explained_variance).copy()
        min_scale = np.finfo(scale.dtype).eps
        scale[scale < min_scale] = min_scale
        transformed /= scale
    return np.asarray(transformed, dtype=np.float64)


@register_atom(witness_incremental_pca_inverse_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _incremental_component_count_matches(X, state), "X width must match fitted component count")
@icontract.require(lambda state: _incremental_state_valid(state), "state must be a fitted incremental PCA state")
@icontract.ensure(lambda result, state: _incremental_inverse_valid(result, state), "reconstruction must be a finite feature matrix")
def incremental_pca_inverse_transform(
    X: NDArray[np.float64],
    state: IncrementalPCAState,
) -> NDArray[np.float64]:
    """Map incremental PCA coordinates back to feature space."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    if state.whiten:
        scaled_components = np.sqrt(state.explained_variance[:, np.newaxis]) * state.components
        return np.asarray(np.dot(checked_x, scaled_components) + state.mean, dtype=np.float64)
    return np.asarray(np.dot(checked_x, state.components) + state.mean, dtype=np.float64)


@register_atom(witness_truncated_svd_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _has_enough_samples(X), "X must contain at least two samples")
@icontract.require(lambda X: np.asarray(X).shape[1] >= 2, "X must contain at least two features")
@icontract.require(lambda X: _has_positive_feature_variance(X), "X must have positive feature variance")
@icontract.require(lambda n_components, X: _truncated_svd_components_valid(n_components, X), "n_components must fit the dense randomized rank bound")
@icontract.require(
    lambda algorithm, n_iter, n_oversamples, power_iteration_normalizer, random_state, tol: _truncated_svd_options_valid(
        algorithm,
        n_iter,
        n_oversamples,
        power_iteration_normalizer,
        random_state,
        tol,
    ),
    "randomized truncated SVD options must be supported",
)
@icontract.ensure(lambda result: _truncated_svd_state_valid(result), "truncated SVD state must contain finite fitted factors")
def truncated_svd_fit(
    X: NDArray[np.float64],
    n_components: int = 2,
    *,
    algorithm: str = "randomized",
    n_iter: int = 5,
    n_oversamples: int = 10,
    power_iteration_normalizer: str = "auto",
    random_state: int | None = None,
    tol: float = 0.0,
) -> TruncatedSVDState:
    """Fit dense randomized truncated SVD state without centering inputs."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2, ensure_min_features=2)
    u, singular_values, vt = _randomized_svd(
        checked_x,
        n_components,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
        flip_sign=False,
    )
    _, vt = svd_flip(u, vt, u_based_decision=False)
    transformed = np.dot(checked_x, vt.T)
    explained_variance = np.var(transformed, axis=0)
    total_variance = float(np.var(checked_x, axis=0).sum())
    explained_variance_ratio = explained_variance / total_variance

    return TruncatedSVDState(
        components=np.asarray(vt, dtype=np.float64).copy(),
        explained_variance=np.asarray(explained_variance, dtype=np.float64).copy(),
        explained_variance_ratio=np.asarray(explained_variance_ratio, dtype=np.float64).copy(),
        singular_values=np.asarray(singular_values, dtype=np.float64).copy(),
        n_components=int(vt.shape[0]),
        n_features_in=int(checked_x.shape[1]),
        algorithm=algorithm,
        n_iter=int(n_iter),
        n_oversamples=int(n_oversamples),
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
        tol=float(tol),
    )


@register_atom(witness_truncated_svd_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted truncated SVD state")
@icontract.require(lambda state: _truncated_svd_state_valid(state), "state must be a fitted truncated SVD state")
@icontract.ensure(lambda result, state: _transformed_valid(result, state), "projection must be a finite component matrix")
def truncated_svd_transform(
    X: NDArray[np.float64],
    state: TruncatedSVDState,
) -> NDArray[np.float64]:
    """Project samples onto fitted truncated SVD components."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.components.T), dtype=np.float64)


@register_atom(witness_truncated_svd_inverse_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _component_count_matches(X, state), "X width must match fitted component count")
@icontract.require(lambda state: _truncated_svd_state_valid(state), "state must be a fitted truncated SVD state")
@icontract.ensure(lambda result, state: _inverse_transformed_valid(result, state), "reconstruction must be a finite feature matrix")
def truncated_svd_inverse_transform(
    X: NDArray[np.float64],
    state: TruncatedSVDState,
) -> NDArray[np.float64]:
    """Map truncated SVD component coordinates back to feature space."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    return np.asarray(np.dot(checked_x, state.components), dtype=np.float64)


@register_atom(witness_kernel_pca_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _has_enough_samples(X), "X must contain at least two samples")
@icontract.require(lambda X: _has_positive_centered_variance(X), "X must have positive centered variance")
@icontract.require(lambda n_components, X: _kernel_pca_components_valid(n_components, X), "n_components must be a positive integer")
@icontract.require(
    lambda kernel, gamma, degree, coef0, alpha, fit_inverse_transform, eigen_solver, tol, max_iter, iterated_power, remove_zero_eig, random_state, copy_X, n_jobs: _kernel_pca_options_valid(
        kernel,
        gamma,
        degree,
        coef0,
        alpha,
        fit_inverse_transform,
        eigen_solver,
        tol,
        max_iter,
        iterated_power,
        remove_zero_eig,
        random_state,
        copy_X,
        n_jobs,
    ),
    "KernelPCA options must select the dense linear-kernel path",
)
@icontract.ensure(lambda result: _kernel_pca_state_valid(result), "KernelPCA state must contain a finite centered-kernel eigensystem")
def kernel_pca_fit(
    X: NDArray[np.float64],
    n_components: int = 2,
    *,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: float = 3.0,
    coef0: float = 1.0,
    alpha: float = 1.0,
    fit_inverse_transform: bool = False,
    eigen_solver: str = "dense",
    tol: float = 0.0,
    max_iter: int | None = None,
    iterated_power: int | str = "auto",
    remove_zero_eig: bool = False,
    random_state: int | None = None,
    copy_X: bool = True,
    n_jobs: int | None = None,
) -> KernelPCAState:
    """Fit dense linear KernelPCA state from a sample-by-feature matrix."""
    del degree, coef0, alpha, fit_inverse_transform, tol, max_iter, iterated_power, random_state, n_jobs
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2, copy=copy_X)
    n_samples, n_features = checked_x.shape
    resolved_components = min(n_samples, int(n_components))
    gamma_value = 1.0 / n_features if gamma is None else float(gamma)

    kernel_matrix = np.asarray(np.dot(checked_x, checked_x.T), dtype=np.float64)
    centerer_rows = np.sum(kernel_matrix, axis=0) / n_samples
    centerer_all = float(np.sum(centerer_rows) / n_samples)
    centered_kernel = kernel_matrix.copy()
    predicted_column_means = (np.sum(centered_kernel, axis=1) / n_samples)[:, np.newaxis]
    centered_kernel -= centerer_rows
    centered_kernel -= predicted_column_means
    centered_kernel += centerer_all

    eigenvalues, eigenvectors = linalg.eigh(
        centered_kernel,
        subset_by_index=(n_samples - resolved_components, n_samples - 1),
    )
    eigenvalues = _check_psd_eigenvalues(eigenvalues, enable_warnings=False)
    eigenvectors, _ = svd_flip(u=eigenvectors, v=None)
    indices = eigenvalues.argsort()[::-1]

    return KernelPCAState(
        eigenvalues=np.asarray(eigenvalues[indices], dtype=np.float64).copy(),
        eigenvectors=np.asarray(eigenvectors[:, indices], dtype=np.float64).copy(),
        X_fit=np.asarray(checked_x, dtype=np.float64).copy(),
        kernel_centerer_rows=np.asarray(centerer_rows, dtype=np.float64).copy(),
        kernel_centerer_all=centerer_all,
        n_components=int(resolved_components),
        n_features_in=int(n_features),
        gamma=gamma_value,
        kernel=kernel,
        eigen_solver=eigen_solver,
        remove_zero_eig=bool(remove_zero_eig),
        fit_inverse_transform=False,
    )


@register_atom(witness_kernel_pca_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _kernel_pca_feature_count_matches(X, state), "X feature count must match fitted KernelPCA state")
@icontract.require(lambda state: _kernel_pca_state_valid(state), "state must be a fitted dense linear KernelPCA state")
@icontract.ensure(lambda result, state: _kernel_pca_transform_valid(result, state), "projection must be a finite component matrix")
def kernel_pca_transform(
    X: NDArray[np.float64],
    state: KernelPCAState,
) -> NDArray[np.float64]:
    """Project samples with a fitted dense linear KernelPCA state."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    kernel_matrix = np.asarray(np.dot(checked_x, state.X_fit.T), dtype=np.float64)
    predicted_column_means = (np.sum(kernel_matrix, axis=1) / state.X_fit.shape[0])[:, np.newaxis]
    kernel_matrix -= state.kernel_centerer_rows
    kernel_matrix -= predicted_column_means
    kernel_matrix += state.kernel_centerer_all

    non_zero_indices = np.flatnonzero(state.eigenvalues)
    scaled_eigenvectors = np.zeros_like(state.eigenvectors)
    scaled_eigenvectors[:, non_zero_indices] = (
        state.eigenvectors[:, non_zero_indices] / np.sqrt(state.eigenvalues[non_zero_indices])
    )
    return np.asarray(np.dot(kernel_matrix, scaled_eigenvectors), dtype=np.float64)


@register_atom(witness_factor_analysis_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _has_enough_samples(X), "X must contain at least two samples")
@icontract.require(lambda X: np.asarray(X).shape[1] >= 1, "X must contain at least one feature")
@icontract.require(lambda X: _has_positive_feature_variance(X), "X must have positive feature variance")
@icontract.require(lambda n_components, X: _factor_components_valid(n_components, X), "n_components must fit the feature count")
@icontract.require(lambda noise_variance_init, X: _factor_noise_init_valid(noise_variance_init, X), "noise variances must be positive per-feature values")
@icontract.require(
    lambda tol, max_iter, svd_method, iterated_power, rotation, random_state: _factor_options_valid(
        tol,
        max_iter,
        svd_method,
        iterated_power,
        rotation,
        random_state,
    ),
    "factor analysis options must be supported",
)
@icontract.ensure(lambda result: _factor_state_valid(result), "factor analysis state must contain finite loading and noise estimates")
def factor_analysis_fit(
    X: NDArray[np.float64],
    n_components: int | None = None,
    *,
    tol: float = 1e-2,
    copy: bool = True,
    max_iter: int = 1000,
    noise_variance_init: tuple[float, ...] | None = None,
    svd_method: str = "lapack",
    iterated_power: int = 3,
    rotation: None = None,
    random_state: int | None = 0,
) -> FactorAnalysisState:
    """Fit dense Lapack factor analysis loading and noise state."""
    del copy, iterated_power, random_state
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, ensure_min_samples=2, copy=True)
    n_samples, n_features = checked_x.shape
    resolved_components = n_features if n_components is None else int(n_components)
    mean = np.mean(checked_x, axis=0)
    centered_x = np.asarray(checked_x - mean, dtype=np.float64)

    nsqrt = np.sqrt(float(n_samples))
    loglike_constant = n_features * np.log(2.0 * np.pi) + resolved_components
    variance = np.var(centered_x, axis=0)
    if noise_variance_init is None:
        psi = np.ones(n_features, dtype=np.float64)
    else:
        psi = np.asarray(noise_variance_init, dtype=np.float64).copy()

    loglike: list[float] = []
    old_loglike = -np.inf
    small = 1e-12
    components = np.empty((resolved_components, n_features), dtype=np.float64)
    iteration = 0
    for iteration in range(max_iter):
        sqrt_psi = np.sqrt(psi) + small
        _, singular_values, vt = linalg.svd(centered_x / (sqrt_psi * nsqrt), full_matrices=False, check_finite=False)
        unexplained_variance = squared_norm(singular_values[resolved_components:])
        selected_singular_values = singular_values[:resolved_components]
        selected_vt = vt[:resolved_components]
        squared_singular_values = selected_singular_values**2
        components = np.sqrt(np.maximum(squared_singular_values - 1.0, 0.0))[:, np.newaxis] * selected_vt
        components *= sqrt_psi

        current_loglike = loglike_constant
        current_loglike += float(np.sum(np.log(squared_singular_values)))
        current_loglike += float(unexplained_variance + np.sum(np.log(psi)))
        current_loglike *= -n_samples / 2.0
        loglike.append(float(current_loglike))
        if (current_loglike - old_loglike) < tol:
            break
        old_loglike = current_loglike
        psi = np.maximum(variance - np.sum(components**2, axis=0), small)

    return FactorAnalysisState(
        components=np.asarray(components, dtype=np.float64).copy(),
        noise_variance=np.asarray(psi, dtype=np.float64).copy(),
        mean=np.asarray(mean, dtype=np.float64).copy(),
        loglike=np.asarray(loglike, dtype=np.float64).copy(),
        n_iter=int(iteration + 1),
        n_components=int(resolved_components),
        n_features_in=int(n_features),
        tol=float(tol),
        max_iter=int(max_iter),
        svd_method=svd_method,
        rotation=rotation,
    )


@register_atom(witness_factor_analysis_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _factor_feature_count_matches(X, state), "X feature count must match fitted factor analysis state")
@icontract.require(lambda state: _factor_state_valid(state), "state must be a fitted factor analysis state")
@icontract.ensure(lambda result, state: _factor_transform_valid(result, state), "latent factors must be a finite component matrix")
def factor_analysis_transform(
    X: NDArray[np.float64],
    state: FactorAnalysisState,
) -> NDArray[np.float64]:
    """Compute expected latent factor means for samples."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    identity = np.eye(state.n_components)
    centered_x = checked_x - state.mean
    weighted_components = state.components / state.noise_variance
    latent_covariance = linalg.inv(identity + np.dot(weighted_components, state.components.T))
    projected = np.dot(centered_x, weighted_components.T)
    return np.asarray(np.dot(projected, latent_covariance), dtype=np.float64)


@register_atom(witness_factor_analysis_covariance)
@icontract.require(lambda state: _factor_state_valid(state), "state must be a fitted factor analysis state")
@icontract.ensure(lambda result, state: _factor_square_matrix_valid(result, state), "covariance must be a finite symmetric feature matrix")
def factor_analysis_covariance(state: FactorAnalysisState) -> NDArray[np.float64]:
    """Compute covariance implied by factor loadings and diagonal noise."""
    covariance = np.dot(state.components.T, state.components)
    covariance.flat[:: len(covariance) + 1] += state.noise_variance
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_factor_analysis_precision)
@icontract.require(lambda state: _factor_state_valid(state), "state must be a fitted factor analysis state")
@icontract.ensure(lambda result, state: _factor_square_matrix_valid(result, state), "precision must be a finite symmetric feature matrix")
def factor_analysis_precision(state: FactorAnalysisState) -> NDArray[np.float64]:
    """Compute inverse covariance implied by fitted factor state."""
    if state.n_components == 0:
        return np.diag(1.0 / state.noise_variance).astype(np.float64)
    if state.n_components == state.n_features_in:
        return np.asarray(linalg.inv(factor_analysis_covariance(state)), dtype=np.float64)

    components = state.components
    precision = np.dot(components / state.noise_variance, components.T)
    precision.flat[:: len(precision) + 1] += 1.0
    precision = np.dot(components.T, np.dot(linalg.inv(precision), components))
    precision /= state.noise_variance[:, np.newaxis]
    precision /= -state.noise_variance[np.newaxis, :]
    precision.flat[:: len(precision) + 1] += 1.0 / state.noise_variance
    return np.asarray(precision, dtype=np.float64)


@register_atom(witness_factor_analysis_score_samples)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _factor_feature_count_matches(X, state), "X feature count must match fitted factor analysis state")
@icontract.require(lambda state: _factor_state_valid(state), "state must be a fitted factor analysis state")
@icontract.ensure(lambda result, X: _factor_score_samples_valid(result, X), "sample scores must be finite per-row values")
def factor_analysis_score_samples(
    X: NDArray[np.float64],
    state: FactorAnalysisState,
) -> NDArray[np.float64]:
    """Compute per-sample log likelihood under fitted factor state."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    centered_x = checked_x - state.mean
    precision = factor_analysis_precision(state)
    n_features = checked_x.shape[1]
    log_like = -0.5 * (centered_x * np.dot(centered_x, precision)).sum(axis=1)
    log_like -= 0.5 * (n_features * np.log(2.0 * np.pi) - fast_logdet(precision))
    return np.asarray(log_like, dtype=np.float64)


@register_atom(witness_factor_analysis_score)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _factor_feature_count_matches(X, state), "X feature count must match fitted factor analysis state")
@icontract.require(lambda state: _factor_state_valid(state), "state must be a fitted factor analysis state")
@icontract.ensure(lambda result: bool(np.isfinite(result)), "score must be finite")
def factor_analysis_score(
    X: NDArray[np.float64],
    state: FactorAnalysisState,
) -> float:
    """Compute average sample log likelihood under fitted factor state."""
    return float(np.mean(factor_analysis_score_samples(X, state)))
