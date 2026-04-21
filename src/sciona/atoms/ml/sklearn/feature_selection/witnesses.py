"""Ghost witnesses for sklearn feature-selection score atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import UnivariateSelectionState

DiscreteFeatureSpec = str | bool | tuple[int, ...] | tuple[bool, ...]


def _check_xy(X: AbstractArray, y: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have equal sample count")
    return int(X.shape[0]), int(X.shape[1])


def witness_mutual_info_continuous_continuous(
    x: AbstractArray,
    y: AbstractArray,
    *,
    n_neighbors: int = 3,
) -> AbstractArray:
    """Describe a scalar continuous-continuous mutual-information estimate."""
    _check_vector_pair(x, y)
    _check_n_neighbors(n_neighbors)
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_mutual_info_continuous_discrete(
    continuous: AbstractArray,
    discrete: AbstractArray,
    *,
    n_neighbors: int = 3,
) -> AbstractArray:
    """Describe a scalar continuous-discrete mutual-information estimate."""
    _check_vector_pair(continuous, discrete)
    _check_n_neighbors(n_neighbors)
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_mutual_info_pair(
    x: AbstractArray,
    y: AbstractArray,
    *,
    x_discrete: bool,
    y_discrete: bool,
    n_neighbors: int = 3,
) -> AbstractArray:
    """Describe a scalar pairwise mutual-information estimate."""
    del x_discrete, y_discrete
    _check_vector_pair(x, y)
    _check_n_neighbors(n_neighbors)
    return AbstractArray(shape=(), dtype="float64", min_val=0.0)


def witness_mutual_info_regression(
    X: AbstractArray,
    y: AbstractArray,
    *,
    discrete_features: DiscreteFeatureSpec = "auto",
    n_neighbors: int = 3,
    copy: bool = True,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe one mutual-information score per feature for regression."""
    del copy, random_state, n_jobs
    _n_samples, n_features = _check_xy(X, y)
    _check_discrete_features(discrete_features, n_features)
    _check_n_neighbors(n_neighbors)
    return AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)


def witness_mutual_info_classif(
    X: AbstractArray,
    y: AbstractArray,
    *,
    discrete_features: DiscreteFeatureSpec = "auto",
    n_neighbors: int = 3,
    copy: bool = True,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> AbstractArray:
    """Describe one mutual-information score per feature for classification."""
    del copy, random_state, n_jobs
    _n_samples, n_features = _check_xy(X, y)
    _check_discrete_features(discrete_features, n_features)
    _check_n_neighbors(n_neighbors)
    return AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)


def witness_f_classif(X: AbstractArray, y: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe ANOVA score and p-value vectors for class labels."""
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values


def witness_chi2(X: AbstractArray, y: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe chi-square score and p-value vectors for class labels."""
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values


def witness_r_regression(
    X: AbstractArray,
    y: AbstractArray,
    *,
    center: bool = True,
    force_finite: bool = True,
) -> AbstractArray:
    """Describe one Pearson correlation value per input feature."""
    del center, force_finite
    _n_samples, n_features = _check_xy(X, y)
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_f_regression(
    X: AbstractArray,
    y: AbstractArray,
    *,
    center: bool = True,
    force_finite: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe regression F-score and p-value vectors per feature."""
    del center, force_finite
    _n_samples, n_features = _check_xy(X, y)
    scores = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)
    p_values = AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0, max_val=1.0)
    return scores, p_values


def witness_select_k_best_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    k: int | str = 10,
) -> AbstractArray:
    """Describe fitting a top-k univariate feature selector."""
    _, n_features = _check_xy(X, y)
    _check_score_func(score_func)
    if not (k == "all" or (isinstance(k, int) and not isinstance(k, bool) and k >= 0)):
        raise ValueError("k must be a non-negative integer or 'all'")
    return AbstractArray(shape=(n_features,), dtype="bool")


def witness_select_percentile_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    percentile: float = 10.0,
) -> AbstractArray:
    """Describe fitting a percentile univariate feature selector."""
    _, n_features = _check_xy(X, y)
    _check_score_func(score_func)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must lie in [0, 100]")
    return AbstractArray(shape=(n_features,), dtype="bool")


def witness_select_fpr_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    alpha: float = 0.05,
) -> AbstractArray:
    """Describe fitting a false-positive-rate selector."""
    return _witness_alpha_selector_fit(X, y, score_func=score_func, alpha=alpha)


def witness_select_fdr_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    alpha: float = 0.05,
) -> AbstractArray:
    """Describe fitting a false-discovery-rate selector."""
    return _witness_alpha_selector_fit(X, y, score_func=score_func, alpha=alpha)


def witness_select_fwe_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    alpha: float = 0.05,
) -> AbstractArray:
    """Describe fitting a family-wise-error selector."""
    return _witness_alpha_selector_fit(X, y, score_func=score_func, alpha=alpha)


def witness_generic_univariate_select_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str = "f_classif",
    mode: str = "percentile",
    param: int | float | str = 1e-5,
) -> AbstractArray:
    """Describe fitting a univariate selector with a configurable strategy."""
    _, n_features = _check_xy(X, y)
    _check_score_func(score_func)
    if mode not in {"percentile", "k_best", "fpr", "fdr", "fwe"}:
        raise ValueError("unsupported selection mode")
    if mode == "k_best" and not (param == "all" or (isinstance(param, int) and not isinstance(param, bool) and param >= 0)):
        raise ValueError("k_best param must be a non-negative integer or 'all'")
    if mode == "percentile" and not (isinstance(param, (int, float)) and not isinstance(param, bool) and 0.0 <= float(param) <= 100.0):
        raise ValueError("percentile param must lie in [0, 100]")
    if mode in {"fpr", "fdr", "fwe"} and not (
        isinstance(param, (int, float)) and not isinstance(param, bool) and 0.0 <= float(param) <= 1.0
    ):
        raise ValueError("alpha param must lie in [0, 1]")
    return AbstractArray(shape=(n_features,), dtype="bool")


def witness_univariate_selection_transform(X: AbstractArray, state: UnivariateSelectionState) -> AbstractArray:
    """Describe retaining fitted univariate selector columns."""
    n_samples, n_features = _check_x(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted selector state")
    return AbstractArray(shape=(n_samples, int(state.support_mask.sum())), dtype=X.dtype)


def _check_x(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_vector_pair(x: AbstractArray, y: AbstractArray) -> None:
    if len(x.shape) != 1 or len(y.shape) != 1:
        raise ValueError("variables must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("variables must have equal sample count")


def _check_n_neighbors(n_neighbors: int) -> None:
    if not (isinstance(n_neighbors, int) and not isinstance(n_neighbors, bool) and n_neighbors >= 1):
        raise ValueError("n_neighbors must be positive")


def _check_discrete_features(discrete_features: DiscreteFeatureSpec, n_features: int) -> None:
    if discrete_features == "auto" or isinstance(discrete_features, bool):
        return
    if isinstance(discrete_features, tuple):
        if all(isinstance(item, bool) for item in discrete_features) and len(discrete_features) == n_features:
            return
        if all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item < n_features for item in discrete_features):
            return
    raise ValueError("discrete_features must match feature count")


def _check_score_func(score_func: str) -> None:
    if score_func not in {"f_classif", "chi2", "f_regression"}:
        raise ValueError("unsupported score_func")


def _witness_alpha_selector_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    score_func: str,
    alpha: float,
) -> AbstractArray:
    _, n_features = _check_xy(X, y)
    _check_score_func(score_func)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return AbstractArray(shape=(n_features,), dtype="bool")
