"""Ghost witnesses for selected sklearn preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_2d(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_1d_or_2d(X: AbstractArray) -> tuple[int, ...]:
    if len(X.shape) not in {1, 2}:
        raise ValueError("X must be 1D or 2D")
    return tuple(int(dim) for dim in X.shape)


def _valid_norm(norm: str) -> bool:
    return norm in {"l1", "l2", "max"}


def witness_add_dummy_feature(X: AbstractArray, value: float = 1.0) -> AbstractArray:
    """Describe insertion of a leading constant feature column."""
    del value
    n_samples, n_features = _check_2d(X)
    return AbstractArray(shape=(n_samples, n_features + 1), dtype=X.dtype)


def witness_binarize(X: AbstractArray, threshold: float = 0.0, copy: bool = True) -> AbstractArray:
    """Describe elementwise thresholding to a binary matrix."""
    del threshold, copy
    return AbstractArray(shape=_check_2d(X), dtype=X.dtype, min_val=0.0, max_val=1.0)


def witness_binarizer_transform(
    X: AbstractArray,
    threshold: float = 0.0,
    copy: bool = True,
) -> AbstractArray:
    """Describe stateless Binarizer.transform output."""
    return witness_binarize(X, threshold=threshold, copy=copy)


def witness_normalize(
    X: AbstractArray,
    norm: str = "l2",
    *,
    axis: int = 1,
    copy: bool = True,
    return_norm: bool = False,
) -> AbstractArray | tuple[AbstractArray, AbstractArray]:
    """Describe normalization along rows or columns."""
    del copy
    n_samples, n_features = _check_2d(X)
    if not _valid_norm(norm):
        raise ValueError("norm must be 'l1', 'l2', or 'max'")
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    normalized = AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)
    if return_norm:
        norm_count = n_features if axis == 0 else n_samples
        return normalized, AbstractArray(shape=(norm_count,), dtype="float64", min_val=0.0)
    return normalized


def witness_normalizer_transform(
    X: AbstractArray,
    norm: str = "l2",
    copy: bool = True,
) -> AbstractArray:
    """Describe stateless Normalizer.transform output."""
    result = witness_normalize(X, norm=norm, axis=1, copy=copy, return_norm=False)
    if isinstance(result, tuple):
        return result[0]
    return result


def witness_scale(
    X: AbstractArray,
    *,
    axis: int = 0,
    with_mean: bool = True,
    with_std: bool = True,
    copy: bool = True,
) -> AbstractArray:
    """Describe mean-centering and variance scaling output."""
    del with_mean, with_std, copy
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    return AbstractArray(shape=_check_1d_or_2d(X), dtype=X.dtype)


def witness_maxabs_scale(
    X: AbstractArray,
    *,
    axis: int = 0,
    copy: bool = True,
) -> AbstractArray:
    """Describe maximum-absolute-value scaling output."""
    del copy
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    return AbstractArray(shape=_check_1d_or_2d(X), dtype=X.dtype)


def witness_minmax_scale(
    X: AbstractArray,
    feature_range: tuple[float, float] = (0, 1),
    *,
    axis: int = 0,
    copy: bool = True,
) -> AbstractArray:
    """Describe min-max scaling output."""
    del copy
    if feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range minimum must be smaller than maximum")
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    return AbstractArray(shape=_check_1d_or_2d(X), dtype=X.dtype)


def witness_robust_scale(
    X: AbstractArray,
    *,
    axis: int = 0,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    copy: bool = True,
    unit_variance: bool = False,
) -> AbstractArray:
    """Describe median and quantile-range scaling output."""
    del with_centering, with_scaling, copy, unit_variance
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    q_min, q_max = quantile_range
    if not 0 <= q_min <= q_max <= 100:
        raise ValueError("invalid quantile range")
    return AbstractArray(shape=_check_1d_or_2d(X), dtype=X.dtype)
