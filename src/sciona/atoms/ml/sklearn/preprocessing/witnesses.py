"""Ghost witnesses for selected sklearn preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import (
    KernelCentererState,
    LabelBinarizerState,
    LabelEncoderState,
    MaxAbsScalerState,
    MinMaxScalerState,
    RobustScalerState,
    StandardScalerState,
)


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


def witness_kernel_centerer_fit(
    K: AbstractArray,
) -> AbstractArray:
    """Describe the per-training-sample kernel means learned during fitting."""
    n_samples, n_features = _check_2d(K)
    if n_samples != n_features:
        raise ValueError("kernel matrix must be square")
    return AbstractArray(shape=(n_samples,), dtype=K.dtype)


def witness_kernel_centerer_transform(
    K: AbstractArray,
    state: KernelCentererState,
    copy: bool = True,
) -> AbstractArray:
    """Describe centering a kernel block with fitted training-kernel means."""
    del copy
    n_samples, n_features = _check_2d(K)
    if n_features != state.n_features_in:
        raise ValueError("kernel columns must match fitted training samples")
    return AbstractArray(shape=(n_samples, n_features), dtype=K.dtype)


def witness_maxabs_scaler_partial_fit(
    X: AbstractArray,
    state: MaxAbsScalerState | None = None,
) -> AbstractArray:
    """Describe fitting per-feature maximum absolute values."""
    n_samples, n_features = _check_2d(X)
    if n_samples < 1:
        raise ValueError("X must contain at least one sample")
    if state is not None and state.n_features_in != n_features:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_features,), dtype="float64", min_val=0.0)


def witness_maxabs_scaler_fit(
    X: AbstractArray,
) -> AbstractArray:
    """Describe fresh fitting for MaxAbsScaler state."""
    return witness_maxabs_scaler_partial_fit(X, state=None)


def witness_maxabs_scaler_transform(
    X: AbstractArray,
    state: MaxAbsScalerState,
    copy: bool = True,
    clip: bool = False,
) -> AbstractArray:
    """Describe applying fitted maximum-absolute-value scaling."""
    del copy, clip
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_maxabs_scaler_inverse_transform(
    X: AbstractArray,
    state: MaxAbsScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe undoing fitted maximum-absolute-value scaling."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_minmax_scaler_partial_fit(
    X: AbstractArray,
    feature_range: tuple[float, float] = (0, 1),
    state: MinMaxScalerState | None = None,
) -> AbstractArray:
    """Describe fitting per-feature minima and maxima."""
    n_samples, n_features = _check_2d(X)
    if n_samples < 1:
        raise ValueError("X must contain at least one sample")
    if feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range minimum must be smaller than maximum")
    if state is not None and state.n_features_in != n_features:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_minmax_scaler_fit(
    X: AbstractArray,
    feature_range: tuple[float, float] = (0, 1),
) -> AbstractArray:
    """Describe fresh fitting for MinMaxScaler state."""
    return witness_minmax_scaler_partial_fit(X, feature_range=feature_range, state=None)


def witness_minmax_scaler_transform(
    X: AbstractArray,
    state: MinMaxScalerState,
    copy: bool = True,
    clip: bool = False,
) -> AbstractArray:
    """Describe applying fitted min-max scaling."""
    del copy, clip
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_minmax_scaler_inverse_transform(
    X: AbstractArray,
    state: MinMaxScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe undoing fitted min-max scaling."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_robust_scaler_fit(
    X: AbstractArray,
    *,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    unit_variance: bool = False,
) -> AbstractArray:
    """Describe fitting robust per-feature center and scale statistics."""
    del with_centering, with_scaling, unit_variance
    n_samples, n_features = _check_2d(X)
    if n_samples < 1:
        raise ValueError("X must contain at least one sample")
    q_min, q_max = quantile_range
    if not 0 <= q_min <= q_max <= 100:
        raise ValueError("invalid quantile range")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_robust_scaler_transform(
    X: AbstractArray,
    state: RobustScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe applying fitted robust center and scale statistics."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_robust_scaler_inverse_transform(
    X: AbstractArray,
    state: RobustScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe undoing fitted robust center and scale statistics."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_standard_scaler_partial_fit(
    X: AbstractArray,
    state: StandardScalerState | None = None,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe fitting mean and variance statistics for standard scaling."""
    del with_mean, with_std
    n_samples, n_features = _check_2d(X)
    if n_samples < 1:
        raise ValueError("X must contain at least one sample")
    if state is not None and state.n_features_in != n_features:
        raise ValueError("X feature count must match fitted state")
    if sample_weight is not None and tuple(sample_weight.shape) != (n_samples,):
        raise ValueError("sample_weight must have one value per sample")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_standard_scaler_fit(
    X: AbstractArray,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe fresh fitting for StandardScaler state."""
    return witness_standard_scaler_partial_fit(
        X,
        state=None,
        with_mean=with_mean,
        with_std=with_std,
        sample_weight=sample_weight,
    )


def witness_standard_scaler_transform(
    X: AbstractArray,
    state: StandardScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe applying fitted standard scaling."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_standard_scaler_inverse_transform(
    X: AbstractArray,
    state: StandardScalerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe undoing fitted standard scaling."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_label_encoder_fit(y: AbstractArray) -> AbstractArray:
    """Describe learning sorted unique labels from a target vector."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or a column vector")
    n_samples = int(y.shape[0])
    return AbstractArray(shape=(n_samples,), dtype=y.dtype)


def witness_label_encoder_fit_transform(y: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe learning classes and encoded targets in one pass."""
    classes = witness_label_encoder_fit(y)
    return classes, AbstractArray(shape=(int(y.shape[0]),), dtype="int64", min_val=0.0)


def witness_label_encoder_transform(
    y: AbstractArray,
    state: LabelEncoderState,
) -> AbstractArray:
    """Describe mapping labels to integer class positions."""
    del state
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or a column vector")
    return AbstractArray(shape=(int(y.shape[0]),), dtype="int64", min_val=0.0)


def witness_label_encoder_inverse_transform(
    y: AbstractArray,
    state: LabelEncoderState,
) -> AbstractArray:
    """Describe mapping integer class positions back to labels."""
    del state
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or a column vector")
    return AbstractArray(shape=(int(y.shape[0]),), dtype="object")


def witness_label_binarize(
    y: AbstractArray,
    *,
    classes: AbstractArray,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> AbstractArray:
    """Describe one-vs-all target-label binarization."""
    del sparse_output
    if neg_label >= pos_label:
        raise ValueError("neg_label must be strictly less than pos_label")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    n_outputs = 1 if int(classes.shape[0]) == 2 else int(classes.shape[0])
    return AbstractArray(shape=(int(y.shape[0]), n_outputs), dtype="int64", min_val=float(neg_label), max_val=float(pos_label))


def witness_label_binarizer_fit(
    y: AbstractArray,
    *,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> AbstractArray:
    """Describe learning label-binarizer classes and target type."""
    del sparse_output
    if neg_label >= pos_label:
        raise ValueError("neg_label must be strictly less than pos_label")
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    n_classes = int(y.shape[1]) if len(y.shape) == 2 else int(y.shape[0])
    return AbstractArray(shape=(n_classes,), dtype=y.dtype)


def witness_label_binarizer_transform(
    y: AbstractArray,
    state: LabelBinarizerState,
) -> AbstractArray:
    """Describe transforming labels with fitted binarizer state."""
    classes = AbstractArray(shape=(int(state.classes.shape[0]),), dtype="object")
    return witness_label_binarize(
        y,
        classes=classes,
        neg_label=state.neg_label,
        pos_label=state.pos_label,
        sparse_output=state.sparse_output,
    )


def witness_label_binarizer_fit_transform(
    y: AbstractArray,
    *,
    neg_label: int = 0,
    pos_label: int = 1,
    sparse_output: bool = False,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe fitting label-binarizer state and transforming labels."""
    classes = witness_label_binarizer_fit(
        y,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    transformed = witness_label_binarize(
        y,
        classes=classes,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    return classes, transformed


def witness_label_binarizer_inverse_transform(
    Y: AbstractArray,
    state: LabelBinarizerState,
    threshold: float | None = None,
) -> AbstractArray:
    """Describe inverse label-binarization output."""
    del threshold
    if len(Y.shape) != 2:
        raise ValueError("Y must be 2D")
    if state.y_type == "multilabel-indicator":
        return AbstractArray(shape=Y.shape, dtype="int64", min_val=0.0, max_val=1.0)
    return AbstractArray(shape=(int(Y.shape[0]),), dtype="object")


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
