"""Ghost witnesses for selected sklearn preprocessing atoms."""

from __future__ import annotations

import math

from sciona.ghost.abstract import AbstractArray

from .state_models import (
    KernelCentererState,
    LabelBinarizerState,
    LabelEncoderState,
    MaxAbsScalerState,
    MinMaxScalerState,
    MultiLabelBinarizerState,
    PolynomialFeaturesState,
    PowerTransformerState,
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


def witness_multi_label_binarizer_fit(
    y: AbstractArray,
    *,
    classes: AbstractArray | None = None,
    sparse_output: bool = False,
) -> AbstractArray:
    """Describe learning the multilabel class ordering."""
    del sparse_output
    if classes is not None:
        if len(classes.shape) != 1:
            raise ValueError("classes must be 1D")
        return AbstractArray(shape=classes.shape, dtype=classes.dtype)
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must describe samples of label iterables")
    n_classes = int(y.shape[1]) if len(y.shape) == 2 else int(y.shape[0])
    return AbstractArray(shape=(n_classes,), dtype="object")


def witness_multi_label_binarizer_transform(
    y: AbstractArray,
    state: MultiLabelBinarizerState,
) -> AbstractArray:
    """Describe transforming label sets to an indicator matrix."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must describe samples of label iterables")
    return AbstractArray(shape=(int(y.shape[0]), int(state.classes.shape[0])), dtype="int64", min_val=0.0, max_val=1.0)


def witness_multi_label_binarizer_fit_transform(
    y: AbstractArray,
    *,
    classes: AbstractArray | None = None,
    sparse_output: bool = False,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe fitting multilabel classes and returning indicators."""
    fitted_classes = witness_multi_label_binarizer_fit(y, classes=classes, sparse_output=sparse_output)
    indicators = AbstractArray(shape=(int(y.shape[0]), int(fitted_classes.shape[0])), dtype="int64", min_val=0.0, max_val=1.0)
    return fitted_classes, indicators


def witness_multi_label_binarizer_inverse_transform(
    yt: AbstractArray,
    state: MultiLabelBinarizerState,
) -> AbstractArray:
    """Describe converting multilabel indicators back to label tuples."""
    if len(yt.shape) != 2:
        raise ValueError("yt must be 2D")
    if int(yt.shape[1]) != int(state.classes.shape[0]):
        raise ValueError("indicator width must match fitted classes")
    return AbstractArray(shape=(int(yt.shape[0]),), dtype="object")


def witness_polynomial_features_fit(
    X: AbstractArray,
    degree: int | tuple[int, int] = 2,
    *,
    interaction_only: bool = False,
    include_bias: bool = True,
    order: str = "C",
) -> AbstractArray:
    """Describe learning polynomial power rows for input features."""
    if order not in {"C", "F"}:
        raise ValueError("order must be 'C' or 'F'")
    _, n_features = _check_2d(X)
    min_degree, max_degree = _polynomial_degree_bounds(degree, include_bias)
    n_outputs = _polynomial_output_count(n_features, min_degree, max_degree, interaction_only, include_bias)
    return AbstractArray(shape=(n_outputs, n_features), dtype="int64")


def witness_polynomial_features_transform(
    X: AbstractArray,
    state: PolynomialFeaturesState,
) -> AbstractArray:
    """Describe polynomial feature expansion shape."""
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, state.n_output_features), dtype=X.dtype)


def witness_polynomial_features_fit_transform(
    X: AbstractArray,
    degree: int | tuple[int, int] = 2,
    *,
    interaction_only: bool = False,
    include_bias: bool = True,
    order: str = "C",
) -> tuple[AbstractArray, AbstractArray]:
    """Describe fitting polynomial powers and expanding features."""
    powers = witness_polynomial_features_fit(
        X,
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
        order=order,
    )
    n_samples, _ = _check_2d(X)
    return powers, AbstractArray(shape=(n_samples, int(powers.shape[0])), dtype=X.dtype)


def witness_power_transformer_fit(
    X: AbstractArray,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
) -> AbstractArray:
    """Describe learning per-feature power-transform lambdas."""
    del standardize
    if method not in {"yeo-johnson", "box-cox"}:
        raise ValueError("method must be 'yeo-johnson' or 'box-cox'")
    _, n_features = _check_2d(X)
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_power_transformer_transform(
    X: AbstractArray,
    state: PowerTransformerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe applying fitted power-transform lambdas."""
    del copy
    n_samples, n_features = _check_2d(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_power_transformer_inverse_transform(
    X: AbstractArray,
    state: PowerTransformerState,
    copy: bool = True,
) -> AbstractArray:
    """Describe undoing fitted power-transform lambdas."""
    return witness_power_transformer_transform(X, state, copy=copy)


def witness_power_transformer_fit_transform(
    X: AbstractArray,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
    copy: bool = True,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe fitting and applying power-transform lambdas."""
    del copy
    lambdas = witness_power_transformer_fit(X, method=method, standardize=standardize)
    n_samples, n_features = _check_2d(X)
    return lambdas, AbstractArray(shape=(n_samples, n_features), dtype=X.dtype)


def witness_power_transform(
    X: AbstractArray,
    method: str = "yeo-johnson",
    *,
    standardize: bool = True,
    copy: bool = True,
) -> AbstractArray:
    """Describe stateless fit-transform power transformation."""
    _, transformed = witness_power_transformer_fit_transform(
        X,
        method=method,
        standardize=standardize,
        copy=copy,
    )
    return transformed


def _polynomial_degree_bounds(degree: int | tuple[int, int], include_bias: bool) -> tuple[int, int]:
    if isinstance(degree, int):
        if degree == 0 and not include_bias:
            raise ValueError("degree zero without bias has empty output")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        return 0, degree
    if len(degree) != 2:
        raise ValueError("degree tuple must have length two")
    min_degree, max_degree = degree
    if min_degree < 0 or min_degree > max_degree:
        raise ValueError("degree bounds must satisfy 0 <= min <= max")
    if max_degree == 0 and not include_bias:
        raise ValueError("degree zero without bias has empty output")
    return int(min_degree), int(max_degree)


def _polynomial_output_count(
    n_features: int,
    min_degree: int,
    max_degree: int,
    interaction_only: bool,
    include_bias: bool,
) -> int:
    if interaction_only:
        total = sum(
            math.comb(n_features, degree)
            for degree in range(max(1, min_degree), min(max_degree, n_features) + 1)
        )
    else:
        total = math.comb(n_features + max_degree, max_degree) - 1
        if min_degree > 0:
            previous_degree = min_degree - 1
            total -= math.comb(n_features + previous_degree, previous_degree) - 1
    if include_bias:
        total += 1
    return total


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
