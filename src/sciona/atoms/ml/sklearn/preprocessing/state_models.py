"""State containers for sklearn preprocessing atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KernelCentererState:
    """Learned means for centering kernel matrices in feature space."""

    k_fit_rows: NDArray[np.float64]
    k_fit_all: float
    n_features_in: int


@dataclass(frozen=True)
class MaxAbsScalerState:
    """Learned maximum-absolute-value scale factors for each feature."""

    scale: NDArray[np.float64]
    max_abs: NDArray[np.float64]
    n_features_in: int
    n_samples_seen: int


@dataclass(frozen=True)
class MinMaxScalerState:
    """Learned min/max scale factors for mapping features into a range."""

    min_: NDArray[np.float64]
    scale: NDArray[np.float64]
    data_min: NDArray[np.float64]
    data_max: NDArray[np.float64]
    data_range: NDArray[np.float64]
    feature_range: tuple[float, float]
    n_features_in: int
    n_samples_seen: int


@dataclass(frozen=True)
class RobustScalerState:
    """Learned medians and quantile-range scales for robust feature scaling."""

    center: NDArray[np.float64] | None
    scale: NDArray[np.float64] | None
    with_centering: bool
    with_scaling: bool
    quantile_range: tuple[float, float]
    unit_variance: bool
    n_features_in: int


@dataclass(frozen=True)
class StandardScalerState:
    """Learned means, variances, and scales for standard feature scaling."""

    mean: NDArray[np.float64] | None
    var: NDArray[np.float64] | None
    scale: NDArray[np.float64] | None
    n_samples_seen: NDArray[np.float64]
    with_mean: bool
    with_std: bool
    n_features_in: int


@dataclass(frozen=True)
class LabelEncoderState:
    """Learned sorted label classes for target encoding."""

    classes: NDArray[np.object_]


@dataclass(frozen=True)
class LabelBinarizerState:
    """Learned target classes and label-binarization configuration."""

    classes: NDArray[np.object_]
    y_type: str
    sparse_input: bool
    neg_label: int
    pos_label: int
    sparse_output: bool


@dataclass(frozen=True)
class MultiLabelBinarizerState:
    """Learned class ordering for multilabel indicator encoding."""

    classes: NDArray[np.object_]
    sparse_output: bool


@dataclass(frozen=True)
class PolynomialFeaturesState:
    """Learned polynomial expansion powers and feature-count metadata."""

    powers: NDArray[np.int_]
    n_features_in: int
    n_output_features: int
    min_degree: int
    max_degree: int
    interaction_only: bool
    include_bias: bool
    order: str


@dataclass(frozen=True)
class PowerTransformerState:
    """Learned per-feature power-transform lambdas and scaling statistics."""

    lambdas: NDArray[np.float64]
    method: str
    standardize: bool
    mean: NDArray[np.float64] | None
    scale: NDArray[np.float64] | None
    n_features_in: int
