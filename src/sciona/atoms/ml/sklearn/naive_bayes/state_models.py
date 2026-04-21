"""State containers for sklearn naive Bayes atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianNBState:
    """Learned dense Gaussian naive Bayes class statistics."""

    classes: NDArray[np.int64]
    class_count: NDArray[np.float64]
    class_prior: NDArray[np.float64]
    theta: NDArray[np.float64]
    var: NDArray[np.float64]
    epsilon: float
    n_features_in: int


@dataclass(frozen=True)
class MultinomialNBState:
    """Learned dense multinomial naive Bayes counts and log probabilities."""

    classes: NDArray[np.int64]
    class_count: NDArray[np.float64]
    feature_count: NDArray[np.float64]
    class_log_prior: NDArray[np.float64]
    feature_log_prob: NDArray[np.float64]
    alpha: float
    fit_prior: bool
    n_features_in: int


@dataclass(frozen=True)
class ComplementNBState:
    """Learned dense complement naive Bayes counts and class weights."""

    classes: NDArray[np.int64]
    class_count: NDArray[np.float64]
    feature_count: NDArray[np.float64]
    feature_all: NDArray[np.float64]
    class_log_prior: NDArray[np.float64]
    feature_log_prob: NDArray[np.float64]
    alpha: float
    fit_prior: bool
    norm: bool
    n_features_in: int
