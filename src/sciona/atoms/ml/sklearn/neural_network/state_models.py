"""State containers for sklearn neural-network atoms."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class BernoulliRBMState:
    """Learned dense Bernoulli restricted Boltzmann machine parameters."""

    components: NDArray[np.float64]
    intercept_hidden: NDArray[np.float64]
    intercept_visible: NDArray[np.float64]
    h_samples: NDArray[np.float64]
    learning_rate: float
    batch_size: int
    n_iter: int
    n_features_in: int
    n_components: int
