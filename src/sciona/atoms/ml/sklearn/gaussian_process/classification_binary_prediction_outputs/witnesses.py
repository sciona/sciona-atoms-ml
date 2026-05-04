"""Ghost witnesses for binary Gaussian-process classification prediction output atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def witness_gpc_binary_predict_positive_class_mask(
    f_star: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Describe sklearn's hard-decision positive-class mask from latent means."""
    return np.asarray(f_star, dtype=np.float64) > 0.0


def witness_gpc_binary_predict_labels(
    positive_class_mask: NDArray[np.bool_],
    classes: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Describe sklearn's class-label lookup from the positive-class mask."""
    class_values = np.asarray(classes)
    return np.asarray(np.where(positive_class_mask, class_values[1], class_values[0]))
