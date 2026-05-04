"""Binary Gaussian-process classification prediction output atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_predict_labels,
    witness_gpc_binary_predict_positive_class_mask,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _bool_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.size >= 1 and array.dtype == np.bool_)


def _class_pair(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape == (2,))


@register_atom(witness_gpc_binary_predict_positive_class_mask)
@icontract.require(lambda f_star: _finite_vector(f_star), "f_star must be a finite one-dimensional vector")
@icontract.ensure(
    lambda result, f_star: _bool_vector(result) and np.asarray(result).shape == np.asarray(f_star).shape,
    "positive-class mask must be a boolean vector aligned with f_star",
)
def gpc_binary_predict_positive_class_mask(
    f_star: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Compute sklearn's binary Gaussian-process positive-class mask from latent means."""
    return np.asarray(np.asarray(f_star, dtype=np.float64) > 0.0, dtype=np.bool_)


@register_atom(witness_gpc_binary_predict_labels)
@icontract.require(
    lambda positive_class_mask: _bool_vector(positive_class_mask),
    "positive_class_mask must be a one-dimensional boolean vector",
)
@icontract.require(lambda classes: _class_pair(classes), "classes must be a one-dimensional pair of class labels")
@icontract.ensure(
    lambda result, positive_class_mask: np.asarray(result).ndim == 1
    and np.asarray(result).shape == np.asarray(positive_class_mask).shape,
    "predicted labels must align with the positive_class_mask length",
)
def gpc_binary_predict_labels(
    positive_class_mask: NDArray[np.bool_],
    classes: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Map sklearn's binary Gaussian-process positive-class mask onto class labels."""
    mask = np.asarray(positive_class_mask, dtype=np.bool_)
    class_values = np.asarray(classes)
    return np.asarray(np.where(mask, class_values[1], class_values[0]))
