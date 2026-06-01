from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_center_data,
    witness_pca_svd_decompose,
    witness_calculate_pca_variance,
)

@register_atom(witness_center_data, name="center_data")
@icontract.require(lambda X: X.ndim == 2, "Precondition failed: X.ndim == 2")
@icontract.ensure(lambda result, X: X_centered.shape == X.shape, "Postcondition failed: X_centered.shape == X.shape")
def center_data(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute mean and center the input matrix.

    Args:
        X: NDArray[np.float64]

    Returns:
        X_centered: NDArray[np.float64]
    """
    import numpy
    return numpy.mean(X=X) # type: ignore

@register_atom(witness_pca_svd_decompose, name="pca_svd_decompose")
@icontract.require(lambda X_centered, n_components: n_components <= min(X_centered.shape), "Precondition failed: n_components <= min(X_centered.shape)")
@icontract.ensure(lambda result, X_centered, n_components: result is not None, "Postcondition failed: result is not None")
def pca_svd_decompose(X_centered: NDArray[np.float64], n_components: int) -> NDArray[np.float64]:
    """Perform SVD decomposition on centered data and compute loadings.

    Args:
        X_centered: NDArray[np.float64]
        n_components: int

    Returns:
        components: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.svd(X_centered=X_centered, n_components=n_components) # type: ignore

@register_atom(witness_calculate_pca_variance, name="calculate_pca_variance")
@icontract.require(lambda singular_values, n_samples: n_samples > 1, "Precondition failed: n_samples > 1")
@icontract.ensure(lambda result, singular_values, n_samples: result is not None, "Postcondition failed: result is not None")
def calculate_pca_variance(singular_values: NDArray[np.float64], n_samples: int) -> NDArray[np.float64]:
    """Calculate explained variance metrics.

    Args:
        singular_values: NDArray[np.float64]
        n_samples: int

    Returns:
        explained_variance: NDArray[np.float64]
    """
    import sklearn.decomposition
    return sklearn.decomposition.PCA(singular_values=singular_values, n_samples=n_samples) # type: ignore

