"""Ghost witnesses for selected sklearn decomposition atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_pca_fit(
    X: AbstractArray,
    n_components: int | float | None = None,
    *,
    whiten: bool = False,
    copy: bool = True,
    svd_solver: str = "full",
) -> AbstractArray:
    """Describe fitting PCA components from a dense sample matrix."""
    del whiten, copy
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if n_samples < 2:
        raise ValueError("PCA requires at least two samples")
    if svd_solver != "full":
        raise ValueError("this atom exposes the full-SVD PCA fit path")
    if n_components is None:
        width = min(n_samples, n_features)
    elif isinstance(n_components, int) and not isinstance(n_components, bool):
        if n_components < 0 or n_components > min(n_samples, n_features):
            raise ValueError("n_components must fit the full-SVD rank bound")
        width = n_components
    elif isinstance(n_components, float):
        if not 0.0 < n_components < 1.0:
            raise ValueError("fractional n_components must lie in (0, 1)")
        width = min(n_samples, n_features)
    else:
        raise ValueError("n_components must be None, an integer, or a float fraction")
    return AbstractArray(shape=(width, n_features), dtype="float64")
