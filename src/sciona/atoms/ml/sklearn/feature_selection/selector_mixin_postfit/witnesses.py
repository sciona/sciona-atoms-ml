"""Ghost witnesses for SelectorMixin post-fit helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_support_mask(support_mask: AbstractArray) -> int:
    if len(support_mask.shape) != 1:
        raise ValueError("support_mask must be 1D")
    n_features = int(support_mask.shape[0])
    if n_features < 1:
        raise ValueError("support_mask must be nonempty")
    return n_features


def witness_selector_support_indices(
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe integer feature indices selected by a support mask."""
    n_features = _check_support_mask(support_mask)
    return AbstractArray(shape=(n_features,), dtype="int64", min_val=0)


def witness_selector_transform_dense(
    X: AbstractArray,
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe dense feature selection with a supplied support mask."""
    n_features = _check_support_mask(support_mask)
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if int(X.shape[1]) != n_features:
        raise ValueError("X and support_mask must agree on feature count")
    return AbstractArray(shape=(int(X.shape[0]), n_features), dtype="float64")


def witness_selector_inverse_transform_dense(
    X_selected: AbstractArray,
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe dense inverse feature selection with zero-filled dropped columns."""
    n_features = _check_support_mask(support_mask)
    if len(X_selected.shape) not in {1, 2}:
        raise ValueError("X_selected must be 1D or 2D")
    n_samples = 1 if len(X_selected.shape) == 1 else int(X_selected.shape[0])
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_selector_feature_names_out(
    input_features: AbstractArray,
    support_mask: AbstractArray,
) -> AbstractArray:
    """Describe feature-name masking by a support mask."""
    n_features = _check_support_mask(support_mask)
    if len(input_features.shape) != 1 or int(input_features.shape[0]) != n_features:
        raise ValueError("input_features must be 1D and match support_mask length")
    return AbstractArray(shape=(n_features,), dtype="str")
