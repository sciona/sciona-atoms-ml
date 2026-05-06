"""Sklearn coordinate-descent path-residual prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_rescaled_train_sample_weight,
    witness_cd_path_residuals_resolved_precompute,
    witness_cd_path_residuals_test_sample_weight,
    witness_cd_path_residuals_train_sample_count,
    witness_cd_path_residuals_train_sample_weight,
    witness_cd_path_residuals_use_gram_precompute,
    witness_cd_path_residuals_use_sample_weight_branch,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_cd_path_residuals_use_sample_weight_branch)
@icontract.ensure(
    lambda result, sample_weight: isinstance(result, bool) and result == (sample_weight is not None),
    "sample-weight branch must match sample_weight is not None",
)
def cd_path_residuals_use_sample_weight_branch(sample_weight: object) -> bool:
    """Return whether _path_residuals uses the sample-weight branch."""
    return sample_weight is not None


@register_atom(witness_cd_path_residuals_train_sample_weight)
@icontract.require(lambda sample_weight: sample_weight is not None, "sample_weight must be provided")
@icontract.require(lambda train: hasattr(train, "__iter__"), "train must be indexable")
@icontract.ensure(
    lambda result, sample_weight, train: np.array_equal(np.asarray(result), np.asarray(sample_weight)[train]),
    "training sample weights must equal sample_weight[train]",
)
def cd_path_residuals_train_sample_weight(sample_weight: object, train: object) -> np.ndarray:
    """Return the training sample-weight slice used by _path_residuals."""
    return np.asarray(sample_weight)[train]


@register_atom(witness_cd_path_residuals_test_sample_weight)
@icontract.require(lambda sample_weight: sample_weight is not None, "sample_weight must be provided")
@icontract.require(lambda test: hasattr(test, "__iter__"), "test must be indexable")
@icontract.ensure(
    lambda result, sample_weight, test: np.array_equal(np.asarray(result), np.asarray(sample_weight)[test]),
    "test sample weights must equal sample_weight[test]",
)
def cd_path_residuals_test_sample_weight(sample_weight: object, test: object) -> np.ndarray:
    """Return the test sample-weight slice used by _path_residuals."""
    return np.asarray(sample_weight)[test]


@register_atom(witness_cd_path_residuals_train_sample_count)
@icontract.require(
    lambda x_train_shape: isinstance(x_train_shape, tuple) and len(x_train_shape) >= 1 and _positive_int(x_train_shape[0]),
    "x_train_shape must have a positive first dimension",
)
@icontract.ensure(
    lambda result, x_train_shape: _positive_int(result) and int(result) == int(x_train_shape[0]),
    "train sample count must equal X_train.shape[0]",
)
def cd_path_residuals_train_sample_count(x_train_shape: tuple[int, ...]) -> int:
    """Return the training sample count used to rescale sample weights."""
    return int(x_train_shape[0])


@register_atom(witness_cd_path_residuals_rescaled_train_sample_weight)
@icontract.require(
    lambda train_sample_weight: isinstance(train_sample_weight, np.ndarray) and train_sample_weight.size >= 1,
    "train_sample_weight must be a nonempty ndarray",
)
@icontract.require(
    lambda train_sample_count: _positive_int(train_sample_count),
    "train_sample_count must be positive",
)
@icontract.ensure(
    lambda result, train_sample_count: isinstance(result, np.ndarray)
    and np.isclose(float(np.sum(result)), float(train_sample_count)),
    "rescaled training sample weights must sum to the training sample count",
)
def cd_path_residuals_rescaled_train_sample_weight(
    train_sample_weight: np.ndarray, train_sample_count: int
) -> np.ndarray:
    """Return the training sample weights rescaled to sum to n_samples."""
    return train_sample_weight * (int(train_sample_count) / np.sum(train_sample_weight))


@register_atom(witness_cd_path_residuals_use_gram_precompute)
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be positive")
@icontract.ensure(
    lambda result, y_ndim: isinstance(result, bool) and result == (int(y_ndim) == 1),
    "Gram-precompute branch must match y.ndim == 1",
)
def cd_path_residuals_use_gram_precompute(y_ndim: int) -> bool:
    """Return whether _path_residuals preserves path_params['precompute']."""
    return int(y_ndim) == 1


@register_atom(witness_cd_path_residuals_resolved_precompute)
@icontract.require(
    lambda use_gram_precompute: isinstance(use_gram_precompute, bool),
    "use_gram_precompute must be boolean",
)
@icontract.ensure(
    lambda result, path_precompute, use_gram_precompute: (
        result == path_precompute if use_gram_precompute else result is False
    ),
    "resolved precompute must preserve path_params['precompute'] only for mono-output",
)
def cd_path_residuals_resolved_precompute(
    path_precompute: object, use_gram_precompute: bool
) -> object:
    """Return the resolved precompute value used by _path_residuals."""
    return path_precompute if use_gram_precompute else False
