"""Bagging out-of-bag scoring helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_classifier_oob_accuracy,
    witness_bagging_oob_uncovered_mask,
    witness_bagging_regressor_oob_r2,
)

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _sample_index_block_valid(block: object, n_samples: int) -> bool:
    values = np.asarray(block)
    return bool(
        _positive_int(n_samples)
        and values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
    )

def _aligned_sample_index_blocks(sample_index_blocks: tuple[NDArray[np.int64], ...], n_samples: int) -> bool:
    return bool(
        _positive_int(n_samples)
        and len(sample_index_blocks) >= 1
        and all(_sample_index_block_valid(block, n_samples) for block in sample_index_blocks)
    )

def _oob_rows(sample_indices: NDArray[np.int64], n_samples: int) -> NDArray[np.int64]:
    in_bag_mask = np.zeros(n_samples, dtype=np.bool_)
    in_bag_mask[np.asarray(sample_indices, dtype=np.int64)] = True
    return np.asarray(np.flatnonzero(~in_bag_mask), dtype=np.int64)

def _mask_valid(result: object, n_samples: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (n_samples,) and values.dtype == np.bool_)

def _encoded_targets_valid(y_encoded: object) -> bool:
    values = np.asarray(y_encoded)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
    )

def _prediction_totals_valid(prediction_totals: object, y_encoded: object) -> bool:
    try:
        totals = np.asarray(prediction_totals, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    encoded = np.asarray(y_encoded)
    return bool(
        _encoded_targets_valid(y_encoded)
        and totals.ndim == 2
        and totals.shape[0] == encoded.shape[0]
        and totals.shape[1] >= 1
        and np.all(np.isfinite(totals))
        and np.all(totals >= 0.0)
        and np.all(encoded < totals.shape[1])
    )

def _regression_targets_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _aligned_regression_vectors(y_true: object, predictions: object) -> bool:
    truth = np.asarray(y_true, dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    return bool(
        _regression_targets_valid(y_true)
        and _regression_targets_valid(predictions)
        and truth.shape == preds.shape
    )

@register_atom(witness_bagging_oob_uncovered_mask)
@icontract.require(
    lambda sample_index_blocks, n_samples: _aligned_sample_index_blocks(sample_index_blocks, n_samples),
    "sample index blocks must be a nonempty tuple of valid in-bag index vectors",
)
@icontract.ensure(
    lambda result, n_samples: _mask_valid(result, n_samples),
    "uncovered mask must be a boolean vector with one entry per sample",
)
def bagging_oob_uncovered_mask(
    sample_index_blocks: tuple[NDArray[np.int64], ...],
    *,
    n_samples: int,
) -> NDArray[np.bool_]:
    """Mark samples that were never out-of-bag for any estimator."""
    covered = np.zeros(n_samples, dtype=np.bool_)
    for sample_indices in sample_index_blocks:
        covered[_oob_rows(np.asarray(sample_indices, dtype=np.int64), n_samples)] = True
    return np.asarray(~covered, dtype=np.bool_)

@register_atom(witness_bagging_classifier_oob_accuracy)
@icontract.require(
    lambda y_encoded, prediction_totals: _prediction_totals_valid(prediction_totals, y_encoded),
    "y_encoded must be a nonnegative integer vector aligned with a finite nonnegative sample-by-class total matrix",
)
@icontract.ensure(
    lambda result: isinstance(result, float) and np.isfinite(result) and 0.0 <= result <= 1.0,
    "classifier OOB accuracy must be a finite value in [0, 1]",
)
def bagging_classifier_oob_accuracy(
    y_encoded: NDArray[np.int64],
    prediction_totals: NDArray[np.float64],
) -> float:
    from sklearn.metrics import accuracy_score, r2_score
    """Return the fraction of rows whose largest class total matches the encoded label."""
    encoded = np.asarray(y_encoded, dtype=np.int64)
    totals = np.asarray(prediction_totals, dtype=np.float64)
    return float(accuracy_score(encoded, np.argmax(totals, axis=1)))

@register_atom(witness_bagging_regressor_oob_r2)
@icontract.require(
    lambda y_true, predictions: _aligned_regression_vectors(y_true, predictions),
    "y_true and predictions must be aligned finite 1D regression vectors",
)
@icontract.ensure(
    lambda result: isinstance(result, float) and np.isfinite(result),
    "regressor OOB r2 must be a finite value",
)
def bagging_regressor_oob_r2(
    y_true: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> float:
    from sklearn.metrics import accuracy_score, r2_score
    """Return how much of the target variation the averaged predictions explain."""
    return float(r2_score(np.asarray(y_true, dtype=np.float64), np.asarray(predictions, dtype=np.float64)))
