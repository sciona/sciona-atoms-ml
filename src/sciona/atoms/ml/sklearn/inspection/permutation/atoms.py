"""Permutation-importance aggregation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_mean,
    witness_permutation_importance_std,
    witness_permutation_importance_summary,
    witness_permutation_importance_values,
)


def _finite_score(score: float) -> bool:
    return bool(isinstance(score, (int, float)) and not isinstance(score, bool) and np.isfinite(float(score)))


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _feature_vector_valid(result: NDArray[np.float64], importances: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(importances)
    return bool(values.shape == (source.shape[0],) and np.all(np.isfinite(values)))


def _importance_matrix_valid(result: NDArray[np.float64], permuted_scores: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(permuted_scores)
    return bool(values.shape == source.shape and np.all(np.isfinite(values)))


def _summary_valid(
    result: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    permuted_scores: NDArray[np.float64],
) -> bool:
    mean, spread, importances = result
    source = np.asarray(permuted_scores)
    return bool(
        _importance_matrix_valid(importances, source)
        and _feature_vector_valid(mean, importances)
        and _feature_vector_valid(spread, importances)
    )


@register_atom(witness_permutation_importance_values)
@icontract.require(lambda baseline_score: _finite_score(baseline_score), "baseline_score must be finite")
@icontract.require(lambda permuted_scores: _finite_matrix(permuted_scores), "permuted_scores must be a finite 2D matrix")
@icontract.ensure(lambda result, permuted_scores: _importance_matrix_valid(result, permuted_scores), "importance values must match score shape")
def permutation_importance_values(
    baseline_score: float,
    permuted_scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute raw feature importance values from baseline and shuffled scores."""
    scores = np.asarray(permuted_scores, dtype=np.float64)
    return np.asarray(float(baseline_score) - scores, dtype=np.float64)


@register_atom(witness_permutation_importance_mean)
@icontract.require(lambda importances: _finite_matrix(importances), "importances must be a finite 2D matrix")
@icontract.ensure(lambda result, importances: _feature_vector_valid(result, importances), "mean must produce one finite value per feature")
def permutation_importance_mean(importances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute mean feature importance over repeated shuffles."""
    values = np.asarray(importances, dtype=np.float64)
    return np.asarray(np.mean(values, axis=1), dtype=np.float64)


@register_atom(witness_permutation_importance_std)
@icontract.require(lambda importances: _finite_matrix(importances), "importances must be a finite 2D matrix")
@icontract.ensure(lambda result, importances: _feature_vector_valid(result, importances), "spread must produce one finite value per feature")
def permutation_importance_std(importances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute feature importance spread over repeated shuffles."""
    values = np.asarray(importances, dtype=np.float64)
    return np.asarray(np.std(values, axis=1), dtype=np.float64)


@register_atom(witness_permutation_importance_summary)
@icontract.require(lambda baseline_score: _finite_score(baseline_score), "baseline_score must be finite")
@icontract.require(lambda permuted_scores: _finite_matrix(permuted_scores), "permuted_scores must be a finite 2D matrix")
@icontract.ensure(lambda result, permuted_scores: _summary_valid(result, permuted_scores), "summary arrays must align by feature")
def permutation_importance_summary(
    baseline_score: float,
    permuted_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean, spread, and raw permutation importance arrays."""
    importances = permutation_importance_values(baseline_score, permuted_scores)
    return (
        permutation_importance_mean(importances),
        permutation_importance_std(importances),
        importances,
    )
