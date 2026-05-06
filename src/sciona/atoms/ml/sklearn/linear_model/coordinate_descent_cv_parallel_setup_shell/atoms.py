"""Sklearn coordinate-descent CV parallel-setup atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_best_mse_initial,
    witness_cd_cv_fold_count,
    witness_cd_cv_folds,
    witness_cd_cv_path_job_count,
    witness_cd_cv_path_job_kwargs,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _split_pair(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2


@register_atom(witness_cd_cv_folds)
@icontract.require(lambda cv_splits: isinstance(cv_splits, Iterable), "cv_splits must be iterable")
@icontract.ensure(
    lambda result: isinstance(result, list),
    "fold materialization must return a concrete list",
)
@icontract.ensure(
    lambda result: all(_split_pair(item) for item in result),
    "each fold entry must be a (train, test) pair",
)
def cd_cv_folds(cv_splits: Iterable[tuple[object, object]]) -> list[tuple[object, object]]:
    """Return sklearn's list(cv.split(...)) materialization."""
    return list(cv_splits)


@register_atom(witness_cd_cv_fold_count)
@icontract.require(lambda folds: isinstance(folds, Sequence), "folds must be a sequence")
@icontract.ensure(
    lambda result, folds: _positive_int(result) and int(result) == len(folds),
    "fold count must equal len(folds)",
)
def cd_cv_fold_count(folds: Sequence[object]) -> int:
    """Return the number of materialized CV folds."""
    return len(folds)


@register_atom(witness_cd_cv_path_job_kwargs)
@icontract.ensure(
    lambda result, this_alphas, this_l1_ratio, x_dtype_type: isinstance(result, dict)
    and result == {
        "alphas": this_alphas,
        "l1_ratio": this_l1_ratio,
        "X_order": "F",
        "dtype": x_dtype_type,
    },
    "job kwargs must match sklearn's _path_residuals delayed-call payload",
)
def cd_cv_path_job_kwargs(this_alphas: object, this_l1_ratio: object, x_dtype_type: object) -> dict[str, object]:
    """Return sklearn's delayed(_path_residuals) kwarg mapping for one CV job."""
    return {
        "alphas": this_alphas,
        "l1_ratio": this_l1_ratio,
        "X_order": "F",
        "dtype": x_dtype_type,
    }


@register_atom(witness_cd_cv_path_job_count)
@icontract.require(lambda l1_ratios: isinstance(l1_ratios, Sequence), "l1_ratios must be a sequence")
@icontract.require(lambda folds: isinstance(folds, Sequence), "folds must be a sequence")
@icontract.ensure(
    lambda result, l1_ratios, folds: _positive_int(result) and int(result) == len(l1_ratios) * len(folds),
    "job count must equal the l1_ratio x fold product",
)
def cd_cv_path_job_count(l1_ratios: Sequence[object], folds: Sequence[object]) -> int:
    """Return the number of delayed _path_residuals jobs scheduled by LinearModelCV.fit."""
    return len(l1_ratios) * len(folds)


@register_atom(witness_cd_cv_best_mse_initial)
@icontract.require(lambda fold_count: _positive_int(fold_count), "fold_count must be positive")
@icontract.ensure(
    lambda result: isinstance(result, float) and np.isinf(result) and result > 0.0,
    "best_mse must initialize to positive infinity",
)
def cd_cv_best_mse_initial(fold_count: int) -> float:
    """Return sklearn's initial best_mse value before scanning CV results."""
    del fold_count
    return float(np.inf)
