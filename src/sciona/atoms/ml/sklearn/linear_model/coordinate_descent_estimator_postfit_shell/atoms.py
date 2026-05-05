"""Sklearn coordinate-descent estimator post-fit shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_nonfinite_parameter_guard_required,
    witness_cd_estimator_nonfinite_parameter_message,
    witness_cd_estimator_single_target_branch,
    witness_cd_estimator_single_target_coef,
    witness_cd_estimator_single_target_dual_gap,
    witness_cd_estimator_single_target_n_iter,
    witness_cd_estimator_sparse_coef,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_tensor(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_cd_estimator_single_target_branch)
@icontract.require(lambda n_targets: _positive_int(n_targets), "n_targets must be positive")
@icontract.ensure(
    lambda result, n_targets: isinstance(result, bool) and result == (int(n_targets) == 1),
    "single-target branch must match n_targets == 1",
)
def cd_estimator_single_target_branch(n_targets: int) -> bool:
    """Return whether ElasticNet.fit should collapse post-fit outputs to 1D."""
    return int(n_targets) == 1


@register_atom(witness_cd_estimator_single_target_n_iter)
@icontract.require(
    lambda n_iter_list: isinstance(n_iter_list, Sequence) and len(n_iter_list) >= 1 and all(_positive_int(v) for v in n_iter_list),
    "n_iter_list must be a nonempty sequence of positive integers",
)
@icontract.ensure(
    lambda result, n_iter_list: _positive_int(result) and int(result) == int(n_iter_list[0]),
    "single-target n_iter must equal the first iteration count",
)
def cd_estimator_single_target_n_iter(n_iter_list: Sequence[int]) -> int:
    """Return the collapsed single-target iteration count."""
    return int(n_iter_list[0])


@register_atom(witness_cd_estimator_single_target_coef)
@icontract.require(lambda coef_matrix: _finite_tensor(coef_matrix), "coef_matrix must be finite")
@icontract.ensure(
    lambda result, coef_matrix: np.array_equal(
        np.asarray(result),
        np.asarray(coef_matrix)[0],
    ),
    "single-target coef must equal coef_matrix[0]",
)
def cd_estimator_single_target_coef(coef_matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the collapsed single-target coefficient vector."""
    return np.asarray(coef_matrix)[0]


@register_atom(witness_cd_estimator_single_target_dual_gap)
@icontract.require(lambda dual_gaps: _finite_tensor(dual_gaps), "dual_gaps must be finite")
@icontract.ensure(
    lambda result, dual_gaps: np.isfinite(float(result)) and np.isclose(float(result), float(np.asarray(dual_gaps)[0])),
    "single-target dual gap must equal dual_gaps[0]",
)
def cd_estimator_single_target_dual_gap(dual_gaps: NDArray[np.floating]) -> float:
    """Return the collapsed single-target dual gap."""
    return float(np.asarray(dual_gaps)[0])


@register_atom(witness_cd_estimator_nonfinite_parameter_guard_required)
@icontract.require(lambda coef: np.asarray(coef).size >= 1, "coef must be array-like and nonempty")
@icontract.require(lambda intercept: np.asarray(intercept).size >= 1 or np.isscalar(intercept), "intercept must be scalar or array-like")
@icontract.ensure(
    lambda result, coef, intercept: isinstance(result, bool)
    and result == (not all(np.isfinite(w).all() for w in [np.asarray(coef), np.asarray(intercept)])),
    "guard predicate must match the sklearn all-finite check",
)
def cd_estimator_nonfinite_parameter_guard_required(coef: object, intercept: object) -> bool:
    """Return whether ElasticNet.fit should raise for non-finite fitted parameters."""
    return not all(np.isfinite(w).all() for w in [np.asarray(coef), np.asarray(intercept)])


@register_atom(witness_cd_estimator_nonfinite_parameter_message)
@icontract.ensure(
    lambda result, coef, intercept: isinstance(result, str)
    and result
    == (
        "Coordinate descent iterations resulted in non-finite parameter"
        " values. The input data may contain large values and need to"
        " be preprocessed."
    ),
    "non-finite parameter message must match sklearn formatting",
)
def cd_estimator_nonfinite_parameter_message(coef: object, intercept: object) -> str:
    """Return the non-finite parameter ValueError message used by ElasticNet.fit."""
    del coef, intercept
    return (
        "Coordinate descent iterations resulted in non-finite parameter"
        " values. The input data may contain large values and need to"
        " be preprocessed."
    )


@register_atom(witness_cd_estimator_sparse_coef)
@icontract.require(lambda coef: _finite_tensor(coef), "coef must be a finite numeric array")
@icontract.ensure(
    lambda result, coef: sparse.isspmatrix_csr(result)
    and np.array_equal(result.toarray(), np.atleast_2d(np.asarray(coef))),
    "sparse_coef_ must be the CSR representation of coef_",
)
def cd_estimator_sparse_coef(coef: object) -> sparse.csr_matrix:
    """Return the sparse_coef_ representation of coef_."""
    return sparse.csr_matrix(np.atleast_2d(np.asarray(coef)))
