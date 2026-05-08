"""Sklearn ElasticNet multi-target post-fit shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_multitarget_branch,
    witness_cd_estimator_multitarget_coef,
    witness_cd_estimator_multitarget_dual_gap,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_2d_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape[0] >= 1
        and value.shape[1] >= 1
        and np.all(np.isfinite(value))
    )


def _finite_vector(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 1
        and value.shape[0] >= 1
        and np.all(np.isfinite(value))
    )


@register_atom(witness_cd_estimator_multitarget_branch)
@icontract.require(lambda n_targets: _positive_int(n_targets), "n_targets must be positive")
@icontract.ensure(
    lambda result, n_targets: isinstance(result, bool) and result == (int(n_targets) != 1),
    "multi-target branch must match n_targets != 1",
)
def cd_estimator_multitarget_branch(n_targets: int) -> bool:
    """Return whether ElasticNet.fit keeps post-loop outputs as multi-target arrays."""
    return int(n_targets) != 1


@register_atom(witness_cd_estimator_multitarget_coef)
@icontract.require(
    lambda n_targets: _positive_int(n_targets) and int(n_targets) != 1,
    "n_targets must select the multi-target branch",
)
@icontract.require(lambda coef_matrix: _finite_2d_array(coef_matrix), "coef_matrix must be a finite 2D ndarray")
@icontract.require(
    lambda coef_matrix, n_targets: coef_matrix.shape[0] == int(n_targets),
    "coef_matrix rows must match n_targets",
)
@icontract.ensure(
    lambda result, coef_matrix, n_targets: result is coef_matrix and result.shape[0] == int(n_targets),
    "multi-target coef assignment must preserve the coefficient matrix object",
)
def cd_estimator_multitarget_coef(
    coef_matrix: NDArray[np.floating], n_targets: int
) -> NDArray[np.floating]:
    """Return the coefficient matrix assigned to self.coef_ for multi-target output."""
    return coef_matrix


@register_atom(witness_cd_estimator_multitarget_dual_gap)
@icontract.require(
    lambda n_targets: _positive_int(n_targets) and int(n_targets) != 1,
    "n_targets must select the multi-target branch",
)
@icontract.require(lambda dual_gaps: _finite_vector(dual_gaps), "dual_gaps must be a finite 1D ndarray")
@icontract.require(
    lambda dual_gaps, n_targets: dual_gaps.shape[0] == int(n_targets),
    "dual_gaps length must match n_targets",
)
@icontract.ensure(
    lambda result, dual_gaps, n_targets: result is dual_gaps and result.shape[0] == int(n_targets),
    "multi-target dual-gap assignment must preserve the dual-gap vector object",
)
def cd_estimator_multitarget_dual_gap(
    dual_gaps: NDArray[np.floating], n_targets: int
) -> NDArray[np.floating]:
    """Return the dual-gap vector assigned to self.dual_gap_ for multi-target output."""
    return dual_gaps
