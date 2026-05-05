"""Sklearn coordinate-descent CV post-fit shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_delete_l1_ratio_required,
    witness_cd_cv_fit_coef,
    witness_cd_cv_fit_dual_gap,
    witness_cd_cv_fit_intercept,
    witness_cd_cv_fit_n_iter,
    witness_cd_cv_fit_return_self,
    witness_cd_cv_refit_uses_sample_weight,
)


def _finite_value(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_cd_cv_refit_uses_sample_weight)
@icontract.ensure(
    lambda result, sample_weight: isinstance(result, bool) and result == (sample_weight is not None),
    "sample-weight branch must match sample_weight is not None",
)
def cd_cv_refit_uses_sample_weight(sample_weight: object) -> bool:
    """Return whether LinearModelCV.fit should refit with sample_weight."""
    return sample_weight is not None


@register_atom(witness_cd_cv_delete_l1_ratio_required)
@icontract.require(
    lambda has_l1_ratio_attr: isinstance(has_l1_ratio_attr, bool),
    "has_l1_ratio_attr must be boolean",
)
@icontract.ensure(
    lambda result, has_l1_ratio_attr: isinstance(result, bool) and result == (not has_l1_ratio_attr),
    "cleanup branch must match not hasattr(self, 'l1_ratio')",
)
def cd_cv_delete_l1_ratio_required(has_l1_ratio_attr: bool) -> bool:
    """Return whether LinearModelCV.fit should delete l1_ratio_ after refit."""
    return not has_l1_ratio_attr


@register_atom(witness_cd_cv_fit_coef)
@icontract.require(lambda model_coef: _finite_value(model_coef), "model_coef must be finite")
@icontract.ensure(
    lambda result, model_coef: np.array_equal(np.asarray(result), np.asarray(model_coef)),
    "coef passthrough must equal model.coef_",
)
def cd_cv_fit_coef(model_coef: object) -> object:
    """Return the fitted coef_ copied from the refit model."""
    return model_coef


@register_atom(witness_cd_cv_fit_intercept)
@icontract.require(lambda model_intercept: _finite_value(model_intercept), "model_intercept must be finite")
@icontract.ensure(
    lambda result, model_intercept: np.array_equal(np.asarray(result), np.asarray(model_intercept)),
    "intercept passthrough must equal model.intercept_",
)
def cd_cv_fit_intercept(model_intercept: object) -> object:
    """Return the fitted intercept_ copied from the refit model."""
    return model_intercept


@register_atom(witness_cd_cv_fit_dual_gap)
@icontract.require(lambda model_dual_gap: _finite_value(model_dual_gap), "model_dual_gap must be finite")
@icontract.ensure(
    lambda result, model_dual_gap: np.array_equal(np.asarray(result), np.asarray(model_dual_gap)),
    "dual-gap passthrough must equal model.dual_gap_",
)
def cd_cv_fit_dual_gap(model_dual_gap: object) -> object:
    """Return the fitted dual_gap_ copied from the refit model."""
    return model_dual_gap


@register_atom(witness_cd_cv_fit_n_iter)
@icontract.require(
    lambda model_n_iter: (
        isinstance(model_n_iter, Sequence)
        and not isinstance(model_n_iter, (str, bytes))
        and len(model_n_iter) >= 1
    )
    or isinstance(model_n_iter, (int, np.integer)),
    "model_n_iter must be a positive integer or a nonempty sequence",
)
@icontract.ensure(
    lambda result, model_n_iter: result == model_n_iter,
    "n_iter passthrough must equal model.n_iter_",
)
def cd_cv_fit_n_iter(model_n_iter: object) -> object:
    """Return the fitted n_iter_ copied from the refit model."""
    return model_n_iter


@register_atom(witness_cd_cv_fit_return_self)
@icontract.ensure(
    lambda result, estimator_identity: result is estimator_identity,
    "fit return must pass self through unchanged",
)
def cd_cv_fit_return_self(estimator_identity: object) -> object:
    """Return self from LinearModelCV.fit."""
    return estimator_identity
