"""Sklearn coordinate-descent CV unweighted refit callback-shell atoms."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_refit_unweighted_fit_args,
    witness_cd_cv_refit_unweighted_fit_call_required,
    witness_cd_cv_refit_unweighted_fit_kwargs,
    witness_cd_cv_refit_unweighted_fitted_model,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_cd_cv_refit_unweighted_fit_call_required)
@icontract.ensure(
    lambda result, sample_weight: _bool(result) and result == (sample_weight is None),
    "unweighted refit branch must match sample_weight is None",
)
def cd_cv_refit_unweighted_fit_call_required(sample_weight: object) -> bool:
    """Return whether LinearModelCV.fit refits with model.fit(X, y)."""
    return sample_weight is None


@register_atom(witness_cd_cv_refit_unweighted_fit_args)
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight must be absent")
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "unweighted fit positional args must preserve X and y identity",
)
def cd_cv_refit_unweighted_fit_args(
    X: object, y: object, sample_weight: object
) -> tuple[object, object]:
    """Return the positional args passed to model.fit(X, y)."""
    return (X, y)


@register_atom(witness_cd_cv_refit_unweighted_fit_kwargs)
@icontract.require(lambda sample_weight: sample_weight is None, "sample_weight must be absent")
@icontract.ensure(
    lambda result: isinstance(result, dict) and result == {},
    "unweighted fit kwargs must be empty",
)
def cd_cv_refit_unweighted_fit_kwargs(sample_weight: object) -> dict[str, object]:
    """Return the empty kwargs payload for model.fit(X, y)."""
    return {}


@register_atom(witness_cd_cv_refit_unweighted_fitted_model)
@icontract.ensure(
    lambda result, model_after_fit: result is model_after_fit,
    "unweighted refit callback shell must preserve the fitted model identity",
)
def cd_cv_refit_unweighted_fitted_model(model_after_fit: object) -> object:
    """Return the model object after the deferred unweighted refit callback."""
    return model_after_fit
