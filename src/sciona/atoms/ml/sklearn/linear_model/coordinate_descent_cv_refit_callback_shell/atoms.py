"""Sklearn coordinate-descent CV refit callback-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_cv_refit_fitted_model,
    witness_cd_cv_refit_set_params_kwargs,
    witness_cd_cv_refit_set_params_result,
    witness_cd_cv_refit_weighted_fit_kwargs,
)


@register_atom(witness_cd_cv_refit_set_params_kwargs)
@icontract.require(lambda common_params: isinstance(common_params, dict), "common_params must be a dict")
@icontract.ensure(
    lambda result, common_params: isinstance(result, dict) and result == common_params,
    "set_params kwargs must preserve the common-parameter mapping",
)
def cd_cv_refit_set_params_kwargs(common_params: dict[object, object]) -> dict[object, object]:
    """Return the kwargs payload expanded into model.set_params(**common_params)."""
    return dict(common_params)


@register_atom(witness_cd_cv_refit_set_params_result)
@icontract.ensure(
    lambda result, model_after_set_params: result is model_after_set_params,
    "set_params callback shell must preserve the model object identity",
)
def cd_cv_refit_set_params_result(model_after_set_params: object) -> object:
    """Return the model object after the deferred set_params callback."""
    return model_after_set_params


@register_atom(witness_cd_cv_refit_weighted_fit_kwargs)
@icontract.require(
    lambda sample_weight: sample_weight is not None,
    "sample_weight must be present for the weighted refit branch",
)
@icontract.ensure(
    lambda result, sample_weight: isinstance(result, dict)
    and result == {"sample_weight": sample_weight},
    "weighted fit kwargs must map sample_weight through unchanged",
)
def cd_cv_refit_weighted_fit_kwargs(sample_weight: object) -> dict[str, object]:
    """Return the kwargs payload for model.fit(X, y, sample_weight=sample_weight)."""
    return {"sample_weight": sample_weight}


@register_atom(witness_cd_cv_refit_fitted_model)
@icontract.ensure(
    lambda result, model_after_fit: result is model_after_fit,
    "refit callback shell must preserve the fitted model object identity",
)
def cd_cv_refit_fitted_model(model_after_fit: object) -> object:
    """Return the model object after the deferred refit callback."""
    return model_after_fit
