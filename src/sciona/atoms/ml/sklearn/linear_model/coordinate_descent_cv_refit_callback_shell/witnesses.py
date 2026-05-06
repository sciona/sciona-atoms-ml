"""Ghost witnesses for sklearn coordinate-descent CV refit callback-shell atoms."""

from __future__ import annotations


def witness_cd_cv_refit_set_params_kwargs(common_params: object) -> object:
    """Describe the model.set_params(**common_params) kwarg shell in LinearModelCV.fit."""
    return common_params


def witness_cd_cv_refit_set_params_result(model_after_set_params: object) -> object:
    """Describe the model object after set_params(...) in LinearModelCV.fit."""
    return model_after_set_params


def witness_cd_cv_refit_weighted_fit_kwargs(sample_weight: object) -> object:
    """Describe the weighted model.fit(..., sample_weight=...) kwarg shell in LinearModelCV.fit."""
    return sample_weight


def witness_cd_cv_refit_fitted_model(model_after_fit: object) -> object:
    """Describe the fitted model object returned by the refit callback in LinearModelCV.fit."""
    return model_after_fit
